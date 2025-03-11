import asyncio
from .loader import DocSerializer
from typing import AsyncGenerator, Dict, Optional, Union, List
from .embeddings import HFEmbedder
from .chunker import ABCChunker, ChunkerFactory, SemanticSplitter, RecursiveSplitter
from .vectordb import ConnectorFactory, ABCVectorDB
from langchain_core.documents.base import Document
from pathlib import Path
from ..utils import SingletonMeta, SingletonABCMeta


class Indexer(metaclass=SingletonMeta):
    """This class bridges static files with the vector store database.
    """
    def __init__(self, config, logger, device=None) -> None:
        embedder = HFEmbedder(embedder_config=config.embedder, device=device)
        self.serializer = DocSerializer(data_dir=config.paths.data_dir, llm_config=config.llm)
        self.chunker: ABCChunker = ChunkerFactory.create_chunker(config, embedder=embedder.get_embeddings())
        self.vectordb = ConnectorFactory.create_vdb(config, logger=logger, embeddings=embedder.get_embeddings())
        self.logger = logger
        self.logger.info("Indexer initialized...")

        self.n_concurrent_loading = config.insertion.get("n_concurrent_loading", 2) # Number of concurrent loading operations
        self.n_concurrent_chunking = config.insertion.get("n_concurrent_chunking", 2) # Number of concurrent chunking operations

    async def chunk(self, doc, gpu_semaphore: asyncio.Semaphore) -> AsyncGenerator[Document, None]:
        if isinstance(self.chunker, SemanticSplitter):
            async with gpu_semaphore:
                chunks = await self.chunker.split_document(doc)
        else:
            chunks = await self.chunker.split_document(doc)

        self.logger.info(f"{Path(doc.metadata['source']).name} CHUNKED")
        return chunks
        
        
    async def add_files2vdb(self, path: str | list[str], metadata: Optional[Dict] = {}, collection_name : Optional[str] = None):
        """Add a files to the vector database in async mode"""
        gpu_semaphore = asyncio.Semaphore(self.n_concurrent_chunking) # Only allow max_concurrent_gpu_ops GPU operation at a time
        doc_generator: AsyncGenerator[Document, None] = self.serializer.serialize_documents(path, metadata=metadata, recursive=True, n_concurrent_ops=self.n_concurrent_loading)

        chunk_tasks = []
        try:
            async for doc in doc_generator:
                task = asyncio.create_task(self.chunk(doc, gpu_semaphore))
                chunk_tasks.append(task)

            # Await all tasks concurrently
            results = await asyncio.gather(*chunk_tasks)
            all_chunks = sum(results, [])

            if all_chunks:
                await self.vectordb.async_add_documents(all_chunks, collection_name=collection_name)
                self.logger.debug(f"Documents {path} added.")
        
        except Exception as e:
            self.logger.error(f"An exception as occured: {e}")
            raise Exception(f"An exception as occured: {e}")
        
    
    def delete_files(self, filters: Union[Dict, List[Dict]], collection_name: Optional[str] = None):
        deleted_files = []
        not_found_files = []

        for filter in filters:
            try:
                key = next(iter(filter))
                value = filter[key]
                # Get points associated with the file name
                points = self.vectordb.get_file_points(filter, collection_name)
                if not points:
                    self.logger.info(f"No points found for {key}: {value}")
                    not_found_files.append(filter)
                    continue

                # Delete the points
                self.vectordb.delete_points(points, collection_name)
                deleted_files.append(filter)

            except Exception as e:
                self.logger.error(f"Error in `delete_files` for {key} {value}: {e}")
        
        return deleted_files, not_found_files