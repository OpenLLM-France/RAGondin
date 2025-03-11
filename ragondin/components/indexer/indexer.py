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
        self.max_queued_batches = config.insertion.get("max_queued_batches", 4) # Maximum number of batches that can be queued
        self.batch_size = config.insertion.get("batch_size", 3) # Number of documents in each batch for vdb insertion
    
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
        doc_generator: AsyncGenerator[Document, None] = self.serializer.serialize_documents(path, recursive=True, n_concurrent_ops=self.n_concurrent_loading)
        batch_queue = asyncio.Queue(maxsize=self.max_queued_batches)

        producer_task = asyncio.create_task(
            producer(doc_generator, lambda x: self.chunk(x, gpu_semaphore), batch_queue, document_batch_size=self.batch_size, max_queued_batches=self.max_queued_batches, logger=self.logger)
        )
        consumer_tasks = [
            asyncio.create_task(
                consumer(
                    consumer_id=i, batch_queue=batch_queue, 
                    logger=self.logger,
                    vdb=self.vectordb,
                    collection_name=collection_name
                )
            ) for i in range(self.max_queued_batches)
        ]
            
        # Run producer and consumer concurrently
        try:
            # Wait for producer to complete and queue to be empty
            await producer_task
            
            # Wait for queue to be empty and all tasks to be marked as done
            await batch_queue.join()

            # Cancel any remaining consumer tasks
            for task in consumer_tasks:
                task.cancel()

            # Wait for all consumers to complete
            await asyncio.gather(*consumer_tasks, return_exceptions=True)

            self.logger.debug(f"Documents {path} added.")        

        except Exception as e:
            self.logger.error(f"An exception as occured: {e}")

            # Cancel all tasks in case of error
            producer_task.cancel()
            for task in consumer_tasks:
                task.cancel()
            raise
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

    
async def producer(doc_generator: AsyncGenerator[Document, None], chunker: callable, batch_queue: asyncio.Queue, document_batch_size: int=2, max_queued_batches: int=2, logger=None):
    current_batch = []
    try:
        async for doc in doc_generator:
            chunks = await chunker(doc)
            current_batch.append(chunks)

            if len(current_batch) == document_batch_size:
                await batch_queue.put(current_batch)
                current_batch = []
        
        # Put remaining documents
        if current_batch:
            await batch_queue.put(current_batch)
    
    finally:
        # Send one None for each consumer
        for _ in range(max_queued_batches):
            await batch_queue.put(None)


async def consumer(consumer_id, batch_queue: asyncio.Queue, logger, vdb: ABCVectorDB, collection_name: Optional[str] = None):
    while True:
        batch = await batch_queue.get()
        try:
            if batch is None:  # End signal
                batch_queue.task_done()
                logger.debug(f"Consumer {consumer_id} ended")
                break
        
            all_chunks = sum(batch, [])
            if all_chunks:
                await vdb.async_add_documents(all_chunks)

        except Exception as e:
            logger.error(f"Consumer {consumer_id} error processing batch: {e}")

        finally:
            batch_queue.task_done()