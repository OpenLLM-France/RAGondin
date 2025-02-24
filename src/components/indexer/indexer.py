from omegaconf import OmegaConf
from .loader import DocSerializer
from typing import AsyncGenerator
from .embeddings import HFEmbedder
from .chunker import ABCChunker, ChunkerFactory
from .vectordb import ConnectorFactory, ABCVectorDB
from langchain_core.documents.base import Document



class Indexer:
    """This class bridges static files with the vector store database.
    """
    def __init__(self, config, logger, device=None) -> None:
        embedder = HFEmbedder(embedder_config=config.embedder, device=device)        
        self.serializer = DocSerializer(data_dir=config.paths.root_dir / 'data', llm_config=config.llm)
        self.chunker: ABCChunker = ChunkerFactory.create_chunker(config, embedder=embedder.get_embeddings())
        self.vectordb = ConnectorFactory.create_vdb(config, logger=logger, embeddings=embedder.get_embeddings())
        self.logger = logger
        self.logger.info("Indexer initialized...")
        
    async def add_files2vdb(self, path):
        """Add a files to the vector database in async mode"""
        try:
            doc_generator: AsyncGenerator[Document, None] = self.serializer.serialize_documents(path, recursive=True)
            await self.vectordb.async_add_documents(
                doc_generator=doc_generator, 
                chunker=self.chunker, 
                document_batch_size=2,
                max_concurrent_gpu_ops=4,
                max_queued_batches=2
            )
            self.logger.info(f"Documents from {path} added.")
        except Exception as e:
            raise Exception(f"An exception as occured: {e}")


#   await self.vectordb.async_add_documents(
#                 doc_generator=doc_generator, 
#                 chunker=self.chunker, 
#                 document_batch_size=3,
#                 max_concurrent_gpu_ops=6,
#                 max_queued_batches=2
#             )