import gc
import inspect
from typing import Dict, List, Optional

import ray
import torch
from config import load_config
from langchain_core.documents.base import Document
from loguru import logger

from .chunker import ABCChunker, ChunkerFactory
from ..reranker import Reranker
from .embeddings import HFEmbedder
from .loaders.loader import DocSerializer
from .vectordb import ConnectorFactory
from ..utils import SingletonMeta

# Load the configuration
config = load_config()

# Set ray resources
NUM_GPUS = config.ray.get("num_gpus")
NUM_CPUS = config.ray.get("num_cpus")
N_PARALLEL_INDEXATION = config.ray.get("n_parallel_indexation")

if torch.cuda.is_available():
    gpu, cpu = NUM_GPUS, NUM_CPUS
else:
    gpu, cpu = 0, NUM_CPUS


@ray.remote(
    num_cpus=cpu,
    num_gpus=gpu,
    max_task_retries=2,
    max_restarts=-1,
    concurrency_groups={"compute": N_PARALLEL_INDEXATION},
)
class Indexer(metaclass=SingletonMeta):
    def __init__(self, config, logger, device=None):
        self.config = config
        self.enable_insertion = self.config.vectordb["enable"]
        self.embedder = HFEmbedder(embedder_config=config.embedder, device=device)
        self.serializer = DocSerializer(data_dir=config.paths.data_dir, config=config)
        self.chunker: ABCChunker = ChunkerFactory.create_chunker(
            config, embedder=self.embedder.get_embeddings()
        )
        self.vectordb = ConnectorFactory.create_vdb(
            config, logger=logger, embeddings=self.embedder.get_embeddings()
        )
        self.logger = logger

        # reranker
        self.reranker = None
        self.reranker_enabled = config.reranker["enable"]
        if self.reranker_enabled:
            self.logger.debug(f"Reranker enabled: {self.reranker_enabled}")
            self.reranker = Reranker(self.logger, config)

        self.logger.info("Indexer initialized...")

    async def serialize(self, path: str, metadata: Optional[Dict] = {}):
        self.logger.info(f"Starting serialization of documents from {path}...")
        doc: Document = await self.serializer.serialize_document(
            path, metadata=metadata
        )
        self.logger.info("Serialization completed.")
        return doc

    async def chunk(self, doc: Document, file_path: str):
        if doc is not None:
            self.logger.info("Starting chunking")
            chunks = await self.chunker.split_document(doc)
            self.logger.info(f"Chunking completed for {file_path}")
            return chunks
        else:
            self.logger.info(f"No chunks for {file_path}")
            return []

    @ray.method(concurrency_group="compute")
    async def add_file(
        self,
        path: str | list[str],
        metadata: Optional[Dict] = {},
        partition: Optional[str] = None,
    ):
        partition = self._check_partition_str(partition)
        metadata = {**metadata, "partition": partition}
        doc = await self.serialize(path, metadata=metadata)
        chunks = await self.chunk(doc, path)

        try:
            if self.enable_insertion:
                await self.vectordb.async_add_documents(chunks)
                self.logger.debug(f"Documents {path} added.")
            else:
                self.logger.debug(
                    f"Documents {path} handled but not added to the database."
                )
        except Exception as e:
            self.logger.error(f"An exception as occured: {e}")
            raise Exception(f"An exception as occured: {e}")
        finally:
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    def delete_file(self, file_id: str, partition: str):
        if not self.enable_insertion:
            self.logger.error(
                "Vector database is not enabled, however, the delete_files method was called."
            )
            return
        try:
            # Get points associated with the file name
            points = self.vectordb.get_file_points(file_id, partition)
            if not points:
                self.logger.info(f"No points found for file_id: {file_id}")
                return
            # Delete the points
            self.vectordb.delete_file_points(points, file_id, partition)
            self.logger.info(f"File {file_id} deleted.")
        except Exception as e:
            self.logger.error(f"Error in `delete_files` for file_id {file_id}: {e}")
            raise

        return True

    async def update_file_metadata(self, file_id: str, metadata: dict, partition: str):
        """
        Updates the metadata of a file in the vector database.
        Args:
            file_id (str): The ID of the file to be updated.
            metadata (Dict): The new metadata to be associated with the file.
            collection_name (Optional[str]): The name of the collection in which the file is stored. Defaults to None.
        Returns:
            None
        """
        if not self.enable_insertion:
            self.logger.error(
                "Vector database is not enabled, however, the update_file_metadata method was called."
            )
            return

        try:
            # Get existing chunks associated with the file name
            docs = self.vectordb.get_file_chunks(file_id, partition)

            # Update the metadata
            for doc in docs:
                doc.metadata.update(metadata)

            # Delete the existing chunks
            self.delete_file(file_id, partition)

            # Add the updated chunks
            if self.enable_insertion:
                await self.vectordb.async_add_documents(docs)

            self.logger.info(f"Metadata for file {file_id} updated.")
        except Exception as e:
            self.logger.error(
                f"Error in `update_file_metadata` for file_id {file_id}: {e}"
            )
            raise

    async def asearch(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: int = 0.6,
        partition: Optional[str | List[str]] = None,
        filter: Optional[Dict] = {},
    ) -> List[Document]:
        partition = self._check_partition_list(partition)

        results = await self.vectordb.async_search(
            query=query,
            partition=partition,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            filter=filter,
        )

        # if self.reranker_enabled:
        #     self.logger.debug(f"Reranker enabled: {self.reranker_enabled}")
        #     results = await self.reranker.rerank(query, documents=results, top_k=top_k)

        return results

    def check_file_exists_in_partition(self, file_id: str, partition: str):
        return self.vectordb.file_exists(file_id=file_id, partition=partition)

    def delete_partition(self, partition: str):
        return self.vectordb.delete_partition(partition)

    def sample_chunk_ids(self, partition: str, n_ids: int = 100):
        return self.vectordb.sample_chunk_ids(partition=partition, n_ids=n_ids)

    def _check_partition_list(self, partition: Optional[str]):
        if partition is None:
            self.logger.warning("Partition not provided. Using default partition.")
            partition = [self.default_partition]
        elif isinstance(partition, str):
            partition = [partition]
        elif (not isinstance(partition, list)) or (
            not all(isinstance(p, str) for p in partition)
        ):
            raise ValueError("Partition should be a string or a list of strings.")
        return partition

    def _check_partition_str(self, partition: Optional[str]):
        if partition is None:
            self.logger.warning("Partition not provided. Using default partition.")
            partition = self.default_partition
        elif not isinstance(partition, str):
            raise ValueError("Partition should be a string.")
        return partition

    async def _delegate_vdb_call(self, method_name: str, *args, **kwargs):
        """Execute the method on the local vectordb, handling async methods."""
        method = getattr(self.vectordb, method_name)
        if not callable(method):
            raise AttributeError(f"Method {method_name} not found/callable")

        result = method(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    def _is_method_async(self, method_name: str) -> bool:
        """Helper method to check if a vectordb method is async."""
        method = getattr(self.vectordb, method_name, None)
        return inspect.iscoroutinefunction(method) if method else False
