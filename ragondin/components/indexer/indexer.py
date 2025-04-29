import gc
import inspect
from typing import Dict, List, Optional

import ray
import torch
from config import load_config
from langchain_core.documents.base import Document
from loguru import logger

from .chunker import ABCChunker, ChunkerFactory
from .embeddings import HFEmbedder
from .loaders.loader import DocSerializer
from .vectordb import ConnectorFactory
from components.reranker import Reranker

# Load the configuration
config = load_config()

# Set ray resources
NUM_GPUS = config.ray.get("num_gpus")
NUM_CPUS = config.ray.get("num_cpus")

if torch.cuda.is_available():
    gpu, cpu = NUM_GPUS, NUM_CPUS
else:
    gpu, cpu = 0, NUM_CPUS


@ray.remote(num_cpus=cpu, num_gpus=gpu, max_task_retries=2)
class IndexerWorker:
    """This class bridges static files with the vector store database.*"""

    def __init__(self) -> None:
        """
        Initializes the Indexer class with the given configuration, logger, and optional device.

        Args:
            config (Config): Configuration object containing settings for the embedder, paths, llm, and insertion.
            logger (Logger): Logger object for logging information.
            device (str, optional): Device to be used by the embedder. Defaults to None.
        """
        from config import load_config
        from loguru import logger

        from .embeddings import HFEmbedder
        from .vectordb import ConnectorFactory

        self.config = load_config()
        self.embedder = HFEmbedder(embedder_config=self.config.embedder, device=None)
        self.serializer = DocSerializer(
            data_dir=self.config.paths.data_dir, config=self.config
        )
        self.chunker: ABCChunker = ChunkerFactory.create_chunker(
            self.config, embedder=self.embedder.get_embeddings()
        )
        self.vectordb = ConnectorFactory.create_vdb(
            self.config, logger=logger, embeddings=self.embedder.get_embeddings()
        )
        self.logger = logger
        self.n_concurrent_loading = config.insertion.get(
            "n_concurrent_loading", 2
        )  # Number of concurrent loading operations
        self.n_concurrent_chunking = config.insertion.get(
            "n_concurrent_chunking", 2
        )  # Number of concurrent chunking operations
        self.default_partition = "_default"
        self.enable_insertion = self.config.vectordb["enable"]
        self.logger.info("Indexer worker initialized.")

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

    def _check_partition_str(self, partition: Optional[str]):
        if partition is None:
            self.logger.warning("Partition not provided. Using default partition.")
            partition = self.default_partition
        elif not isinstance(partition, str):
            raise ValueError("Partition should be a string.")
        return partition


@ray.remote(max_restarts=-1)
class Indexer:
    def __init__(self):
        config = load_config()
        self.config = config
        device = None
        self.logger = logger
        self.enable_insertion = self.config.vectordb["enable"]
        self.embedder = HFEmbedder(embedder_config=config.embedder, device=device)
        self.vectordb = ConnectorFactory.create_vdb(
            config, logger, embeddings=self.embedder.get_embeddings()
        )
        self.reranker = None
        self.reranker_enabled = config.reranker["enable"]
        self.reranker_top_k = int(config.reranker["top_k"])
        if self.reranker_enabled:
            self.logger.debug("Reranker enabled")
            self.reranker = Reranker(self.logger, config)
        logger.info("Indexer supervisor actor initialized.")

    def get_worker(self):
        return IndexerWorker.remote()

    async def add_file(self, path, metadata, partition):
        worker = self.get_worker()
        await worker.add_file.remote(path, metadata, partition)
        ray.kill(worker)

    def delete_file(self, file_id: str, partition: str):
        """
        Deletes files from the vector database based on the provided filters.
        Args:
            filters (Union[Dict, List[Dict]]): A dictionary or list of dictionaries containing the filters to identify files to be deleted.
            collection_name (Optional[str]): The name of the collection from which files should be deleted. Defaults to None.
        Returns:
            Tuple[List[Dict], List[Dict]]: A tuple containing two lists:
                - deleted_files: A list of filters for the files that were successfully deleted.
                - not_found_files: A list of filters for the files that were not found in the database.
        Raises:
            Exception: If an error occurs during the deletion process, it is logged and the function continues with the next filter.
        """
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
        similarity_threshold: int = 0.80,
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
        if self.reranker_enabled:
            results = await self.reranker.rerank(query, results, self.reranker_top_k)
        return results
    def delete_partition(self, partition: str):
        return self.vectordb.delete_partition(partition)

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
