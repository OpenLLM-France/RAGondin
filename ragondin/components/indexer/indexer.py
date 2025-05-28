import asyncio
import gc
import inspect
import traceback
from typing import Dict, List, Optional

import ray
import torch
from config import load_config
from langchain_core.documents.base import Document
from langchain_openai import OpenAIEmbeddings
from loguru import logger
from ray.util.actor_pool import ActorPool

from .chunker import ABCChunker, ChunkerFactory
from .vectordb import ConnectorFactory

# Load the configuration
config = load_config()

# Set ray resources
NUM_GPUS = config.ray.get("num_gpus")
POOL_SIZE = config.ray.get("pool_size")
MAX_TASKS_PER_WORKER = config.ray.get("max_tasks_per_worker")

if torch.cuda.is_available():
    gpu = NUM_GPUS
else:
    gpu = 0


@ray.remote(num_gpus=gpu, max_task_retries=2)
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
        from langchain_openai import OpenAIEmbeddings
        from loguru import logger

        from .vectordb import ConnectorFactory

        self.config = load_config()
        self.embedder = OpenAIEmbeddings(
            model=self.config.embedder.get("model_name"),
            base_url=self.config.embedder.get("base_url"),
            api_key=self.config.embedder.get("api_key"),
        )
        self.serializer = ray.get_actor("DocSerializer", namespace="ragondin")
        self.chunker: ABCChunker = ChunkerFactory.create_chunker(
            self.config, embedder=self.embedder
        )
        self.vectordb = ConnectorFactory.create_vdb(
            self.config, logger=logger, embeddings=self.embedder
        )
        self.logger = logger
        self.default_partition = "_default"
        self.enable_insertion = self.config.vectordb["enable"]
        self.task_state_manager = ray.get_actor(
            "TaskStateManager", namespace="ragondin"
        )
        self.task_count = 0
        self.logger.info("Indexer worker initialized.")

    async def serialize(self, path: str, metadata: Optional[Dict] = {}):
        self.logger.info(f"Starting serialization of documents from {path}...")
        doc: Document = await self.serializer.serialize_document.remote(
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
        task_id: str,
        path: str | list[str],
        metadata: Optional[Dict] = {},
        partition: Optional[str] = None,
    ):
        self.task_count += 1

        partition = self._check_partition_str(partition)
        metadata = {**metadata, "partition": partition}

        # Serialize document
        self.task_state_manager.set_state.remote(task_id, "SERIALIZING")
        doc = await self.serialize(path, metadata=metadata)

        # Chunk docs
        self.task_state_manager.set_state.remote(task_id, "CHUNKING")
        chunks = await self.chunk(doc, path)

        # Add chunks to the vector database
        try:
            if self.enable_insertion:
                self.task_state_manager.set_state.remote(task_id, "INSERTING")
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

    async def get_task_count(self):
        return self.task_count


@ray.remote(max_restarts=-1)
class Indexer:
    def __init__(self):
        config = load_config()
        self.config = config
        self.logger = logger
        self.enable_insertion = self.config.vectordb["enable"]
        self.embedder = OpenAIEmbeddings(
            model=self.config.embedder.get("model_name"),
            base_url=self.config.embedder.get("base_url"),
            api_key=self.config.embedder.get("api_key"),
        )
        self.vectordb = ConnectorFactory.create_vdb(
            config, logger, embeddings=self.embedder
        )
        self.task_state_manager = ray.get_actor(
            "TaskStateManager", namespace="ragondin"
        )
        self.queue = ray.get_actor("IndexerQueue", namespace="ragondin")
        logger.info("Indexer supervisor actor initialized.")

    async def add_file(self, path, metadata, partition):
        # Retrieve task_id from the Ray runtime context
        task_id = ray.get_runtime_context().get_task_id()

        self.task_state_manager.set_state.remote(task_id, "QUEUED")

        # Get a free worker from the pool
        worker = await self.queue.get_worker.remote()

        try:
            # Send task to the worker
            await worker.add_file.remote(task_id, path, metadata, partition)

        except Exception as e:
            self.logger.error(f"Error in `add_file` for path {path}: {e}")
            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            await self.task_state_manager.set_state.remote(task_id, "FAILED")
            await self.task_state_manager.set_error.remote(task_id, tb)
            raise
        finally:
            task_count = await worker.get_task_count.remote()
            if task_count >= MAX_TASKS_PER_WORKER:
                logger.info("Worker reached max task count, restarting...")
                ray.kill(worker)
                worker = IndexerWorker.remote()
            self.queue.push.remote(worker)
        self.task_state_manager.set_state.remote(task_id, "COMPLETED")

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
        return results

    def delete_partition(self, partition: str):
        return self.vectordb.delete_partition(partition)

    async def get_task_status(self, task_id: str):
        """Get the status of a task."""
        return await self.task_state_manager.get_state.remote(task_id)

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


@ray.remote
class TaskStateManager:
    def __init__(self):
        self.states = {}
        self.errors = {}
        self.lock = asyncio.Lock()

    async def set_state(self, task_id: str, state: str):
        async with self.lock:
            self.states[task_id] = state

    async def set_error(self, task_id: str, tb_str: str):
        async with self.lock:
            self.errors[task_id] = tb_str

    async def get_state(self, task_id: str) -> Optional[str]:
        async with self.lock:
            state = self.states.get(task_id, None)
            return state

    async def get_error(self, task_id: str) -> Optional[str]:
        async with self.lock:
            return self.errors.get(task_id)

    async def get_all_states(self):
        async with self.lock:
            return dict(self.states)


@ray.remote(concurrency_groups={"worker_queue": 1})
class IndexerQueue:
    def __init__(self):
        self.pool = ActorPool([IndexerWorker.remote() for _ in range(POOL_SIZE)])

    @ray.method(concurrency_group="worker_queue")
    async def get_worker(self):
        """Get a free worker from the pool."""
        while True:
            if self.pool.has_free():
                worker = self.pool.pop_idle()
                return worker
            await asyncio.sleep(1)

    def push(self, worker):
        """Push a worker back to the pool."""
        self.pool.push(worker)

    def pool_info(self):
        return {
            "total": POOL_SIZE,
            "idle": len(self.pool._idle_actors),
        }
