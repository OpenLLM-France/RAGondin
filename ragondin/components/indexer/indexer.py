import asyncio
import gc
import inspect
import traceback
from typing import Dict, List, Optional, Union

import ray
import torch
from config import load_config
from langchain_core.documents.base import Document
from langchain_openai import OpenAIEmbeddings
from loguru import logger

from .chunker import BaseChunker, ChunkerFactory
from .vectordb import ConnectorFactory


@ray.remote(max_restarts=-1, concurrency_groups={"insertion": 1})
class Indexer:
    def __init__(self):
        # Load config, logger
        self.config = load_config()
        self.logger = logger

        # Initialize the embeddings
        self.embedder = OpenAIEmbeddings(
            model=self.config.embedder.get("model_name"),
            base_url=self.config.embedder.get("base_url"),
            api_key=self.config.embedder.get("api_key"),
        )

        # Get the global serializer queue actor by name
        self.serializer_queue = ray.get_actor("SerializerQueue", namespace="ragondin")

        # Initialize chunker
        self.chunker: BaseChunker = ChunkerFactory.create_chunker(
            self.config, embedder=self.embedder
        )

        # Initialize vectordb connector
        self.vectordb = ConnectorFactory.create_vdb(
            self.config, self.logger, embeddings=self.embedder
        )

        # Task‐state actor (to record states & errors)
        self.task_state_manager = ray.get_actor(
            "TaskStateManager", namespace="ragondin"
        )

        self.default_partition = "_default"
        self.enable_insertion = self.config.vectordb["enable"]
        self.handle = ray.get_actor("Indexer", namespace="ragondin")
        self.logger.info("Indexer actor initialized.")

    async def serialize(
        self, task_id: str, path: str, metadata: Optional[Dict] = {}
    ) -> Document:
        self.logger.info(f"Starting serialization of documents from {path}...")

        # Call the remote serializer
        doc: Document = await self.serializer_queue.submit_document.remote(
            task_id, path, metadata=metadata
        )
        self.logger.info(f"Serialization completed for {path}")
        return doc

    async def chunk(self, doc: Document, file_path: str) -> List[Document]:
        self.logger.info(f"Starting chunking for {file_path}")
        chunks = await self.chunker.split_document(doc)
        self.logger.info(f"Chunking completed for {file_path}")
        return chunks

    async def add_file(
        self,
        path: Union[str, List[str]],
        metadata: Optional[Dict] = {},
        partition: Optional[str] = None,
    ):
        """
        Index a file into the vector database.
        """
        task_id = ray.get_runtime_context().get_task_id()
        self.task_state_manager.set_state.remote(task_id, "QUEUED")
        # 1. Check/normalize partition
        partition = self._check_partition_str(partition)

        # 2. Merge partition into metadata
        metadata = {**metadata, "partition": partition}

        # 3. Serialize
        doc = await self.serialize(task_id, path, metadata=metadata)
        # 4. Chunk
        self.task_state_manager.set_state.remote(task_id, "CHUNKING")
        chunks = await self.chunk(doc, str(path))

        # 5. Insert into vectordb (if enabled)
        if self.enable_insertion and chunks:
            try:
                await self.task_state_manager.set_state.remote(task_id, "INSERTING")
                await self.handle.insert_documents.remote(chunks)
                self.logger.debug(f"Documents {path} added to vectordb.")
            except Exception as e:
                self.logger.error(f"Error during insertion: {e}")
                tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                await self.task_state_manager.set_state.remote(task_id, "FAILED")
                await self.task_state_manager.set_error.remote(task_id, tb)
                raise
            finally:
                # Clean up GPU memory if needed
                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
        else:
            self.logger.debug(
                f"Vectordb insertion skipped (enable_insertion={self.enable_insertion})."
            )

        # 6. Mark task as completed
        await self.task_state_manager.set_state.remote(task_id, "COMPLETED")

        return True

    @ray.method(concurrency_group="insertion")
    async def insert_documents(self, chunks):
        try:
            await self.vectordb.async_add_documents(chunks)
        except Exception as e:
            self.logger.error(f"Error inserting documents: {e}")
            raise

    def delete_file(self, file_id: str, partition: str) -> bool:
        """
        Delete all chunks associated with a given file_id in a partition.
        """
        if not self.enable_insertion:
            self.logger.error(
                "Vector database is not enabled, but delete_file was called."
            )
            return False

        try:
            points = self.vectordb.get_file_points(file_id, partition)
            if not points:
                self.logger.info(f"No points found for file_id: {file_id}")
                return False

            self.vectordb.delete_file_points(points, file_id, partition)
            self.logger.info(f"Deleted file {file_id} from partition {partition}.")
            return True
        except Exception as e:
            self.logger.error(f"Error in delete_file for {file_id}: {e}")
            raise

    async def update_file_metadata(self, file_id: str, metadata: Dict, partition: str):
        """
        Update metadata for all chunks of a given file_id.
        This re‐inserts the chunks with updated metadata.
        """
        if not self.enable_insertion:
            self.logger.error(
                "Vector database is not enabled, but update_file_metadata was called."
            )
            return

        try:
            # 1. Fetch existing chunks for file_id
            docs = self.vectordb.get_file_chunks(file_id, partition)

            # 2. Update metadata
            for doc in docs:
                doc.metadata.update(metadata)

            # 3. Delete old chunks
            self.delete_file(file_id, partition)

            # 4. Insert updated chunks
            await self.vectordb.async_add_documents(docs)
            self.logger.info(f"Metadata updated for file {file_id}.")
        except Exception as e:
            self.logger.error(f"Error in update_file_metadata for {file_id}: {e}")
            raise

    async def asearch(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.80,
        partition: Optional[Union[str, List[str]]] = None,
        filter: Optional[Dict] = {},
    ) -> List[Document]:
        """
        Asynchronously search the vector database for documents matching the query.
        """
        partition_list = self._check_partition_list(partition)
        results = await self.vectordb.async_search(
            query=query,
            partition=partition_list,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            filter=filter,
        )
        return results

    async def get_task_status(self, task_id: str) -> Optional[str]:
        return await self.task_state_manager.get_state.remote(task_id)

    def _check_partition_str(self, partition: Optional[str]) -> str:
        """
        Normalize a single partition string (or default) to a valid string.
        """
        if partition is None:
            self.logger.warning("partition not provided; using default.")
            return self.default_partition
        if not isinstance(partition, str):
            raise ValueError("Partition must be a string.")
        return partition

    def delete_partition(self, partition: str):
        return self.vectordb.delete_partition(partition)

    def _check_partition_list(
        self, partition: Optional[Union[str, List[str]]]
    ) -> List[str]:
        """
        Normalize partition to a list of strings for searching.
        """
        if partition is None:
            self.logger.warning("partition not provided; using default.")
            return [self.default_partition]
        if isinstance(partition, str):
            return [partition]
        if isinstance(partition, list) and all(isinstance(p, str) for p in partition):
            return partition
        raise ValueError("Partition must be a string or a list of strings.")

    async def _delegate_vdb_call(self, method_name: str, *args, **kwargs):
        method = getattr(self.vectordb, method_name, None)
        if method is None or not callable(method):
            raise AttributeError(f"Method {method_name} not found on vectordb.")
        result = method(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    def _is_method_async(self, method_name: str) -> bool:
        """
        Check if a vectordb method is defined as a coroutine.
        """
        method = getattr(self.vectordb, method_name, None)
        return inspect.iscoroutinefunction(method) if method else False


@ray.remote
class TaskStateManager:
    """
    Actor to manage task states and errors.
    """

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
