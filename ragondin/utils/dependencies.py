import ray
import ray.actor
from config import load_config

from components import ABCVectorDB
from components.indexer.indexer import Indexer, TaskStateManager
from components.indexer.loaders.pdf_loaders.marker import MarkerPool
from components.indexer.loaders.serializer import SerializerQueue

def get_or_create_actor(name, cls, namespace="ragondin", **options):
    from utils.logger import get_logger
    logger= get_logger()
    logger.info(f"Getting or creating actor: {name} in namespace: {namespace}")
    try:
        return ray.get_actor(name, namespace=namespace)
    except Exception as e:
        logger.info(f"Actor {name} not found, creating a new one: {e}")
        return cls.options(name=name, namespace=namespace, **options).remote()
    
class VDBProxy:
    """Class that delegates method calls to the remote vectordb."""

    def __init__(self, indexer_actor: ray.actor.ActorHandle):
        self.indexer_actor = indexer_actor  # Reference to the remote actor

    def __getattr__(self, method_name):
        # Check if the method is async on the remote vectordb
        is_async = ray.get(self.indexer_actor._is_method_async.remote(method_name))

        if is_async:
            # Return an async coroutine for async methods
            async def async_wrapper(*args, **kwargs):
                result_ref = self.indexer_actor._delegate_vdb_call.remote(
                    method_name, *args, **kwargs
                )
                return await result_ref

            return async_wrapper

        else:
            # Return a blocking wrapper for sync methods
            def sync_wrapper(*args, **kwargs):
                return ray.get(
                    self.indexer_actor._delegate_vdb_call.remote(
                        method_name, *args, **kwargs
                    )
                )

            return sync_wrapper


# load config
config = load_config()


def get_vectordb() -> ABCVectorDB:
    indexer = ray.get_actor("Indexer", namespace="ragondin")
    return VDBProxy(indexer_actor=indexer)

def get_task_state_manager():
    return get_or_create_actor("TaskStateManager", TaskStateManager, lifetime="detached")
def get_serializer_queue():
    return get_or_create_actor("SerializerQueue", SerializerQueue)
def get_marker_pool():
    if config.loader.file_loaders.get("pdf") == "MarkerLoader":
        return get_or_create_actor("MarkerPool", MarkerPool)
def get_indexer():
    return get_or_create_actor("Indexer", Indexer)
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@ray.remote
class Test:
    def __init__(self):
        from dataclasses import dataclass, field
        from typing import Any, Dict, Optional
        @dataclass
        class TaskInfo:
            state: Optional[str] = None
            error: Optional[str] = None
            details: Dict[str, Any] = field(default_factory=dict)
        self.TaskInfo = TaskInfo
        self.tasks: Dict[str, TaskInfo] = {}
        self.lock = asyncio.Lock()

    async def _ensure_task(self, task_id: str):
        """Helper to get-or-create the TaskInfo object under lock."""
        if task_id not in self.tasks:
            self.tasks[task_id] = self.TaskInfo()
        return self.tasks[task_id]

    async def set_state(self, task_id: str, state: str):
        async with self.lock:
            info = await self._ensure_task(task_id)
            info.state = state

    async def set_error(self, task_id: str, tb_str: str):
        async with self.lock:
            info = await self._ensure_task(task_id)
            info.error = tb_str

    async def set_details(
        self, task_id: str, *, file_id: str, partition: int, metadata: dict
    ):
        async with self.lock:
            info = await self._ensure_task(task_id)
            info.details = {
                "file_id": file_id,
                "partition": partition,
                "metadata": metadata,
            }

    async def get_state(self, task_id: str) -> Optional[str]:
        async with self.lock:
            info = self.tasks.get(task_id)
            return info.state if info else None

    async def get_error(self, task_id: str) -> Optional[str]:
        async with self.lock:
            info = self.tasks.get(task_id)
            return info.error if info else None

    async def get_details(self, task_id: str) -> Optional[dict]:
        async with self.lock:
            info = self.tasks.get(task_id)
            return info.details if info else None

    async def get_all_states(self) -> Dict[str, str]:
        async with self.lock:
            return {tid: info.state for tid, info in self.tasks.items()}

    async def get_all_info(self) -> Dict[str, dict]:
        async with self.lock:
            return {
                task_id: {
                    "state": info.state,
                    "error": info.error,
                    "details": info.details,
                }
                for task_id, info in self.tasks.items()
            }

test1 = Test.remote()
test2 = Test.remote()
test3 = get_or_create_actor("TestActor", Test)