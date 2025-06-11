import asyncio
import gc
from pathlib import Path
from typing import Dict, Optional, Union

import ray
import torch
from config import load_config
from langchain_core.documents.base import Document

from . import get_loader_classes

config = load_config()

# Set ray resources
if torch.cuda.is_available():
    NUM_GPUS = config.ray.get("num_gpus")
else: # On CPU
    NUM_GPUS = 0

POOL_SIZE = config.ray.get("pool_size")
MAX_TASKS_PER_WORKER = config.ray.get("max_tasks_per_worker")


@ray.remote(num_gpus=NUM_GPUS)
class DocSerializer:
    def __init__(self, data_dir=None, **kwargs) -> None:
        from config import load_config
        from loguru import logger

        self.logger = logger
        self.config = load_config()
        self.data_dir = data_dir
        self.kwargs = kwargs
        self.kwargs["config"] = self.config
        self.save_markdown = self.config.loader.get("save_markdown", False)
        self.task_state_manager = ray.get_actor(
            "TaskStateManager", namespace="ragondin"
        )
        # Initialize loader classes:
        self.loader_classes = get_loader_classes(config=self.config)

    async def serialize_document(
        self,
        task_id: str,
        path: Union[str, Path],
        metadata: Optional[Dict] = {},
    ) -> Document:
        # Set task state
        self.task_state_manager.set_state.remote(task_id, "SERIALIZING")

        p = Path(path)
        file_ext = p.suffix

        # Get appropriate loader for the file type
        loader_cls = self.loader_classes.get(file_ext)
        if loader_cls is None:
            self.logger.info(f"No loader available for {p.name}")
            return None

        self.logger.debug(f"Loading document: {p.name}")
        loader = loader_cls(**self.kwargs)  # Propagate kwargs here!

        metadata = {"page_sep": loader.page_sep, **metadata}

        try:
            # Load the doc
            doc: Document = await loader.aload_document(
                file_path=path, metadata=metadata, save_markdown=self.save_markdown
            )

            # Clean up resources
            del loader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            return doc
        except Exception as e:
            self.logger.error(f"Error loading document {path}: {e}")
            raise e


@ray.remote
class SerializerQueue:
    """
    Manages a pool of DocSerializer actors. Each actor can handle up to N
    concurrent serialize_document() calls. The total concurrency is M * N,
    where M = POOL_SIZE and N = MAX_TASKS_PER_WORKER.
    """

    def __init__(self):
        from loguru import logger

        self.logger = logger
        # 1) Spawn M DocSerializer actors (no config argument passed)
        self.actors = [DocSerializer.remote() for _ in range(POOL_SIZE)]

        # 2) Keep track of how many tasks each actor is handling
        self.load: Dict[ray.actor.ActorHandle, int] = {a: 0 for a in self.actors}

        # 3) A lock so that picking/incrementing is atomic
        self._lock = asyncio.Lock()

        self.logger.info(
            f"SerializerQueue initialized with {POOL_SIZE} actors, each with a max capacity of {MAX_TASKS_PER_WORKER} concurrent tasks."
        )

    async def submit_document(
        self,
        task_id: str,
        path: Union[str, Path],
        metadata: Optional[Dict] = {},
    ) -> Document:
        """
        Find any DocSerializer actor whose current load < MAX_TASKS_PER_WORKER,
        increment its load, call serialize_document on it, then decrement load.
        If none are immediately available, wait until one frees up.
        """

        # Get an available actor for the task
        while True:
            async with self._lock:
                for actor, count in self.load.items():
                    if count < MAX_TASKS_PER_WORKER:
                        chosen = actor
                        self.load[actor] += 1
                        break
                else:
                    chosen = None  # none free

            if chosen is not None:
                break  # we have an actor
            await asyncio.sleep(0.1)  # wait *without* holding the lock

        # Serialize_document on that actor
        try:
            doc: Document = await chosen.serialize_document.remote(
                task_id, path, metadata
            )
        except Exception as e:
            self.logger.error(f"Error serializing document {path}: {e}")
            raise
        finally:
            # Free up the actor
            async with self._lock:
                self.load[chosen] -= 1

        return doc

    async def pool_info(self) -> Dict[str, int]:
        """
        Return a summary of current load per actor and total capacity.
        """
        total_capacity = POOL_SIZE * MAX_TASKS_PER_WORKER
        current_load = sum(self.load.values())
        return {
            "pool_size": POOL_SIZE,
            "max_tasks_per_worker": MAX_TASKS_PER_WORKER,
            "total_capacity": total_capacity,
            "current_load": current_load,
            "free_slots": total_capacity - current_load,
        }
