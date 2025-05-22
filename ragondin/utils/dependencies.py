import ray
import ray.actor
from components import ABCVectorDB
from components.indexer.indexer import Indexer
from components.indexer.indexer_deployment import Indexer as IndexerForDeployment
from components.indexer.indexer_deployment import TaskStateManager
from config import load_config
from loguru import logger
from ray.util.state import get_task
import asyncio
from typing import Optional

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

# Initialize components once
local_deployment = config.ray["local_deployment"]

if local_deployment:
    indexer = Indexer.remote(config, logger)
else:
    indexer = IndexerForDeployment.remote()

vectordb: ABCVectorDB = VDBProxy(
    indexer_actor=indexer
)  # vectordb is not of type ABCVectorDB, but it mimics it


def get_indexer():
    return indexer


logger.info("Starting TaskStateManager actor")
task_state_manager = TaskStateManager.options(
            name="TaskStateManager",
            lifetime="detached",
            namespace="ragondin"
        ).remote()