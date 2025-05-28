import ray
import ray.actor
from components import ABCVectorDB
from components.indexer.indexer import Indexer, IndexerQueue, TaskStateManager
from components.indexer.loaders.serializer import DistDocSerializer
from config import load_config


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

# Create global indexer supervisor actor
indexer = Indexer.options(name="Indexer", namespace="ragondin").remote()

# Create vectordb instance
vectordb: ABCVectorDB = VDBProxy(
    indexer_actor=indexer
)  # vectordb is not of type ABCVectorDB, but it mimics it


def get_indexer():
    return indexer


# Create task state manager actor
task_state_manager = TaskStateManager.options(
    name="TaskStateManager", lifetime="detached", namespace="ragondin"
).remote()

# Create indexer queue actor
indexer_queue = IndexerQueue.options(name="IndexerQueue", namespace="ragondin").remote()

# Create document serializer actor
serializer = DistDocSerializer.options(
    name="DocSerializer", namespace="ragondin"
).remote(data_dir=config.paths.data_dir, config=config)
