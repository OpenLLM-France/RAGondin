from components import ConnectorFactory, HFEmbedder, Indexer, ABCVectorDB
from config import load_config
from loguru import logger
import ray


class VDBProxy:
    """Serializable class that delegates method calls to the remote vectordb."""

    def __init__(self, indexer_actor: ray.actor.ActorHandle):
        self.indexer_actor = indexer_actor  # Reference to the remote actor

    def __getattr__(self, method_name):
        def remote_method(*args, **kwargs):
            return ray.get(
                self.indexer_actor._delegate_vdb_call.remote(
                    method_name, *args, **kwargs
                )
            )

        return remote_method


# load config
config = load_config()
# Initialize components once
indexer = Indexer.remote(config, logger)
vectordb: ABCVectorDB = VDBProxy(indexer_actor=indexer)


def get_indexer():
    return indexer
