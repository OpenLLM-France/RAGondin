from components.indexer.embeddings import HFEmbedder
from components.indexer.indexer import Indexer
from components.indexer.vectordb import ConnectorFactory
from config import load_config
from loguru import logger

# load config
config = load_config()
# Initialize components once
indexer = Indexer.remote(config, logger)
embedder = HFEmbedder(embedder_config=config.embedder)
vectordb = ConnectorFactory.create_vdb(
    config, logger=logger, embeddings=embedder.get_embeddings()
)


def get_indexer():
    return indexer
