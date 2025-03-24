from components import ConnectorFactory, HFEmbedder, Indexer
from config import load_config
from loguru import logger

# load config
config = load_config()
# Initialize components once
indexer = Indexer.remote(config, logger)
embedder = HFEmbedder(embedder_config=config.embedder, device="cpu")


vectordb = ConnectorFactory.create_vdb(
    config, logger=logger, embeddings=embedder.get_embeddings()
)


def get_indexer():
    return indexer
