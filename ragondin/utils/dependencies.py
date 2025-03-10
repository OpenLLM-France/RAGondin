from components.indexer.indexer import Indexer
from config import load_config
from loguru import logger

# load config
config = load_config()
# Initialize components once
indexer = Indexer(config, logger)

def get_indexer():
    return indexer