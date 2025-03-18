from components import Indexer
from config.config import load_config
from fastapi import APIRouter, Depends
from utils.dependencies import get_indexer

# load config
config = load_config()
DATA_DIR = config.paths.data_dir
# Create an APIRouter instance
router = APIRouter()


@router.get("/collections/", summary="Get existant collections")
async def get_collections(indexer: Indexer = Depends(get_indexer)) -> list[str]:
    return await indexer.vectordb.get_collections()