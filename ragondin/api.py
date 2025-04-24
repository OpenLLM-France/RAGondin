import ray

if not ray.is_initialized():
    ray.init(dashboard_host="0.0.0.0")

import json
from enum import Enum
from pathlib import Path
from typing import Literal

import uvicorn
from chainlit.utils import mount_chainlit
from components import RagPipeline
from config import load_config
from fastapi import Depends, FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from loguru import logger
from pydantic import BaseModel
from routers.indexer import router as indexer_router

from routers.openai import router as openai_router
from routers.search import router as search_router
from routers.extract import router as extract_router
from routers.partition import router as partition_router
from utils.dependencies import vectordb

config = load_config()
DATA_DIR = Path(config.paths.data_dir)

ragPipe = RagPipeline(config=config, vectordb=vectordb, logger=logger)


class Tags(Enum):
    VDB = "VectorDB operations"
    INDEXER = ("Indexer",)
    SEARCH = ("Semantic Search",)
    OPENAI = ("OpenAI Compatible API",)
    EXTRACT = ("Document extracts",)
    PARTITION = ("Partitions & files",)


class AppState:
    def __init__(self, config):
        self.config = config
        self.model_name = config.llm.model
        self.ragpipe = ragPipe
        self.vectordb = vectordb
        self.data_dir = Path(config.paths.data_dir)


class ChatMsg(BaseModel):
    role: Literal["user", "assistant"]
    content: str


mapping = {"user": HumanMessage, "assistant": AIMessage}

app = FastAPI()

app.state.app_state = AppState(config)
app.mount(
    "/static", StaticFiles(directory=DATA_DIR.resolve(), check_dir=True), name="static"
)


def static_base_url_dependency(request: Request) -> str:
    return f"{request.url.scheme}://{request.url.hostname}:{request.url.port}/static"


@app.get("/health_check", summary="Toy endpoint to check that the api is up")
async def health_check(static_base_url: str = Depends(static_base_url_dependency)):
    logger.info(f"URL: {static_base_url}")
    return "RAG API is up."


# Mount the default front
# mount_chainlit(app, "./chainlit/app_front2.py", path="/chainlit")
mount_chainlit(app, "./chainlit/app_front.py", path="/chainlit")

# Mount the indexer router
app.include_router(indexer_router, prefix="/indexer", tags=[Tags.INDEXER])
# Mount the extract router
app.include_router(extract_router, prefix="/extract", tags=[Tags.EXTRACT])
# Mount the search router
app.include_router(search_router, prefix="/search", tags=[Tags.SEARCH])
# Mount the partition router
app.include_router(partition_router, prefix="/partition", tags=[Tags.PARTITION])
# Mount the openai router
app.include_router(openai_router, prefix="/v1", tags=[Tags.OPENAI])


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8083, reload=True, proxy_headers=True)
