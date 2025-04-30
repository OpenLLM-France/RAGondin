import os

import ray
from dotenv import dotenv_values

SHARED_ENV = os.environ.get("SHARED_ENV", None)

env_vars = dotenv_values(SHARED_ENV) if SHARED_ENV else {}
env_vars["PYTHONPATH"] = "/app/ragondin"


ray.init(
    address="auto", runtime_env={"working_dir": "/app/ragondin", "env_vars": env_vars}
)

import json
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

import uvicorn
from chainlit.utils import mount_chainlit
from config import load_config
from fastapi import Depends, FastAPI, Request, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from langchain_core.messages import AIMessage, HumanMessage
from loguru import logger
from pydantic import BaseModel
from routers.extract import router as extract_router
from routers.indexer import router as indexer_router

from routers.openai import router as openai_router
from routers.partition import router as partition_router
from routers.search import router as search_router
from utils.dependencies import vectordb
import os

from components import RagPipeline

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
        self.ragpipe = ragPipe
        self.vectordb = vectordb
        self.data_dir = Path(config.paths.data_dir)


# Read the token from env (or None if not set)
AUTH_TOKEN: Optional[str] = os.getenv("AUTH_TOKEN")
security = HTTPBearer()


# Dependency to verify token
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if AUTH_TOKEN is None:
        return  # Auth disabled
    if credentials.credentials != AUTH_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid or missing token"
        )


# Apply globally only if AUTH_TOKEN is set
dependencies = [Depends(verify_token)] if AUTH_TOKEN else []
app = FastAPI(dependencies=dependencies)


app.state.app_state = AppState(config)
app.mount(
    "/static", StaticFiles(directory=DATA_DIR.resolve(), check_dir=True), name="static"
)


@app.get("/health_check", summary="Toy endpoint to check that the api is up")
async def health_check(request: Request):
    # TODO : Error reporting about llm and vlm
    return "RAG API is up."


# Mount the default front
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
    uvicorn.run("api:app", host="0.0.0.0", port=8080, reload=True, proxy_headers=True)
