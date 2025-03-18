import asyncio
from typing import Annotated, List, Optional, Dict, Any, Union
from fastapi import FastAPI, File, UploadFile, HTTPException, status, Request, Depends
from enum import Enum
import json
import os
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
import uvicorn
from typing import Literal
from pathlib import Path
from components import RagPipeline
from config import load_config
from loguru import logger
from chainlit.utils import mount_chainlit
from routers.indexer import router as indexer_router
from routers.openai_api import router as openai_router
from utils.dependencies import indexer
import time
import uuid
# Importer les modèles depuis le nouveau module
from models.chatCompletion import (
    OpenAIMessage, OpenAICompletionRequest, OpenAICompletionChoice,
    OpenAIUsage, OpenAICompletion, OpenAICompletionChunkChoice,
    OpenAICompletionChunk
)

# Charger la configuration
config = load_config()
DATA_DIR = Path(config.paths.data_dir)

# Classe pour stocker les ressources globales
class AppState:
    def __init__(self, config):
        self.config = config
        self.model_name = config.llm.model
        self.ragpipe = RagPipeline(config=config, vectordb=indexer.vectordb, logger=logger)
        self.data_dir = Path(config.paths.data_dir)

# Initialiser l'état de l'application
app_state = AppState(config)

# Les classes liées à l'API OpenAI
class Tags(Enum):
    VDB = "VectorDB operations"
    LLM = "LLM Calls",
    INDEXER = "Indexer"
    OPENAI = "OpenAI Compatible API"

class ChatMsg(BaseModel):
    role: Literal["user", "assistant"]
    content: str

mapping = {
    "user": HumanMessage, 
    "assistant": AIMessage
}

app = FastAPI()
app.state.app_state = app_state
app.mount('/static', StaticFiles(directory=DATA_DIR.resolve(), check_dir=True), name='static')


def get_app_state(request: Request) -> AppState:
    return request.app.state.app_state

def static_base_url_dependency(request: Request) -> str:
    return f"{request.url.scheme}://{request.client.host}:{request.url.port}/static"

def source2url(s: dict, static_base_url: str):
    s['url'] = f"{static_base_url}/{s['sub_url_path']}"
    s.pop("source")
    s.pop('sub_url_path')
    return s



@app.get("/health_check/", summary="Toy endpoint to check that the api is up")
async def health_check():
    return "RAG API is up."

mount_chainlit(app, './chainlit/app_front.py', path="/chainlit") # mount the default front

# Mount the indexer router
app.include_router(indexer_router, prefix="/indexer", tags=[Tags.INDEXER])

# Mount the OpenAI compatible API router
app.include_router(openai_router, prefix="/v1", tags=[Tags.OPENAI])

if __name__ == "__main__":
    uvicorn.run('api:app', host="0.0.0.0", port=8083, reload=True, proxy_headers=True) # 8083

# uvicorn api:app --reload --port 8083 --host 0.0.0.0