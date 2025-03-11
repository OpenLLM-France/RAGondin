import asyncio
from typing import Annotated
from fastapi import FastAPI, File, UploadFile, HTTPException, status, Request, Depends
from enum import Enum
import json
import os
from pydantic import BaseModel
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
from utils.dependencies import indexer

config = load_config()
DATA_DIR = Path(config.paths.data_dir)

ragPipe = RagPipeline(config=config, vectordb=indexer.vectordb, logger=logger)

class Tags(Enum):
    VDB = "VectorDB operations"
    LLM = "LLM Calls",
    INDEXER = "Indexer"

class ChatMsg(BaseModel):
    role: Literal["user", "assistant"]
    content: str

mapping = {
    "user": HumanMessage, 
    "assistant": AIMessage
}

app = FastAPI()
app.mount('/static', StaticFiles(directory=DATA_DIR.resolve(), check_dir=True), name='static')


def static_base_url_dependency(request: Request) -> str:
    return f"{request.url.scheme}://{request.client.host}:{request.url.port}/static"


def source2url(s: dict, static_base_url: str):
    s['url'] = f"{static_base_url}/{s['sub_url_path']}"
    s.pop("source")
    s.pop('sub_url_path')
    return s


@app.get("/collections/",
          summary="Get existant collections",
          tags=[Tags.VDB]
          )
async def get_collections() -> list[str]:
    return await indexer.vectordb.get_collections()


@app.post("/generate/",
          summary="Given a question, this endpoint allows to generate an answer grounded on the documents in the VectorDB",
          tags=[Tags.LLM]
          )
async def get_answer(
    new_user_input: str, chat_history: list[ChatMsg]=None,
    static_base_url: str = Depends(static_base_url_dependency)
    ):

    msgs: list[HumanMessage | AIMessage] = None
    if chat_history:
        msgs = [mapping[chat_msg.role](content=chat_msg.content) for chat_msg in chat_history]
    answer_stream, context, sources = await ragPipe.run(question=new_user_input, chat_history=msgs)
    # print(sources)
    
    sources = list(map(lambda x: source2url(x, static_base_url), sources))
    src_json = json.dumps(sources)

    async def send_chunk():
        async for token in answer_stream:
            yield token.content
          
    return StreamingResponse(
        send_chunk(), 
        media_type="text/event-stream",
        headers={"X-Metadata-Sources": src_json},
    )

@app.get("/health_check/", summary="Toy endpoint to check that the api is up")
async def health_check():
    return "RAG API is up."


mount_chainlit(app, './chainlit/app_front.py', path="/chainlit") # mount the default front

# Mount the indexer router
app.include_router(indexer_router, prefix="/indexer", tags=[Tags.INDEXER])


if __name__ == "__main__":
    uvicorn.run('api:app', host="0.0.0.0", port=8083, reload=True, proxy_headers=True) # 8083

# uvicorn api:app --reload --port 8083 --host 0.0.0.0