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
from utils.dependencies import vectordb

config = load_config()
DATA_DIR = Path(config.paths.data_dir)

ragPipe = RagPipeline(config=config, vectordb=vectordb, logger=logger)


class Tags(Enum):
    VDB = "VectorDB operations"
    LLM = ("LLM Calls",)
    INDEXER = ("Indexer",)
    SEARCH = ("Semantic Search",)
    OPENAI = "OpenAI Compatible API"


class AppState:
    def __init__(self, config):
        self.config = config
        self.model_name = config.llm.model
        self.ragpipe = ragPipe
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


def source2url(s: dict, static_base_url: str):
    s["url"] = f"{static_base_url}/{s['sub_url_path']}"
    s.pop("source")
    s.pop("sub_url_path")
    return s


@app.post(
    "/{partition}/generate",
    summary="Given a question, this endpoint allows to generate an answer grounded on the documents in the VectorDB",
    tags=[Tags.LLM],
)
async def get_answer(
    partition: str,
    new_user_input: str,
    chat_history: list[ChatMsg] = None,
    static_base_url: str = Depends(static_base_url_dependency),
):
    """
    Asynchronously generates an answer to a user's input based on chat history and returns a streaming response.
    Args:
        new_user_input (str): The new input from the user.
        chat_history (list[ChatMsg], optional): The history of chat messages. Defaults to None.
        static_base_url (str, optional): The base URL for static resources. Defaults to the value provided by static_base_url_dependency.
    Returns:
        StreamingResponse: A streaming response containing the generated answer and metadata sources.
    Raises:
        Any exceptions raised by the dependencies or the ragPipe.run method.
    Notes:
        - The function converts the chat history into a list of HumanMessage or AIMessage objects.
        - It runs the ragPipe pipeline to generate an answer stream, context, and sources.
        - The sources are converted to URLs using the static_base_url and included in the response headers as JSON.
    """
    msgs: list[HumanMessage | AIMessage] = None
    if chat_history:
        msgs = [
            mapping[chat_msg.role](content=chat_msg.content)
            for chat_msg in chat_history
        ]
    answer_stream, context, sources = await ragPipe.run(
        partition=[partition], question=new_user_input, chat_history=msgs
    )
    # print(sources)
    sources = list(map(lambda x: source2url(x, static_base_url), sources))
    src_json = json.dumps(sources)

    async def send_chunk():
        """
        Asynchronously sends chunks of data from an answer stream.

        This coroutine function iterates over an asynchronous stream of tokens
        and yields the content of each token.

        Yields:
            str: The content of each token from the answer stream.
        """
        async for token in answer_stream:
            yield token.content

    return StreamingResponse(
        send_chunk(),
        media_type="text/event-stream",
        headers={"X-Metadata-Sources": src_json},
    )


@app.get("/health_check", summary="Toy endpoint to check that the api is up")
async def health_check():
    return "RAG API is up."


mount_chainlit(
    app, "./chainlit/app_front.py", path="/chainlit"
)  # mount the default front


# Mount the search router
app.include_router(search_router, prefix="", tags=[Tags.SEARCH])
# Mount the openai router
app.include_router(openai_router, prefix="/v1", tags=[Tags.OPENAI])
# Mount the indexer router
app.include_router(indexer_router, prefix="/indexer", tags=[Tags.INDEXER])

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8083, reload=True, proxy_headers=True)
