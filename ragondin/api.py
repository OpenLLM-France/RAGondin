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
from utils.dependencies import indexer
import time
import uuid

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

# Classes pour la compatibilité OpenAI
class OpenAIMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class OpenAICompletionRequest(BaseModel):
    model: str = Field(..., description="model name")
    messages: List[OpenAIMessage]
    temperature: Optional[float] = Field(0.7)
    top_p: Optional[float] = Field(1.0)
    stream: Optional[bool] = Field(False)
    max_tokens: Optional[int] = Field(None)

class OpenAICompletionChoice(BaseModel):
    index: int
    message: OpenAIMessage
    finish_reason: str

class OpenAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class OpenAICompletion(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[OpenAICompletionChoice]
    usage: OpenAIUsage

class OpenAICompletionChunkChoice(BaseModel):
    index: int
    delta: Dict[str, str]
    finish_reason: Optional[str] = None

class OpenAICompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[OpenAICompletionChunkChoice]


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

@app.post("/generate/",
          summary="Given a question, this endpoint allows to generate an answer grounded on the documents in the VectorDB",
          tags=[Tags.LLM]
          )
async def get_answer(
    new_user_input: str, 
    chat_history: list[ChatMsg]=None,
    static_base_url: str = Depends(static_base_url_dependency),
    app_state: AppState = Depends(get_app_state)
    ):

    msgs: list[HumanMessage | AIMessage] = None
    if chat_history:
        msgs = [mapping[chat_msg.role](content=chat_msg.content) for chat_msg in chat_history]
    answer_stream, context, sources = await app_state.ragpipe.run(question=new_user_input, chat_history=msgs)
    
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

# Nouvel endpoint compatible OpenAI
@app.post("/v1/chat/completions",
          summary="OpenAI compatible chat completion endpoint using RAG",
          tags=[Tags.OPENAI]
          )
async def openai_chat_completion(
    request: OpenAICompletionRequest,
    static_base_url: str = Depends(static_base_url_dependency),
    app_state: AppState = Depends(get_app_state)
):
    # Récupérer le dernier message utilisateur
    # Et convertir l'historique au format attendu par le pipeline RAG
    user_messages = [msg for msg in request.messages if msg.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="Au moins un message utilisateur est requis")
    
    new_user_input = user_messages[-1].content
    
    # Convertir l'historique de conversation
    chat_history = []
    for msg in request.messages[:-1]:  # Exclure le dernier message utilisateur
        if msg.role in ["user", "assistant"]:
            chat_history.append(ChatMsg(role=msg.role, content=msg.content))
    
    msgs = None
    if chat_history:
        msgs = [mapping[chat_msg.role](content=chat_msg.content) for chat_msg in chat_history]
    
    # Appeler le pipeline RAG avec app_state
    answer_stream, context, sources = await app_state.ragpipe.run(question=new_user_input, chat_history=msgs)
    
    # Traiter les sources
    sources = list(map(lambda x: source2url(x, static_base_url), sources))
    src_json = json.dumps(sources)
    
    # Création de l'ID de réponse
    response_id = f"chatcmpl-{str(uuid.uuid4())}"
    created_time = int(time.time())
    model_name = app_state.model_name
    
    if request.stream:
        # Réponse en streaming compatible OpenAI
        async def stream_response():
            full_response = ""
            # Envoyer un premier chunk avec le rôle
            chunk = OpenAICompletionChunk(
                id=response_id,
                created=created_time,
                model=model_name,
                choices=[
                    OpenAICompletionChunkChoice(
                        index=0,
                        delta={"role": "assistant"},
                        finish_reason=None
                    )
                ]
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
            
            # Envoyer les tokens un par un
            async for token in answer_stream:
                full_response += token.content
                chunk = OpenAICompletionChunk(
                    id=response_id,
                    created=created_time,
                    model=model_name,
                    choices=[
                        OpenAICompletionChunkChoice(
                            index=0,
                            delta={"content": token.content},
                            finish_reason=None
                        )
                    ]
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
            
            # Envoyer un chunk final
            chunk = OpenAICompletionChunk(
                id=response_id,
                created=created_time,
                model=model_name,
                choices=[
                    OpenAICompletionChunkChoice(
                        index=0,
                        delta={},
                        finish_reason="stop"
                    )
                ]
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
            
        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
            headers={"X-Metadata-Sources": src_json},
        )
    else:
        # Réponse non-streaming
        full_response = ""
        async for token in answer_stream:
            full_response += token.content
        
        completion = OpenAICompletion(
            id=response_id,
            created=created_time,
            model=model_name,
            choices=[
                OpenAICompletionChoice(
                    index=0,
                    message=OpenAIMessage(role="assistant", content=full_response),
                    finish_reason="stop"
                )
            ],
            usage=OpenAIUsage(
                prompt_tokens=100,  # Valeurs approximatives
                completion_tokens=len(full_response.split()) * 4 // 3,  # Estimation
                total_tokens=100 + len(full_response.split()) * 4 // 3
            )
        )
        
        return completion

mount_chainlit(app, './chainlit/app_front.py', path="/chainlit") # mount the default front

# Mount the indexer router
app.include_router(indexer_router, prefix="/indexer", tags=[Tags.INDEXER])

if __name__ == "__main__":
    uvicorn.run('api:app', host="0.0.0.0", port=8083, reload=True, proxy_headers=True) # 8083

# uvicorn api:app --reload --port 8083 --host 0.0.0.0