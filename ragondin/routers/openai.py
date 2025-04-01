import json
import time
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from models.openai import (
    OpenAICompletion,
    OpenAICompletionChoice,
    OpenAICompletionChunk,
    OpenAICompletionChunkChoice,
    OpenAICompletionRequest,
    OpenAIMessage,
    OpenAIUsage,
)
from pydantic import BaseModel


# Classe pour les messages du chat
class ChatMsg(BaseModel):
    role: Literal["user", "assistant"]
    content: str


# Mapping pour les types de messages
mapping = {"user": HumanMessage, "assistant": AIMessage}

# Créer un router pour les endpoints OpenAI
router = APIRouter()


# Fonctions utilitaires pour le router
def get_app_state(request: Request):
    return request.app.state.app_state


def static_base_url_dependency(request: Request) -> str:
    return f"{request.url.scheme}://{request.url.hostname}:{request.url.port}/static"


def source2url(s: dict, static_base_url: str):
    s["url"] = f"{static_base_url}/{s['sub_url_path']}"
    s.pop("source")
    s.pop("sub_url_path")
    return s


@router.post(
    "/chat/completions", summary="OpenAI compatible chat completion endpoint using RAG"
)
async def openai_chat_completion(
    request: OpenAICompletionRequest,
    static_base_url: str = Depends(static_base_url_dependency),
    app_state=Depends(get_app_state),
):
    # Get the last user message
    # And convert the chat history to the format expected by the RAG pipeline
    user_messages = [msg for msg in request.messages if msg.role == "user"]
    if not user_messages:
        raise HTTPException(
            status_code=400, detail="At least one user message is required"
        )

    new_user_input = user_messages[-1].content

    # Convert chat history to the format expected by the RAG pipeline
    chat_history = []
    for msg in request.messages[:-1]:  # Exclure le dernier message utilisateur
        if msg.role in ["user", "assistant"]:
            chat_history.append(ChatMsg(role=msg.role, content=msg.content))

    msgs = None
    if chat_history:
        msgs = [
            mapping[chat_msg.role](content=chat_msg.content)
            for chat_msg in chat_history
        ]

    # Run RAG pipeline
    answer_stream, context, sources = await app_state.ragpipe.run(
        partition=["all"], question=new_user_input, chat_history=msgs
    )

    # Handle the sources
    sources = list(map(lambda x: source2url(x, static_base_url), sources))
    src_json = json.dumps(sources)

    # Create response-id
    response_id = f"chatcmpl-{str(uuid.uuid4())}"
    created_time = int(time.time())
    model_name = app_state.model_name

    if request.stream:
        # Openai compatible streaming response
        async def stream_response():
            full_response = ""
            chunk = OpenAICompletionChunk(
                id=response_id,
                created=created_time,
                model=model_name,
                choices=[
                    OpenAICompletionChunkChoice(
                        index=0, delta={"role": "assistant"}, finish_reason=None
                    )
                ],
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
                            finish_reason=None,
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

            # Send final chunk
            chunk = OpenAICompletionChunk(
                id=response_id,
                created=created_time,
                model=model_name,
                choices=[
                    OpenAICompletionChunkChoice(index=0, delta={}, finish_reason="stop")
                ],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
            headers={"X-Metadata-Sources": src_json},
        )
    else:
        # Non streaming response
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
                    finish_reason="stop",
                )
            ],
            usage=OpenAIUsage(
                prompt_tokens=100,  # Valeurs approximatives
                completion_tokens=len(full_response.split()) * 4 // 3,  # Estimation
                total_tokens=100 + len(full_response.split()) * 4 // 3,
            ),
        )

        return completion
