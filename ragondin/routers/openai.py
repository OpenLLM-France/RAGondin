import json
import time
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from langchain_core.messages import AIMessage, HumanMessage
from models.openai import (
    OpenAIChatCompletion,
    OpenAIChatCompletionChoice,
    OpenAICompletion,
    OpenAICompletionChoice,
    OpenAICompletionChunk,
    OpenAICompletionChunkChoice,
    OpenAIChatCompletionRequest,
    OpenAICompletionRequest,
    OpenAIMessage,
    OpenAIUsage,
    OpenAILogprobs,
    ChoiceLogprobs,
    ChatCompletionTokenLogprob,
)
from pydantic import BaseModel
from urllib.parse import urlparse, quote
from loguru import logger


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


@router.get("/models", summary="OpenAI-compatible model listing endpoint")
async def list_models(app_state=Depends(get_app_state)):
    # Get available partitions from your backend
    partitions = app_state.vectordb.list_partitions() + ["all"]

    # Format them as OpenAI models
    models = [
        {
            "id": f"ragondin-{partition}",
            "object": "model",
            "created": 0,
            "owned_by": "RAGondin",
        }
        for partition in partitions
    ]

    return JSONResponse(content={"object": "list", "data": models})


def __get_partition_name(model_name, app_state):
    if not model_name.startswith("ragondin-"):
        raise HTTPException(status_code=404, detail="Model not found")

    partition = model_name.split("-")[1]

    if partition != "all" and not app_state.vectordb.partition_exists(partition):
        raise HTTPException(status_code=404, detail="Model not found")

    return partition


def __create_md_links(sources, static_base_url):
    sources = list(map(lambda x: source2url(x, static_base_url), sources))
    markdowns_links = []

    for doc in sources:
        encoded_url = quote(doc["url"], safe=":/")
        parsed_url = urlparse(doc["url"])
        doc_name = parsed_url.path.split("/")[-1]

        if "pdf" in doc_name.lower():
            encoded_url = f"{encoded_url}#page={doc['page']}"
        s = f"* {doc['doc_id']} : [{doc_name}]({encoded_url})"

        markdowns_links.append(s)

    src_md = "\n## Sources: \n" + "\n".join(markdowns_links) if markdowns_links else ""

    return markdowns_links, src_md


@router.post(
    "/chat/completions", summary="OpenAI compatible chat completion endpoint using RAG"
)
async def openai_chat_completion(
    request: OpenAIChatCompletionRequest,
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

    # Load model name and partition
    model_name = request.model
    partition = __get_partition_name(model_name, app_state)

    # Run RAG pipeline
    answer_stream, context, sources = await app_state.ragpipe.run(
        partition=[partition], question=new_user_input, chat_history=msgs
    )

    # Handle the sources
    markdowns_links, src_md = __create_md_links(sources, static_base_url)

    # Create response-id
    response_id = f"chatcmpl-{str(uuid.uuid4())}"
    created_time = int(time.time())

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
            ## Send Sources
            chunk_src = OpenAICompletionChunk(
                id=response_id,
                created=created_time,
                model=model_name,
                choices=[
                    OpenAICompletionChunkChoice(
                        index=0,
                        delta={"content": src_md},
                        finish_reason=None,
                    )
                ],
            )

            chunk = OpenAICompletionChunk(
                id=response_id,
                created=created_time,
                model=model_name,
                choices=[
                    OpenAICompletionChunkChoice(index=0, delta={}, finish_reason="stop")
                ],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
            # yield f"data: {chunk_src.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
            headers={"X-Metadata-Sources": json.dumps(markdowns_links)},
        )
    else:
        # Non streaming response
        full_response = ""
        async for token in answer_stream:
            full_response += token.content

        completion = OpenAIChatCompletion(
            id=response_id,
            created=created_time,
            model=model_name,
            choices=[
                OpenAIChatCompletionChoice(
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


@router.post("/completions", summary="OpenAI compatible completion endpoint using RAG")
async def openai_completion(
    request: OpenAICompletionRequest,
    static_base_url: str = Depends(static_base_url_dependency),
    app_state=Depends(get_app_state),
):
    print("strat openai_completion(...)")
    # Load model name and partition
    model_name = request.model
    partition = __get_partition_name(model_name, app_state)

    # Run RAG pipeline
    output, context, sources = await app_state.ragpipe.completion(
        partition=[partition],
        query=request.prompt,
        chat_history=None,
        llm_config=None,
    )

    # Create response-id
    response_id = f"chatcmpl-{str(uuid.uuid4())}"
    created_time = int(time.time())

    if request.stream:
        raise
    else:
        # Non streaming response
        output = await output
        metadata = output.response_metadata
        usage = metadata["token_usage"]

        logprobs = {
            "token_logprobs": [ r["logprob"] for r in metadata["logprobs"]["content"] ],
            "top_logprobs": [
                {
                    v["token"]: v["logprob"] for v in entry["top_logprobs"]
                }
                for entry in metadata["logprobs"]["content"]
            ]
        }

        completion = OpenAICompletion(
            id=response_id,
            created=created_time,
            model=model_name,
            choices=[
                OpenAICompletionChoice(
                    index=0,
                    text=output.content,
                    logprobs=logprobs,
                    finish_reason=metadata["finish_reason"],
                )
            ],
            usage=OpenAIUsage(
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                total_tokens=usage["total_tokens"],
            ),
        )
        print("after completion object creation")

        return completion
