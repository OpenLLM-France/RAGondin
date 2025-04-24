import json
from fastapi import APIRouter, Body, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from models.openai import (
    OpenAIChatCompletionRequest,
    OpenAICompletionRequest,
)
from urllib.parse import urlparse, quote
from loguru import logger

# Créer un router pour les endpoints OpenAI
router = APIRouter()


# Fonctions utilitaires pour le router
def get_app_state(request: Request):
    return request.app.state.app_state


@router.get("/models", summary="OpenAI-compatible model listing endpoint")
async def list_models(app_state=Depends(get_app_state)):
    # Get available partitions from your backend
    partitions = app_state.ragpipe.vectordb.list_partitions()

    # Format them as OpenAI models
    models = [
        {
            "id": f"ragondin-{partition.partition}",
            "object": "model",
            "created": int(partition.created_at.timestamp()),
            "owned_by": "RAGondin",
        }
        for partition in partitions
    ]

    models.append(
        {
            "id": "ragondin-all",
            "object": "model",
            "created": 0,
            "owned_by": "RAGondin",
        }
    )

    return JSONResponse(content={"object": "list", "data": models})


def __get_partition_name(model_name, app_state):
    if not model_name.startswith("ragondin-"):
        raise HTTPException(status_code=404, detail="Model not found")

    partition = model_name.split("ragondin-")[1]
    if partition != "all" and not app_state.vectordb.partition_exists(partition):
        raise HTTPException(status_code=404, detail="Model not found")

    return partition


def static_base_url_dependency(request: Request) -> str:
    return f"{request.url.scheme}://{request.url.hostname}:{request.url.port}/static"


def __prepare_sources(request: Request, sources: list):
    links = []
    for source in sources:
        filename = source["filename"]
        file_url = str(request.url_for("static", path=filename))
        encoded_url = quote(file_url, safe=":/")
        links.append(
            {
                "doc_id": source["doc_id"],
                "file_url": encoded_url,
                "filename": source["filename"],
                "_id": source["_id"],
                "chunk_url": str(
                    request.url_for("get_extract", extract_id=source["_id"])
                ),
                "page": source["page"],
            }
        )

    return links


@router.post(
    "/chat/completions", summary="OpenAI compatible chat completion endpoint using RAG"
)
async def openai_chat_completion(
    request2: Request,
    request: OpenAIChatCompletionRequest = Body(...),
    app_state=Depends(get_app_state),
    # static_base_url=Depends(static_base_url_dependency),
):
    try:
        # Get the last user message
        if not request.messages:
            raise HTTPException(
                status_code=400, detail="At least one user message is required"
            )

        # Load model name and partition
        model_name = request.model
        partition = __get_partition_name(model_name, app_state)

        # Run RAG pipeline
        llm_output, _, sources = await app_state.ragpipe.chat_completion(
            partition=[partition], payload=request.model_dump()
        )

        # Handle the sources
        metadata = __prepare_sources(request2, sources)

        if request.stream:
            # Openai compatible streaming response
            async def stream_response():
                async for chunk in llm_output:
                    yield chunk

            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream",
                headers={"X-Metadata-Sources": json.dumps(metadata)},
            )
        else:
            # get the next chunk item of an async generator async
            try:
                chunk = await llm_output.__anext__()
                return chunk
            except StopAsyncIteration:
                raise HTTPException(status_code=500, detail="No response from LLM")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/completions", summary="OpenAI compatible completion endpoint using RAG")
async def openai_completion(
    request2: Request,
    request: OpenAICompletionRequest,
    static_base_url: str = Depends(static_base_url_dependency),
    app_state=Depends(get_app_state),
):
    # Get the last user message
    if not request.prompt:
        raise HTTPException(status_code=400, detail="tThe prompt is required")

    if request.stream:
        raise HTTPException(
            status_code=400, detail="Streaming is not supported for this endpoint"
        )

    # Load model name and partition
    model_name = request.model
    if not model_name.startswith("ragondin-"):
        raise HTTPException(status_code=404, detail="Model not found")
    partition = model_name.split("ragondin-")[1]
    if partition != "all" and not app_state.ragpipe.vectordb.partition_exists(
        partition
    ):
        raise HTTPException(status_code=404, detail="Model not found")
    # Run RAG pipeline
    llm_output, _, sources = await app_state.ragpipe.completions(
        partition=[partition], payload=request.model_dump()
    )

    # Handle the sources
    try:
        complete_response = await llm_output.__anext__()
        return complete_response
    except StopAsyncIteration:
        raise HTTPException(status_code=500, detail="No response from LLM")
