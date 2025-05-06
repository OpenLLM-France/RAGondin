import json
from fastapi import APIRouter, Body, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from openai import AsyncOpenAI
from models.openai import (
    OpenAIChatCompletionRequest,
    OpenAICompletionRequest,
)
from urllib.parse import urlparse, quote
from loguru import logger
from config.config import load_config
from fastapi import status


config = load_config()
# Créer un router pour les endpoints OpenAI
router = APIRouter()


# Fonctions utilitaires pour le router
def get_app_state(request: Request):
    return request.app.state.app_state


async def check_llm_model_availability(request: Request):
    models = {"VLM": config.llm, "LLM": config.vlm}
    for model_type, param in models.items():
        try:
            client = AsyncOpenAI(api_key=param["api_key"], base_url=param["base_url"])
            openai_models = await client.models.list()
            available_models = {m.id for m in openai_models.data}
            if param["model"] not in available_models:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Only these models ({available_models}) are available for your `{model_type}`. Please check your configuration file.",
                )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Error while checking the `{model_type}` endpoint: {str(e)}",
            )


@router.get("/models", summary="OpenAI-compatible model listing endpoint")
async def list_models(
    app_state=Depends(get_app_state), _: None = Depends(check_llm_model_availability)
):
    # Get available partitions from your backend
    partitions = app_state.vectordb.list_partitions()

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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Model not found"
        )

    partition = model_name.split("ragondin-")[1]
    if partition != "all" and not app_state.vectordb.partition_exists(partition):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Model not found"
        )

    return partition


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
    _: None = Depends(check_llm_model_availability),
):
    # TODO: Better handle error
    try:
        # Get the last user message
        if not request.messages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one user message is required",
            )
        if not request.messages[-1].role == "user":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The last message must be from the user",
            )
        if not request.messages[-1].content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The last message must have content",
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
                return JSONResponse(
                    content=chunk, headers={"X-Metadata-Sources": json.dumps(metadata)}
                )
            except StopAsyncIteration:
                err_str = "No response from LLM"
                logger.debug(err_str)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=err_str,
                )
    except Exception as e:
        err_str = f"Error: {str(e)}"
        logger.debug(err_str)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=err_str,
        )


@router.post("/completions", summary="OpenAI compatible completion endpoint using RAG")
async def openai_completion(
    request2: Request,
    request: OpenAICompletionRequest,
    app_state=Depends(get_app_state),
    _: None = Depends(check_llm_model_availability),
):
    # Get the last user message
    if not request.prompt:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="tThe prompt is required"
        )

    if request.stream:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Streaming is not supported for this endpoint",
        )

    # Load model name and partition
    model_name = request.model
    if not model_name.startswith("ragondin-"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Model not found"
        )
    partition = model_name.split("ragondin-")[1]
    if partition != "all" and not app_state.vectordb.partition_exists(partition):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Model not found"
        )
    # Run RAG pipeline
    llm_output, _, sources = await app_state.ragpipe.completions(
        partition=[partition], payload=request.model_dump()
    )

    # Handle the sources
    metadata = __prepare_sources(request2, sources)

    try:
        complete_response = await llm_output.__anext__()
        return JSONResponse(
            content=complete_response,
            headers={"X-Metadata-Sources": json.dumps(metadata)},
        )
    except StopAsyncIteration:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No response from LLM",
        )
