import json
from fastapi import APIRouter, Depends, HTTPException, Request
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
            "created": 0,  # TODO: Add created time
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
        llm_output, context, sources = await app_state.ragpipe.chat_completion(
            partition=[partition], payload=request.model_dump()
        )

        # Handle the sources
        markdowns_links, src_md = __create_md_links(sources, static_base_url)

        if request.stream:
            # Openai compatible streaming response
            async def stream_response():
                async for chunk in llm_output:
                    yield chunk

            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream",
                headers={"X-Metadata-Sources": json.dumps(markdowns_links)},
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
    partition = __get_partition_name(model_name, app_state)
    # Run RAG pipeline
    llm_output, context, sources = await app_state.ragpipe.completions(
        partition=[partition], payload=request.model_dump()
    )

    # Handle the sources
    # markdowns_links, src_md = __create_md_links(sources, static_base_url)

    try:
        complete_response = await llm_output.__anext__()
        return complete_response
    except StopAsyncIteration:
        raise HTTPException(status_code=500, detail="No response from LLM")
