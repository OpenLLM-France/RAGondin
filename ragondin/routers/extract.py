from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from utils.dependencies import Indexer, get_indexer, vectordb
from loguru import logger

# Create an APIRouter instance
router = APIRouter()


@router.get("/{extract_id}")
async def get_extract(extract_id: str, indexer: Indexer = Depends(get_indexer)):
    try:
        doc = vectordb.get_chunk_by_id(extract_id)
        if doc is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Extract '{extract_id}' not found.",
            )
    except Exception as e:
        err_str = f"Failed to retrieve extract: {str(e)}"
        logger.debug(err_str)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=err_str
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"page_content": doc.page_content, "metadata": doc.metadata},
    )
