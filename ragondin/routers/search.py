from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from utils.dependencies import Indexer, get_indexer, vectordb

# Create an APIRouter instance
router = APIRouter()


@router.get("")
async def search_multiple_partitions(
    request: Request,
    partitions: Optional[List[str]] = Query(
        default=["all"], description="List of partitions to search"
    ),
    text: str = Query(..., description="Text to search semantically"),
    top_k: int = Query(5, description="Number of top results to return"),
    indexer: Indexer = Depends(get_indexer),
):
    try:
        results = await indexer.asearch.remote(
            query=text, top_k=top_k, partition=partitions
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    documents = [
        {
            "link": str(
                request.url_for("get_extract", extract_id=doc.metadata["_id"])
            )
        }
        for doc in results
    ]

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"documents": documents}
    )


@router.get("/partition/{partition}")
async def search_one_partition(
    request: Request,
    partition: str,
    text: str = Query(..., description="Text to search semantically"),
    top_k: int = Query(5, description="Number of top results to return"),
    indexer: Indexer = Depends(get_indexer),
):
    try:
        results = await indexer.asearch.remote(
            query=text,
            top_k=top_k,
            partition=partition
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    documents = [
        {
            "link": str(
                request.url_for("get_extract", extract_id=doc.metadata["_id"])
            )
        }
        for doc in results
    ]

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"documents": documents}
    )


@router.get("/partition/{partition}/file/{file_id}")
async def search_file(
    request: Request,
    partition: str,
    file_id: str,
    query: str = Query(..., description="Text to search semantically"),
    top_k: int = Query(5, description="Number of top results to return"),
    indexer: Indexer = Depends(get_indexer),
):
    try:
        results = await indexer.asearch.remote(
            query=query,
            top_k=top_k,
            partition=partition,
            filter={"file_id": file_id}
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    documents = [
        {
            "link": str(
                request.url_for("get_extract", extract_id=doc.metadata["_id"])
            )
        }
        for doc in results
    ]

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"documents": documents}
    )

