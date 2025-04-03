from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from utils.dependencies import Indexer, get_indexer, vectordb

# Create an APIRouter instance
router = APIRouter()


@router.get("/{partition}")
async def list_files(
    request: Request,
    partition: str,
    indexer: Indexer = Depends(get_indexer),
):
    # Check if partition exists
    if not vectordb.partition_exists(partition):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Partition '{partition}' not found."
        )

    try:
        results = vectordb.list_files(partition=partition)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    documents = [
        {
            "link": str(request.url_for("get_file", partition=partition, file_id=file))
        }
        for file in results
    ]

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"documents": documents}
    )


@router.get("/{partition}/file/{file_id}")
async def get_file(
    request: Request,
    partition: str,
    file_id: str,
    indexer: Indexer = Depends(get_indexer),
):
    # Check if file exists
    if not vectordb.file_exists(file_id, partition):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{file_id}' not found in partition '{partition}'."
        )

    try:
        results = await indexer.get_file_chunks.remote(
            partition=partition, file_id=file_id, include_id=True
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

    metadata = (
        {k: v for k, v in results[0].metadata.items() if k != "_id"}
        if results else {}
    )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "metadata": metadata,
            "documents": documents
        }
    )