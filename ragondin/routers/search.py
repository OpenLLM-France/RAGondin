from loguru import logger
from typing import Optional, Any, List
from fastapi import APIRouter, HTTPException, Depends, Query, Request
from fastapi.responses import JSONResponse
from pathlib import Path
from components import Indexer
from utils.dependencies import get_indexer


# Create an APIRouter instance
router = APIRouter()

@router.get("/", response_model=None)
async def search_multiple_partitions(
    request: Request,
    partitions: Optional[List[str]] = Query(["all"], description="List of partitions to search"),
    text: str = Query(..., description="Text to search semantically"),
    top_k: int = Query(5, description="Number of top results to return"),
    indexer: Indexer = Depends(get_indexer),
):
    try:
        logger.info(f"Searching for '{text}' in partitions: {partitions}")
        # Perform the search using the Indexer
        results = await indexer.asearch(query=text, top_k=top_k, partition=partitions)
        
        # Construct HATEOAS response
        documents = [
            {"link": str(request.url_for("get_extract", extract_id=doc.metadata["_id"]))}
            for doc in results
        ]
        
        # Return results
        return JSONResponse(content={"Documents": documents}, status_code=200)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Handle errors
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/partition/{partition}", response_model=None)
async def search_one_partition(
    request : Request,
    partition : str,
    text: str = Query(..., description="Text to search semantically"),
    top_k: int = Query(5, description="Number of top results to return"),
    indexer: Indexer = Depends(get_indexer)
    ):
    try:
        # Perform the search using the Indexer
        results = await indexer.asearch(query=text, top_k=top_k, partition=partition)
        # Transforming the results (assuming they are LangChain documents)
        # Construct HATEOAS response
        documents = [
            {"link": str(request.url_for("get_extract", extract_id=doc.metadata["_id"]))}
            for doc in results
        ]
        
        # Return results
        return JSONResponse(content={"Documents": documents}, status_code=200)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Handle errors
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/partition/{partition}/file/{file_id}", response_model=None)
async def search_file(
    request : Request,
    partition : str,
    file_id: str,
    query: str = Query(..., description="Text to search semantically"),
    top_k: int = Query(5, description="Number of top results to return"),
    indexer: Indexer = Depends(get_indexer),
    
    ):
    try:
        # Perform the search using the Indexer
        results = await indexer.asearch(query=query, top_k=top_k, partition=partition, filter= {"file_id": file_id})
        
        # Construct HATEOAS response
        documents = [
            {"link": str(request.url_for("get_extract", extract_id=doc.metadata["_id"]))}
            for doc in results
        ]
        
        # Return results
        return JSONResponse(content={"Documents": documents}, status_code=200)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Handle errors
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{extract_id}", response_model=None)
async def get_extract(
    extract_id : str,
    indexer: Indexer = Depends(get_indexer)
    ):
    try:
        doc = indexer.vectordb.get_chunk_by_id(extract_id)
        return JSONResponse(content={"page_content": doc.page_content, "metadata": doc.metadata}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))