from typing import List, Optional

from components import Indexer
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from utils.dependencies import get_indexer

# Create an APIRouter instance
router = APIRouter()


@router.get("/", response_model=None)
async def search_multiple_partitions(
    request: Request,
    namespace: Optional[List[str]] = Query(
        ["all"], description="List of namespaces to search"
    ),
    text: str = Query(..., description="Text to search semantically"),
    top_k: int = Query(5, description="Number of top results to return"),
    indexer: Indexer = Depends(get_indexer),
):
    try:
        # Perform the search using the Indexer
        results = await indexer.asearch(query=text, top_k=top_k, partition=namespace)

        # Construct HATEOAS response
        documents = [
            {
                "link": str(
                    request.url_for("get_extract", extract_id=doc.metadata["_id"])
                )
            }
            for doc in results
        ]

        # Return results
        return JSONResponse(content=f"Documents: {documents}", status_code=200)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Handle errors
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/namespace/{namespace}", response_model=None)
async def search_one_partition(
    request: Request,
    namespace: str,
    text: str = Query(..., description="Text to search semantically"),
    top_k: int = Query(5, description="Number of top results to return"),
    indexer: Indexer = Depends(get_indexer),
):
    try:
        # Perform the search using the Indexer
        results = await indexer.asearch(query=text, top_k=top_k, partition=namespace)
        # Transforming the results (assuming they are LangChain documents)
        # Construct HATEOAS response
        documents = [
            {
                "link": str(
                    request.url_for("get_extract", extract_id=doc.metadata["_id"])
                )
            }
            for doc in results
        ]

        # Return results
        return JSONResponse(content=f"Documents: {documents}", status_code=200)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Handle errors
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/namespace/{namespace}/file/{file_id}", response_model=None)
async def search_file(
    request: Request,
    namespace: str,
    file_id: str,
    query: str = Query(..., description="Text to search semantically"),
    top_k: int = Query(5, description="Number of top results to return"),
    indexer: Indexer = Depends(get_indexer),
):
    try:
        # Perform the search using the Indexer
        results = await indexer.asearch(
            query=query, top_k=top_k, partition=namespace, filter={"file_id": file_id}
        )

        # Construct HATEOAS response
        documents = [
            {
                "link": str(
                    request.url_for("get_extract", extract_id=doc.metadata["_id"])
                )
            }
            for doc in results
        ]

        # Return results
        return JSONResponse(content=f"Documents: {documents}", status_code=200)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Handle errors
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{item_id}", response_model=None)
async def get_extract(item_id: str, indexer: Indexer = Depends(get_indexer)):
    try:
        doc = indexer.vectordb.get_chunk_by_id(item_id)
        return JSONResponse(
            content={"page_content": doc.page_content, "metadata": doc.metadata},
            status_code=200,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
