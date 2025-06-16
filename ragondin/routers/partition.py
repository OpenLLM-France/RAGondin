from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from utils.dependencies import Indexer, get_indexer, vectordb
import ray
from loguru import logger


# Create an APIRouter instance
router = APIRouter()


@router.get("/")
async def list_existant_partitions(request: Request):
    try:
        partitions = []
        for p in vectordb.list_partitions():
            partitions.append(
                {"partition": p.partition, "created_at": int(p.created_at.timestamp())}
            )

        return JSONResponse(
            status_code=status.HTTP_200_OK, content={"partitions": partitions}
        )
    except Exception as e:
        logger.debug(str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.delete("/{partition}/")
async def delete_partition(partition: str, indexer: Indexer = Depends(get_indexer)):
    """
    Delete a partition.
    """
    try:
        deleted = ray.get(indexer.delete_partition.remote(partition))

    except Exception as e:
        err_str = f"Error while deleting partition '{partition}': {str(e)}"
        logger.debug(err_str)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=err_str,
        )

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Partition '{partition}' not found.",
        )

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{partition}/")
async def list_files(
    request: Request,
    partition: str,
    indexer: Indexer = Depends(get_indexer),
):
    # Check if partition exists
    if not vectordb.partition_exists(partition):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Partition '{partition}' not found.",
        )

    try:
        results = vectordb.list_files(partition=partition)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.debug(str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )

    files = [
        {"link": str(request.url_for("get_file", partition=partition, file_id=file))}
        for file in results
    ]

    return JSONResponse(status_code=status.HTTP_200_OK, content={"files": files})


@router.get("/check-file/{partition}/file/{file_id}")
async def check_file_exists_in_partition(
    request: Request,
    partition: str,
    file_id: str,
):
    # Check if file exists
    exists = vectordb.file_exists(file_id, partition)
    if not exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{file_id}' not found in partition '{partition}'.",
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=f"File '{file_id}' exists in partition '{partition}'.",
        )


@router.get("/{partition}/file/{file_id}")
async def get_file(
    request: Request,
    partition: str,
    file_id: str,
):
    # Check if file exists
    if not vectordb.file_exists(file_id, partition):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{file_id}' not found in partition '{partition}'.",
        )

    try:
        results = vectordb.get_file_chunks(
            partition=partition, file_id=file_id, include_id=True
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.debug(str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )

    documents = [
        {"link": str(request.url_for("get_extract", extract_id=doc.metadata["_id"]))}
        for doc in results
    ]

    metadata = (
        {k: v for k, v in results[0].metadata.items() if k != "_id"} if results else {}
    )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"metadata": metadata, "documents": documents},
    )

@router.get("/{partition}/sample")
async def sample_chunks(
    request: Request, partition: str, n_ids: int = 200, seed: int | None = None
):
    # Check if partition exists
    if not vectordb.partition_exists(partition):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Partition '{partition}' not found.",
        )

    try:
        list_ids = vectordb.sample_chunk_ids(
            partition=partition, n_ids=n_ids, seed=seed
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )

    chunks = [
        {"link": str(request.url_for("get_extract", extract_id=id))} for id in list_ids
    ]

    return JSONResponse(status_code=status.HTTP_200_OK, content={"chunk_urls": chunks})

@router.get("/{partition}/chunks")
async def list_all_chunks(
    request: Request,
    partition: str,
):
    # Check if partition exists
    if not vectordb.partition_exists(partition):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Partition '{partition}' not found.",
        )

    try:
        chunks = vectordb.list_chunk_ids(partition=partition)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )

    chunks = [
        {
            "link": str(request.url_for("get_extract", extract_id=chunk["Chunk ID"])),
            "Chunk's content": chunk["Chunk's content"],
            "Embedding vector": chunk["Embedding vector"],
            "Original file's ID": chunk["Original file's ID"]
        }
        for chunk in chunks
    ]

    return JSONResponse(status_code=status.HTTP_200_OK, content={"All chunks' details": chunks})

@router.get("/{partition}/clusters")
async def list_clusters(
    request: Request,
    partition: str,
):
    result = vectordb.clusterizer(partition)

    return JSONResponse(
        status_code=status.HTTP_200_OK, content={"clusters": result}
    )