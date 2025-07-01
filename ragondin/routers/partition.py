import ray
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from utils.dependencies import Indexer, get_indexer, vectordb
from utils.logger import get_logger

logger = get_logger()

router = APIRouter()


@router.get("/")
async def list_existant_partitions(request: Request):
    try:
        partitions = [
            {"partition": p.partition, "created_at": int(p.created_at.timestamp())}
            for p in vectordb.list_partitions()
        ]
        logger.debug(
            "Returned list of existing partitions.", partition_count=len(partitions)
        )
        return JSONResponse(
            status_code=status.HTTP_200_OK, content={"partitions": partitions}
        )
    except Exception:
        logger.exception("Failed to list partitions.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list partitions.",
        )


@router.delete("/{partition}")
async def delete_partition(partition: str, indexer: Indexer = Depends(get_indexer)):
    try:
        deleted = ray.get(indexer.delete_partition.remote(partition))
    except Exception:
        logger.exception("Failed to delete partition", partition=partition)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete partition",
        )

    if not deleted:
        logger.warning("Partition not found for deletion", partition=partition)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Partition not found",
        )

    logger.debug("Partition successfully deleted.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{partition}")
async def list_files(
    request: Request,
    partition: str,
    indexer: Indexer = Depends(get_indexer),
):
    log = logger.bind(partition=partition)

    if not vectordb.partition_exists(partition):
        log.warning("Partition not found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Partition '{partition}' not found",
        )

    try:
        results = vectordb.list_files(partition=partition)
        log.debug("Listed files in partition", file_count=len(results))
    except ValueError as e:
        log.warning(f"Invalid partition value: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception:
        log.exception("Failed to list files in partition")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list files",
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
    log = logger.bind(partition=partition, file_id=file_id)
    exists = vectordb.file_exists(file_id, partition)
    if not exists:
        log.warning("File not found in partition.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{file_id}' not found in partition '{partition}'.",
        )

    log.debug("File exists in partition.")
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
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch file chunks.",
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
    request: Request, partition: str, include_embedding: bool = True
):
    # Check if partition exists
    if not vectordb.partition_exists(partition):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Partition '{partition}' not found.",
        )

    try:
        chunks = vectordb.list_all_chunk(
            partition=partition, include_embedding=include_embedding
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )

    chunks = [
        {
            "link": str(
                request.url_for("get_extract", extract_id=chunk.metadata["_id"])
            ),
            "content": chunk.page_content,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]
    return JSONResponse(status_code=status.HTTP_200_OK, content={"chunks": chunks})
