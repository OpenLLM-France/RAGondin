from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from utils.dependencies import Indexer, get_indexer, vectordb
import ray
from utils.logger import get_logger

logger = get_logger()

router = APIRouter()


@router.get("/")
async def list_existant_partitions(request: Request):
    log = logger.bind(endpoint="/partitions")
    try:
        partitions = [
            {"partition": p.partition, "created_at": int(p.created_at.timestamp())}
            for p in vectordb.list_partitions()
        ]
        log.info("Returned list of existing partitions.", count=len(partitions))
        return JSONResponse(
            status_code=status.HTTP_200_OK, content={"partitions": partitions}
        )
    except Exception:
        log.exception("Failed to list partitions.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list partitions.",
        )


@router.delete("/{partition}/")
async def delete_partition(partition: str, indexer: Indexer = Depends(get_indexer)):
    log = logger.bind(partition=partition)
    try:
        deleted = ray.get(indexer.delete_partition.remote(partition))
    except Exception:
        log.exception("Failed to delete partition.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete partition.",
        )

    if not deleted:
        log.warning("Partition not found for deletion.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Partition '{partition}' not found.",
        )

    log.info("Partition successfully deleted.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{partition}/")
async def list_files(
    request: Request,
    partition: str,
    indexer: Indexer = Depends(get_indexer),
):
    log = logger.bind(partition=partition)

    if not vectordb.partition_exists(partition):
        log.warning("Partition not found.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Partition '{partition}' not found.",
        )

    try:
        results = vectordb.list_files(partition=partition)
        log.info("Listed files in partition.", file_count=len(results))
    except ValueError as e:
        log.warning(f"Invalid partition value: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception:
        log.exception("Failed to list files in partition.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list files."
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

    log.info("File exists in partition.")
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
            detail="Failed to fetch file chunks."
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