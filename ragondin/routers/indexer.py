import json
from pathlib import Path
from typing import Any, Optional

import ray
from config.config import load_config
from fastapi import (
    APIRouter,
    Depends,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse
from loguru import logger
from utils.dependencies import Indexer, get_indexer, vectordb

# load config
config = load_config()
DATA_DIR = config.paths.data_dir
ACCEPTED_FILE_FORMATS = dict(config.loader["file_loaders"]).keys()
FORBIDDEN_CHARS_IN_FILE_ID = set("/")  # set('"<>#%{}|\\^`[]')


# Create an APIRouter instance
router = APIRouter()


def is_file_id_valid(file_id: str) -> bool:
    return not any(c in file_id for c in FORBIDDEN_CHARS_IN_FILE_ID)


async def validate_file_id(file_id: str):
    if not is_file_id_valid(file_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File ID contains forbidden characters: {', '.join(FORBIDDEN_CHARS_IN_FILE_ID)}",
        )

    if not file_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="File ID cannot be empty."
        )
    return file_id


async def validate_file_format(file: UploadFile):
    file_extension = (
        file.filename.split(".")[-1].lower() if "." in file.filename else ""
    )
    if file_extension not in ACCEPTED_FILE_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file format: {file_extension}. Supported formats are: {', '.join(ACCEPTED_FILE_FORMATS)}",
        )

    return file


async def validate_metadata(metadata: Optional[Any] = Form(None)):
    try:
        processed_metadata = metadata or "{}"
        processed_metadata = json.loads(processed_metadata)
        return processed_metadata
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON in metadata"
        )


@router.post("/partition/{partition}/file/{file_id}")
async def add_file(
    request: Request,
    partition: str,
    file_id: str = Depends(validate_file_id),
    file: UploadFile = Depends(validate_file_format),
    metadata: dict = Depends(validate_metadata),
    indexer: Indexer = Depends(get_indexer),
):
    # Check if file exists
    if vectordb.file_exists(file_id, partition):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"File '{file_id}' already exists in partition {partition}",
        )

    # Add file_id to metadata
    metadata["file_id"] = file_id

    # Create a temporary directory to store files
    save_dir = Path(DATA_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save the uploaded file
    file_path = save_dir / Path(file.filename).name

    metadata.update({"source": str(file_path), "filename": file.filename})
    logger.info(f"Processing file: {file.filename} and saving to {file_path}")
    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
    except Exception as e:
        err_str = f"Failed to save file: {str(e)}"
        logger.debug(err_str)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=err_str,
        )

    # Queue the file for indexing
    try:
        task = indexer.add_file.remote(
            path=file_path, metadata=metadata, partition=partition
        )
        # TODO: More specific errors with details and appropriate error codes
    except Exception as e:
        err_str = f"Indexing error: {str(e)}"
        logger.debug(err_str)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=err_str,
        )

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            "task_status_url": str(
                request.url_for("get_task_status", task_id=task.task_id().hex())
            )
        },
    )


@router.delete("/partition/{partition}/file/{file_id}")
async def delete_file(
    partition: str, file_id: str, indexer: Indexer = Depends(get_indexer)
):
    """
    Delete a file in a specific partition.
    """
    try:
        deleted = ray.get(indexer.delete_file.remote(file_id, partition))
    except Exception as e:
        err_str = f"Error while deleting file '{file_id}': {str(e)}"
        logger.debug(err_str)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=err_str,
        )

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{file_id}' not found in partition '{partition}'.",
        )

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.put("/partition/{partition}/file/{file_id}")
async def put_file(
    request: Request,
    partition: str,
    file_id: str = Depends(validate_file_id),
    file: UploadFile = Depends(validate_file_format),
    metadata: dict = Depends(validate_metadata),
    indexer: Indexer = Depends(get_indexer),
):
    # Validate file existence
    if not vectordb.file_exists(file_id, partition):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{file_id}' not found in partition '{partition}'.",
        )

    # Delete old file
    try:
        ray.get(indexer.delete_file.remote(file_id, partition))
        logger.info(f"File {file_id} deleted.")
    except Exception as e:
        err_str = f"Failed to delete existing file: {str(e)}"
        logger.debug(err_str)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=err_str,
        )

    metadata["file_id"] = file_id

    # Save uploaded file
    save_dir = Path(DATA_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save the uploaded file
    file_path = save_dir / Path(file.filename).name
    metadata.update(
        {
            "source": str(file_path),
            "filename": file.filename,
        }
    )

    logger.info(f"Processing file: {file.filename} and saving to {file_path}")

    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
    except Exception as e:
        err_str = f"Failed to save file: {str(e)}"
        logger.debug(err_str)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=err_str,
        )

    # Queue indexing task
    try:
        task = indexer.add_file.remote(
            path=file_path, metadata=metadata, partition=partition
        )
    except Exception as e:
        err_str = f"Indexing error: {str(e)}"
        logger.debug(err_str)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=err_str,
        )

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "task_status_url": str(
                request.url_for("get_task_status", task_id=task.task_id().hex())
            )
        },
    )


@router.patch("/partition/{partition}/file/{file_id}")
async def patch_file(
    partition: str,
    file_id: str = Depends(validate_file_id),
    metadata: Optional[Any] = Depends(validate_metadata),
    indexer: Indexer = Depends(get_indexer),
):
    # Check if file exists
    if not vectordb.file_exists(file_id, partition):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{file_id}' not found in partition '{partition}'.",
        )

    metadata["file_id"] = file_id

    # Update metadata in indexer
    try:
        ray.get(indexer.update_file_metadata.remote(file_id, metadata, partition))
    except Exception as e:
        err_str = f"Failed to update metadata: {str(e)}"
        logger.debug(err_str)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=err_str,
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": f"Metadata for file '{file_id}' successfully updated."},
    )


@router.get("/task/{task_id}")
async def get_task_status(task_id: str, indexer: Indexer = Depends(get_indexer)):
    try:
        state = await indexer.get_task_status.remote(task_id)
    except Exception:
        logger.warning(f"Task {task_id} not found.")
        state = None

    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Task '{task_id}' not found."
        )
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"task_id": task_id, "task_state": state},
    )
