import json
from pathlib import Path
from typing import Any, Optional

import ray
from config.config import load_config
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse
from loguru import logger
from ray.util.state import get_task
from utils.dependencies import Indexer, get_indexer, vectordb

# load config
config = load_config()
DATA_DIR = config.paths.data_dir
# Create an APIRouter instance
router = APIRouter()


@router.post("/partition/{partition}/file/{file_id}")
async def add_file(
    request: Request,
    partition: str,
    file_id: str,
    file: UploadFile = File(...),
    metadata: Optional[Any] = Form(None),
    indexer: Indexer = Depends(get_indexer),
):
    # Check if file exists
    if vectordb.file_exists(file_id, partition):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"File '{file_id}' already exists in partition {partition}",
        )
    # Load metadata
    try:
        metadata = metadata or "{}"
        metadata = json.loads(metadata)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON in metadata"
        )
    if not isinstance(metadata, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Metadata must be a dictionary",
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}",
        )

    # Queue the file for indexing
    try:
        task = indexer.add_file.remote(
            path=file_path, metadata=metadata, partition=partition
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing error: {str(e)}",
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error while deleting file '{file_id}': {str(e)}",
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
    file_id: str,
    file: UploadFile = File(...),
    metadata: Optional[Any] = Form(None),
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete existing file: {str(e)}",
        )

    # Parse metadata
    try:
        metadata = metadata or "{}"
        metadata = json.loads(metadata)
        if not isinstance(metadata, dict):
            raise ValueError("Metadata is not a dictionary.")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid metadata: {str(e)}",
        )

    metadata["file_id"] = file_id

    # Save uploaded file
    save_dir = Path(DATA_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / Path(file.filename).name
    logger.info(f"Processing file: {file.filename} and saving to {file_path}")

    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}",
        )

    # Queue indexing task
    try:
        task = indexer.add_file.remote(
            path=file_path, metadata=metadata, partition=partition
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing error: {str(e)}",
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
    file_id: str,
    metadata: Optional[Any] = Form(None),
    indexer: Indexer = Depends(get_indexer),
):
    # Check if file exists
    if not vectordb.file_exists(file_id, partition):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{file_id}' not found in partition '{partition}'.",
        )

    # Parse metadata
    try:
        metadata = metadata or "{}"
        metadata = json.loads(metadata)
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a JSON object.")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid metadata: {str(e)}",
        )

    metadata["file_id"] = file_id

    # Update metadata in indexer
    try:
        ray.get(indexer.update_file_metadata.remote(file_id, metadata, partition))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update metadata: {str(e)}",
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": f"Metadata for file '{file_id}' successfully updated."},
    )


@router.get("/task/{task_id}")
async def get_task_status(task_id: str, indexer: Indexer = Depends(get_indexer)):
    task = get_task(task_id)

    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Task '{task_id}' not found."
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"task_id": task_id, "task_state": task.state},
    )
