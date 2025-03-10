from loguru import logger
from typing import Optional, Any, List
from fastapi import APIRouter, HTTPException, File, UploadFile, Depends, Form
from fastapi.responses import JSONResponse
from pathlib import Path
from components import Indexer
from models.indexer import SearchRequest, DeleteFilesRequest
from utils.dependencies import get_indexer
from config.config import load_config
import json

#load config
config = load_config()
DATA_DIR = config.paths.data_dir
# Create an APIRouter instance
router = APIRouter()


@router.post("/add-files/", response_model=None)
async def add_files(
    files: List[UploadFile] = File(...),
    metadata: Optional[Any] = Form(...),
    collection_name: Optional[str] = Form(None),
    indexer: Indexer = Depends(get_indexer)):
    try:
        # Load metadata
        metadata = json.loads(metadata)
        if metadata is None:
            metadata = [{}] * len(files)
        elif isinstance(metadata, list):
            if len(metadata) != len(files):
                raise HTTPException(status_code=400, detail="Number of metadata entries should match the number of files.")
        elif isinstance(metadata, dict):
            metadata = [metadata]
        else:
            raise HTTPException(status_code=400, detail="Metadata should be a dictionary or a list of dictionaries.")
        # Load collection name from the request
        if collection_name:
            if not isinstance(collection_name, str):
                raise HTTPException(status_code=400, detail="collection_name must be a string.")

        sub_dir = collection_name if collection_name else config.vectordb.default_collection_name
        # Create a temporary directory to store files
        save_dir = Path(DATA_DIR) / sub_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the uploaded files
        for i, file in enumerate(files):
            file_path = save_dir / Path(file.filename).name
            logger.info(f"Processing file: {file.filename} and saving to {file_path}")
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())
            # Now pass the file path to the Indexer
            await indexer.add_files2vdb(path=file_path, metadata=metadata[i], collection_name=collection_name)

        return JSONResponse(content={"message": f"Files processed and added to the vector database : {[file.filename for file in files]}"}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search/", response_model=None)
async def search(query_params: SearchRequest, indexer: Indexer = Depends(get_indexer)):
    try:
        # Extract query and top_k from request
        query = query_params.query
        top_k = query_params.top_k
        collection_name = query_params.collection_name if query_params.collection_name else None
        logger.info(f"Searching for query: {query} in collection: {collection_name}")
        # Perform the search using the Indexer
        results = await indexer.vectordb.async_search(query=query, top_k=top_k, collection_name=collection_name)
        
        # Transforming the results (assuming they are LangChain documents)
        documents = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in results]
        
        # Return results
        return JSONResponse(content={f"Documents": documents}, status_code=200)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Handle errors
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete-files/",response_model=None)
async def delete_files(request : DeleteFilesRequest, indexer: Indexer = Depends(get_indexer)):
    """
    Delete points in Vector DB associated with the given file names.

    Args:
        file_names (List[str]): A list of file names whose points are to be deleted.

    Returns:
        JSONResponse: A confirmation message including details of files processed.
    """
    # Check filters format
    filters = request.filters
    if isinstance(filters, dict):
        if len(filters) != 1:
            raise HTTPException(status_code=400, detail="Each filter must be a dictionary with exactly one key-value pair.")
        filters = [filters]
    elif isinstance(filters, list):
        for f in filters:
            if not isinstance(f, dict) or len(f) != 1:
                raise HTTPException(status_code=400, detail="Each filter must be a dictionary with exactly one key-value pair.")
    else:
        raise HTTPException(status_code=400, detail="Filters must be a dictionary or a list of dictionaries.")
        
    try:
        collection_name = request.collection_name if request.collection_name else None
        deleted_files, not_found_files = indexer.delete_files(filters, collection_name)
        return {
            "message": "File processing completed.",
            "files_deleted": deleted_files,
            "files_not_found": not_found_files,
        }

    except Exception as e:
        # Handle errors
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync-db/", response_model=None)
async def sync_db(indexer: Indexer = Depends(get_indexer)):
    try:
        data_dir = Path(DATA_DIR)
        if not data_dir.exists():
            raise HTTPException(status_code=400, detail="DATA_DIR does not exist")

        sync_summary = {}

        for collection_path in data_dir.iterdir():
            if collection_path.is_dir():  # Ensure it's a collection folder
                collection_name = collection_path.name
                up_to_date_files = []
                missing_files = []

                for file_path in collection_path.iterdir():
                    if file_path.is_file() and file_path.suffix != ".md":
                        if indexer.vectordb.file_exists(file_path.name, collection_name):
                            up_to_date_files.append(file_path.name)
                        else:
                            missing_files.append(file_path.name)
                            await indexer.add_files2vdb(path=file_path, metadata={}, collection_name=collection_name)

                if not missing_files:
                    logger.info(f"Collection '{collection_name}' is already up to date.")
                else:
                    logger.info(f"Collection '{collection_name}' updated. Added files: {missing_files}")

                sync_summary[collection_name] = {
                    "up_to_date": up_to_date_files,
                    "added": missing_files
                }

        return JSONResponse(content={"message": "Database sync completed.", "details": sync_summary}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))