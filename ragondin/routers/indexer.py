from loguru import logger
from typing import Optional, Any, List
from fastapi import APIRouter, HTTPException, File, UploadFile, Depends, Form, Query
from fastapi.responses import JSONResponse
from pathlib import Path
from components import Indexer
from utils.dependencies import get_indexer
from config.config import load_config
import json

#load config
config = load_config()
DATA_DIR = config.paths.data_dir
# Create an APIRouter instance
router = APIRouter()


@router.post("/{partition}/{file_id}", response_model=None)
async def add_file(
    partition: str,
    file_id: str, 
    file: UploadFile = File(...),
    metadata: Optional[Any] = Form(None),
    indexer: Indexer = Depends(get_indexer)):
    try:
        # Load metadata
        metadata = metadata or "{}" 
        metadata = json.loads(metadata)
        if not isinstance(metadata, dict):
            raise HTTPException(status_code=400, detail="Metadata should be a dictionary")
        
        # CHeck partition
        if not isinstance(partition, str):
            raise HTTPException(status_code=400, detail="partition must be a string.")
            
        # Add file_id to metadata
        metadata["file_id"] = file_id

        # Create a temporary directory to store files
        save_dir = Path(DATA_DIR)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the uploaded file
        file_path = save_dir / Path(file.filename).name
        logger.info(f"Processing file: {file.filename} and saving to {file_path}")
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        # Now pass the file path to the Indexer
        await indexer.add_files2vdb(path=file_path, metadata=metadata, partition=partition)

        return JSONResponse(content={"message": f"File processed and added to the vector database : {file.filename}"}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=None)
async def search_multiple_partitions(
    partitions: List[str] = Query(..., description="Comma-separated list of partitions to search"),
    query: str = Query(..., description="Text to search semantically"),
    top_k: int = Query(5, description="Number of top results to return"),
    indexer: Indexer = Depends(get_indexer),
):
    try:
        # Perform the search using the Indexer
        results = await indexer.asearch(query=query, top_k=top_k, partition=partitions)
        
        # Transforming the results (assuming they are LangChain documents)
        documents = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in results]
        
        # Return results
        return JSONResponse(content={"Documents": documents}, status_code=200)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Handle errors
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{partition}", response_model=None)
async def search_one_partition(
    partition : str,
    query: str = Query(..., description="Text to search semantically"),
    top_k: int = Query(5, description="Number of top results to return"),
    indexer: Indexer = Depends(get_indexer)
    ):
    try:
        # Perform the search using the Indexer
        results = await indexer.asearch(query=query, top_k=top_k, partition=partition)
        
        # Transforming the results (assuming they are LangChain documents)
        documents = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in results]
        
        # Return results
        return JSONResponse(content={"Documents": documents}, status_code=200)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Handle errors
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{partition}/{file_id}", response_model=None)
async def search_file(
    partition : str,
    file_id: str,
    query: str = Query(..., description="Text to search semantically"),
    top_k: int = Query(5, description="Number of top results to return"),
    indexer: Indexer = Depends(get_indexer)
    ):
    try:
        # Perform the search using the Indexer
        results = await indexer.asearch(query=query, top_k=top_k, partition=partition, filter= {"file_id": file_id})
        
        # Transforming the results (assuming they are LangChain documents)
        documents = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in results]
        
        # Return results
        return JSONResponse(content={"Documents": documents}, status_code=200)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Handle errors
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{partition}/{file_id}", response_model=None)
async def delete_file(
    partition: str,
    file_id: str,
    indexer: Indexer = Depends(get_indexer)):
    """
    Delete a file in a specific partition.
    """    
    try:
        deleted = indexer.delete_file(file_id, partition)
        if deleted:
            return JSONResponse(
                content={"message": "File successfully deleted", "file_id": file_id},
                status_code=200
            )
        else:
            return JSONResponse(
                content={"message": "File not found", "file_id": file_id},
                status_code=404
            )
    except Exception as e:
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