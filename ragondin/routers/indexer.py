from loguru import logger
from typing import Optional, Any, List
from fastapi import APIRouter, HTTPException, status, File, UploadFile, Depends
from fastapi.responses import JSONResponse
from pathlib import Path
from crud.qdrant import QdrantCRUD
from models.indexer import SearchRequest, DeleteFilesRequest
from utils.dependencies import get_qdrant_crud
from config.config import load_config

#load config
config = load_config()

# Create an APIRouter instance
router = APIRouter()


@router.post("/add-files/", response_model=None)
async def add_files(files: List[UploadFile] = File(...), qdrant_crud: QdrantCRUD = Depends(get_qdrant_crud)):
    try:
        # Create a temporary directory to store files
        temp_dir = Path(config.paths.data_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the uploaded files to the temporary directory
        file_paths = []
        for file in files:
            file_path = temp_dir / Path(file.filename).name
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())
            file_paths.append(file_path)
        
        # Now pass the directory path to the Indexer
        await qdrant_crud.add_files(path=temp_dir)
        
        return JSONResponse(content={"message": "Files processed and added to the vector database."}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.post("/search/", response_model=None)
async def search(query_params: SearchRequest, qdrant_crud: QdrantCRUD = Depends(get_qdrant_crud)):
    try:
        # Extract query and top_k from request
        query = query_params.query
        top_k = query_params.top_k
        
        # Perform the search using the Indexer
        results = await qdrant_crud.search(query, top_k)
        
        # Transforming the results (assuming they are LangChain documents)
        documents = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in results]
        
        # Return results
        return JSONResponse(content={"results": documents}, status_code=200)

    except Exception as e:
        # Handle errors
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete-files/",response_model=None)
async def delete_files(request : DeleteFilesRequest, qdrant_crud: QdrantCRUD = Depends(get_qdrant_crud)):
    """
    Delete points in Qdrant associated with the given file names.

    Args:
        file_names (List[str]): A list of file names whose points are to be deleted.

    Returns:
        JSONResponse: A confirmation message including details of files processed.
    """

    try:
        deleted_files, not_found_files = qdrant_crud.delete_files(request.file_names)
        return {
            "message": "File processing completed.",
            "files_deleted": deleted_files,
            "files_not_found": not_found_files,
        }

    except Exception as e:
        # Handle errors
        raise HTTPException(status_code=500, detail=str(e))