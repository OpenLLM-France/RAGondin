import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, File, Form, UploadFile, Request, Response, status
from fastapi.responses import JSONResponse

from ..models.indexer import (
    IndexationRequest, DeletionRequest, MetadataUpdateRequest, SearchRequest,
    IndexationResult, DeletionResult, MetadataUpdateResult, SearchResult
)
from ..utils.mcp_dependencies import get_mcp_dependencies, MCPDependencies
from ..utils.dependencies import vectordb
from config.config import load_config

# Charger la configuration
config = load_config()
DATA_DIR = config.paths.data_dir

router = APIRouter(
    prefix="/partition",
    tags=["indexation"],
    responses={404: {"description": "Opération non trouvée"}},
)

@router.post("/{partition}/file/{file_id}")
async def add_file(
    request: Request,
    partition: str,
    file_id: str,
    file: UploadFile = File(...),
    metadata: Optional[Any] = Form(None),
    dependencies: MCPDependencies = Depends(get_mcp_dependencies)
):
    """Ajoute un fichier dans une partition spécifique"""
    result = await dependencies.add_file(partition, file_id, file, metadata)
    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content=result
    )

@router.delete("/{partition}/file/{file_id}")
async def delete_file(
    partition: str,
    file_id: str,
    dependencies: MCPDependencies = Depends(get_mcp_dependencies)
):
    """Supprime un fichier d'une partition spécifique"""
    await dependencies.delete_file(partition, file_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@router.put("/{partition}/file/{file_id}")
async def put_file(
    request: Request,
    partition: str,
    file_id: str,
    file: UploadFile = File(...),
    metadata: Optional[Any] = Form(None),
    dependencies: MCPDependencies = Depends(get_mcp_dependencies)
):
    """Met à jour un fichier dans une partition spécifique"""
    result = await dependencies.update_file(partition, file_id, file, metadata)
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content=result
    )

@router.patch("/{partition}/file/{file_id}")
async def patch_file(
    partition: str,
    file_id: str,
    metadata: Optional[Any] = Form(None),
    dependencies: MCPDependencies = Depends(get_mcp_dependencies)
):
    """Met à jour les métadonnées d'un fichier"""
    result = await dependencies.update_metadata(partition, file_id, metadata)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=result
    )

@router.post("/sync-db/")
async def sync_db(dependencies: MCPDependencies = Depends(get_mcp_dependencies)):
    """Synchronise la base de données avec les fichiers"""
    result = await dependencies.sync_database()
    return JSONResponse(
        content=result,
        status_code=200
    ) 