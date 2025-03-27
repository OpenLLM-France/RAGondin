from typing import List, Optional
from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from ..utils.api_dependencies import get_api_dependencies, APIDependencies
from ..utils.router_manager import RouterManager, MCPToolRoute

# Créer une instance du gestionnaire de routes
router_manager = RouterManager(__file__)

async def search_multiple_partitions(
    request: Request,
    partitions: Optional[List[str]] = None,
    text: str = None,
    top_k: int = 5,
    dependencies: APIDependencies = Depends(get_api_dependencies)
) -> JSONResponse:
    """Recherche dans plusieurs partitions"""
    try:
        results = await dependencies.search_multiple_partitions(
            request=request,
            partitions=partitions,
            text=text,
            top_k=top_k
        )
        return JSONResponse(content=results, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@MCPToolRoute(request_type="GET", 
              path="/partition/{partition}", 
              router=router, 
              mcp_server=mcp_server)
async def search_one_partition(
    request: Request,
    partition: str,
    text: str,
    top_k: int = 5,
    dependencies: APIDependencies = Depends(get_api_dependencies)
) -> JSONResponse:
    """Recherche dans une partition spécifique"""
    try:
        results = await dependencies.search_one_partition(
            request=request,
            partition=partition,
            text=text,
            top_k=top_k
        )
        return JSONResponse(content=results, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def search_file(
    request: Request,
    partition: str,
    file_id: str,
    query: str,
    top_k: int = 5,
    dependencies: APIDependencies = Depends(get_api_dependencies)
) -> JSONResponse:
    """Recherche dans un fichier spécifique"""
    try:
        results = await dependencies.search_file(
            request=request,
            partition=partition,
            file_id=file_id,
            query=query,
            top_k=top_k
        )
        return JSONResponse(content=results, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def get_extract(
    extract_id: str,
    dependencies: APIDependencies = Depends(get_api_dependencies)
) -> JSONResponse:
    """Récupère un extrait par son ID"""
    try:
        results = await dependencies.get_extract(extract_id=extract_id)
        return JSONResponse(content=results, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Ajouter les routes au gestionnaire
router_manager.append("get", "/", search_multiple_partitions)
router_manager.append("get", "/partition/{partition}", search_one_partition)
router_manager.append("get", "/partition/{partition}/file/{file_id}", search_file)
router_manager.append("get", "/{extract_id}", get_extract)

# Configurer le router et les ressources MCP
router = router_manager.setup_router()
mcp_tool = router_manager.setup_mcp() 