from typing import List
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from ..utils.api_dependencies import get_api_dependencies, APIDependencies

router = APIRouter(
    prefix="/collections",
    tags=["collections"],
    responses={404: {"description": "Collection non trouvée"}},
)

@router.get("/")
async def get_collections(
    dependencies: APIDependencies = Depends(get_api_dependencies)
) -> JSONResponse:
    """Récupère toutes les collections existantes"""
    try:
        collections = await dependencies.get_collections()
        return JSONResponse(content={"collections": collections}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 