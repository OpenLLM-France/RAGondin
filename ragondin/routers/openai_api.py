from typing import Optional, List, Dict, Any, Union
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field
from omegaconf import OmegaConf

from ..models.openai import (
    OpenAICompletionRequest, OpenAICompletion, OpenAICompletionChoice,
    OpenAICompletionChunk, OpenAICompletionChunkChoice, OpenAIMessage,
    OpenAIUsage, ChatMsg, mapping, Tool, ToolChoice
)
from ..utils.api_dependencies import get_api_dependencies
from config import load_config

router = APIRouter(prefix="/v1", tags=["openai"])

@router.post("/chat/completions")
async def chat_completions(
    request: OpenAICompletionRequest,
    static_base_url: str,
    app_state: Any,
    api_deps = Depends(get_api_dependencies)
):
    """Endpoint pour la complétion de chat OpenAI
    
    Args:
        request: Requête de complétion
        static_base_url: URL de base pour les ressources statiques
        app_state: État de l'application
        api_deps: Dépendances de l'API
        
    Returns:
        Réponse de complétion (streaming ou non)
    """
    try:
        # Appeler le service de complétion
        return await api_deps.chat_completions(
            request=request,
            static_base_url=static_base_url,
            app_state=app_state
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de la complétion de chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 