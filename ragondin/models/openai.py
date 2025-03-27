from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field


# Classes pour la compatibilité OpenAI
class OpenAIMessage(BaseModel):
    """Modèle représentant un message dans l'API OpenAI."""

    role: str
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None


class OpenAICompletionRequest(BaseModel):
    """Modèle représentant une requête de complétion pour l'API OpenAI."""

    model: str
    messages: List[ChatMsg]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, ToolChoice]] = None


class OpenAICompletionChoice(BaseModel):
    """Modèle représentant un choix de complétion dans l'API OpenAI."""

    index: int
    message: OpenAIMessage
    finish_reason: Optional[str] = None


class OpenAIUsage(BaseModel):
    """Modèle représentant les statistiques d'utilisation dans l'API OpenAI."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAICompletion(BaseModel):
    """Modèle représentant une réponse de complétion dans l'API OpenAI."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[OpenAICompletionChoice]
    usage: OpenAIUsage


class OpenAICompletionChunkChoice(BaseModel):
    """Modèle représentant un choix de segment de complétion en streaming dans l'API OpenAI."""

    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None


class OpenAICompletionChunk(BaseModel):
    """Modèle représentant un segment de complétion en streaming dans l'API OpenAI."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[OpenAICompletionChunkChoice]


class ChatMsg(BaseModel):
    """Modèle pour un message de chat"""
    role: str
    content: str


class Tool(BaseModel):
    """Modèle pour un outil OpenAI"""
    type: str = Field(default="function")
    function: Dict[str, Any]


class ToolChoice(BaseModel):
    """Modèle pour le choix d'un outil OpenAI"""
    type: str = Field(default="function")
    function: Dict[str, str]


# Mapping des rôles vers les types de messages
mapping = {
    "user": ChatMsg,
    "assistant": ChatMsg,
    "system": ChatMsg,
    "tool": ChatMsg
}
