from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# Classes pour la compatibilité OpenAI
class OpenAIMessage(BaseModel):
    """Modèle représentant un message dans l'API OpenAI."""

    role: Literal["user", "assistant", "system"]
    content: str


class OpenAICompletionRequest(BaseModel):
    """Modèle représentant une requête de complétion pour l'API OpenAI."""

    model: str = Field(..., description="model name")
    messages: List[OpenAIMessage]
    temperature: Optional[float] = Field(0.7)
    top_p: Optional[float] = Field(1.0)
    stream: Optional[bool] = Field(False)
    max_tokens: Optional[int] = Field(None)


class OpenAICompletionChoice(BaseModel):
    """Modèle représentant un choix de complétion dans l'API OpenAI."""

    index: int
    message: OpenAIMessage
    finish_reason: str


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
    delta: Dict[str, str]
    finish_reason: Optional[str] = None
    metadata: str = None


class OpenAICompletionChunk(BaseModel):
    """Modèle représentant un segment de complétion en streaming dans l'API OpenAI."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[OpenAICompletionChunkChoice]
