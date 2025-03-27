from typing import List, Optional, Dict, Any, Type, TypeVar, Union
from pydantic import BaseModel
from loguru import logger
import openai
from mcp import MCPTool, MCPResource
from fastapi import HTTPException
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)

ModelT = TypeVar("ModelT", bound=BaseModel)

class ChatMessage(BaseModel):
    """Modèle pour les messages du chat"""
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    """Modèle pour les requêtes de complétion"""
    messages: List[ChatMessage]
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, str]]] = None

class ChatCompletionResponse(BaseModel):
    """Modèle pour les réponses de complétion"""
    id: str
    choices: List[Dict[str, Any]]
    created: int
    model: str
    usage: Dict[str, int]

class OnPremiseLLMClient:
    """Client pour interagir avec notre LLM on-premise via MCP"""
    
    def __init__(self, base_url: str, api_key: str):
        """Initialise le client
        
        Args:
            base_url: URL de base de notre API on-premise
            api_key: Clé API pour l'authentification
        """
        self.base_url = base_url
        self.api_key = api_key
        self.mcp_tool = MCPTool()
        self._setup_resources()
        
    def _setup_resources(self):
        """Configure les ressources MCP"""
        # Ressource pour la complétion de chat
        chat_resource = MCPResource(
            name="chat/completions",
            description="Génère des réponses de chat en utilisant notre LLM on-premise",
            parameters={
                "messages": List[ChatMessage],
                "model": str,
                "temperature": float,
                "max_tokens": Optional[int],
                "top_p": float,
                "frequency_penalty": float,
                "presence_penalty": float,
                "stop": Optional[List[str]],
                "tools": Optional[List[Dict[str, Any]]],
                "tool_choice": Optional[Union[str, Dict[str, str]]]
            }
        )
        
        # Wrapper la fonction de complétion
        wrapped_func = chat_resource(self.chat_completion)
        self.mcp_tool.add_resource(chat_resource)
        
    async def chat_completion(
        self,
        messages: List[ChatMessage],
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, str]]] = None
    ) -> ChatCompletionResponse:
        """Génère une complétion de chat
        
        Args:
            messages: Liste des messages du chat
            model: Nom du modèle à utiliser
            temperature: Température pour la génération
            max_tokens: Nombre maximum de tokens à générer
            top_p: Paramètre top_p pour la génération
            frequency_penalty: Pénalité de fréquence
            presence_penalty: Pénalité de présence
            stop: Liste des séquences d'arrêt
            tools: Liste des outils disponibles
            tool_choice: Choix de l'outil à utiliser
            
        Returns:
            Réponse de complétion
        """
        try:
            # Préparer la requête
            request = ChatCompletionRequest(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
                tools=tools,
                tool_choice=tool_choice
            )
            
            # Configurer le client OpenAI avec notre URL on-premise
            client = openai.OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
            
            # Appeler l'API
            response = await client.chat.completions.create(
                **request.model_dump(exclude_none=True)
            )
            
            # Convertir la réponse en notre format
            return ChatCompletionResponse(
                id=response.id,
                choices=[
                    {
                        "message": {
                            "role": choice.message.role,
                            "content": choice.message.content,
                            "tool_calls": [
                                {
                                    "id": tool_call.id,
                                    "type": tool_call.type,
                                    "function": {
                                        "name": tool_call.function.name,
                                        "arguments": tool_call.function.arguments
                                    }
                                }
                                for tool_call in choice.message.tool_calls
                            ] if hasattr(choice.message, "tool_calls") else None
                        },
                        "finish_reason": choice.finish_reason
                    }
                    for choice in response.choices
                ],
                created=response.created,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            )
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de la complétion: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
            
    async def generate_str(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, str]]] = None
    ) -> str:
        """Génère une réponse textuelle
        
        Args:
            message: Message de l'utilisateur
            system_prompt: Prompt système optionnel
            tools: Liste des outils disponibles
            tool_choice: Choix de l'outil à utiliser
            
        Returns:
            Réponse générée
        """
        messages = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=message))
        
        response = await self.chat_completion(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice
        )
        return response.choices[0]["message"]["content"]
        
    async def generate_structured(
        self,
        message: str,
        response_model: Type[ModelT],
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, str]]] = None
    ) -> ModelT:
        """Génère une réponse structurée
        
        Args:
            message: Message de l'utilisateur
            response_model: Modèle Pydantic pour la réponse
            system_prompt: Prompt système optionnel
            tools: Liste des outils disponibles
            tool_choice: Choix de l'outil à utiliser
            
        Returns:
            Réponse structurée
        """
        messages = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=message))
        
        response = await self.chat_completion(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice
        )
        content = response.choices[0]["message"]["content"]
        return response_model.model_validate_json(content) 