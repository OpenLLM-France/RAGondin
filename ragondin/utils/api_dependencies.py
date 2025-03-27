from typing import Optional, Dict, Any, Callable, List, Union
from functools import lru_cache
from fastapi import Depends, HTTPException, status, UploadFile, Form, Request
from pathlib import Path
import json
from loguru import logger
import openai
from fastapi.responses import StreamingResponse
import uuid
import time
from mcp import MCPTool, MCPResource, MCPPrompt
from mcp.types import CallToolRequest, CallToolRequestParams, CallToolResult
from pydantic_core import from_json

from ..models.indexer import (
    IndexationRequest, DeletionRequest, MetadataUpdateRequest, SearchRequest,
    IndexationResult, DeletionResult, MetadataUpdateResult, SearchResult
)
from ..models.openai import (
    OpenAICompletionRequest, OpenAICompletion, OpenAICompletionChoice,
    OpenAICompletionChunk, OpenAICompletionChunkChoice, OpenAIMessage,
    OpenAIUsage, ChatMsg, mapping
)
from ..controllers.indexer_controller import IndexerController
from ..components.indexer.indexer import Indexer
from config import load_config
from omegaconf import OmegaConf
from .dependencies import vectordb

def source2url(s: dict, static_base_url: str):
    s["url"] = f"{static_base_url}/{s['sub_url_path']}"
    s.pop("source")
    s.pop("sub_url_path")
    return s

def get_indexer(config: OmegaConf = Depends(load_config)) -> Indexer:
    """Dépendance pour obtenir une instance d'Indexer
    
    Args:
        config: Configuration du système
        
    Returns:
        Instance d'Indexer configurée
    """
    return Indexer(config=config, logger=logger)


def get_indexer_controller(config: OmegaConf = Depends(load_config)) -> IndexerController:
    """Dépendance pour obtenir une instance du contrôleur d'indexation
    
    Args:
        config: Configuration du système
        
    Returns:
        Instance du contrôleur d'indexation
    """
    return IndexerController(config=config)


class APIDependencies:
    """Classe qui gère les dépendances de l'API et les opérations d'API"""
    
    def __init__(self, config: OmegaConf):
        """Initialise les dépendances avec une configuration
        
        Args:
            config: Configuration du système
        """
        self.config = config
        self.controller = IndexerController(config=config)
        self.data_dir = Path(config.paths.data_dir)
        self.mcp_tool = MCPTool()
        self._setup_mcp_resources()
        
    def _setup_mcp_resources(self):
        """Configure les ressources MCP"""
        # Ressource pour lister les outils
        tools_resource = MCPResource(
            name="tools",
            description="Liste tous les outils disponibles",
            parameters={}
        )
        self.mcp_tool.add_resource(tools_resource)
        
        # Prompt pour la contextualisation des outils
        contextualize_prompt = MCPPrompt(
            name="contextualize_tools",
            description="Prompt pour contextualiser la question utilisateur avec les outils disponibles",
            template="""Tu es un assistant IA qui peut utiliser des outils pour répondre aux questions.
Voici les outils disponibles:

{available_tools}

Historique du chat:
{chat_history}

Question de l'utilisateur: {question}

Analyse la question et décide si l'utilisation d'outils est nécessaire.
Si oui, structure ta réponse pour utiliser les outils appropriés.
Si non, réponds directement à la question.
Réponds toujours en français.""",
            parameters={
                "question": str,
                "chat_history": List[Dict[str, str]],
                "available_tools": List[Dict[str, Any]]
            }
        )
        self.mcp_tool.add_prompt(contextualize_prompt)
        
    async def add_file(
        self,
        partition: str,
        file_id: str,
        file: UploadFile,
        metadata: Optional[Any] = Form(None)
    ) -> Dict[str, Any]:
        """Ajoute un fichier dans une partition spécifique
        
        Args:
            partition: Nom de la partition
            file_id: Identifiant du fichier
            file: Fichier à uploader
            metadata: Métadonnées optionnelles
            
        Returns:
            Réponse JSON avec le statut de l'opération
        """
        # Vérifier si le fichier existe déjà
        if vectordb.file_exists(file_id, partition):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"File '{file_id}' already exists in partition {partition}",
            )

        # Charger les métadonnées
        try:
            metadata = metadata or "{}"
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="Invalid JSON in metadata"
            )
        if not isinstance(metadata, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Metadata must be a dictionary",
            )

        # Ajouter file_id aux métadonnées
        metadata["file_id"] = file_id

        # Créer un répertoire temporaire pour stocker les fichiers
        save_dir = self.data_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        # Sauvegarder le fichier uploadé
        file_path = save_dir / Path(file.filename).name
        try:
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save file: {str(e)}",
            )

        # Indexer le fichier
        try:
            result = await self.controller.index_documents(
                IndexationRequest(
                    path=file_path,
                    metadata=metadata,
                    partition=partition
                )
            )
            if not result.success:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=result.message
                )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Indexing error: {str(e)}",
            )

        return {
            "message": f"File '{file_id}' successfully indexed in partition '{partition}'"
        }
        
    async def delete_file(self, partition: str, file_id: str) -> None:
        """Supprime un fichier d'une partition spécifique
        
        Args:
            partition: Nom de la partition
            file_id: Identifiant du fichier à supprimer
        """
        try:
            result = await self.controller.delete_document(
                DeletionRequest(
                    file_id=file_id,
                    partition=partition
                )
            )
            if not result.success:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=result.message
                )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error while deleting file '{file_id}': {str(e)}",
            )
            
    async def update_file(
        self,
        partition: str,
        file_id: str,
        file: UploadFile,
        metadata: Optional[Any] = Form(None)
    ) -> Dict[str, Any]:
        """Met à jour un fichier dans une partition spécifique
        
        Args:
            partition: Nom de la partition
            file_id: Identifiant du fichier
            file: Nouveau fichier
            metadata: Nouvelles métadonnées optionnelles
            
        Returns:
            Réponse JSON avec le statut de l'opération
        """
        # Vérifier l'existence du fichier
        if not vectordb.file_exists(file_id, partition):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File '{file_id}' not found in partition '{partition}'.",
            )

        # Supprimer l'ancien fichier
        try:
            result = await self.controller.delete_document(
                DeletionRequest(
                    file_id=file_id,
                    partition=partition
                )
            )
            if not result.success:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=result.message
                )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete existing file: {str(e)}",
            )

        # Parser les métadonnées
        try:
            metadata = metadata or "{}"
            metadata = json.loads(metadata)
            if not isinstance(metadata, dict):
                raise ValueError("Metadata is not a dictionary.")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid metadata: {str(e)}",
            )

        metadata["file_id"] = file_id

        # Sauvegarder le nouveau fichier
        save_dir = self.data_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        file_path = save_dir / Path(file.filename).name
        try:
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save file: {str(e)}",
            )

        # Indexer le nouveau fichier
        try:
            result = await self.controller.index_documents(
                IndexationRequest(
                    path=file_path,
                    metadata=metadata,
                    partition=partition
                )
            )
            if not result.success:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=result.message
                )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Indexing error: {str(e)}",
            )

        return {
            "message": f"File '{file_id}' successfully updated in partition '{partition}'"
        }
        
    async def update_metadata(
        self,
        partition: str,
        file_id: str,
        metadata: Optional[Any] = Form(None)
    ) -> Dict[str, Any]:
        """Met à jour les métadonnées d'un fichier
        
        Args:
            partition: Nom de la partition
            file_id: Identifiant du fichier
            metadata: Nouvelles métadonnées
            
        Returns:
            Réponse JSON avec le statut de l'opération
        """
        # Vérifier l'existence du fichier
        if not vectordb.file_exists(file_id, partition):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File '{file_id}' not found in partition '{partition}'.",
            )

        # Parser les métadonnées
        try:
            metadata = metadata or "{}"
            metadata = json.loads(metadata)
            if not isinstance(metadata, dict):
                raise ValueError("Metadata must be a JSON object.")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid metadata: {str(e)}",
            )

        metadata["file_id"] = file_id

        # Mettre à jour les métadonnées
        try:
            result = await self.controller.update_metadata(
                MetadataUpdateRequest(
                    file_id=file_id,
                    metadata=metadata,
                    partition=partition
                )
            )
            if not result.success:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=result.message
                )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update metadata: {str(e)}",
            )

        return {
            "message": f"Metadata for file '{file_id}' successfully updated."
        }
        
    async def sync_database(self) -> Dict[str, Any]:
        """Synchronise la base de données avec les fichiers
        
        Returns:
            Réponse JSON avec le résumé de la synchronisation
        """
        try:
            if not self.data_dir.exists():
                raise HTTPException(status_code=400, detail="DATA_DIR does not exist")

            sync_summary = {}

            for collection_path in self.data_dir.iterdir():
                if collection_path.is_dir():  # S'assurer que c'est un dossier de collection
                    collection_name = collection_path.name
                    up_to_date_files = []
                    missing_files = []

                    for file_path in collection_path.iterdir():
                        if file_path.is_file() and file_path.suffix != ".md":
                            if vectordb.file_exists(file_path.name, collection_name):
                                up_to_date_files.append(file_path.name)
                            else:
                                missing_files.append(file_path.name)
                                await self.controller.index_documents(
                                    IndexationRequest(
                                        path=file_path,
                                        metadata={},
                                        partition=collection_name
                                    )
                                )

                    if not missing_files:
                        logger.info(f"Collection '{collection_name}' is already up to date.")
                    else:
                        logger.info(f"Collection '{collection_name}' updated. Added files: {missing_files}")

                    sync_summary[collection_name] = {
                        "up_to_date": up_to_date_files,
                        "added": missing_files,
                    }

            return {
                "message": "Database sync completed.",
                "details": sync_summary
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def search_multiple_partitions(
        self,
        request: Request,
        partitions: Optional[List[str]] = None,
        text: str = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Recherche dans plusieurs partitions
        
        Args:
            request: Requête FastAPI pour la génération des URLs
            partitions: Liste des partitions à rechercher
            text: Texte à rechercher
            top_k: Nombre de résultats à retourner
            
        Returns:
            Résultats de la recherche avec liens HATEOAS
        """
        try:
            results = await self.controller.search_documents(
                SearchRequest(
                    query=text,
                    top_k=top_k,
                    partition=partitions
                )
            )

            # Construire la réponse HATEOAS
            documents = [
                {
                    "link": str(
                        request.url_for("get_extract", extract_id=doc.metadata["_id"])
                    )
                }
                for doc in results
            ]

            return {"Documents": documents}

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def search_one_partition(
        self,
        request: Request,
        partition: str,
        text: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Recherche dans une partition spécifique
        
        Args:
            request: Requête FastAPI pour la génération des URLs
            partition: Partition à rechercher
            text: Texte à rechercher
            top_k: Nombre de résultats à retourner
            
        Returns:
            Résultats de la recherche avec liens HATEOAS
        """
        try:
            results = await self.controller.search_documents(
                SearchRequest(
                    query=text,
                    top_k=top_k,
                    partition=partition
                )
            )

            # Construire la réponse HATEOAS
            documents = [
                {
                    "link": str(
                        request.url_for("get_extract", extract_id=doc.metadata["_id"])
                    )
                }
                for doc in results
            ]

            return {"Documents": documents}

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def search_file(
        self,
        request: Request,
        partition: str,
        file_id: str,
        query: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Recherche dans un fichier spécifique
        
        Args:
            request: Requête FastAPI pour la génération des URLs
            partition: Partition contenant le fichier
            file_id: Identifiant du fichier
            query: Texte à rechercher
            top_k: Nombre de résultats à retourner
            
        Returns:
            Résultats de la recherche avec liens HATEOAS
        """
        try:
            results = await self.controller.search_documents(
                SearchRequest(
                    query=query,
                    top_k=top_k,
                    partition=partition,
                    filter={"file_id": file_id}
                )
            )

            # Construire la réponse HATEOAS
            documents = [
                {
                    "link": str(
                        request.url_for("get_extract", extract_id=doc.metadata["_id"])
                    )
                }
                for doc in results
            ]

            return {"Documents": documents}

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def get_extract(self, extract_id: str) -> Dict[str, Any]:
        """Récupère un extrait par son ID
        
        Args:
            extract_id: Identifiant de l'extrait
            
        Returns:
            Contenu et métadonnées de l'extrait
        """
        try:
            doc = vectordb.get_chunk_by_id(extract_id)
            return {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def get_collections(self) -> List[str]:
        """Récupère toutes les collections existantes
        
        Returns:
            Liste des noms des collections
        """
        try:
            return await self.controller.vectordb.get_collections()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """Récupère la liste des outils disponibles via MCP
        
        Returns:
            Liste des outils avec leurs descriptions et paramètres
        """
        try:
            tools = await self.mcp_tool.get_tools()
            return tools
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des outils: {str(e)}")
            return []
            
    async def call_tool(self, tool_call: Dict[str, Any]) -> Any:
        """Appelle un outil via MCP
        
        Args:
            tool_call: Appel d'outil à exécuter
            
        Returns:
            Résultat de l'appel d'outil
        """
        try:
            request = CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(
                    name=tool_call["function"]["name"],
                    arguments=from_json(tool_call["function"]["arguments"], allow_partial=True)
                )
            )
            result = await self.mcp_tool.call_tool(request)
            return result
        except Exception as e:
            logger.error(f"Erreur lors de l'appel de l'outil: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
            
    async def get_contextualized_prompt(
        self,
        question: str,
        chat_history: List[Dict[str, str]],
        available_tools: List[Dict[str, Any]]
    ) -> str:
        """Récupère le prompt contextualisé via MCP
        
        Args:
            question: Question de l'utilisateur
            chat_history: Historique du chat
            available_tools: Liste des outils disponibles
            
        Returns:
            Prompt contextualisé
        """
        try:
            # Formater l'historique du chat
            formatted_history = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in chat_history
            ])
            
            # Formater les outils disponibles
            formatted_tools = json.dumps(available_tools, indent=2)
            
            # Récupérer le prompt formaté
            prompt = await self.mcp_tool.get_prompt(
                "contextualize_tools",
                {
                    "question": question,
                    "chat_history": formatted_history,
                    "available_tools": formatted_tools
                }
            )
            return prompt
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du prompt contextualisé: {str(e)}")
            return question  # En cas d'erreur, retourner la question brute
            
    async def chat_completions(
        self,
        request: OpenAICompletionRequest,
        static_base_url: str,
        app_state: Any
    ) -> StreamingResponse | OpenAICompletion:
        """Gère la logique de complétion de chat OpenAI avec RAG
        
        Args:
            request: Requête de complétion OpenAI
            static_base_url: URL de base pour les ressources statiques
            app_state: État de l'application
            
        Returns:
            Réponse de l'API OpenAI (streaming ou non)
        """
        # Récupérer le dernier message utilisateur
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(
                status_code=400, detail="At least one user message is required"
            )

        new_user_input = user_messages[-1].content

        # Convertir l'historique des messages
        chat_history = []
        for msg in request.messages[:-1]:  # Exclure le dernier message utilisateur
            if msg.role in ["user", "assistant"]:
                chat_history.append(ChatMsg(role=msg.role, content=msg.content))

        msgs = None
        if chat_history:
            msgs = [
                mapping[chat_msg.role](content=chat_msg.content)
                for chat_msg in chat_history
            ]

        # Récupérer les outils disponibles
        available_tools = await self.get_available_tools()
        
        # Récupérer le prompt contextualisé
        contextualized_prompt = await self.get_contextualized_prompt(
            question=new_user_input,
            chat_history=[{"role": msg.role, "content": msg.content} for msg in chat_history],
            available_tools=available_tools
        )

        # Exécuter le pipeline RAG avec le prompt contextualisé
        answer_stream, context, sources = await app_state.ragpipe.run(
            partition=["all"], question=contextualized_prompt, chat_history=msgs
        )

        # Gérer les sources
        sources = list(map(lambda x: source2url(x, static_base_url), sources))
        src_json = json.dumps(sources)

        # Créer l'ID de réponse
        response_id = f"chatcmpl-{str(uuid.uuid4())}"
        created_time = int(time.time())
        model_name = app_state.model_name

        if request.stream:
            # Réponse streaming compatible OpenAI
            async def stream_response():
                full_response = ""
                chunk = OpenAICompletionChunk(
                    id=response_id,
                    created=created_time,
                    model=model_name,
                    choices=[
                        OpenAICompletionChunkChoice(
                            index=0, delta={"role": "assistant"}, finish_reason=None
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

                # Envoyer les tokens un par un
                async for token in answer_stream:
                    full_response += token.content
                    chunk = OpenAICompletionChunk(
                        id=response_id,
                        created=created_time,
                        model=model_name,
                        choices=[
                            OpenAICompletionChunkChoice(
                                index=0,
                                delta={"content": token.content},
                                finish_reason=None,
                            )
                        ],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"

                # Vérifier si la réponse contient des appels d'outils
                if hasattr(answer_stream, "tool_calls") and answer_stream.tool_calls:
                    # Gérer les appels d'outils
                    tool_results = []
                    for tool_call in answer_stream.tool_calls:
                        result = await self.call_tool(tool_call)
                        tool_results.append((tool_call.id, result))
                    
                    # Envoyer les appels d'outils
                    chunk = OpenAICompletionChunk(
                        id=response_id,
                        created=created_time,
                        model=model_name,
                        choices=[
                            OpenAICompletionChunkChoice(
                                index=0,
                                delta={
                                    "tool_calls": [
                                        {
                                            "id": tool_call.id,
                                            "type": tool_call.type,
                                            "function": {
                                                "name": tool_call.function.name,
                                                "arguments": tool_call.function.arguments
                                            }
                                        }
                                        for tool_call in answer_stream.tool_calls
                                    ]
                                },
                                finish_reason="tool_calls",
                            )
                        ],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"
                    
                    # Envoyer les résultats des outils
                    for tool_id, result in tool_results:
                        chunk = OpenAICompletionChunk(
                            id=response_id,
                            created=created_time,
                            model=model_name,
                            choices=[
                                OpenAICompletionChunkChoice(
                                    index=0,
                                    delta={
                                        "role": "tool",
                                        "tool_call_id": tool_id,
                                        "content": str(result)
                                    },
                                    finish_reason=None,
                                )
                            ],
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n"
                else:
                    # Envoyer le chunk final sans outils
                    chunk = OpenAICompletionChunk(
                        id=response_id,
                        created=created_time,
                        model=model_name,
                        choices=[
                            OpenAICompletionChunkChoice(index=0, delta={}, finish_reason="stop")
                        ],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"
                    
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream",
                headers={"X-Metadata-Sources": src_json},
            )
        else:
            # Réponse non streaming
            full_response = ""
            async for token in answer_stream:
                full_response += token.content

            # Vérifier si la réponse contient des appels d'outils
            if hasattr(answer_stream, "tool_calls") and answer_stream.tool_calls:
                # Récupérer les outils disponibles
                available_tools = await self.get_available_tools()
                
                # Gérer les appels d'outils
                tool_results = []
                for tool_call in answer_stream.tool_calls:
                    result = await self.call_tool(tool_call)
                    tool_results.append((tool_call.id, result))
                
                # Créer la réponse avec les appels d'outils
                completion = OpenAICompletion(
                    id=response_id,
                    created=created_time,
                    model=model_name,
                    choices=[
                        OpenAICompletionChoice(
                            index=0,
                            message=OpenAIMessage(
                                role="assistant",
                                content=full_response,
                                tool_calls=[
                                    {
                                        "id": tool_call.id,
                                        "type": tool_call.type,
                                        "function": {
                                            "name": tool_call.function.name,
                                            "arguments": tool_call.function.arguments
                                        }
                                    }
                                    for tool_call in answer_stream.tool_calls
                                ]
                            ),
                            finish_reason="tool_calls",
                        )
                    ],
                    usage=OpenAIUsage(
                        prompt_tokens=100,  # Valeurs approximatives
                        completion_tokens=len(full_response.split()) * 4 // 3,  # Estimation
                        total_tokens=100 + len(full_response.split()) * 4 // 3,
                    ),
                )
            else:
                # Réponse normale sans outils
                completion = OpenAICompletion(
                    id=response_id,
                    created=created_time,
                    model=model_name,
                    choices=[
                        OpenAICompletionChoice(
                            index=0,
                            message=OpenAIMessage(role="assistant", content=full_response),
                            finish_reason="stop",
                        )
                    ],
                    usage=OpenAIUsage(
                        prompt_tokens=100,  # Valeurs approximatives
                        completion_tokens=len(full_response.split()) * 4 // 3,  # Estimation
                        total_tokens=100 + len(full_response.split()) * 4 // 3,
                    ),
                )

            return completion


# Fonction utilitaire pour obtenir une instance des dépendances de l'API
def get_api_dependencies(config: OmegaConf = Depends(load_config)) -> APIDependencies:
    """Fournit une instance des dépendances de l'API
    
    Args:
        config: Configuration du système
        
    Returns:
        Instance de APIDependencies
    """
    return APIDependencies(config=config) 