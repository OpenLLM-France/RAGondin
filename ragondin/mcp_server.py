import os
import json
from typing import Dict, List, Any, Optional

from mcp.server.fastmcp import FastMCP
import openai

from .models.indexer import (
    IndexationRequest, DeletionRequest, MetadataUpdateRequest, SearchRequest,
    IndexationResult, DeletionResult, MetadataUpdateResult, SearchResult
)
from .utils.api_dependencies import MCPDependencies, get_mcp_dependencies
from .utils.config import Config, get_config


# Initialisation du serveur MCP
mcp = FastMCP("RAGondin")

# Création des dépendances
config = get_config()
dependencies = MCPDependencies(config)

# Configuration de l'API OpenAI
openai.api_key = os.environ.get("OPENAI_API_KEY")


@mcp.resource("indexation://partitions")
async def get_partitions_resource() -> str:
    """Récupère la liste des partitions disponibles sous forme de ressource"""
    partitions = await dependencies.provider.get_partitions()
    return json.dumps(partitions, indent=2)


@mcp.resource("indexation://files/{partition}")
async def get_files_in_partition_resource(partition: str) -> str:
    """Récupère la liste des fichiers dans une partition spécifique"""
    files = await dependencies.provider.get_files_in_partition(partition=partition)
    return json.dumps(files, indent=2)


@mcp.resource("document://{partition}/{file_id}")
async def get_document_chunks_resource(partition: str, file_id: str) -> str:
    """Récupère les chunks d'un document spécifique"""
    # Simuler une recherche pour trouver tous les chunks d'un document spécifique
    search_request = SearchRequest(
        query="",  # Recherche vide pour récupérer tous les chunks
        top_k=100,  # Nombre élevé pour récupérer tous les chunks
        similarity_threshold=0.0,  # Seuil minimal pour tout récupérer
        partition=partition,
        filter={"file_id": file_id}  # Filtre sur l'ID du fichier
    )
    
    results = await dependencies.search_documents(search_request)
    
    # Formater les résultats
    formatted_chunks = [
        {
            "content": r.content,
            "metadata": r.metadata
        } for r in results
    ]
    
    return json.dumps(formatted_chunks, indent=2)


@mcp.tool()
async def index_documents(
    path: str,
    metadata: Optional[Dict[str, Any]] = None,
    partition: Optional[str] = None
) -> str:
    """Indexe un ou plusieurs documents
    
    Args:
        path: Chemin ou liste de chemins vers les fichiers à indexer
        metadata: Métadonnées à associer aux documents
        partition: Partition dans laquelle indexer les documents
        
    Returns:
        Résultat de l'opération d'indexation au format JSON
    """
    request = IndexationRequest(
        path=path,
        metadata=metadata or {},
        partition=partition
    )
    
    result = await dependencies.index_documents(request)
    return json.dumps(result.dict(), indent=2)


@mcp.tool()
async def delete_document(
    file_id: str,
    partition: str
) -> str:
    """Supprime un document
    
    Args:
        file_id: ID du fichier à supprimer
        partition: Partition contenant le fichier
        
    Returns:
        Résultat de l'opération de suppression au format JSON
    """
    request = DeletionRequest(
        file_id=file_id,
        partition=partition
    )
    
    result = await dependencies.delete_document(request)
    return json.dumps(result.dict(), indent=2)


@mcp.tool()
async def update_metadata(
    file_id: str,
    metadata: Dict[str, Any],
    partition: str
) -> str:
    """Met à jour les métadonnées d'un document
    
    Args:
        file_id: ID du fichier à mettre à jour
        metadata: Nouvelles métadonnées
        partition: Partition contenant le fichier
        
    Returns:
        Résultat de l'opération de mise à jour au format JSON
    """
    request = MetadataUpdateRequest(
        file_id=file_id,
        metadata=metadata,
        partition=partition
    )
    
    result = await dependencies.update_metadata(request)
    return json.dumps(result.dict(), indent=2)


@mcp.tool()
async def search_documents(
    query: str,
    top_k: int = 5,
    similarity_threshold: float = 0.80,
    partition: Optional[str] = None,
    filter: Optional[Dict[str, Any]] = None
) -> str:
    """Recherche des documents
    
    Args:
        query: Requête de recherche
        top_k: Nombre de résultats à retourner
        similarity_threshold: Seuil de similarité minimal
        partition: Partition dans laquelle effectuer la recherche
        filter: Filtres supplémentaires
        
    Returns:
        Résultats de recherche au format JSON
    """
    request = SearchRequest(
        query=query,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        partition=partition,
        filter=filter or {}
    )
    
    results = await dependencies.search_documents(request)
    
    # Convertir les résultats en dictionnaires pour la sérialisation JSON
    results_dict = [r.dict() for r in results]
    return json.dumps(results_dict, indent=2)


@mcp.prompt()
async def rag_prompt(
    query: str,
    context_documents: Optional[List[Dict[str, Any]]] = None
) -> dict:
    """Crée un prompt RAG pour Anthropic
    
    Args:
        query: La question de l'utilisateur
        context_documents: Documents de contexte (optionnel)
        
    Returns:
        Prompt structuré pour le modèle
    """
    # Si les documents de contexte ne sont pas fournis, rechercher dans l'index
    if not context_documents:
        search_request = SearchRequest(
            query=query,
            top_k=5,
            similarity_threshold=0.80
        )
        
        results = await dependencies.search_documents(search_request)
        context_documents = [
            {
                "content": r.content,
                "metadata": r.metadata
            } for r in results
        ]
    
    # Construction du contexte
    context_str = ""
    for i, doc in enumerate(context_documents):
        context_str += f"\n--- Document {i+1} ---\n"
        context_str += f"Contenu: {doc['content']}\n"
        context_str += f"Métadonnées: {json.dumps(doc['metadata'])}\n"
    
    # Construire le prompt pour Claude
    return {
        "messages": [
            {
                "role": "user",
                "content": f"""Voici une question de l'utilisateur. Utilise le contexte fourni pour y répondre de façon précise.
                
## Question:
{query}

## Contexte:
{context_str}

Réponds uniquement en te basant sur les informations du contexte. Si le contexte ne contient pas assez d'informations pour répondre, indique-le clairement."""
            }
        ],
        "system": "Tu es un assistant IA spécialisé dans la recherche documentaire, appelé RAGondin. Tu fournis des réponses précises basées uniquement sur les documents de contexte qui te sont fournis."
    }


@mcp.tool()
async def call_openai_api(
    prompt: str,
    model: str = "gpt-3.5-turbo",
    max_tokens: int = 1000,
) -> str:
    """Appelle l'API OpenAI pour générer une réponse
    Args:
        prompt: Le prompt pour l'API OpenAI
        model: Le modèle à utiliser
        max_tokens: Nombre maximum de tokens dans la réponse
    Returns:
        Réponse générée par l'API OpenAI
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erreur lors de l'appel à l'API OpenAI: {str(e)}"


if __name__ == "__main__":
    mcp.run() 