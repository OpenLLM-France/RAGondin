import os
import json
from typing import Dict, Any, List, Optional

import anthropic
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from starlette.routes import Mount

from mcp.server.fastmcp import FastMCP

from .utils.mcp_dependencies import MCPDependencies, get_mcp_dependencies
from .models.indexer import (
    IndexationRequest, DeletionRequest, MetadataUpdateRequest, SearchRequest,
    IndexationResult, DeletionResult, MetadataUpdateResult, SearchResult
)


# Initialisation de FastAPI
app = FastAPI(title="RAGondin MCP API", description="API RAGondin avec intégration MCP et Anthropic")

# Initialisation du serveur MCP
mcp_server = FastMCP("RAGondin MCP")

# Client Anthropic
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


class RAGQueryRequest(BaseModel):
    """Modèle de requête pour la génération RAG"""
    query: str = Field(..., description="Question de l'utilisateur")
    partition: Optional[str] = Field(None, description="Partition spécifique à utiliser")
    top_k: int = Field(5, description="Nombre de résultats à récupérer")
    similarity_threshold: float = Field(0.75, description="Seuil de similarité")
    model: str = Field("claude-3-sonnet-20240229", description="Modèle Anthropic à utiliser")
    max_tokens: int = Field(1000, description="Nombre maximum de tokens dans la réponse")
    temperature: float = Field(0.7, description="Température pour la génération")


class RAGResponse(BaseModel):
    """Modèle de réponse pour la génération RAG"""
    query: str = Field(..., description="Question originale")
    answer: str = Field(..., description="Réponse générative")
    sources: List[Dict[str, Any]] = Field(..., description="Sources utilisées")
    model: str = Field(..., description="Modèle utilisé")


# Montage du serveur MCP sur FastAPI
app.routes.append(Mount("/mcp", app=mcp_server.sse_app()))


# Définition des ressources MCP
@mcp_server.resource("indexation://partitions")
async def get_partitions_resource(
    dependencies: MCPDependencies = Depends(get_mcp_dependencies)
) -> str:
    """Récupère la liste des partitions disponibles"""
    partitions = await dependencies.provider.get_partitions()
    return json.dumps(partitions, indent=2)


@mcp_server.resource("indexation://files/{partition}")
async def get_files_in_partition_resource(
    partition: str,
    dependencies: MCPDependencies = Depends(get_mcp_dependencies)
) -> str:
    """Récupère la liste des fichiers dans une partition"""
    files = await dependencies.provider.get_files_in_partition(partition=partition)
    return json.dumps(files, indent=2)


# Définition des outils MCP
@mcp_server.tool()
async def index_documents(
    path: str,
    metadata: Optional[Dict[str, Any]] = None,
    partition: Optional[str] = None,
    dependencies: MCPDependencies = Depends(get_mcp_dependencies)
) -> str:
    """Indexe des documents"""
    request = IndexationRequest(
        path=path,
        metadata=metadata or {},
        partition=partition
    )
    
    result = await dependencies.index_documents(request)
    return json.dumps(result.dict(), indent=2)


@mcp_server.tool()
async def search_documents(
    query: str,
    top_k: int = 5,
    similarity_threshold: float = 0.80,
    partition: Optional[str] = None,
    filter: Optional[Dict[str, Any]] = None,
    dependencies: MCPDependencies = Depends(get_mcp_dependencies)
) -> str:
    """Recherche des documents"""
    request = SearchRequest(
        query=query,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        partition=partition,
        filter=filter or {}
    )
    
    results = await dependencies.search_documents(request)
    results_dict = [r.dict() for r in results]
    return json.dumps(results_dict, indent=2)


# Définition des prompts MCP
@mcp_server.prompt()
async def rag_prompt(
    query: str,
    context_documents: List[Dict[str, Any]],
) -> dict:
    """Crée un prompt RAG pour Anthropic"""
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


# Routes FastAPI
@app.post("/rag/query", response_model=RAGResponse)
async def rag_query(
    request: RAGQueryRequest,
    dependencies: MCPDependencies = Depends(get_mcp_dependencies)
) -> RAGResponse:
    """Génère une réponse à une question en utilisant RAG et Anthropic Claude
    
    Args:
        request: Paramètres de la requête
        dependencies: Dépendances MCP
        
    Returns:
        Réponse générative avec sources
    """
    try:
        # Recherche de documents pertinents
        search_request = SearchRequest(
            query=request.query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            partition=request.partition
        )
        
        search_results = await dependencies.search_documents(search_request)
        
        if not search_results:
            raise HTTPException(
                status_code=404,
                detail="Aucun document pertinent trouvé pour cette requête"
            )
        
        # Préparation du contexte pour le prompt
        context_documents = [
            {
                "content": r.content,
                "metadata": r.metadata,
                "score": r.score
            } for r in search_results
        ]
        
        # Création du prompt pour Claude
        prompt_data = await rag_prompt(
            query=request.query,
            context_documents=context_documents
        )
        
        # Appel à Anthropic
        message = client.messages.create(
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system=prompt_data.get("system", ""),
            messages=prompt_data.get("messages", [])
        )
        
        # Construction de la réponse
        return RAGResponse(
            query=request.query,
            answer=message.content[0].text,
            sources=context_documents,
            model=request.model
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du traitement de la requête RAG: {str(e)}"
        )


@app.post("/indexation/index", response_model=IndexationResult)
async def api_index_documents(
    request: IndexationRequest,
    dependencies: MCPDependencies = Depends(get_mcp_dependencies)
) -> IndexationResult:
    """Indexe des documents via l'API
    
    Args:
        request: Paramètres d'indexation
        dependencies: Dépendances MCP
        
    Returns:
        Résultat de l'indexation
    """
    try:
        return await dependencies.index_documents(request)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'indexation des documents: {str(e)}"
        )


@app.post("/indexation/search", response_model=List[SearchResult])
async def api_search_documents(
    request: SearchRequest,
    dependencies: MCPDependencies = Depends(get_mcp_dependencies)
) -> List[SearchResult]:
    """Recherche des documents via l'API
    
    Args:
        request: Paramètres de recherche
        dependencies: Dépendances MCP
        
    Returns:
        Résultats de la recherche
    """
    try:
        return await dependencies.search_documents(request)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la recherche: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ragondin.mcp_api:app", host="0.0.0.0", port=8000, reload=True) 