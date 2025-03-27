import os
import json
import asyncio
from typing import Dict, Any, List, Optional

import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client import MCPClient
from mcp.server.fastmcp import FastMCP

from .mcp_server import mcp_server

# Paramètres du serveur MCP
server_params = StdioServerParameters(
    command="python",
    args=["-m", "ragondin.mcp_server"],
    env={"ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY")}
)

# Client Anthropic
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


async def run_rag_query(query: str, partition: Optional[str] = None) -> str:
    """Exécute une requête RAG en utilisant le serveur MCP et Anthropic Claude
    
    Args:
        query: La question de l'utilisateur
        partition: Partition spécifique à utiliser (optionnel)
        
    Returns:
        Réponse du modèle
    """
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialiser la connexion
            await session.initialize()
            
            # Chercher les documents pertinents
            filter_args = {}
            if partition:
                filter_args["partition"] = partition
                
            results_json = await session.call_tool(
                "search_documents", 
                arguments={
                    "query": query,
                    "top_k": 5,
                    "similarity_threshold": 0.75,
                    **({"partition": partition} if partition else {})
                }
            )
            
            # Parser les résultats
            results = json.loads(results_json)
            
            if not results:
                return "Aucun document pertinent trouvé pour cette requête."
            
            # Obtenir le prompt RAG
            prompt_result = await session.get_prompt(
                "rag_prompt",
                arguments={
                    "query": query,
                    "context_documents": results
                }
            )
            
            # Appeler Anthropic Claude avec le prompt
            message = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                system=prompt_result.get("system", ""),
                messages=prompt_result.get("messages", [])
            )
            
            return message.content[0].text


async def index_documents_with_mcp(
    path: str, 
    metadata: Optional[Dict[str, Any]] = None,
    partition: Optional[str] = None
) -> str:
    """Indexe des documents en utilisant le serveur MCP
    
    Args:
        path: Chemin vers les documents à indexer
        metadata: Métadonnées à associer (optionnel)
        partition: Partition à utiliser (optionnel)
        
    Returns:
        Résultat de l'indexation
    """
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialiser la connexion
            await session.initialize()
            
            # Indexer les documents
            result_json = await session.call_tool(
                "index_documents",
                arguments={
                    "path": path,
                    **({"metadata": metadata} if metadata else {}),
                    **({"partition": partition} if partition else {})
                }
            )
            
            return result_json


async def list_partitions() -> List[str]:
    """Liste les partitions disponibles
    
    Returns:
        Liste des partitions
    """
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialiser la connexion
            await session.initialize()
            
            # Lire la ressource des partitions
            content, _ = await session.read_resource("indexation://partitions")
            return json.loads(content)


async def run_openai_query(prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int = 1000) -> str:
    """Exécute une requête en utilisant l'API OpenAI via le MCP
    Args:
        prompt: Le prompt pour l'API OpenAI
        model: Le modèle à utiliser
        max_tokens: Nombre maximum de tokens dans la réponse
    Returns:
        Réponse générée par l'API OpenAI
    """
    async with MCPClient(mcp_server) as session:
        result = await session.call_tool("call_openai_api", prompt=prompt, model=model, max_tokens=max_tokens)
        return result


async def main():
    """Exemple d'utilisation de l'intégration MCP avec RAGondin et Anthropic Claude"""
    # Vérification de la clé API Anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("⚠️ Veuillez définir la variable d'environnement ANTHROPIC_API_KEY")
        return
    
    # Exemple: Indexer un document
    print("📑 Indexation d'un document...")
    result = await index_documents_with_mcp(
        path="./data/sample.txt",
        metadata={"source": "exemple", "auteur": "RAGondin"},
        partition="test"
    )
    print(f"Résultat de l'indexation: {result}")
    
    # Exemple: Lister les partitions
    print("\n🗂️ Partitions disponibles:")
    partitions = await list_partitions()
    print(partitions)
    
    # Exemple: Requête RAG
    print("\n❓ Réponse à une question...")
    question = "Que contient le document sample.txt?"
    answer = await run_rag_query(question, partition="test")
    print(f"Question: {question}")
    print(f"Réponse: {answer}")


if __name__ == "__main__":
    asyncio.run(main()) 