from typing import Dict, List, Tuple, Callable, Any
from pathlib import Path
from fastapi import APIRouter
from mcp import MCPTool, MCPResource
from mcp.server.fastmcp import FastMCP

from fastapi import FastAPI
from fastmcp import MCP
from fastmcp.tools import Tool
from typing import get_type_hints, Callable
from pydantic import create_model
from inspect import signature

app = FastAPI()
mcp = MCP()

def MCPToolRoute(
    request_type: str = "POST",
    path: str = "/",
    router: APIRouter = None,
    mcp_server :FastMCP=None,
    prefix: str = ""
):


    def decorator(func: Callable):
        sig = signature(func)
        hints = get_type_hints(func)
        name = func.__name__
        fields = {
            param: (hints[param], ...)
            for param in sig.parameters
        }
        InputModel = create_model(f"{name.capitalize()}Input", **fields)

        # Créer la fonction FastAPI qui appelle func


        # Attacher la route dynamiquement
        route_func = getattr(router, request_type.lower())(path)(func)

        # Enregistrer comme outil FastMCP

        if request_type == "POST":
            mcp_server.tool(name)(func)
        elif request_type == "GET":
            mcp_server.resource(f"{prefix}/{path}")(func)

        # Retourner la fonction décorée FastAPI
        return route_func

    return decorator


class RouterManager:
    """Gestionnaire de routes API et de ressources MCP"""
    
    def __init__(self, file_path: str):
        """Initialise le gestionnaire avec le chemin du fichier
        
        Args:
            file_path: Chemin du fichier contenant les routes
        """
        self.file_path = file_path
        self.prefix = Path(file_path).stem  # Nom du fichier sans extension
        self.router = APIRouter(prefix=f"/{self.prefix}", tags=[self.prefix])
        self.mcp_server = FastMCP(self.prefix)
        self.mcp_tool = MCPTool()
        self.routes: List[Tuple[str, str, Callable]] = []  # [(request_type, path, func)]
        self.api_wrappers: Dict[Callable, Callable] = {}  # {func: wrapped_func}
        self.mcp_wrappers: Dict[Callable, Callable] = {}  # {func: wrapped_func}
        
    def append(self, request_type: str, path: str, func: Callable) -> None:
        """Ajoute une route à configurer
        
        Args:
            request_type: Type de requête (get, post, etc.)
            path: Chemin de la route
            func: Fonction à wrapper
        """
        self.routes.append((request_type, path, func))
        
    def setup_router(self) -> APIRouter:
        """Configure le router FastAPI avec toutes les routes
        
        Returns:
            Router FastAPI configuré
        """
        for request_type, path, func in self.routes:
            # Wrapper la fonction avec le décorateur approprié
            wrapped_func = getattr(self.router, request_type)(path)(func)
            self.api_wrappers[func] = wrapped_func
            
        return self.router
        
    def setup_mcp(self) -> MCPTool:
        """Configure les ressources MCP pour les routes GET
        
        Returns:
            MCPTool configuré
        """
        for request_type, path, func in self.routes:
            if request_type == "get":
                # Créer une ressource MCP pour chaque route GET
                resource = MCPResource(
                    name=f"{self.prefix}/{path}",
                    description=f"Resource for {path}",
                    parameters=func.__annotations__ if hasattr(func, '__annotations__') else {}
                )
                
                # Wrapper la fonction pour la ressource MCP
                wrapped_func = resource(func)
                self.mcp_wrappers[func] = wrapped_func
                
                # Ajouter la ressource au tool
                self.mcp_tool.add_resource(resource)
            else:
                # Si ce n'est pas une route GET, c'est un outil
                tool = MCPTool()
                
                # Wrapper la fonction pour l'outil
                wrapped_func = tool(func)
                self.mcp_wrappers[func] = wrapped_func
                
                # Ajouter l'outil au tool
                self.mcp_tool.add_tool(tool)
                
        return self.mcp_tool 