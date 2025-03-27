from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
from langchain_core.documents.base import Document

class IndexationRequest(BaseModel):
    """Modèle de requête pour l'indexation de documents"""
    path: Union[str, List[str]] = Field(..., description="Chemin ou liste de chemins vers les fichiers à indexer")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Métadonnées à associer aux documents")
    partition: Optional[str] = Field(None, description="Partition dans laquelle indexer les documents")

class DeletionRequest(BaseModel):
    """Modèle de requête pour la suppression de documents"""
    file_id: str = Field(..., description="ID du fichier à supprimer")
    partition: str = Field(..., description="Partition contenant le fichier")

class MetadataUpdateRequest(BaseModel):
    """Modèle de requête pour la mise à jour de métadonnées"""
    file_id: str = Field(..., description="ID du fichier à mettre à jour")
    metadata: Dict[str, Any] = Field(..., description="Nouvelles métadonnées à associer")
    partition: str = Field(..., description="Partition contenant le fichier")

class SearchRequest(BaseModel):
    """Modèle de requête pour la recherche de documents"""
    query: str = Field(..., description="Requête de recherche")
    top_k: int = Field(default=5, description="Nombre de résultats à retourner")
    similarity_threshold: float = Field(default=0.80, description="Seuil de similarité minimal")
    partition: Optional[Union[str, List[str]]] = Field(None, description="Partition(s) dans laquelle(s) effectuer la recherche")
    filter: Dict[str, Any] = Field(default_factory=dict, description="Filtres supplémentaires")

class SearchResult(BaseModel):
    """Modèle de résultat pour la recherche de documents"""
    content: str = Field(..., description="Contenu du document")
    metadata: Dict[str, Any] = Field(..., description="Métadonnées associées au document")
    score: float = Field(..., description="Score de similarité")
    
    @classmethod
    def from_document(cls, doc: Document, score: Optional[float] = None) -> "SearchResult":
        """Crée un résultat de recherche à partir d'un document LangChain"""
        return cls(
            content=doc.page_content,
            metadata=doc.metadata,
            score=score or doc.metadata.get("score", 0.0)
        )

class IndexationResult(BaseModel):
    """Modèle de résultat pour l'indexation de documents"""
    success: bool = Field(..., description="Indique si l'opération a réussi")
    message: str = Field(..., description="Message de résultat")
    file_count: int = Field(default=0, description="Nombre de fichiers indexés")
    chunk_count: int = Field(default=0, description="Nombre de chunks indexés")

class DeletionResult(BaseModel):
    """Modèle de résultat pour la suppression de documents"""
    success: bool = Field(..., description="Indique si l'opération a réussi")
    message: str = Field(..., description="Message de résultat")
    deleted_points: int = Field(default=0, description="Nombre de points supprimés")

class MetadataUpdateResult(BaseModel):
    """Modèle de résultat pour la mise à jour de métadonnées"""
    success: bool = Field(..., description="Indique si l'opération a réussi")
    message: str = Field(..., description="Message de résultat")
    updated_chunks: int = Field(default=0, description="Nombre de chunks mis à jour")
