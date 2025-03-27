from typing import List, Optional
from ..models.indexer import (
    IndexationRequest, DeletionRequest, MetadataUpdateRequest, SearchRequest,
    IndexationResult, DeletionResult, MetadataUpdateResult, SearchResult
)
from ..components.indexer.indexer import Indexer
from ..utils.config import Config
import logging
import ray

logger = logging.getLogger(__name__)

class IndexerController:
    """Contrôleur pour les opérations d'indexation
    
    Cette classe expose des méthodes de haut niveau pour interagir avec
    le système d'indexation, en servant d'intermédiaire entre les
    requêtes API et l'Indexer.
    """
    
    def __init__(self, config: Config) -> None:
        """Initialise le contrôleur avec une configuration
        
        Args:
            config: Configuration du système
        """
        self.config = config
        self.indexer = Indexer.remote(config=config, logger=logger)
        self.logger = logger
    
    async def index_documents(self, request: IndexationRequest) -> IndexationResult:
        """Indexe des documents selon les paramètres de la requête
        
        Args:
            request: Paramètres d'indexation incluant chemin, métadonnées et partition
            
        Returns:
            Résultat de l'opération d'indexation
        """
        try:
            self.logger.info(f"Démarrage de l'indexation des documents: {request.path}")
            
            # Utiliser directement l'Indexer avec Ray
            await ray.get(self.indexer.add_files2vdb.remote(
                path=request.path,
                metadata=request.metadata,
                partition=request.partition
            ))
            
            files_count = 1 if isinstance(request.path, str) else len(request.path)
            
            return IndexationResult(
                success=True,
                message=f"Indexation réussie de {files_count} fichier(s)",
                file_count=files_count,
                chunk_count=0  # On ne peut pas facilement compter les chunks ici
            )
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'indexation: {str(e)}")
            return IndexationResult(
                success=False,
                message=f"Échec de l'indexation: {str(e)}",
                file_count=0,
                chunk_count=0
            )
    
    async def delete_document(self, request: DeletionRequest) -> DeletionResult:
        """Supprime un document selon les paramètres de la requête
        
        Args:
            request: Paramètres de suppression incluant l'ID du fichier et la partition
            
        Returns:
            Résultat de l'opération de suppression
        """
        try:
            self.logger.info(f"Suppression du document: {request.file_id} dans {request.partition}")
            
            # Utiliser directement l'Indexer avec Ray
            await ray.get(self.indexer.delete_file.remote(
                file_id=request.file_id,
                partition=request.partition
            ))
            
            return DeletionResult(
                success=True,
                message=f"Suppression réussie du document {request.file_id}",
                deleted_points=1  # On ne peut pas facilement compter les points supprimés
            )
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la suppression: {str(e)}")
            return DeletionResult(
                success=False,
                message=f"Échec de la suppression: {str(e)}",
                deleted_points=0
            )
    
    async def update_metadata(self, request: MetadataUpdateRequest) -> MetadataUpdateResult:
        """Met à jour les métadonnées d'un document
        
        Args:
            request: Paramètres de mise à jour incluant l'ID du fichier, 
                    les nouvelles métadonnées et la partition
            
        Returns:
            Résultat de l'opération de mise à jour
        """
        try:
            self.logger.info(f"Mise à jour des métadonnées pour: {request.file_id}")
            
            # Utiliser directement l'Indexer avec Ray
            await ray.get(self.indexer.update_file_metadata.remote(
                file_id=request.file_id,
                metadata=request.metadata,
                partition=request.partition
            ))
            
            return MetadataUpdateResult(
                success=True,
                message=f"Mise à jour réussie des métadonnées pour {request.file_id}",
                updated_chunks=1  # On ne peut pas facilement compter les chunks mis à jour
            )
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la mise à jour des métadonnées: {str(e)}")
            return MetadataUpdateResult(
                success=False,
                message=f"Échec de la mise à jour des métadonnées: {str(e)}",
                updated_chunks=0
            )
    
    async def search_documents(self, request: SearchRequest) -> List[SearchResult]:
        """Recherche des documents selon les paramètres de la requête
        
        Args:
            request: Paramètres de recherche incluant requête, top_k, seuil, partition et filtres
            
        Returns:
            Liste des résultats de recherche
        """
        try:
            self.logger.info(f"Recherche pour la requête: {request.query}")
            
            # Utiliser directement l'Indexer avec Ray
            documents = await ray.get(self.indexer.asearch.remote(
                query=request.query,
                top_k=request.top_k,
                similarity_threshold=request.similarity_threshold,
                partition=request.partition,
                filter=request.filter
            ))
            
            # Convertir les documents en résultats de recherche
            results = [SearchResult.from_document(doc) for doc in documents]
            
            return results
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la recherche: {str(e)}")
            return [] 