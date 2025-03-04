from components import Indexer
from config import load_config
from qdrant_client import models
from typing import List
from loguru import logger

# Load the configuration
config = load_config()

class QdrantCRUD :
    def __init__(self, indexer: Indexer = Indexer(config=config, logger=logger)):
        self.indexer = indexer
        self.collection_name = config.vectordb.collection_name
        self.logger = self.indexer.logger

    async def add_files(self, path: str):
        """
        Add files to Qdrant
        """
        try:
            await self.indexer.add_files2vdb(path=path)
        except Exception as e:
            self.logger.error(f"Couldn't add directory {path} to Qdrant : {e}")
            raise
        
    async def search(self, query: str, top_k: int):
        return await self.indexer.vectordb.async_search(query, top_k)


    def get_file_points(self, file_name: str):
        """
        Get the points associated with a file from Qdrant
        """
        try:
            # Scroll through all vectors
            has_more = True
            offset = None
            results = []

            while has_more:
                response = self.indexer.vectordb.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=models.Filter(must=[]),  # No initial filter for substring
                    limit=100,
                    offset=offset,
                )
                
                # Add points that contain the filename in metadata.source
                results.extend(
                    point
                    for point in response[0]
                    if file_name in point.payload.get("metadata", {}).get("source", "")
                )
                has_more = response[1]  # Check if there are more results
                offset = response[1] if has_more else None

            # Return list of result ids
            return [res.id for res in results]
        
        except Exception as e:
            self.logger.error(f"Couldn't get file points for file {file_name}: {e}")
            raise


    def delete_points(self, points: list):
        """
        Delete points from Qdrant
        """
        try:
            self.indexer.vectordb.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=points)
            )
        except Exception as e:
            self.logger.error(f"Error in `delete_points`: {e}")
    
    def delete_files(self, file_names: List[str]):
        deleted_files = []
        not_found_files = []

        for file_name in file_names:
            try:
                # Get points associated with the file name
                points = self.get_file_points(file_name)
                print(file_name, len(points))
                if not points:
                    self.logger.info(f"No points found for file: {file_name}")
                    not_found_files.append(file_name)
                    continue

                # Delete the points
                self.delete_points(points)
                deleted_files.append(file_name)

            except Exception as e:
                self.logger.error(f"Error in `delete_files` for file {file_name}: {e}")
        
        return deleted_files, not_found_files

    def update_file ():
        pass
        