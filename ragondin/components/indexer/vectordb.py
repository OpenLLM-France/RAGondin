from abc import abstractmethod, ABC
import asyncio
from typing import Union, Optional
from qdrant_client import QdrantClient, models
from pymilvus import MilvusClient
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents.base import Document
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from .chunker import ABCChunker
from loguru import logger


# https://python-client.qdrant.tech/qdrant_client.qdrant_client

class ABCVectorDB(ABC):
    """
    Abstract base class for a Vector Database.
    This class defines the interface for a vector database connector.
    """

    @abstractmethod
    async def get_collections(self):
        pass

    @abstractmethod
    async def async_add_documents(self, chunks, collection_name : Optional[str] = None):
        pass

    @abstractmethod
    async def async_search(self, query: str, top_k: int = 5, collection_name : Optional[str] = None) -> list[Document]:
        pass
    
    @abstractmethod
    async def async_multy_query_search(self, queries: list[str], top_k_per_query: int = 5) -> list[Document]:
        pass
    
    @abstractmethod
    def get_file_points(self, filter: dict, collection_name):
        pass
    
    @abstractmethod
    def delete_points(self, points: list, collection_name: Optional[str] = None):
        pass
    
    @abstractmethod
    def file_exists(self, file_name: str, collection_name: Optional[str] = None):
        pass

    @abstractmethod
    def collection_exists(self, collection_name: str):
        pass


class MilvusDB(ABCVectorDB):
    "Concrete class to use a Milvus DB"

    def __init__(
            self,
            host,
            port,
            embeddings: HuggingFaceBgeEmbeddings | HuggingFaceEmbeddings=None,
            collection_name: str = None,
            logger = None,
            hybrid_mode=True):

        """
        Initialize Milvus.

        Args:
            host (str): The host of the Milvus server.
            port (int): The port of the Milvus server.
            collection_name (str): The name of the collection in the Milvus database.
            embeddings (list): The embeddings.
        """

        self.logger = logger
        self.embeddings: Union[HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings] = embeddings
        self.port = port
        self.host = host
        self.uri = f"http://{host}:{port}"
        self.client = MilvusClient(uri=self.uri)
        self.sparse_embeddings = BM25BuiltInFunction() if hybrid_mode else None
        #self.retrieval_mode = RetrievalMode.HYBRID if hybrid_mode else RetrievalMode.DENSE
        self.index_params = {"metric_type": "IP"}
        #logger.info(f"VectorDB retrieval mode: {self.retrieval_mode}")

        # Initialize collection-related attributes
        self.default_collection_name = None
        self._collection_name = None
        self.vector_store = None

        # Set the initial collection name (if provided)
        if collection_name:
            self.default_collection_name = collection_name
            self.collection_name = collection_name

    @property
    def collection_name(self):
        return self._collection_name
    
    @collection_name.setter
    def collection_name(self, name: str):
        if not name:
            if self.default_collection_name is None:
                raise ValueError("Collection name cannot be empty.")
            name = self.default_collection_name
        
        self.vector_store = Milvus(
            connection_args={"uri": self.uri},
            collection_name=name,
            embedding_function=self.embeddings,
            auto_id=True,
            index_params=self.index_params,
            primary_field="_id",
            enable_dynamic_field=True,
            #builtin_function=self.sparse_embeddings
        ) 
        self.logger.info(f"The Collection named `{name}` loaded.")

        if self.default_collection_name is None:
            self.default_collection_name = name
            self.logger.info(f"Default collection name set to `{name}`.")
        self._collection_name = name

    async def get_collections(self) -> list[str]:
        return self.client.list_collections()
    
    async def async_search(self, query: str, top_k: int = 5,similarity_threshold: int=0.80, collection_name : Optional[str] = None) -> list[Document]:
        if collection_name is None :
            if self.default_collection_name is None:
                raise ValueError("Collection name not provided and no default collection name set.")
            self.collection_name = self.default_collection_name
        elif not self.collection_exists(collection_name):
            raise ValueError(f"Collection {collection_name} does not exist.")
        else :    
            self.collection_name = collection_name

        docs_scores = await self.vector_store.asimilarity_search_with_relevance_scores(query=query, k=top_k, score_threshold=similarity_threshold)
        docs = [doc for doc, score in docs_scores]
        return docs
    
    async def async_multy_query_search(self, queries: list[str], top_k_per_query: int = 5, similarity_threshold: int=0.80, collection_name : Optional[str] = None) -> list[Document]:
        # Set the collection name
        self.collection_name = collection_name
        
        # Gather all search tasks concurrently
        search_tasks = [self.async_search(query=query, top_k=top_k_per_query, similarity_threshold=similarity_threshold, collection_name=collection_name) for query in queries]
        retrieved_results = await asyncio.gather(*search_tasks)
        # Process the retrieved documents
        retrieved_chunks = {}
        for retrieved in retrieved_results:
            if retrieved:
                for document in retrieved:
                    retrieved_chunks[document.metadata["_id"]] = document
        return list(retrieved_chunks.values())
    
    async def async_add_documents(self, chunks: list[Document], collection_name : Optional[str] = None) -> None:
        # Set the collection name
        self.collection_name = collection_name
        await self.vector_store.aadd_documents(chunks)
        self.logger.debug("CHUNKS INSERTED")
    
    def get_file_points(self, filter: dict, collection_name : Optional[str] = None, limit: int = 100):
        """
        Get the points associated with a file from Milvus
        """
        try:
            key = next(iter(filter))
            value = filter[key]
            
            # Adjust filter expression based on the type of value
            if isinstance(value, str):
                filter_expression = f"{key} == '{value}'"  # For strings, enclose in single quotes
            elif isinstance(value, int):
                filter_expression = f"{key} == {value}"  # For integers, leave as is
            else:
                raise ValueError(f"Unsupported filter value type: {type(value)}")
                
            # Pagination parameters
            offset = 0
            results = []

            while True:
                response = self.client.query(
                    collection_name=collection_name if collection_name else self.default_collection_name,
                    filter=filter_expression,
                    output_fields=["_id"],  # Only fetch IDs
                    limit=limit,
                    offset=offset
                )

                if not response:
                    break  # No more results

                results.extend([res["_id"] for res in response])
                offset += len(response)  # Move offset forward

                if limit == 1:
                    return [response[0]["_id"]] if response else []                

            return results  # Return list of result IDs

        except Exception as e:
            self.logger.error(f"Couldn't get file points for {key} {value}: {e}")
            raise

    def delete_points(self, points: list, collection_name: Optional[str] = None):
        """
        Delete points from Milvus
        """
        try:
            self.client.delete(
                collection_name=collection_name if collection_name else self.default_collection_name,
                ids=points
            )
        except Exception as e:
            self.logger.error(f"Error in `delete_points`: {e}")
        pass

    def file_exists(self, file_name: str, collection_name: Optional[str] = None):
        """
        Check if a file exists in Qdrant
        """
        try:
            # Get points associated with the file name
            points = self.get_file_points({"file_name": file_name}, collection_name, limit=1)
            return True if points else False
        except Exception as e:
            self.logger.error(f"Error in `file_exists` for {file_name}: {e}")
            return False
    
    def collection_exists(self, collection_name: str):
        """
        Check if a collection exists in Milvus
        """
        return self.client.has_collection(collection_name)


class QdrantDB(ABCVectorDB):
    """
    Concrete class for a Qdrant Vector Database.
    This class implements the **`BaseVectorDd`** interface for a Qdrant database.
    """

    def __init__(
            self, 
            host, 
            port, 
            embeddings: HuggingFaceBgeEmbeddings | HuggingFaceEmbeddings=None,
            collection_name: str = None, 
            logger = None,
            hybrid_mode=True,
        ):
        """
        Initialize Qdrant_Connector.

        Args:
            host (str): The host of the Qdrant server.
            port (int): The port of the Qdrant server.
            collection_name (str): The name of the collection in the Qdrant database.
            embeddings (list): The embeddings.
        """

        self.logger = logger
        self.embeddings: Union[HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings] = embeddings
        self.port = port
        self.host = host
        self.client = QdrantClient(
            port=port, host=host,
            prefer_grpc=False,
        )

        self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25") if hybrid_mode else None
        self.retrieval_mode = RetrievalMode.HYBRID if hybrid_mode else RetrievalMode.DENSE
        logger.info(f"VectorDB retrieval mode: {self.retrieval_mode}")
        
        # Initialize collection-related attributes
        self.default_collection_name = None
        self._collection_name = None
        self.vector_store = None

        # Set the initial collection name (if provided)
        if collection_name:
            self.default_collection_name = collection_name
            self.collection_name = collection_name
        
    @property
    def collection_name(self):
        return self._collection_name
    
    @collection_name.setter
    def collection_name(self, name: str):
        if not name:
            if self.default_collection_name is None:
                raise ValueError("Collection name cannot be empty.")
            name = self.default_collection_name

        if self.client.collection_exists(collection_name=name):
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=name,
                embedding=self.embeddings,
                sparse_embedding=self.sparse_embeddings,
                retrieval_mode=self.retrieval_mode,
            ) 
            self.logger.warning(f"Collection `{name}` LOADED.")
        else:
            self.vector_store = QdrantVectorStore.construct_instance(
                embedding=self.embeddings,
                sparse_embedding=self.sparse_embeddings,
                collection_name=name,
                client_options={'port': self.port, 'host':self.host},
                retrieval_mode=self.retrieval_mode,
            )
            self.logger.debug(f"Collection `{name}` CREATED.")
        
        if self.default_collection_name is None:
            self.default_collection_name = name
            self.logger.info(f"Default collection name set to `{name}`.")
        self._collection_name = name


    async def get_collections(self) -> list[str]:
        return [c.name for c in self.client.get_collections().collections]

    async def async_search(self, query: str, top_k: int = 5, similarity_threshold: int=0.80, collection_name : Optional[str] = None) -> list[Document]:
        if collection_name is None :
            if self.default_collection_name is None:
                raise ValueError("Collection name not provided and no default collection name set.")
            self.collection_name = self.default_collection_name
        elif not self.collection_exists(collection_name):
            raise ValueError(f"Collection {collection_name} does not exist.")
        else :    
            self.collection_name = collection_name
        docs_scores = await self.vector_store.asimilarity_search_with_relevance_scores(query=query, k=top_k, score_threshold=similarity_threshold)
        docs = [doc for doc, score in docs_scores]
        return docs
    

    async def async_multy_query_search(self, queries: list[str], top_k_per_query: int = 5, similarity_threshold: int=0.80, collection_name : Optional[str] = None) -> list[Document]:
        # Set the collection name
        self.collection_name = collection_name
        # Gather all search tasks concurrently
        search_tasks = [self.async_search(query=query, top_k=top_k_per_query, similarity_threshold=similarity_threshold, collection_name=collection_name) for query in queries]
        retrieved_results = await asyncio.gather(*search_tasks)
        retrieved_chunks = {}
        # Process the retrieved documents
        for retrieved in retrieved_results:
            if retrieved:
                for document in retrieved:
                    retrieved_chunks[document.metadata["_id"]] = document
        return list(retrieved_chunks.values())
    

    async def async_add_documents(self, chunks: list[Document], collection_name : Optional[str] = None) -> None:
        # Set the collection name
        self.collection_name = collection_name
        
        await self.vector_store.aadd_documents(chunks)
        self.logger.debug("CHUNKS INSERTED")
    
    
    def get_file_points(self, filter: dict, collection_name : Optional[str] = None, limit: int = 100):
        """
        Get the points associated with a file from Qdrant
        """
        try:
            # Scroll through all vectors
            has_more = True
            offset = None
            results = []

            key = next(iter(filter))
            value = filter[key]

            while has_more:
                response = self.client.scroll(
                    collection_name=collection_name if collection_name else self.default_collection_name,
                    scroll_filter=models.Filter(must=[models.FieldCondition(key=f"metadata.{key}",match=models.MatchValue(value=value))]),
                    limit=limit,
                    offset=offset,
                )
                
                # Add points that contain the filename in metadata.source
                results.extend(response[0])
                has_more = response[1]  # Check if there are more results
                offset = response[1] if has_more else None

                if limit == 1:
                    return [results[0].id] if results else []

            # Return list of result ids
            return [res.id for res in results]
        
        except Exception as e:
            self.logger.error(f"Couldn't get file points for {key} {value}: {e}")
            raise
        

    def delete_points(self, points: list, collection_name: Optional[str] = None):
        """
        Delete points from Qdrant
        """
        try:
            self.indexer.vectordb.client.delete(
                collection_name=collection_name if collection_name else self.default_collection_name,
                points_selector=models.PointIdsList(points=points)
            )
        except Exception as e:
            self.logger.error(f"Error in `delete_points`: {e}")


    def file_exists(self, file_name: str, collection_name: Optional[str] = None):
        """
        Check if a file exists in Qdrant
        """
        try:
            # Get points associated with the file name
            points = self.get_file_points({"file_name": file_name}, collection_name, limit=1)
            return True if points else False
        except Exception as e:
            self.logger.error(f"Error in `file_exists` for {file_name}: {e}")
            return False
    
    def collection_exists(self, collection_name: str):
        """
        Check if a collection exists in Qdrant
        """
        return self.client.collection_exists(collection_name)

class ConnectorFactory:
    CONNECTORS = {
        "milvus": MilvusDB,
        "qdrant": QdrantDB
    }

    @staticmethod
    def create_vdb(config, logger, embeddings) -> ABCVectorDB:

        if config["vectordb"]["enable"] == False:
            logger.info("Vector database is not enabled. Skipping initialization.")
            return None

        # Extract parameters
        dbconfig = dict(config.vectordb)
        name = dbconfig.pop("connector_name")
        dbconfig.pop("enable")
        vdb_cls = ConnectorFactory.CONNECTORS.get(name)
        if not vdb_cls:
            raise ValueError(f"VECTORDB '{name}' is not supported.")

        dbconfig['embeddings'] = embeddings
        dbconfig['logger'] = logger

        return vdb_cls(**dbconfig)