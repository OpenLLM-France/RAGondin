from abc import abstractmethod, ABC
import asyncio
from typing import Union
from qdrant_client import QdrantClient, models
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
    async def async_add_documents(self, chunks):
        pass

    @abstractmethod
    async def async_search(self, query: str, top_k: int = 5) -> list[Document]:
        pass
    
    @abstractmethod
    async def async_multy_query_search(self, queries: list[str], top_k_per_query: int = 5) -> list[Document]:
        pass


class QdrantDB(ABCVectorDB):
    """
    Concrete class for a Qdrant Vector Database.
    This class implements the **`BaseVectorDd`** interface for a Qdrant database.
    """

    def __init__(
            self, 
            host, port, 
            embeddings: HuggingFaceBgeEmbeddings | HuggingFaceEmbeddings=None,
            collection_name: str = None, logger = None,
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
        self._collection_name = None
        self.vector_store = None

        # Set the initial collection name (if provided)
        if collection_name:
            self.collection_name = collection_name
        
    @property
    def collection_name(self):
        return self._collection_name
    
    @collection_name.setter
    def collection_name(self, name: str):
        if not name:
            raise ValueError("Collection name cannot be empty.")
        
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


    async def get_collections(self) -> list[str]:
        return [c.name for c in self.client.get_collections().collections]

    async def async_search(self, query: str, top_k: int = 5, similarity_threshold: int=0.80) -> list[Document]:
        docs_scores = await self.vector_store.asimilarity_search_with_relevance_scores(query=query, k=top_k, score_threshold=similarity_threshold)
        docs = [doc for doc, score in docs_scores]
        return docs
    

    async def async_multy_query_search(self, queries: list[str], top_k_per_query: int = 5, similarity_threshold: int=0.80) -> list[Document]:
        # Gather all search tasks concurrently
        search_tasks = [self.async_search(query=query, top_k=top_k_per_query, similarity_threshold=similarity_threshold) for query in queries]
        retrieved_results = await asyncio.gather(*search_tasks)
        
        retrieved_chunks = {}
        # Process the retrieved documents
        for retrieved in retrieved_results:
            if retrieved:
                for document in retrieved:
                    retrieved_chunks[document.metadata["_id"]] = document
        return list(retrieved_chunks.values())
    

    async def async_add_documents(self, chunks: list[Document]) -> None:
        await self.vector_store.aadd_documents(chunks)
        self.logger.debug("CHUNKS INSERTED")


class ConnectorFactory:
    CONNECTORS = {
        "qdrant": QdrantDB
    }

    @staticmethod
    def create_vdb(config, logger, embeddings) -> ABCVectorDB:
        # Extract parameters
        dbconfig = dict(config.vectordb)
        name = dbconfig.pop("connector_name")
        vdb_cls = ConnectorFactory.CONNECTORS.get(name)
        if not vdb_cls:
            raise ValueError(f"VECTORDB '{name}' is not supported.")

        dbconfig['embeddings'] = embeddings
        dbconfig['logger'] = logger

        return vdb_cls(**dbconfig)