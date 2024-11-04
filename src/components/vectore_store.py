from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import Coroutine, Generator, Union
from qdrant_client import QdrantClient, models
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
from langchain_core.documents.base import Document
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode


# https://python-client.qdrant.tech/qdrant_client.qdrant_client

class BaseVectorDd(metaclass=ABCMeta):
    """
    Abstract base class for a Vector Database Connector.
    This class defines the interface for a vector database connector.
    """
    @abstractmethod
    def async_add_documents(self, index_name, chunks, embeddings):
        """
        Abstract method for building an index in the vector database.

        Args:
            index_name (str): The name of the index to be built.
            chunks (list): The chunks of data to be indexed.
            embeddings (list): The embeddings of the data.
        """
        pass

    @abstractmethod
    async def async_sim_search(self, query: str, top_k: int = 5) -> list[Document]:
        """
        Abstract method for performing a similarity search in the vector database.

        Args:
            query (list): The query vector.
            top_k (int): The number of top similar vectors to return.

        Returns:
            list: The top_k similar vectors.
        """
        pass
    
    @abstractmethod
    def async_sim_search_with_score(self, query, top_k):
        """
        Abstract method for performing a similarity search in the vector database.

        Args:
            query (list): The query vector.
            top_k (int): The number of top similar vectors to return.

        Returns:
            list: The top_k similar vectors.
        """
        pass


class QdrantDB(BaseVectorDd):
    """
    Concrete class for a Qdrant Vector Database.
    This class implements the **`VectorDBConnector`** interface for a Qdrant database.
    """

    def __init__(
            self, 
            host, port, 
            embeddings: HuggingFaceBgeEmbeddings | HuggingFaceEmbeddings=None,
            collection_name: str = None, logger = None,
            hybrid_mode=False,
        ):
        """
        Initialize Qdrant_Connector.

        Args:
            host (str): The host of the Qdrant server.
            port (int): The port of the Qdrant server.
            collection_name (str): The name of the collection in the Qdrant database.
            embeddings (list): The embeddings.
        """
        self.collection_name = collection_name
        self.logger = logger
        self.embeddings: Union[HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings] = embeddings
        self.client = QdrantClient(
            port=port,
            host=host,
            prefer_grpc=False,
        )

        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25") if hybrid_mode else None
        self.retrieval_mode = RetrievalMode.HYBRID if hybrid_mode else RetrievalMode.DENSE
        
        if self.client.collection_exists(collection_name=collection_name):
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=collection_name,
                embedding=embeddings,
                sparse_embedding=sparse_embeddings,
                retrieval_mode=self.retrieval_mode,
            ) 
            self.logger.warning(f"The Collection named `{collection_name}` loaded.")
        else:
            self.vector_store = QdrantVectorStore.construct_instance(
                embedding=embeddings,
                sparse_embedding=sparse_embeddings,
                collection_name=collection_name,
                client_options={'port': port, 'host':host},
                retrieval_mode=self.retrieval_mode,
            )
            self.logger.info(f"As the collection `{collection_name}` is non-existant, it's created.")


    async def async_add_documents(self, batch_gen: Generator[list[Document], None, None]) -> None:
        """
        Add docs to the vectore store in a async mode.

        Args:
            chuncked_docs (list): The chunks of data to be indexed.
        """
        # https://github.com/langchain-ai/langchain/issues/15310

        async for item in batch_gen:
            s = await self.vector_store.aadd_documents(item)
            print(s)


    async def async_sim_search(self, query: str, top_k: int = 5):
        return await self.vector_store.asimilarity_search(
            query, k=top_k
        )

    async def async_sim_search_with_score(self, query: str, top_k: int = 5) -> list[tuple[Document, float]]:
        """
        Perform an async similarity search in the Qdrant database.

        Args:
            query (list): The query vector.
            top_k (int): The number of top similar vectors to return.

        Returns:
            list: The top_k similar vectors.
        """
        return self.vector_store.asimilarity_search_with_score(query=query, k=top_k)
    

    async def async_multy_query_sim_search(self, queries: list[str], top_k_per_queries: int = 5) -> list[Document]:
        """
        Perform a similarity search in the Qdrant database for multiple queries.

        This method takes a list of queries and performs a similarity search for each query.
        The results of all searches are combined into a set to remove duplicates, and then returned as a list.

        Args:
            queries (list[str]): The list of query vectors.
            top_k_per_queries (int): The number of top similar vectors to return for each query.

        Returns:
            list: The combined results of the similarity searches for all queries.
        """
        # documents = list(
        #     map(lambda q: self.vector_store.similarity_search(query=q, k=top_k_per_queries), queries)
        # )

        retrieved_chunks = {}
        for query in queries:
            retrieved = await self.async_sim_search(query=query, top_k=top_k_per_queries)
            for document in retrieved:
                retrieved_chunks[id(document)] = document
        return list(retrieved_chunks.values())

    async def async_multy_query_sim_search_with_scores(self, queries: list[str], top_k_per_queries: int = 5) -> list[tuple[Document, float]]:
        """
        Perform a similarity search in the Qdrant database for multiple queries.

        This method takes a list of queries and performs a similarity search for each query.
        The results of all searches are combined into a set to remove duplicates, and then returned as a list.

        Args:
            queries (list[str]): The list of query vectors.
            top_k_per_queries (int): The number of top similar vectors to return for each query.

        Returns:
            list: The combined results of the similarity searches for all queries.
        """
        retrieved_chunks = defaultdict(lambda: (None, float('-inf')))
        for query in queries:
            retrieved = await self.async_sim_search_with_score(query=query, k=top_k_per_queries)
            for document, score in retrieved:
                if score > retrieved_chunks[id(document)][1]:
                    retrieved_chunks[id(document)] = (document, score)
                        
        return list(retrieved_chunks.values())
    

CONNECTORS = {
    "qdrant": QdrantDB
}