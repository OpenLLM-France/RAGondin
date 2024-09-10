from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import Union
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
from langchain_core.documents.base import Document
from langchain_qdrant import QdrantVectorStore
from langchain_community.vectorstores import Qdrant


import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

# https://python-client.qdrant.tech/qdrant_client.qdrant_client

class BaseVectorDdConnector(metaclass=ABCMeta):
    """
    Abstract base class for a Vector Database Connector.
    This class defines the interface for a vector database connector.
    """

    @abstractmethod
    def disconnect(self):
        """
        Abstract method for disconnecting from the vector database.
        """
        pass

    @abstractmethod
    def add_documents(self, index_name, chunks, embeddings):
        """
        Abstract method for building an index in the vector database.

        Args:
            index_name (str): The name of the index to be built.
            chunks (list): The chunks of data to be indexed.
            embeddings (list): The embeddings of the data.
        """
        pass

    @abstractmethod
    def insert_vector(self, vector, payload):
        """
        Abstract method for inserting a vector into the vector database.

        Args:
            vector (list): The vector to be inserted.
            payload (dict): The payload associated with the vector.
        """
        pass

    @abstractmethod
    def similarity_search_with_score(self, query, top_k):
        """
        Abstract method for performing a similarity search in the vector database.

        Args:
            query (list): The query vector.
            top_k (int): The number of top similar vectors to return.

        Returns:
            list: The top_k similar vectors.
        """
        pass


class Qdrant_Connector(BaseVectorDdConnector):
    """
    Concrete class for a Qdrant Vector Database Connector.
    This class implements the **`VectorDBConnector`** interface for a Qdrant database.
    """

    def __init__(
            self, host, port, 
            embeddings: HuggingFaceBgeEmbeddings | HuggingFaceEmbeddings=None,
            collection_name: str = "my_documents"
        ):
        """
        Initialize Qdrant_Connector.

        Args:
            host (str): The host of the Qdrant server.
            port (int): The port of the Qdrant server.
            collection_name (str): The name of the collection in the Qdrant database.
            embeddings (list): The embeddings of the data.
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name

        self.embeddings: Union[HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings] = embeddings
        
        self.client = QdrantClient(
            port=port,
            host=host, # if None, localhost will be used
            prefer_grpc=False,
            location=":memory:"
        )
        self.vector_store = None

        if self.client.collection_exists(collection_name=collection_name):
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=embeddings
            )  
            logger.info(f"Collection {self.collection_name} loaded.")
        

    def disconnect(self):
        """
        Disconnect from the Qdrant database.
        Note: Qdrant does not require explicit disconnection.
        """
        pass


    def add_documents(self, chuncked_docs: list[Document]) -> None:
        """
        Build an index in the Qdrant database.

        Args:
            chuncked_docs (list): The chunks of data to be indexed.
        """
        if self.vector_store is None:
            self.vector_store = QdrantVectorStore.from_documents(
                documents=chuncked_docs, 
                embedding=self.embeddings
            )
            logger.info(f"Collection {self.collection_name} created.")
        else:
            self.vector_store.add_documents(chuncked_docs)



    def insert_vector(self, vector, payload):
        """
        Insert a vector into the Qdrant database.

        Args:
            vector (list): The vector to be inserted.
            payload (dict): The payload associated with the vector.
        """
        # Implement the method to insert a vector into Qdrant
        pass

    def similarity_search_with_score(self, query: str, top_k: int = 5) -> list[tuple[Document, float]]:
        """
        Perform a similarity search in the Qdrant database.

        Args:
            query (list): The query vector.
            top_k (int): The number of top similar vectors to return.

        Returns:
            list: The top_k similar vectors.
        """
        return self.vector_store.similarity_search_with_score(query=query, k=top_k)

    def similarity_search(self, query: str, top_k: int = 5) -> list[Document]:
        """
        Perform a similarity search in the Qdrant database.

        Args:
            query (list): The query vector.
            top_k (int): The number of top similar vectors to return.

        Returns:
            list: The top_k similar vectors.
        """
        return self.vector_store.similarity_search(query=query, k=top_k)

    def multy_query_similarity_search(self, queries: list[str], top_k_per_queries: int = 5) -> list[Document]:
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
        retrieved_chunks = {}
        for query in queries:
            retrieved = self.vector_store.similarity_search(query=query, k=top_k_per_queries)
            for document in retrieved:
                retrieved_chunks[id(document)] = document
        
        return list(retrieved_chunks.values())

    def multy_query_similarity_search_with_scores(self, queries: list[str], top_k_per_queries: int = 5) -> list[tuple[Document, float]]:
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
            retrieved = self.vector_store.similarity_search_with_score(query=query, k=top_k_per_queries)
            for document, score in retrieved:
                if score > retrieved_chunks[id(document)][1]:
                    retrieved_chunks[id(document)] = (document, score)
                        
        return list(retrieved_chunks.values())
    



class Qdrant_ConnectorLegacy(BaseVectorDdConnector):
    """
    Concrete class for a Qdrant Vector Database Connector.
    This class implements the **`VectorDBConnector`** interface for a Qdrant database.
    """

    def __init__(
            self, host, port, 
            embeddings: HuggingFaceBgeEmbeddings | HuggingFaceEmbeddings=None,
            collection_name: str = "my_documents"
        ):
        """
        Initialize Qdrant_Connector.

        Args:
            host (str): The host of the Qdrant server.
            port (int): The port of the Qdrant server.
            collection_name (str): The name of the collection in the Qdrant database.
            embeddings (list): The embeddings of the data.
        """
        self.collection_name = collection_name
        self.embeddings: Union[HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings] = embeddings

        if host is None:
            self.db = Qdrant
        else:
            self.host = host
            self.port = port
            self.url = f"http://{self.host}:{self.port}"

            self.client = QdrantClient(
                url=self.url, 
                prefer_grpc=False
            )

            self.db = Qdrant(
                client=self.client, 
                embeddings=embeddings, 
                collection_name=self.collection_name
            )
  

    def disconnect(self):
        """
        Disconnect from the Qdrant database.
        Note: Qdrant does not require explicit disconnection.
        """
        pass

    def add_documents(self, chuncked_docs: list[Document]) -> None:
        """
        Build an index in the Qdrant database.

        Args:
            chuncked_docs (list): The chunks of data to be indexed.
        """
        try: # create a new collection with that name
            self.db.from_documents(
                chuncked_docs,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                url = self.url
            )
            print(4)
        except: # when the collection already exists
            self.db = self.db.from_documents(
                chuncked_docs,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                location=":memory:"
            )


    def insert_vector(self, vector, payload):
        """
        Insert a vector into the Qdrant database.

        Args:
            vector (list): The vector to be inserted.
            payload (dict): The payload associated with the vector.
        """
        # Implement the method to insert a vector into Qdrant
        pass

    def similarity_search_with_score(self, query: str, top_k: int = 5) -> list[tuple[Document, float]]:
        """
        Perform a similarity search in the Qdrant database.

        Args:
            query (list): The query vector.
            top_k (int): The number of top similar vectors to return.

        Returns:
            list: The top_k similar vectors.
        """
        return self.vector_store.similarity_search_with_score(query=query, k=top_k)

    def similarity_search(self, query: str, top_k: int = 5) -> list[Document]:
        """
        Perform a similarity search in the Qdrant database.

        Args:
            query (list): The query vector.
            top_k (int): The number of top similar vectors to return.

        Returns:
            list: The top_k similar vectors.
        """
        return self.vector_store.similarity_search(query=query, k=top_k)

    def multy_query_similarity_search(self, queries: list[str], top_k_per_queries: int = 5) -> list[Document]:
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
        retrieved_chunks = {}
        for query in queries:
            retrieved = self.vector_store.similarity_search(query=query, k=top_k_per_queries)
            for document in retrieved:
                retrieved_chunks[id(document)] = document
        return list(retrieved_chunks.values())

    def multy_query_similarity_search_with_scores(self, queries: list[str], top_k_per_queries: int = 5) -> list[tuple[Document, float]]:
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
            retrieved = self.vector_store.similarity_search_with_score(query=query, k=top_k_per_queries)
            for document, score in retrieved:
                if score > retrieved_chunks[id(document)][1]:
                    retrieved_chunks[id(document)] = (document, score)
                        
        return list(retrieved_chunks.values())


CONNECTORS = {
    "qdrant": Qdrant_Connector
}