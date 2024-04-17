from abc import ABCMeta, abstractmethod
from typing import Union

from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
from langchain_core.documents.base import Document


class VectorDB_Connector:
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
    def build_index(self, index_name, chunks, embeddings):
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


class Qdrant_Connector(VectorDB_Connector):
    """
    Concrete class for a Qdrant Vector Database Connector.
    This class implements the VectorDB_Connector interface for a Qdrant database.
    """

    def __init__(self, host, port, embeddings: Union[HuggingFaceBgeEmbeddings,HuggingFaceEmbeddings], collection_name: str = "my_documents"):
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
                url=self.url, prefer_grpc=False
            )
            self.db = Qdrant(client=self.client, embeddings=embeddings, collection_name=self.collection_name)

    def disconnect(self):
        """
        Disconnect from the Qdrant database.
        Note: Qdrant does not require explicit disconnection.
        """
        pass

    def build_index(self, chuncked_docs : list[Document]) -> None:
        """
        Build an index in the Qdrant database.

        Args:
            chuncked_docs (list): The chunks of data to be indexed.
        """
        try :
            self.db.from_documents(
                documents= chuncked_docs,
                embedding = self.embeddings,
                collection_name=self.collection_name
            )
        except:
            self.db = self.db.from_documents(
                documents= chuncked_docs,
                embedding = self.embeddings,
                collection_name=self.collection_name,
                location = ":memory:"
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

    def similarity_search_with_score(self, query: str, top_k: int = 5):
        """
        Perform a similarity search in the Qdrant database.

        Args:
            query (list): The query vector.
            top_k (int): The number of top similar vectors to return.

        Returns:
            list: The top_k similar vectors.
        """
        return self.db.similarity_search_with_score(query=query, k=top_k)

    def similarity_search(self, query: str, top_k: int = 5):
        """
        Perform a similarity search in the Qdrant database.

        Args:
            query (list): The query vector.
            top_k (int): The number of top similar vectors to return.

        Returns:
            list: The top_k similar vectors.
        """
        return self.db.similarity_search(query=query, k=top_k)