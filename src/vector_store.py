from abc import ABCMeta, abstractmethod
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant


class VectorDB_Connector:
    @abstractmethod
    def disconnect(self):
        pass

    @abstractmethod
    def build_index(self, index_name, chunks, embeddings):
        pass

    @abstractmethod
    def insert_vector(self, vector, payload):
        pass

    @abstractmethod
    def similarity_search_with_score(self, query, top_k):
        pass


class Qdrant_Connector(VectorDB_Connector):
    def __init__(self, host, port, collection_name, embeddings):
        self.host = host
        self.port = port
        self.url = f"http://{self.host}:{self.port}"
        self.collection_name = collection_name
        print(self.collection_name)
        self.embeddings = embeddings
        self.client = QdrantClient(
                    url=self.url, prefer_grpc=False
        )
        self.db = Qdrant(client=self.client, embeddings=embeddings, collection_name=self.collection_name)


    def disconnect(self):
        # Qdrant does not require explicit disconnection
        pass

    def build_index(self, chunks):
        self.db.from_documents(
            chunks,
            self.embeddings,
            collection_name=self.collection_name
        )

    def insert_vector(self, vector, payload):
        # Implement the method to insert a vector into Qdrant
        pass

    def similarity_search_with_score(self, query, top_k):
        docs = self.db.similarity_search_with_score(query=query, k=top_k)
        return docs

