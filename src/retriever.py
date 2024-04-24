# Import necessary modules and classes
from typing import Union
from src.llm import LLM
from src.prompt import Prompt
from src.reranker import Reranker
from src.vector_store import Qdrant_Connector

# Define the types of retrievers
TYPES = ["similarity"]

# Define the Retriever class
class Retriever:
    """
    The Retriever class is responsible for retrieving relevant documents based on the type of retrieval method specified.

    Attributes
    ----------
    type : str
        The type of retrieval method. It can be "similarity".
    params : dict
        The parameters for the retrieval method.
    reranker : Reranker, optional
        An instance of the Reranker class used for reranking the retrieved documents.
    """

    def __init__(self, params: dict, reranker: Reranker = None, type: str = "similarity") -> None:
        """
        Constructs all the necessary attributes for the Retriever object.

        Parameters
        ----------
            params : dict
                The parameters for the retrieval method.
            reranker : Reranker, optional
                An instance of the Reranker class used for reranking the retrieved documents.
            type : str
                The type of retrieval method. It can be "similarity".
        """
        self.params = params
        self.reranker = reranker
        self.type = type

    def retrieve(self, question: str, db: Qdrant_Connector) -> list[str]:
        """
        Retrieves relevant documents based on the type of retrieval method specified.

        Parameters
        ----------
            question : str
                The question to retrieve documents for.
            db : Qdrant_Connector
                The Qdrant_Connector instance to use for retrieving documents.

        Returns
        -------
            list[str]
                The list of retrieved documents.
        """
        if self.type == "similarity":
            retrieved_chunks = db.similarity_search(query=question, top_k=self.params["top_k"])
        else:
            raise ValueError(f"Invalid type. Choose from {TYPES}")
        retrieved_chunks_txt = [chunk.page_content for chunk in retrieved_chunks]
        if self.reranker is None:
            return retrieved_chunks_txt
        reranked_docs_txt = self.reranker.rerank(query=question, docs=retrieved_chunks_txt,
                                                 k=self.params['top_k_rerank'])
        return reranked_docs_txt

# Define the MultiQueryRetriever class
class MultiQueryRetriever(Retriever):
    """
    The MultiQueryRetriever class is a subclass of the Retriever class that retrieves relevant documents based on multiple queries.

    Attributes
    ----------
    llm : LLM
        An instance of the LLM class used for generating multiple queries.
    prompt_multi_queries : Prompt
        An instance of the Prompt class used for generating prompts for multiple queries.
    """

    def __init__(self, params: dict,llm: LLM, prompt_multi_queries: Prompt, reranker: Reranker = None) -> None:
        """
        Constructs all the necessary attributes for the MultiQueryRetriever object.

        Parameters
        ----------
            params : dict
                The parameters for the retrieval method.
            llm : LLM
                An instance of the LLM class used for generating multiple queries.
            prompt_multi_queries : Prompt
                An instance of the Prompt class used for generating prompts for multiple queries.
            reranker : Reranker, optional
                An instance of the Reranker class used for reranking the retrieved documents.
        """
        self.llm = llm
        self.prompt_multi_queries = prompt_multi_queries
        super().__init__(params, reranker)

    def retrieve(self, question: str, db: Qdrant_Connector) -> list[str]:
        """
        Retrieves relevant documents based on multiple queries.

        Parameters
        ----------
            question : str
                The question to retrieve documents for.
            db : Qdrant_Connector
                The Qdrant_Connector instance to use for retrieving documents.

        Returns
        -------
            list[str]
                The list of retrieved documents.
        """
        multi_prompt = self.prompt_multi_queries.get_prompt(question=question, k_multi_queries=3)
        generated_questions = Prompt.generate_multi_query(self.llm, multi_prompt)

        if self.type == "similarity":
            retrieved_chunks = db.multy_query_similarity_search(queries=generated_questions,
                                                            top_k_per_queries=self.params["top_k"])
        else:
            raise ValueError(f"Invalid type. Choose from {TYPES}")
        retrieved_chunks_txt = [chunk.page_content for chunk in retrieved_chunks]
        if self.reranker is None:
            return retrieved_chunks_txt
        reranked_docs_txt = self.reranker.rerank(query=question, docs=retrieved_chunks_txt,
                                                 k=self.params['top_k_rerank'])
        return reranked_docs_txt