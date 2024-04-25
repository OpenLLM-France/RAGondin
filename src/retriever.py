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
    """

    def __init__(self, params: dict, type_retriever: str = "similarity") -> None:
        """
        Constructs all the necessary attributes for the Retriever object.

        Parameters
        ----------
            params : dict
                The parameters for the retrieval method.
            reranker : Reranker, optional
                An instance of the Reranker class used for reranking the retrieved documents.
            type_retriever : str
                The type of retrieval method. It can be "similarity".
        """
        self.params = params
        self.type = type_retriever

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
        return retrieved_chunks_txt

    def retrieve_with_scores(self, question: str, db: Qdrant_Connector) -> list[tuple[str, float]]:
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
            retrieved_chunks = db.similarity_search_with_score(query=question, top_k=self.params["top_k"])
        else:
            raise ValueError(f"Invalid type. Choose from {TYPES}")
        retrieved_chunks_with_score = [tuple((chunk.page_content, score)) for chunk, score in retrieved_chunks]
        return retrieved_chunks_with_score


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

    def __init__(self, params: dict, llm: LLM, prompt_multi_queries: Prompt) -> None:
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
        """
        self.llm = llm
        self.prompt_multi_queries = prompt_multi_queries
        super().__init__(params)

    def retrieve_with_scores(self, question: str, db: Qdrant_Connector):
        multi_prompt = self.prompt_multi_queries.get_prompt(question=question, k_multi_queries=3)
        generated_questions = Prompt.generate_multi_query(self.llm, multi_prompt)

        if self.type == "similarity":
            retrieved_chunks = db.multy_query_similarity_search_with_scores(queries=generated_questions,
                                                                            top_k_per_queries=self.params["top_k"])
        else:
            raise ValueError(f"Invalid type. Choose from {TYPES}")
        retrieved_chunks_with_score = [tuple((chunk.page_content, score)) for chunk, score in retrieved_chunks]
        return retrieved_chunks_with_score

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
        return retrieved_chunks_txt


class HybridRetriever(Retriever):
    """
    The HybridRetriever class is a subclass of the Retriever class that retrieves relevant documents
    based on multiple retrieval methods. It combines the results from multiple retrievers, each with
    a specified weight, to produce a final list of retrieved documents.

    Attributes
    ----------
    retrievers : list[tuple[float, Retriever]]
        A list of tuples, each containing a weight and an instance of a Retriever subclass. The weight
        determines the influence of the retriever's results on the final list of retrieved documents.
    """

    def __init__(self, params: dict, retrievers: list[tuple[float, Retriever]]) -> None:
        """
        Constructs all the necessary attributes for the HybridRetriever object.

        Parameters
        ----------
        params : dict
            The parameters for the retrieval method.
        retrievers : list[tuple[float, Retriever]]
            A list of tuples, each containing a weight and an instance of a Retriever subclass.
        """
        super().__init__(params, type_retriever="hybrid")
        self.retrievers = retrievers

    def retrieve(self, question: str, db: Qdrant_Connector) -> list[str]:
        """
        Retrieves relevant documents based on the results of multiple retrievers. Each retriever's results
        are weighted according to the weight specified in the 'retrievers' attribute. The results are then
        combined and sorted by score to produce the final list of retrieved documents.

        Parameters
        ----------
        question : str
            The question to retrieve documents for.
        db : Qdrant_Connector
            The Qdrant_Connector instance to use for retrieving documents.

        Returns
        -------
        list[str]
            The list of retrieved documents, sorted by score.
        """
        retrieved_chunks_txt = {}
        for weight, retriever in self.retrievers:
            chunks_with_score = retriever.retrieve_with_scores(question, db)
            for chunk, score in chunks_with_score:
                if chunk in retrieved_chunks_txt:
                    retrieved_chunks_txt[chunk] += weight*score
                else:
                    retrieved_chunks_txt[chunk] = weight*score
        retrieved_chunks_txt = sorted(retrieved_chunks_txt.items(), key=lambda x: x[1], reverse=True)
        retrieved_chunks_txt = [chunk for chunk, weight in retrieved_chunks_txt]
        return retrieved_chunks_txt[:self.params["top_k"]]