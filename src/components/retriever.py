# Import necessary modules and classes
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import Union
from .llm import LLM
from .prompt import Prompt
from .reranker import Reranker
from .vector_store import Qdrant_Connector, BaseVectorDdConnector
from enum import Enum

CRITERIAS = ["similarity"]

class BaseRetriever(metaclass=ABCMeta):
    """Abstract class for the base retreiver.
    """
    @abstractmethod
    def __init__(self, criteria: str = "similarity", top_k: int = 6, **extra_args) -> None:
        pass

    @abstractmethod
    def retrieve(self, question: str, db: Qdrant_Connector) -> list[str]:
        pass

    @abstractmethod
    def retrieve_with_scores(self, question: str, db: Qdrant_Connector) -> list[tuple[str, float]]:
        pass


# Define the Retriever class
class SingleRetriever(BaseRetriever):
    def __init__(self, criteria: str = "similarity", top_k: int = 6, **extra_args) -> None:
        """Constructs all the necessary attributes for the Retriever object.

        Args:
            criteria (str, optional): Retrieval criteria. Defaults to "similarity".
            top_k (int, optional): top_k most similar documents to retreive. Defaults to 6.
        """
        self.top_k = top_k
        if criteria not in CRITERIAS:
            ValueError(f"Invalid type. Choose from {CRITERIAS}")

        self.criteria = criteria

    def retrieve(self, question: str, db: BaseVectorDdConnector | Qdrant_Connector) -> list[str]:
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
        if self.criteria == "similarity":
            retrieved_chunks = db.similarity_search(
                query=question, 
                top_k=self.top_k
            )
        else:
            raise ValueError(f"Invalid type. Choose from {CRITERIAS}")
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
        if self.criteria == "similarity":
            retrieved_chunks = db.similarity_search_with_score(query=question, top_k=self.params["top_k"])
        else:
            raise ValueError(f"Invalid type. Choose from criteria from {CRITERIAS}")
        retrieved_chunks_with_score = [tuple((chunk.page_content, score)) for chunk, score in retrieved_chunks]
        return retrieved_chunks_with_score


# Define the MultiQueryRetriever class
class MultiQueryRetriever(SingleRetriever):
    def __init__(
            self, 
            criteria: str = "similarity", top_k: int = 6,
            **extra_args
            ) -> None:
        """
        The MultiQueryRetriever class is a subclass of the Retriever class that retrieves relevant documents based on multiple queries.
        Given a query, multiple similar are generated with an llm. Retreival is done with each one them and finally a subset is chosen.

        Attributes
        ----------
        Args:
            criteria (str, optional): Retrieval criteria. Defaults to "similarity".
            top_k (int, optional): top_k most similar documents to retreive. Defaults to 6.
            extra_args (dict): contains additionals arguments for this type of retreiver.
        """
        super().__init__(criteria, top_k)
        
        try:
            prompt_multi_queries = extra_args.get("prompt_multi_queries")
            if not isinstance(prompt_multi_queries, Prompt):
                raise TypeError(f"`prompt_multi_queries` should be of type {Prompt}")

            llm = extra_args.get("llm")
            if not isinstance(llm, LLM):
                raise TypeError(f"`llm` should be of type {LLM}")
        
            k_multi_queries = extra_args.get("k_multi_queries")
            if not isinstance(k_multi_queries, int):
                raise TypeError(f"`k_multi_queries` should be of type {int}")
            
            self.prompt_multi_queries: Prompt = prompt_multi_queries
            self.llm: LLM = llm
            self.k_multi_queries: int = k_multi_queries

        except Exception as e:
            raise KeyError(f"An Error has occured: {e}")


    def retrieve_with_scores(self, question: str, db: BaseVectorDdConnector | Qdrant_Connector):
        msg_prompts = self.prompt_multi_queries.get_prompt(
            question=question, 
            k_multi_queries=self.k_multi_queries
        )
        # generate similar questions
        generated_questions = Prompt.generate_multi_query(self.llm, msg_prompts=msg_prompts)

        if self.criteria == "similarity":
            retrieved_chunks = db.multy_query_similarity_search_with_scores(queries=generated_questions, top_k_per_queries=self.top_k)
        else:
            raise ValueError(f"Invalid, {self.criteria} should be from {CRITERIAS}")
        
        retrieved_chunks_with_score = [(chunk.page_content, score) for chunk, score in retrieved_chunks ]
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
        multi_prompt_dict = self.prompt_multi_queries.get_prompt(question=question, k_multi_queries=self.k_multi_queries)
        generated_questions = Prompt.generate_multi_query(self.llm, msg_prompts=multi_prompt_dict)
        if self.criteria == "similarity":
            retrieved_chunks = db.multy_query_similarity_search(
                queries=generated_questions, 
                top_k_per_queries=self.top_k
            )
        else:
            raise ValueError(f"Invalid type. Choose from {CRITERIAS}")
        retrieved_chunks_txt = [chunk.page_content for chunk in retrieved_chunks]
        return retrieved_chunks_txt


class HybridRetriever(SingleRetriever):
    #TODO: This class has not been review yet in order to make it coherent with other retreivers
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

    def __init__(self, params: dict, retrievers: list[tuple[float, SingleRetriever]]) -> None:
        """
        Constructs all the necessary attributes for the HybridRetriever object.

        Parameters
        ----------
        params : dict
            The parameters for the retrieval method.
        retrievers : list[tuple[float, Retriever]]
            A list of tuples, each containing a weight and an instance of a Retriever subclass.
        """
        super().__init__(params, criteria="hybrid")
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
        retrieved_chunks_txt = defaultdict(float)
        for weight, retriever in self.retrievers:
            chunks_with_score = retriever.retrieve_with_scores(question, db)
            for chunk, score in chunks_with_score:
                retrieved_chunks_txt[chunk] += weight * score
                
        retrieved_chunks_txt = sorted(retrieved_chunks_txt.items(), key=lambda x: x[1], reverse=True)
        retrieved_chunks_txt = [chunk for chunk, weight in retrieved_chunks_txt]
        return retrieved_chunks_txt[:self.params["top_k"]]



RETREIEVERS = {
    "single": SingleRetriever,
    "multiQuery": MultiQueryRetriever
}

def get_retreiver_cls(retreiver_type: str) -> BaseRetriever:
    # Retrieve the chunker class from the map and instantiate it
    retreiver = RETREIEVERS.get(retreiver_type, None)
    if retreiver is None:
        raise ValueError(f"Unknown retreiver type: {retreiver_type}")
    return retreiver