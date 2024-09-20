# Import necessary modules and classes
from abc import ABCMeta, abstractmethod
from pathlib import Path
from .llm import LLM2
# from .prompt import generate_multi_query, MultiQueryPrompt
from .vector_store import Qdrant_Connector, BaseVectorDdConnector
from loguru import logger
# from .utils import load_prompt
from langchain_core.prompts import ChatPromptTemplate
from .prompt import load_sys_template
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads


dir_path = Path(__file__).parent
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
            logger.info("Get relevant documents.")
        else:
            raise ValueError(f"Invalid type. Choose from {CRITERIAS}")
        retrieved_chunks_txt = [chunk for chunk in retrieved_chunks]
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
            llm: ChatOpenAI = extra_args.get("llm")
            if not isinstance(llm, ChatOpenAI):
                raise TypeError(f"`llm` should be of type {LLM2}")
        
            k_queries = extra_args.get("k_queries")
            if not isinstance(k_queries, int):
                raise TypeError(f"`k_queries` should be of type {int}")
            
            multi_queries_template = load_sys_template(
                dir_path / "prompts/multi_query_prompt_template.txt"
            )
            self.k_queries = k_queries
            prompt_multi_queries: ChatPromptTemplate = ChatPromptTemplate.from_template(
                multi_queries_template
            )
            self.generate_queries = (
                prompt_multi_queries 
                | llm
                | StrOutputParser() 
                | (lambda x: x.split("[SEP]"))
            )

        except Exception as e:
            raise KeyError(f"An Error has occured: {e}")


    def retrieve_with_scores(self, question: str, db: BaseVectorDdConnector | Qdrant_Connector):
        logger.info("generate different perspectives of the question ...")
        generated_questions = self.generate_queries.invoke(
            {
                "question": question, 
                "k_queries": self.k_queries
            }
        )
        
        if self.criteria == "similarity":
            retrieved_chunks = db.multy_query_similarity_search_with_scores(queries=generated_questions, top_k_per_queries=self.top_k)
        else:
            raise ValueError(f"Invalid, {self.criteria} should be from {CRITERIAS}")
        
        retrieved_chunks_with_score = [(chunk, score) for chunk, score in retrieved_chunks ]
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

        # generate different perspectives of the question
        generated_questions = self.generate_queries.invoke(
            {"question": question, "k_queries": self.k_queries}
        )

        if self.criteria == "similarity":
            retrieved_chunks = db.multy_query_similarity_search(
                queries=generated_questions, 
                top_k_per_queries=self.top_k
            )
        else:
            raise ValueError(f"Invalid type. Choose from {CRITERIAS}")
        return retrieved_chunks




class HyDeRetreiver(SingleRetriever):
    def __init__(self, criteria: str = "similarity", top_k: int = 6, **extra_args) -> None:
        super().__init__(criteria, top_k, **extra_args)
        try:
            llm = extra_args.get("llm")
            if not isinstance(llm, ChatOpenAI):
                raise TypeError(f"`llm` should be of type {ChatOpenAI}")
            
            self.llm: ChatOpenAI = llm

            hyde_template = load_sys_template(
                dir_path / "prompts/hyde.txt"
            )

            hyde_prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(
                hyde_template
            )
            self.generate_hyde = (
                hyde_prompt 
                | llm
                | StrOutputParser() 
            )

        except Exception as e:
            raise ArithmeticError(f"An error occured: {e}")
        

    def get_hyde(self, question: str):
        generated_document = (self.generate_hyde
                              .invoke({"question": question})
                            )
        return generated_document

    def retrieve(self, question: str, db: BaseVectorDdConnector | Qdrant_Connector) -> list[str]:
        hyde = self.get_hyde(question)
        logger.info("generate hyde document...")
        docs = super().retrieve(hyde, db)
        return docs # + super().retrieve(question, db) # hyde + single retreiver for stronger results
    
    def retrieve_with_scores(self, question: str, db: Qdrant_Connector) -> list[tuple[str, float]]:
        hyde = self.get_hyde(question)
        return super().retrieve_with_scores(hyde, db)



RETREIEVERS = {
    "single": SingleRetriever,
    "multiQuery": MultiQueryRetriever,
    "hyde": HyDeRetreiver,
}

def get_retreiver_cls(retreiver_type: str) -> BaseRetriever:
    # Retrieve the chunker class from the map and instantiate it
    retreiver = RETREIEVERS.get(retreiver_type, None)
    if retreiver is None:
        raise ValueError(f"Unknown retreiver type: {retreiver_type}")
    return retreiver


def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]
