"""Module for building prompts for LLMs"""

from langchain.prompts import PromptTemplate

from src.llm import LLM

TYPE = ["basic", "multi_query"]


class Prompt:
    """Class for generating prompts to provide context for LLMs."""

    def __init__(self, template: PromptTemplate = None, type_template: str = 'basic') -> None:
        """
        Initialize Prompt.

        Args:
            template (PromptTemplate, optional): Template for generating prompts.
                If None, use default template.
            type_template (str): Type of template to use. It can be either "basic" or "multi_query".
        """
        self.typed_template = type_template
        if type_template == "basic":
            self.template = template or DEFAULT_TEMPLATE
        if type_template == "multi_query":
            self.template = template or DEFAULT_MULTI_QUERY_TEMPLATE

    def get_prompt(self, question: str, docs: list[str] = None, k_multi_queries: int = 3) -> str:
        """
        Generate prompt from documents and question.

        Args:
            docs (list[str]): List of document strings.
            question (str): Question string in the case of type_template is "basic".
            k_multi_queries (int): Number of multi queries to generate in the case of type_template is "multi_query".
        Returns:
            str: Final generated prompt.
        """
        if self.typed_template == "basic":
            context = self._build_context(docs)
            return self.template.format(context=context, question=question)
        if self.typed_template == "multi_query":
            return self.create_multi_query_prompt(question, template=self.template, k=k_multi_queries)

    def _build_context(self, docs: list[str]) -> str:
        """Build context string from list of documents."""
        context = "Extracted documents:\n"
        for i, doc in enumerate(docs):
            context += f"Document {i}:::\n{doc}\n"
        return context

    @staticmethod
    def create_multi_query_prompt(question: str, template: PromptTemplate = None, k: int = 3) -> str:
        multi_query_template = template or DEFAULT_MULTI_QUERY_TEMPLATE
        return multi_query_template.format(k=str(k), question=question)

    @staticmethod
    def generate_multi_query(llm: LLM, prompt: str) -> list[str]:
        return llm.run(prompt).split("####Questions:")[1].split("[SEP]")


BASIC_PROMPT_TEMPLATE = """
<|system|>
Answer the question in french only using the following french context to help:

{context}

</s>
<|user|>
{question}
</s>
<|assistant|>

 """

DEFAULT_TEMPLATE = PromptTemplate(input_variables=["context", "question"], template=BASIC_PROMPT_TEMPLATE)

BASIC_MULTI_QUERY_PROMPT_TEMPLATE = """You are an AI language model assistant in french. Your task is 
    to generate {k} different versions of the given user 
    question to retrieve relevant documents from a vector  database. 
    By generating multiple perspectives on the user question, 
    your goal is to help the user overcome some of the limitations 
    of distance-based similarity search. Provide these alternative 
    questions separated by [SEP].
    Be sure to include [SEP] between each question, without anything else and those questions must be in french.
    #### Original question: {question}
    ####Questions:
    """

DEFAULT_MULTI_QUERY_TEMPLATE = PromptTemplate(input_variables=["k", "question"],
                                              template=BASIC_MULTI_QUERY_PROMPT_TEMPLATE)
