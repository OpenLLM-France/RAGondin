"""Module for building prompts for LLMs"""

from langchain.prompts import PromptTemplate


class Prompt:
    """Class for generating prompts to provide context for LLMs."""

    def __init__(self, template: PromptTemplate = None) -> None:
        """
        Initialize Prompt.

        Args:
            template (PromptTemplate, optional): Template for generating prompts.
                If None, use default template.
        """
        self.template = template or DEFAULT_TEMPLATE

    def get_prompt(self, docs: list[str], question: str) -> str:
        """
        Generate prompt from documents and question.

        Args:
            docs (list[str]): List of document strings.
            question (str): Question string.

        Returns:
            str: Final generated prompt.
        """
        context = self._build_context(docs)
        return self.template.format(context=context, question=question)

    def _build_context(self, docs: list[str]) -> str:
        """Build context string from list of documents."""
        context = "Extracted documents:\n"
        for i, doc in enumerate(docs):
            context += f"Document {i}:::\n{doc}\n"
        return context



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

DEFAULT_TEMPLATE = PromptTemplate(input_variables=["context", "question"],template=BASIC_PROMPT_TEMPLATE)





