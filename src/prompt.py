"""Module for building prompts for LLMs"""

import asyncio
from enum import Enum
from langchain.prompts import PromptTemplate
from .llm import LLM
from pathlib import Path
from pathlib import Path


class TemplateType(str, Enum):
    BASIC = "basic" 
    MULTI_QUERY = "multiQuery"


def get_msgs(file_path: Path):
    with open(file_path, mode="r") as f:
        txt = f.read()
        sys_msg, user_msg = txt.split("&&&\n")
        return sys_msg, user_msg


# PROMPT_FILE

class Prompt:
    """Class for generating prompts to provide context for LLMs."""

    def __init__(self, type_template: str = 'basic') -> None:
        """
        Initialize Prompt.

        Args:
            template (PromptTemplate, optional): Template for generating prompts.
                If None, use default template.
            type_template (str): Type of template to use. It can be either "basic" or "multi_query".
        """

        dir_path = Path(__file__).parent
        if type_template == TemplateType.BASIC:
            msg_templates = get_msgs(dir_path / "prompts/basic_prompt_template.txt")

        elif type_template == TemplateType.MULTI_QUERY:
            msg_templates = get_msgs(dir_path / "prompts/multi_query_prompt_template.txt")
        else:
            raise ValueError(f"This `template_type` isn't supported")
        
        self.typed_template = type_template
        self.msg_templates = msg_templates

    def get_prompt(self, question: str, docs: list[str] = None, k_multi_queries: int = 3) -> dict:
        """
        Generate prompt from documents and question.

        Args:
            docs (list[str]): List of document strings.
            question (str): Question string in the case of type_template is "basic".
            k_multi_queries (int): Number of multi queries to generate in the case of type_template is "multi_query".
        Returns:
            str: Final generated prompt.
        """
        if self.typed_template == TemplateType.BASIC:
            context = self._build_context(docs)
            sys_template, user_template = self.msg_templates

            sys_prompt = PromptTemplate.from_template(sys_template).format(context=context)
            user_prompt = PromptTemplate.from_template(user_template).format(question=question)
            print(user_prompt)
            return {"system":sys_prompt, "user":user_prompt}
        
        if self.typed_template == TemplateType.MULTI_QUERY:
            sys_template, user_template = self.msg_templates
            sys_prompt = PromptTemplate.from_template(sys_template)\
                .format(k=str(k_multi_queries))
            user_prompt = PromptTemplate.from_template(user_template).format(question=question)

            return {"system":sys_prompt, "user":user_prompt}
            

    def _build_context(self, docs: list[str]) -> str:
        """Build context string from list of documents."""
        context = "Extracted documents:\n"
        for i, doc in enumerate(docs):
            context += f"Document {i}:::\n{doc}\n"
        return context

 
    @staticmethod
    def generate_multi_query(llm: LLM, msg_prompts: dict) -> list[str]:
        questions = llm.run(msg_prompts)
        # q_l = questions[-1].split("[SEP]")
        print(questions)
        return questions
    

BASIC_PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Vous devez répondre à la question en français sans aucune faute de grammaire ni d'orthographe en utilisant strictement le contexte suivant pour vous aider: 
{context}<|eot_id|> 
<|start_header_id|>user<|end_header_id|>

{question}<|eot_id|> 
<|start_header_id|>assistant<|end_header_id|>
"""

DEFAULT_TEMPLATE = PromptTemplate(input_variables=["context", "question"], template=BASIC_PROMPT_TEMPLATE)

BASIC_MULTI_QUERY_PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an AI language model assistant in french. Your task is 
to generate {k} different versions of the given user 
question to retrieve relevant documents from a vector database.
By generating multiple perspectives on the user question, 
your goal is to help the user overcome some of the limitations 
of distance-based similarity search. Provide these alternative 
questions separated by [SEP].

You MUST absolutely generate {k} questions, in french, in the same line, separated only by [SEP]. Here is the format to follow
####Generated Questions: "question 1"[SEP]"question 2"[SEP]"question 3"[SEP]..."question {k}"<|eot_id|>
<|start_header_id|>user<|end_header_id|>

#### Original question: {question}
####Generated Questions: """

DEFAULT_MULTI_QUERY_TEMPLATE = PromptTemplate(input_variables=["k", "question"],
                                              template=BASIC_MULTI_QUERY_PROMPT_TEMPLATE)
