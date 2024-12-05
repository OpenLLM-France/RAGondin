from typing import Literal
from pydantic import BaseModel, Field
from llama_index.llms.langchain import LangChainLLM
from langchain_openai import ChatOpenAI
from langchain_core.documents.base import Document
from llama_index.core.llms import ChatMessage, MessageRole
import asyncio

from .llm import LLM

sys_prompt = """

You are an expert at judging documents' relevancy with respect to a given user query/input.
* For CVs pay attention to the keys before judging them as relevant.
"""

class GradeDocuments(BaseModel):
    """Evaluates document's relevancy using a binary scoring system."""

    relevance_score: Literal['yes', 'maybe', 'no'] = Field(
        description="Document relevance classification:\n"
                    "- yes: The document is highly relevant to the query's main concepts\n"
                    "- maybe: Somewhat relevant with shared keywords"
                    "- no: Document has minimal or no meaningful connection"
    )


# Evaluation Criteria:
# - 'yes': The document directly addresses the key concepts, provides substantive information, and closely aligns with the query's primary intent
# - 'maybe': The document contains partial or tangential information related to the query, with some meaningful connections but not comprehensive coverage
# - 'no': The document has minimal or no meaningful relationship to the query's core objectives

class Grader:
    def __init__(self, config, logger=None) -> None:
        lc_llm = LLM(config=config, logger=None).client
        llm = LangChainLLM(
            llm=lc_llm
        )

        # structured llm
        self.sllm = llm.as_structured_llm(
            output_cls=GradeDocuments
        )
        self.logger = logger

    
    async def grade(self, user_input: str, docs: list[Document], n_workers=8):
        n_workers = min(n_workers, len(docs))
        sem = asyncio.Semaphore(n_workers)
        async def eval_document(user_input, doc: Document):
            async with sem:
                query_tmpl = f"""User input: {user_input}\n Retrieved Document: {doc.page_content}"""
                try:
                    messages = [
                        ChatMessage(role=MessageRole.SYSTEM, content=sys_prompt),
                        ChatMessage(role=MessageRole.USER, content=query_tmpl)
                    ]
                    res = await self.sllm.achat(messages=messages)

                    return res.raw.relevance_score
                except Exception as e:
                    self.logger.info(f"An Exception occured: {e}")
                    return 'yes'
        
        tasks = [eval_document(user_input=user_input, doc=d) for d in docs]
        grades = await asyncio.gather(*tasks)

        # filtering
        relevants_docs = list(filter(lambda x: x[1] != 'no', zip(docs, grades)))
        return [doc for doc, grade in relevants_docs] 