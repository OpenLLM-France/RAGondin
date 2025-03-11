from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents.base import Document
import asyncio
from .llm import LLM
from .utils import llmSemaphore

sys_prompt = """You are an expert at carefully judging documents' relevancy with respect to user's query."""
class DocumentGrade(BaseModel):
    """Evaluates document's relevancy with respect to a user query.
    """
    relevance_score: Literal['highly_relevant', 'somewhat_relevant', 'irrelevant'] = Field(
        description="Document relevance classification:\n")


class Grader:
    def __init__(self, config, logger=None):
        llm: ChatOpenAI = LLM(config=config, logger=logger).client
        # structured llm
        self.sllm = llm.with_structured_output(DocumentGrade)
        self.logger = logger

    async def _grade_doc(self, user_input, doc: Document, semaphore=llmSemaphore):
        async with semaphore:
            try:
                query_template = ("""User query: {user_input}\n""" 
                    """Retrieved Document: {content}"""
                )

                template = ChatPromptTemplate.from_messages(
                    [
                        ('system', sys_prompt),
                        ('user', query_template)
                    ]
                )
                # Create a PromptValue from the template
                prompt_value = template.invoke(
                    {
                        "user_input": user_input,
                        "content": doc.page_content
                    }
                )
                result: DocumentGrade = await self.sllm.ainvoke(prompt_value)
                
                return result.relevance_score
            except Exception as e:
                self.logger.debug(f"An Exception occured. Couldn't grade this document: {e}")
    
    async def grade_docs(self, user_input: str, docs: list[Document], batch_size=6):
        batch_size = min(batch_size, len(docs))
        self.logger.debug(f"{len(docs)} documents to assess relevancy.")

        tasks = [
            self._grade_doc(user_input=user_input, doc=d) for d in docs
        ]
        grades: list[DocumentGrade] = await asyncio.gather(*tasks)

        # Filter out irrelevant documents
        relevant_docs = list(
            filter(lambda doc_grade: doc_grade[1] != 'irrelevant', zip(docs, grades))
        )
        relevant_docs = [doc for doc, _ in relevant_docs]
        self.logger.debug(f"{len(relevant_docs)} relevant documents.")
        return relevant_docs