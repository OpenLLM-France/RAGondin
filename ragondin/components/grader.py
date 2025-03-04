from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents.base import Document
import asyncio
from .llm import LLM

sys_prompt = """You are an expert at carefully judging documents' relevancy with respect to a user query/input."""
class DocumentGrade(BaseModel):
    """Evaluates document's relevancy with respect to a user query."""
    relevance_score: Literal['highly_relevant', 'somewhat_relevant', 'irrelevant'] = Field(
        description="Document relevance classification:\n")


class Grader:
    def __init__(self, config, logger=None):
        llm: ChatOpenAI = LLM(config=config, logger=logger).client
        # structured llm
        self.sllm = llm.with_structured_output(DocumentGrade)
        self.logger = logger

    async def _grade_doc(self, user_input, doc: Document, sem: asyncio.Semaphore):
        async with sem:
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
                self.logger.info(f"An Exception occured. Couldn't grade this document: {e}")
    
    async def grade_docs(self, user_input: str, docs: list[Document], batch_size=6):
        batch_size = min(batch_size, len(docs))
        self.logger.info(f"{len(docs)} documents to assess relevancy.")

        sem = asyncio.Semaphore(batch_size)
        tasks = [
            self._grade_doc(user_input=user_input, doc=d, sem=sem) for d in docs
        ]
        grades: list[DocumentGrade] = await asyncio.gather(*tasks)

        # Filter out irrelevant documents
        relevant_docs = list(
            filter(lambda doc_grade: doc_grade[1] != 'irrelevant', zip(docs, grades))
        )
        relevant_docs = [doc for doc, _ in relevant_docs]
        return relevant_docs





# from typing import Literal
# from pydantic import BaseModel, Field
# from .llm import LLM
# from llama_index.core.llms import ChatMessage, MessageRole
# from llama_index.llms.langchain import LangChainLLM

# class Grader(BaseModel):
#     """Evaluates document's relevancy using a binary scoring system."""

#     relevance_score: Literal['yes', 'maybe', 'no'] = Field(
#         description="Document relevance classification:\n"
#                     "- yes: The document is highly relevant to the query's main concepts\n"
#                     "- maybe: Somewhat relevant with shared keywords"
#                     "- no: Document has minimal or no meaningful connection"
#     )


# class Grader:
#     def __init__(self, config, logger=None) -> None:
#         lc_llm = LLM(config=config, logger=None).client
#         llm = LangChainLLM(
#             llm=lc_llm
#         )
#         # structured llm
#         self.sllm = llm.as_structured_llm(
#             output_cls=GradeDocuments
#         )
#         self.logger = logger

#     async def _eval_document(self, user_input, doc: Document, sem: asyncio.Semaphore):
#         async with sem:
#             query_tmpl = f"""User input: {user_input}\n Retrieved Document: {doc.page_content}"""
#             try:
#                 messages = [
#                     ChatMessage(role=MessageRole.SYSTEM, content=sys_prompt),
#                     ChatMessage(role=MessageRole.USER, content=query_tmpl)
#                 ]
#                 res = await self.sllm.achat(messages=messages)

#                 return res.raw.relevance_score
#             except Exception as e:
#                 self.logger.info(f"An Exception occured: {e}")
#                 return 'yes'
    
#     async def grade(self, user_input: str, docs: list[Document], n_workers=6):
#         n_workers = min(n_workers, len(docs))
#         self.logger.info(f"{len(docs)} documents to grade.")
#         sem = asyncio.Semaphore(n_workers)
#         tasks = [
#             self._eval_document(user_input=user_input, doc=d, sem=sem) for d in docs
#         ]
#         grades = await asyncio.gather(*tasks)

#         # Filter relevant documents
#         relevant_docs = [
#             doc for doc, grade in zip(docs, grades) if grade != 'no'
#         ]
#         return relevant_docs