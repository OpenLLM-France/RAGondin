from typing import Literal
from pydantic import BaseModel, Field
from llama_index.llms.langchain import LangChainLLM
from langchain_openai import ChatOpenAI
from langchain_core.documents.base import Document
from llama_index.core.llms import ChatMessage, MessageRole
import asyncio


sys_prompt = """You are an expert at judging documents' relevancy with respect to a given user query/input."""


class GradeDocuments(BaseModel):
    """Evaluates document relevance using a binary scoring system."""

    relevance_score: Literal['yes', 'no'] = Field(
        description="Document relevance classification:\n"
                    "- yes: Document strongly matches the query's main concepts\n"
                    "- no: Document has minimal or no meaningful connection"
    )


class Grader:
    def __init__(self, config, logger=None) -> None:
        settings = {
            'model': config.llm["name"],
            'base_url': config.llm["base_url"],
            'api_key': config.llm['api_key'],
            'timeout': 60,
            'temperature': config.llm["temperature"],
            'max_tokens': config.llm["max_tokens"], 
        }
        # langchain llm
        lc_llm = ChatOpenAI(**settings) # TODO: consider retry strategy
        # \
        #     .with_retry(
        #         retry_if_exception_type=(Exception,), 
        #         wait_exponential_jitter=False, stop_after_attempt=3
        # )

        llm = LangChainLLM(
            llm=lc_llm
        )

        # structured llm
        self.sllm = llm.as_structured_llm(
            output_cls=GradeDocuments
        )
        self.logger = logger

    
    async def grade(self, user_input: str, docs: list[Document], n_workers=6):
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
        relevants_docs = list(filter(lambda x: x[1] == 'yes', zip(docs, grades)))
        print(1)
        return [doc for doc, grade in relevants_docs] 