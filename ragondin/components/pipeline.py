import copy
from enum import Enum
import sys
from pathlib import Path
from langchain_core.documents.base import Document
from openai import AsyncOpenAI
from loguru import logger

from .grader import Grader
from .indexer import ABCVectorDB
from .llm import LLM
from .reranker import Reranker
from .retriever import ABCRetriever, RetrieverFactory
from .utils import format_context, load_sys_template


class RAGMODE(Enum):
    SIMPLERAG = "SimpleRag"
    CHATBOTRAG = "ChatBotRag"


class RetrieverPipeline:
    def __init__(self, config, vectordb: ABCVectorDB, logger=None) -> None:
        self.config = config
        self.logger = logger

        # vectordb
        self.vectordb: ABCVectorDB = vectordb

        # retriever
        self.retriever: ABCRetriever = RetrieverFactory.create_retriever(
            config=config, logger=self.logger
        )

        # reranker
        self.reranker = None
        self.reranker_enabled = config.reranker["enable"]
        self.reranker_top_k = int(config.reranker["top_k"])
        if self.reranker_enabled:
            self.logger.debug(f"Reranker enabled: {self.reranker_enabled}")
            self.reranker = Reranker(self.logger, config)

        # grader
        self.grader: Grader = None
        self.grader_enabled = config.grader["enable"]
        if self.grader_enabled:
            self.grader = Grader(config, logger=self.logger)

    async def retrieve_docs(self, partition: list[str], query: str) -> list[Document]:
        docs = await self.retriever.retrieve(
            partition=partition, query=query, db=self.vectordb
        )
        self.logger.debug(f"{len(docs)} Documents retreived")

        if docs:
            # grade and filter out irrelevant docs
            if self.grader_enabled:
                docs = await self.grader.grade_docs(user_input=query, docs=docs)

            # rerank documents
            if self.reranker_enabled:
                docs = await self.reranker.rerank(
                    query, documents=docs, top_k=self.reranker_top_k
                )

            else:
                docs = docs[: self.reranker_top_k]

        return docs


class RagPipeline:
    def __init__(self, config, vectordb: ABCVectorDB, logger=None) -> None:
        self.config = config
        self.logger = self.set_logger(config) if logger is None else logger

        # retriever pipeline
        self.retriever_pipeline = RetrieverPipeline(
            config=config, vectordb=vectordb, logger=self.logger
        )

        self.prompts_dir = Path(config.paths.prompts_dir)
        # contextualizer prompt
        self.contextualizer_pmpt = load_sys_template(
            self.prompts_dir / config.prompt["contextualizer_pmpt"]
        )

        # rag sys prompt
        self.rag_sys_prompt: str = load_sys_template(
            self.prompts_dir / config.prompt["rag_sys_pmpt"]
        )

        self.rag_mode = config.rag["mode"]
        self.chat_history_depth = config.rag["chat_history_depth"]

        self.llm_client = LLM(config.llm, self.logger)
        self.vlm_client = LLM(config.vlm, self.logger)
        self.contextualizer = AsyncOpenAI(
            base_url=config.vlm["base_url"], api_key=config.vlm["api_key"]
        )

    async def generate_query(self, messages: list[dict]) -> str:
        match RAGMODE(self.rag_mode):
            case RAGMODE.SIMPLERAG:
                # For SimpleRag, we don't need to contextualize the query as the chat history is not taken into account
                last_msg = messages[-1]
                return last_msg["content"]

            case RAGMODE.CHATBOTRAG:
                # Contextualize the query based on the chat history
                chat_history = ""
                for m in messages:
                    chat_history += f"{m['role']}: {m['content']}\n"

                params = dict(self.config.llm_params)
                params.pop("max_retries")
                response = await self.contextualizer.chat.completions.create(
                    model=self.config.vlm["model"],
                    messages=[
                        {"role": "system", "content": self.contextualizer_pmpt},
                        {
                            "role": "user",
                            "content": f"Given the following chat, generate a query. \n{chat_history}\n",
                        },
                    ],
                    **params,
                )

                contextualized_query = response.choices[0].message.content
                return contextualized_query

    async def _prepare_for_chat_completion(self, partition: list[str], payload: dict):
        messages = payload["messages"]
        self.logger.info(f"Chat Length before {len(messages)}")
        messages = messages[-self.chat_history_depth :]  # limit history depth
        self.logger.info(f"Chat Length After {len(messages)}")

        # 1. get the query
        query = await self.generate_query(messages)
        self.logger.debug(f"Query: {query}")

        # 2. get docs
        docs = await self.retriever_pipeline.retrieve_docs(
            partition=partition, query=query
        )
        logger.info(f"{len(docs)} Documents retrieved")

        # 3. Format the retrieved docs
        context, sources = format_context(docs)

        # 4. prepare the output
        messages: list = copy.deepcopy(messages)

        # prepend the messages with the system prompt
        messages.insert(
            0,
            {
                "role": "system",
                "content": self.rag_sys_prompt.format(context=context),
            },
        )
        # messages.append({"role": "tool", "name": "retriever", "content": f"Here are the retrieved documents: {context}"})

        payload["messages"] = messages
        return payload, docs

    async def _prepare_for_completions(self, partition: list[str], payload: dict):
        prompt = payload["prompt"]

        # 1. get the query
        query = await self.generate_query(
            messages=[{"role": "user", "content": prompt}]
        )
        self.logger.debug(f"Query: {query}")

        # 2. get docs
        docs = await self.retriever_pipeline.retrieve_docs(
            partition=partition, query=query
        )

        # 3. Format the retrieved docs
        context, sources = format_context(docs)

        # 4. prepare the output
        prompt = (
            f"Given the context: \n{context}\n" if docs else ""
        ) + f"Complete the following prompt: {prompt}"
        payload["prompt"] = prompt

        return payload, docs

    async def completions(self, partition: list[str], payload: dict):
        payload, docs = await self._prepare_for_completions(
            partition=partition, payload=payload
        )
        llm_output = self.llm_client.completions(request=payload)
        return llm_output, docs

    async def chat_completion(self, partition: list[str], payload: dict):
        try:
            payload, docs = await self._prepare_for_chat_completion(
                partition=partition, payload=payload
            )

            llm_output = self.llm_client.chat_completion(request=payload)
            return llm_output, docs
        except Exception as e:
            raise e

    @staticmethod
    def set_logger(config):
        verbose = config.verbose
        if bool(verbose["verbose"]):
            level = verbose["level"]
        else:
            level = "ERROR"
        logger.remove()
        logger.add(sys.stderr, level=level)
        return logger
