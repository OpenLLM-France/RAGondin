import gc
import sys
from collections import deque
from pathlib import Path

import torch
from langchain_core.documents.base import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from loguru import logger

from .grader import Grader
from .indexer import ABCVectorDB
from .llm import LLM
from .reranker import Reranker
from .retriever import ABCRetriever, RetrieverFactory
from .utils import format_context, load_sys_template


class RagPipeline:
    def __init__(self, config, vectordb: ABCVectorDB, logger=None) -> None:
        self.config = config
        self.logger = self.set_logger(config) if logger is None else logger
        self.vectordb: ABCVectorDB = vectordb

        self.reranker = None
        self.reranker_enabled = config.reranker["enable"]
        logger.info(f"Reranker enabled: {self.reranker_enabled}")
        self.reranker_top_k = int(config.reranker["top_k"])
        if self.reranker_enabled:
            self.reranker = Reranker(self.logger, config)

        self.prompts_dir = Path(config.paths.prompts_dir)
        self.rag_sys_prompt: str = load_sys_template(
            self.prompts_dir / config.prompt["rag_sys_pmpt"]
        )

        self.context_pmpt_tmpl = config.prompt["context_pmpt_tmpl"]

        self.retriever: ABCRetriever = RetrieverFactory.create_retriever(
            config=config, logger=self.logger
        )

        self.grader: Grader = None
        self.grader_enabled = config.grader["enable"]
        if self.grader_enabled:
            self.grader = Grader(config, logger=self.logger)

        self.rag_mode = config.rag["mode"]
        self.chat_history_depth = config.rag["chat_history_depth"]

        self._chat_history: deque = deque(maxlen=self.chat_history_depth)
        self.llm_client = LLM(config.llm, self.logger)

    async def get_contextualized_docs(
        self, partition: list[str], question: str, chat_history: list
    ) -> list[Document]:
        """With this function, the new question is reformulated as a standalone question that takes into account the chat_history.
        The new contextualized question is better suited for retreival.
        This contextualisation allows to have a RAG agent that also takes into account history, so chatbot RAG.

        Args:
            `question` (str): The user question
            `chat_history` (list): The conversation history
        """
        if (
            self.rag_mode == "SimpleRag"
        ):  # for the SimpleRag, we don't need the contextualize as questions are treated independently regardless of the chat_history
            docs = await self.retriever.retrieve(
                partition=partition, question=question, db=self.vectordb
            )
            contextualized_question = question

        if self.rag_mode == "ChatBotRag":
            template = load_sys_template(
                self.prompts_dir
                / self.context_pmpt_tmpl  # get the prompt for contextualizing
            )

            contextualize_q_prompt = ChatPromptTemplate.from_template(template)

            history_aware_retriever = (
                contextualize_q_prompt | self.llm_client.client | StrOutputParser()
            )
            input_ = {
                "query": question,
                "chat_history": chat_history,
            }
            contextualized_question = await history_aware_retriever.ainvoke(input_)
            logger.debug(f"Query: {contextualized_question}")

            docs = await self.retriever.retrieve(
                partition=partition, question=contextualized_question, db=self.vectordb
            )
            logger.debug(f"{len(docs)} Documents retreived")
            gc.collect()
            torch.cuda.empty_cache()
        return docs, contextualized_question

    async def run(
        self,
        partition: list[str],
        question: str = "",
        chat_history: list[AIMessage | HumanMessage] = None,
        llm_config: dict = None,
    ):
        rag_chain, input_, context, sources = await self._prepare_output(
            partition, question, chat_history, llm_config
        )
        return rag_chain.astream(input_), context, sources

    async def completion(
        self,
        partition: list[str],
        question: str = "",
        chat_history: list[AIMessage | HumanMessage] = None,
        llm_config: dict = None,
    ):
        rag_chain, input_, context, sources = await self._prepare_output(
            partition, question, chat_history, llm_config
        )

        return rag_chain.ainvoke(input_), context, sources

    async def _prepare_output(
        self,
        partition: list[str],
        question: str = "",
        chat_history: list[AIMessage | HumanMessage] = None,
        llm_config: dict = None,
    ):
        if chat_history:  # when the user provides chat_history (in api_mode)
            chat_history = chat_history[-self.chat_history_depth :]
        else:
            chat_history = list(self._chat_history)  # use the saved chat history

        # 1. contextualize the question and retreive relevant documents
        docs, contextualized_question = await self.get_contextualized_docs(
            partition=partition, question=question, chat_history=chat_history
        )

        if docs:
            # grade and filter irrelevant docs
            if self.grader_enabled:
                docs = self.grader.filter_docs(
                    user_input=contextualized_question, docs=docs
                )

            # 2. rerank documents is asked
            if self.reranker_enabled:
                docs = await self.reranker.rerank(
                    contextualized_question, chunks=docs, k=self.reranker_top_k
                )
            else:
                docs = docs[: self.reranker_top_k]

        # 3. Format the retrieved docs
        context, sources = format_context(docs)

        # 4. run the rag_chain
        llm_client: ChatOpenAI = self.llm_client.client
        llm_client = llm_client.with_config(config=llm_config)

        rag_chain, input_ = _prepare_pipeline(
            question=question,
            chat_history=chat_history,
            context=context,
            sys_pmpt_tmpl=self.rag_sys_prompt,
            llm_client=llm_client,
        )

        return rag_chain, input_, context, sources

    def update_history(self, question: str, answer: str):
        self._chat_history.extend(
            [
                HumanMessage(content=question),
                AIMessage(content=answer),
            ]
        )

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


def _prepare_pipeline(
    question: str,
    context: str,
    chat_history: list[AIMessage | HumanMessage],
    sys_pmpt_tmpl: str,
    llm_client: ChatOpenAI,
):
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", sys_pmpt_tmpl),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    rag_chain = qa_prompt | llm_client.with_retry(stop_after_attempt=2)
    input_ = {
        "input": question,
        "context": context,
        "chat_history": chat_history,
    }
    return rag_chain, input_
