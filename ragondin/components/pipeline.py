import gc
import sys
import torch
from langchain_core.documents.base import Document
from .llm import LLM
from .utils import format_context, load_sys_template
from .reranker import Reranker
from .retriever import ABCRetriever, RetrieverFactory
from .indexer import ABCVectorDB

from loguru import logger
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    MessagesPlaceholder, 
    ChatPromptTemplate
)
from langchain_core.messages import (
    AIMessage, 
    HumanMessage
)
from pathlib import Path
from collections import deque
from .grader import Grader

class RagPipeline:
    def __init__(self, config, vectordb: ABCVectorDB, logger=None) -> None:
        self.config = config
        self.logger = self.set_logger(config) if logger is None else logger
        self.vectordb: ABCVectorDB = vectordb
            
        self.reranker = None
        if config.reranker["model_name"]:
            self.reranker = Reranker(self.logger, config)

        self.reranker_top_k = int(config.reranker["top_k"])

        self.prompts_dir = Path(config.paths.prompts_dir)
        self.rag_sys_prompt: str = load_sys_template(self.prompts_dir / config.prompt['rag_sys_pmpt'])
        
        self.context_pmpt_tmpl = config.prompt['context_pmpt_tmpl']

        self.retriever: ABCRetriever = RetrieverFactory.create_retriever(config=config, logger=self.logger)

        self.grader: Grader = None
        if config.grader['grade_documents']:
            self.grader = Grader(config, logger=self.logger)

        self.rag_mode = config.rag["mode"]
        self.chat_history_depth = config.rag["chat_history_depth"]

        self._chat_history: deque = deque(maxlen=self.chat_history_depth)
        self.llm_client = LLM(config, self.logger)

    async def get_contextualized_docs(self, partition: list[str], question: str, chat_history: list)-> list[Document]:
        """With this function, the new question is reformulated as a standalone question that takes into account the chat_history.
        The new contextualized question is better suited for retreival. 
        This contextualisation allows to have a RAG agent that also takes into account history, so chatbot RAG.

        Args:
            `question` (str): The user question
            `chat_history` (list): The conversation history
        """        
        if self.rag_mode == "SimpleRag": # for the SimpleRag, we don't need the contextualize as questions are treated independently regardless of the chat_history
            docs = await self.retriever.retrieve(
                partition=partition,
                question=question, 
                db=self.vectordb
            )
            contextualized_question = question

        if self.rag_mode == "ChatBotRag":
            template = load_sys_template(
                self.prompts_dir / self.context_pmpt_tmpl # get the prompt for contextualizing
            )
    
            contextualize_q_prompt = ChatPromptTemplate.from_template(template)

            history_aware_retriever = (
                contextualize_q_prompt
                | self.llm_client.client
                | StrOutputParser()
            )
            input_ = {
                "query": question, 
                "chat_history": chat_history,
            }
            contextualized_question = await history_aware_retriever.ainvoke(input_)  
            logger.debug(f"Query: {contextualized_question}")

            docs = await self.retriever.retrieve(
                partition=partition,
                question=contextualized_question, 
                db=self.vectordb
            )
            logger.debug(f"{len(docs)} Documents retreived")
            gc.collect()
            torch.cuda.empty_cache()
        return docs, contextualized_question
    

    async def run(self, partition : list[str], question: str="", chat_history: list[AIMessage | HumanMessage]=None):
        if chat_history: # when the user provides chat_history (in api_mode)
            chat_history = chat_history[-self.chat_history_depth:]
        else:
            chat_history = list(self._chat_history) # use the saved chat history
      
        # 1. contextualize the question and retreive relevant documents
        docs, contextualized_question = await self.get_contextualized_docs(partition=partition, question=question, chat_history=chat_history) 

        
        if docs:
            # grade and filter irrelevant docs
            docs = await self.grader.grade_docs(user_input=contextualized_question, docs=docs) if self.grader else docs

            # 2. rerank documents is asked
            if self.reranker:
                docs = await self.reranker.rerank(contextualized_question, chunks=docs, k=self.reranker_top_k)
            
        # 3. Format the retrieved docs
        context, sources = format_context(docs)

        # 4. run the llm for inference
        answer = self.llm_client.run(
            question=question, 
            chat_history=chat_history,
            context=context,
            sys_pmpt_tmpl=self.rag_sys_prompt
        )
    
        self.free_memory()
        
        return answer, context, sources


    def free_memory(self):
        gc.collect()
        torch.cuda.empty_cache()

    
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
            level = verbose['level']
        else:
            level = 'ERROR'
        logger.remove()
        logger.add(sys.stderr, level=level)
        return logger