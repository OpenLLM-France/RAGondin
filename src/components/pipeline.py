import gc
import sys

import torch
from .chunker import ABCChunker, ChunkerFactory
from langchain_core.documents.base import Document
from .llm import LLM
from .utils import format_context, load_sys_template
from .reranker import Reranker
from .retriever import ABCRetriever, RetrieverFactory
from .vectordb import ConnectorFactory
from .embeddings import HFEmbedder
from omegaconf import OmegaConf
from .loader import DocSerializer
from loguru import logger
from typing import AsyncGenerator
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

class Indexer:
    """This class bridges static files with the vector store database.
    """
    def __init__(self, config: OmegaConf, logger, device=None) -> None:
        embedder = HFEmbedder(embedder_config=config.embedder, device=device)
        self.serializer = DocSerializer(root_dir=config.paths.root_dir)
        self.chunker: ABCChunker = ChunkerFactory.create_chunker(config, embedder=embedder.get_embeddings())
        self.vectordb = ConnectorFactory.create_vdb(config, logger=logger, embeddings=embedder.get_embeddings())
        self.logger = logger
        self.logger.info("Indexer initialized...")


    async def add_files2vdb(self, path):
        """Add a files to the vector database in async mode"""
        try:
            doc_generator: AsyncGenerator[Document, None] = self.serializer.serialize_documents(path, recursive=True)
            await self.vectordb.async_add_documents(
                doc_generator=doc_generator, 
                chunker=self.chunker, 
                document_batch_size=4
            )
            self.logger.info(f"Documents from {path} added.")
        except Exception as e:
            raise Exception(f"An exception as occured: {e}")


class RagPipeline:
    def __init__(self, config, device="cpu") -> None:
        self.config = config
        # print(self.config)
        self.logger = self.set_logger(config)
        self.indexer = Indexer(config, self.logger, device=device)
            
        self.reranker = None
        if config.reranker["model_name"]:
            self.reranker = Reranker(self.logger, config)

        self.reranker_top_k = int(config.reranker["top_k"])

        self.prompts_dir = Path(config.paths.prompts_dir)

        self.qa_sys_prompt: str = load_sys_template(
            self.prompts_dir / config.prompt['rag_sys_pmpt']
        )
        
        self.context_pmpt_tmpl = config.prompt['context_pmpt_tmpl']

        self.retriever: ABCRetriever = RetrieverFactory.create_retriever(config=config, logger=self.logger)

        self.grader: Grader = None
        if config.grader['grade_documents']:
            self.grader = Grader(config, logger=self.logger)

        self.rag_mode = config.rag["mode"]

        self.chat_history_depth = config.rag["chat_history_depth"]

        self._chat_history: deque = deque(maxlen=self.chat_history_depth)
        self.llm_client = LLM(
            config, self.logger, 
        )

    async def get_contextualize_docs(self, question: str, chat_history: list)-> list[Document]:
        """With this function, the new question is reformulated as a standalone question that takes into account the chat_history.
        The new contextualized question is better suited for retreival. 
        This contextualisation allows to have a RAG agent that also takes into account history, so chatbot RAG.

        Args:
            `question` (str): The user question
            `chat_history` (list): The conversation history
        """        
        if self.rag_mode == "SimpleRag": # for the SimpleRag, we don't need the contextualize as questions are treated independently regardless of the chat_history
            logger.info("Documents retreived...")
            docs = await self.retriever.retrieve(
                question, 
                db=self.indexer.vectordb
            )
            contextualized_question = question

        if self.rag_mode == "ChatBotRag":
            logger.info("Contextualizing the question...")
            
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

            logger.info("Generating contextualized question for retreival...") 
            contextualized_question = await history_aware_retriever.ainvoke(input_) # TODO: this is the bootleneck, the model answers sometimes instead of reformulating  
            print("==>", contextualized_question)
            logger.info("Documents retreived...")

            docs = await self.retriever.retrieve(
                contextualized_question, 
                db=self.indexer.vectordb
            )
            gc.collect()
            torch.cuda.empty_cache()
        return docs, contextualized_question
    

    async def run(self, question: str="", chat_history: list[AIMessage | HumanMessage]=None):
        if chat_history: # when the user provides chat_history (in api_mode)
            chat_history = chat_history[-self.chat_history_depth:]
        else:
            chat_history = list(self._chat_history) # use the saved chat history
      
        # 1. contextualize the question and retreive relevant documents
        docs, contextualized_question = await self.get_contextualize_docs(question, chat_history) 

        # grade and filter irrelevant docs
        docs = await self.grader.grade(user_input=contextualized_question, docs=docs) if self.grader else docs

        if docs:
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
            sys_pmpt_tmpl=self.qa_sys_prompt
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