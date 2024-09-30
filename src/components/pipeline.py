from pathlib import Path
from .chunker import Docs, CHUNKERS
from .llm import LLM
from .utils import format_context, load_sys_template
from .reranker import Reranker
from .retriever import get_retreiver_cls, BaseRetriever
from .vector_store import CONNECTORS
from .embeddings import HFEmbedder
from .config import Config
from loguru import logger
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import (
    MessagesPlaceholder, 
    ChatPromptTemplate
)

from collections import deque
from langchain_core.messages import AIMessage, HumanMessage


dir_path = Path(__file__).parent


class Doc2VdbPipe:
    """This class bridges static files with the vector database.
    """
    def __init__(self, config: Config) -> None:
        # init chunker 
        self.chunker = CHUNKERS[config.chunker_name](
            chunk_size=config.chunk_size, 
            chunk_overlap=config.chunk_overlap, 
            chunker_args=config.chunker_args
        )

        # init the embedder model
        embedder = HFEmbedder(
            model_type=config.em_model_type,
            model_name=config.em_model_name, 
            model_kwargs=config.model_kwargs, 
            encode_kwargs=config.encode_kwargs
        )

        # init the connector
        self.connector = CONNECTORS[config.db_connector](
            host=config.host,
            port=config.port,
            collection_name=config.collection_name,
            embeddings=embedder.embedding
        )

    async def add_file2vdb(self, file_path: str | Path):
        """Add a file to the vector database in async mode"""
        docs = Docs()
        try:
            docs.load_file(file_path=file_path) # populuate the docs object
            # TODO: Think about chunking method with respect to file type
            docs_splited = self.chunker.split(docs=docs.get_docs())
            await self.connector.aadd_documents(docs_splited)
            logger.info(f"Documents from {file_path} added.")
        except Exception as e:
            raise Exception(f"An exception as occured: {e}")


class RagPipeline:
    def __init__(self, config: Config) -> None:
        docvdbPipe = Doc2VdbPipe(config=config)
        self.docvdbPipe = docvdbPipe
        logger.info("File to VectorDB Connector initialized...")
        
        self.reranker = None
        if config.reranker_model_name is not None:
            self.reranker = Reranker(
                model_name=config.reranker_model_name,
            )
            logger.info("Reranker initialized...")
        

        self.qa_sys_prompt: ChatPromptTemplate = load_sys_template(
            dir_path / "prompts/basic_sys_prompt_template.txt"
        )

        llm_client = LLM(
            model_name=config.model_name, 
            base_url=config.base_url, api_key=config.api_key, timeout=config.timeout, 
            max_tokens=config.max_tokens
        )
        self.llm_client = llm_client

        retreiver_cls = get_retreiver_cls(retreiver_type=config.retreiver_type)
        logger.info("Init Retriever...")

        extra_params = config.retriever_extra_params
        if config.retreiver_type in ["hyde", "multiQuery"]:
            extra_params["llm"] = llm_client.client # add an llm to extra parameters for these types of retreivers

            self.retriever: BaseRetriever = retreiver_cls(
                criteria=config.criteria,
                top_k=config.top_k,
                **extra_params
            )
        if config.retreiver_type == "single": # for single retreiver
            self.retriever: BaseRetriever = retreiver_cls(
                criteria=config.criteria,
                top_k=config.top_k
            )
            if config.retriever_extra_params: 
                logger.info(f"'retriever_extra_params' is not used for the `{config.retreiver_type}` retreiver")


        self.reranker_top_k = config.reranker_top_k
        self.rag_mode = config.rag_mode
        self.chat_history_depth = config.chat_history_depth
        self._chat_history: deque = deque(maxlen=config.chat_history_depth)

    def get_contextualize_docs(self, question: str, chat_history: list):
        """With this function, the new question is reformulated as a standalone question that takes into acoount the chat_history.
        The new question is then used for retreival. So this allows to have RAG agent that acts as a chatbot aswell.

        Args:
            `question` (str): The new question
            `chat_history` (list): The history
        """        
        if self.rag_mode == "SimpleRag": # for the SimpleRag, we don't need the contextualize as question are treated independently regardless of the chat_history
            logger.info("Documents retreived...")
            docs = self.retriever.retrieve(
                question, 
                db=self.docvdbPipe.connector
            )

        if self.rag_mode == "ChatBotRag":
            logger.info("Contextualizing the question")
            
            sys_prompt = load_sys_template(
                dir_path / "prompts/contextualize_sys_prompt_template.txt" # get the prompt for contextualizing
            )
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", sys_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "Here is the question to contextualize: '{input}'"),
                ]
            )
            history_aware_retriever = (
                contextualize_q_prompt
                | self.llm_client.client 
                | StrOutputParser()
            )

            input_ = {"input": question, "chat_history": chat_history}
            logger.info("Generating contextualized question for retreival...") 
            contextualized_question = history_aware_retriever.invoke(input_) # TODO: this is the bootleneck, the model answers sometimes instead of reformulating  
            logger.info("Documents retreived...")

            docs = self.retriever.retrieve(
                contextualized_question, 
                db=self.docvdbPipe.connector
            )
        return docs 

    def run(self, question: str="", chat_history_api: list[AIMessage | HumanMessage] = None):
        if chat_history_api is None: 
            chat_history = list(self._chat_history) # use the saved chat history
        else:
            # this is for when the user provides chat_history (in api_mode)
            chat_history = chat_history[self.chat_history_depth:]

        # 1. contextualize the question and retreive relevant documents
        docs = self.get_contextualize_docs(question, chat_history) 

        # 2. rerank documents is asked
        if self.reranker is not None:
            docs = self.reranker.rerank(question, docs=docs, k=self.reranker_top_k)
        
        # 3. Format the retrieved docs
        context, sources = format_context(docs)

        # 4. run the llm for inference
        answer = self.llm_client.run(
            question=question, 
            chat_history=chat_history,
            context=context, sys_prompt_tmpl=self.qa_sys_prompt)

        return answer, context, sources

    
    def update_history(self, question: str, answer: str):
        self._chat_history.extend(
            [
                HumanMessage(content=question),
                AIMessage(content=answer),
            ]
        )