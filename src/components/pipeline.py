from pathlib import Path
from .chunker import Docs, get_chunker_cls, CHUNKERS
from .llm import LLM, LLM2, template_from_sys_template
from .prompt import BasicPrompt, MultiQueryPrompt, format_context, get_sys_template
from .reranker import Reranker
from .retriever import get_retreiver_cls, BaseRetriever
from .vector_store import CONNECTORS
from .embeddings import HFEmbedder
from .config import Config
from openai import OpenAI, AsyncOpenAI
from loguru import logger
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

dir_path = Path(__file__).parent


class Doc2VdbPipe:
    """This class bridges static files with the vector database.
    """
    def __init__(self, config: Config) -> None:
        # get chunker 
        self.chunker = CHUNKERS[config.chunker_name](
            chunk_size=config.chunk_size, 
            chunk_overlap=config.chunk_overlap, 
            chunker_args=config.chunker_args
        )

        # get the embedder model
        embedder = HFEmbedder(
            model_type=config.em_model_type,
            model_name=config.em_model_name, 
            model_kwargs=config.model_kwargs, 
            encode_kwargs=config.encode_kwargs
        )

        # define the connector
        self.connector = CONNECTORS[config.db_connector](
            host=config.host,
            port=config.port,
            collection_name=config.collection_name,
            embeddings=embedder.embedding
        )

    def load_files2db(self, data_path: Path = None):
        """Add files from a `data_path`

        Args:
            data_path (Path, optional): Data Path. Defaults to None.

        Raises:
            Exception:
        """
        # get the data
        docs = Docs()
        docs.load(dir_path=data_path) # populate docs object
        docs_splited = self.chunker.split(docs=docs.get_docs())
        
        try:
            self.connector.add_documents(docs_splited)
        except Exception as e:
            raise Exception(f"An exception as occured: {e}")
      
    def load_file2db(self, file_path: str | Path):
        """Add a file to the vector data base"""
        docs = Docs()
        docs.load_file(file_path=file_path) # populuate the docs object

        # TODO: Think about chunking method with respect to file type
        docs_splited = self.chunker.split(docs=docs.get_docs())
        self.connector.add_documents(docs_splited)
        logger.info(f"Documents from {file_path} added.")


class RagPipeline:
    def __init__(self, config: Config) -> None:
        docvdbPipe = Doc2VdbPipe(config=config)
        logger.info(f"Doc to Vector database initialised")
        docvdbPipe.load_files2db(data_path=config.data_path)
        self.docvdbPipe = docvdbPipe
        
        print("Reranker...")
        self.reranker = None
        if config.reranker_model_name is not None:
            self.reranker = Reranker(
                model_name=config.reranker_model_name,
            )
        
        print("Prompt...")
        self.prompt: ChatPromptTemplate = get_sys_template(
            dir_path / "prompts/basic_sys_prompt_template.txt"
        )

        self.llm_client = LLM2(
            model_name=config.model_name, 
            base_url=config.base_url, api_key=config.api_key, timeout=config.timeout, 
            max_tokens=config.max_tokens, 
            streaming=True
        )

        print("Retriever...")
        retreiver_cls = get_retreiver_cls(retreiver_type=config.retreiver_type)
        extra_params = config.retriever_extra_params

        if config.retreiver_type == "multiQuery":
            extra_params["llm"] = LLM(
                client=OpenAI(
                    base_url=config.base_url, 
                    api_key=config.api_key, 
                    timeout=config.timeout,
                ),
                model_name=config.model_name,
                max_tokens=config.max_tokens,
                chat_mode="SimpleLLM"
            )
            extra_params["prompt_multi_queries"] = MultiQueryPrompt()
            self.retriever: BaseRetriever = retreiver_cls(
                criteria=config.criteria,
                top_k=config.top_k,
                **extra_params
            )
        if config.retreiver_type == "single":
            self.retriever: BaseRetriever = retreiver_cls(
                criteria=config.criteria,
                top_k=config.top_k
            )
            if config.retriever_extra_params: 
                logger.info(f"'retriever_extra_params' is not used in {config.retreiver_type} retreiver")

        self.reranker_top_k = config.reranker_top_k
        self.rag_mode = config.rag_mode
        self._chat_history: list = []
    

    def get_contextualize_docs(self, question: str, chat_history: list):
        if self.rag_mode == "SimpleRag":
            docs = self.retriever.retrieve(
                question, 
                db=self.docvdbPipe.connector
            )
        if self.rag_mode == "ChatBotRag":
            sys_prompt = get_sys_template(
                dir_path / "prompts/contextualize_sys_prompt_template.txt"
            )
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", sys_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
            history_aware_retriever = (
                contextualize_q_prompt
                | self.llm_client.client 
                | StrOutputParser()
                | (lambda x: self.retriever.retrieve(x, db=self.docvdbPipe.connector))
            )

            input_ = {"input": question, "chat_history": chat_history}
            docs = history_aware_retriever.invoke(input_)

        return docs

            

    def run(self, question: str="", chat_history: list = None):
        docs = self.get_contextualize_docs(question, chat_history)

        if self.reranker is not None:
            docs = self.reranker.rerank(question, docs=docs, k=self.reranker_top_k)
        context = format_context(docs)

        answer = self.llm_client.run(
            question=question, 
            chat_history=chat_history,
            context=context, sys_msg=self.prompt)

        return answer, context
