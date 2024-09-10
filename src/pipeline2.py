import copy
from pathlib import Path
from .chunker import Docs, get_chunker_cls, CHUNKERS
from .llm import LLM
from .prompt import Prompt
from .reranker import Reranker
from .retriever import SingleRetriever, MultiQueryRetriever, get_retreiver_cls, BaseRetriever
from .vector_store import CONNECTORS, BaseVectorDdConnector
from .embeddings import HFEmbedder, BaseEmbedder
from .config import Config
from openai import OpenAI, AsyncOpenAI

class Doc2VdbPipe:
    def __init__(self, config: Config) -> None: 
        self.chunker = CHUNKERS[config.chunker_name](
            chunk_size=config.chunk_size, 
            chunk_overlap=config.chunk_overlap, 
            chunker_args=config.chunker_args
        )

        # TODO: Is it necessary to save this?
        embedder = HFEmbedder(
            model_type=config.em_model_type,
            model_name=config.em_model_name, 
            model_kwargs=config.model_kwargs, 
            encode_kwargs=config.encode_kwargs
        )

        self.connector = CONNECTORS[config.db_connector](
            host=config.host,
            port=config.port,
            collection_name=config.collection_name,
            embeddings=embedder.embedding
        )


    def load_files2db(self, data_path: Path = None):
        # get the data
        docs = Docs()
        docs.load(dir_path=data_path) # populate docs object
        docs_splited = self.chunker.split(docs=docs.get_docs())
        
        try:
            self.connector.add_documents(docs_splited)
        except Exception as e:
            raise Exception(f"An exception as occured: {e}")
      
    def load_file2db(self, file_path: str | Path):
        docs = Docs()
        docs.load_file(file_path=file_path) # populuate the docs object

        # TODO: Think about chunking method with respect to file type
        docs_splited = self.chunker.split(docs=docs.get_docs())
        self.connector.add_documents(docs_splited)



class RagPipeline:
    def __init__(self, config: Config) -> None:
        print("Doc2VdbPipe...")
        docvdbPipe = Doc2VdbPipe(config=config)
        docvdbPipe.load_files2db(data_path=config.data_path)
        self.docvdbPipe = docvdbPipe
        
        print("Reranker...")
        self.reranker = None
        if config.reranker_model_name is not None:
            self.reranker = Reranker(
                model_name=config.reranker_model_name,
            )
        
        print("Prompt...")
        self.prompt = Prompt(type_template=config.prompt_template)        
        self.llm_client = LLM(
            client=AsyncOpenAI(
                base_url=config.base_url, 
                api_key=config.api_key, 
                timeout=config.timeout
            ),
            model_name=config.model_name,
            max_tokens=config.max_tokens
        )

        print("Retriever...")
        retreiver_cls = get_retreiver_cls(retreiver_type=config.retreiver_type)
        extra_params = config.retriever_extra_params
        if config.retreiver_type == "multiQuery":
            extra_params["llm"] = LLM(
                client=OpenAI(
                    base_url=config.base_url, 
                    api_key=config.api_key, 
                    timeout=config.timeout
                ),
                model_name=config.model_name,
                max_tokens=config.max_tokens
            )
            extra_params["prompt_multi_queries"] = Prompt(type_template=config.retreiver_type)
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

        self.reranker_top_k = config.reranker_top_k

    async def run(self, question: str=""):
        docs_txt = self.retriever.retrieve(
            question, 
            db=self.docvdbPipe.connector
        )
        if self.reranker is not None:
            docs_txt = self.reranker.rerank(question, docs=docs_txt, k=self.reranker_top_k)

        prompt_dict = self.prompt.get_prompt(question, docs=docs_txt)
        answer = await self.llm_client.async_run(prompt_dict)
        return answer
