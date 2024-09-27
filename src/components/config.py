import os
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
from dataclasses import dataclass, field

# Load env variables from .env file
load_dotenv(dotenv_path="../")


@dataclass
class Config:
    """This class encapsulates the configurations for the application, 
    including settings for paths, directories, and LLM-specific parameters.
    """
    # Docs
    data_path: Path = Path("experiments/test_data").absolute()
    # temp_folder: Path = Path(__file__).parent.absolute() / "temp_files"
    chunker_name: str = "recursive_splitter"
    chunk_size: int = 1000
    chunk_overlap: int = 100 # TODO: Better express it with a percentage
    chunker_args: dict = field(default_factory= dict) # additional attributes specific to chunker
    
    # Embedding Model
    em_model_type: str = 'huggingface'
    em_model_name: str = "thenlper/gte-base"
    model_kwargs: dict = field(default_factory= lambda: {"device": "cpu"})
    encode_kwargs: dict = field(default_factory= lambda: {"normalize_embeddings": True})

    # Vector DB
    host: str = 'localhost'
    port: int = 6333
    collection_name: str = "docs_vdb"
    db_connector: int = "qdrant"

    # LLM Client    
    base_url: str = os.getenv('MODEL_URL', '')
    api_key: str = os.getenv('API_KEY', '')
    model_name: str = 'meta-llama-31-8b-it'
    timeout: int = 60
    rag_mode: Literal["ChatBotRag", "SimpleRag"] = "ChatBotRag"
    chat_history_depth: int = 4
    max_tokens: int = 1000

    # Reranker
    reranker_model_name: str | None = "colbert-ir/colbertv2.0"
    reranker_top_k: int = 5 # number of docs to return after reranking

    # retriever
    retreiver_type: Literal["hyde", "single", "multiQuery"] = "single"
    criteria: str = "similarity"
    top_k: int = 6
    retriever_extra_params: dict = field( # multiQuery retreiver type
        default_factory=lambda: {
            "k_queries": 3 # the llm will be added when creating the pipeline
        }
    )
    









