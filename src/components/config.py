import os
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass, field


# Load env variables from .env file
load_dotenv(dotenv_path="../")

@dataclass
class Config:
    """This class encapsulates the configurations for the application, 
    including settings for paths, directories, prompts, and LLM-specific parameters.
    """
    # Docs
    data_path: Path = "./experiments/test_data" # Put None
    chunker_name: str = "recursive_splitter"
    chunk_size: int = 800
    chunk_overlap: int = 80 # TODO: Better express it with a percentage
    chunker_args: dict = field(default_factory= dict) # additional attributes specific to chunker
    
    # Embedding Model
    em_model_type: str = 'huggingface'
    em_model_name: str = "thenlper/gte-base"
    model_kwargs: dict = field(default_factory= lambda: {"device": "cpu"})
    encode_kwargs: dict = field(default_factory= lambda: {"normalize_embeddings": True})

    # Vector DB
    host: str = None
    port: int = 0
    collection_name: str = "my_docs"
    db_connector: int = "qdrant"

    # LLM Client    
    base_url: str = os.getenv('MODEL_URL', '')
    api_key: str = os.getenv('API_KEY', '')
    model_name: str = 'meta-llama-31-8b-it'
    timeout: int = 60
    prompt_template = "basic"
    max_tokens: int = 1000

    # Reranker
    reranker_model_name: str | None = "colbert-ir/colbertv2.0"
    reranker_top_k: int = 5 # number of docs to return after reranking

    # retriever
    retreiver_type: str = "hyde"
    criteria: str = "similarity"
    top_k: int = 5
    retriever_extra_params: dict = field( # multiQuery retreiver type
        default_factory=lambda: {
            "k_multi_queries": 3 # llm and the prompt template will be added
        }
    )
    









