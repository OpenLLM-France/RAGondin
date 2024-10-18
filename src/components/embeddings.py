from abc import ABCMeta, abstractmethod
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
import torch


class BaseEmbedder(metaclass=ABCMeta):
    """Abstract class for embedders
    """
    @abstractmethod
    def get_embeddings(self):
        pass

HG_EMBEDDER_TYPE = {
    "huggingface_bge": HuggingFaceBgeEmbeddings,
    "huggingface": HuggingFaceEmbeddings
}

class HFEmbedder(BaseEmbedder):
    """Factory class for loading HuggingFace embeddings models backend models.
    """
    def __init__(self, config, device=None) -> None:
        """Initialize Embeddings.

        Args:
            model_type (str): Type of embedding model to use. Defaults to 'huggingface'.
            model_name (str): Name of specific model.. Defaults to "thenlper/gte-small".
        Raises:
            ValueError: If invalid `model_type` passed or non-existant `model_name`.
        """

        model_type = config.embedder["type"]
        if model_type in HG_EMBEDDER_TYPE:
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            try:
                model_name = config.embedder["name"]
                self.embedding = HG_EMBEDDER_TYPE[model_type](
                    model_name=model_name,
                    model_kwargs={"device": device},
                    encode_kwargs={"normalize_embeddings": True}
                )
            except Exception as e:
                raise ValueError(f"An error occured: {e}")
        else:
            raise ValueError(f"{model_type} is not a supported `model_type`")

    def get_embeddings(self) -> HuggingFaceBgeEmbeddings:
        """
        Return the generated embeddings. This will be used in the Vector DB.
        """
        return self.embedding
