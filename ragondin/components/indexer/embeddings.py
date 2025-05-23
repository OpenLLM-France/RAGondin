from abc import ABC, abstractmethod

import torch
from langchain_huggingface import HuggingFaceEmbeddings
from omegaconf import OmegaConf
from ..utils import SingletonABCMeta


class ABCEmbedder(ABC):
    @abstractmethod
    def get_embeddings(self):
        """Return the embeddings model instance.

        Returns:
            Any: The embeddings model instance that can generate vector representations.
        """
        pass


class HFEmbedder(ABCEmbedder, metaclass=SingletonABCMeta):
    def __init__(self, embedder_config: OmegaConf, device=None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self.embedding = HuggingFaceEmbeddings(
                model_name=embedder_config["model_name"],
                model_kwargs={"device": device, "trust_remote_code": True},
                encode_kwargs={"normalize_embeddings": True, "convert_to_tensor": True},
            )
        except Exception as e:
            raise ValueError(f"An error occurred during model initialization: {e}")

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """Retrieve the initialized embedding model."""
        return self.embedding
