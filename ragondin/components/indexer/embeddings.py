from abc import abstractmethod, ABC
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
from omegaconf import OmegaConf


class ABCEmbedder(ABC):
    """Abstract base class defining the interface for embedder implementations.

    This class serves as a template for creating embedder classes that convert text
    into vector representations using various embedding models.
    """

    @abstractmethod
    def get_embeddings(self):
        """Return the embeddings model instance.

        Returns:
            Any: The embeddings model instance that can generate vector representations.
        """
        pass

class HFEmbedder(ABCEmbedder):
    """Factory class for loading and managing HuggingFace embedding models.

    This class handles the initialization and configuration of various HuggingFace
    embedding models, supporting both BGE and standard HuggingFace embeddings.

    Args:
        embedder_config (OmegaConf): Configuration object containing model parameters
        device (str, optional): Device to run the model on ('cuda' or 'cpu'). 
            Defaults to None, which auto-selects based on CUDA availability.
    
    Raises:
        ValueError: If the specified model type is not supported or if initialization fails.
    """

    def __init__(self, embedder_config: OmegaConf, device=None) -> None:
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        try:
            self.embedding = HuggingFaceEmbeddings(
                model_name=embedder_config["model_name"],
                model_kwargs={"device": device, 'trust_remote_code': True},
                encode_kwargs={"normalize_embeddings": True}
            )
        except Exception as e:
            raise ValueError(f"An error occurred during model initialization: {e}")
    


    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """Retrieve the initialized embedding model."""
        return self.embedding