from abc import abstractmethod, ABC
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
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


# Dictionary mapping embedding model types to their corresponding classes
HG_EMBEDDER_TYPE = {
    "huggingface_bge": HuggingFaceBgeEmbeddings,
    "huggingface": HuggingFaceEmbeddings
}


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
        # Extract model type from config
        model_type = embedder_config["type"]

        if model_type in HG_EMBEDDER_TYPE:
            # Auto-select device if none specified
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            try:
                model_name = embedder_config["model_name"]
                self.embedding = HG_EMBEDDER_TYPE[model_type](
                    model_name=model_name,
                    model_kwargs={"device": device, 'trust_remote_code': True},
                    encode_kwargs={"normalize_embeddings": True}
                )
            except Exception as e:
                raise ValueError(f"An error occurred during model initialization: {e}")
        else:
            raise ValueError(f"{model_type} is not a supported `model_type`")


    def get_embeddings(self) -> HuggingFaceBgeEmbeddings:
        """Retrieve the initialized embedding model.

        Returns:
            HuggingFaceBgeEmbeddings: The configured embedding model instance
            ready for generating embeddings.
        """
        return self.embedding