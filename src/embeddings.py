from abc import ABCMeta, abstractmethod
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings


class BaseEmbedder(metaclass=ABCMeta):
    """_Abstract
    """
    @abstractmethod
    def get_embeddings(self):
        pass


HG_EMBEDDER_TYPE = {
    "huggingface_bge": HuggingFaceBgeEmbeddings,
    "huggingface": HuggingFaceEmbeddings
}

class HFEmbedder(BaseEmbedder):
    """Factory class to generate embeddings using different backend models.
    """
    def __init__(
            self, 
            model_type: str = 'huggingface', 
            model_name: str = "thenlper/gte-small", 
            model_kwargs = {"device": "cpu"},
            encode_kwargs = {"normalize_embeddings": True}
        ) -> None:
        """
        Initialize Embeddings.

        Args:
            model_type (str): Type of embedding model to use.
            model_name (str): Name of specific model.
            model_kwargs (dict): kwargs to pass to model constructor.
            encode_kwargs (dict): kwargs for encoding documents.

        Raises:
            ValueError: If invalid model_type passed.
        """

        # cach_folder
        if model_type in HG_EMBEDDER_TYPE:
            try:
                self.embedding = HG_EMBEDDER_TYPE[model_type](
                    model_name=model_name,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs
                )

            except Exception as e:
                raise ValueError(f"An error occured: {e}")
        else:
            raise ValueError(f"{model_type} is not a valid `model_type`")

    def get_embeddings(self) -> HuggingFaceBgeEmbeddings:
        """
        Return the generated embeddings.
        """
        return self.embedding
