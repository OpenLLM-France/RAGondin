from langchain_community.embeddings import HuggingFaceBgeEmbeddings


class Embeddings:
    """
    Factory class to generate embeddings using different backend models.
    """

    def __init__(self, model_type, model_name, model_kwargs, encode_kwargs) -> None:
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
        if model_type == "huggingface_bge":
            self.embeddings = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
        else:
            raise ValueError(f"{model_type} is not a valid model_type")

    def get_embeddings(self) -> HuggingFaceBgeEmbeddings:
        """
        Return the generated embeddings.
        """
        return self.embeddings

