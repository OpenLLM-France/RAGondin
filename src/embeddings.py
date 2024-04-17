from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings


class Embeddings:
    """
    Factory class to generate embeddings using different backend models.
    """

    def __init__(self, model_type: str = 'huggingface', model_name: str = "thenlper/gte-small", model_kwargs=None,
                 encode_kwargs=None) -> None:
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
        if model_kwargs is None:
            model_kwargs = {"device": "cuda"}
        if encode_kwargs is None:
            encode_kwargs = {"normalize_embeddings": True}
        if model_type == "huggingface_bge":
            self.embeddings = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
        elif model_type == "huggingface":
            self.embeddings = HuggingFaceEmbeddings(
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
