"""Module for document reranking using RAG models."""

from ragatouille import RAGPretrainedModel


class Reranker:
    """Reranks documents for a query using a RAG model."""

    def __init__(self, model_name: str = "colbert-ir/colbertv2.0"):
        """
        Initialize Reranker.

        Args:
            model_name (str): Name of pretrained RAGondin model to use.
        """
        self.model = RAGPretrainedModel.from_pretrained(model_name)

    def rerank(self, question: str, docs: list[str], k: int = 5) -> list[str]:
        """
        Rerank documents for a query.

        Args:
            question (str): Search query.
            docs (list[str]): List of document strings.
            k (int): Number of documents to return.

        Returns:
            list[str]: Top k reranked document strings.
        """
        ranked_docs = self.model.rerank(question, docs, k=k)
        return [doc["content"] for doc in ranked_docs]
