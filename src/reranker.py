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
        docs_cleaned = [doc for doc in drop_duplicate(docs)]
        k = min(k, len(docs_cleaned)) # k must be <= the number of documents
        ranked_docs = self.model.rerank(question, docs_cleaned, k=k)

        # drop d
        return [doc["content"] for doc in ranked_docs]


def drop_duplicate(L: list[str], key=None):
    seen = set()
    for s in L:
        val = s if key is None else key(s)
        if val not in seen:
            seen.add(val)
            yield s


if __name__ == "__main__":
    q = 'Comment vas-tu?'
    resp = ["je n'y comprends rien", "je n'aime pas la politique", "je vais bien",  "la bonne communication"]
    reranker = Reranker()
    print(reranker.model.rerank(q, resp, k=3))