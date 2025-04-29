import asyncio
import gc

import torch
from langchain_core.documents.base import Document
from loguru import logger
from sentence_transformers import CrossEncoder


class Reranker:
    """Reranks documents for a query using a RAG model."""

    def __init__(self, logger, config):
        """
        Initialize Reranker.

        Args:
            model_name (str): Name of pretrained RAGondin model to use.
        """
        self.model = CrossEncoder(
            config.reranker["model_name"],
            automodel_args={"torch_dtype": "auto"},
            trust_remote_code=True,
            device="cuda:0"
        )

        # Semaphore to limit concurrent GPU operations
        self.semaphore = asyncio.Semaphore(5)  # Only allow 5 GPU operation at a time
        self.logger = logger
        self.logger.debug("Reranker initialized...")

    async def rerank(
        self, query: str, chunks: list[Document], k: int = 5
    ) -> list[Document]:
        """
        Rerank documents by relevancy with respect to the given query.

        Args:
            query (str): Search query.
            docs (list[str]): List of document strings.
            k (int): Number of documents to return.

        Returns:
            list[str]: Top k reranked document strings.
        """

        logger.debug("Reranking documents")
        async with self.semaphore:
            k = min(k, len(chunks))  # k must be <= the number of documents
            rankings = self.model.rank(query, [doc.page_content for doc in chunks], return_documents=True, convert_to_tensor=True)
            ranked_txt = [{
                'content': ranking['text'],
                'score': ranking['score'],
                'rank': rankings.index(ranking),
                'result_index': ranking['corpus_id']
                } for ranking in rankings[:k]]
            # ranked_txt = await asyncio.to_thread(
            #     lambda: self.model.rerank(
            #         query, [d.page_content for d in chunks], k=k, bsize="auto"
            #     )
            # )
            gc.collect()
            torch.cuda.empty_cache()
            ranked_docs = [doc for doc in original_docs(ranked_txt, chunks)]
            return ranked_docs


def original_docs(ranked_txt, docs: list[Document]):
    for doc_txt in ranked_txt:
        for doc in docs:
            if doc_txt["content"] == doc.page_content:
                yield doc
                docs.remove(doc)
                break


if __name__ == "__main__":
    q = "Comment vas-tu?"
    resp = [
        "je n'y comprends rien",
        "je n'aime pas la politique",
        "je vais bien",
        "la bonne communication",
    ]
    reranker = Reranker()
    print(reranker.model.rerank(q, resp, k=3))