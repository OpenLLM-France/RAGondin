import gc
import copy
import torch
import asyncio
from enum import Enum
from ragatouille import RAGPretrainedModel
from sentence_transformers import CrossEncoder
from langchain_core.documents.base import Document
from .utils import SingletonMeta


class RerankerType(Enum):
    CROSSENCODER = "crossencoder"
    COLBERT = "colbert"


class Reranker(metaclass=SingletonMeta):
    def __init__(self, logger, config):
        reranker_type = config.reranker["reranker_type"]
        model_name = config.reranker["model_name"]

        self.logger = logger
        self.semaphore = asyncio.Semaphore(
            5
        )  # Only allow 5 reranking operation at a time

        self.reranker_type = RerankerType(reranker_type)
        self.logger.debug(
            f"Reranker type: {self.reranker_type}, Model name: {model_name}"
        )

        match self.reranker_type:
            case RerankerType.CROSSENCODER:
                self.model = CrossEncoder(
                    model_name=model_name,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    trust_remote_code=True,
                )
            case RerankerType.COLBERT:
                # Initialize ColBERT model here
                self.model = RAGPretrainedModel.from_pretrained(
                    pretrained_model_name_or_path=model_name
                )
            case _:
                raise ValueError(
                    "reranker_type must be either 'crossencoder' or 'colbert'"
                )

        self.logger.debug(f"{self.reranker_type} Reranker initialized...")

    async def rerank(
        self, query: str, documents: list[Document], top_k: int = 6
    ) -> list[Document]:
        self.logger.debug("Reranking documents")
        top_k = min(top_k, len(documents))

        async with self.semaphore:
            match self.reranker_type:
                case RerankerType.CROSSENCODER:
                    reranked_docs = await asyncio.to_thread(
                        lambda: self.___crossencoder_rerank(query, documents, top_k)
                    )
                    return reranked_docs

                case RerankerType.COLBERT:
                    reranked_docs = await asyncio.to_thread(
                        lambda: self.___colbert_rerank(query, documents, top_k)
                    )
                    return reranked_docs

    def ___crossencoder_rerank(
        self, query: str, documents: list[Document], top_k: int
    ) -> list[Document]:
        with torch.no_grad():
            docs_txt = [doc.page_content for doc in documents]
            results = self.model.rank(query=query, documents=docs_txt, top_k=top_k)

        gc.collect()
        torch.cuda.empty_cache()
        return [documents[r["corpus_id"]] for r in results]

    def ___colbert_rerank(
        self, query: str, documents: list[Document], top_k: int
    ) -> list[Document]:
        with torch.no_grad():
            docs_txt = [doc.page_content for doc in documents]
            results = self.model.rerank(
                query=query, documents=docs_txt, k=top_k, bsize="auto"
            )

        gc.collect()
        torch.cuda.empty_cache()
        return [doc for doc in original_docs(results, documents)]


def original_docs(ranked_txt: list[str], docs: list[Document]):
    docs = copy.deepcopy(docs)
    for doc_txt in ranked_txt:
        for doc in docs:
            if doc_txt["content"] == doc.page_content:
                yield doc
                docs.remove(doc)
                break
