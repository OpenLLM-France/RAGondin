from ragatouille import RAGPretrainedModel


class Reranker:
    def __init__(self, docs, num_docs_final: int = 5):
        self.reranker = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
        

    def rerank(self, query : str, docs: list[str], num_docs_final: int = 5) -> list[str]:
        relevant_docs = self.reranker.rerank(query, relevant_docs, k=num_docs_final)
        relevant_docs = [doc["content"] for doc in relevant_docs]
        return relevant_docs


### TODO : Compressor class