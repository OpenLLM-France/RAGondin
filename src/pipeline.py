import configparser

from src.chunker import Docs, Chunker
from src.llm import LLM
from src.prompt import Prompt
from src.reranker import Reranker
from vector_store import Qdrant_Connector
from embeddings import Embeddings
from flask import Flask, request
from openai import OpenAI

app = Flask(__name__)


@app.route('/inference', methods=['POST'])
def inference():
    try:
        query = request.form.get('query')
        print(request)
        config = configparser.ConfigParser()
        config.read("src/config.ini")
        host = config.get("VECTOR_DB", "host")
        port = config.get("VECTOR_DB", "port")
        collection = config.get("VECTOR_DB", "collection")
        model_type = config.get("EMBEDDINGS", "model_type")
        model_name = config.get("EMBEDDINGS", "model_name")
        model_kwargs = dict(config.items("EMBEDDINGS.MODEL_KWARGS"))
        encode_kwargs = dict(config.items("EMBEDDINGS.ENCODE_KWARGS"))

        embeddings = Embeddings(model_type, model_name, model_kwargs, encode_kwargs).get_embeddings()

        connector = Qdrant_Connector(host, port, collection, embeddings)

        k = 5
        print("Results:")
        docs = connector.similarity_search_with_score(query, 5)
        for i in docs:
            doc, score = i
            print({"score": score, "content": doc.page_content, "metadata": doc.metadata})

        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",
        )

        # TODO: create dynamic content for a query and retrieved chunks
        content = ""

        completion = client.chat.completions.create(
            model="TheBloke/Vigostral-7B-Chat-AWQ",
            messages=[
                {"role": "user", "content": content}
            ],
        )

        return completion.choices[0].message
    except Exception as e:
        print(e)


class RAG:
    """
    This class represents a RAG (Retrieval-Augmented Generation) model.

    It uses a Qdrant_Connector to retrieve relevant chunks of data, a Reranker to rerank these chunks,
    a Prompt to generate a prompt from the reranked chunks, and a LLM (Language Model) to generate an output
    from the prompt.

    Attributes:
        llm (LLM): An instance of the LLM class.
        connector (Qdrant_Connector): An instance of the Qdrant_Connector class.
        reranker (Reranker): An instance of the Reranker class.
        prompt (Prompt): An instance of the Prompt class.
    """
    def __init__(self, llm: LLM, connector: Qdrant_Connector, reranker: Reranker, prompt: Prompt):
        """
        The constructor for the RAG class.

        Args:
            llm (LLM): An instance of the LLM class.
            connector (Qdrant_Connector): An instance of the Qdrant_Connector class.
            reranker (Reranker): An instance of the Reranker class.
            prompt (Prompt): An instance of the Prompt class.
        """
        self.llm = llm
        self.connector = connector
        self.reranker = reranker
        self.prompt = prompt

    def run(self, question: str, top_k: int = 10, top_k_rerank: int = 5) -> str:
        """
        This method retrieves relevant chunks of data, reranks them, generates a prompt from the reranked chunks,
        and generates an output from the prompt.

        Args:
            question (str): The question to be answered.
            top_k (int): The number of top similar vectors to return from the similarity search.
            top_k_rerank (int): The number of top documents to return after reranking.

        Returns:
            str: The generated output from the LLM.
        """
        retrieved_chunks = self.connector.similarity_search(query=question, top_k=top_k)
        retrieved_chunks_txt = [chunk.page_content for chunk in retrieved_chunks]
        reranked_docs_txt = self.reranker.rerank(query=question, docs=retrieved_chunks_txt, k=top_k_rerank)
        prompt_txt = self.prompt.get_prompt(docs=reranked_docs_txt, question=question)
        return self.llm.run(prompt_txt)
