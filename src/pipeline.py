import configparser
from .chunker import Docs, RecursiveSplitter
from .llm import LLM
from .prompt import Prompt
from .reranker import Reranker
from .retriever import SingleRetriever
from .vector_store import Qdrant_Connector
from .embeddings import HFEmbedder
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

        embeddings = HFEmbedder(model_type, model_name, model_kwargs, encode_kwargs).get_embeddings()

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
    def __init__(self, llm: LLM, connector: Qdrant_Connector, retriever : SingleRetriever, prompt: Prompt, reranker: Reranker= None):
        """
        The constructor for the RAG class.

        Args:
            llm (LLM): An instance of the LLM class.
            connector (Qdrant_Connector): An instance of the Qdrant_Connector class.
            retriever (Retriever): An instance of the Retriever class.
            prompt (Prompt): An instance of the Prompt class.
            reranker (Reranker): An instance of the Reranker class.
        """
        self.llm = llm
        self.connector = connector
        self.prompt = prompt
        self.retriever = retriever
        self.reranker = reranker

    def run(self, question: str) -> str:
        """
        This method retrieves relevant chunks of data, reranks them, generates a prompt from the reranked chunks,
        and generates an output from the prompt.

        Args:
            question (str): The question to be answered.
        Returns:
            str: The generated output from the LLM.
        """
        docs_txt = self.retriever.retrieve(question, self.connector)
        if self.reranker is not None:
            docs_txt = self.reranker.rerank(question=question, docs=docs_txt)
        prompt_txt = self.prompt.get_prompt(docs=docs_txt, question=question)
        return self.llm.run(prompt_txt)
