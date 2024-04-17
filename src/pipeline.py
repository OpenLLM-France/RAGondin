import configparser
from vector_store import Qdrant_Connector
from embeddings import Embeddings
from llm import LLM
from flask import Flask, request
from openai import OpenAI
from chunker import Chunker
from chunker import Docs
from reranker import Reranker
from prompt import Prompt
access_token = "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"

app=Flask(__name__)
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
        
        
        #TODO: create dynamic content for a query and retrieved chunks
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

if __name__ == '__main__':
    #app.run(host='0.0.0.0', debug=True, port=80)
    from huggingface_hub import InferenceClient

    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    model_kwargs = {"device": "mps"}
    emb = Embeddings(model_kwargs= model_kwargs)
    connector = Qdrant_Connector(host=None,port=0, embeddings=emb.get_embeddings(),collection_name="my_collection")

    print("Loading documents...")
    docs = Docs()
    dir_path = "../experiments/test_data"
    docs.load(dir_path)

    print("Chunking...")
    chunker = Chunker(chunk_size=1000, chunk_overlap=200)
    docs.chunk(chunker)
    connector.build_index(chuncked_docs=docs.get_chunks())
    question = "Qu'elle sont les 5 phases du jeu MESBG?"
    retrived_chunks = connector.similarity_search(query="Bonjour, qui est le plus grand roi de France?", top_k=10)
    retrived_chunks_txt = [chunk.page_content for chunk in retrived_chunks]

    print("Reranking...")
    reranker = Reranker()
    reranked_docs_txt = reranker.rerank(query=question,docs=retrived_chunks_txt,k=5)

    prompt = Prompt()
    prompt_txt = prompt.get_prompt(docs = reranked_docs_txt, question=question)

    llm_client = InferenceClient(
        model=repo_id,
        timeout=120,
        token=access_token
    )
    llm = LLM(llm_client)
    print(llm.generate_output(prompt_txt))



