import configparser
from vector_store import Qdrant_Connector
from embeddings import Embeddings
from LLM import LLM
from flask import Flask, request
from openai import OpenAI
from reranker import Reranker

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

    repo_id = "mistralai/Mixtral-7B-Instruct-v0.1"

    llm_client = InferenceClient(
        model=repo_id,
        timeout=120,
    )
    llm = LLM(llm_client)
    print(llm.generate_output('Bonjour, qui est le plus grand roi de France?'))