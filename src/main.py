from src.evaluation import Evaluator
from vector_store import Qdrant_Connector
from embeddings import Embeddings
from llm import LLM
from chunker import Chunker
from chunker import Docs
from reranker import Reranker
from prompt import Prompt
from acess_token import access_token
from huggingface_hub import InferenceClient

if __name__ == '__main__':

    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    model_kwargs = {"device": "mps"}
    emb = Embeddings(model_kwargs=model_kwargs)
    connector = Qdrant_Connector(host=None, port=0, embeddings=emb.get_embeddings(), collection_name="my_collection")

    print("Loading documents...")
    docs = Docs()
    dir_path = "../experiments/test_data"
    docs.load(dir_path)

    print("Chunking...")
    chunker: Chunker = Chunker(chunk_size=1000, chunk_overlap=200)
    docs.chunk(chunker)

    llm_client = InferenceClient(
        model=repo_id,
        timeout=120,
        token=access_token
    )
    llm = LLM(llm_client)

    Eval: Evaluator = Evaluator(save_path="eval", llm=llm, docs=docs)
    print("Generating questions...")

    Eval.generate_questions(5)

    print(Eval.questions)

    print("Generating critique for each QA couple...")

    Eval.critique_questions()

    print("Filtering questions...")

    print(Eval.questions)

    Eval.filter_questions(groundness_trsh=3, relevance_trsh=1, standalone_trsh=3)

    print(Eval.filtered_questions)

    connector.build_index(chuncked_docs=docs.get_chunks())
    question = "Qu'elle sont les 5 phases du jeu MESBG?"
    retrived_chunks = connector.similarity_search(query="Bonjour, qui est le plus grand roi de France?", top_k=10)
    retrived_chunks_txt = [chunk.page_content for chunk in retrived_chunks]

    print("Reranking...")
    reranker = Reranker()
    reranked_docs_txt = reranker.rerank(query=question, docs=retrived_chunks_txt, k=5)

    prompt = Prompt()
    prompt_txt = prompt.get_prompt(docs=reranked_docs_txt, question=question)

    print(llm.generate_output(prompt_txt))



