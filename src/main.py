# Import necessary modules and classes
from src.evaluation import Evaluator
from src.pipeline import RAG
from vector_store import Qdrant_Connector
from embeddings import Embeddings
from llm import LLM
from chunker import Chunker
from chunker import Docs
from reranker import Reranker
from prompt import Prompt
from acess_token import access_token
from huggingface_hub import InferenceClient

# Main execution
if __name__ == '__main__':

    # Define the model repository ID and model kwargs
    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    model_kwargs = {"device": "mps"}

    # Initialize embeddings
    emb = Embeddings(model_kwargs=model_kwargs)

    # Initialize Qdrant connector
    connector = Qdrant_Connector(host=None, port=0, embeddings=emb.get_embeddings(), collection_name="my_collection")

    print("Loading documents...")
    # Load documents
    docs = Docs()
    dir_path = "../experiments/test_data"
    docs.load(dir_path)

    print("Chunking...")
    # Chunk documents
    chunker: Chunker = Chunker(chunk_size=1000, chunk_overlap=200)
    docs.chunk(chunker)

    # Initialize LLM client
    llm_client = InferenceClient(
        model=repo_id,
        timeout=120,
        token=access_token
    )
    llm = LLM(llm_client)

    # Initialize evaluator
    Eval: Evaluator = Evaluator(save_path="eval", llm=llm, docs=docs)
    print("Generating questions...")

    # Generate questions
    Eval.generate_questions(5)

    print(Eval.questions)

    print("Generating critique for each QA couple...")
    # Generate critique for each QA couple
    Eval.critique_questions()

    print("Filtering questions...")
    # Filter questions
    print(Eval.questions)
    Eval.filter_questions(groundness_trsh=3, relevance_trsh=1, standalone_trsh=3)

    print(Eval.filtered_questions)

    # Build index
    connector.build_index(chuncked_docs=docs.get_chunks())

    # Initialize prompt and reranker
    prompt = Prompt()
    reranker = Reranker()

    # Initialize RAG
    rag = RAG(llm, connector, reranker, prompt)

    # Define question
    question = "Qu'elle sont les 5 phases du jeu MESBG?"

    print("Answering question...")
    # Answer question
    print(rag.run(question))

    print("Evaluating...")
    # Evaluate
    print(Eval.evaluate(rag)[0])