# Import necessary modules and classes
from src.evaluation import Evaluator
from src.pipeline import RAG
from src.retriever import SingleRetriever, MultiQueryRetriever, HybridRetriever
from src.vector_store import Qdrant_Connector
from src.embeddings import HFEmbedder
from src.llm import LLM
from src.chunker import RecursiveSplitter, Docs
from src.reranker import Reranker
from src.prompt import Prompt
from huggingface_hub import InferenceClient
from openai import OpenAI
import pprint as pp

# Main execution
if __name__ == '__main__':
    # Define the model repository ID and model kwargs
    # repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    # access_token = "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"
    model_kwargs = {"device": "cpu"}

    # Initialize embeddings
    emb = HFEmbedder(model_kwargs=model_kwargs, model_name="thenlper/gte-base")

    # Initialize Qdrant connector"
    print("Qdrant connection...")
    connector = Qdrant_Connector(host=None, port=0, embeddings=emb.get_embeddings(), collection_name = "documents")

    # Loading documents and Chunking...
    docs = Docs()
    dir_path = "./experiments/test_data" # "/home/linagora/workspace/Ahmath/RAG/RAGondin-1/experiments/test_data"

    # Load documents
    docs.load(dir_path)

    # Chunk documents
    print("Chunking...")
    chunker: RecursiveSplitter = RecursiveSplitter(chunk_size=1000, chunk_overlap=100)
    docs.chunk(chunker)

    # Initialize LLM client
    print("Initialize LLM client...")
    llm_client = OpenAI(
        base_url="https://chat.ai.linagora.exaion.com/v1/",
        api_key="sk-7Gqg14u-mGlX-egix20lgg",
        timeout=60
    )

    llm = LLM(llm_client)
    llm_multi_queries = LLM(llm_client)

    # # Initialize evaluator
    # print("\nInitialize evaluator...")
    # Eval: Evaluator = Evaluator(llm=llm_multi_queries)

    # # Generate questions
    # print("\nGenerate questions...")
    # Eval.generate_questions(6, docs=docs)
    # pp.pprint(Eval.questions)

    # # Generate critique for each QA couple
    # print("\nGenerating critique for each QA couple...")
    # Eval.critique_questions()

    # # Filter questions
    # print("\nFiltering questions...")
    # Eval.filter_questions(groundness_trsh=1, relevance_trsh=1, standalone_trsh=1)
    # pp.pprint(Eval.filtered_questions)

    # Build index
    connector.add_documents(chuncked_docs=docs.get_chunks())

    # Initialize prompt and reranker
    reranker = Reranker()
    
    k = 5
    retriever = SingleRetriever(params={"top_k": k}) # basic one
    
    retriever_multi = MultiQueryRetriever(
        params={"top_k": k}, 
        llm=llm_multi_queries,
        prompt_multi_queries=Prompt(type_template='multi_query'),
        k_multi_queries=10
    )
    retriever_final = HybridRetriever(params={"top_k": k}, retrievers=[(0.5, retriever_multi), (0.5, retriever_multi)])

    # Initialize RAG
    prompt = Prompt(type_template='basic')
    rag = RAG(llm=llm, connector=connector, retriever=retriever, prompt=prompt, reranker=reranker)

    
    # Answer question
    # Define question
    while True:
        question = input("Ask your question: ") # Comment la diversité s'intégre dans la RH des entreprises?
        print("Answering question...")
        print(rag.run(question))

        
    # Evaluate
    # print("Evaluating...")
    # eval_dataset = Eval.evaluate(rag)
    # print(eval_dataset[:6])
