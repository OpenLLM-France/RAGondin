import json
import numpy as np
from loguru import logger
from tqdm import tqdm
from reranker import Reranker
import time

# load the model
evaluate_with_reranking = True

# model_name = "jinaai/jina-colbert-v2"
# reranker_type = "colbert"

model_name = "Alibaba-NLP/gte-multilingual-reranker-base"
reranker_type = "crossencoder"

reranker = Reranker(reranker_type=reranker_type, model_name=model_name)


def compute_hits(relevant_chunks, all_retrieved_chunks):
    hits = []
    retrieved_ids = [c["id"] for c in all_retrieved_chunks]
    for actual_node in relevant_chunks:
        hits.append(actual_node["id"] in retrieved_ids)
    return hits


def compute_inverted_ranks(relevant_chunks, all_retrieved_chunks):
    # see link: https://chatgpt.com/share/6813f998-2e88-8002-a472-6af2e9a64b61
    inverted_ranks = []
    retrieved_ids = [c["id"] for c in all_retrieved_chunks]

    ranks = []
    for actual_node in relevant_chunks:
        actual_id = actual_node["id"]
        try:
            rank = retrieved_ids.index(actual_id) + 1
            ranks.append(rank)
        except ValueError:
            logger.debug(f"ValueError: {actual_id} not found in retrieved_ids")

        if ranks:
            inverted_ranks.append(1 / min(ranks))
        else:
            inverted_ranks.append(0)

    return inverted_ranks


# load json file
path = "./output/retrieved_chunks_gte_Qwen2.json"
with open(path, "r", encoding="utf-8") as json_file:
    question_relevant_chunks = json.load(json_file)

HITS = []
INVERTED_RANKS = []

start = time.time()

for row in tqdm(question_relevant_chunks, desc="Computing metrics"):
    file_name = row["file"]  # get the file name
    question = row["question"]  # get the question
    relevant_chunks = row["relevant_chunks"]  # get the relevant chunks.

    all_retrieved_chunks = row["all_retrieved_chunks"]  # get the all retrieved chunks

    if evaluate_with_reranking:
        all_retrieved_chunks = reranker.rerank(
            query=question, chunks=all_retrieved_chunks, top_k=5
        )
    HITS.extend(compute_hits(relevant_chunks, all_retrieved_chunks))
    INVERTED_RANKS.extend(compute_inverted_ranks(relevant_chunks, all_retrieved_chunks))

end = time.time()

hit_rate = np.array(HITS).mean()
Mrr = np.array(INVERTED_RANKS).mean()

print(f"Hit Rate: {hit_rate}")
print(f"MRR: {Mrr}")
print(f"Time taken: {end - start} seconds")
