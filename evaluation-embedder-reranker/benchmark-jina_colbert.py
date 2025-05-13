import json
import asyncio
import numpy as np
import time
from tqdm.asyncio import tqdm
from collections import Counter

from ragatouille import RAGPretrainedModel


model = RAGPretrainedModel.from_pretrained("jinaai/jina-colbert-v2")
list_time = []

def apply_permutation(base_list, target_list_1, target_list_2, permuted_base):
    # Create a mapping from value to its new index in the permuted list
    index_map = {value: i for i, value in enumerate(permuted_base)}

    # Use the index map to reorder the target list
    reordered_target_1 = [None] * len(target_list_1)
    reordered_target_2 = [None] * len(target_list_1)
    for i, value in enumerate(base_list):
        new_index = index_map[value]
        reordered_target_1[new_index] = target_list_1[i]
        reordered_target_2[new_index] = target_list_2[i]

    return reordered_target_1, reordered_target_2

async def retrieve_docs(entry, semaphore=asyncio.Semaphore(5)):
    async with semaphore:
        question = entry["question"]
        all_chunks = entry["all_retrieved_chunks"]

        # Chunks response to the question
        all_relevant_chunks_id = entry["article_ids"].split(',') # List[Str]

        # Best chunk
        true_chunk_id = all_relevant_chunks_id[0]

        list_file_content = [all_chunks[i]["content"] for i in range(len(all_chunks))]
        list_chunks_id = [all_chunks[i]["article_id"] for i in range(len(all_chunks))]
        list_file_name = [all_chunks[i]["filename"] for i in range(len(all_chunks))]
        
        start = time.time()
        ranked_txt = await asyncio.to_thread(
            lambda: model.rerank(
                question, list_file_content, k=min(len(list_file_content), 5), bsize="auto"
            )
        )
        list_time.append(time.time() - start)

        list_index = [ranked_txt[i]['result_index'] for i in range(len(ranked_txt))]
        # list_chunks_id, list_file_name = apply_permutation(list_file_content, list_chunks_id, list_file_name, [ranked_txt[i]['content'] for i in range(len(ranked_txt))])
        list_chunks_id = [list_chunks_id[list_index[i]] for i in range(len(list_index))]
        list_file_name = [list_file_name[list_index[i]] for i in range(len(list_index))]
        # Get hit rate 
        hit_rate = 1 if true_chunk_id in list_chunks_id else 0
        # Get MRR
        Mrr = 1 / (list_chunks_id.index(true_chunk_id) + 1) if hit_rate == 1 else 0
        # Get precision
        counter = Counter(list_chunks_id)
        total = sum(counter[element] for element in (set(all_relevant_chunks_id) & set(list_chunks_id)))
        Precision = total / len(list_chunks_id) if len(list_chunks_id) > 0 else 0
        # Get recall 
        Recall = len(set(all_relevant_chunks_id) & set(list_chunks_id)) / len(all_relevant_chunks_id) if len(all_relevant_chunks_id) > 0 else 0
        return [hit_rate, Mrr, Precision, Recall]



async def main():
    out_file = "./output/retrieved_chunks_OrdalieTech.json"

    json_file = open(out_file, "r", encoding="utf-8")
    list_questions = json.load(json_file)
    tasks = []
    for entry in list_questions:
        task = asyncio.create_task(retrieve_docs(entry))
        tasks.append(task)
        
    evaluation_questions = await tqdm.gather(*tasks)  # list[[hit rate, MRR, Precision, Recall]]
    table_evaluation = np.array(evaluation_questions)

    hit_rate = np.mean(table_evaluation[:, 0])
    Mrr = np.mean(table_evaluation[:, 1])
    Precision = np.mean(table_evaluation[:, 2])
    Recall = np.mean(table_evaluation[:, 3])
    print(f"Hit Rate: {hit_rate}")
    print(f"MRR: {Mrr}")
    print(f"Precision: {Precision}")
    print(f"Recall: {Recall}")
    print(f"Time per query: {sum(list_time) / len(list_time)}")

if __name__ == "__main__":
    asyncio.run(main())