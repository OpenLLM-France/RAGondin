import json
import asyncio
import numpy as np
import torch
from tqdm.asyncio import tqdm
from collections import Counter

from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name_or_path = "Alibaba-NLP/gte-multilingual-reranker-base"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name_or_path, trust_remote_code=True,
    torch_dtype=torch.float16
)
model.eval()


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
        file_name = entry["file"]
        question = entry["question"]
        relevant_docs = entry["relevant chunk"]

        response_chunks_id = [relevant_docs[i]["id"] for i in range(len(relevant_docs))]

        list_file_content = entry["reranker's input"]["file content"]
        list_chunks_id = entry["reranker's input"]["chunks id"]
        list_file_name = entry["reranker's input"]["file name"]
        
        with torch.no_grad():
            inputs = tokenizer([[question, list_file_content[i]] for i in range(len(list_file_content))], padding=True, truncation=True, return_tensors='pt', max_length=512)
            scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
            ordered_list = torch.argsort(scores, descending=True).tolist()

        list_chunks_id, list_file_name = apply_permutation(
            list(range(len(scores))), 
            list_chunks_id,
            list_file_name,
            ordered_list)
        
        list_chunks_id=list_chunks_id[:5]
        list_file_name=list_file_name[:5]

        # Get hit rate 
        hit_rate = 1 if file_name in list_file_name else 0
        # Get MRR
        Mrr = 1 / (list_file_name.index(file_name) + 1) if hit_rate == 1 else 0
        # Get precision
        counter = Counter(list_chunks_id)
        total = sum(counter[element] for element in (set(response_chunks_id) & set(list_chunks_id)))
        Precision = total / len(list_chunks_id) if len(list_chunks_id) > 0 else 0
        # Get recall 
        Recall = len(set(response_chunks_id) & set(list_chunks_id)) / len(response_chunks_id) if len(response_chunks_id) > 0 else 0
        return [hit_rate, Mrr, Precision, Recall]


async def main():
    out_file = "./output/question_and_chunks.json"

    json_file = open(out_file, "r", encoding="utf-8")
    list_questions = json.load(json_file)
    tasks = []
    for entry in list_questions:
        try:
            task = asyncio.create_task(retrieve_docs(entry))
            tasks.append(task)
        except Exception as e:
            print(f"Error occurred: {e}")
            print("Breaking the loop due to an exception.")
            break

    if not tasks:
        print("No evaluations were completed due to an error.")
        return
    
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
    
if __name__ == "__main__":
    asyncio.run(main())