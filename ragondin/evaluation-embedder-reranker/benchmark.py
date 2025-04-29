import json
import asyncio
import numpy as np
import httpx
from tqdm.asyncio import tqdm
from collections import Counter

async def get_file_name(url, semaphore=asyncio.Semaphore(2)):
    async with semaphore:
        async with httpx.AsyncClient(timeout=httpx.Timeout(4 * 60.0), http2=True
        ) as client:
            response = await client.get(
                url=url
            )
            response.raise_for_status()  # Raise an exception for HTTP errors

            response_data = response.json()
            metadata = response_data.get("metadata", [])
            file_name = metadata["filename"]
            return file_name

async def retrieve_docs(entry, semaphore=asyncio.Semaphore(2)):
    async with semaphore:
        file_name = entry["file"]
        question = entry["question"]
        relevant_docs = entry["relevant chunk"]

        response_chunks_id = [relevant_docs[i]["id"] for i in range(len(relevant_docs))]

        params = {
            "text": question,  # The text to search semantically
            "top_k": 5,        # Number of top results to return (default is 5)
        }
        base_url = "http://163.114.159.151:8087"

        # Send the request using httpx
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(4 * 60.0), http2=True
        ) as client:
            response = await client.get(
                url=f"{base_url}/search/partition/frwiki",  # Replace 'frwiki' with the desired partition
                params=params,
            )
            response.raise_for_status()  # Raise an exception for HTTP errors

            response_data = response.json()

            # Extract the documents
            documents = response_data.get("documents", [])
            document_links = [doc["link"] for doc in documents]

            # Extract the content and metadata
            list_chunks_id = [int(document_links[i].split('extract/')[1]) for i in range(len(document_links))]
            list_file_name = [await get_file_name(document_links[i]) for i in range(len(document_links))]

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
    out_file = "./evaluation-embedder-reranker/complete_dataset_v2.json"

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
    
if __name__ == "__main__":
    asyncio.run(main())