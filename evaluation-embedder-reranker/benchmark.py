import json
import asyncio
import numpy as np
import httpx
from tqdm.asyncio import tqdm
from collections import Counter
from sentence_transformers import CrossEncoder
from langchain_core.documents.base import Document


model_name = "jinaai/jina-reranker-v2-base-multilingual"
model = CrossEncoder(
    model_name,
    automodel_args={"torch_dtype": "auto"},
    trust_remote_code=True,
)


async def get_file_name_and_content(url, semaphore=asyncio.Semaphore(5)):
    async with semaphore:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(4 * 60.0), http2=True
        ) as client:
            response = await client.get(url=url)
            response.raise_for_status()

            data: dict = response.json()
            file_name = data.get("metadata", {}).get("metadata", {})
            page_content = data.get("page_content", "")

            return file_name, page_content


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


async def retrieve_docs(
    entry,
    api_base_url: str,
    partition: str = "frwiki",
    top_k: int = 10,
    semaphore=asyncio.Semaphore(5),
):
    async with semaphore:
        file_name = entry["file"]
        question = entry["question"]
        relevant_chunks = entry["relevant_chunk"]

        response_chunks_id = [chunk["id"] for chunk in relevant_chunks]

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(10 * 60.0), http2=True
        ) as client:
            resp = await client.get(
                url=f"{api_base_url}/search/partition/{partition}",  # Replace 'frwiki' with the desired partition
                params={"text": question, "top_k": top_k},
            )
            resp.raise_for_status()  # Raise an exception for HTTP errors
            data: dict = resp.json()

            # Extract the documents
            chunks = data.get("documents", [])
            chunk_links = [doc["link"] for doc in chunks]

            # Extract the content and metadata
            list_chunks_id = [
                int(chunk_links[i].split("extract/")[1])
                for i in range(len(chunk_links))
            ]
            list_file_name_and_content = [
                await get_file_name_and_content(chunk_links[i])
                for i in range(len(chunk_links))
            ]

            list_file_name = [
                list_file_name_and_content[i][0]
                for i in range(len(list_file_name_and_content))
            ]
            list_file_content = [
                list_file_name_and_content[i][1]
                for i in range(len(list_file_name_and_content))
            ]

            rankings = model.rank(
                question,
                list_file_content,
                return_documents=True,
                convert_to_tensor=True,
            )
            list_index = [rankings[i]["corpus_id"] for i in range(len(rankings))]
            # list_chunks_id, list_file_name = apply_permutation(list_file_content, list_chunks_id, list_file_name, [ranked_txt[i]['content'] for i in range(len(ranked_txt))])
            list_chunks_id = [
                list_chunks_id[list_index[i]] for i in range(len(list_index))
            ]
            list_file_name = [
                list_file_name[list_index[i]] for i in range(len(list_index))
            ]
            # Get hit rate
            hit_rate = 1 if file_name in list_file_name else 0
            # Get MRR
            Mrr = 1 / (list_file_name.index(file_name) + 1) if hit_rate == 1 else 0
            # Get precision
            counter = Counter(list_chunks_id)
            total = sum(
                counter[element]
                for element in (set(response_chunks_id) & set(list_chunks_id))
            )
            Precision = total / len(list_chunks_id) if len(list_chunks_id) > 0 else 0
            # Get recall
            Recall = (
                len(set(response_chunks_id) & set(list_chunks_id))
                / len(response_chunks_id)
                if len(response_chunks_id) > 0
                else 0
            )
            return [hit_rate, Mrr, Precision, Recall]


async def main():
    output_file = "./output/question_relevant_chunks.json"
    endpoint_base_url = f"http://163.114.159.151:8080"
    partition = "frwiki"

    with open(output_file, "r", encoding="utf-8") as json_file:
        question_relevant_chunks = json.load(json_file)

    tasks = []
    for entry in question_relevant_chunks:
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

    evaluation_questions = await tqdm.gather(
        *tasks
    )  # list[[hit rate, MRR, Precision, Recall]]
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
