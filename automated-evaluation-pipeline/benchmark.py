import os
import asyncio
import httpx
import json
import numpy as np
from loguru import logger
from openai import AsyncOpenAI

from tqdm.asyncio import tqdm

def evaluate(list_ids: list[int], list_reference: list[int]) -> float:
    ...

async def source_score_per_question(
    chunk_id_reference: list[int],
    chunk_id_llm: list[int],
    sempahore: asyncio.Semaphore = None,
):
    async with sempahore:
        source_evaluation_score = evaluate(retrieved_chunks_ids, chunk_id)
        return source_evaluation_score


# async def response_score_per_question(
#     query: str,
#     llm_answer: str,
#     partition: str,
#     ragondin_api_base_url: str = None,
#     sempahore: asyncio.Semaphore = None,
# ):
#     async with semaphore:
#         retries = 3
#         for attempt in range(retries):
#             try:
#                 async with httpx.AsyncClient(timeout=httpx.Timeout(400)) as client:
#                     response = await client.get(
#                         url=f"{ragondin_api_base_url}/v1/chat/completions",

#                     )

async def retrieve_response_and_docs(
    query: str,
    partition: str,
    ragondin_base_url: str,
    semaphore=asyncio.Semaphore(10)
):
    async with semaphore:
        base_url = f"{ragondin_base_url}/v1"
        auth_key = 'sk-1234'
        client = AsyncOpenAI(api_key=auth_key, base_url=base_url)

        settings = {
            "model": f"ragondin-{partition}",
            "messages": [{
                "role": "user",
                "content": query,
            }],
            "temperature": 0.7,
            "stream": False,
            "frequency_penalty": 0.4,
        }

        try:
            res = await client.chat.completions.create(
                **settings
            )
            response_llm = res.choices[0].message.content
            list_source_chunk_ids = [item['id'] for item in json.loads(res.extra)['sources']]

            return response_llm, list_source_chunk_ids
        except Exception as e:
            logger.debug(f"Error fetching chunks and response: {e}")

async def main():
    data_file = open("./dataset.json", "r", encoding="utf-8")
    list_input = json.load(data_file)

    num_port = os.environ.get("APP_PORT")
    num_host = "163.114.159.68"  # "localhost"
    ragondin_api_base_url = f"http://{num_host}:{num_port}"
    partition = "benchmark"

    tasks = [
        retrieve_response_and_docs(
            query=input["question"],
            partition=partition,
            ragondin_base_url=ragondin_api_base_url,
            semaphore=asyncio.Semaphore(10),
        )
        for input in list_input
    ]

    ragondin_retrieval = await tqdm.gather(*tasks, desc="Fetching")
    responses_llm, metadata_llm = map(list, zip(*ragondin_retrieval))

    # print(f"Source evaluation - nDCG: {np.array(score).mean()}")

if __name__ == '__main__':
    asyncio.run(main())