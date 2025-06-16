import os
import asyncio
import httpx
import json
from loguru import logger

from tqdm.asyncio import tqdm

async def __get_relevant_chunks(
    query: str,
    chunk_source: list(int),
    partition: str = "all",
    top_k: int = 8,
    ragondin_api_base_url: str = None,
    sempahore: asyncio.Semaphore = None,
):
    async with sempahore:
        retries = 3  # Number of retries
        delay = 1  # Delay in seconds between retries
        for attempt in range(retries):
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(4 * 60)) as client:
                    res = await client.get(
                        url=f"{ragondin_api_base_url}/search/partition/{partition}",
                        params={
                            "text": query,
                            "top_k": top_k,
                        },
                    )
                    res.raise_for_status()
                    data: dict = res.json()
                # Extract the documents
                retrieved_chunks = [doc for doc in data.get("documents", [])]
                # Extract the content and metadata
                chunks_tasks = [fetch_chunk_data(chunk) for chunk in retrieved_chunks]
                chunks = await asyncio.gather(*chunks_tasks)
                return chunks
            except Exception as e:
                logger.debug(f"Attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(delay)  # Wait before retrying
                else:
                    logger.debug(f"Error fetching chunks after {retries} attempts: {e}")
                    return None

async def main():
    data_file = open("./dataset.json", "r", encoding="utf-8")
    list_input = json.load(data_file)

    num_port = os.environ.get("APP_PORT")
    num_host = "163.114.159.68"  # "localhost"
    ragondin_api_base_url = f"http://{num_host}:{num_port}"
    partition = "benchmark"

    tasks = [
        __get_relevant_chunks(
            query=input["question"],
            chunk_source=input["chunk ids"],
            partition=partition,
            top_k=5,
            ragondin_api_base_url=ragondin_api_base_url,
            sempahore=asyncio.Semaphore(10),
        )
        for input in list_input
    ]


