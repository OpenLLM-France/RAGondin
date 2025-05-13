# Documents that has informations about the query

import json
import csv
import asyncio
import itertools
from tqdm.asyncio import tqdm

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents.base import Document
from langchain_openai import ChatOpenAI
import httpx
from pydantic import BaseModel, Field
from typing import Literal
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

import os

settings = {
    "temperature": 0.2,
    "max_retries": 3,
    "timeout": 60,
    "model": "Qwen2.5-VL-7B-Instruct",
    "api_key": os.environ.get("VLM_API_KEY"),
    "base_url": os.environ.get("VLM_BASE_URL"),
}

sys_pmpt = """Tu es un expert en évaluation de la pertinence des documents. Ta tâche consiste à évaluer les documents par rapport à une question posée par l'utilisateur"""


class DocRelevancy(BaseModel):
    relevancy: Literal["oui", "non"] = Field(
        description="Indique si le document contient des informations pertinentes par rapport à la question posée"
    )


sllm = (
    ChatOpenAI(**settings)
    .with_structured_output(DocRelevancy)
    .with_retry(stop_after_attempt=2)
)


async def fetch_chunk_data(chunk_url) -> Document:
    async with httpx.AsyncClient(timeout=httpx.Timeout(4 * 10)) as client:
        response = await client.get(chunk_url)
        response.raise_for_status()  # raises exception for 4xx/5xx responses
        data = response.json()
        metadata = data.get("metadata", {})
        return {
            "id": metadata.get("_id"),
            "filename": metadata.get("filename"),
            "article_id": metadata.get("filename").split('.')[0],
            "content": data.get("page_content"),
        }


async def __get_relevant_chunks(
    query: str,
    partition: str = "articles",
    top_k: int = 8,
    ragondin_api_base_url: str = None,
    sempahore: asyncio.Semaphore = None,
    llm_semaphore: asyncio.Semaphore = None,
    add_chunk_relevancy: bool = False,
):
    async with sempahore:
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
            chunk_links = [doc["link"] for doc in data.get("documents", [])]
            # Extract the content and metadata
            chunks_tasks = [fetch_chunk_data(link) for link in chunk_links]
            chunks = await asyncio.gather(*chunks_tasks)
            return chunks
        except Exception as e:
            logger.debug(f"Error fetching chunks: {e}")
            return None 


async def main():
    input_file = "./output/questions.csv"
    output_file = "./output/retrieved_chunks_OrdalieTech.json"

    partition = "articles"
    top_k = 10

    llm_semaphore = asyncio.Semaphore(20)
    semaphore = asyncio.Semaphore(10)

    ragondin_api_base_url = "http://163.114.159.68:8080"

    # load json file
    csv_file = open(input_file, "r", encoding="utf-8")
    question_answer = csv.DictReader(csv_file)

    tasks = [
        __get_relevant_chunks(
            query=entry["question"],
            partition=partition,
            top_k=top_k,
            ragondin_api_base_url=ragondin_api_base_url,
            sempahore=semaphore,
            llm_semaphore=llm_semaphore,
            add_chunk_relevancy=True,
        )
        for i, entry in enumerate(itertools.islice(question_answer, 3)) 
    ]

    data = await tqdm.gather(*tasks, desc="Generating data for evaluation")

    data2 = []
    for entry, chunks in zip(question_answer, data):
        entry["all_retrieved_chunks"] = chunks
        data2.append(entry)

    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(data2, json_file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    asyncio.run(main())
