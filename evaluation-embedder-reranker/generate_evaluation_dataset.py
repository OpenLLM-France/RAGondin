# Documents that has informations about the query

import json
import asyncio
from tqdm.asyncio import tqdm

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents.base import Document
from langchain_openai import ChatOpenAI
import httpx
from pydantic import BaseModel, Field
from typing import Literal
from loguru import logger
import os
from dotenv import load_dotenv

load_dotenv()


settings = {
    "temperature": 0.2,
    "max_retries": 3,
    "timeout": 60,
    "model": "Qwen2.5-VL-7B-Instruct",
    "api_key": os.environ.get("VLM_API_KEY"),
    "base_url": os.environ.get("VLM_BASE_URL"),
}

sys_prompt = """Tu es un expert en génération de questions. Ta tâche consiste à générer une question pertinente qui pourrait être posée par un utilisateur et qui pourrait être répondue par le document donné"""
sllm = ChatOpenAI(**settings).with_retry(stop_after_attempt=2)


async def sample_chunk_links_for_evaluation(
    partition: str = "frwiki",
    n_chunks: int = 500,
    ragondin_api_base_url: str = None,
    seed: int | None = None,
):
    async with httpx.AsyncClient(timeout=httpx.Timeout(4 * 60)) as client:
        res = await client.get(
            url=f"{ragondin_api_base_url}/partition/{partition}/sample",
            params={
                "n_ids": n_chunks,
                "seed": seed,
            },
        )
        res.raise_for_status()
        data: dict = res.json()
        return data.get("chunk_urls", [])


async def fetch_chunk_data(chunk_url) -> Document:
    async with httpx.AsyncClient(timeout=httpx.Timeout(4 * 10)) as client:
        response = await client.get(chunk_url)
        response.raise_for_status()  # raises exception for 4xx/5xx responses
        data = response.json()
        metadata = data.get("metadata", {})
        return {
            "id": metadata.get("_id"),
            "filename": metadata.get("filename"),
            "content": data.get("page_content"),
        }


async def generate_question(chunk, llm_semaphore: asyncio.Semaphore = None):
    user_message = (
        """Voici le document"""
        f"""# Document\n\n{chunk["content"]}\n"""
        """Génère une question qui pourrait être posée par un utilisateur et qui pourrait être répondue par ce document"""
    )
    messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=user_message),
    ]
    try:
        async with llm_semaphore:
            question = await sllm.ainvoke(
                messages
            )  # Use the LLM to generate a question
            return question.content
    except Exception as e:
        logger.debug(f"Error in `generate_question`: {e}")


async def contruct_evaluation_data(
    chunk_url: str,
    llm_semaphore: asyncio.Semaphore = None,
    semaphore: asyncio.Semaphore = None,
):
    async with semaphore:
        chunk = await fetch_chunk_data(chunk_url)
        question = await generate_question(chunk, llm_semaphore=llm_semaphore)
        output = {
            "question": question,
            "true_relevant_chunk": [chunk],
        }
        return output


async def main():
    output_file = "./output/evaluation_data.json"
    partition = "frwiki"
    seed = 2025
    llm_semaphore = asyncio.Semaphore(20)
    semaphore = asyncio.Semaphore(10)

    ragondin_api_base_url = "http://163.114.159.151:8087"
    n_chunks_for_evaluation = 300

    chunk_urls = await sample_chunk_links_for_evaluation(
        partition=partition,
        n_chunks=n_chunks_for_evaluation,
        ragondin_api_base_url=ragondin_api_base_url,
        seed=seed,
    )
    tasks = [
        contruct_evaluation_data(
            chunk_url["link"], llm_semaphore=llm_semaphore, semaphore=semaphore
        )
        for chunk_url in chunk_urls
    ]
    data = await tqdm.gather(*tasks, desc="Generating data for evaluation")
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    asyncio.run(main())
