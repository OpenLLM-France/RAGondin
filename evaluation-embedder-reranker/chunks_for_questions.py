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
from tenacity import retry, stop_after_attempt, wait_fixed

settings = {
    "temperature": 0.2,
    "max_retries": 3,
    "timeout": 60,
    "model": "Qwen2.5-VL-7B-Instruct",
    "api_key": "sk-1234",
    "base_url": "https://chat.lucie.ovh.linagora.com/v1",
}

endpoint_base_url = "http://163.114.159.151:8087"

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


async def infer_chunk_relevancy(
    question: str, chunk: Document, llm_semaphore=asyncio.Semaphore(10)
):
    async with llm_semaphore:
        chunk_id = chunk.metadata.get("_id")
        content = chunk.page_content

        user_message = f"""Voici un document:\n\n{content}\nRéponds 'oui' ou 'non' s'il y a des informations concernant cette question: {question}"""

        messages = [
            SystemMessage(content=sys_pmpt),
            HumanMessage(content=user_message),
        ]
        chunk_info = {
            "id": chunk_id,
            "filename": chunk.metadata.get("filename", ""),
            "content": content,
        }

        try:
            output: DocRelevancy = await sllm.ainvoke(messages)
            return output.relevancy, chunk_info
        except Exception as e:
            logger.debug(f"Error in `infer_chunk_relevancy`: {e}")
            return "non", chunk_info


async def get_relevant_chunks(
    question, chunks: list[Document], llm_semaphore=asyncio.Semaphore(10)
):
    tasks = [
        infer_chunk_relevancy(
            question=question, chunk=chunk, llm_semaphore=llm_semaphore
        )
        for chunk in chunks
    ]
    tagged_chunks = await tqdm.gather(*tasks)

    relevant_chunks = []
    all_retrieved_chunks = []

    for tag, chunk_info in tagged_chunks:
        if tag == "oui":
            relevant_chunks.append(chunk_info)
        all_retrieved_chunks.append(chunk_info)

    return relevant_chunks, all_retrieved_chunks

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def fetch_chunk(chunk_url):
    async with httpx.AsyncClient(timeout=httpx.Timeout(10 * 60)) as client:
        response = await client.get(chunk_url)
        response.raise_for_status()  # raises exception for 4xx/5xx responses
        data = response.json()
        return Document(page_content=data["page_content"], metadata=data["metadata"])


async def __get_relevant_chunks(
    entry: dict,
    partition: str = "frwiki",
    top_k=8,
    question_semaphore=asyncio.Semaphore(10),
    llm_semaphore=asyncio.Semaphore(10),
):
    try:
        query = entry["question"]
        async with question_semaphore:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10 * 60), http2=True) as client:
                res = await client.get(
                    url=f"{endpoint_base_url}/search/partition/{partition}",
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
        chunks_tasks = [fetch_chunk(link) for link in chunk_links]
        chunks = await asyncio.gather(*chunks_tasks)

        relevant_chunks, all_retrieved_chunks = await get_relevant_chunks(
            query, chunks, llm_semaphore=llm_semaphore
        )
        entry["relevant chunks"] = relevant_chunks
        entry["all retrieved chunks"] = all_retrieved_chunks

        list_chunks_id = [chunk["id"] for chunk in all_retrieved_chunks]
        list_file_name = [chunk["filename"] for chunk in all_retrieved_chunks]
        list_file_content = [chunk["content"] for chunk in all_retrieved_chunks]
        entry["reranker's input"] = {
                "chunks id": list_chunks_id,
                "file name": list_file_name,
                "file content": list_file_content
            }
        return entry

    except Exception as e:
        logger.debug(f"Error fetching chunks: {e}")
        return None


async def main():
    input_file = "./output/generated_question.json"
    output_file = "./output/question_and_chunks.json"
    partition = "frwiki"
    top_k = 10
    question_semaphore = asyncio.Semaphore(
        10
    )  # number of questions to treat in parallel
    llm_semaphore = asyncio.Semaphore(15)  # number of parallel llm requests

    with open(input_file, "r", encoding="utf-8") as json_file:
        questions = json.load(json_file)

    tasks = [
        __get_relevant_chunks(
            entry,
            partition=partition,
            top_k=top_k,
            question_semaphore=question_semaphore,
            llm_semaphore=llm_semaphore,
        )
        for entry in questions
    ]

    question_relevant_chunks = await tqdm.gather(
        *tasks, desc="Getting relevant chunks for questions", total=len(tasks)
    )
    question_relevant_chunks = [
        entry for entry in question_relevant_chunks if entry is not None and entry["relevant chunks"]
    ]

    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(question_relevant_chunks, json_file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    asyncio.run(main())
