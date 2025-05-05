# Documents that has informations about the query

import json
import os
import asyncio
from tqdm.asyncio import tqdm
from openai import OpenAI
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
import httpx
from pydantic import BaseModel, Field
from typing import Literal

settings = {
    "temperature": 0.2,
    "max_retries": 3,
    "timeout": 60,
    "model": "Qwen2.5-VL-7B-Instruct",  # "Qwen2.5-VL-7B-Instruct",
    "api_key": "sk-qEoN3RedKamDQNuZJH7F9Q",  # 'sk-Xt9LmQvJw2ZaBf_T7NpYdC',
    "base_url": "https://chat.lucie.ovh.linagora.com/v1",  # http://91.134.49.99:8000/v1/",
}

enpoint_base_url = f"http://163.114.159.151:8087"

sys_pmpt = """\nTu es un expert en évaluation de la pertinence des documents. Ta tâche consiste à évaluer les documents par rapport à une question posée par l'utilisateur"""


class DocRelevancy(BaseModel):
    relevancy: Literal["yes", "no"] = Field(
        description="Indicate if the document contains information related to the question."
    )


sllm = (
    ChatOpenAI(**settings)
    .with_structured_output(DocRelevancy)
    .with_retry(stop_after_attempt=2)
)


async def get_relevant_docs(question, chunks):
    async def get_doc(chunk, question, semaphore=asyncio.Semaphore(10)):
        async with semaphore:
            chunk_id = chunk.metadata.get("_id")
            content = chunk.page_content.split("=> chunk: ")[1]
            message = HumanMessage(
                content=f"Selon ce document:\n\n{content}\nRéponds 'oui' ou 'non' s'il y a des informations concernant cette question: {question}"
            )
            output: DocRelevancy = await sllm.ainvoke([message])
            print(output)

            if output.relevancy == "yes":
                return {"id": chunk_id, "content": content}

    tasks = [get_doc(chunk, question) for chunk in chunks]
    list_relevant_chunk = await tqdm.gather(*tasks)
    list_result = [chunk for chunk in list_relevant_chunk if chunk is not None]
    return list_result


async def __fetch_chunk(chunk_url):
    async with httpx.AsyncClient(timeout=httpx.Timeout(4 * 10)) as client:
        response = await client.get(chunk_url)
        response.raise_for_status()  # raises exception for 4xx/5xx responses
        data = response.json()
        return Document(page_content=data["page_content"], metadata=data["metadata"])


async def __get_relevant_chunks(
    entry: dict, partition: str = "frwiki", top_k=10, semaphore=asyncio.Semaphore(10)
):
    async with semaphore:
        query = entry["question"]
        search_endpoint = f"{enpoint_base_url}/search/partition/{partition}?text={query}&top_k={top_k}"
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(4 * 60)) as client:
                # Send the request using httpx
                response = await client.get(search_endpoint)
                response.raise_for_status()
                response_data = response.json()

                # Extract the documents
                documents = response_data.get("documents", [])
                document_links = [doc["link"] for doc in documents]

                # Extract the content and metadata
                chunks_tasks = [__fetch_chunk(link) for link in document_links]
                chunks = await asyncio.gather(*chunks_tasks)
        except Exception as e:
            print(f"Error fetching chunks: {e}")

        entry["relevant_chunk"] = await get_relevant_docs(query, chunks)
        return entry


async def main():
    file = "./output/generated_question.json"

    with open(file, "r", encoding="utf-8") as json_file:
        list_question = json.load(json_file)

    complete_list_questions = [__get_relevant_chunks(entry) for entry in list_question]
    complete_list_questions = await tqdm.gather(
        *complete_list_questions,
        desc="Getting relevant chunks",
        total=len(complete_list_questions),
    )

    with open(
        "./output/questions_and_chunks.json",
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(complete_list_questions, json_file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    asyncio.run(main())
