import random
import asyncio
import os
import httpx
import json
import time
import numpy as np
import hdbscan
import ast


# import umap.umap_ as umap
from dotenv import load_dotenv
from loguru import logger
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tqdm.asyncio import tqdm
import pickle

load_dotenv()  # Charge les variables du .env

BASE_URL = os.environ["BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL = os.environ["MODEL"]

settings = {
    "temperature": 0.2,
    "max_retries": 3,
    "timeout": 60,
    "base_url": BASE_URL,
    "model": MODEL,
    "api_key": API_KEY,
}
llm = ChatOpenAI(**settings).with_retry(stop_after_attempt=2)


question_tmpl = """You are an expert in generating questions based on chunks.
Given a set of text chunks, your task is to generate a relevant question that requires all of the provided chunks to be answered.

You should generate a question that is clear, concise, and directly related to the content of the chunks.
The output only that question, without any additional text or explanation.
"""

answer_tmpl = """You are an expert in answering questions based on given chunks.
Given a question and a set of text chunks, your task is to provide a comprehensive answer that utilizes all of the provided chunks.
The answer should be clear, concise, and directly address the question using the information from the chunks.
The output should only be the answer, without any additional text or explanation."""


def format_chunks(chunks: list[str]):
    chunks_str = ""
    for i, chunk in enumerate(chunks, start=1):
        chunks_str += f"Chunk {i}:\n{chunk}\n"
        chunks_str += "-" * 40 + "\n"
    return chunks_str.strip()  # Remove trailing newline and spaces


async def question_answer(chunks: list[str], semaphore=asyncio.Semaphore(10)):
    async with semaphore:
        chunks_str = format_chunks(chunks)

        # generate a question based on the chunks
        messages = [
            {"role": "system", "content": question_tmpl},
            {
                "role": "user",
                "content": f"Here are the chunks:\n{chunks_str}. Generaye the question in the same language as the chunks.",
            },
        ]
        output = await llm.ainvoke(messages)
        llm_question = output.content.strip()

        # generate an answer based on the question and chunks
        messages = [
            {"role": "system", "content": answer_tmpl},
            {
                "role": "user",
                "content": f"Here are the chunks:\n{chunks_str}\n\nQuestion: {llm_question}.\n Generate the answer in the same language as the chunks.",
            },
        ]
        output = await llm.ainvoke(messages)
        llm_answer = output.content.strip()

        return llm_question, llm_answer


async def get_all_chunks(url: str, semaphore=asyncio.Semaphore(10)) -> dict:
    async with semaphore:
        retries = 3
        for attempt in range(retries):
            try:
                async with httpx.AsyncClient(timeout=400) as client:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    all_chunks_list = resp.json()["chunks"]
                if not all_chunks_list:
                    raise ValueError("No chunks found.")
                return all_chunks_list
            except Exception as e:
                logger.debug(f"Attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(1)  # Wait before retrying
                else:
                    logger.debug(f"Error fetching chunks after {retries} attempts: {e}")
                    return None


async def generate_questions_from_clusters(clusters, n_min=1, n_max=3):
    list_questions = []
    tasks = []
    for cluster_label, chunks in clusters.items():  # cluster loop
        question_count = 10
        while question_count > 0:
            n = random.randint(n_min, min(n_max, len(chunks)))
            sampled_chunks = random.sample(chunks, n)
            chunks_text = [c["text"] for c in sampled_chunks]

            prompt = (
                "--------------------------------\n"
                f"{chunks_text[0]}\n"
                "--------------------------------\n"
                f"{chunks_text[1]}\n"
                "--------------------------------\n"
                f"{chunks_text[2]}"
            )
            task = asyncio.create_task(question_answer(prompt))
            tasks.append(task)

            question_count -= 1

    questions_and_answers = await tqdm.gather(*tasks, desc="Generation progress")

    for question, llm_answer in questions_and_answers:
        list_questions.append(
            {
                "question": question,
                "chunk ids": [c["id"] for c in sampled_chunks],
                "LLM answer": llm_answer,
            }
        )
    return list_questions


async def main():
    num_port = "8087"  # os.environ.get("APP_PORT")
    num_host = "163.114.159.68"  # "localhost"
    ragondin_api_base_url = f"http://{num_host}:{num_port}"
    partition = "benchmark"

    url = f"{ragondin_api_base_url}/partition/{partition}/chunks"

    start = time.time()
    all_chunks_list = await get_all_chunks(url)
    pause = time.time()
    logger.info(f"Clusters retrieval time: {pause - start} seconds")

    ids, chunk_contents, chunk_embeddings, file_ids = map(
        list,
        zip(
            *[
                (
                    chunk["metadata"]["_id"],
                    chunk["content"],
                    chunk["metadata"]["vector"],
                    chunk["metadata"]["file_id"],
                )
                for chunk in all_chunks_list
            ]
        ),
    )

    d = {}
    for i, chunk in enumerate(all_chunks_list):
        content = chunk["content"]
        metadata = chunk["metadata"]
        metadata.pop("vector", None)  # Remove vector from metadata

        d[i] = {"content": content, "metadata": metadata}

    embeddings = np.array(list(map(ast.literal_eval, chunk_embeddings)))

    # save data
    os.makedirs("./data", exist_ok=True)
    pickle.dump(
        d, open("./data/chunks_data.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL
    )
    pickle.dump(
        embeddings,
        open("./data/chunks_embeddings.pkl", "wb"),
        protocol=pickle.HIGHEST_PROTOCOL,
    )

    # reducer = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.1, metric="cosine")
    # embeddings_reduced = reducer.fit_transform(embeddings)

    # hdb = hdbscan.HDBSCAN(min_cluster_size=5, metric="euclidean")
    # labels = hdb.fit_predict(embeddings_reduced)

    # clusters = {}
    # for idx, label in enumerate(labels):
    #     if label == -1:
    #         continue  # -1 == bruit
    #     clusters.setdefault(int(label), []).append(
    #         {
    #             "id": ids[idx],
    #             "text": chunk_contents[idx][:100],
    #             "file_id": file_ids[idx],
    #         }
    #     )

    # for label, items in clusters.items():
    #     logger.info(f"Cluster {label}: {[item['id'] for item in items]}")

    # questions = await generate_questions_from_clusters(clusters)

    # limited_clusters = {k: clusters[k] for k in list(clusters.keys())[:3]}
    # questions = await generate_questions_from_clusters(limited_clusters)
    # logger.info(f"Questions generated time: ({time.time() - pause}) seconds")

    # with open("./dataset.json", "w", encoding="utf-8") as f:
    #     json.dump(questions, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    asyncio.run(main())
