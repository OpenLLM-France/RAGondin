import random
import asyncio
import os
import httpx
import json
import time
import numpy as np
import hdbscan
import umap
import umap.umap_ as umap
from dotenv import load_dotenv
from loguru import logger
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tqdm.asyncio import tqdm

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
    "api_key":API_KEY
}
llm = ChatOpenAI(**settings).with_retry(stop_after_attempt=2)

async def call_llm(prompt: str, semaphore=asyncio.Semaphore(10)) -> str:
    async with semaphore:
        message = HumanMessage(
        content=[
            {"type": "text", "text": prompt}
        ]
        )
        resp = await llm.ainvoke([message])
        resp = resp.content
        return resp
async def question_answer_task(chunks: str, semaphore=asyncio.Semaphore(10)):
    async with semaphore:
        question_prompt = (
            "Générez une question qui concerne ces chunks:\n"
            f"{chunks}"
        )
        question = await call_llm(question_prompt)
        
        answer_prompt = (
            "À propos de ces chunks:\n"
            f"{chunks}"
            "--------------------------------\n"
            f"Répondez à cette question: {question}"
        )
        response_llm = await call_llm(answer_prompt)
        return question, response_llm

async def get_all_chunks(url: str, semaphore=asyncio.Semaphore(10)) -> dict:
    async with semaphore:
        retries = 3
        for attempt in range(retries):
            try:
                async with httpx.AsyncClient(timeout=400) as client:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    all_chunks_list = resp.json()["All chunks' details"]
                if not all_chunks_list:
                    raise ValueError("No clusters found.")
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
            n = 3       # random.randint(n_min, min(n_max, len(chunks)))
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
            task = asyncio.create_task(
                question_answer_task(prompt)
            )
            tasks.append(task)

            question_count-= 1

    questions_and_answers = await tqdm.gather(*tasks, desc="Generation progress")
    
    
    for (question, llm_answer) in questions_and_answers:
        list_questions.append({
            "question": question,
            "chunk ids": [c["id"] for c in sampled_chunks],
            "LLM answer": llm_answer,
        })
    return list_questions

async def main():
    num_port = os.environ.get("APP_PORT")
    num_host = "163.114.159.68"  # "localhost"
    ragondin_api_base_url = f"http://{num_host}:{num_port}"
    partition = "benchmark"
    url = f"{ragondin_api_base_url}/partition/{partition}/chunks"

    start = time.time()
    all_chunks_list = await get_all_chunks(url)
    pause = time.time()
    logger.info(f"Clusters retrieval time: {pause - start} seconds")

    ids, chunk_contents, chunk_embeddings, file_ids = map(list, zip(*[(chunk["Chunk ID"], chunk["Chunk's content"], chunk["Embedding vector"], chunk["Original file's ID"]) for chunk in all_chunks_list]))

    embeddings = np.array(chunk_embeddings)

    reducer = umap.UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.1,
        metric='cosine'
    )
    embeddings_reduced = reducer.fit_transform(embeddings)

    hdb = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean')
    labels = hdb.fit_predict(embeddings_reduced)

    clusters = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue  # -1 == bruit
        clusters.setdefault(int(label), []).append(
            {
                "id": ids[idx],
                "text": chunk_contents[idx][:100],
                "file_id": file_ids[idx],
            }
        )

    for label, items in clusters.items():
        logger.info(f"Cluster {label}: {[item['id'] for item in items]}")

    #questions = await generate_questions_from_clusters(clusters)

    limited_clusters = {k: clusters[k] for k in list(clusters.keys())[:3]}
    questions = await generate_questions_from_clusters(limited_clusters)
    logger.info(f"Questions generated time: ({time.time() - pause}) seconds")
    
    with open("./dataset.json", "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    asyncio.run(main())