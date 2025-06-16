import random
import asyncio
import os
import httpx
import json
import time
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

async def get_clusters(url: str, semaphore=asyncio.Semaphore(10)) -> dict:
    async with semaphore:
        retries = 3
        for attempt in range(retries):
            try:
                async with httpx.AsyncClient(timeout=400) as client:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    clusters = resp.json()["clusters"]
                if not clusters:
                    raise ValueError("No clusters found.")
                return clusters
            except Exception as e:
                logger.debug(f"Attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(1)  # Wait before retrying
                else:
                    logger.debug(f"Error fetching chunks after {retries} attempts: {e}")
                    return None


async def generate_questions_from_clusters(clusters, n_min=1, n_max=3):
    list_questions = []
    question_tasks, response_tasks = [], []
    for cluster_label, chunks in clusters.items():  # cluster loop
        question_count = 10
        while question_count > 0:
            n = 3       # random.randint(n_min, min(n_max, len(chunks)))
            sampled_chunks = random.sample(chunks, n)
            chunks_text = [c["text"] for c in sampled_chunks]

            question_prompt = (
                "Générez une question qui concerne ces chunks:\n"
                "--------------------------------\n"
                f"{chunks_text[0]}\n"
                "--------------------------------\n"
                f"{chunks_text[1]}\n"
                "--------------------------------\n"
                f"{chunks_text[2]}"
            )
            question_task = asyncio.create_task(
                call_llm(question_prompt)
            )
            question_tasks.append(question_task)

            answer_prompt = (
                "À propos de ces chunks:\n"
                "--------------------------------\n"
                f"{chunks_text[0]}\n"
                "--------------------------------\n"
                f"{chunks_text[1]}\n"
                "--------------------------------\n"
                f"{chunks_text[2]}\n"
                "--------------------------------\n"
                f"Répondez à cette question: {questions[-1]}"
            )
            response_task = asyncio.create_task(
                call_llm(answer_prompt)
            )
            response_tasks.append(response_task)

            question_count-= 1

    questions = await tqdm.gather(*question_tasks, desc="Question generation progress")
    llm_answers = await tqdm.gather(*response_tasks, desc="Question generation progress")
    
    for question, llm_answer in zip(questions, llm_answers):
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
    url = f"{ragondin_api_base_url}/partition/{partition}/clusters"

    start = time.time()
    clusters = await get_clusters(url)
    pause = time.time()
    logger.info(f"Clusters retrieval time: {pause - start} seconds")

    #questions = await generate_questions_from_clusters(clusters)

    limited_clusters = {k: clusters[k] for k in list(clusters.keys())[:3]}
    questions = await generate_questions_from_clusters(limited_clusters)
    logger.info(f"Questions generated time: ({time.time() - pause}) seconds")
    with open("generated_questions.json", "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    asyncio.run(main())