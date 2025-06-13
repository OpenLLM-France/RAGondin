import json
import asyncio
import os
import httpx

from loguru import logger
from dotenv import load_dotenv
from langchain_core.messages import HumanMesssage
from langchain_openai import ChatOpenAI
load_dotenv()

vlm_config = {
    "temperature": 0.2
    "timeout": 60
    "max_retries": 2
    "logprobs": true,
    "base_url": os.environ.get("BASE_URL")
    "model": os.environ.get("MODEL")
    "api_key": os.environ.get("API_KEY")
}

llm = ChatOpenAI(**vlm_config).with_retry(stop_after_attemp=2)

async def generate_question(prompt:str) -> str:
    try:
        message = HumanMessage()
async def main():
    num_port = os.environ.get("APP_PORT")
    num_host = "163.114.159.68"  # "localhost"
    ragondin_api_base_url = f"http://{num_host}:{num_port}"
    partition = "benchmark"

    retries = 3
    for attemps in range(retries):
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                res = await client.get(
                    url=f"{ragondin_api_base_url}/partition/{partition}/clusters"
                )
                res.raise_for_status()
                list_clusters = res.json().get("clusters").items()
                list_questions = []

                for cluster_label, chunks in list_clusters:
                    list_ids, list_texts, list_file_ids = map(list, zip(*[(chunk["id"], chunk["text"], chunk["file_id"]) for chunk in chunks]))
                    question = "..."
                    list_questions.append(
                        {
                            "cluster_label": cluster_label,
                            "question": question,
                            "chunk_ids": list_ids,
                            "texts": list_texts,
                            "file_ids": list_file_ids,
                        }
                    )
                
                with open("./data_files/list_questions.json", "w", encoding="utf-8") as f:
                    json.dump(list_questions, f, indent=4, ensure_ascii=False)
        
        except Exception as e:
            logger.debug(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(1)  # Wait before retrying
            else:
                logger.debug(f"Error fetching chunks after {retries} attempts: {e}")
                return None

if __name__ == "__main__":
    asyncio.run(main())