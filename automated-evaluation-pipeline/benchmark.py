import os
import asyncio
import json
import math
from typing import Literal
import numpy as np
from loguru import logger
from openai import AsyncOpenAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

load_dotenv()




def relevance(val, true_chunk_ids):
    return 1 if val in true_chunk_ids else 0


def source_score_per_question(
    chunk_id_reference: list[int],
    chunk_id_llm: list[int],
):
    val_DCG = 0
    for i, val in enumerate(chunk_id_llm):
        val_DCG += relevance(val, chunk_id_reference) / math.log2(i + 2)
    iDCG = 0
    for i in range(min(len(chunk_id_reference), len(chunk_id_llm))):
        iDCG += 1 / math.log2(i + 2)
    return val_DCG / iDCG


async def retrieve_response_and_docs(
    query: str, partition: str, ragondin_base_url: str, semaphore=asyncio.Semaphore(10)
):
    async with semaphore:
        base_url = f"{ragondin_base_url}/v1"
        auth_key = "sk-1234"
        client = AsyncOpenAI(api_key=auth_key, base_url=base_url)

        settings = {
            "model": f"ragondin-{partition}",
            "messages": [
                {
                    "role": "user",
                    "content": query,
                }
            ],
            "temperature": 0.2,
            "stream": False,
        }

        try:
            res = await client.chat.completions.create(**settings)
            response_llm = res.choices[0].message.content
            list_source_chunk_ids = [
                item["_id"] for item in json.loads(res.extra)["sources"]
            ]

            return response_llm, list_source_chunk_ids
        except Exception as e:
            logger.debug(f"Error fetching chunks and response: {e}")


llm_judge_settings = {
    "model": os.environ.get("MODEL"),
    "base_url": os.environ.get("BASE_URL"),
    "api_key": os.environ.get("API_KEY"),
    "temperature": 0.2,
    "max_tokens": 1000,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
}


class JudgeResponse(BaseModel):
    output: Literal["compelte", "incomplete", "somewhat_complete", "irrelevant"] = (
        Field(
            ...,
            description="The output of the LLM judge. It can be one of the following: "
            "'complete', 'incomplete', 'somewhat_complete', 'irrelevant'. "
            "This indicates how well the generated answer matches the true answer.",
        )
    )


judge_prompt = """You are an expert judge evaluating the quality of a response to a question.
Given a query (`query`) and the true answer (`true_answer`), you will evaluate the response of a language model (LLM) `generated_answer` to the query with respect to the `true` factual answer.
"""

llm_judge = ChatOpenAI(**llm_judge_settings).with_structured_output(JudgeResponse)


async def response_score_per_question(
    query: str,
    llm_answer: str,
    ragondin_answer: str,
    sempahore: asyncio.Semaphore = asyncio.Semaphore(10),
):
    s = f"""Here are the needed details to evaluate the response of a language model (LLM) to a question:
    query: {query}
    true_answer: {ragondin_answer}
    generated_answer: {llm_answer}
    """
    async with sempahore:
        try:
            response = await llm_judge.ainvoke(
                [
                    {"role": "system", "content": judge_prompt},
                    {"role": "user", "content": s},
                ]
            )
            return response.output
        except Exception as e:
            logger.debug(f"Error evaluating response: {e}")
            return "error"


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

    llm_answer_chunk_ids_l = await tqdm.gather(*tasks, desc="Fetching")

    responses = []
    scores = []

    llm_judge_tasks = []

    for (llm_response, ids_l), input_ in zip(llm_answer_chunk_ids_l, list_input):
        true_ids = [c["id"] for c in input_["chunks"]]
        responses.append(llm_response)
        score = source_score_per_question(
            chunk_id_reference=true_ids, chunk_id_llm=ids_l
        )
        scores.append(score)

        llm_judge_tasks.append(
            response_score_per_question(
                query=input_["question"],
                llm_answer=input_["llm_answer"],
                ragondin_answer=llm_response,
            )
        )

    llm_judge_scores = await tqdm.gather(*llm_judge_tasks, desc="Evaluating responses")

    print(f"LLM Judge Scores: {llm_judge_scores}")
    print(f"Source evaluation - nDCG: {np.array(scores).mean()}")


if __name__ == "__main__":
    asyncio.run(main())
