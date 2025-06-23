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

# Retrieving responses and document sources fom RAGondin
async def retrieve_response_and_docs_ragondin(
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


# Sources retrieval evaluation
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


# Response retrieval evaluation
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

class CompletionEvaluationResponse(BaseModel):
    output: Literal["complete", "mostly_complete", "partially_complete", "incomplete"] = (
        Field(
            ...,
            description="The output of the LLM judge. It can be one of the following: "
            "'complete', 'mostly_complete', 'partially_complete', 'incomplete'. "
            "This indicates how well the generated answer matches the true answer.",
        )
    )

class PrecisionEvaluationResponse(BaseModel):
    output: Literal["Highly_precise", "mostly_precise", "low_precision", "imprecise"] = (
        Field(
            ...,
            description="The output of the LLM judge. It can be one of the following: "
            "'Highly_precise', 'mostly_precise', 'low_precision', 'imprecise'. "
            "This indicates how well the generated answer matches the true answer.",
        )
    )
complettion_judge_prompt = """You are an expert judge evaluating the completeness of a language model's response to a question.
Given:
    A query (query)
    The correct or reference answer (true_answer)
    A generated response (generated_answer)
Your task is to assess how complete the generated_answer is in relation to the true_answer. Focus on whether the generated response fully covers, partially covers, or omits important elements found in the true_answer.
Consider:
    Does the response address all key points in the true_answer?
    Are there any significant omissions or gaps?
    Is the response thorough or only partial?"""

precision_judge_prompt = """You are an expert judge evaluating the precision of a language model's response to a question.
Given:
    A query (query)
    The correct or reference answer (true_answer)
    A generated response (generated_answer)
Your task is to assess how precisely the generated_answer aligns with the true_answer. Focus on whether the generated response contains only accurate, relevant, and specific information, without unnecessary or incorrect additions.
Consider:
    Does the response stay focused on what was asked?
    Does it avoid unrelated, vague, or incorrect information?
    Is the content specific and factually aligned with the true_answer?
"""

llm_completion_judge = ChatOpenAI(**llm_judge_settings).with_structured_output(CompletionEvaluationResponse)
llm_precision_judge = ChatOpenAI(**llm_judge_settings).with_structured_output(PrecisionEvaluationResponse)


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
            response_for_completion = await llm_completion_judge.ainvoke(
                [
                    {"role": "system", "content": complettion_judge_prompt},
                    {"role": "user", "content": s},
                ]
            )

            response_for_precision = await llm_precision_judge.ainvoke(
                [
                    {"role": "system", "content": precision_judge_prompt},
                    {"role": "user", "content": s},
                ]
            )
            return response_for_completion.output, response_for_precision.output
        except Exception as e:
            logger.debug(f"Error evaluating response: {e}")
            return "error"


async def main():
    data_file = open("./dataset.json", "r", encoding="utf-8")
    list_response_answer_reference = json.load(data_file)

    num_port = os.environ.get("APP_PORT")
    num_host = "localhost"
    ragondin_api_base_url = f"http://{num_host}:{num_port}"
    partition = "benchmark"

    tasks = [
        retrieve_response_and_docs_ragondin(
            query=resp_ans_reference["question"],
            partition=partition,
            ragondin_base_url=ragondin_api_base_url,
            semaphore=asyncio.Semaphore(10),
        )
        for resp_ans_reference in list_response_answer_reference
    ]

    ragondin_answer_chunk_ids_l = await tqdm.gather(*tasks, desc="Fetching")

    scores = []

    response_judge_tasks = []

    for (ragondin_response, ragondin_chunk_ids), input_reference in zip(ragondin_answer_chunk_ids_l, list_response_answer_reference):
        chunk_id_reference = [c["id"] for c in input_reference["chunks"]]
        score = source_score_per_question(
            chunk_id_reference=chunk_id_reference, chunk_id_llm=ragondin_chunk_ids
        )
        scores.append(score)

        resp_eval_task = asyncio.create_task(
            response_score_per_question(
                query=input_reference["question"],
                llm_answer=input_reference["llm_answer"],
                ragondin_answer=ragondin_response,
            )
        )
        response_judge_tasks.append(
            resp_eval_task
        )

    llm_judge_scores = await tqdm.gather(*response_judge_tasks, desc="Evaluating responses")

    response_completion_evaluation_result, response_precision_evaluation_result = {}, {}
    for completion_evaluation, precision_evaluation in llm_judge_scores:
        response_completion_evaluation_result[completion_evaluation] = (
            response_completion_evaluation_result.get(completion_evaluation, 0) + 1
        )
        response_precision_evaluation_result[precision_evaluation] = (
            response_precision_evaluation_result.get(precision_evaluation, 0) + 1
        )
    
    print(f"LLM completion Scores: {response_completion_evaluation_result}\n")
    print(f"LLM precision Scores: {response_precision_evaluation_result}\n")
    print(f"Source evaluation - nDCG: {np.array(scores).mean()}")


if __name__ == "__main__":
    asyncio.run(main())
