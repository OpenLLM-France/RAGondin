# Documents that has informations about the query

import json
import os
import asyncio
from tqdm.asyncio import tqdm
from openai import OpenAI
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from utils.dependencies import vectordb

settings = {
    'temperature': 0.2,
    'max_retries': 3,
    'timeout':60,
    'model': "Qwen2.5-VL-7B-Instruct", # "Qwen2.5-VL-7B-Instruct",
    'api_key': "sk-qEoN3RedKamDQNuZJH7F9Q", # 'sk-Xt9LmQvJw2ZaBf_T7NpYdC',
    'base_url': "https://chat.lucie.ovh.linagora.com/v1", # http://91.134.49.99:8000/v1/",
}

llm = ChatOpenAI(**settings).with_retry(stop_after_attempt=5)

async def get_doc(chunk, client, question, semaphore=asyncio.Semaphore(10)):
    async with semaphore:
        chunk_id = chunk.metadata.get("_id")
        content = chunk.page_content.split("=> chunk: ")[1]
        # file_content = open(os.path.join("./data", file_name), "rb").read().decode('utf-8') # bytes like

        # messages = [
        #     {"role": "user", "content": f"Selon ce document:\n\n{file_content}\nRéponds 'oui' ou 'non' s'il y a des informations concernant cette question: {question}"},
        # ]
        
        # response = client.chat.completions.create(
        #     messages=messages,
        #     model='Qwen2.5-VL-7B-Instruct',
        #     temperature=0,
        #     top_p=1,
        #     n=1,
        #     max_tokens=50,
        #     stream=False,
        #     stop=None
        # )

        # resp = response.choices[0].message.content
        # if "oui" in resp.lower():
        #     return file_name
        # return
    
        message = HumanMessage(
            content=[
                {
                    "type": "text", 
                    "text": f"Selon ce document:\n\n{content}\nRéponds 'oui' ou 'non' s'il y a des informations concernant cette question: {question}"
                }
            ]
        )

        output = await llm.ainvoke([message])
        resp = output.content

        if "oui" in resp.lower():
            return  {"id":chunk_id,
                     "content": content}
        return
    
async def get_relevant_docs(client, question, chunks):
    tasks = [get_doc(chunk, client, question) for chunk in chunks]
    list_relevant_chunk = await tqdm.gather(*tasks)
    list_result = [chunk for chunk in list_relevant_chunk if chunk is not None]
    return list_result

async def main():
    file = "./ragondin/evaluation-embedder-reranker/output_v2.json"
    client = OpenAI(api_key="sk-qEoN3RedKamDQNuZJH7F9Q", base_url="https://chat.lucie.ovh.linagora.com/v1")

    with open(file, "r", encoding="utf-8") as json_file:
        list_question = json.load(json_file)
        complete_list_questions = []

        count_question = 0

        for entry in list_question:
            question = entry["question"]
            chunks = await vectordb.async_search(query=question, partition=['frwiki'], top_k=10, similarity_threshold=0.95)
        
            entry["relevant chunk"] = await get_relevant_docs(client, question, chunks)

            complete_list_questions.append(entry)
            count_question += 1

            print(f"Search relevant docs for {question}: {entry["relevant chunk"]}")
            print("-" * 50)

        with open("./ragondin/evaluation-embedder-reranker/complete_dataset_v2.json", "w", encoding="utf-8") as json_file:
            json.dump(complete_list_questions, json_file, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    asyncio.run(main())