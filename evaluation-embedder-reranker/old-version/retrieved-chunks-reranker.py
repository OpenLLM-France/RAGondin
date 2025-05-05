import json
import asyncio
import numpy as np
import httpx
from tqdm.asyncio import tqdm

async def get_file_name_and_content(url, semaphore=asyncio.Semaphore(5)):
    async with semaphore:
        async with httpx.AsyncClient(timeout=httpx.Timeout(4 * 60.0), http2=True
        ) as client:
            response = await client.get(
                url=url
            )
            response.raise_for_status()  # Raise an exception for HTTP errors

            response_data = response.json()
            metadata = response_data.get("metadata", [])
            file_name = metadata["filename"]

            page_content = response_data.get("page_content", [])

            return file_name, page_content

async def retrieve_docs(entry, semaphore=asyncio.Semaphore(5)):
    async with semaphore:
        question = entry["question"]
        
        params = {
            "text": question,  # The text to search semantically
            "top_k": 10,        # Number of top results to return (default is 5)
        }
        base_url = "http://163.114.159.151:8087"

        # Send the request using httpx
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(10 * 60.0), http2=True
        ) as client:
            response = await client.get(
                url=f"{base_url}/search/partition/frwiki",  # Replace 'frwiki' with the desired partition
                params=params,
            )
            response.raise_for_status()  # Raise an exception for HTTP errors

            response_data = response.json()

            # Extract the documents
            documents = response_data.get("documents", [])
            document_links = [doc["link"] for doc in documents]

            # Extract the content and metadata
            list_chunks_id = [int(document_links[i].split('extract/')[1]) for i in range(len(document_links))]
            list_file_name_and_content = [await get_file_name_and_content(document_links[i]) for i in range(len(document_links))]

            list_file_name = [list_file_name_and_content[i][0] for i in range(len(list_file_name_and_content))] 
            list_file_content = [list_file_name_and_content[i][1] for i in range(len(list_file_name_and_content))]
            entry["reranker's input"] = {
                "chunks id": list_chunks_id,
                "file name": list_file_name,
                "file content": list_file_content
            }

            return entry

async def main():
    out_file = "./output/complete_dataset.json"

    json_file = open(out_file, "r", encoding="utf-8")
    list_questions = json.load(json_file)
    tasks = []
    for entry in list_questions:
        try:
            task = asyncio.create_task(retrieve_docs(entry))
            tasks.append(task)
        except Exception as e:
            print(f"Error occurred: {e}")
            print("Breaking the loop due to an exception.")
            break

    if not tasks:
        print("No evaluations were completed due to an error.")
        return
    
    list_entries = await tqdm.gather(*tasks)
    with open("./output/complete_dataset.json", "w", encoding="utf-8") as json_file:
        json.dump(list_entries, json_file, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    asyncio.run(main())