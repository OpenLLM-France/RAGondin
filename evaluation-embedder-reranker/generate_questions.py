# This file is for retrieving questions for every docs in the dataset
import asyncio
import json
from openai import AsyncOpenAI
from pathlib import Path
from tqdm.asyncio import tqdm

# get list of txt files in the data directory in .data using pathlib
txt_files = Path("../data").glob("*.txt")

# output file
out_file = "./output/generated_question.json"

api_key = "sk-qEoN3RedKamDQNuZJH7F9Q"
base_url = "https://chat.lucie.ovh.linagora.com/v1"
model_name = "Qwen2.5-VL-7B-Instruct"

client = AsyncOpenAI(api_key=api_key, base_url=base_url)


async def generate_question(in_file: Path, semaphore: asyncio.Semaphore):
    async with semaphore:
        with open(in_file, "rb") as file:
            file_content = file.read().decode("utf-8")[:10000]  # bytes like

        messages = [
            {
                "role": "user",
                "content": f"Voici le contenu du fichier '{in_file.name}':\n\n{file_content}\nPosez une question sur ce contenu ?",
            },
        ]

        response = await client.chat.completions.create(
            messages=messages,
            model=model_name,
            temperature=0.3,
            max_tokens=100,
            stream=False,
        )
        question = response.choices[0].message.content
        return {"file": in_file.name, "question": question}


async def main():
    semaphore = asyncio.Semaphore(
        10
    )  # Limit the number of concurrent tasks  # noqa: F821
    tasks = [generate_question(file_path, semaphore) for file_path in txt_files]

    list_questions = await tqdm.gather(
        *tasks, desc="Generating questions", total=len(tasks)
    )

    # Save the questions to a JSON file
    with open(out_file, "w", encoding="utf-8") as json_file:
        json.dump(list_questions, json_file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    asyncio.run(main())
