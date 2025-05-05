# This file is for retrieving questions for every docs in the dataset
import os
import json
from openai import OpenAI

out_file = "./evaluation-embedder-reranker/output/generated_question.json"
list_questions = []

client = OpenAI(api_key="sk-qEoN3RedKamDQNuZJH7F9Q", base_url="https://chat.lucie.ovh.linagora.com/v1")
for file_name in os.listdir("./data"):
    if not file_name.endswith(".txt"):
        continue

    file_content = open(os.path.join("./data", file_name), "rb").read().decode('utf-8')[:10000] # bytes like

    messages = [
        {"role": "user", "content": f"Voici le contenu du fichier '{file_name}':\n\n{file_content}\nPosez une question sur ce contenu ?"},
    ]

    response = client.chat.completions.create(
        messages=messages,
        model='Qwen2.5-VL-7B-Instruct',
        temperature=0,
        top_p=1,
        n=1,
        max_tokens=100,
        stream=False,
        stop=None
    )
    question = response.choices[0].message.content
    list_questions.append({
        "file": file_name,
        "question": question
    })
    
    print(f"Generated Question for {file_name}: {question}")
    print("-" * 50)

with open(out_file, "w", encoding="utf-8") as json_file:
    json.dump(list_questions, json_file, indent=4, ensure_ascii=False)