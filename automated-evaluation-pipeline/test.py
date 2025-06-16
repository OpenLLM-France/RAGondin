import os
import httpx
import json
from dotenv import load_dotenv

load_dotenv()  # Charge les variables du .env

BASE_URL = os.environ["BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL = os.environ["MODEL"]

def call_llm(chunks_texts):
    if len(chunks_texts) == 1:
        prompt = f"Génère une question dont la réponse se trouve uniquement dans ce texte :\n{chunks_texts[0]}"
    else:
        prompt = (
            "Génère une question dont la réponse nécessite d'utiliser toutes les informations suivantes :\n"
            + "\n\n".join([f"Fragment {i+1} : {txt}" for i, txt in enumerate(chunks_texts)])
        )
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 128,
        "temperature": 0.7,
    }
    response = httpx.post(f"{BASE_URL}/chat/completions", headers=headers, json=data, timeout=60)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# Exemple d'utilisation :
chunks = ["Texte 1", "Texte 2"]
question = call_llm(chunks)
print(question)






llm_config = {
    "api_key": os.environ.get("API_KEY"),
    "base_url": os.environ.get("BASE_URL"),
    "model": os.environ.get("MODEL"),
}

llm = LLM(llm_config)

async def call_llm(chunks_texts):
    if len(chunks_texts) == 1:
        prompt = f"Génère une question dont la réponse se trouve uniquement dans ce texte :\n{chunks_texts[0]}"
    else:
        prompt = (
            "Génère une question dont la réponse nécessite d'utiliser toutes les informations suivantes :\n"
            + "\n\n".join([f"Fragment {i+1} : {txt}" for i, txt in enumerate(chunks_texts)])
        )
    request = {
        "model": llm_config["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 128,
        "temperature": 0.7,
    }
    response = await llm.chat_completion(request)
    return response["choices"][0]["message"]["content"]