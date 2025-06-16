import random
import asyncio
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
    response = httpx.post(f"{BASE_URL}/chat/completions", headers=headers, json=data, timeout=400)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


async def generate_questions_from_clusters(clusters, n_min=1, n_max=3):
    questions = []
    for cluster_label, chunks in clusters.items():
        n = random.randint(n_min, min(n_max, len(chunks)))
        sampled_chunks = random.sample(chunks, n)
        chunks_texts = [c["text"] for c in sampled_chunks]
        question = await call_llm(chunks_texts)
        questions.append({
            "cluster_label": cluster_label,
            "question": question,
            "chunk_ids": [c["id"] for c in sampled_chunks],
            "texts": chunks_texts,
        })
    return questions

async def main():
    # 1. Récupère les clusters via l'API
    url = "http://163.114.159.68:8090/partition/benchmark/clusters"
    async with httpx.AsyncClient(timeout=400) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        clusters = resp.json()["clusters"]

    # 2. Génère les questions
    #questions = await generate_questions_from_clusters(clusters)

    # Limiter aux 3 premiers clusters
    limited_clusters = {k: clusters[k] for k in list(clusters.keys())[:3]}
    questions = await generate_questions_from_clusters(limited_clusters)
    

    # 3. Sauvegarde le résultat
    with open("generated_questions.json", "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    asyncio.run(main())