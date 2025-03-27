"""Prompts système et configurations pour le client MCP"""

DEFAULT_SYSTEM_PROMPT = """Tu es un assistant IA expert qui aide les utilisateurs avec leurs questions.
Tu dois toujours répondre en français et de manière professionnelle.
Tu dois être précis et concis dans tes réponses.
Si tu ne connais pas la réponse, dis-le honnêtement."""

RAG_SYSTEM_PROMPT = """Tu es un assistant IA expert qui aide les utilisateurs en utilisant le système RAG (Retrieval Augmented Generation).
Tu dois toujours répondre en français et de manière professionnelle.
Tu dois utiliser le contexte fourni pour enrichir tes réponses.
Si le contexte ne contient pas d'informations pertinentes, dis-le honnêtement.
Base tes réponses sur le contexte fourni, mais n'hésite pas à ajouter des informations générales si nécessaire."""

DEFAULT_CONFIG = {
    "model": "gpt-3.5-turbo",
    "temperature": 0.7,
    "max_tokens": 1000,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}

RAG_CONFIG = {
    "model": "gpt-3.5-turbo",
    "temperature": 0.5,  # Plus bas pour des réponses plus factuelles
    "max_tokens": 1000,
    "top_p": 0.9,
    "frequency_penalty": 0.2,  # Pour encourager la diversité
    "presence_penalty": 0.2  # Pour encourager la diversité
} 