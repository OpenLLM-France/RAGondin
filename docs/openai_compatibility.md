# API Compatible OpenAI pour RAGondin

## Introduction

RAGondin propose désormais une API compatible avec les standards OpenAI, ce qui permet d'intégrer facilement RAGondin avec des applications, bibliothèques et frameworks existants qui supportent déjà l'API OpenAI.

## Endpoints disponibles

### Chat Completions

```
POST /v1/chat/completions
```

Cet endpoint est compatible avec l'API `/v1/chat/completions` d'OpenAI. Il permet de générer des réponses basées sur une conversation en utilisant le moteur RAG de RAGondin.

#### Format de requête

```json
{
  "model": "nom-du-modele",
  "messages": [
    {"role": "system", "content": "Tu es un assistant utile."},
    {"role": "user", "content": "Bonjour, peux-tu m'aider?"}
  ],
  "temperature": 0.7,
  "stream": true
}
```

#### Format de réponse

En mode non-streaming:
```json
{
  "id": "chatcmpl-123abc",
  "object": "chat.completion",
  "created": 1677858242,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Bonjour! Oui, je suis là pour vous aider. Comment puis-je vous assister aujourd'hui?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

En mode streaming, chaque chunk est envoyé dans le format:
```
data: {"id":"chatcmpl-123abc","object":"chat.completion.chunk","created":1677858242,"model":"gpt-3.5-turbo-0613","choices":[{"index":0,"delta":{"content":"Bonjour"},"finish_reason":null}]}

data: {"id":"chatcmpl-123abc","object":"chat.completion.chunk","created":1677858242,"model":"gpt-3.5-turbo-0613","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

...

data: [DONE]
```

## Utilisation avec la bibliothèque OpenAI 

Vous pouvez utiliser l'API RAGondin directement avec la bibliothèque officielle OpenAI comme ceci:

```python
from openai import OpenAI

# Initialiser le client avec notre API
client = OpenAI(
    # Cette clé est ignorée par notre API, mais nécessaire pour la bibliothèque
    api_key="sk-fake-key",
    # Point important: l'URL de notre API
    base_url="http://localhost:8083"
)

# Appel à l'API
response = client.chat.completions.create(
    model="RAGondin",  # Nom du modèle (valeur factice)
    messages=[
        {"role": "system", "content": "Tu es un assistant utile."},
        {"role": "user", "content": "Qu'est-ce que RAGondin?"}
    ],
    stream=True
)

# Traitement de la réponse en streaming
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## Différences avec l'API OpenAI

Bien que compatible, notre API présente quelques différences avec l'API OpenAI officielle:

1. **Authentification**: Notre API n'exige pas de clé API valide, bien que la bibliothèque client nécessite qu'une clé factice soit fournie.
2. **Paramètres du modèle**: Le nom du modèle fourni dans la requête est ignoré, nous utilisons le modèle configuré dans la configuration RAGondin.
3. **Paramètres supplémentaires**: Certains paramètres comme `top_logprobs`, `response_format`, etc. ne sont pas pris en charge.
4. **Metadata**: Notre API renvoie des informations supplémentaires sur les sources utilisées dans l'en-tête `X-Metadata-Sources`.

## Exemples d'intégration

### Avec JavaScript/TypeScript (ChatGPT UI, etc.)

```javascript
import OpenAI from 'openai';

const openai = new OpenAI({
  apiKey: 'sk-fake-key',
  baseURL: 'http://localhost:8083',
});

async function callAPI() {
  const stream = await openai.chat.completions.create({
    model: 'RAGondin',
    messages: [
      { role: 'system', content: 'Tu es un assistant utile.' },
      { role: 'user', content: 'Explique-moi RAGondin.' }
    ],
    stream: true,
  });
  
  for await (const chunk of stream) {
    process.stdout.write(chunk.choices[0]?.delta?.content || '');
  }
}

callAPI();
```

### Avec LangChain

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

chat = ChatOpenAI(
    model="RAGondin",
    openai_api_key="sk-fake-key",
    openai_api_base="http://localhost:8083/v1",
)

messages = [
    SystemMessage(content="Tu es un assistant utile."),
    HumanMessage(content="Qu'est-ce que RAGondin?"),
]

response = chat(messages)
print(response.content)
``` 