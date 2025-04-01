(Due to technical issues, the search service is temporarily unavailable.)

Here's the corrected and fully English version of your README:

# RAGondin 

RAGondin is a project dedicated to experimenting with advanced RAG (Retrieval-Augmented Generation) techniques to improve the quality of such systems. We start with a vanilla implementation and build up to more advanced techniques to address challenges and edge cases in RAG applications.  

![](RAG_architecture.png)

## Goals
- Experiment with advanced RAG techniques
- Develop evaluation metrics for RAG applications
- Collaborate with the community to innovate and push the boundaries of RAG applications

## Current Features
- **Supported File Formats**  
The current branch handles the following file types: `pdf`, `docx`, `doc`, `odt`, `pptx`, `ppt`, `txt`. Other formats (csv, html, etc.) will be added in future releases.

- **Chunking**  
Differents chunking strategies are implemented: **`semantic` and `recursive` chunking**.
Currently **semantic chunker** is used to process all supported file types. Future releases will implement format-specific chunkers (e.g., specialized CSV chunking, Markdown chunker, etc).

- **Indexing & Search**  
After chunking, data is indexed in a **Qdrant** vector database using the multilingual embedding model `HIT-TMG/KaLM-embedding-multilingual-mini-v1` (ranked highly on the MTEB benchmark). The same model embeds user queries for semantic search (Dense Search).  
    * **Hybrid Search**: Combines **`semantic search`** with keyword search (using **`BM25`**) to handle domain-specific jargon and coded product names that might not exist in the embedding model's training data.

- **Retriever**  
Supports three retrieval modes:  
    * **Single Retriever**: Standard query-based document retrieval  
    * **MultiQuery**: Generates augmented query variations using an LLM, then combines results  
    * **HyDE**: Generates a hypothetical answer using an LLM, then retrieves documents matching this answer  

- **Grader**: Filters out irrelevant documents after retrieval.  
- **Reranker**: Uses a multilingual reranking model to reorder documents by relevance with respect to the user's query. This part is important because the retriever returns documents that are semantically similar to the query. However, similarity is not synonymous with relevance, so rerankers are essential for reordering documents and filtering out less relevant ones. This helps reduce hallucination.

- **RAG Types**:  
    * **SimpleRAG**: Basic implementation without chat history  
    * **ChatBotRAG**: Version that maintains conversation context  

## Configurations
- `.env`: Stores your LLM API key (`API_KEY`)  and your `BASE_URL` see the **`.env.example`**

## Usage

#### 1. Clone the repository:
```bash
git clone https://github.com/OpenLLM-France/RAGondin.git
cd RAGondin
git checkout dev
```

#### 2. Create uv environment and install dependencies:
Requirements: Python3.12 and uv installed

```bash
# Create a new environment with all dependencies
uv init
uv sync
```

#### 3. Create a .env file
Copy the content of **.env.exemple** to **.env**. Then add the **BASE_URL** and **API_KEY**.

#### 4.Deployment
1. **Load all the documents that you want to extract the informations**

Add the files (word, excel, pptx, pdf, etc.) into the './data/' folder or add it later on the web interface.

2. **Launch the app**:
```bash
# launch the api
docker-compose up -d --build
```

### 🧠 API Overview

This FastAPI-powered backend offers capabilities for document-based question answering (RAG), semantic search, and document indexing across multiple partitions. It exposes endpoints for interacting with a vector database and managing document ingestion, processing, and querying.

---

### 📍 Endpoints Summary

---

#### 🔍 LLM Calls

**`POST /{partition}/generate`**  
Generates an answer to a user’s input based on a chat history and a document corpus in a given partition. Supports asynchronous streaming response.

---

#### 📦 Indexer

**`POST /indexer/partition/{partition}/file/{file_id}`**  
Uploads a file (with optional metadata) to a specific partition.

- **Inputs:**
  - `file` (form-data): binary – File to upload  
  - `metadata` (form-data): JSON string – Metadata for the file (e.g. `{"file_type": "pdf"}`)

- **Returns:**
  - `201 Created` with a JSON containing the task status URL

---

**`PUT /indexer/partition/{partition}/file/{file_id}`**  
Replaces an existing file in the partition. Deletes existing entry and creates a new indexation task.

- **Inputs:**
  - `file` (form-data): binary – File to upload  
  - `metadata` (form-data): JSON string – Metadata for the file (e.g. `{"file_type": "pdf"}`)

- **Returns:**
  - `201 Created` with a JSON containing the task status URL

---

**`PATCH /indexer/partition/{partition}/file/{file_id}`**  
Updates the metadata of an existing file without reindexing.

- **Inputs:**
  - `metadata` (form-data): JSON string – Metadata for the file (e.g. `{"file_type": "pdf"}`)

---

**`DELETE /indexer/partition/{partition}/file/{file_id}`**  
Deletes a file from a specific partition.

---

**`GET /indexer/task/{task_id}`**  
Retrieves the status of an asynchronous indexing task.

---

#### 🔎 Semantic Search

**`GET /search/`**  
Searches across multiple partitions using a semantic query.

- **Inputs:**
  - `partitions` (query, optional): List[str] – Partitions to search (default: `["all"]`)  
  - `text` (query, required): string – Text to search semantically  
  - `top_k` (query, optional): int – Number of top results to return (default: `5`)

- **Returns:**
  - `200 OK` with a JSON list of document links (HATEOAS style)

---

**`GET /search/partition/{partition}`**  
Searches within a specific partition.

- **Inputs:**
  - `text` (query, required): string – Text to search semantically  
  - `top_k` (query, optional): int – Number of top results to return (default: `5`)

- **Returns:**
  - `200 OK` with a JSON list of document links (HATEOAS style)

---

**`GET /search/partition/{partition}/file/{file_id}`**  
Searches within a specific file in a partition.

- **Inputs:**
  - `text` (query, required): string – Text to search semantically  
  - `top_k` (query, optional): int – Number of top results to return (default: `5`)

- **Returns:**
  - `200 OK` with a JSON list of document links (HATEOAS style)

---

#### 📄 Document Extract Details

**`GET /extract/{extract_id}`**  
Fetches a specific extract by its ID.

- **Returns:**
  - Extract text content  
  - Metadata (JSON)

---

#### 💬 OpenAI-Compatible Chat

**`POST /v1/chat/completions`**  
OpenAI-compatible chat completion endpoint using a Retrieval-Augmented Generation (RAG) pipeline. Accepts `model`, `messages`, `temperature`, `top_p`, etc.




## Contribute
Contributions are welcome! Please follow standard GitHub workflow:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Disclaimer
This repository is for research and educational purposes only. While we strive for correctness, we cannot guarantee fitness for any particular purpose. Use at your own risk.

## License
MIT License - See [LICENSE](LICENSE) file for details.
```
