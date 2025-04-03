# RAGondin 

RAGondin is a project dedicated to experimenting with advanced RAG (Retrieval-Augmented Generation) techniques to improve the quality of such systems. We start with a vanilla implementation and build up to more advanced techniques to address challenges and edge cases in RAG applications.  

![](RAG_architecture.png)

## Goals
- Experiment with advanced RAG techniques
- Develop evaluation metrics for RAG applications
- Collaborate with the community to innovate and push the boundaries of RAG applications

## Current Features
- **Supported File Formats**  
The current branch supports the following file types: `pdf`, `docx`, `doc`, `pptx`, `ppt`, and `txt`. Additional formats such as `odt`, `csv`, and `html` will be introduced in future updates. All supported file types are converted to Markdown, with images replaced by captions generated using a Vision Language Model (VLM) (Refer to the **Configuration** section for more details). Then the markdown output is chunked and indexed in milvus.

- **Chunking**  
Differents chunking strategies are implemented: **`semantic` and `recursive` chunking**.
Currently **recursive chunker** is used to process all supported file types. Future releases will implement format-specific chunkers (e.g., specialized CSV chunking, Markdown chunker, etc). You can find the specifics of this chunker in the file *`.hydra_config/chunker/recursive_splitter.yaml`*

  ```yml
  name: recursive_splitter
  chunk_size: 1500
  chunk_overlap: 300
  contextual_retrieval: true
  ```
Here the **`chunk_size` and `chunk_overlap`** are expressed in terms of tokens and not characters. For a quick test, contextual_retrieval (see. [contextual retrievel](!https://www.anthropic.com/news/contextual-retrieval) ) can be set to false

- **Indexing & Search**  
After chunking, data is indexed in a **Milvus** vector database using the multilingual embedding model `HIT-TMG/KaLM-embedding-multilingual-mini-v1` (ranked highly on the MTEB benchmark). The same model embeds user queries for semantic search (Dense Search).  
    * **Hybrid Search**: Combines **`semantic search`** with keyword search (using **`BM25`**) to handle domain-specific jargon and coded product names that might not exist in the embedding model's training data. By default, our search pipeline  uses hybrid search.

- **Retriever**  
Supports three retrieval modes: **`multiQuery`, `single` and `hyde`**
    * **single**: Standard query-based document retrieval  
    * **multiQuery**: Generates augmented query variations using an LLM, then combines results. This is the default setting and is to most relevant one according to our tests. 
    * **HyDE**: Generates a hypothetical answer using an LLM, then retrieves documents matching this answer

- **Grader**: Filters out irrelevant documents after retrieval. Currently it's set to be **`false`**.
- **Reranker**: Uses a multilingual reranking model to reorder documents by relevance with respect to the user's query. This part is important because the retriever returns documents that are semantically similar to the query. However, similarity is not synonymous with relevance, so rerankers are essential for reordering documents and filtering out less relevant ones. This helps reduce hallucination by weeding out irrelevant ones.

- **RAG Types**:  
    * **SimpleRAG**: Basic implementation without chat history  
    * **ChatBotRAG**: Version that maintains conversation context. 

## Configurations
- `.env`: Store your LLM API key (`API_KEY`)  and your `BASE_URL` in a **`.env`** at the root of the project (see the **`.env.example`** for references). An **`VLM`(Vision language model)** endpoint is expected. The **`VLM`** is used to caption extracted images from files. 

## Usage

#### 1. Clone the repository:
```bash
git clone https://github.com/OpenLLM-France/RAGondin.git
cd RAGondin
git checkout dev
```

#### 2. Create uv environment and install dependencies:
**Requirements**: Ensure you have Python 3.12 installed along with `uv`. For detailed installation instructions, refer to the [uv official documentation](https://docs.astral.sh/uv/getting-started/installation/#pypi).

* To install `uv`, you can use either `pip` (if already available) or `curl`. Additional installation methods are outlined in the [documentation](https://docs.astral.sh/uv/getting-started/installation/#pypi).
```bash
# with pip
pip install uv

# with curl
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
# Create a new environment with all dependencies
uv sync # There is already 
```

#### 3. Create a .env file
Copy the content of **.env.exemple** to **.env**. Then add the **BASE_URL** and **API_KEY**.

#### 4.Deployment: Launch the app
The application can be launched in either a GPU or CPU environment, depending on your device's capabilities. Use the following commands:

```bash
# Launch with GPU support (recommended for faster processing)
docker compose up --build

# Launch with CPU only (useful if GPU is unavailable)
docker compose --profile cpu up --build
```
> **Note**: The initial launch may take longer due to the installation of required dependencies. Once the application is up and running, you can access the web interface at `http://localhost:8080/chainlit` (8080 = APP_PORT in your **`.env`**) to manage documents, execute searches, or interact with the RAG pipeline.


* **Running on CPU**:  
  For quick testing on a CPU, adjust the following parameters to reduce computational load (at the cost of performance and quality):  
  - Set **`top_k=4`** for the **`reranker`** in `.hydra_config/config.yaml`.  
  - Set **`top_k=5`** for the **`retriever`** in `.hydra_config/retriever/base.yaml`.  

* **Running on GPU**:  
  Optimal values are already configured for GPU usage. However, you can modify these settings if you wish to experiment with different configurations depending on the capacity of your llm-endpoint and host environment.

Now, that your app is launched, files can be added in order to chat with your documents. The following sections deals with that.

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
