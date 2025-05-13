# 🦫 RAGondin — The Open RAG Experimentation Playground

RAGondin is a lightweight, modular and extensible Retrieval-Augmented Generation (RAG) framework designed to explore and test advanced RAG techniques — 100% open source and focused on experimentation, not lock-in.

> Built by the OpenLLM France community, RAGondin offers a sovereign-by-design alternative to mainstream RAG stacks like LangChain or Haystack.

---

## 📑 Table of Contents

- [✨ Features](#-features)
- [🚀 Getting Started](#-getting-started)
- [⚙️ Configuration](#️-configuration)
- [🔁 Launch RAGondin](#️-launch-ragondin)
- [🔍 API Endpoints](#-api-endpoints)
- [🏗️ Architecture](#-architecture)
- [🔧 Contributing](#-contributing)
- [📄 License](#-license)

---

## ✨ Features

- 🔎 Document parsing for `pdf`, `docx`, `doc`, `pptx`, `ppt`, and `txt`. Future updates will expand support to include formats such as `odt`, `csv`, audio and video files, and `html`.  
- 🤖 RAGondin will expose a fully **OpenAI-compatible API** (`/v1/chat/completions`), allowing seamless integration with any tool or framework that supports OpenAI:
> RAG context will be transparently injected before forwarding to the selected LLM backend.  
> ✅ Users retain full sovereignty: your model, your data, your vector store.
- 📚 RAGondin will support **multi-tenant vector indexing**, allowing organizations to isolate and manage multiple RAG knowledge bases across departments or teams:
> Each user or team accesses a logically separated vector index, ensuring data isolation and context-specific retrieval.
- 📦 Vector store powered by Milvus
- 🧩 Plugin-style architecture (easily add new retrievers, chunkers, pipelines)
- ✂️ Multi-strategy chunking (by tokens, sentences, etc.)
- 🔁 Hybrid search (dense + sparse + keyword search)
- ⚙️ Declarative `.env` configuration for backends (embedding, LLM, index, etc.)

---

## 🚀 Getting Started

### 1. Clone the repository:
```bash
git clone https://github.com/OpenLLM-France/RAGondin.git
cd RAGondin
git checkout main # or dev if you want to try the dev branch
```

#### Environment Setup

First, the users are suggested to run RAGondin in a virtual environment (for all the necessary libraries and packages). It can be done easily with:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Create uv environment and install dependencies:
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
uv sync
```

---

## ⚙️ Configuration

Add a `.env` file at the root of the project to configure the LLM (Language Model) and VLM (Vision Language Model) settings. 

It is mandatory to configure the LLM settings (`API_KEY`, `BASE_URL`, `MODEL_NAME`) as well as the VLM settings (`API_KEY`, `BASE_URL`, `MODEL_NAME`). The **`VLM`** is specifically utilized for generating captions for images extracted from files during the vectorization process. If you plan to use the same model for both LLM and VLM functionalities, you can reuse the same settings for both.

For PDF file indexing, multiple options are available:
- **`MarkerLoader` and `DoclingLoader`** are recommended for the best performance (requires GPU).
- **PyMuPDF4LLMLoader** or **PyMuPDFLoader**: Suggested for non-GPU users. Not that these loader doesn't handle Non-searchable PDF nor does it handle images (**`We will add it`**).

Concerning the audio and video files, we use OpenAI's Whisper model to convert the audio into plain text. The file extensions supported by RAGondin are: .wav, .mp3, .mp4, .ogg, .flv, .wma, .aac. You can also choose the model used for transcription for speed or precision. Here are all the Whisper's models: tiny, base, small, medium, large, turbo. For more information, checkout [OpenAI Whisper](https://github.com/openai/whisper)

Other file formats are pre-configured with optimal settings.

```bash
# LLM settings
BASE_URL=
API_KEY=
MODEL=

# VLM settings
VLM_BASE_URL=
VLM_API_KEY=
VLM_MODEL=

# App
APP_PORT=8080
APP_HOST=0.0.0.0

# To enable HTTP authentication via HTTPBearer
AUTH_TOKEN=super-secret-token

## More settings can be added (see .env.example)

# Loaders
PDFLoader=DoclingLoader

# Audio
WHISPER_MODEL=base
```
---

## 🔁 Launch RAGondin

Make sure that you have Docker Desktop in disposition. If not, check out the installation in the official website [Docker](!https://www.docker.com/).

The application can be launched in either a GPU or CPU environment, depending on your device's capabilities. Use the following commands:

```bash
# Launch with GPU support (recommended for faster processing)
docker compose up --build

# Launch with CPU only (useful if GPU is unavailable)
docker compose --profile cpu up --build
```

Once it is running, you can check everything is fine by doing:
```bash
curl http://localhost:0/health_check
```

> **Note**: The initial launch is longer due to the installation of required dependencies. Once the application is up and running, you can access the api documentation at `http://localhost:8080/docs` (8080 is the APP_PORT variable determined in your **`.env`**) to manage documents, execute searches, or interact with the RAG pipeline (see the **next section** about the api for more details). A default chat ui is also deployed using [chainlit](!https://docs.chainlit.io/get-started/overview). You can access to it at `http://localhost:8080/chainlit` chat with your documents with our RAG engine behind it.


* **Running on CPU**:  
  For quick testing on a CPU, you can optimize performance by reducing computational load with the following adjustments in the **`.env`** file:
  - Set **`RERANKER_TOP_K=6`** or even lower to limit the number of documents processed by the reranker. You can actually go further and disable the reranker by **`RERANKER_ENABLED=false`** cause it's a costly operation.
  - Set **`RETRIEVER_TOP_K=4`** to reduce the number of documents retrieved during the search phase.

  These changes may impact performance and result quality but are suitable for lightweight testing.

* **Running on GPU**:  
  Optimal values are already configured for GPU usage. However, you can modify these settings if you wish to experiment with different configurations depending on the capacity of machine.

Now, that your app is launched, files can be added in order to chat with your documents. The following sections deals with that.

---

## 🔍 API Endpoints

| Endpoint        | Method | Description                    |
|----------------|--------|--------------------------------|
| `/ingest`       | POST   | Ingest a document              |
| `/query`        | POST   | Ask a question (RAG pipeline)  |
| `/search`       | GET    | Search documents in index      |

📌 Full OpenAPI spec coming soon.

**`POST /{partition}/generate`**  
Generates an answer to a user’s input based on a chat history and a document corpus in a given partition. Supports asynchronous streaming response.

### 📦 Indexer

**`POST /indexer/partition/{partition}/file/{file_id}`**  
Uploads a file (with optional metadata) to a specific partition.

- **Inputs:**
  - `file` (form-data): binary – File to upload  
  - `metadata` (form-data): JSON string – Metadata for the file (e.g. `{"file_type": "pdf"}`)

- **Returns:**
  - `201 Created` with a JSON containing the task status URL

**`PUT /indexer/partition/{partition}/file/{file_id}`**  
Replaces an existing file in the partition. Deletes existing entry and creates a new indexation task.

- **Inputs:**
  - `file` (form-data): binary – File to upload  
  - `metadata` (form-data): JSON string – Metadata for the file (e.g. `{"file_type": "pdf"}`)

- **Returns:**
  - `201 Created` with a JSON containing the task status URL

**`PATCH /indexer/partition/{partition}/file/{file_id}`**  
Updates the metadata of an existing file without reindexing.

- **Inputs:**
  - `metadata` (form-data): JSON string – Metadata for the file (e.g. `{"file_type": "pdf"}`)

**`DELETE /indexer/partition/{partition}/file/{file_id}`**  
Deletes a file from a specific partition.

**`GET /indexer/task/{task_id}`**  
Retrieves the status of an asynchronous indexing task.

### 🔎 Semantic Search

**`GET /search/`**  
Searches across multiple partitions using a semantic query.

- **Inputs:**
  - `partitions` (query, optional): List[str] – Partitions to search (default: `["all"]`)  
  - `text` (query, required): string – Text to search semantically  
  - `top_k` (query, optional): int – Number of top results to return (default: `5`)

- **Returns:**
  - `200 OK` with a JSON list of document links (HATEOAS style)

**`GET /search/partition/{partition}`**  
Searches within a specific partition.

- **Inputs:**
  - `text` (query, required): string – Text to search semantically  
  - `top_k` (query, optional): int – Number of top results to return (default: `5`)

- **Returns:**
  - `200 OK` with a JSON list of document links (HATEOAS style)

**`GET /search/partition/{partition}/file/{file_id}`**  
Searches within a specific file in a partition.

- **Inputs:**
  - `text` (query, required): string – Text to search semantically  
  - `top_k` (query, optional): int – Number of top results to return (default: `5`)

- **Returns:**
  - `200 OK` with a JSON list of document links (HATEOAS style)

### 📄 Document Extract Details

**`GET /extract/{extract_id}`**  
Fetches a specific extract by its ID.

- **Returns:**
  - Extract text content  
  - Metadata (JSON)

### 💬 OpenAI-Compatible Chat

**`POST /v1/chat/completions`**  
OpenAI-compatible chat completion endpoint using a Retrieval-Augmented Generation (RAG) pipeline. Accepts `model`, `messages`, `temperature`, `top_p`, etc.

### ℹ️ Utils

**`GET /health_check`**

Simple endpoint to ensure the server is running.

---

## 🏗️ Architecture

```mermaid
graph TD
    A[User Query] --> B[RAGondin Backend]
    B --> C[Retriever (Milvus)]
    B --> D[Chunker & Preprocessing]
    C --> E[Relevant Chunks]
    E --> F[LLM Completion]
    F --> G[Final Answer]
```

🧩 Designed for plug & play: each component can be swapped independently.

---

## 🔧 Contributing

We ❤️ contributions!

1. Fork the project
2. Create your feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -am 'feat: add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a pull request 🚀

Read our [CONTRIBUTING.md](CONTRIBUTING.md) for coding standards, test setup, and more.

---

## 📄 License

## License

RAGondin is released under the MIT License - See [LICENSE](LICENSE) file for details.
Feel free to use, remix, and build upon — with attribution.

---

## 🙌 Credits

A project initiated by [OpenLLM France](https://github.com/OpenLLM-France) and inspired by the spirit of libre experimentation.

---

## 🔗 Related Projects

- [LUCIE-7B](https://github.com/OpenLLM-France/lucie-7b) – Open LLM from scratch
- [OpenLLM France](https://openllm.fr) – French-speaking community for open-source generative AI