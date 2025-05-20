# RAGondin 

RAGondin is a project dedicated to experimenting with advanced RAG (Retrieval-Augmented Generation) techniques to improve the quality of such systems. We start with a vanilla implementation and build up to more advanced techniques to address challenges and edge cases in RAG applications.  

![](RAG_architecture.png)

## Goals
- Experiment with advanced RAG techniques
- Develop evaluation metrics for RAG applications
- Collaborate with the community to innovate and push the boundaries of RAG applications

## Current Features
This section provides a detailed explanation of the currently supported features.

The **`.hydra_config`** directory contains all the configuration files for the application. These configurations are structured using the [Hydra configuration framework](https://hydra.cc/docs/intro/). This directory will be referenced for setting up the RAG (Retrieval-Augmented Generation) pipeline.

### Supported File Formats
This branch currently supports the following file types:

* **Text and Document Files**: `txt`, `pdf`, `docx`, `doc`, `pptx`
* **Audio Files**: `wav`, `mp3`, `mp4`, `ogg`, `flv`, `wma`, `aac`

For all supported formats, content is converted to **Markdown**. Images within documents are replaced with captions generated using a **Vision Language Model (VLM)**. (See the **Configuration** section for more details.) The resulting Markdown is then chunked and indexed using the [Milvus vector database](https://milvus.io/).


> [!NOTE]
> **Upcoming support**: Future updates aim to include additional formats such as `csv`, `odt`, `html`, and image file types.

### Chunking
Several chunking strategies are supported, including **`semantic`** and **`recursive`**. By default, the **recursive chunker** is applied to all supported file types for its efficiency and low memory usage. This is the **recommended strategy** for most use cases. Future updates may introduce format-specific chunkers (e.g., for CSV, Markdown, etc.). The recursive chunker configuration is located at: *`.hydra_config/chunker/recursive_splitter.yaml`*.

```yml
# .hydra_config/chunker/recursive_splitter.yaml
defaults:
  - base
name: recursive_splitter
chunk_size: 1000
chunk_overlap: 100
```

The **`chunk_size`** and **`chunk_overlap`** values are expressed in **tokens**, not characters. For enhanced retrieval, enable the **contextual retrieval** feature—a technique introduced by Anthropic to improve retrieval performance ([Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)). To activate it, set **`CONTEXTUAL_RETRIEVAL=true`** in your **`.env`** file. Refer to the **`Usage`** section for further instructions.


### **Indexing**

Once chunked, document fragments are ingested into the **Milvus** vector database using the `jinaai/jina-embeddings-v3` multilingual embedding model—top performer on the [MTEB benchmark](https://huggingface.co/spaces/mteb/leaderboard). To switch models, set the **`EMBEDDER_MODEL`** variable in your **`.env`** file to any Hugging Face–compatible alternative (e.g., `"sentence-transformers/all-MiniLM-L6-v2"` for faster throughput).

> \[!IMPORTANT]
> Choose an embedding model that aligns with your document languages and offers a suitable context window. The default model supports both English and French.

---

### **Retriever & Search**

#### Search Pipeline

After indexing, you can execute searches via our **hybrid search** pipeline, which blends **semantic search** with **BM25** keyword matching. This hybrid approach ensures you retrieve both topically relevant chunks and those containing exact keywords.

#### Retrieval Strategies

Three strategies feed into the hybrid search—available only in the **RAG pipeline** (see **OpenAI Compatible API**), not in the standalone **Semantic Search** endpoints:

* **multiQuery**: Uses an LLM to generate multiple query reformulations, merging their results for superior relevance (default and most effective based on benchmarks).
* **single**: Executes a single, straightforward retrieval query.
* **HyDE**: Generates a hypothetical answer via LLM, then retrieves chunks similar to that synthetic response.

---

#### **Reranker**

Finally, retrieved documents are re-ordered by relevance using a multilingual reranking model. By default, we employ **`jinaai/jina-colbert-v2`** from Hugging Face.

> [!IMPORTANT]
> The retriever fetches documents that are semantically similar to the query. However, semantic similarity doesn't always equate to relevance. Therefore, rerankers are crucial for reordering results and filtering out less pertinent documents, thereby reducing the likelihood of hallucinations.

### RAG Type
* **SimpleRAG**: Basic implementation without chat history taken into account.
* **ChatBotRAG**: Version that maintains conversation context. 

## Usage

### 1. Clone the repository:
```bash
git clone https://github.com/OpenLLM-France/RAGondin.git
cd RAGondin
git checkout main # or a given release
```

### 2. Create uv environment and install dependencies:
>[!IMPORTANT] 
> Ensure you have Python 3.12 installed along with `uv`. For detailed installation instructions for uv, refer to the [uv official documentation](https://docs.astral.sh/uv/getting-started/installation/#pypi).

* To install `uv`, you can use either `pip` (if already available) or `curl`. Additional installation methods are outlined in the [documentation](https://docs.astral.sh/uv/getting-started/installation/#pypi).
```bash
# with pip
pip install uv

# with curl
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
# Create a new environment with all dependencies
cd RAGondin/
uv sync
```

### 3. Create a `.env` File

Create a `.env` file at the root of the project, mirroring the structure of `.env.example`, to configure your environment.

* Define your LLM-related variables (`API_KEY`, `BASE_URL`, `MODEL_NAME`) and VLM (Vision Language Model) settings (`VLM_API_KEY`, `VLM_BASE_URL`, `VLM_MODEL_NAME`).

> [!IMPORTANT]
> The **VLM** is used for generating captions from images extracted from files, and for tasks like summarizing chat history. The same endpoint can serve as both your LLM and VLM.

* For PDF indexing, multiple loader options are available. Set your choice using the **`PDFLoader`** variable:

  * **`MarkerLoader`** and **`DoclingLoader`** are recommended for optimal performance, especially on OCR-processed PDFs. They support both GPU and CPU execution, though CPU runs may be slower.
  * For lightweight testing on CPU, use **`PyMuPDF4LLMLoader`** or **`PyMuPDFLoader`**.

    > ⚠️ These do **not** support non-searchable PDFs or image-based content.


* Audio & Video Files
  - Audio and video content is transcribed using OpenAI’s **Whisper** model. Supported model sizes include: `tiny`, `base`, `small`, `medium`, `large`, and `turbo`. For details, refer to the [OpenAI Whisper repository](https://github.com/openai/whisper).The default transcription model is set via the **`WHISPER_MODEL`** variable, which defaults to `'base'`.

Other file formats (`txt`, `docx`, `doc`, `pptx`) are pre-configured.

* set `AUTH_TOKEN` to enable HTTP authentication via HTTPBearer. If not provided the endpoints will be accessible without any restrictions

```bash
# This is the minimal settings required.

# LLM settings
BASE_URL=
API_KEY=
MODEL=
LLM_SEMAPHORE=10

# VLM settings
VLM_BASE_URL=
VLM_API_KEY=
VLM_MODEL=
VLM_SEMAPHORE=10

# App
APP_PORT=8080
APP_HOST=0.0.0.0

# RAY
RAY_DEDUP_LOGS=0
RAY_DASHBOARD_PORT=8265

# To enable HTTP authentication via HTTPBearer for the api endpoints
AUTH_TOKEN=super-secret-token

# Loaders
PDFLoader=DoclingLoader

# Audio
WHISPER_MODEL=base

# Vector db VDB Milvus
VDB_HOST=milvus
VDB_CONNECTOR_NAME=milvus

# EMBEDDER
EMBEDDER_MODEL=jinaai/jina-embeddings-v3

# RETRIEVER
CONTEXTUAL_RETRIEVAL=true
RETRIEVER_TOP_K=12

# RERANKER
RERANKER_ENABLED=false
RERANKER_MODEL=jinaai/jina-reranker-v2-base-multilingual
RERANKER_MODEL_TYPE=crossencoder
RERANKER_TOP_K=5 # number of documents to return after reranking
```
* **Running on CPU**:
  For quick testing on CPU, you can reduce computational load by adjusting the following settings in the **`.env`** file:

- Set **`RERANKER_TOP_K=5`** or lower to limit the number of documents returned by the reranker. This defines how many documents are included in the LLM's context—reduce it if your LLM has a limited context window (4–5 is usually sufficient for an 8k context). You can also disable the reranker entirely with **`RERANKER_ENABLED=false`**, as it is a costly operation.

> [!WARNING]
> These adjustments may affect performance and result quality but are appropriate for lightweight testing.


* **Running on GPU**:
The default values are well-suited for GPU usage. However, you can adjust them as needed to experiment with different configurations based on your machine’s capabilities.

### 4.Deployment: Launch the app
The application can be launched in either in GPU or CPU environment, depending on your device's capabilities. Use the following commands:

```bash
# Launch with GPU support (recommended for faster processing)
docker compose up --build # or 'down' # to stop it

# Launch with CPU only
docker compose --profile cpu up --build # or '--profile cpu down' to stop it
```

Once it is running, you can check everything is fine by doing:
```bash
curl http://localhost:8080/health_check
```

> [!IMPORTANT]
> The initial launch is longer due to the installation of required dependencies. Once the application is up and running, you can access the fastapi documentation at `http://localhost:8080/docs` (8080 is the APP_PORT variable determined in your **`.env`**) to manage documents, execute searches, or interact with the RAG pipeline (see the **next section** about the api for more details). A default chat ui is also deployed using [chainlit](!https://docs.chainlit.io/get-started/overview). You can access to it at `http://localhost:8080/chainlit` chat with your documents with our RAG engine behind it.


### 5. Distributed deployment in a Ray cluster

To scale RAGondin across multiple machines using Ray, follow these steps:

---

#### ✅ 1. Set environment variables

Make sure to set your `.env` file with the necessary variables. In addition to the usual ones, add **Ray-specific variables** from `.env.example`:
```env
RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
CONFIG_PATH=/ray_mount/.hydra_config
DATA_DIR=/ray_mount/data
HF_HOME=/ray_mount/model_weights
HF_HUB_CACHE=/ray_mount/model_weights/hub
DATA_VOLUME_DIRECTORY=/app/volumes
SHARED_ENV=/ray_mount/.env

RAY_NUM_CPUS=4
RAY_NUM_GPUS=0.6
```

The last 2 settings define the per-actor resource requirements. Adjust them according to your workload and GPU size. For example, a single indexation uses at most 7GB VRAM, and each GPU has 16GB, setting `RAY_NUM_GPUS=0.5` allows **2 concurrent indexers per node**, meaning a 2-node cluster can handle **up to 4 concurrent indexation tasks**.

---

#### 📁 2. Set up shared storage (e.g. with NFS)

Workers need shared access to:
- `.env`
- `.hydra_config`
- SQLite DB (`/volumes`)
- Uploaded files (`/data`)
- Model weights (e.g. `/model_weights` if using HF local cache)

Example using **NFS**:

**On the Ray head node (NFS server):**

```bash
sudo mkdir -p /ray_mount
```

Add to `/etc/exports`:
```
/ray_mount 192.168.42.0/24(rw,sync,no_subtree_check)
```

Update and enable the NFS service:
```bash
sudo exportfs -a
sudo systemctl restart nfs-kernel-server
```

**On other Ray nodes (clients):**

```bash
sudo mount -t nfs 192.168.42.226:/ray_mount /ray_mount
```

To make the mount persistent:
```bash
echo "192.168.42.226:/ray_mount /ray_mount nfs defaults 0 0" | sudo tee -a /etc/fstab
```

Then copy required files to the shared mount:
```bash
sudo cp -r .hydra_config /ray_mount/
sudo cp .env /ray_mount/
sudo mkdir /ray_mount/volumes /ray_mount/data /ray_mount/model_weights
sudo chown -R ubuntu:ubuntu /ray_mount
```

---

#### 🐳 3. Launch `ragondin` with Ray in host mode

Use a dedicated Ray-enabled docker compose file:

```bash
docker compose -f docker-compose-ray.yaml up -d
```

The `ragondin` container must run in `host` mode so that the Ray head node is reachable by other machines. Make sure `RAY_RUNTIME_ENV_HOOK` is set properly to support `uv`:

```env
RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
```

---

#### 🔗 4. Join the cluster from other nodes

Run the following on other machines to connect to the Ray cluster:

```bash
uv run ray start --address='192.168.201.85:6379'
```

Or if you're not using `uv`:

```bash
ray start --address='192.168.201.85:6379'
```

Replace `192.168.201.85` with your head node’s actual IP address.



### 6. 🧠 API Overview

This FastAPI-powered backend offers capabilities for document-based question answering (RAG), semantic search, and document indexing across multiple partitions. It exposes endpoints for interacting with a vector database and managing document ingestion, processing, and querying.

---
### 📍 Endpoints Summary

For all the following endpoints, make sure to include your authentication token **AUTH_TOKEN** in the HTTP request header if authentication is enabled.
---

#### 📦 Indexer

**`POST /indexer/partition/{partition}/file/{file_id}`**  
Uploads a file (with optional metadata) to a specific partition.

- **Inputs:**
  - `file` (form-data): binary – File to upload  
  - `metadata` (form-data): JSON string – Metadata for the file (e.g. `{"file_type": "pdf"}`)

- **Returns:**
  - `201 Created` with a JSON containing the task status URL
  - `409 Conflit` if a file with the same id in the same partition already exists

---

**`PUT /indexer/partition/{partition}/file/{file_id}`**  
Replaces an existing file in the partition. Deletes existing entry and creates a new indexation task.

- **Inputs:**
  - `file` (form-data): binary – File to upload  
  - `metadata` (form-data): JSON string – Metadata for the file (e.g. `{"file_type": "pdf"}`)

- **Returns:**
  - `202 Accepted` with a JSON containing the task status URL

---

**`PATCH /indexer/partition/{partition}/file/{file_id}`**  
Updates the metadata of an existing file without reindexing.

- **Inputs:**
  - `metadata` (form-data): JSON string – Metadata for the file (e.g. `{"file_type": "pdf"}`)

- **Returns:**
  - `200 Ok` if metadata for that file is successfully updated


---

**`DELETE /indexer/partition/{partition}/file/{file_id}`**  
Deletes a file from a specific partition.

- **Returns:**
  - `204 No content`
  - `404 Not found` if the file is not found in the partition


> [!NOTE]  
> Once the rag is running you can attack these endpoints in order to index multiple files. Check this [data_indexer.py](./utility/data_indexer.py) in the [ 📁utility](./utility/) folder.


---

**`GET /indexer/task/{task_id}`**  
Retrieves the status of an asynchronous indexing task (see the **`POST /indexer/partition/{partition}/file/{file_id}`** endpoint).


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
  - `400 bad request` if the field `partitions` isn't correctly set

---

**`GET /search/partition/{partition}`**  
Searches within a specific partition.

- **Inputs:**
  - `text` (query, required): string – Text to search semantically  
  - `top_k` (query, optional): int – Number of top results to return (default: `5`)

- **Returns:**
  - `200 OK` with a JSON list of document links (HATEOAS style)
  - `400 bad request` if the field `partitions` isn't correctly set

---

**`GET /search/partition/{partition}/file/{file_id}`**  
Searches within a specific file in a partition.

- **Inputs:**
  - `text` (query, required): string – Text to search semantically  
  - `top_k` (query, optional): int – Number of top results to return (default: `5`)

- **Returns:**
  - `200 OK` with a JSON list of document links (HATEOAS style)
  - `400 bad request` if the field `partitions` isn't correctly set
---

#### 📄 Document Extract Details

**`GET /extract/{extract_id}`**  
Fetches a specific extract by its ID.

- **Returns:**
  - `content` and `metadata` of the extract (an extract is a chunk) inn JSON format

---

#### 💬 OpenAI-Compatible Chat

OpenAI API compatibility enables seamless integration with existing tools and workflows that follow the OpenAI interface. It makes it easy to use popular UIs like **`OpenWebUI`** without the need for custom adapters.

For the following OpenAI-compatible endpoints, when using an OpenAI client, provide your `AUTH_TOKEN` as the `api_key` if authentication is enabled; otherwise, you can use any placeholder value such as `'sk-1234'`.

* **`GET /v1/models`**
This endpoint allows to list all existant **`models`**

> [!NOTE]  
> Model names follow the pattern **`ragondin-{partition_name}`**, where **`partition_name`** refers to a data partition containing specific files. These “models” aren’t standalone LLMs (like GPT-4 or Llama), but rather placeholders that tell your LLM endpoint to generate responses using only the data from the chosen partition. To query the entire vector database, use the special model name **`partition-all`**.


* **`POST /v1/chat/completions`**  
OpenAI-compatible chat completion endpoint using a Retrieval-Augmented Generation (RAG) pipeline. Accepts `model`, `messages`, `temperature`, `top_p`, etc.

* **`POST /v1/completions`**
Same for this endpoint


> [!TIP]
To test these endpoint with openai client, you can refer to the the [openai_compatibility_guide.ipynb](./utility/openai_compatibility_guide.ipynb) notebook from the [📁 utility](./utility/) folder
---

#### ℹ️ Utils

**`GET /health_check`**

Simple endpoint to ensure the server is running.


## Contribute
Contributions are welcome! Please follow standard GitHub workflow:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Disclaimer
This repository is for research and educational purposes only. While we strive for correctness, we cannot guarantee fitness for any particular purpose. Use at your own risk.

## License
MIT License - See [LICENSE](LICENSE) file for details.

## Troubleshooting

### Error on dependencies installation

After running `uv sync`, if you have this error:

```bash
error: Distribution `ray==2.43.0 @ registry+https://pypi.org/simple` can't be installed because it doesn't have a source distribution or wheel for the current platform

hint: You're using CPython 3.13 (`cp313`), but `ray` (v2.43.0) only has wheels with the following Python ABI tag: `cp312`
```

This means your uv installation relies on cpython 3.13 while you are using python 3.12.

To solve it, please run:
```bash
uv venv --python=3.12
uv sync
```

## TODO
[] Better manage logs