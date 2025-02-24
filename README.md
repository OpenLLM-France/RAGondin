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
A **semantic chunker** is used to process all supported file types. Future releases will implement format-specific chunkers (e.g., specialized CSV chunking).

- **Indexing & Search**  
After chunking, data is indexed in a **Qdrant** vector database using the multilingual embedding model `HIT-TMG/KaLM-embedding-multilingual-mini-v1` (ranked highly on the MTEB benchmark). The same model embeds user queries for semantic search (Dense Search).  
    * **Hybrid Search**: Combines semantic search with keyword search (BM25) to handle domain-specific jargon and coded product names that might not exist in the embedding model's training data.

- **Retriever**  
Supports three retrieval modes:  
    * **Single Retriever**: Standard query-based document retrieval  
    * **MultiQuery**: Generates augmented query variations using an LLM, then combines results  
    * **HyDE**: Generates a hypothetical answer using an LLM, then retrieves documents matching this answer  

- **Grader**: Filters out irrelevant documents after retrieval  
- **Reranker**: Uses a multilingual reranking model to reorder documents by relevance  

- **RAG Types**:  
    * **SimpleRAG**: Basic implementation without chat history  
    * **ChatBotRAG**: Version that maintains conversation context  

## Configurations
- `.env`: Stores your LLM API key (`API_KEY`)  and your `BASE_URL`

## Usage

#### 1. Clone the repository:
```bash
git clone https://github.com/OpenLLM-France/RAGondin.git
git checkout main
```

#### 2. Create poetry environment and install dependencies:
Requirements: Python3.12 and poetry installed

```bash
# Create a new environment using Poetry
poetry env use python3.12
poetry config virtualenvs.in-project true

# Install dependencies
poetry install
```

#### 3. Run the Chainlit app
1. **Prepare Qdrant collection** (using `manage_collection.py`):
```bash
# Create/update collection (default collection from .hydra_config/config.yaml)
python3 manage_collection.py -f {folder_path} 

# Specify collection name
python3 manage_collection.py -f {folder_path} -c {collection_name}

# Delete collection
python3 manage_collection.py -d {collection_name}
```

2. **Launch the app and the api**:
```bash
# launch the api
uvicorn api:app --reload --port 8082 --host 0.0.0.0
```

```bash
# launch the frontend
chainlit run app_front.py --host 0.0.0.0 --port 8081 --root-path /chainlit
```
Access the chatbot interface at `http://localhost:8081/chainlit` (port may vary). The LLM will ground its responses in documents from your VectorDB.

3. **Copilot Mode**:  
Use the provided `test_copilot.html` template or create your own:
```html
<!doctype html>
<head>
<meta charset="utf-8" />
</head>
<body>
<script src="http://localhost:8081/copilot/index.js"></script>
<script>
window.mountChainlitWidget({
    chainlitServer: "http://localhost:8000",
    theme: "dark"
});
</script>
</body>
```

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