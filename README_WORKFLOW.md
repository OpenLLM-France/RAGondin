# RAGondin workflow Documentation

## The Indexer process

By running the command:

```bash
uv run python3 ragondin/manage_collection.py -f '/data'
```
the Indexer class will be trigered and all the files in **/data** will be vectorized through " processes:

- **Serialize**
- **Chunking**
- **Vectorize**

Then, all the chunks will be stored inside the **./vdb/volumes/milvus** folder.

For more details, check the documentation in **./ragondin/components/indexer/indexer.py** and to modify chunker configuration, check the settings in **./.hydra_config/chunker/**

## Retrieval, Augmentation and Generation: Related chunks + LLM inference retrieved

We then put our question in form of a prompt via the web interface. This action will trigger API calls through './ragondin/chainlit/' then to './ragondin/api.py'. The full pipeline (inside *./ragondin/components/pipeline.py*) will be executed:

### 1, Retrieval: Related chunks retrieved
- **Question contextualization**: Retrieve the document chunks that are related to the prompts
### 2, Augmentation: Get the top k most related document chunks
- **Documents rerank**: Using the **RAGPretrainedModel** to return a given number of document strings which are considered the most related.
### 3, Generation:
- **Reformat the answer and give the source**
- **Get the inference by using llm**

Once finished, the answer will be yielded by StreamingResponse and the web interface will display the answer.
