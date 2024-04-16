# RAG 

![](RAG_architecture.png)

To run the qdrant docker container: 

```
docker run -p 6333:6333  -v $(pwd)/data/qdrant/storage:/qdrant/storage     qdrant/qdrant
```

