1. Qui est Alexandre Zapolsky
2. Que signifie open weights
3. Qui est Agathe  BERTHELE?
4. Quels sont les fonctionnalités de LinShare
5. Parle moi des projets de Linagora avec la DTNUM
6. Quel est le projet de Linagora avec MAIF
7. Quel sont les evenements dont Linagora à participé
8. Fais-moi un résumé de l'evenement SOFINS 2023.

# FLAT Index
## COSINE
```python
{"metric_type": "BM25", "index_type": "SPARSE_INVERTED_INDEX"},  # For sparse vector*
{"metric_type": "COSINE", "index_type": "FLAT"},  # For dense vector
* reranker: rrf
```
* reranker: RRF
1. Qui est Alexandre Zapolsky? ==> Good Answer

2. Que signifie open weights? ==> Good

3. Qui est Agathe  BERTHELE? ==> Correct

4. Quels sont les fonctionnalités de LinShare ==> Correct Answer

5. Parle moi des projets de Linagora avec la DTNUM? ==> Not found

6. Quel est le projet de Linagora avec MAIF? ==> Correct Answer

7. Quel sont les evenements dont Linagora à participé? ==> Very Good

8. Fait moi un résumé de l'evenement SOFINS 2023. ==> Found

## IP
```python
{"metric_type": "BM25", "index_type": "SPARSE_INVERTED_INDEX"},  # For sparse vector*
{"metric_type": "IP", "index_type": "FLAT"},  # For dense vector
* reranker: rrf
```
* reranker: RRF
1. Qui est Alexandre Zapolsky? ==> Good 
2. Que signifie open weights? ==> Good
3. Qui est Agathe  BERTHELE? ==> Good 
4. Quels sont les fonctionnalités de LinShare
5. Parle moi des projets de Linagora avec la DTNUM? ==> Found
6. Quel est le projet de Linagora avec MAIF? 
7. Quel sont les evenements dont Linagora à participé? => Good
8. Fait moi un résumé de l'evenement SOFINS 2023. ==> Found

# IVF_FLAT_IP

```python
  collection_name: vdb_test_ivf_flat_ip

INDEX_PARAMS = [
    {"metric_type": "BM25", "index_type": "SPARSE_INVERTED_INDEX"},  # For sparse vector
    {
        "metric_type": "IP",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},
    },  # For dense vector
]


SEARCH_PARAMS = [
    {"metric_type": "IP", "params": {"nprobe": 10}},
    {"metric_type": "BM25", "params": {"drop_ratio_build": 0.2}},
]
```
1. Qui est Alexandre Zapolsky ==> Good
2. Que signifie "open weights" ==> Good
3. Qui est Agathe  BERTHELE? ==> Good
4. Quels sont les fonctionnalités de LinShare ==> Good
5. Parle moi des projets de Linagora avec la DTNUM
6. Quel est le projet de Linagora avec MAIF ==> Good
7. Quels sont les evenements dont Linagora à participé ==> Good
8. Fais-moi un résumé de l'evenement SOFINS 2023. ==> Good

# IVF_FLAT_COSINE

```python
collection_name: ivf_flat_cosine1
metric = "COSINE"
INDEX_PARAMS = [
    {"metric_type": "BM25", "index_type": "SPARSE_INVERTED_INDEX"},  # For sparse vector
    {
        "metric_type": metric,
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},
    },  # For dense vector
]


SEARCH_PARAMS = [
    {"metric_type": metric, "params": {"nprobe": 10}},
    {"metric_type": "BM25", "params": {"drop_ratio_build": 0.2}},
]
```

1. Qui est Alexandre Zapolsky ==> Good
2. Que signifie "open weights" ==> Good
3. Qui est Agathe  BERTHELE? ==> Good
4. Quels sont les fonctionnalités de LinShare
5. Parle moi des projets de Linagora avec la DTNUM
6. Quel est le projet de Linagora avec MAIF ==> Good
7. Quel sont les evenements dont Linagora à participé
8. Fais-moi un résumé de l'evenement SOFINS 2023.

---------------

```python
INDEX_PARAMS = [
    {"metric_type": "BM25", "index_type": "SPARSE_INVERTED_INDEX"},  # For sparse vector
    {"metric_type": "COSINE", "index_type": "FLAT"},  # For dense vector
]

SEARCH_PARAMS = None

########
metric = "COSINE"
INDEX_PARAMS = [
    {"metric_type": "BM25", "index_type": "SPARSE_INVERTED_INDEX"},  # For sparse vector
    {
        "metric_type": metric,
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},
    },  # For dense vector
]


SEARCH_PARAMS = [
    {"metric_type": metric, "params": {"nprobe": 10}},
    {"metric_type": "BM25", "params": {"drop_ratio_build": 0.2}},
]


```

# HNSW_IP

```python
INDEX_PARAMS = [
    {"metric_type": "BM25", "index_type": "SPARSE_INVERTED_INDEX"},  # For sparse vector
    {
        "metric_type": "IP",
        "index_type": "IVF_FLAT",
        "params": {"M": 16, "efConstruction": 40},
    },  # For dense vector
]

SEARCH_PARAMS = [
    {"metric_type": "IP", "params": {"ef": 10}},
    {"metric_type": "BM25", "params": {"drop_ratio_build": 0.2}},
]

# Duration: 2522.2139
```

1. Qui est Alexandre Zapolsky ==> Good
2. Que signifie "open weights" ==> Good
3. Qui est Agathe  BERTHELE? ==> Good
4. Quels sont les fonctionnalités de LinShare ==> Good
5. Est ce que Linagora travaille avec la DTNUM ==> Good
6. Quel est le projet de Linagora avec MAIF ==> 
7. Quel sont les evenements dont Linagora à participé ==>
8. Fais-moi un résumé de l'evenement SOFINS 2023. ==> Good


# IVF_HNSW


#############

* https://chat.deepseek.com/a/chat/s/340f605d-4014-477f-a231-df4a9aaf357a
* https://www.perplexity.ai/search/which-search-algo-is-the-best-lC5FLvrhToapaMLXQz94aw
* https://www.pinecone.io/learn/series/faiss/vector-indexes/
* Comparaison table: https://superlinked.com/vectorhub/articles/vector-indexes


-------------

# Tester

* Ajout de differents endpoints pour (un pour chat et un pour l'indexation)
* Pour utiliser et deployer uniquement les APIs du Search
* Tester sur GPUs & CPUs: QA (Quality assessment)
