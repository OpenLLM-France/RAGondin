* https://www.analyticsvidhya.com/blog/2024/07/hit-rate-mrr-and-mmr-metrics/
* https://www.evidentlyai.com/ranking-metrics/precision-recall-at-k

## Hit Rate
**`Definition`**: Le hit rate, c’est le pourcentage de fois où on obtient le bon chunk parmi toutes les chunks récupérés.

![alt text](./assets/image-1.png)
![alt text](./assets/image.png)

> Pb with Hit Rate: Cela ne prend pas en compte la position du chunk pertinent parmi ceux récupérés. Dans l'exemple suivant on a 2 retrievers qui ont le même hit rate et pourtant le retriever 2 classe mieux les documents pertinents  (à la première position) et clairement ce retriever serait plus preferable: C'est là que le MRR (Mean Reciprocal Rank) est considéré
![alt text](./assets/image-2.png)

## MRR
**`Definition`**: Le MRR, indique à quelle position moyenne se trouve la première bonne réponse.

> **`Interprétation`** :
* MRR proche de 1: le bon résultat est souvent en tête.
* MRR proche de 0: le bon résultat est rarement trouvé ou très bas dans la liste.

![alt text](./assets/image-3.png)
![alt text](./assets/image-4.png)
https://huggingface.co/datasets/maastrichtlawtech/bsard

https://huggingface.co/datasets/antoinelb7/alloprof

https://huggingface.co/datasets/lyon-nlp/mteb-fr-retrieval-syntec-s2p
 
https://huggingface.co/collections/lyon-nlp/mteb-french-68013afc32b4a4201abf2f7b
 