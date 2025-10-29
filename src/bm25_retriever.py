"""
bm25_retriever.py
Retrieval BM25 su indice Lucene (PySerini) pre-costruito.
Restituisce i top-k passaggi più rilevanti per una query.

Note:

L'indice deve essere già costruito con PySerini (vedi esempio sotto).
Hardcoded path: '../indexes/bm25_index'
Costruzione indice (eseguire una volta):
python -m pyserini.index.lucene
-collection JsonCollection
-input ../data/collection
-index ../indexes/bm25_index
-generator DefaultLuceneDocumentGenerator
-threads 8
-storePositions -storeDocvectors -storeRaw
"""

import json
from typing import List, Dict
from pyserini.search.lucene import LuceneSearcher


def bm25_retrieve(query, k=10):
    """
    Execute the search using Okapi BM25 implemented by PySerini
    :param query: user query
    :param k: number of passages to retrieve
    :return: list of top documents (passages)
    """
    searcher = LuceneSearcher('../indexes/bm25_index')
    hits = searcher.search(query, k=k)

    top_contents = []

    for hit in hits:
        jsondoc = json.loads(hit.raw)
        top_contents.append(jsondoc["contents"])

    return top_contents


def bm25_batch_retrieve(
    queries: List[str],
    k: int = 50,
    batch_size: int = 256
) -> Dict[str, List[str]]:
    """
    Execute batch search using Okapi BM25 for multiple queries
    :param queries: list of user queries
    :param k: number of passages to retrieve per query
    :param batch_size: number of queries to process in parallel
    :return: dictionary mapping each query to its list of top document contents
    """
    searcher = LuceneSearcher('../indexes/bm25_index')
    results = {}
    
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        
        batch_hits = searcher.batch_search(batch, qids=[str(j) for j in range(len(batch))], k=k, threads=8)
        
        for idx, query in enumerate(batch):
            qid = str(idx)
            hits = batch_hits.get(qid, [])
            
            top_contents = []
            for hit in hits:
                jsondoc = json.loads(hit.raw)
                top_contents.append(jsondoc["contents"])
            
            results[query] = top_contents
    
    return results