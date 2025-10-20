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

Uso CLI (esempio):
python bm25_retriever.py
"""

import json
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

    passages = []
    top_contents = []

    for i in range(len(hits)):
        jsondoc = json.loads(hits[i].lucene_document.get('raw'))
        passages = (jsondoc["contents"])
        top_contents.append(passages)

    return top_contents

def main():
    query = "what is gravitational time dilation?"
    print(f"BM25 example with query: {query}")
    docs= bm25_retrieve(query, k=5)
    for doc in docs:
        print(f"Text: {doc[:400]}...")
    
    
if __name__=="__main__":
    main()