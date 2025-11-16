"""
bm25_retriever.py
Retrieval BM25 su indice Lucene (PySerini) pre-costruito.
Restituisce i top-k passaggi più rilevanti per una query.

Note:

L'indice deve essere già costruito con PySerini (vedi esempio sotto).
Costruzione indice (eseguire una volta):
python -m pyserini.index.lucene \
    -collection JsonCollection \
    -input ../data/collection \
    -index ../indexes/bm25_index \
    -generator DefaultLuceneDocumentGenerator \
    -threads 8 \
    -storePositions -storeDocvectors -storeRaw

Uso CLI:

Singola query:
python bm25_retriever.py --bm25_index_dir ../indexes/bm25_index \
    --query "When did Apollo 11 land?" \
    --k 5 --threads 8

File con query (una per riga):
python bm25_retriever.py --bm25_index_dir ../indexes/bm25_index \
    --queries_file ./queries.txt \
    --k 50 --batch_size 256 --threads 8
"""

import json
import argparse
from typing import List, Dict
from pyserini.search.lucene import LuceneSearcher


def bm25_retrieve(query: str, searcher: LuceneSearcher = None, k: int = 50) -> List[str]:
    """
    Execute the search using Okapi BM25 implemented by PySerini
    :param query: user query
    :param k: number of passages to retrieve
    :return: list of top documents (passages)
    """
    if searcher is None:
        raise ValueError("A LuceneSearcher instance must be provided.")
    hits = searcher.search(query, k=k)

    top_contents = []
    for hit in hits:
        jsondoc = json.loads(hit.raw)
        top_contents.append(jsondoc["contents"])

    return top_contents


def bm25_batch_retrieve(
    queries: List[str],
    searcher: LuceneSearcher = None,
    k: int = 50,
    batch_size: int = 256,
    threads: int = 8,
) -> Dict[str, List[str]]:
    """
    Execute batch search using Okapi BM25 for multiple queries
    :param queries: list of user queries
    :param k: number of passages to retrieve per query
    :param batch_size: number of queries to process in parallel
    :param threads: number of threads for PySerini batch_search
    :return: dictionary mapping each query to its list of top document contents
    """
    if searcher is None:
        raise ValueError("A LuceneSearcher instance must be provided.")

    results: Dict[str, List[str]] = {}

    for i in range(0, len(queries), batch_size):
        batch = queries[i : i + batch_size]
        qids = [str(j) for j in range(len(batch))]
        batch_hits = searcher.batch_search(batch, qids=qids, k=k, threads=threads)

        for idx, query in enumerate(batch):
            qid = str(idx)
            hits = batch_hits.get(qid, [])

            top_contents: List[str] = []
            for hit in hits:
                jsondoc = json.loads(hit.raw)
                top_contents.append(jsondoc["contents"])

            results[query] = top_contents

    return results


def create_bm25_searcher(index_dir: str) -> LuceneSearcher:
    return LuceneSearcher(index_dir)


def main():
    parser = argparse.ArgumentParser(description="BM25 retrieval (PySerini) with optional batch mode and threads control.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--bm25_index_dir", required=True, help="Directory dell'indice BM25 (PySerini)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", required=True, help="Singola query da cercare")
    parser.add_argument("--k", type=int, default=50, help="Numero di documenti da recuperare per query")
    args = parser.parse_args()

    searcher = create_bm25_searcher(args.bm25_index_dir)
    docs = bm25_retrieve(args.query, searcher=searcher, k=args.k)
    print(json.dumps({args.query: docs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()