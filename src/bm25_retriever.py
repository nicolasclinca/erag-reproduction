"""
bm25_retriever.py
BM25 retrieval over a pre-built Lucene (PySerini) index.
Returns the top-k most relevant passages for a query.

Notes:

The index must already be built with PySerini (see example below).
Index construction (run once):
python -m pyserini.index.lucene \
    -collection JsonCollection \
    -input ../data/collection \
    -index ../indexes/bm25_index \
    -generator DefaultLuceneDocumentGenerator \
    -threads 8 \
    -storePositions -storeDocvectors -storeRaw

CLI usage:

Single query:
python bm25_retriever.py --bm25_index_dir ../indexes/bm25_index \
    --query "When did Apollo 11 land?" \
    --k 5 --threads 8
"""

import json
import argparse
from typing import List, Dict, TypedDict
from pyserini.search.lucene import LuceneSearcher


class RetrievedDoc(TypedDict):
    doc_id: str
    score: float
    contents: str


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
        doc = searcher.doc(hit.docid)
        jsondoc = json.loads(doc.raw())
        top_contents.append(jsondoc["contents"])

    return top_contents


def bm25_batch_retrieve(
    queries: List[str],
    searcher: LuceneSearcher = None,
    k: int = 50,
    batch_size: int = 256,
    threads: int = 8,
) -> Dict[str, List[RetrievedDoc]]:
    """
    Execute batch search using Okapi BM25 for multiple queries.

    Returns:
        {query: [{"doc_id": str, "score": float, "contents": str}, ...]}
    """
    if searcher is None:
        raise ValueError("A LuceneSearcher instance must be provided.")

    results: Dict[str, List[RetrievedDoc]] = {}

    for i in range(0, len(queries), batch_size):
        batch = queries[i : i + batch_size]
        qids = [str(j) for j in range(len(batch))]
        batch_hits = searcher.batch_search(batch, qids=qids, k=k, threads=threads)

        for idx, query in enumerate(batch):
            qid = str(idx)
            hits = batch_hits.get(qid, []) or []

            out_list: List[RetrievedDoc] = []
            for hit in hits:
                doc_id = str(hit.docid)
                score = float(getattr(hit, "score", 0.0))

                contents = ""
                try:
                    doc = searcher.doc(hit.docid)
                    if doc is not None:
                        raw = doc.raw()
                        js = json.loads(raw) if raw else {}
                        contents = js.get("contents") or js.get("text") or ""
                except Exception:
                    contents = ""

                out_list.append({"doc_id": doc_id, "score": score, "contents": contents})

            results[query] = out_list

    return results


def create_bm25_searcher(index_dir: str) -> LuceneSearcher:
    return LuceneSearcher(index_dir)


def main():
    parser = argparse.ArgumentParser(description="BM25 retrieval (PySerini) with optional batch mode and threads control.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--bm25_index_dir", required=True, help="BM25 index directory (PySerini)")
    parser.add_argument("--query", required=True, help="Single query to search")
    parser.add_argument("--k", type=int, default=50, help="Number of documents to retrieve per query")
    args = parser.parse_args()

    searcher = create_bm25_searcher(args.bm25_index_dir)
    docs = bm25_retrieve(args.query, searcher=searcher, k=args.k)
    print(json.dumps({args.query: docs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()