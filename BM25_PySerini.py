import json
from pyserini.search.lucene import LuceneSearcher


# BASH COMMAND
# """ python -m pyserini.index.lucene \
#   -collection JsonCollection \
#   -input ./Progetto/Files/Passages \
#   -index ./Progetto/Files/indexes \
#   -generator DefaultLuceneDocumentGenerator \
#   -threads 1 \
#   -storePositions -storeDocvectors -storeRaw
# """


def bm25_pyserini_retrieve(query, k=10):
    """
    Execute the search using Okapi BM25 implemented by PySerini
    :param query: user query
    :param k: numeber of passages to be retrieved
    :return: lists of top k documents with relative scores and indices
    """
    searcher = LuceneSearcher('./Files/indexes')
    hits = searcher.search(query, k=k)

    passages = []
    # forse non vanno passati tutti in uscita
    top_contents = []
    top_scores = []
    top_ids = []

    for i in range(len(hits)):
        jsondoc = json.loads(hits[i].lucene_document.get('raw'))
        passages = (jsondoc["contents"])

        top_contents.append(passages)
        top_ids.append(hits[i].docid)
        top_scores.append(hits[i].score)

    return top_ids, top_contents, top_scores


# # Example
def bm25_example():
    query = "what is gravitational time dilation?"
    indices, docs, scores = bm25_pyserini_retrieve(query, k=5)
    for idx, doc, score in zip(indices, docs, scores):
        print(f"  - Index: {idx}, Score: {score:.4f}\n    Text: {doc[:200]}...")


# bm25_example()
