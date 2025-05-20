import json
import torch
import faiss
import torch.nn.functional as F
from pyserini.search.lucene import LuceneSearcher
from build_indexes import *


def bm25_pyserini_retrieve(query, k=10):
    """
    Execute the search using Okapi BM25 implemented by PySerini
    :param query: user query
    :param k: numeber of passages to be retrieved
    :return: lists of top k documents with relative scores and indices
    """
    searcher = LuceneSearcher('../indexes/bm25_index')
    hits = searcher.search(query, k=k)

    passages = []
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



# Dense retrieval function using Contriever and Faiss (Inner Product)
def dense_retrieve(query, index, original_docs, k=5, max_length=256):
    """
    Execute a dense retrieval search: retrieves the top-k documents for a query using the Contriever model
    and a Faiss index (expecting normalized vectors and IP metric).
    Maintains the original function signature.
    :param query: the query for the LLM
    :param index: doc indices
    :param original_docs:
    :param k: only top k documents are returned
    :param max_length:
    :return: the top k documents' indices, content and distances to the query
    """
    tokenizer, model, device = setup_dense_retriever()
    model.eval()
    with torch.no_grad():
        # Tokenize query and move to device
        inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get model outputs
        outputs = model(**inputs)

        # Perform the same mean pooling as for documents
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        mean_embedding = sum_embeddings / sum_mask

        # Normalize the query embedding - Crucial for Inner Product
        normalized_embedding = F.normalize(mean_embedding, p=2, dim=1)

        # Move to CPU and convert to NumPy array
        query_embedding_np = normalized_embedding.cpu().numpy()

    # Faiss requires normalized query vector for IndexFlatIP search
    faiss.normalize_L2(query_embedding_np)

    # Search the Faiss index
    distances, indices = index.search(query_embedding_np, k)

    top_docs = [original_docs[i] for i in indices[0]]

    # Return indices and distances (scores)
    return indices[0], top_docs, distances[0]


def retrieve_documents(query, method='BM25', k=5):
    """
    Select Retrieve Method between Okapi BM25 and Meta Contriever
    :param query: user query
    :param method: the retrieval method (default: BM25)
    :param k: number of documents to be retrieved
    :return: list of the k most relevant documents
    """
    if method == 'BM25':
        _, docs, _ = bm25_pyserini_retrieve(query, k)
        return docs
    #elif method == 'dense':
    #    _, docs, _ = dense_retrieve(query, index, original_docs=documents, k=k)
    #    return docs
    else:
        raise ValueError("Unsupported retrieval method. Use 'BM25' or 'dense'.")
    
    
    
if __name__=="__main__":
    print("Esempio con BM25")
    bm25_example()