from BM25_PySerini import bm25_pyserini_retrieve
from Gemini_LLM import text_generator


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


# Main RAG pipeline function
def rag_pipeline(query, retrieval_method='BM25', k=5):
    """
    General RAG Pipeline: takes query, retrieval method and number of documents to be retrieved.
    :param query: user query
    :param retrieval_method: method used to retrieve docs (default: BM25)
    :param k: number of documents to be retrieved (default: 5)
    :return: the answer to the query
    """
    retrieved_docs = retrieve_documents(query, method=retrieval_method, k=k)
    print(f"Retrieved documents ({retrieval_method}):")
    for i, doc in enumerate(retrieved_docs):
        print(f"{i + 1}. {doc[:150]}...")
    # Generate response using Gemini
    answer = text_generator(query, retrieved_docs)
    return answer


# Example usage:
def rag_pipeline_example():
    query = "What is gravitational time dilation?"
    answer_bm25 = rag_pipeline(query, retrieval_method='BM25', k=5)
    print("\nAnswer (BM25):", answer_bm25)

    # Test with dense retrieval
    #answer_dense = rag_pipeline(query, retrieval_method='dense', k=5)
    #print("\nAnswer (Dense):", answer_dense)


rag_pipeline_example()
