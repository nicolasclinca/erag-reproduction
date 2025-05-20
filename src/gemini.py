# Text generation with Gemini API
import google.generativeai as genai
from retrieval_models import *

# Configure Gemini API (replace with your API key)
gemini_key = "GEMINI_KEY_HERE"
genai.configure(api_key=gemini_key)


def gemini_text_generator(query, retrieved_docs):
    """
    Takes the user textual query and the contextual retrieved documents, and returns
    the answer by Gemini 2.0 Flash
    :param query: user query
    :param retrieved_docs: list of retrieved documents
    :return: textual Answer by Gemini AI
    """
    context = " ".join(retrieved_docs)
    prompt = f"Answer this question: {query}. Context: {context} answer: "
    response = genai.GenerativeModel('gemini-2.0-flash').generate_content(prompt)
    return response.text


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
    answer = gemini_text_generator(query, retrieved_docs)
    return answer


# Example usage:
def rag_pipeline_example():
    query = "What is gravitational time dilation?"
    answer_bm25 = rag_pipeline(query, retrieval_method="BM25", k=5)
    print("\nAnswer (BM25):", answer_bm25)

    # Test with dense retrieval
    #answer_dense = rag_pipeline(query, retrieval_method='dense', k=5)
    #print("\nAnswer (Dense):", answer_dense)


rag_pipeline_example()