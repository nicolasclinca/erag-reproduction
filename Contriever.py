

import torch
import faiss
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# Flag to use GPU
use_gpu = True
if use_gpu:
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = 'cpu'

# --- Model Loading and Device Setup ---
print("Loading Contriever and tokenizer...")
model_name = "facebook/contriever"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

model.to(device)
print(f"Model moved to device: {device}")
# --- End Model Loading ---


def encode_documents(docs, batch_size=16, max_length=256):
    """
    Encodes a list of documents into normalized embeddings using Contriever.
    Maintains the original function signature.
    :param docs: list of documents to be embedded
    :param batch_size: size of the sample batch
    :param max_length: max length of the tensor
    :return: embedding vector for dense retrival
    """
    all_embeddings = []
    model.eval()

    if use_gpu and torch.cuda.is_available():
      model.to(device)

    with torch.no_grad():
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i:i+batch_size]
            # Tokenize and move tensors to the correct device
            inputs = tokenizer(batch_docs, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get model outputs
            outputs = model(**inputs)

            # Perform mean pooling (weighted by attention mask)
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask

            # Normalize embeddings to unit length (L2 norm) - Crucial for Inner Product
            normalized_embeddings = F.normalize(mean_embeddings, p=2, dim=1)

            # Move embeddings to CPU and convert to NumPy
            all_embeddings.append(normalized_embeddings.cpu().numpy())

    # Concatenate all batch embeddings into a single NumPy array
    embeddings = np.concatenate(all_embeddings, axis=0)
    return embeddings


print("Encoding documents...")
doc_embeddings = encode_documents(documents, batch_size=16)
print("Document embedding shape:", doc_embeddings.shape)

# --- Build Faiss Index using Inner Product ---
embedding_dim = doc_embeddings.shape[1]
# Use IndexFlatIP for Inner Product similarity
index = faiss.IndexFlatIP(embedding_dim)

"""
Faiss *requires* normalized vectors for IndexFlatIP.
Although encode_documents already returns normalized vectors,
it's good practice and sometimes necessary for specific Faiss indexes
to explicitly normalize *again* right before adding.
"""
print("Normalizing embeddings for Faiss (redundant but safe)...")
faiss.normalize_L2(doc_embeddings)

print("Adding documents to Faiss index...")
index.add(doc_embeddings)
print(f"Faiss index built with {index.ntotal} documents using Inner Product (IP).")
# --- End Faiss Indexing ---


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