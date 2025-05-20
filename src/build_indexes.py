import json
import torch
import faiss
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# Pyserini BM25 indexing
#python -m pyserini.index.lucene -collection JsonCollection -input ../data/collection -index ../indexes/bm25_index -generator DefaultLuceneDocumentGenerator -threads 8 -storePositions -storeDocvectors -storeRaw


def data_loading(input_file):
    # --- Loading --- #
    documents = []
    try:
        with open(input_file, "r", encoding="utf-8") as fin:
            for i, line in enumerate(fin):
                try:
                    data = json.loads(line)
                    documents.append(data["contents"])
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON on line {i+1}")
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        exit()
    if not documents:
        print("No documents loaded. Exiting.")
        exit()
    return documents


def setup_dense_retriever(model_name="facebook/contriever"):
    """Initiliaze dense setup"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) 
    print(f"Model moved to device: {device}")
    return tokenizer, model, device



def encode_documents(docs, batch_size=16, max_length=256):
    """
    Encodes a list of documents into normalized embeddings using Contriever.
    Maintains the original function signature.
    :param docs: list of documents to be embedded
    :param batch_size: size of the sample batch
    :param max_length: max length of the tensor
    :return: embedding vector for dense retrival
    """
    tokenizer, model, device = setup_dense_retriever()
    all_embeddings = []
    model.eval()
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


def faiss_index():
    documents = data_loading(input_file="../data/collection/wikipedia_passages.jsonl")
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
    
    index_file = '../indexes/faiss_index'
    faiss.write_index(index, index_file)
    print(f"FAISS index salvato in {index_file}")
    # --- End Faiss Indexing ---
    

if __name__=="__main__":
    faiss_index()
