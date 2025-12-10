"""
dataset_builder.py
Caricamento dataset KILT e creazione di dataset augmented con documenti retrieved.
Formato output: [{"query": str, "retrieved_docs": [doc1, ...], "gold_answer": str}, ...]
Supporta BM25 e Contriever per il retrieval.

Uso CLI:

BM25
python dataset_builder.py --datasets ../data/nq-train-kilt.jsonl \
    --method BM25 \
    --bm25_index_dir ../indexes/bm25_index \
    --k 50
    
Contriever
python dataset_builder.py --datasets ../data/nq-train-kilt.jsonl \
    --method Contriever \
    --faiss_index ./index_out_full/ivfpq_opq_contriever.faiss \
    --collection ../data/collection/wikipedia_passages.jsonl \
    --offsets ./index_out_full/collection_offsets.u64.bin \
    --nprobe 64 \
    --k 50 \
    --batch_size 256
"""

import json
import argparse
import os

from contriever_retriever import DenseRetriever
from bm25_retriever import bm25_batch_retrieve, create_bm25_searcher


def load_expected_outputs(filename):
    """Loads all queries and their gold answers from the KILT file.
    Returns a dict {query: [gold_answer1, gold_answer2, ...]}."""
    expected = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            query = record["input"].strip()
            golds = [out["answer"].strip() for out in record.get("output", []) if "answer" in out]
            if golds:
                expected[query] = golds
    return expected


def retrieval_results(queries, method='BM25', k=50, retriever=None, batch_size=256):
    """
    Restituisce {query: [doc1, doc2, ..., dock]}.
    """
    if method == 'BM25':
        return bm25_batch_retrieve(queries, searcher=retriever, k=k, batch_size=batch_size)
    elif method == 'Contriever':
        return retriever.contriever_batch_retrieve(queries=queries, k=k, batch_size=batch_size)
    else:
        raise ValueError(f"Unknown method: {method}")
    

def select_answer(gold_answers, max_words=250):
    """
    Seleziona una singola risposta da una lista di risposte gold.
    - Prima risposta con <= max_words parole.
    - Se nessuna soddisfa il vincolo, prende la risposta più breve.
    """
    if not gold_answers:
        return ""
    valid_answers = [a for a in gold_answers if isinstance(a, str) and a.strip()]
    if not valid_answers:
        return gold_answers[0]

    lengths = [len(a.split()) for a in valid_answers]

    for ans, l in zip(valid_answers, lengths):
        if l <= max_words:
            return ans
    min_idx = min(range(len(valid_answers)), key=lambda i: lengths[i])
    return valid_answers[min_idx]


def augment_with_documents(dataset, retrieved_results, max_words):
    """
    Restituisce una lista di dizionari nel formato:
        {"query": str,
         "retrieved_docs": [doc1, doc2, ..., dock],
         "gold_answer": str}
    """
    augmented_data = []
    for query, gold_answers in dataset.items():
        retrieved_docs = retrieved_results.get(query, [])
        target_text = select_answer(gold_answers, max_words)
        augmented_data.append({
            "query": query,
            "retrieved_docs": retrieved_docs,
            "gold_answer": target_text
        })
    return augmented_data


def create_retriever(method='BM25', bm25_index_dir=None, faiss_index=None,
                     collection=None, offsets=None, nprobe=64, in_memory=False):
    if method == 'BM25':
        if not bm25_index_dir:
            raise ValueError("--bm25_index_dir è obbligatorio con --method BM25")
        retriever = create_bm25_searcher(bm25_index_dir)
    elif method == 'Contriever':
        missing = []
        if not faiss_index:
            missing.append("--faiss_index")
        if not collection:
            missing.append("--collection")
        if missing:
            raise ValueError(f"Con --method Contriever servono: --faiss_index e --collection (mancanti: {', '.join(missing)})")
        retriever = DenseRetriever(
                index_path=faiss_index,
                collection_path=collection,
                offsets_path=offsets,
                nprobe=nprobe,
                in_memory=in_memory
            )
    else:
        raise ValueError(f"Unknown method: {method}")
    return retriever


def augment_datasets(args):
    """
    Processa una lista di dataset (args.datasets).
    - Istanzia il retriever Contriever se richiesto da args.method.
    - Per ogni dataset:
        * Carica le query e i gold answer
        * Esegue il retrieval (BM25 o Contriever)
        * Salva il dataset arricchito in <same_dir>/<basename>-augmented.json
    """

    # Istanzia il retriever
    retriever = create_retriever(method=args.method, bm25_index_dir=args.bm25_index_dir, 
                                 faiss_index=args.faiss_index, collection=args.collection, 
                                 offsets=args.offsets, nprobe=args.nprobe, in_memory=args.in_memory)

    for dataset_path in args.datasets:
        if not os.path.exists(dataset_path):
            print(f"File not found: '{dataset_path}'. Skipped.")
            continue

        # Carica query e gold
        expected_outputs = load_expected_outputs(filename=dataset_path)
        queries = list(expected_outputs.keys())
        print(f"Loaded {len(queries)} queries")

        # Retrieval
        retrieved_results = retrieval_results(queries=queries, method=args.method, k=args.k,
                                              retriever=retriever, batch_size=args.batch_size)

        # Augment e salvataggio
        augmented = augment_with_documents(expected_outputs, retrieved_results, args.max_answer_words)
        dirn = args.augmented_datasets if args.augmented_datasets else (os.path.dirname(dataset_path) or ".")
        os.makedirs(dirn, exist_ok=True)
        base = os.path.splitext(os.path.basename(dataset_path))[0]
        out_path = os.path.join(dirn, f"{base}-augmented-{args.method}.json")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(augmented, f, indent=4)

        print(f"Saved augmented dataset for '{dataset_path}' -> '{out_path}' ({len(augmented)} samples).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment datasets with retrieved documents",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--datasets", type=str, nargs="+", required=True,
        help="List of paths to the datasets to process (separated by space).")
    parser.add_argument("--augmented_datasets", type=str, default=None,
        help="Directory dove salvare i dataset augmentati. Default: stessa cartella del dataset.")
    parser.add_argument("--method", type=str, choices=["BM25", "Contriever"], default="BM25",
        help="Retrieval method (BM25 or Contriever)")
    parser.add_argument("--k", type=int, default=50,
        help="Number of documents to retrieve for each query (default: 50)")
    parser.add_argument("--batch_size", type=int, default=256,
        help="Batch size for dense retrieval (default: 256)")
    parser.add_argument("--bm25_index_dir", type=str, help="Directory indice BM25 (PySerini)")
    parser.add_argument("--faiss_index", type=str, help="Path indice FAISS (.faiss) per Contriever")
    parser.add_argument("--collection", type=str, 
        help="Path JSONL collezione (id, contents) per Contriever")
    parser.add_argument("--offsets", type=str, default=None, help="Offsets binari uint64 (opzionale)")
    parser.add_argument("--nprobe", type=int, default=64, help="FAISS nprobe")
    parser.add_argument("--in_memory", action="store_true", 
        help="Carica tutta la collezione in RAM (solo mini-run)")
    parser.add_argument("--max_answer_words", type=int, default=250,
        help="Numero massimo di parole per la risposta (default: 250).")    
    args = parser.parse_args()
    augment_datasets(args)
    print("Process completed.")
