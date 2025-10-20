"""
data_loader.py
Caricamento dataset KILT e creazione di dataset augmented con documenti retrieved.
Formato output: [{"query": str, "retrieved_docs": [doc1, ...], "gold_answer": str}, ...]
Supporta BM25 e Contriever per il retrieval.

Uso CLI:

BM25
python data_loader.py --datasets ../data/nq-train-kilt.jsonl ../data/nq-dev-kilt.jsonl
--method BM25
--k 50

Contriever
python data_loader.py --datasets ../data/nq-train-kilt.jsonl
--method Contriever
--k 50
--batch_size 256
"""

import json
from bm25_retriever import bm25_retrieve
import argparse
import os
from contriever_retriever import DenseRetriever


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
        return {query: bm25_retrieve(query, k=k) for query in queries}
    elif method == 'Contriever':
        return retriever.contriever_batch_retrieve(queries=queries, k=k, batch_size=batch_size)
    else:
        raise ValueError(f"Unknown method: {method}")


def augment_with_retrieved_documents(dataset, retrieved_results):
    """
    Restituisce una lista di dizionari nel formato:
        {"query": str,
         "retrieved_docs": [doc1, doc2, ..., dock],
         "gold_answer": str}
    """
    augmented_data = []
    for query, gold_answers in dataset.items():
        retrieved_docs = retrieved_results.get(query, [])
        target_text = gold_answers[0] # Since it's a multi-answer task, we will only take the first answer
        augmented_data.append({"query": query, "retrieved_docs": retrieved_docs, "gold_answer": target_text})
    return augmented_data


def augment_datasets(args):
    """
    Processa una lista di dataset (args.datasets).
    - Istanzia il retriever Contriever se richiesto da args.method.
    - Per ogni dataset:
        * Carica le query e i gold answer
        * Esegue il retrieval (BM25 o Contriever)
        * Salva il dataset arricchito in <same_dir>/<basename>-augmented.json
    """
    datasets = args.datasets or []

    if not datasets:
        print("No dataset provided in --datasets.")
        return

    # Istanzia il retriever se usiamo Contriever
    retriever = None
    if args.method == 'Contriever':
        retriever = DenseRetriever(
            index_path="./index_out_full/ivfpq_opq_contriever.faiss",
            collection_path="../data/collection/wikipedia_passages.jsonl",
            offsets_path="./index_out_full/collection_offsets.u64.bin",
            nprobe=64,
        )

    for dataset_path in datasets:
        if not dataset_path:
            continue

        if not os.path.exists(dataset_path):
            print(f"File not found: '{dataset_path}'. Skipped.")
            continue

        try:
            # Carica query e gold
            expected_outputs = load_expected_outputs(filename=dataset_path)
            queries = list(expected_outputs.keys())
            print(f"Loaded {len(queries)} queries")

            # Retrieval
            retrieved_results = retrieval_results(
                queries=queries,
                method=args.method,
                k=args.k,
                retriever=retriever,
                batch_size=args.batch_size,
            )

            # Augment e salvataggio
            augmented = augment_with_retrieved_documents(expected_outputs, retrieved_results)
            dirn = os.path.dirname(dataset_path) or "."
            base = os.path.splitext(os.path.basename(dataset_path))[0]
            out_path = os.path.join(dirn, f"{base}-augmented.json")

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(augmented, f, indent=4)

            print(f"Saved augmented dataset for '{dataset_path}' -> '{out_path}' ({len(augmented)} samples).")

        except Exception as e:
            print(f"Error during the processing of '{dataset_path}': {e}")


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Augment datasets with retrieved documents")
        parser.add_argument(
            "--datasets",
            type=str,
            nargs="+",
            required=True,
            help="List of paths to the datasets to process (separated by space)."
        )
        parser.add_argument(
            "--method",
            type=str,
            choices=["BM25", "Contriever"],
            default="BM25",
            help="Retrieval method (BM25 or Contriever)"
        )
        parser.add_argument(
            "--k",
            type=int,
            default=50,
            help="Number of documents to retrieve for each query (default: 50)"
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=256,
            help="Batch size for dense retrieval (default: 256)"
        )
        args = parser.parse_args()
        augment_datasets(args)
        print("Process completed.")
    except Exception as e:
        print(f"Error in creating the augmented dataset: {e}")
