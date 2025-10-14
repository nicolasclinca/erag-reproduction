import json
from src.bm25_retriever import bm25_retrieve
import argparse
import os
from typing import Dict, List, Tuple
from src.contriever_retriever import DenseRetriever


"""
Uso CLI:
    python data_loader.py --datasets ../data/nq-train-kilt.jsonl ../data/nq-dev-kilt.jsonl ../data/fever-train-kilt.jsonl ../data/fever-dev-kilt.jsonl ../data/hotpotqa-train-kilt.jsonl ../data/hotpotqa-dev-kilt.jsonl ../data/triviaqa-train-kilt.jsonl ../data/triviaqa-dev-kilt.jsonl ../data/wow-train-kilt.jsonl ../data/wow-dev-kilt.jsonl --method Contriever
"""

# Load all expected outputs from the KILT file
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


def retrieval_results(queries, method='BM25', k=50):
    if method == 'BM25':
        # For each query, retrieval_results[query] = [doc1, doc2, ..., doc50]
        retrieve_res = {query: bm25_retrieve(query, k=k) for query in queries}
    elif method == 'Contriever':
        # Percorsi hardcoded (da gestire meglio con variabili d'ambiente o argomenti)
        index_path = "./index_out_full/ivfpq_opq_contriever.faiss"
        collection_path = "../data/collection/wikipedia_passages.jsonl"
        offsets_path = "./index_out_full/collection_offsets.u64.bin"
        batch_size = 256
        nprobe = 64

        # if not index_path or not collection_path:
        #     raise ValueError()

        # Istanzia il retriever (usa offsets per accesso random veloce se disponibile)
        retr = DenseRetriever(
            index_path=index_path,
            collection_path=collection_path,
            offsets_path=offsets_path,
            nprobe=nprobe,
            in_memory=False,  # True only for mini-run; full-scale: False
        )

        # Batch inference
        retrieve_res: Dict[str, List[str]] = {}
        for i in range(0, len(queries), batch_size):
            batch_q = queries[i : i + batch_size]
            results = retr.batch_dense_retrieve(batch_q, k=k, return_cosine=False)
            for q, r in zip(batch_q, results):
                retrieve_res[q] = r["documents"]
    return retrieve_res


# Add retrieved documents to the dataset
def augment_with_retrieved_documents(nq_dataset, retrieval_results):
    augmented_data = []
    for query, gold_answers in nq_dataset.items():
        # Retrieve corresponding documents for the query
        retrieved_docs = retrieval_results.get(query, [])
        # Target answers (since it's a multi-answer task, we will just take the first answer)
        target_text = gold_answers[0]
        augmented_data.append({"query": query, "retrieved_docs": retrieved_docs, "gold_answer": target_text})
    return augmented_data


def augment_datasets(args):
    """
    Process a list of datasets (args.datasets). For each path:
      - call retrieval_results(filename=path, method=args.method)
      - call augment_with_retrieved_documents(...)
      - save the result in <same_dir>/<basename>-augmented.json
    """

    method = args.method
    datasets = args.datasets or []

    if not datasets:
        print("No dataset provided in --datasets.")
        return

    for dataset_path in datasets:
        if not dataset_path:
            continue

        if not os.path.exists(dataset_path):
            print(f"File not found: '{dataset_path}'. Skipped.")
            continue

        try:
            # Recupera i risultati di retrieval e costruisce l'augmented dataset
            expected_outputs = load_expected_outputs(filename=dataset_path)
            queries = list(expected_outputs.keys())
            print(f"Loaded {len(queries)} queries")
            retrieve_res = retrieval_results(queries=queries, method=method)
            augmented = augment_with_retrieved_documents(expected_outputs, retrieve_res)

            # Costruisce il percorso di output
            dirn = os.path.dirname(dataset_path) or "."
            base = os.path.splitext(os.path.basename(dataset_path))[0]
            out_path = os.path.join(dirn, f"{base}-augmented.json")

            # Salva in JSON (utf-8)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(augmented, f, indent=4)

            print(f"Saved augmented dataset for '{dataset_path}' -> '{out_path}' ({len(augmented)} samples).")

        except Exception as e:
            # Non interrompiamo l'elaborazione degli altri dataset: segnaliamo l'errore e continuiamo
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
            required=True,
            default="BM25",
            help="Retrieval method (BM25 or Contriever)"
        )
        args = parser.parse_args()
        augment_datasets(args)
        print("Process completed.")
    except Exception as e:
        print(f"Error in creating the augmented dataset: {e}")

    
    