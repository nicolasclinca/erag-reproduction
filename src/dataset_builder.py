"""
dataset_builder.py
Caricamento dataset KILT e creazione di dataset augmented con documenti retrieved.
Formato output: [{"query": str, "retrieved_docs": [doc1, ...], "gold_answer": str}, ...]

Supporta:
- bm25 (PySerini LuceneSearcher su indice BM25)
- contriever (FAISS + JSONL collection)
- dpr/bge/tct (FAISS sharded PySerini-style + docstore Lucene per docid->contents) via dense_sharded_retriever.py

Note dense-sharded (dpr/bge/tct):
- Gli indici sono in 120 shard (part_0..part_N) e vengono caricati una volta.
- L'encoder query è selezionato in base al metodo (dpr/bge/tct).
- Retrieval: search su ogni shard (top per_shard_k) + merge top-k globale.

Uso CLI:

bm25
python dataset_builder.py --datasets ../data/nq-train-kilt.jsonl \
    --method bm25 \
    --bm25_index_dir ../indexes/bm25_index \
    --k 50

contriever
python dataset_builder.py --datasets ../data/nq-train-kilt.jsonl \
    --method contriever \
    --faiss_index ./index_out_full/ivfpq_opq_contriever.faiss \
    --collection ../data/collection/wikipedia_passages.jsonl \
    --offsets ./index_out_full/collection_offsets.u64.bin \
    --nprobe 64 \
    --k 50 \
    --batch_size 256

bge (dense-sharded)
python dataset_builder.py --datasets ../data/nq-train-kilt.jsonl \
    --method bge \
    --dense_index_root_dir ../indexes/wiki-bge-118m \
    --docstore_index_dir ../indexes/wiki_docstore_lucene \
    --k 50 \
    --batch_size 256 \
    --dense_threads 8 \
    --dense_encode_batch_size 32
"""

import json
import argparse
import os

from contriever_retriever import DenseRetriever
from bm25_retriever import bm25_batch_retrieve, create_bm25_searcher
from dense_sharded_retriever import ShardedFaissSearcher, dense_sharded_batch_retrieve
from query_encoders import build_query_encoder


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
        augmented_data.append({"query": query, "retrieved_docs": retrieved_docs, "gold_answer": target_text})
    return augmented_data


def retrieval_results(
    queries,
    method="bm25",
    k=50,
    retriever=None,
    batch_size=256,
    dense_threads=8,
    dense_encode_batch_size=32,
    per_shard_k=None,
):
    """
    Returns {query: [doc_contents1, doc_contents2, ...]}.

    Nota: bm25_batch_retrieve e contriever_batch_retrieve ora ritornano anche doc_id e score,
    quindi qui estraiamo solo i contents per non rompere l'augmenting attuale.
    """
    method = (method or "").lower()

    if method == "bm25":
        full = bm25_batch_retrieve(
            queries, searcher=retriever, k=k, batch_size=batch_size
        )
        return {q: [d.get("contents", "") for d in docs] for q, docs in full.items()}

    if method == "contriever":
        full = retriever.contriever_batch_retrieve(
            queries=queries, k=k, batch_size=batch_size
        )
        return {q: [d.get("contents", "") for d in docs] for q, docs in full.items()}

    if method in ("dpr", "bge", "tct"):
        full = dense_sharded_batch_retrieve(
            queries=queries,
            searcher=retriever,
            k=k,
            batch_size=batch_size,
            threads=dense_threads,
            encode_batch_size=dense_encode_batch_size,
            per_shard_k=per_shard_k,
        )
        return {q: [d.get("contents", "") for d in docs] for q, docs in full.items()}

    raise ValueError(f"Unknown method: {method}")


def create_retriever(args):
    """
    Crea il retriever in base ad args.method (lowercase):
    - bm25 -> LuceneSearcher
    - contriever -> DenseRetriever (FAISS + JSONL)
    - dpr/bge/tct -> ShardedFaissSearcher + query encoder fixed
    """
    if args.method == "bm25":
        if not args.bm25_index_dir:
            raise ValueError("--bm25_index_dir è obbligatorio con --method bm25")
        return create_bm25_searcher(args.bm25_index_dir)

    if args.method == "contriever":
        missing = []
        if not args.faiss_index: missing.append("--faiss_index")
        if not args.collection: missing.append("--collection")
        if missing:
            raise ValueError(f"Con --method contriever servono: --faiss_index e --collection (mancanti: {', '.join(missing)})")
        return DenseRetriever(index_path=args.faiss_index, collection_path=args.collection, offsets_path=args.offsets, nprobe=args.nprobe, in_memory=args.in_memory)

    if args.method in ("dpr", "bge", "tct"):
        missing = []
        if not args.dense_index_root_dir: missing.append("--dense_index_root_dir")
        if not args.docstore_index_dir: missing.append("--docstore_index_dir")
        if missing:
            raise ValueError(f"Con --method {args.method} servono: dense_index_root_dir e docstore_index_dir (mancanti: {', '.join(missing)})")

        query_encoder = build_query_encoder(encoder_type=args.method)
        return ShardedFaissSearcher(
            index_root_dir=args.dense_index_root_dir, query_encoder=query_encoder, docstore_index_dir=args.docstore_index_dir,
            max_loaded_docid_shards=args.max_loaded_docid_shards, faiss_threads=args.dense_threads, mmap=args.dense_mmap,
            assert_inner_product=True
        )

    raise ValueError(f"Unknown method: {args.method}")


def augment_datasets(args):
    """
    Processa una lista di dataset (args.datasets).
    - Istanzia il retriever richiesto da args.method.
    - Per ogni dataset:
        * Carica le query e i gold answer
        * Esegue il retrieval
        * Salva il dataset arricchito in <out_dir>/<basename>-augmented-<method>.json
    """
    retriever = create_retriever(args)

    for dataset_path in args.datasets:
        if not os.path.exists(dataset_path):
            print(f"File not found: '{dataset_path}'. Skipped.")
            continue

        expected_outputs = load_expected_outputs(filename=dataset_path)
        queries = list(expected_outputs.keys())
        print(f"Loaded {len(queries)} queries")

        retrieved_results = retrieval_results(queries=queries, method=args.method, k=args.k, retriever=retriever, 
                                              batch_size=args.batch_size, dense_threads=args.dense_threads, 
                                              dense_encode_batch_size=args.dense_encode_batch_size, 
                                              per_shard_k=args.per_shard_k)

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
    parser.add_argument("--method", type=str.lower, choices=["bm25", "contriever", "dpr", "bge", "tct"], 
                        default="bm25", help="Retrieval method (bm25, contriever, dpr, bge, tct)")
    parser.add_argument("--k", type=int, default=50,
                        help="Number of documents to retrieve for each query (default: 50)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for retrieval (default: 256)")

    # BM25 args
    parser.add_argument("--bm25_index_dir", type=str, help="Directory indice BM25 (PySerini)")

    # Contriever args
    parser.add_argument("--faiss_index", type=str, help="Path indice FAISS (.faiss) per Contriever")
    parser.add_argument("--collection", type=str, help="Path JSONL collezione (id, contents) per Contriever")
    parser.add_argument("--offsets", type=str, default=None, help="Offsets binari uint64 (opzionale)")
    parser.add_argument("--nprobe", type=int, default=64, help="FAISS nprobe")
    parser.add_argument("--in_memory", action="store_true", help="Carica tutta la collezione in RAM (solo mini-run)")

    # Dense sharded args (dpr/bge/tct)
    parser.add_argument("--dense_index_root_dir", type=str, default=None,
                        help="Directory root con shard part_0..part_N (FAISS PySerini-style).")
    parser.add_argument("--docstore_index_dir", type=str, default=None,
                        help="Indice Lucene con storeRaw per docid->contents (può essere anche l'indice BM25 se storeRaw).")
    parser.add_argument("--max_loaded_docid_shards", type=int, default=16,
                        help="LRU cache size per shard docid (dense sharded)")
    parser.add_argument("--dense_threads", type=int, default=8, help="FAISS omp threads (CPU)")
    parser.add_argument("--dense_encode_batch_size", type=int, default=32,
                        help="Batch size per query encoding (dense sharded)")
    parser.add_argument("--per_shard_k", type=int, default=None,
                        help="Quanti risultati per shard prima del merge (default: k*4 capped).")
    parser.add_argument("--dense_mmap", dest="dense_mmap", action="store_true",
                        help="Usa FAISS mmap per dense-sharded (default)")
    parser.add_argument("--no_dense_mmap", dest="dense_mmap", action="store_false",
                        help="Disabilita FAISS mmap per dense-sharded")
    parser.set_defaults(dense_mmap=True)

    parser.add_argument("--max_answer_words", type=int, default=250,
                        help="Numero massimo di parole per la risposta (default: 250).")

    args = parser.parse_args()
    augment_datasets(args)
    print("Process completed.")