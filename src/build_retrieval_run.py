"""
build_retrieval_run.py

Creates retrieval "run files" (trec-rag style) from a single KILT dataset (WOW/FEVER/NQ).
- Loads queries from a KILT .jsonl file (typical fields: id, input, output).
- Runs retrieval with one or more selectable methods: bm25, contriever, dpr, bge, tct
- Saves one CSV per method with columns:
    query_id, doc_id, score, run_id
  where run_id is the method name.

Default experiment layout
-------------------------
By default, outputs are written to:

  <exp_root>/<dataset>/retrieval-runs/<dataset>_<run_id>.csv

where:
- exp_root default: repo_root/exp-output
- dataset is taken from --dataset, or inferred from the dataset filename
- run_id is the method name (bm25, contriever, dpr, bge, tct)

Note on duplicate queries:
- Retrieval functions in the codebase are indexed by "query text".
  If identical queries exist in the dataset, here we run retrieval only once
  and replicate the results for all query_id associated with that text.

CLI usage:

BM25:
python build_retrieval_run.py \
  --exp_root ../exp-output \
  --dataset nq \
  --dataset_path ../data/nq-train-kilt.jsonl \
  --methods bm25 \
  --bm25_index_dir ../indexes/bm25_index \
  --k 50 \
  --overwrite

Multiple methods:
python build_retrieval_run.py \
  --exp_root ../exp-output \
  --dataset nq \
  --dataset_path ../data/nq-train-kilt.jsonl \
  --methods bm25 contriever \
  --bm25_index_dir ../indexes/bm25_index \
  --faiss_index ./index_out_full/ivfpq_opq_contriever.faiss \
  --collection ../data/collection/wikipedia_passages.jsonl \
  --offsets ./index_out_full/collection_offsets.u64.bin \
  --nprobe 64 \
  --k 50 \
  --batch_size 256 \
  --overwrite
"""

from __future__ import annotations

import os
import json
import csv
import argparse
from typing import Dict, List, Tuple, Optional, Iterable, TypedDict
from collections import defaultdict

from dataset_builder import create_retriever
from bm25_retriever import bm25_batch_retrieve
from dense_sharded_retriever import dense_sharded_batch_retrieve


class RetrievedDoc(TypedDict):
    doc_id: str
    score: float
    contents: str


# -----------------------------
# Experiment paths (reuse when available)
# -----------------------------
def default_exp_root() -> str:
    """
    Default exp-output root, relative to repository layout:
      repo_root/exp-output
    where repo_root is the parent directory of this src file.
    """
    try:
        from build_all_erag_qrels import default_exp_root as _default_exp_root

        return _default_exp_root()
    except Exception:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        return os.path.join(repo_root, "exp-output")


def dataset_dir(exp_root: str, dataset: str) -> str:
    return os.path.join(exp_root, dataset)


def retrieval_runs_default_dir(exp_root: str, dataset: str) -> str:
    return os.path.join(dataset_dir(exp_root, dataset), "retrieval-runs")


def _sanitize_component(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "run"
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "run"


def infer_dataset_name_from_kilt_path(dataset_path: str) -> Optional[str]:
    """
    Best-effort heuristic to infer dataset name from a KILT dataset filename.
    Examples:
      nq-train-kilt.jsonl -> nq
      nq-dev-kilt.jsonl   -> nq
      fever-train-kilt    -> fever
      wow-dev-kilt        -> wow
    """
    base = os.path.splitext(os.path.basename(dataset_path or ""))[0]
    if not base:
        return None

    base_low = base.lower()
    for sep in ("-", "_"):
        if sep in base_low:
            head = base_low.split(sep, 1)[0].strip()
            if head:
                return head

    return base_low.strip() or None


def output_path_for_method(exp_root: str, dataset: str, run_id: str) -> str:
    out_dir = retrieval_runs_default_dir(exp_root, dataset)
    fname = f"{_sanitize_component(dataset)}_{_sanitize_component(run_id)}.csv"
    return os.path.join(out_dir, fname)


# -----------------------------
# KILT loading
# -----------------------------
def load_kilt_queries_from_file(
    path: str,
    max_examples: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """
    Load (query_id, query_text) from a KILT jsonl file.
    - query_id: record["id"] if present, otherwise line index
    - query_text: record["input"].strip()
    """
    out: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_examples is not None and i >= max_examples:
                break
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except Exception:
                continue

            query = (rec.get("input") or "").strip()
            if not query:
                continue

            qid = str(rec.get("id", i))
            out.append((qid, query))
    return out


# -----------------------------
# Retrieval dispatch
# -----------------------------
def retrieve_unique_queries(
    unique_queries: List[str],
    method: str,
    retriever,
    *,
    k: int,
    batch_size: int,
    bm25_threads: int,
    contriever_return_cosine: bool,
    dense_threads: int,
    dense_encode_batch_size: int,
    per_shard_k: Optional[int],
) -> Dict[str, List[RetrievedDoc]]:
    """
    Run retrieval for a list of unique queries (strings).
    Returns {query_text: [ {doc_id, score, contents}, ... ]}.
    """
    method = (method or "").lower()

    if method == "bm25":
        return bm25_batch_retrieve(
            unique_queries,
            searcher=retriever,
            k=k,
            batch_size=batch_size,
            threads=bm25_threads,
        )

    if method == "contriever":
        return retriever.contriever_batch_retrieve(
            queries=unique_queries,
            k=k,
            batch_size=batch_size,
            return_cosine=contriever_return_cosine,
        )

    if method in ("dpr", "bge", "tct"):
        return dense_sharded_batch_retrieve(
            queries=unique_queries,
            searcher=retriever,
            k=k,
            batch_size=batch_size,
            threads=dense_threads,
            encode_batch_size=dense_encode_batch_size,
            per_shard_k=per_shard_k,
        )

    raise ValueError(f"Unknown method: {method}")


# -----------------------------
# Output writing
# -----------------------------
def iter_run_rows(
    qid_query_pairs: List[Tuple[str, str]],
    retrieved_by_query: Dict[str, List[RetrievedDoc]],
    run_id: str,
) -> Iterable[Tuple[str, str, float, str]]:
    """
    Yields (query_id, doc_id, score, run_id), replicating results for duplicate queries.
    """
    query2qids: Dict[str, List[str]] = defaultdict(list)
    for qid, q in qid_query_pairs:
        query2qids[q].append(qid)

    for q, qids in query2qids.items():
        docs = retrieved_by_query.get(q, []) or []
        for qid in qids:
            for d in docs:
                yield (qid, str(d.get("doc_id", "")), float(d.get("score", 0.0)), run_id)


def write_run_csv(
    rows: Iterable[Tuple[str, str, float, str]],
    out_path: str,
    score_precision: int = 6,
) -> None:
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "doc_id", "score", "run_id"])
        fmt = f"{{:.{int(score_precision)}f}}"
        for qid, docid, score, runid in rows:
            w.writerow([qid, docid, fmt.format(score), runid])


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build one or more retrieval run CSV files from a single KILT dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--dataset_path", type=str, required=True, help="Path to a single KILT .jsonl dataset file.")
    parser.add_argument("--max_examples", type=int, default=None, help="Limit examples in the dataset (debug).")

    parser.add_argument(
        "--methods",
        type=str.lower,
        nargs="+",
        choices=["bm25", "contriever", "dpr", "bge", "tct"],
        required=True,
        help="One or more retrieval methods to run. One output CSV is produced per method.",
    )

    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument(
        "--exp_root",
        type=str,
        default=None,
        help="Experiment output root (default: repo_root/exp-output).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (e.g., nq). If not provided, it is inferred from the dataset filename.",
    )

    parser.add_argument("--overwrite", action="store_true", help="Overwrite output files if they exist.")
    parser.add_argument("--score_precision", type=int, default=6, help="Score decimals in output.")

    # BM25 args
    parser.add_argument("--bm25_index_dir", type=str, default=None, help="Directory indice BM25 (PySerini)")
    parser.add_argument("--bm25_threads", type=int, default=8, help="Threads for PySerini batch_search.")

    # Contriever args
    parser.add_argument("--faiss_index", type=str, default=None, help="Path indice FAISS (.faiss) per Contriever")
    parser.add_argument("--collection", type=str, default=None, help="Path JSONL collezione (id, contents) per Contriever")
    parser.add_argument("--offsets", type=str, default=None, help="Offsets binari uint64 (opzionale)")
    parser.add_argument("--nprobe", type=int, default=64, help="FAISS nprobe")
    parser.add_argument("--in_memory", action="store_true", help="Carica tutta la collezione in RAM (solo mini-run)")
    parser.add_argument(
        "--contriever_return_cosine",
        action="store_true",
        help="Usa approx_cosine come score (altrimenti usa -L2).",
    )

    # Dense sharded args (dpr/bge/tct)
    parser.add_argument("--dense_index_root_dir", type=str, default=None, help="Directory root shard part_0..part_N")
    parser.add_argument("--docstore_index_dir", type=str, default=None, help="Indice Lucene docstore (storeRaw)")
    parser.add_argument("--max_loaded_docid_shards", type=int, default=16, help="LRU cache size per shard docid")
    parser.add_argument("--dense_threads", type=int, default=8, help="FAISS omp threads (CPU)")
    parser.add_argument("--dense_encode_batch_size", type=int, default=32, help="Batch size per query encoding")
    parser.add_argument("--per_shard_k", type=int, default=None, help="Risultati per shard prima del merge")
    parser.add_argument("--dense_mmap", dest="dense_mmap", action="store_true", help="Use FAISS mmap (default)")
    parser.add_argument("--no_dense_mmap", dest="dense_mmap", action="store_false", help="Disable FAISS mmap")
    parser.set_defaults(dense_mmap=True)

    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")

    exp_root = os.path.abspath(args.exp_root or default_exp_root())
    dataset_name = args.dataset or infer_dataset_name_from_kilt_path(args.dataset_path)
    if not dataset_name:
        raise ValueError("dataset is required when it cannot be inferred from dataset_path.")

    methods = list(dict.fromkeys([m.lower().strip() for m in (args.methods or []) if m and m.strip()]))
    if not methods:
        raise ValueError("methods is empty.")

    # 1) Load queries once
    qid_query_pairs = load_kilt_queries_from_file(args.dataset_path, max_examples=args.max_examples)
    if not qid_query_pairs:
        raise RuntimeError("No queries loaded from dataset.")

    query2qids: Dict[str, List[str]] = defaultdict(list)
    for qid, q in qid_query_pairs:
        query2qids[q].append(qid)
    unique_queries = list(query2qids.keys())

    print(f"Dataset: {dataset_name}")
    print(f"exp_root: {exp_root}")
    print(f"Loaded {len(qid_query_pairs)} (qid,query) pairs; unique queries = {len(unique_queries)}")
    print(f"Methods: {methods}")

    # 2) Run retrieval per method and write one CSV per method
    out_dir = retrieval_runs_default_dir(exp_root, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    for method in methods:
        run_id = method
        out_path = output_path_for_method(exp_root, dataset_name, run_id=run_id)

        if os.path.exists(out_path) and not args.overwrite:
            raise FileExistsError(f"Output exists: {out_path} (use --overwrite)")

        print("\n----------------------------------------")
        print(f"Method: {method}")
        print(f"Output: {out_path}")

        args.method = method  # used by create_retriever()
        retriever = create_retriever(args)

        retrieved_by_query = retrieve_unique_queries(
            unique_queries,
            method=method,
            retriever=retriever,
            k=args.k,
            batch_size=args.batch_size,
            bm25_threads=args.bm25_threads,
            contriever_return_cosine=args.contriever_return_cosine,
            dense_threads=args.dense_threads,
            dense_encode_batch_size=args.dense_encode_batch_size,
            per_shard_k=args.per_shard_k,
        )

        rows = iter_run_rows(qid_query_pairs, retrieved_by_query, run_id=run_id)
        write_run_csv(rows, out_path, score_precision=args.score_precision)

        print(f"Saved run file -> {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()