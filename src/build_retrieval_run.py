"""
build_retrieval_run.py

Creates a retrieval "run file" (trec-rag style) from a KILT dataset (WOW/FEVER/NQ).
- Loads queries from one or more KILT .jsonl files (typical fields: id, input, output).
- Runs retrieval with a selectable method: bm25, contriever, dpr, bge, tct
- Saves results in a CLI-selectable format: csv, json, txt
  with columns/fields: query_id, doc_id, score, run_id
  (run_id default = name of the method used).

Note on duplicate queries:
- Retrieval functions in the codebase are indexed by "query text".
  If identical queries exist in the dataset, here we run retrieval only once
  and replicate the results for all query_id associated with that text.

CLI usage:

BM25:
python build_retrieval_run.py \
  --datasets ../data/nq-train-kilt.jsonl \
  --method bm25 \
  --bm25_index_dir ../indexes/bm25_index \
  --k 50 \
  --output ../runs/nq_bm25.txt \
  --format txt

Contriever:
python build_retrieval_run.py \
  --datasets ../data/nq-train-kilt.jsonl \
  --method contriever \
  --faiss_index ./index_out_full/ivfpq_opq_contriever.faiss \
  --collection ../data/collection/wikipedia_passages.jsonl \
  --offsets ./index_out_full/collection_offsets.u64.bin \
  --nprobe 64 \
  --k 50 \
  --batch_size 256 \
  --output ../runs/nq_contriever.csv \
  --format csv

Dense-sharded (bge/dpr/tct):
python build_retrieval_run.py \
  --datasets ../data/nq-train-kilt.jsonl \
  --method bge \
  --dense_index_root_dir ../indexes/wiki-bge-118m \
  --docstore_index_dir ../indexes/wiki_docstore_lucene \
  --k 50 \
  --dense_threads 8 \
  --dense_encode_batch_size 32 \
  --output ../runs/nq_bge.json \
  --format json
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
# KILT loading
# -----------------------------
def load_kilt_queries_from_file(
    path: str,
    max_examples: Optional[int] = None,
    qid_prefix: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """
    Load (query_id, query_text) from a KILT jsonl file.
    - query_id: record["id"] if present, otherwise line index
    - query_text: record["input"].strip()

    qid_prefix (optional): prefix to prepend to query_id (e.g., dataset basename).
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
            if qid_prefix:
                qid = f"{qid_prefix}:{qid}"

            out.append((qid, query))
    return out


def load_kilt_queries(
    datasets: List[str],
    max_examples: Optional[int] = None,
    prefix_with_dataset: bool = True,
) -> List[Tuple[str, str]]:
    """
    Load (qid, query) from a list of files.
    If prefix_with_dataset=True, prefix qid with <basename> to avoid collisions across different datasets.
    """
    all_pairs: List[Tuple[str, str]] = []
    multi = len(datasets) > 1
    for p in datasets:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Dataset not found: {p}")
        prefix = None
        if prefix_with_dataset and multi:
            prefix = os.path.splitext(os.path.basename(p))[0]
        all_pairs.extend(load_kilt_queries_from_file(p, max_examples=max_examples, qid_prefix=prefix))
    return all_pairs


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


def write_run_txt(
    rows: Iterable[Tuple[str, str, float, str]],
    out_path: str,
    score_precision: int = 6,
    sep: str = "\t",
) -> None:
    fmt = f"{{:.{int(score_precision)}f}}"
    with open(out_path, "w", encoding="utf-8") as f:
        for qid, docid, score, runid in rows:
            f.write(f"{qid}{sep}{docid}{sep}{fmt.format(score)}{sep}{runid}\n")


def write_run_json(
    rows: Iterable[Tuple[str, str, float, str]],
    out_path: str,
    score_precision: int = 6,
) -> None:
    """
    Write a streaming JSON array (without keeping everything in RAM).
    """
    fmt = f"{{:.{int(score_precision)}f}}"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("[\n")
        first = True
        for qid, docid, score, runid in rows:
            obj = {
                "query_id": qid,
                "doc_id": docid,
                "score": float(fmt.format(score)),
                "run_id": runid,
            }
            if not first:
                f.write(",\n")
            f.write(json.dumps(obj, ensure_ascii=False))
            first = False
        f.write("\n]\n")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build retrieval run file from KILT dataset(s).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--datasets", type=str, nargs="+", required=True, help="Path(s) to KILT .jsonl dataset(s).")
    parser.add_argument("--max_examples", type=int, default=None, help="Limit examples per dataset (debug).")
    parser.add_argument(
        "--prefix_with_dataset",
        action="store_true",
        help="If multiple datasets, prefix query_id with dataset basename to avoid collisions.",
    )

    parser.add_argument("--method", type=str.lower, choices=["bm25", "contriever", "dpr", "bge", "tct"], required=True)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--output", type=str, required=True, help="Output path.")
    parser.add_argument("--format", choices=["txt", "csv", "json"], default="txt")
    parser.add_argument("--run_id", type=str, default=None, help="Run id (default: method).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if exists.")
    parser.add_argument("--score_precision", type=int, default=6, help="Score decimals in output.")
    parser.add_argument("--txt_sep", type=str, default="\t", help="Separator for txt format (default: tab).")

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

    out_path = args.output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if os.path.exists(out_path) and not args.overwrite:
        raise FileExistsError(f"Output exists: {out_path} (use --overwrite)")

    # 1) Load queries
    qid_query_pairs = load_kilt_queries(
        datasets=args.datasets,
        max_examples=args.max_examples,
        prefix_with_dataset=args.prefix_with_dataset,
    )
    if not qid_query_pairs:
        raise RuntimeError("No queries loaded from dataset(s).")

    query2qids: Dict[str, List[str]] = defaultdict(list)
    for qid, q in qid_query_pairs:
        query2qids[q].append(qid)
    unique_queries = list(query2qids.keys())

    print(f"Loaded {len(qid_query_pairs)} (qid,query) pairs; unique queries = {len(unique_queries)}")

    # 2) Create retriever
    retriever = create_retriever(args)

    # 3) Retrieval on unique queries
    retrieved_by_query = retrieve_unique_queries(
        unique_queries,
        method=args.method,
        retriever=retriever,
        k=args.k,
        batch_size=args.batch_size,
        bm25_threads=args.bm25_threads,
        contriever_return_cosine=args.contriever_return_cosine,
        dense_threads=args.dense_threads,
        dense_encode_batch_size=args.dense_encode_batch_size,
        per_shard_k=args.per_shard_k,
    )

    # 4) Write output
    run_id = args.run_id or args.method
    rows = iter_run_rows(qid_query_pairs, retrieved_by_query, run_id=run_id)

    if args.format == "csv":
        write_run_csv(rows, out_path, score_precision=args.score_precision)
    elif args.format == "json":
        write_run_json(rows, out_path, score_precision=args.score_precision)
    elif args.format == "txt":
        write_run_txt(rows, out_path, score_precision=args.score_precision, sep=args.txt_sep)
    else:
        raise ValueError(f"Unknown format: {args.format}")

    print(f"Saved run file -> {out_path}")


if __name__ == "__main__":
    main()