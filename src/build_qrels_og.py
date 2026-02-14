"""
build_qrels_og.py

Creates a qrels_og file (query_id, doc_id, relevance) in CSV format
starting from:
1) one or more KILT datasets (.jsonl) containing gold evidence in output[*].provenance[*].wikipedia_id
2) the segmented collection (JSONL) produced by preprocess_wikipedia.py with typical fields:
   {"id": "<wikipedia_id>_<segment_id>", "contents": ...}

Relevance rule (segment-level):
- a segment (full doc_id, e.g. "40885965_20") is relevant (relevance=1)
  if the wikipedia_id part of the doc_id is among the gold wikipedia_id for that query_id.
- In practice, here we write ONLY the relevant segments (positive-only qrels):
  all retrieved docs not present in qrels will be considered non-relevant by evaluation tools.

Output:
- CSV with header: query_id,doc_id,relevance

Usage:
python build_qrels_og.py \
  --datasets ../data/nq-train-kilt.jsonl \
  --collection ../data/collection/wikipedia_passages.jsonl \
  --output ../qrels/nq_qrels_og.csv \
  --overwrite
"""

from __future__ import annotations

import os
import json
import csv
import argparse
from typing import Dict, Iterator, List, Optional, Set, Tuple


# -----------------------------
# Collection parsing
# -----------------------------
def iter_collection_doc_ids(collection_path: str, max_docs: Optional[int] = None) -> Iterator[str]:
    """
    Iterate doc_id values (the "id" field) from the JSONL collection.
    """
    with open(collection_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_docs is not None and i >= max_docs:
                break
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except Exception:
                continue

            did = (rec.get("id") or rec.get("doc_id") or "").strip()
            if did:
                yield str(did)


# -----------------------------
# KILT gold loading (qid -> set(wikipedia_id))
# -----------------------------
def load_kilt_gold_wikipedia_ids_from_file(
    path: str,
    max_examples: Optional[int] = None,
    qid_prefix: Optional[str] = None,
) -> Dict[str, Set[str]]:
    """
    Returns: {query_id: set_of_gold_wikipedia_ids}

    query_id:
      - record["id"] if present, otherwise the row index
      - if qid_prefix is provided: f"{qid_prefix}:{qid}"
    """
    gold: Dict[str, Set[str]] = {}

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

            qid = str(rec.get("id", i))
            if qid_prefix:
                qid = f"{qid_prefix}:{qid}"

            wiki_ids: Set[str] = set()
            outputs = rec.get("output") or []
            if isinstance(outputs, list):
                for out in outputs:
                    provs = (out or {}).get("provenance") or []
                    if not isinstance(provs, list):
                        continue
                    for p in provs:
                        wid = (p or {}).get("wikipedia_id", None)
                        if wid is None:
                            continue
                        wiki_ids.add(str(wid))

            gold[qid] = wiki_ids

    return gold


def load_kilt_gold_wikipedia_ids(
    datasets: List[str],
    max_examples: Optional[int] = None,
    prefix_with_dataset: bool = True,
) -> Dict[str, Set[str]]:
    """
    Loads qid->gold_wikipedia_ids from a list of datasets.
    If multiple datasets and prefix_with_dataset=True, prefixes qid with the basename to avoid collisions.
    """
    if not datasets:
        return {}

    multi = len(datasets) > 1
    merged: Dict[str, Set[str]] = {}

    for p in datasets:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Dataset not found: {p}")

        prefix = None
        if prefix_with_dataset and multi:
            prefix = os.path.splitext(os.path.basename(p))[0]

        part = load_kilt_gold_wikipedia_ids_from_file(p, max_examples=max_examples, qid_prefix=prefix)

        # merge (if collision, union the sets)
        for qid, wids in part.items():
            if qid not in merged:
                merged[qid] = set()
            merged[qid].update(wids)

    return merged


def wikipedia_id_from_doc_id(doc_id: str) -> str:
    """
    Extracts the wikipedia_id part from doc_id.
    Example: "40885965_20" -> "40885965"
    If there is no "_", returns the full doc_id.
    """
    doc_id = (doc_id or "").strip()
    if not doc_id:
        return ""
    return doc_id.split("_", 1)[0]


# -----------------------------
# Inversion: wikipedia_id -> [query_id, ...]
# -----------------------------
def build_wikipedia_id_to_qids(gold_wiki_ids_by_qid: Dict[str, Set[str]]) -> Dict[str, List[str]]:
    """
    Build an inverted map:
        {wikipedia_id: [query_id1, query_id2, ...]}
    """
    wid2qids: Dict[str, List[str]] = {}
    for qid, wids in gold_wiki_ids_by_qid.items():
        if not wids:
            continue
        for wid in wids:
            wid = str(wid).strip()
            if not wid:
                continue
            wid2qids.setdefault(wid, []).append(qid)
    return wid2qids


# -----------------------------
# Qrels OG rows (streaming)
# -----------------------------
def iter_qrels_og_rows(
    collection_doc_ids: Iterator[str],
    wid2qids: Dict[str, List[str]],
) -> Iterator[Tuple[str, str, int]]:
    """
    Yields (query_id, doc_id, relevance=1) for all collection segments
    that belong to a gold wikipedia_id for one or more queries.
    """
    for docid in collection_doc_ids:
        wid = wikipedia_id_from_doc_id(docid)
        qids = wid2qids.get(wid, None)
        if not qids:
            continue
        for qid in qids:
            yield qid, docid, 1


def write_qrels_og_csv(rows: Iterator[Tuple[str, str, int]], out_path: str) -> int:
    """
    Write a streaming CSV and return the number of rows written (excluding header).
    """
    n = 0
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "doc_id", "relevance"])
        for qid, docid, rel in rows:
            w.writerow([qid, docid, int(rel)])
            n += 1
    return n


# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build qrels_og CSV (only-positive segment-level qrels) from KILT dataset(s) and a segmented collection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--datasets", type=str, nargs="+", required=True, help="Path(s) to KILT .jsonl dataset(s).")
    parser.add_argument("--collection", type=str, required=True, help="Path to segmented collection JSONL (wikipedia_passages.jsonl).")

    parser.add_argument("--max_examples", type=int, default=None, help="Limit examples per dataset (debug).")
    parser.add_argument(
        "--prefix_with_dataset",
        action="store_true",
        help="If multiple datasets, prefix query_id with dataset basename to avoid collisions.",
    )

    parser.add_argument("--max_collection_docs", type=int, default=None, help="Limit docs read from collection (debug).")

    parser.add_argument("--output", type=str, required=True, help="Output qrels_og CSV path.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if exists.")

    args = parser.parse_args()

    if not os.path.exists(args.collection):
        raise FileNotFoundError(f"Collection not found: {args.collection}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if os.path.exists(args.output) and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output} (use --overwrite)")

    # 1) Load gold wikipedia ids by query_id
    gold_map = load_kilt_gold_wikipedia_ids(
        datasets=args.datasets,
        max_examples=args.max_examples,
        prefix_with_dataset=args.prefix_with_dataset,
    )
    if not gold_map:
        raise RuntimeError("No gold data loaded from dataset(s).")

    wid2qids = build_wikipedia_id_to_qids(gold_map)
    needed_wids = set(wid2qids.keys())

    print(f"Loaded queries: {len(gold_map)}")
    print(f"Unique gold wikipedia_id: {len(needed_wids)}")

    # 2) Stream collection -> stream qrels rows -> write
    coll_docids = iter_collection_doc_ids(args.collection, max_docs=args.max_collection_docs)

    # track missing wikipedia_id (best-effort, by observing seen ids in collection)
    remaining_wids = set(needed_wids)

    def _rows_with_tracking() -> Iterator[Tuple[str, str, int]]:
        nonlocal remaining_wids
        for docid in coll_docids:
            wid = wikipedia_id_from_doc_id(docid)
            if wid in remaining_wids:
                remaining_wids.discard(wid)

            qids = wid2qids.get(wid, None)
            if not qids:
                continue
            for qid in qids:
                yield qid, docid, 1

    n_rows = write_qrels_og_csv(_rows_with_tracking(), args.output)

    print(f"Saved qrels_og -> {args.output} (rows={n_rows})")
    if remaining_wids:
        print(f"Warning: {len(remaining_wids)} gold wikipedia_id not found in collection (example: {next(iter(remaining_wids))})")


if __name__ == "__main__":
    main()