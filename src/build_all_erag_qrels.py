"""
build_all_erag_qrels.py

Dato un file <dataset>_all_runs.csv (query_id, doc_id), calcola le qrels eRAG “model-based”
per TUTTE le coppie (query_id, doc_id) presenti, usando un modello FiD-T5.

Output
------
Scrive <dataset>_all_qrels.csv con header:
  query_id, doc_id, relevance

e include SOLO le righe con relevance > 0.

Nota
----
Qui NON calcoliamo metriche IR (P@k, nDCG, ecc.): ci basta generare la risposta usando
solo quel documento come contesto e calcolare la downstream_metric (EM/Accuracy/F1),
che corrisponde alla relevance da loggare.

Input necessari
---------------
- --all_runs: CSV con colonne query_id, doc_id (eventuali colonne extra vengono ignorate)
- --datasets: uno o più file KILT .jsonl (per query_text e gold answers)
- --collection: JSONL collezione documenti (id, contents)
- --model_dir: directory HuggingFace del modello FiD-T5

Uso CLI
-------
python build_all_erag_qrels.py \
  --all_runs ../input_runs/nq/nq_all_runs.csv \
  --datasets ../data/nq-dev-kilt.jsonl \
  --collection ../data/collection/wikipedia_passages.jsonl \
  --model_dir ../models/fid_t5 \
  --metric em \
  --overwrite
"""

from __future__ import annotations

import os
import json
import csv
import argparse
from typing import Dict, List, Tuple, Optional, Set, Iterable, Any
from functools import partial
from collections import Counter

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from fid_t5 import t5_fid_generator
from metrics import exact_match_metric, f1_metric


# -----------------------------
# Downstream metriche (come build_erag_qrels.py)
# -----------------------------
METRICS = {
    "em": exact_match_metric,
    "f1": f1_metric,
    "accuracy": exact_match_metric,  # nel progetto coincide con EM
}


# -----------------------------
# Load all_runs (query_id, doc_id)
# -----------------------------
def load_all_runs_grouped(all_runs_path: str) -> Tuple[Dict[str, List[str]], Set[str], Set[str]]:
    """
    Carica un CSV <dataset>_all_runs.csv e ritorna:
      - pairs_by_qid: {query_id: [doc_id1, doc_id2, ...]} (dedup per query_id, ordine di apparizione)
      - qids: set(query_id)
      - docids: set(doc_id)

    Nota: non esiste score, quindi l'ordine non è un ranking; è solo un ordine di processazione.
    """
    if not os.path.exists(all_runs_path):
        raise FileNotFoundError(f"all_runs not found: {all_runs_path}")

    pairs_by_qid: Dict[str, List[str]] = {}
    qids: Set[str] = set()
    docids: Set[str] = set()

    # dedup per qid preservando ordine
    seen_per_qid: Dict[str, Set[str]] = {}

    with open(all_runs_path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            qid = str(row.get("query_id", "")).strip()
            did = str(row.get("doc_id", "")).strip()
            if not qid or not did:
                continue

            qids.add(qid)
            docids.add(did)

            if qid not in pairs_by_qid:
                pairs_by_qid[qid] = []
                seen_per_qid[qid] = set()

            if did in seen_per_qid[qid]:
                continue
            seen_per_qid[qid].add(did)
            pairs_by_qid[qid].append(did)

    return pairs_by_qid, qids, docids


# -----------------------------
# Dataset KILT loading (solo qids nel run)
# -----------------------------
def load_kilt_maps_for_qids(
    datasets: List[str],
    needed_qids: Set[str],
    *,
    max_examples: Optional[int] = None,
    prefix_with_dataset: bool = True,
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Returns:
      query_id_to_query: {qid: query_text}
      expected_outputs:  {qid: [gold_answer1, ...]}
    """
    if not datasets:
        raise ValueError("No datasets provided.")

    multi = len(datasets) > 1
    qid2query: Dict[str, str] = {}
    qid2golds: Dict[str, List[str]] = {}
    remaining = set(needed_qids)

    for p in datasets:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Dataset not found: {p}")

        prefix = None
        if prefix_with_dataset and multi:
            prefix = os.path.splitext(os.path.basename(p))[0]

        with open(p, "r", encoding="utf-8") as f:
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

                raw_qid = str(rec.get("id", i))
                qid = f"{prefix}:{raw_qid}" if prefix else raw_qid
                if qid not in remaining:
                    continue

                query_text = (rec.get("input") or "").strip()

                outputs = rec.get("output") or []
                golds: List[str] = []
                if isinstance(outputs, list):
                    for out in outputs:
                        ans = (out or {}).get("answer", None)
                        if isinstance(ans, str) and ans.strip():
                            golds.append(ans.strip())

                qid2query[qid] = query_text
                qid2golds[qid] = golds
                remaining.discard(qid)

                if not remaining:
                    break

        if not remaining:
            break

    return qid2query, qid2golds


# -----------------------------
# Collection loading (solo docids nel run)
# -----------------------------
def _all_numeric_docids(docids: Set[str]) -> bool:
    return bool(docids) and all(str(d).isdigit() for d in docids)


def load_collection_contents_by_id_scan(collection_path: str, needed_docids: Set[str]) -> Dict[str, str]:
    """
    Scan streaming JSONL: match su campo "id" (doc_id stringa tipo 40885965_20).
    """
    doc_map: Dict[str, str] = {}
    remaining = set(needed_docids)

    with open(collection_path, "r", encoding="utf-8") as f:
        for line in f:
            if not remaining:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            did = str(rec.get("id", "")).strip()
            if not did or did not in remaining:
                continue

            contents = rec.get("contents") or rec.get("text") or ""
            doc_map[did] = contents
            remaining.discard(did)

    return doc_map


def load_collection_contents_by_line_offsets(
    collection_path: str,
    offsets_path: str,
    needed_docids: Set[str],
) -> Dict[str, str]:
    """
    Lookup random-access per doc_id numerici (line number) usando offsets uint64.
    """
    import mmap
    import struct

    ids = sorted({int(d) for d in needed_docids})
    doc_map: Dict[str, str] = {}

    with open(offsets_path, "rb") as off_f, open(collection_path, "rb") as col_f:
        mm = mmap.mmap(off_f.fileno(), 0, access=mmap.ACCESS_READ)
        count = len(mm) // 8

        def offset_at(i: int) -> int:
            return struct.unpack_from("<Q", mm, i * 8)[0]

        for i in ids:
            if i < 0 or i >= count:
                continue
            col_f.seek(offset_at(i))
            line = col_f.readline()
            try:
                rec = json.loads(line.decode("utf-8"))
            except Exception:
                continue
            contents = rec.get("contents") or rec.get("text") or ""
            doc_map[str(i)] = contents

        mm.close()

    return doc_map


def load_collection_contents_by_line_scan(collection_path: str, needed_docids: Set[str]) -> Dict[str, str]:
    """
    Fallback lento: scan e cattura solo le linee richieste (doc_id numerico = numero riga).
    """
    needed_idx = {int(d) for d in needed_docids if str(d).isdigit()}
    doc_map: Dict[str, str] = {}

    with open(collection_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i not in needed_idx:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            contents = rec.get("contents") or rec.get("text") or ""
            doc_map[str(i)] = contents
            if len(doc_map) >= len(needed_idx):
                break

    return doc_map


def load_doc_id_to_document(
    collection_path: str,
    needed_docids: Set[str],
    *,
    offsets_path: Optional[str] = None,
) -> Dict[str, str]:
    if not os.path.exists(collection_path):
        raise FileNotFoundError(f"Collection not found: {collection_path}")

    if _all_numeric_docids(needed_docids):
        if offsets_path:
            if not os.path.exists(offsets_path):
                raise FileNotFoundError(f"Offsets not found: {offsets_path}")
            return load_collection_contents_by_line_offsets(collection_path, offsets_path, needed_docids)
        return load_collection_contents_by_line_scan(collection_path, needed_docids)

    return load_collection_contents_by_id_scan(collection_path, needed_docids)


# -----------------------------
# Output helpers
# -----------------------------
def infer_dataset_name_from_all_runs_path(all_runs_path: str) -> str:
    base = os.path.splitext(os.path.basename(all_runs_path))[0]
    suffix = "_all_runs"
    if base.endswith(suffix):
        return base[: -len(suffix)]
    return base


def default_output_path(all_runs_path: str) -> str:
    dataset = infer_dataset_name_from_all_runs_path(all_runs_path)
    out_dir = os.path.dirname(all_runs_path) or "."
    return os.path.join(out_dir, f"{dataset}_all_qrels.csv")


def format_relevance(score: float, score_precision: int = 6) -> Any:
    """
    Formatta relevance come in build_erag_qrels.py:
    - int se è “quasi intero” (EM/accuracy)
    - altrimenti float formattato
    """
    try:
        s = float(score)
    except Exception:
        s = 0.0

    if abs(s - round(s)) < 1e-12:
        return int(round(s))

    fmt = f"{{:.{int(score_precision)}f}}"
    return float(fmt.format(s))


# -----------------------------
# Core: compute and write qrels (only relevance > 0)
# -----------------------------
def write_all_erag_qrels_csv(
    pairs_by_qid: Dict[str, List[str]],
    query_id_to_query: Dict[str, str],
    doc_id_to_document: Dict[str, str],
    expected_outputs: Dict[str, List[str]],
    text_generator,
    downstream_metric_func,
    out_path: str,
    *,
    score_precision: int = 6,
) -> None:
    """
    Esegue generazione+metriche per ogni coppia (qid, docid) in pairs_by_qid,
    e scrive streaming le righe con relevance > 0.

    Strategia batching:
    - Iteriamo per “posizione i” nelle liste doc_id (simile all'ID-mode in erag_mod.eval)
    - Nella stessa iterazione batchiamo solo query_text uniche; duplicati nello stesso batch
      vengono gestiti uno-a-uno (evita collisioni di chiavi nel dict per il generator).
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    qids = list(pairs_by_qid.keys())
    max_len = max((len(lst) for lst in pairs_by_qid.values()), default=0)

    with open(out_path, "w", encoding="utf-8", newline="") as f_out:
        w = csv.writer(f_out)
        w.writerow(["query_id", "doc_id", "relevance"])

        written = 0
        processed = 0

        for i in range(max_len):
            # items: (qid, qtext, doc_id, doc_text)
            items: List[Tuple[str, str, str, str]] = []
            for qid in qids:
                doc_ids = pairs_by_qid.get(qid, []) or []
                if i >= len(doc_ids):
                    continue

                if qid not in query_id_to_query:
                    raise KeyError(f"query_id_to_query missing query_id: {qid}")

                doc_id = str(doc_ids[i])
                qtext = query_id_to_query[qid]
                dtext = doc_id_to_document.get(doc_id, "")

                items.append((qid, qtext, doc_id, dtext))

            if not items:
                continue

            # Batch solo su query_text uniche; duplicati -> uno-a-uno
            qtext_counts = Counter(qtext for (_qid, qtext, _did, _dtext) in items)

            batch_input: Dict[str, List[str]] = {}
            batch_expected: Dict[str, List[str]] = {}
            backmap: List[Tuple[str, str, str]] = []  # (qid, qtext, doc_id)
            dup_items: List[Tuple[str, str, str, str]] = []

            for qid, qtext, doc_id, doc_text in items:
                if qtext_counts[qtext] == 1:
                    batch_input[qtext] = [doc_text]
                    batch_expected[qtext] = expected_outputs.get(qid, []) or []
                    backmap.append((qid, qtext, doc_id))
                else:
                    dup_items.append((qid, qtext, doc_id, doc_text))

            # --- Batch part ---
            if batch_input:
                generated = text_generator(batch_input)
                if set(generated.keys()) != set(batch_input.keys()):
                    raise RuntimeError("The text_generator function did not return outputs for all given inputs.")

                scores = downstream_metric_func(generated, batch_expected)
                if set(scores.keys()) != set(generated.keys()):
                    raise RuntimeError("The downstream_metric function did not return evaluation scores for all given inputs.")

                for qid, qtext, doc_id in backmap:
                    processed += 1
                    s = float(scores.get(qtext, 0.0))
                    if s > 0.0:
                        w.writerow([qid, doc_id, format_relevance(s, score_precision=score_precision)])
                        written += 1

            # --- Duplicate query_text in same batch: one-by-one ---
            for qid, qtext, doc_id, doc_text in dup_items:
                generated = text_generator({qtext: [doc_text]})
                if qtext not in generated:
                    raise RuntimeError("The text_generator function did not return outputs for all given inputs.")
                scores = downstream_metric_func(generated, {qtext: expected_outputs.get(qid, []) or []})
                if qtext not in scores:
                    raise RuntimeError("The downstream_metric function did not return evaluation scores for all given inputs.")

                processed += 1
                s = float(scores.get(qtext, 0.0))
                if s > 0.0:
                    w.writerow([qid, doc_id, format_relevance(s, score_precision=score_precision)])
                    written += 1

            if (i + 1) % 10 == 0:
                print(f"Processed rank-position i={i+1}/{max_len}. Pairs processed so far: {processed}, written: {written}")

        print(f"Done. Total pairs processed: {processed}, written (relevance>0): {written}")
        print(f"Saved qrels -> {out_path}")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a single eRAG qrels CSV from <dataset>_all_runs.csv, keeping only relevance>0 pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--all_runs", type=str, required=True, help="Path to <dataset>_all_runs.csv (query_id, doc_id).")

    parser.add_argument("--datasets", type=str, nargs="+", required=True, help="KILT dataset(s) .jsonl")
    parser.add_argument("--max_examples", type=int, default=None, help="Limit examples per dataset (debug)")
    parser.add_argument(
        "--prefix_with_dataset",
        action="store_true",
        help="If multiple datasets, prefix query_id with dataset basename (must match all_runs).",
    )

    parser.add_argument("--collection", type=str, required=True, help="Collection JSONL with fields id, contents")
    parser.add_argument(
        "--offsets",
        type=str,
        default=None,
        help="Offsets uint64 for random access (only meaningful if doc_id are numeric line ids).",
    )

    parser.add_argument("--model_dir", type=str, required=True, help="FiD-T5 model directory")
    parser.add_argument("--max_input_len", type=int, default=256)
    parser.add_argument("--max_output_len", type=int, default=64)
    parser.add_argument("--num_beams", type=int, default=4)

    parser.add_argument("--metric", choices=list(METRICS.keys()), default="em")
    parser.add_argument("--score_precision", type=int, default=6)

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path. Default: alongside all_runs -> <dataset>_all_qrels.csv",
    )
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    out_path = args.output or default_output_path(args.all_runs)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if os.path.exists(out_path) and not args.overwrite:
        raise FileExistsError(f"Output exists: {out_path} (use --overwrite)")

    downstream_metric_func = METRICS[args.metric]
    print(f"Downstream metric: {args.metric}")

    # 1) load all_runs pairs
    pairs_by_qid, needed_qids, needed_docids = load_all_runs_grouped(args.all_runs)
    if not pairs_by_qid:
        raise RuntimeError("all_runs produced no rows.")
    n_pairs = sum(len(v) for v in pairs_by_qid.values())
    print(f"all_runs loaded: qids={len(needed_qids)}, unique_docids={len(needed_docids)}, total_pairs={n_pairs}")

    # 2) dataset -> query_id_to_query + expected_outputs
    query_id_to_query, expected_outputs = load_kilt_maps_for_qids(
        args.datasets,
        needed_qids,
        max_examples=args.max_examples,
        prefix_with_dataset=args.prefix_with_dataset,
    )
    missing_qids = sorted(list(needed_qids - set(query_id_to_query.keys())))
    if missing_qids:
        raise KeyError(f"{len(missing_qids)} query_id from all_runs not found in dataset(s). Examples: {missing_qids[:10]}")

    # 3) collection -> doc_id_to_document
    doc_id_to_document = load_doc_id_to_document(args.collection, needed_docids, offsets_path=args.offsets)
    missing_docids = sorted(list(needed_docids - set(doc_id_to_document.keys())))
    if missing_docids:
        print(
            f"WARNING: {len(missing_docids)} doc_id from all_runs not found in collection. "
            f"Examples: {missing_docids[:10]}. They will use empty contents."
        )
        for did in missing_docids:
            doc_id_to_document[did] = ""

    # 4) load model + generator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model: {args.model_dir} on {device}")
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir)
    tokenizer = T5Tokenizer.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    t5_generator = partial(
        t5_fid_generator,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_input_len=args.max_input_len,
        max_output_len=args.max_output_len,
        num_beams=args.num_beams,
    )

    # 5) compute relevance and write CSV (only >0)
    torch.cuda.empty_cache()
    write_all_erag_qrels_csv(
        pairs_by_qid=pairs_by_qid,
        query_id_to_query=query_id_to_query,
        doc_id_to_document=doc_id_to_document,
        expected_outputs=expected_outputs,
        text_generator=t5_generator,
        downstream_metric_func=downstream_metric_func,
        out_path=out_path,
        score_precision=args.score_precision,
    )


if __name__ == "__main__":
    main()