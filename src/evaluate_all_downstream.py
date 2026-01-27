"""
evaluate_all_downstream.py

Valutazione end-to-end (FiD) per ogni k in k_values, loggando i punteggi in CSV.

Per ogni run file CSV in --input_folder (es. ../input_runs/nq) produce:
- <input_filename>_downstream.csv
    colonne: query_id, score, k
  (una riga per query per ogni k)

- <input_filename>_mean_downstream.csv
    colonne: score, k
  (una riga per k: media score su tutte le query)

Input
-----
- run file CSV con almeno le colonne:
    query_id, doc_id, score, run_id
  (eventuali colonne extra come relevance vengono ignorate)

- dataset KILT (.jsonl) per:
    query_id -> query_text
    query_id -> lista gold answers

- collection JSONL per:
    doc_id -> document contents

- modello FiD-T5 per generazione.

Note
----
- Il ranking viene ricostruito ordinando per score decrescente e deduplicando doc_id per query_id.
- Se nel dataset esistono query_text duplicate per query_id diversi, la generazione viene deduplicata
  per query_text (come in build_erag_qrels.py), ma la metrica downstream viene calcolata per ogni query_id.

Uso CLI
-------
python evaluate_all_downstream.py \
  --input_folder ../input_runs/nq \
  --datasets ../data/nq-dev-kilt.jsonl \
  --collection ../data/collection/wikipedia_passages.jsonl \
  --model_dir ../models/fid_t5 \
  --metric em \
  --k_values 10 30 50 \
  --overwrite
"""

from __future__ import annotations

import os
import json
import csv
import argparse
from typing import Dict, List, Tuple, Optional, Set, Any
from functools import partial

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
# File listing
# -----------------------------
def list_candidate_run_csv_files(input_folder: str) -> List[str]:
    """
    Lista i CSV che sembrano run file.
    Esclude i file di output generati dagli script precedenti.
    """
    files: List[str] = []
    for name in os.listdir(input_folder):
        p = os.path.join(input_folder, name)
        if not os.path.isfile(p):
            continue
        low = name.lower()
        if not low.endswith(".csv"):
            continue

        # skip outputs commonly produced by other scripts
        if low.endswith("_assessed_run.csv"):
            continue
        if low.endswith("_retrieval_metrics.csv") or low.endswith("_mean_metrics.csv"):
            continue
        if low.endswith("_downstream.csv") or low.endswith("_mean_downstream.csv"):
            continue
        if low.endswith("_all_runs.csv") or low.endswith("_all_qrels.csv"):
            continue

        files.append(p)

    return sorted(files)


def output_paths_for_run(run_path: str, output_folder: str) -> Tuple[str, str]:
    base = os.path.splitext(os.path.basename(run_path))[0]
    per_query = os.path.join(output_folder, f"{base}_downstream.csv")
    mean_path = os.path.join(output_folder, f"{base}_mean_downstream.csv")
    return per_query, mean_path


# -----------------------------
# Run parsing (CSV) -> {qid: [docid,...]}
# -----------------------------
def load_run_grouped_csv(
    run_path: str,
) -> Tuple[Dict[str, List[str]], Set[str], Set[str], str]:
    """
    Ritorna:
      retrieval_results: {query_id: [doc_id1, doc_id2, ...]} ordinati per score desc
      qids: set(query_id)
      docids: set(doc_id)
      run_id: primo run_id non vuoto (fallback: "run")
    """
    per_q: Dict[str, List[Tuple[str, float]]] = {}
    qids: Set[str] = set()
    docids: Set[str] = set()
    run_id = "run"

    with open(run_path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise RuntimeError(f"CSV has no header: {run_path}")

        for row in r:
            qid = str(row.get("query_id", "")).strip()
            docid = str(row.get("doc_id", "")).strip()
            rid = str(row.get("run_id", "")).strip() or "run"

            try:
                score = float(row.get("score", 0.0))
            except Exception:
                score = 0.0

            if not qid or not docid:
                continue

            qids.add(qid)
            docids.add(docid)
            if rid:
                run_id = rid

            per_q.setdefault(qid, []).append((docid, float(score)))

    retrieval_results: Dict[str, List[str]] = {}
    for qid, pairs in per_q.items():
        pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)

        # dedup doc ids preservando l'ordine
        seen: Set[str] = set()
        out: List[str] = []
        for did, _s in pairs_sorted:
            if did in seen:
                continue
            seen.add(did)
            out.append(did)
        retrieval_results[qid] = out

    return retrieval_results, qids, docids, run_id


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
# End-to-end evaluation (FiD) per k
# -----------------------------
def _limit_docids_per_query(retrieval_results: Dict[str, List[str]], k: int) -> Dict[str, List[str]]:
    return {qid: (docs or [])[:k] for qid, docs in retrieval_results.items()}


def _build_generator_input_from_ids(
    retrieval_results_topk: Dict[str, List[str]],
    query_id_to_query: Dict[str, str],
    doc_id_to_document: Dict[str, str],
) -> Dict[str, List[str]]:
    """
    Converte {query_id: [doc_id,...]} in {query_text: [doc_text,...]} per il generator.
    Deduplica per query_text (stessa generazione).
    """
    gen_input: Dict[str, List[str]] = {}
    for qid, docids in retrieval_results_topk.items():
        qtext = query_id_to_query.get(qid, None)
        if not qtext:
            continue

        if qtext in gen_input:
            continue

        docs_text = [doc_id_to_document.get(str(did), "") for did in (docids or [])]
        gen_input[qtext] = docs_text
    return gen_input


def evaluate_e2e_for_run_to_csv(
    retrieval_results: Dict[str, List[str]],
    query_id_to_query: Dict[str, str],
    doc_id_to_document: Dict[str, str],
    expected_outputs: Dict[str, List[str]],
    text_generator,
    downstream_metric_func,
    k_values: List[int],
    per_query_csv: str,
    mean_csv: str,
    *,
    overwrite: bool = False,
    score_precision: int = 6,
) -> None:
    os.makedirs(os.path.dirname(per_query_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(mean_csv) or ".", exist_ok=True)

    for p in (per_query_csv, mean_csv):
        if os.path.exists(p) and not overwrite:
            raise FileExistsError(f"Output exists: {p} (use --overwrite)")

    qids = sorted(retrieval_results.keys())
    k_values = sorted(set(int(x) for x in (k_values or [])))
    if not k_values:
        raise ValueError("k_values is empty.")

    fmt = f"{{:.{int(score_precision)}f}}"

    with open(per_query_csv, "w", encoding="utf-8", newline="") as f_per, open(
        mean_csv, "w", encoding="utf-8", newline=""
    ) as f_mean:
        w_per = csv.writer(f_per)
        w_mean = csv.writer(f_mean)

        w_per.writerow(["query_id", "score", "k"])
        w_mean.writerow(["score", "k"])

        for k in k_values:
            retrieval_topk = _limit_docids_per_query(retrieval_results, k)
            gen_input = _build_generator_input_from_ids(retrieval_topk, query_id_to_query, doc_id_to_document)

            generated_by_qtext = text_generator(gen_input) if gen_input else {}
            if set(generated_by_qtext.keys()) != set(gen_input.keys()):
                raise RuntimeError("The text_generator function did not return outputs for all given inputs.")

            # map back to qid
            generated_by_qid: Dict[str, str] = {}
            gold_by_qid: Dict[str, List[str]] = {}
            for qid in qids:
                qtext = query_id_to_query[qid]
                generated_by_qid[qid] = generated_by_qtext.get(qtext, "")
                gold_by_qid[qid] = expected_outputs.get(qid, []) or []

            scores_by_qid = downstream_metric_func(generated_by_qid, gold_by_qid) if qids else {}
            scores_by_qid = {qid: float(scores_by_qid.get(qid, 0.0)) for qid in qids}

            avg = (sum(scores_by_qid.values()) / len(scores_by_qid)) if scores_by_qid else 0.0

            for qid in qids:
                w_per.writerow([qid, fmt.format(scores_by_qid[qid]), k])

            w_mean.writerow([fmt.format(avg), k])


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate end-to-end downstream (FiD) scores for each run CSV in a folder, for multiple k.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing run CSV files.")
    parser.add_argument(
        "--output_folder",
        type=str,
        default=None,
        help="Where to write outputs (default: same as input_folder).",
    )

    # Dataset
    parser.add_argument("--datasets", type=str, nargs="+", required=True, help="KILT dataset(s) .jsonl")
    parser.add_argument("--max_examples", type=int, default=None, help="Limit examples per dataset (debug)")
    parser.add_argument(
        "--prefix_with_dataset",
        action="store_true",
        help="If multiple datasets, prefix query_id with dataset basename (must match run files).",
    )

    # Collection
    parser.add_argument("--collection", type=str, required=True, help="Collection JSONL with fields id, contents")
    parser.add_argument(
        "--offsets",
        type=str,
        default=None,
        help="Offsets uint64 for random access (only meaningful if doc_id are numeric line ids).",
    )

    # Model / generation
    parser.add_argument("--model_dir", type=str, required=True, help="FiD-T5 model directory")
    parser.add_argument("--max_input_len", type=int, default=256)
    parser.add_argument("--max_output_len", type=int, default=64)
    parser.add_argument("--num_beams", type=int, default=4)

    # Metriche
    parser.add_argument("--metric", choices=list(METRICS.keys()), default="em")
    parser.add_argument("--k_values", type=int, nargs="+", default=[10, 30, 50])
    parser.add_argument("--score_precision", type=int, default=6)

    # Output
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if not os.path.isdir(args.input_folder):
        raise NotADirectoryError(f"Input folder not found: {args.input_folder}")
    output_folder = args.output_folder or args.input_folder
    os.makedirs(output_folder, exist_ok=True)

    run_files = list_candidate_run_csv_files(args.input_folder)
    if not run_files:
        raise FileNotFoundError(f"No run CSV files found in: {args.input_folder}")

    # 1) parse runs (store per file) + union qids (for dataset loading once)
    per_file: List[Tuple[str, Dict[str, List[str]], Set[str], Set[str], str]] = []
    all_needed_qids: Set[str] = set()

    for rp in run_files:
        retrieval_results, qids, docids, run_id = load_run_grouped_csv(rp)
        if not retrieval_results:
            print(f"WARNING: run file produced no rows: {rp} (skipped)")
            continue
        per_file.append((rp, retrieval_results, qids, docids, run_id))
        all_needed_qids |= qids

    if not per_file:
        raise RuntimeError("No valid run files to process.")

    # 2) load dataset maps once (for all qids across files)
    query_id_to_query, expected_outputs = load_kilt_maps_for_qids(
        args.datasets,
        all_needed_qids,
        max_examples=args.max_examples,
        prefix_with_dataset=args.prefix_with_dataset,
    )
    missing_qids = sorted(list(all_needed_qids - set(query_id_to_query.keys())))
    if missing_qids:
        raise KeyError(f"{len(missing_qids)} query_id from run files not found in dataset(s). Examples: {missing_qids[:10]}")

    # 3) load model once
    downstream_metric_func = METRICS[args.metric]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Downstream metric: {args.metric}")
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
        batch_size=1,
    )

    # 4) per-file: load doc contents + evaluate
    for run_path, retrieval_results, qids, docids, run_id in per_file:
        per_query_out, mean_out = output_paths_for_run(run_path, output_folder)

        for p in (per_query_out, mean_out):
            if os.path.exists(p) and not args.overwrite:
                raise FileExistsError(f"Output exists: {p} (use --overwrite)")

        print("\n----------------------------------------")
        print(f"Run: {run_path}")
        print(f"  qids={len(qids)} unique_docids={len(docids)} run_id={run_id}")

        # align expected outputs to the qids in this run
        expected_outputs_this = {qid: expected_outputs.get(qid, []) for qid in retrieval_results.keys()}

        # load collection docs for this run
        doc_id_to_document = load_doc_id_to_document(args.collection, docids, offsets_path=args.offsets)
        missing_docids = sorted(list(docids - set(doc_id_to_document.keys())))
        if missing_docids:
            print(
                f"WARNING: {len(missing_docids)} doc_id from run not found in collection. "
                f"Examples: {missing_docids[:10]}. They will use empty contents."
            )
            for did in missing_docids:
                doc_id_to_document[did] = ""

        # evaluate
        torch.cuda.empty_cache()
        evaluate_e2e_for_run_to_csv(
            retrieval_results=retrieval_results,
            query_id_to_query=query_id_to_query,
            doc_id_to_document=doc_id_to_document,
            expected_outputs=expected_outputs_this,
            text_generator=t5_generator,
            downstream_metric_func=downstream_metric_func,
            k_values=args.k_values,
            per_query_csv=per_query_out,
            mean_csv=mean_out,
            overwrite=args.overwrite,
            score_precision=args.score_precision,
        )

        print(f"Saved per-query downstream -> {per_query_out}")
        print(f"Saved mean downstream      -> {mean_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()