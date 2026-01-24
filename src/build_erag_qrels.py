"""
build_erag_qrels.py

Costruisce qrels eRAG (query_id, doc_id, relevance) dove relevance è la performance downstream
(EM/F1) ottenuta generando una risposta usando SOLO quel documento.

Input:
- run file (csv/json/txt) stile trec-rag: query_id, doc_id, score, run_id
- dataset KILT .jsonl (per query_id->query_text e query_id->gold_answers)
- collection .jsonl (per doc_id->contents)

Output (in output_dir):
- aggregated_<runid>_<metric>.json
- per_input_<runid>_<metric>.json
- qrels_<runid>_<metric>.csv    (query_id, doc_id, relevance)
- triples_<runid>_<metric>.json (debug: query_id, doc_id, score)

Esegue erag_mod.eval in ID-mode:
  retrieval_results: {query_id: [doc_id1, doc_id2, ...]}
  expected_outputs:  {query_id: [gold1, gold2, ...]}
  query_id_to_query: {query_id: query_text}
  doc_id_to_document:{doc_id: doc_text}

Uso CLI:

1) BM25 / Dense-sharded (doc_id stringa tipo "<wikipedia_id>_<segment_id>"):
python build_erag_qrels.py \
  --run ../runs/nq_bm25.csv \
  --run_format csv \
  --datasets ../data/nq-dev-kilt.jsonl \
  --collection ../data/collection/wikipedia_passages.jsonl \
  --model_dir ../models/fid_t5 \
  --metric em \
  --k_values 10 30 50 \
  --output_dir ../logs/erag_qrels/nq_bm25_em \
  --overwrite

2) Contriever (doc_id numerici = row-id della collection) + offsets per lookup veloce:
python build_erag_qrels.py \
  --run ../runs/nq_contriever.csv \
  --run_format csv \
  --datasets ../data/nq-dev-kilt.jsonl \
  --collection ../data/collection/wikipedia_passages.jsonl \
  --offsets ../indexes/index_out_full/collection_offsets.u64.bin \
  --model_dir ../models/fid_t5 \
  --metric em \
  --k_values 10 30 50 \
  --output_dir ../logs/erag_qrels/nq_contriever_em \
  --overwrite
"""

from __future__ import annotations

import os
import json
import csv
import argparse
from typing import Dict, List, Tuple, Optional, Iterable, Any, Set
from functools import partial

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

import erag_mod
from fid_t5 import t5_fid_generator
from metrics import exact_match_metric, f1_metric


# -----------------------------
# Metriche (come evaluation.py)
# -----------------------------
METRICS = {
    "em": exact_match_metric,
    "f1": f1_metric,
    "accuracy": exact_match_metric,  # nel tuo progetto coincide con EM
}


def define_retrieval_metrics(k_values: List[int], metric: str) -> Tuple[int, List[str]]:
    retrieval_metrics: List[str] = []
    for k in k_values:
        retrieval_metrics.extend([f"P_{k}", f"success_{k}"])
        # con f1: relevance non binaria => non possiamo calcolare recall/ndcg/map/rr
        if metric != "f1":
            retrieval_metrics.extend([f"recall_{k}", f"ndcg_cut_{k}", f"map_cut_{k}", f"recip_rank_cut_{k}"])
    return max(k_values), retrieval_metrics


# -----------------------------
# Run parsing (query_id, doc_id, score, run_id)
# -----------------------------
def infer_run_format(run_path: str) -> str:
    ext = os.path.splitext(run_path)[1].lower()
    if ext == ".csv":
        return "csv"
    if ext == ".json":
        return "json"
    return "txt"


def _autodetect_sep(sample_line: str) -> Optional[str]:
    if "\t" in sample_line:
        return "\t"
    if "," in sample_line:
        return ","
    return None  # whitespace split


def iter_run_rows_csv(run_path: str) -> Iterable[Tuple[str, str, float, str]]:
    with open(run_path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            qid = str(row.get("query_id", "")).strip()
            docid = str(row.get("doc_id", "")).strip()
            run_id = str(row.get("run_id", "")).strip() or "run"
            try:
                score = float(row.get("score", 0.0))
            except Exception:
                score = 0.0
            if qid and docid:
                yield qid, docid, score, run_id


def iter_run_rows_txt(run_path: str, sep: Optional[str] = None) -> Iterable[Tuple[str, str, float, str]]:
    detected_sep = sep
    first_data_line_checked = False

    with open(run_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if not first_data_line_checked:
                # skip optional header
                if line.lower().startswith("query_id"):
                    first_data_line_checked = True
                    continue
                if detected_sep is None:
                    detected_sep = _autodetect_sep(line)
                first_data_line_checked = True

            parts = line.split(detected_sep) if detected_sep is not None else line.split()
            if len(parts) < 4:
                continue

            qid = parts[0].strip()
            docid = parts[1].strip()

            try:
                score = float(parts[2])
            except Exception:
                score = 0.0

            run_id = parts[3].strip() or "run"

            if qid and docid:
                yield qid, docid, score, run_id


def iter_run_rows_json(run_path: str) -> Iterable[Tuple[str, str, float, str]]:
    with open(run_path, "r", encoding="utf-8") as f:
        # peek first non-whitespace
        first_char = ""
        while True:
            c = f.read(1)
            if not c:
                break
            if not c.isspace():
                first_char = c
                break
        f.seek(0)

        if first_char == "[":
            data = json.load(f)
            if not isinstance(data, list):
                return
            for obj in data:
                if not isinstance(obj, dict):
                    continue
                qid = str(obj.get("query_id", "")).strip()
                docid = str(obj.get("doc_id", "")).strip()
                run_id = str(obj.get("run_id", "")).strip() or "run"
                try:
                    score = float(obj.get("score", 0.0))
                except Exception:
                    score = 0.0
                if qid and docid:
                    yield qid, docid, score, run_id
        else:
            # jsonl best-effort
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                qid = str(obj.get("query_id", "")).strip()
                docid = str(obj.get("doc_id", "")).strip()
                run_id = str(obj.get("run_id", "")).strip() or "run"
                try:
                    score = float(obj.get("score", 0.0))
                except Exception:
                    score = 0.0
                if qid and docid:
                    yield qid, docid, score, run_id


def iter_run_rows(
    run_path: str,
    run_format: Optional[str] = None,
    run_txt_sep: Optional[str] = None,
) -> Iterable[Tuple[str, str, float, str]]:
    fmt = (run_format or infer_run_format(run_path)).lower()
    if fmt == "csv":
        yield from iter_run_rows_csv(run_path)
    elif fmt == "json":
        yield from iter_run_rows_json(run_path)
    elif fmt == "txt":
        yield from iter_run_rows_txt(run_path, sep=run_txt_sep)
    else:
        raise ValueError(f"Unknown run format: {fmt}")


def load_run_grouped(
    run_path: str,
    *,
    run_format: Optional[str] = None,
    run_txt_sep: Optional[str] = None,
) -> Tuple[Dict[str, List[str]], Set[str], Set[str], str]:
    """
    Ritorna:
      retrieval_results: {query_id: [doc_id1, doc_id2, ...]} ordinati per score desc
      qids: set(query_id)
      docids: set(doc_id)
      run_id: primo run_id non vuoto
    """
    per_q: Dict[str, List[Tuple[str, float]]] = {}
    qids: Set[str] = set()
    docids: Set[str] = set()
    run_id = "run"

    for qid, docid, score, rid in iter_run_rows(run_path, run_format=run_format, run_txt_sep=run_txt_sep):
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
# Output writing
# -----------------------------
def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_qrels_csv(triples: List[Dict[str, Any]], out_path: str, score_precision: int = 6) -> None:
    """
    Scrive CSV con header identico a build_qrels.py: query_id, doc_id, relevance

    - se lo score è intero (EM/accuracy) scrive int 0/1
    - se è float (F1) scrive float formattato
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fmt = f"{{:.{int(score_precision)}f}}"

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "doc_id", "relevance"])
        for t in triples:
            qid = str(t.get("query_id", ""))
            did = str(t.get("doc_id", ""))
            try:
                s = float(t.get("score", 0.0))
            except Exception:
                s = 0.0

            if abs(s - round(s)) < 1e-12:
                rel_out: Any = int(round(s))
            else:
                rel_out = float(fmt.format(s))

            w.writerow([qid, did, rel_out])


# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build eRAG qrels from a retrieval run file, KILT dataset(s), and collection JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Run
    parser.add_argument("--run", type=str, required=True, help="Run file path (csv/json/txt) from build_retrieval_run.py")
    parser.add_argument("--run_format", choices=["txt", "csv", "json"], default=None, help="Override run format")
    parser.add_argument("--run_txt_sep", type=str, default=None, help="Separator for txt parsing (default auto)")

    # Dataset
    parser.add_argument("--datasets", type=str, nargs="+", required=True, help="KILT dataset(s) .jsonl")
    parser.add_argument("--max_examples", type=int, default=None, help="Limit examples per dataset (debug)")
    parser.add_argument(
        "--prefix_with_dataset",
        action="store_true",
        help="If multiple datasets, prefix query_id with dataset basename (must match run).",
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
    parser.add_argument("--k_values", type=int, nargs="+", default=[50])
    parser.add_argument("--score_precision", type=int, default=6)

    # Output
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) metriche retrieval (come evaluation.py)
    _doc_n, retrieval_metrics = define_retrieval_metrics(args.k_values, args.metric)
    downstream_metric_func = METRICS[args.metric]
    print(f"Downstream metric: {args.metric}")
    print(f"Retrieval metrics: {retrieval_metrics}")

    # 2) run -> retrieval_results {qid:[docid,...]}
    retrieval_results, needed_qids, needed_docids, run_id = load_run_grouped(
        args.run, run_format=args.run_format, run_txt_sep=args.run_txt_sep
    )
    if not retrieval_results:
        raise RuntimeError("Run file produced no rows.")
    print(f"Run loaded: qids={len(needed_qids)}, unique_docids={len(needed_docids)}, run_id={run_id}")

    # 3) dataset -> query_id_to_query + expected_outputs
    query_id_to_query, expected_outputs = load_kilt_maps_for_qids(
        args.datasets,
        needed_qids,
        max_examples=args.max_examples,
        prefix_with_dataset=args.prefix_with_dataset,
    )

    missing_qids = sorted(list(needed_qids - set(query_id_to_query.keys())))
    if missing_qids:
        raise KeyError(
            f"{len(missing_qids)} query_id from run not found in dataset(s). Examples: {missing_qids[:10]}"
        )

    # assicura chiavi identiche (erag_mod.eval richiede match)
    expected_outputs = {qid: expected_outputs.get(qid, []) for qid in retrieval_results.keys()}

    # 4) collection -> doc_id_to_document
    doc_id_to_document = load_doc_id_to_document(args.collection, needed_docids, offsets_path=args.offsets)
    missing_docids = sorted(list(needed_docids - set(doc_id_to_document.keys())))
    if missing_docids:
        print(
            f"WARNING: {len(missing_docids)} doc_id from run not found in collection. "
            f"Examples: {missing_docids[:10]}. They will use empty contents."
        )
        for did in missing_docids:
            doc_id_to_document[did] = ""

    # 5) model + generator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model: {args.model_dir} on {device}")
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir)
    tokenizer = T5Tokenizer.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    t5_generator_for_eval = partial(
        t5_fid_generator,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_input_len=args.max_input_len,
        max_output_len=args.max_output_len,
        num_beams=args.num_beams,
    )

    # 6) erag eval (ID-mode)
    torch.cuda.empty_cache()
    print("Running erag_mod.eval (ID-mode)...")
    erag_results = erag_mod.eval(
        retrieval_results=retrieval_results,
        expected_outputs=expected_outputs,
        text_generator=t5_generator_for_eval,
        downstream_metric=downstream_metric_func,
        retrieval_metrics=set(retrieval_metrics),
        inputs_are_ids=True,
        query_id_to_query=query_id_to_query,
        doc_id_to_document=doc_id_to_document,
    )

    # 7) save logs
    tag = f"{run_id}_{args.metric}"

    aggregated_path = os.path.join(args.output_dir, f"aggregated_{tag}.json")
    per_input_path = os.path.join(args.output_dir, f"per_input_{tag}.json")
    triples_path = os.path.join(args.output_dir, f"triples_{tag}.json")
    qrels_csv_path = os.path.join(args.output_dir, f"qrels_{tag}.csv")

    for p in (aggregated_path, per_input_path, triples_path, qrels_csv_path):
        if os.path.exists(p) and not args.overwrite:
            raise FileExistsError(f"Output exists: {p} (use --overwrite)")

    save_json(erag_results.get("aggregated", {}), aggregated_path)
    save_json(erag_results.get("per_input", {}), per_input_path)

    triples = erag_results.get("triples", []) or []
    save_json(triples, triples_path)
    write_qrels_csv(triples, qrels_csv_path, score_precision=args.score_precision)

    print("Saved outputs:")
    print(f"  aggregated: {aggregated_path}")
    print(f"  per_input:  {per_input_path}")
    print(f"  triples:    {triples_path}")
    print(f"  qrels csv:  {qrels_csv_path}")


if __name__ == "__main__":
    main()