"""
build_erag_qrels.py

Script end-to-end per riprodurre eRAG e generare qrels “model-based”.

Obiettivi
---------
1) Generare qrels eRAG (query_id, doc_id, relevance) dove:
   - doc_id proviene dal run file di retrieval
   - relevance è il punteggio downstream (EM/Accuracy binario oppure F1 continuo in [0,1])
     ottenuto facendo generazione usando SOLO quel documento come contesto (eRAG).

2) Calcolare anche:
   - valutazione end-to-end (FiD) per ogni k in k_values, loggando i punteggi in CSV
   - correlazioni (Spearman/Kendall) tra metriche eRAG e punteggi end-to-end

Input
-----
- Run file di retrieval (csv/json/txt) stile trec-rag, generabile con build_retrieval_run.py:
    query_id, doc_id, score, run_id
  Il ranking viene ricostruito ordinando per score decrescente (dedup doc_id per query).

- Dataset KILT (.jsonl):
  usato per costruire:
    * query_id -> query_text
    * query_id -> lista gold answers (expected_outputs)

- Collezione documenti (.jsonl):
  usata per costruire:
    * doc_id -> document contents
  Supporta due casi:
    * doc_id stringa (es. "<wikipedia_id>_<segment_id>"): lookup per match sul campo JSON "id" (scan streaming)
    * doc_id numerico (es. Contriever): interpreta doc_id come row-id nella collection;
      se passi --offsets (uint64) usa random-access (consigliato), altrimenti fa scan lento.

- Modello FiD-T5 (directory HuggingFace) per generazione.

Metriche
--------
- Downstream metric: em | accuracy | f1
- Retrieval metrics:
  per ogni k in k_values:
    P_k, success_k
    + (se metric != "f1") recall_k, ndcg_cut_k, map_cut_k, recip_rank_cut_k

Esecuzione eRAG (ID-mode)
------------------------
Chiama erag_mod.eval con:
  retrieval_results: {query_id: [doc_id1, doc_id2, ...]}
  expected_outputs:  {query_id: [gold1, gold2, ...]}
  query_id_to_query: {query_id: query_text}
  doc_id_to_document:{doc_id: doc_text}

Output (in output_dir)
----------------------
Log eRAG:
- aggregated_<runid>_<metric>.json
- per_input_<runid>_<metric>.json
- triples_<runid>_<metric>.json
    lista di {query_id, doc_id, score} dove score = downstream_metric (label di rilevanza eRAG)
- qrels_<runid>_<metric>.csv
    CSV con header: query_id, doc_id, relevance
    (stessa struttura di build_qrels.py; relevance è int per EM/accuracy, float per F1)

Log end-to-end:
- end_to_end_<runid>_<metric>.csv
    righe: query_id, score, k
- end_to_end_averages_<runid>_<metric>.csv
    righe: k, average_score

Log correlazioni:
- correlations_<runid>_<metric>.json
    per ogni metrica eRAG: Spearman/Kendall rispetto agli end-to-end score (k scelto dal suffisso della metrica)

Uso CLI
-------

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


def _limit_docids_per_query(retrieval_results: Dict[str, List[str]], k: int) -> Dict[str, List[str]]:
    """Ritorna una copia di retrieval_results limitata ai primi k doc_id per query_id."""
    return {qid: (docs or [])[:k] for qid, docs in retrieval_results.items()}


def _build_generator_input_from_ids(
    retrieval_results_topk: Dict[str, List[str]],
    query_id_to_query: Dict[str, str],
    doc_id_to_document: Dict[str, str],
) -> Dict[str, List[str]]:
    """
    Converte {query_id: [doc_id,...]} in {query_text: [doc_text,...]} per il generator.
    Se più query_id condividono lo stesso query_text, li deduplica (stessa generazione).
    """
    gen_input: Dict[str, List[str]] = {}
    for qid, docids in retrieval_results_topk.items():
        qtext = query_id_to_query.get(qid, None)
        if not qtext:
            continue

        if qtext in gen_input:
            continue  # dedup per query text

        docs_text = [doc_id_to_document.get(str(did), "") for did in (docids or [])]
        gen_input[qtext] = docs_text
    return gen_input


def evaluation_e2e_to_csv(
    retrieval_results: Dict[str, List[str]],
    query_id_to_query: Dict[str, str],
    doc_id_to_document: Dict[str, str],
    expected_outputs: Dict[str, List[str]],
    text_generator,
    downstream_metric_func,
    k_values: List[int],
    output_dir: str,
    tag: str,
    *,
    overwrite: bool = False,
    score_precision: int = 6,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Valutazione end-to-end per ogni k in k_values.

    Log:
    - CSV per-query: query_id, score, k
    - CSV medie: k, average_score

    Returns:
      all_e2e_scores: {"k_<k>": {query_id: score}}
      average_e2e_scores: {"k_<k>": avg_score}
    """
    os.makedirs(output_dir, exist_ok=True)

    per_query_csv = os.path.join(output_dir, f"end_to_end_{tag}.csv")
    avg_csv = os.path.join(output_dir, f"end_to_end_averages_{tag}.csv")

    for p in (per_query_csv, avg_csv):
        if os.path.exists(p) and not overwrite:
            raise FileExistsError(f"Output exists: {p} (use --overwrite)")

    qids = sorted(retrieval_results.keys())
    fmt = f"{{:.{int(score_precision)}f}}"

    all_e2e_scores: Dict[str, Dict[str, float]] = {}
    average_e2e_scores: Dict[str, float] = {}

    with open(per_query_csv, "w", encoding="utf-8", newline="") as f_per, open(
        avg_csv, "w", encoding="utf-8", newline=""
    ) as f_avg:
        w_per = csv.writer(f_per)
        w_avg = csv.writer(f_avg)

        w_per.writerow(["query_id", "score", "k"])
        w_avg.writerow(["k", "average_score"])

        for k in sorted(set(int(x) for x in k_values)):
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
            # enforce all qids
            scores_by_qid = {qid: float(scores_by_qid.get(qid, 0.0)) for qid in qids}

            avg = (sum(scores_by_qid.values()) / len(scores_by_qid)) if scores_by_qid else 0.0

            key = f"k_{k}"
            all_e2e_scores[key] = scores_by_qid
            average_e2e_scores[key] = float(avg)

            # write rows
            for qid in qids:
                w_per.writerow([qid, fmt.format(scores_by_qid[qid]), k])
            w_avg.writerow([k, fmt.format(avg)])

    print(f"Saved end-to-end per-query CSV -> {per_query_csv}")
    print(f"Saved end-to-end averages CSV -> {avg_csv}")

    return all_e2e_scores, average_e2e_scores


def compute_correlations_and_log(
    erag_results: Dict[str, Any],
    retrieval_metrics: List[str],
    all_e2e_scores: Dict[str, Dict[str, float]],
    default_k: int,
    output_dir: str,
    tag: str,
    *,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Calcola correlazioni Spearman/Kendall tra metriche eRAG (per_input) e punteggi end-to-end.
    Salva un JSON: correlations_<tag>.json

    Returns:
      correlations: {metric_name: {spearman_corr, spearman_p, kendall_corr, kendall_p}}
    """
    import re
    import scipy.stats as stats

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"correlations_{tag}.json")
    if os.path.exists(out_path) and not overwrite:
        raise FileExistsError(f"Output exists: {out_path} (use --overwrite)")

    per_input = erag_results.get("per_input", {}) or {}

    def _select_k_key_for_metric(metric_name: str, fallback: int) -> Tuple[str, int]:
        m = re.search(r"(\d+)$", metric_name)
        k = int(m.group(1)) if m else int(fallback)
        return f"k_{k}", k

    correlations: Dict[str, Any] = {}

    for metric_name in retrieval_metrics:
        e2e_key, k_used = _select_k_key_for_metric(metric_name, default_k)
        e2e_scores_dict = all_e2e_scores.get(e2e_key, {}) or {}

        if not e2e_scores_dict:
            correlations[metric_name] = {
                "spearman_corr": None,
                "spearman_p": None,
                "kendall_corr": None,
                "kendall_p": None,
                "note": f"Missing end-to-end scores for {e2e_key}",
            }
            continue

        aligned_erag = []
        aligned_e2e = []

        for qid, e2e_score in e2e_scores_dict.items():
            erag_score = (per_input.get(qid, {}) or {}).get(metric_name, 0.0)
            aligned_erag.append(float(erag_score))
            aligned_e2e.append(float(e2e_score))

        spearman_corr, spearman_p = stats.spearmanr(aligned_erag, aligned_e2e)
        kendall_corr, kendall_p = stats.kendalltau(aligned_erag, aligned_e2e)

        correlations[metric_name] = {
            "k_used": k_used,
            "n_pairs": len(aligned_erag),
            "spearman_corr": spearman_corr,
            "spearman_p": spearman_p,
            "kendall_corr": kendall_corr,
            "kendall_p": kendall_p,
        }

        print(f"\nEvaluated {len(aligned_erag)} pairs.")
        print(f"For metric {metric_name} (k={k_used}):")
        print(f"  Spearman correlation: {spearman_corr} (p={spearman_p})")
        print(f"  Kendall correlation:  {kendall_corr} (p={kendall_p})")

    save_json(correlations, out_path)
    print(f"Saved correlations -> {out_path}")
    return correlations


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
    doc_n, retrieval_metrics = define_retrieval_metrics(args.k_values, args.metric)
    downstream_metric_func = METRICS[args.metric]
    print(f"Downstream metric: {args.metric}")
    print(f"Retrieval metrics: {retrieval_metrics}")
    print(f"doc_n (fallback k): {doc_n}")

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

    # allineamento chiavi (erag_mod.eval richiede match)
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

    # 7) save eRAG logs
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

    print("Saved eRAG outputs:")
    print(f"  aggregated: {aggregated_path}")
    print(f"  per_input:  {per_input_path}")
    print(f"  triples:    {triples_path}")
    print(f"  qrels csv:  {qrels_csv_path}")

    # 8) end-to-end per k (CSV) + averages (CSV)
    print("\nEvaluating end-to-end scores for each k...")
    all_e2e_scores, avg_e2e_scores = evaluation_e2e_to_csv(
        retrieval_results=retrieval_results,
        query_id_to_query=query_id_to_query,
        doc_id_to_document=doc_id_to_document,
        expected_outputs=expected_outputs,
        text_generator=t5_generator_for_eval,
        downstream_metric_func=downstream_metric_func,
        k_values=args.k_values,
        output_dir=args.output_dir,
        tag=tag,
        overwrite=args.overwrite,
        score_precision=args.score_precision,
    )

    # 9) correlations eRAG vs end-to-end
    print("\nComputing correlations (eRAG vs end-to-end)...")
    _ = compute_correlations_and_log(
        erag_results=erag_results,
        retrieval_metrics=retrieval_metrics,
        all_e2e_scores=all_e2e_scores,
        default_k=doc_n,
        output_dir=args.output_dir,
        tag=tag,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()