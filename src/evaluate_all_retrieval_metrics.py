"""
evaluate_all_retrieval_metrics.py

Calcola metriche di retrieval (P@k, success@k, recall@k, nDCG@k, MAP@k, MRR@k)
a partire da file assessed run:

  <input_filename>_assessed_run.csv

con colonne:
  query_id, doc_id, score, run_id, relevance

Per ogni file assessed in --input_folder produce:
- <input_filename>_retrieval_metrics.csv
    colonne: query_id, <metrica1>, <metrica2>, ...
    (una riga per query)

- <input_filename>_mean_metrics.csv
    colonne: <metrica1>, <metrica2>, ...
    (una sola riga: media delle metriche su tutte le query)

Scelta metriche in base al tipo di downstream relevance
-------------------------------------------------------
- Se relevance è BINARIA (0/1): calcola
    P_k, success_k, recall_k, ndcg_cut_k, map_cut_k, recip_rank_cut_k
  (come build_erag_qrels.py + erag_mod.py)

- Se relevance è CONTINUA (es. F1 in [0,1]): calcola SOLO
    P_k, success_k
  (come erag_mod.py: le altre metriche non sono supportate con relevance continua)

Uso CLI
-------
python evaluate_all_retrieval_metrics.py \
  --input_folder ../input_runs/nq \
  --k_values 10 30 50 \
  --overwrite

Per relevance continua:
python evaluate_all_retrieval_metrics.py \
  --input_folder ../input_runs/nq \
  --k_values 10 30 50 \
  --continuous_relevance \
  --overwrite
"""

from __future__ import annotations

import os
import csv
import argparse
from typing import Dict, List, Tuple, Optional, Any, Set

import pytrec_eval


# -----------------------------
# Metriche (come build_erag_qrels.py / evaluation.py)
# -----------------------------
def define_retrieval_metrics(k_values: List[int], binary_relevance: bool) -> List[str]:
    retrieval_metrics: List[str] = []
    for k in k_values:
        retrieval_metrics.extend([f"P_{k}", f"success_{k}"])
        if binary_relevance:
            retrieval_metrics.extend([f"recall_{k}", f"ndcg_cut_{k}", f"map_cut_{k}", f"recip_rank_cut_{k}"])
    return retrieval_metrics


# -----------------------------
# File listing
# -----------------------------
def list_assessed_run_files(input_folder: str) -> List[str]:
    files: List[str] = []
    for name in os.listdir(input_folder):
        p = os.path.join(input_folder, name)
        if not os.path.isfile(p):
            continue
        low = name.lower()
        if not low.endswith(".csv"):
            continue
        if low.endswith("_assessed_run.csv"):
            files.append(p)
    return sorted(files)


def out_paths_for_run(run_path: str, output_folder: str) -> Tuple[str, str]:
    base = os.path.splitext(os.path.basename(run_path))[0]  # keeps "..._assessed_run"
    per_query = os.path.join(output_folder, f"{base}_retrieval_metrics.csv")
    mean_path = os.path.join(output_folder, f"{base}_mean_metrics.csv")
    return per_query, mean_path


# -----------------------------
# Parsing assessed run
# -----------------------------
def load_assessed_run(
    run_path: str,
    *,
    binary_relevance: bool,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, Any]]]:
    """
    Returns:
      run:  {qid: {docid: score}}  (score: retrieval score, higher=better)
      qrel: {qid: {docid: relevance}} (int for binary, float for continuous)
    """
    run: Dict[str, Dict[str, float]] = {}
    qrel: Dict[str, Dict[str, Any]] = {}

    # dedup per qid/docid: se ripetuta, tieni score max (e relevance max)
    seen: Dict[Tuple[str, str], Tuple[float, Any]] = {}

    with open(run_path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise RuntimeError(f"CSV has no header: {run_path}")

        needed = {"query_id", "doc_id", "score", "run_id", "relevance"}
        missing = [c for c in sorted(needed) if c not in set(r.fieldnames)]
        if missing:
            raise KeyError(f"Missing columns {missing} in assessed run: {run_path}")

        for row in r:
            qid = str(row.get("query_id", "")).strip()
            did = str(row.get("doc_id", "")).strip()
            if not qid or not did:
                continue

            try:
                score = float(row.get("score", 0.0))
            except Exception:
                score = 0.0

            raw_rel = row.get("relevance", 0)

            if binary_relevance:
                # robust: qualsiasi relevance > 0 diventa 1
                try:
                    rel_f = float(raw_rel)
                except Exception:
                    rel_f = 0.0
                rel_val: Any = int(rel_f > 0.0)
            else:
                try:
                    rel_val = float(raw_rel)
                except Exception:
                    rel_val = 0.0

            key = (qid, did)
            if key in seen:
                prev_score, prev_rel = seen[key]
                if score > prev_score:
                    seen[key] = (score, rel_val)
                else:
                    # tieni relevance max (robusto a inconsistenze)
                    try:
                        if float(rel_val) > float(prev_rel):
                            seen[key] = (prev_score, rel_val)
                    except Exception:
                        pass
            else:
                seen[key] = (score, rel_val)

    for (qid, did), (score, rel_val) in seen.items():
        run.setdefault(qid, {})[did] = float(score)
        qrel.setdefault(qid, {})[did] = rel_val

    return run, qrel


# -----------------------------
# Helpers metriche
# -----------------------------
def _is_recip_rank_k(m: str) -> bool:
    m_low = m.lower()
    return m_low.startswith("recip_rank_") and len(m_low.split("cut_")) == 2 and m_low.split("cut_")[1].isdigit()


def _top_k_run(run_dict: Dict[str, Dict[str, float]], k: int) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for qid, docs in run_dict.items():
        top = sorted(docs.items(), key=lambda x: x[1], reverse=True)[: int(k)]
        out[qid] = {d: float(s) for d, s in top}
    return out


def compute_metrics_binary(
    run: Dict[str, Dict[str, float]],
    qrel: Dict[str, Dict[str, Any]],
    metrics: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Calcolo con pytrec_eval per metriche binary-safe.
    Gestiamo recip_rank_cut_k separatamente (come erag_mod.py).
    """
    qids = sorted(run.keys())
    results: Dict[str, Dict[str, float]] = {qid: {} for qid in qids}

    recip_rank_metrics = {m for m in metrics if _is_recip_rank_k(m)}
    pytrec_metrics = set(metrics) - recip_rank_metrics

    if pytrec_metrics:
        evaluator = pytrec_eval.RelevanceEvaluator(qrel, pytrec_metrics)
        pytrec_results = evaluator.evaluate(run)
        for qid in qids:
            results[qid].update(pytrec_results.get(qid, {}) or {})

    if recip_rank_metrics:
        rr_eval = pytrec_eval.RelevanceEvaluator(qrel, {"recip_rank"})
        ks = sorted({int(m.split("cut_")[1]) for m in recip_rank_metrics})

        rr_by_k: Dict[int, Dict[str, Dict[str, float]]] = {}
        for k in ks:
            run_k = _top_k_run(run, k)
            rr_by_k[k] = rr_eval.evaluate(run_k)  # qid -> {"recip_rank": val}

        for m in recip_rank_metrics:
            k = int(m.split("cut_")[1])
            per_q = rr_by_k.get(k, {}) or {}
            for qid in qids:
                results[qid][m] = float((per_q.get(qid, {}) or {}).get("recip_rank", 0.0))

    # assicura presenza di tutte le metriche
    for qid in qids:
        for m in metrics:
            if m not in results[qid]:
                results[qid][m] = 0.0

    return results


def compute_metrics_continuous(
    run: Dict[str, Dict[str, float]],
    qrel: Dict[str, Dict[str, Any]],
    metrics: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Calcolo manuale per relevance continua:
      - success_k: max relevance nel top-k
      - P_k: mean relevance nel top-k (diviso per k anche se i doc sono meno di k)
    """
    qids = sorted(run.keys())
    results: Dict[str, Dict[str, float]] = {qid: {} for qid in qids}

    for qid in qids:
        # ranking per score desc
        ranked_docids = [d for d, _s in sorted((run.get(qid, {}) or {}).items(), key=lambda x: x[1], reverse=True)]
        labels = qrel.get(qid, {}) or {}

        for metric in metrics:
            if "_" not in metric:
                raise RuntimeError("Continuous relevance supports only metrics with cut (e.g., P_10, success_10).")

            metric_without_cut = metric[: metric.find("_")]
            cut_value = int(metric[metric.find("_") + 1 :])

            top_docids = ranked_docids[:cut_value]

            if metric_without_cut == "success":
                max_value = 0.0
                for d in top_docids:
                    try:
                        max_value = max(max_value, float(labels.get(d, 0.0)))
                    except Exception:
                        max_value = max(max_value, 0.0)
                results[qid][metric] = float(max_value)

            elif metric_without_cut == "P":
                if cut_value <= 0:
                    results[qid][metric] = 0.0
                else:
                    sum_value = 0.0
                    for d in top_docids:
                        try:
                            sum_value += float(labels.get(d, 0.0))
                        except Exception:
                            sum_value += 0.0
                    results[qid][metric] = float(sum_value) / float(cut_value)

            else:
                raise RuntimeError(
                    'Continuous relevance supports only ["success_k", "P_k"]. '
                    "Use --continuous_relevance to restrict metrics."
                )

    return results


def aggregate_means(per_query: Dict[str, Dict[str, float]], metrics: List[str]) -> Dict[str, float]:
    qids = list(per_query.keys())
    if not qids:
        return {m: 0.0 for m in metrics}

    means: Dict[str, float] = {}
    for m in metrics:
        vals = [float((per_query.get(qid, {}) or {}).get(m, 0.0)) for qid in qids]
        means[m] = (sum(vals) / len(vals)) if vals else 0.0
    return means


# -----------------------------
# Output writing
# -----------------------------
def write_per_query_csv(out_path: str, per_query: Dict[str, Dict[str, float]], metrics: List[str]) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    qids = sorted(per_query.keys())

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id"] + metrics)
        for qid in qids:
            row = [qid] + [per_query.get(qid, {}).get(m, 0.0) for m in metrics]
            w.writerow(row)


def write_means_csv(out_path: str, means: Dict[str, float], metrics: List[str]) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(metrics)
        w.writerow([means.get(m, 0.0) for m in metrics])


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute retrieval metrics for assessed run CSV files in a folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing *_assessed_run.csv files.")
    parser.add_argument(
        "--output_folder",
        type=str,
        default=None,
        help="Where to write outputs (default: same as input_folder).",
    )
    parser.add_argument("--k_values", type=int, nargs="+", default=[10, 30, 50])
    parser.add_argument(
        "--continuous_relevance",
        action="store_true",
        help="If set, treat relevance as continuous and compute only P_k and success_k.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output files if they exist.")

    args = parser.parse_args()

    if not os.path.isdir(args.input_folder):
        raise NotADirectoryError(f"Input folder not found: {args.input_folder}")

    output_folder = args.output_folder or args.input_folder
    os.makedirs(output_folder, exist_ok=True)

    binary_relevance = not bool(args.continuous_relevance)
    k_values = sorted(set(int(x) for x in (args.k_values or [])))
    if not k_values:
        raise ValueError("k_values is empty.")

    metrics = define_retrieval_metrics(k_values, binary_relevance=binary_relevance)
    print(f"binary_relevance={binary_relevance}")
    print(f"metrics={metrics}")

    assessed_files = list_assessed_run_files(args.input_folder)
    if not assessed_files:
        raise FileNotFoundError(f"No *_assessed_run.csv files found in: {args.input_folder}")

    processed = 0
    for run_path in assessed_files:
        per_query_out, mean_out = out_paths_for_run(run_path, output_folder)

        for p in (per_query_out, mean_out):
            if os.path.exists(p) and not args.overwrite:
                raise FileExistsError(f"Output exists: {p} (use --overwrite)")

        run, qrel = load_assessed_run(run_path, binary_relevance=binary_relevance)

        # pytrec_eval richiede che le chiavi query coincidano tra run e qrel (come erag_mod)
        if set(run.keys()) != set(qrel.keys()):
            # allinea: se una query non ha doc in qrel (raro), metti dict vuoto
            all_qids = set(run.keys()) | set(qrel.keys())
            run = {qid: run.get(qid, {}) or {} for qid in all_qids}
            qrel = {qid: qrel.get(qid, {}) or {} for qid in all_qids}

        if binary_relevance:
            per_query = compute_metrics_binary(run, qrel, metrics)
        else:
            per_query = compute_metrics_continuous(run, qrel, metrics)

        means = aggregate_means(per_query, metrics)

        write_per_query_csv(per_query_out, per_query, metrics)
        write_means_csv(mean_out, means, metrics)

        processed += 1
        print(f"Processed: {run_path}")
        print(f"  per-query -> {per_query_out}")
        print(f"  means     -> {mean_out}")

    print(f"\nDone. Files processed: {processed}")


if __name__ == "__main__":
    main()