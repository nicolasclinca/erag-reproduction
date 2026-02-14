"""
evaluate_correlations.py

Computes correlations between retrieval metrics (per-query) and downstream score (per-query, per-k),
starting from files already computed by:

- evaluate_all_retrieval_metrics.py
    <run>_assessed_run_retrieval_metrics.csv
    columns: query_id, <metric_1>, <metric_2>, ...

- evaluate_all_downstream.py
    <run>_downstream.csv
    columns: query_id, score, k
    (one row per query for each k)

For each pair (retrieval_metrics, downstream) it writes a CSV:
  <base_downstream>_correlations.csv

Output CSV fields:
  metric, k_downstream, spearman_corr, spearman_p, kendall_corr, kendall_p

Important notes
---------------
- File matching:
    <base>_assessed_run_retrieval_metrics.csv  <->  <base>_downstream.csv
- For each metric (e.g., ndcg_cut_10) it uses downstream with k=10.
  If k is not present in downstream, it prints a WARNING and uses k=50 if available.
  If k=50 is not available either, it skips the metric.
- Query alignment: uses the intersection of query_id between retrieval_metrics and downstream(k),
  sorted by query_id, to guarantee correct pairing.

CLI usage
-------
python evaluate_correlations.py \
  --input_folder ../input_runs/nq \
  --overwrite
"""

from __future__ import annotations

import os
import csv
import re
import argparse
from typing import Dict, List, Tuple, Optional

import scipy.stats as stats


# -----------------------------
# Metrics (as in evaluate_all_retrieval_metrics.py)
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
def list_retrieval_metrics_files(input_folder: str) -> List[str]:
    files: List[str] = []
    for name in os.listdir(input_folder):
        p = os.path.join(input_folder, name)
        if not os.path.isfile(p):
            continue
        low = name.lower()
        if not low.endswith(".csv"):
            continue
        if low.endswith("_mean_metrics.csv"):
            continue
        if low.endswith("_correlations.csv"):
            continue
        if low.endswith("_downstream.csv"):
            continue
        if low.endswith("_retrieval_metrics.csv"):
            files.append(p)
    return sorted(files)


def list_downstream_files(input_folder: str) -> List[str]:
    files: List[str] = []
    for name in os.listdir(input_folder):
        p = os.path.join(input_folder, name)
        if not os.path.isfile(p):
            continue
        low = name.lower()
        if not low.endswith(".csv"):
            continue
        if low.endswith("_mean_downstream.csv"):
            continue
        if low.endswith("_correlations.csv"):
            continue
        if low.endswith("_retrieval_metrics.csv"):
            continue
        if low.endswith("_downstream.csv"):
            files.append(p)
    return sorted(files)


# -----------------------------
# Naming helpers
# -----------------------------
def base_from_retrieval_metrics_path(path: str) -> str:
    """
    Example:
      nq_bge_better_assessed_run_retrieval_metrics.csv -> nq_bge_better
    """
    base = os.path.splitext(os.path.basename(path))[0]

    suffix = "_retrieval_metrics"
    if base.endswith(suffix):
        base = base[: -len(suffix)]

    suffix2 = "_assessed_run"
    if base.endswith(suffix2):
        base = base[: -len(suffix2)]

    return base


def base_from_downstream_path(path: str) -> str:
    """
    Example:
      nq_bge_better_downstream.csv -> nq_bge_better
    """
    base = os.path.splitext(os.path.basename(path))[0]
    suffix = "_downstream"
    if base.endswith(suffix):
        base = base[: -len(suffix)]
    return base


def correlations_out_path(downstream_path: str, output_folder: str) -> str:
    base = base_from_downstream_path(downstream_path)
    return os.path.join(output_folder, f"{base}_correlations.csv")


# -----------------------------
# CSV parsing helpers
# -----------------------------
def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def load_retrieval_metrics_csv(path: str) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    """
    Returns:
      metrics_in_file: list in header order (excluding query_id)
      per_qid: {qid: {metric: value}}
    """
    per_qid: Dict[str, Dict[str, float]] = {}

    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise RuntimeError(f"CSV has no header: {path}")

        if "query_id" not in set(r.fieldnames):
            raise KeyError(f"Missing column 'query_id' in: {path}")

        metrics_in_file = [c for c in r.fieldnames if c != "query_id"]

        for row in r:
            qid = str(row.get("query_id", "")).strip()
            if not qid:
                continue

            d: Dict[str, float] = {}
            for m in metrics_in_file:
                d[m] = _safe_float(row.get(m, 0.0), default=0.0)
            per_qid[qid] = d

    return metrics_in_file, per_qid


def load_downstream_csv(path: str) -> Dict[int, Dict[str, float]]:
    """
    Returns:
      by_k: {k: {qid: score}}
    """
    by_k: Dict[int, Dict[str, float]] = {}

    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise RuntimeError(f"CSV has no header: {path}")

        needed = {"query_id", "score", "k"}
        missing = [c for c in sorted(needed) if c not in set(r.fieldnames)]
        if missing:
            raise KeyError(f"Missing columns {missing} in downstream file: {path}")

        for row in r:
            qid = str(row.get("query_id", "")).strip()
            if not qid:
                continue

            try:
                k = int(row.get("k", 0))
            except Exception:
                continue

            score = _safe_float(row.get("score", 0.0), default=0.0)
            by_k.setdefault(int(k), {})[qid] = float(score)

    return by_k


# -----------------------------
# Correlation helpers
# -----------------------------
def parse_k_from_metric_name(metric_name: str) -> Optional[int]:
    """
    Extracts the trailing k from names like:
      P_10, success_30, ndcg_cut_50, map_cut_10, recip_rank_cut_30
    """
    m = re.search(r"(\d+)$", metric_name or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def select_downstream_k(
    metric_name: str,
    downstream_by_k: Dict[int, Dict[str, float]],
    *,
    fallback_k: int = 50,
) -> Optional[int]:
    """
    Rule:
    - use the metric k if present in downstream
    - otherwise WARNING and use fallback_k if present
    - otherwise None (skip)
    """
    metric_k = parse_k_from_metric_name(metric_name)

    if metric_k is not None and metric_k in downstream_by_k:
        return metric_k

    if fallback_k in downstream_by_k:
        if metric_k is None:
            print(f"WARNING: cannot parse k from metric '{metric_name}'. Using downstream k={fallback_k}.")
        else:
            print(
                f"WARNING: downstream k={metric_k} not available for metric '{metric_name}'. "
                f"Using downstream k={fallback_k}."
            )
        return int(fallback_k)

    if metric_k is None:
        print(
            f"WARNING: cannot parse k from metric '{metric_name}', and downstream k={fallback_k} not available. "
            "Skipping metric."
        )
    else:
        print(
            f"WARNING: downstream k={metric_k} not available for metric '{metric_name}', and downstream k={fallback_k} "
            "not available. Skipping metric."
        )
    return None


def compute_correlations_for_pair(
    retrieval_metrics_path: str,
    downstream_path: str,
    out_path: str,
    *,
    overwrite: bool,
    fallback_k: int = 50,
    restrict_metrics: Optional[List[str]] = None,
) -> None:
    if os.path.exists(out_path) and not overwrite:
        raise FileExistsError(f"Output exists: {out_path} (use --overwrite)")

    metrics_in_file, per_qid_metrics = load_retrieval_metrics_csv(retrieval_metrics_path)
    downstream_by_k = load_downstream_csv(downstream_path)

    if restrict_metrics is not None:
        metrics = [m for m in metrics_in_file if m in set(restrict_metrics)]
    else:
        metrics = list(metrics_in_file)

    if not metrics:
        print(f"WARNING: no metrics found to evaluate for: {retrieval_metrics_path}")
        return

    if not downstream_by_k:
        print(f"WARNING: downstream file has no usable rows: {downstream_path}")
        return

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp_path = out_path + ".tmp"

    with open(tmp_path, "w", encoding="utf-8", newline="") as f_out:
        w = csv.writer(f_out)
        w.writerow(["metric", "k_downstream", "spearman_corr", "spearman_p", "kendall_corr", "kendall_p"])

        for metric_name in metrics:
            k_used = select_downstream_k(metric_name, downstream_by_k, fallback_k=fallback_k)
            if k_used is None:
                continue

            downstream_scores = downstream_by_k.get(k_used, {}) or {}
            qids_common = sorted(set(per_qid_metrics.keys()) & set(downstream_scores.keys()))

            if not qids_common:
                print(
                    f"WARNING: no common query_id between retrieval_metrics and downstream(k={k_used}) "
                    f"for metric '{metric_name}'. Skipping metric."
                )
                continue

            aligned_x = [float((per_qid_metrics.get(qid, {}) or {}).get(metric_name, 0.0)) for qid in qids_common]
            aligned_y = [float(downstream_scores.get(qid, 0.0)) for qid in qids_common]

            spearman_corr, spearman_p = stats.spearmanr(aligned_x, aligned_y)
            kendall_corr, kendall_p = stats.kendalltau(aligned_x, aligned_y)

            w.writerow([metric_name, int(k_used), spearman_corr, spearman_p, kendall_corr, kendall_p])

    os.replace(tmp_path, out_path)
    print(f"Saved correlations -> {out_path}")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Spearman/Kendall correlations between retrieval metrics and downstream scores (per-query).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing *_retrieval_metrics.csv and *_downstream.csv")
    parser.add_argument(
        "--output_folder",
        type=str,
        default=None,
        help="Where to write *_correlations.csv (default: same as input_folder).",
    )
    parser.add_argument(
        "--fallback_k",
        type=int,
        default=50,
        help="Fallback downstream k to use when metric-specific k is missing (as requested, default=50).",
    )

    # optional: allows “defining the metrics” like in the other scripts
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=None,
        help="If set, restrict correlations to metrics implied by these k values (like evaluate_all_retrieval_metrics.py).",
    )
    parser.add_argument(
        "--continuous_relevance",
        action="store_true",
        help="If set with --k_values, restrict to continuous-safe metrics (P_k, success_k).",
    )

    parser.add_argument("--overwrite", action="store_true", help="Overwrite correlation files if they exist.")

    args = parser.parse_args()

    if not os.path.isdir(args.input_folder):
        raise NotADirectoryError(f"Input folder not found: {args.input_folder}")

    output_folder = args.output_folder or args.input_folder
    os.makedirs(output_folder, exist_ok=True)

    retrieval_files = list_retrieval_metrics_files(args.input_folder)
    downstream_files = list_downstream_files(args.input_folder)

    if not retrieval_files:
        raise FileNotFoundError(f"No *_retrieval_metrics.csv files found in: {args.input_folder}")
    if not downstream_files:
        raise FileNotFoundError(f"No *_downstream.csv files found in: {args.input_folder}")

    downstream_map: Dict[str, str] = {}
    for p in downstream_files:
        b = base_from_downstream_path(p)
        if b in downstream_map:
            print(f"WARNING: duplicate downstream base '{b}'. Keeping first: {downstream_map[b]} | ignoring: {p}")
            continue
        downstream_map[b] = p

    restrict_metrics: Optional[List[str]] = None
    if args.k_values is not None:
        k_values = sorted(set(int(x) for x in (args.k_values or [])))
        if not k_values:
            raise ValueError("k_values is empty.")
        binary_relevance = not bool(args.continuous_relevance)
        restrict_metrics = define_retrieval_metrics(k_values, binary_relevance=binary_relevance)
        print(f"Restricting metrics to: {restrict_metrics}")

    processed = 0
    skipped = 0

    for rm_path in retrieval_files:
        base = base_from_retrieval_metrics_path(rm_path)
        ds_path = downstream_map.get(base, None)
        if not ds_path:
            print(f"WARNING: cannot find downstream file for retrieval_metrics='{rm_path}' (base='{base}'). Skipped.")
            skipped += 1
            continue

        out_path = correlations_out_path(ds_path, output_folder)

        print("\n----------------------------------------")
        print(f"Retrieval metrics: {rm_path}")
        print(f"Downstream:        {ds_path}")

        compute_correlations_for_pair(
            retrieval_metrics_path=rm_path,
            downstream_path=ds_path,
            out_path=out_path,
            overwrite=args.overwrite,
            fallback_k=int(args.fallback_k),
            restrict_metrics=restrict_metrics,
        )
        processed += 1

    print("\nDone.")
    print(f"Pairs processed: {processed}")
    print(f"Pairs skipped:   {skipped}")


if __name__ == "__main__":
    main()