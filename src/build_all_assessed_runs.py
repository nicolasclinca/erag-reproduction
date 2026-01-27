"""
build_all_assessed_runs.py

Dato:
- un file <dataset>_all_qrels.csv con colonne: query_id, doc_id, relevance
- una cartella --input_folder contenente run file CSV con colonne: query_id, doc_id, score, run_id

Crea, per ogni file CSV in input_folder, un file:
  <input_filename>_assessed_run.csv

con colonne:
  query_id, doc_id, score, run_id, relevance

Dove:
- query_id, doc_id, score, run_id sono presi dal file di input
- relevance è presa da <dataset>_all_qrels.csv
- se la coppia (query_id, doc_id) non esiste in all_qrels => relevance = 0

Note:
- I file di input sono assunti essere CSV.
- Il file all_qrels può contenere relevance int o float.
- I file che terminano già con _assessed_run.csv vengono ignorati.
- Il file all_qrels viene ignorato se si trova dentro input_folder.

Uso CLI
-------
python build_all_assessed_runs.py \
  --all_qrels ../input_runs/nq/nq_all_qrels.csv \
  --input_folder ../input_runs/nq \
  --overwrite
"""

from __future__ import annotations

import os
import csv
import argparse
from typing import Dict, Tuple, List, Any


def list_csv_files(folder: str) -> List[str]:
    files: List[str] = []
    for name in os.listdir(folder):
        p = os.path.join(folder, name)
        if os.path.isfile(p) and name.lower().endswith(".csv"):
            files.append(p)
    return sorted(files)


def load_qrels_map(all_qrels_path: str) -> Dict[Tuple[str, str], Any]:
    """
    Carica <dataset>_all_qrels.csv e ritorna un mapping:
      (query_id, doc_id) -> relevance

    Se ci sono duplicati, mantiene la relevance massima.
    """
    if not os.path.exists(all_qrels_path):
        raise FileNotFoundError(f"all_qrels not found: {all_qrels_path}")

    rel_map: Dict[Tuple[str, str], Any] = {}

    with open(all_qrels_path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            qid = str(row.get("query_id", "")).strip()
            did = str(row.get("doc_id", "")).strip()
            if not qid or not did:
                continue

            raw_rel = row.get("relevance", 0)
            try:
                rel_val: Any = float(raw_rel)
                # se è intero, lo lasciamo come int (stile codebase)
                if abs(rel_val - round(rel_val)) < 1e-12:
                    rel_val = int(round(rel_val))
            except Exception:
                rel_val = 0

            key = (qid, did)
            if key in rel_map:
                # tieni max (robusto a duplicati)
                try:
                    prev = float(rel_map[key])
                    cur = float(rel_val)
                    rel_map[key] = rel_val if cur > prev else rel_map[key]
                except Exception:
                    rel_map[key] = rel_val
            else:
                rel_map[key] = rel_val

    return rel_map


def assessed_output_path(input_csv_path: str, output_folder: str) -> str:
    base = os.path.splitext(os.path.basename(input_csv_path))[0]
    return os.path.join(output_folder, f"{base}_assessed_run.csv")


def assess_one_run(
    run_path: str,
    out_path: str,
    rel_map: Dict[Tuple[str, str], Any],
) -> Tuple[int, int]:
    """
    Scrive out_path aggiungendo la colonna relevance.
    Returns:
      (n_rows_written, n_rows_with_rel_gt_zero)
    """
    with open(run_path, "r", encoding="utf-8", newline="") as f_in, open(
        out_path, "w", encoding="utf-8", newline=""
    ) as f_out:
        r = csv.DictReader(f_in)
        if not r.fieldnames:
            raise RuntimeError(f"CSV has no header: {run_path}")

        needed = {"query_id", "doc_id", "score", "run_id"}
        missing = [c for c in sorted(needed) if c not in set(r.fieldnames)]
        if missing:
            raise KeyError(f"Missing columns {missing} in run file: {run_path}")

        w = csv.writer(f_out)
        w.writerow(["query_id", "doc_id", "score", "run_id", "relevance"])

        n = 0
        n_pos = 0

        for row in r:
            qid = str(row.get("query_id", "")).strip()
            did = str(row.get("doc_id", "")).strip()
            score = row.get("score", "")
            run_id = row.get("run_id", "")

            rel = rel_map.get((qid, did), 0)

            try:
                rel_f = float(rel)
            except Exception:
                rel_f = 0.0

            if rel_f > 0.0:
                n_pos += 1

            w.writerow([qid, did, score, run_id, rel])
            n += 1

    return n, n_pos


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add eRAG relevance from <dataset>_all_qrels.csv to each run CSV in a folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--all_qrels", type=str, required=True, help="Path to <dataset>_all_qrels.csv")
    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing run CSV files to assess.")
    parser.add_argument(
        "--output_folder",
        type=str,
        default=None,
        help="Folder to write assessed runs (default: same as input_folder).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite assessed run files if they exist.")

    args = parser.parse_args()

    if not os.path.isdir(args.input_folder):
        raise NotADirectoryError(f"Input folder not found: {args.input_folder}")

    output_folder = args.output_folder or args.input_folder
    os.makedirs(output_folder, exist_ok=True)

    rel_map = load_qrels_map(args.all_qrels)
    print(f"Loaded all_qrels: {args.all_qrels} -> {len(rel_map)} pairs")

    all_files = list_csv_files(args.input_folder)
    if not all_files:
        raise FileNotFoundError(f"No .csv files found in: {args.input_folder}")

    all_qrels_abs = os.path.abspath(args.all_qrels)

    processed_files = 0
    total_rows = 0
    total_pos = 0

    for run_path in all_files:
        name = os.path.basename(run_path)

        # skip qrels file if it's inside the folder
        if os.path.abspath(run_path) == all_qrels_abs:
            continue

        # skip the all_runs file
        if name.lower().endswith("_all_runs.csv"):
            continue

        # skip already-assessed outputs
        if name.lower().endswith("_assessed_run.csv"):
            continue

        out_path = assessed_output_path(run_path, output_folder)
        if os.path.exists(out_path) and not args.overwrite:
            raise FileExistsError(f"Output exists: {out_path} (use --overwrite)")

        n, n_pos = assess_one_run(run_path, out_path, rel_map)
        processed_files += 1
        total_rows += n
        total_pos += n_pos

        print(f"Saved assessed run -> {out_path} | rows={n}, relevance>0={n_pos}")

    print("\nDone.")
    print(f"Processed files: {processed_files}")
    print(f"Total rows:      {total_rows}")
    print(f"Total rel>0:     {total_pos}")


if __name__ == "__main__":
    main()