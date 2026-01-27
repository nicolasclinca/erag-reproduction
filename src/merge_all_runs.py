"""
merge_all_runs.py

Unisce più retrieval run file (CSV) in un unico file <dataset>_all_runs.csv.

Input
-----
- Una cartella fornita via --input_folder (es. ../input_runs/nq)
- La cartella contiene SOLO file .csv di run (es. generati da build_qrels_og.py / build_retrieval_run.py)
  con almeno le colonne:
    query_id, doc_id
  (eventuali colonne extra come score, run_id vengono ignorate)

Output
------
- Un CSV con sole colonne: query_id, doc_id
- Nessuna riga duplicata (dedup su coppia (query_id, doc_id))

Naming
------
- Di default scrive in: <input_folder>/<dataset>_all_runs.csv
  dove <dataset> è il basename della cartella input (es. "nq").
- Puoi override con --output.

Uso CLI
-------
python merge_all_runs.py --input_folder ../input_runs/nq
python merge_all_runs.py --input_folder ../input_runs/nq --output ../runs/nq_all_runs.csv --overwrite
"""

from __future__ import annotations

import os
import csv
import argparse
from typing import List, Tuple, Set, Optional


def list_csv_files(input_folder: str) -> List[str]:
    files: List[str] = []
    for name in os.listdir(input_folder):
        p = os.path.join(input_folder, name)
        if os.path.isfile(p) and name.lower().endswith(".csv"):
            files.append(p)
    return sorted(files)


def iter_pairs_from_run_csv(path: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            qid = str(row.get("query_id", "")).strip()
            did = str(row.get("doc_id", "")).strip()
            if qid and did:
                out.append((qid, did))
    return out


def merge_runs(
    input_folder: str,
    output_path: str,
) -> List[Tuple[str, str]]:
    csv_files = list_csv_files(input_folder)
    if not csv_files:
        raise FileNotFoundError(f"No .csv files found in: {input_folder}")

    # evita che l'output (se già presente) venga ri-letto come input
    out_abs = os.path.abspath(output_path)
    csv_files = [p for p in csv_files if os.path.abspath(p) != out_abs]

    seen: Set[Tuple[str, str]] = set()
    merged: List[Tuple[str, str]] = []

    for p in csv_files:
        pairs = iter_pairs_from_run_csv(p)
        for qid, did in pairs:
            key = (qid, did)
            if key in seen:
                continue
            seen.add(key)
            merged.append(key)

    return merged


def write_pairs_csv(pairs: List[Tuple[str, str]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "doc_id"])
        for qid, did in pairs:
            w.writerow([qid, did])


def default_output_path(input_folder: str) -> str:
    dataset = os.path.basename(os.path.normpath(input_folder))
    return os.path.join(input_folder, f"{dataset}_all_runs.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge multiple retrieval run CSV files into a single <dataset>_all_runs.csv (query_id, doc_id) without duplicates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Cartella contenente i run file CSV da unire (es. ../input_runs/nq).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path di output CSV. Default: <input_folder>/<dataset>_all_runs.csv",
    )
    parser.add_argument("--overwrite", action="store_true", help="Sovrascrive l'output se esiste.")

    args = parser.parse_args()

    if not os.path.isdir(args.input_folder):
        raise NotADirectoryError(f"Input folder not found: {args.input_folder}")

    out_path = args.output or default_output_path(args.input_folder)

    if os.path.exists(out_path) and not args.overwrite:
        raise FileExistsError(f"Output exists: {out_path} (use --overwrite)")

    pairs = merge_runs(args.input_folder, out_path)
    write_pairs_csv(pairs, out_path)

    print(f"Input folder: {args.input_folder}")
    print(f"Output file:  {out_path}")
    print(f"Unique pairs: {len(pairs)}")


if __name__ == "__main__":
    main()