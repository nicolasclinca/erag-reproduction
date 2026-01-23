"""
build_qrels.py

Crea un file di qrels (query_id, doc_id, relevance) a partire da:
1) un run file prodotto da build_retrieval_run.py (csv/json/txt)
   contenente: query_id, doc_id, score, run_id
2) uno o più dataset KILT (.jsonl) contenenti gold evidence in output[*].provenance[*].wikipedia_id

Regola di relevance:
- doc_id nel run ha forma tipica: "<wikipedia_id>_<segment_id>" (es. "40885965_20")
- un segmento è rilevante (relevance=1) se la parte wikipedia_id del doc_id
  è presente tra i wikipedia_id gold del dataset per quella query_id.
- altrimenti relevance=0.

Output:
- CSV: header "query_id,doc_id,relevance"
- TXT: "query_id<TAB>doc_id<TAB>relevance" (separatore configurabile)
- JSON: array di oggetti [{"query_id":..,"doc_id":..,"relevance":..}, ...]

Uso:
python build_qrels.py \
  --datasets ../data/nq-train-kilt.jsonl \
  --run ../runs/nq_bm25.txt \
  --output ../qrels/nq_bm25_qrels.txt \
  --format txt \
  --overwrite
"""

from __future__ import annotations

import os
import json
import csv
import argparse
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple


# -----------------------------
# KILT gold loading (qid -> set(wikipedia_id))
# -----------------------------
def load_kilt_gold_wikipedia_ids_from_file(
    path: str,
    max_examples: Optional[int] = None,
    qid_prefix: Optional[str] = None,
) -> Dict[str, Set[str]]:
    """
    Ritorna: {query_id: set_of_gold_wikipedia_ids}

    query_id:
      - record["id"] se presente, altrimenti indice riga
      - se qid_prefix è dato: f"{qid_prefix}:{qid}"
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
    Carica qid->gold_wikipedia_ids da una lista di dataset.
    Se più dataset e prefix_with_dataset=True, prefissa qid con basename per evitare collisioni.
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

        # merge (se collisione, unisci set)
        for qid, wids in part.items():
            if qid not in merged:
                merged[qid] = set()
            merged[qid].update(wids)

    return merged


# -----------------------------
# Run parsing (yield query_id, doc_id)
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


def iter_run_pairs_csv(run_path: str) -> Iterator[Tuple[str, str]]:
    with open(run_path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            qid = str(row.get("query_id", "")).strip()
            docid = str(row.get("doc_id", "")).strip()
            if qid and docid:
                yield qid, docid


def iter_run_pairs_txt(run_path: str, sep: Optional[str] = None) -> Iterator[Tuple[str, str]]:
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
            if len(parts) < 2:
                continue

            qid = parts[0].strip()
            docid = parts[1].strip()
            if qid and docid:
                yield qid, docid


def iter_run_pairs_json(run_path: str) -> Iterator[Tuple[str, str]]:
    """
    Supporta:
    - JSON array (formato scritto da build_retrieval_run.py)
    - JSONL (una riga = un oggetto), best effort
    """
    with open(run_path, "r", encoding="utf-8") as f:
        # peek first non-whitespace char
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
                if qid and docid:
                    yield qid, docid
        else:
            # jsonl fallback
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
                if qid and docid:
                    yield qid, docid


def iter_run_pairs(run_path: str, run_format: Optional[str] = None, run_txt_sep: Optional[str] = None) -> Iterator[Tuple[str, str]]:
    fmt = (run_format or infer_run_format(run_path)).lower()
    if fmt == "csv":
        yield from iter_run_pairs_csv(run_path)
    elif fmt == "json":
        yield from iter_run_pairs_json(run_path)
    elif fmt == "txt":
        yield from iter_run_pairs_txt(run_path, sep=run_txt_sep)
    else:
        raise ValueError(f"Unknown run format: {fmt}")


# -----------------------------
# Qrels building
# -----------------------------
def wikipedia_id_from_doc_id(doc_id: str) -> str:
    """
    Estrae la parte wikipedia_id da doc_id.
    Esempio: "40885965_20" -> "40885965"
    Se non c'è "_", ritorna doc_id intero.
    """
    doc_id = (doc_id or "").strip()
    if not doc_id:
        return ""
    return doc_id.split("_", 1)[0]


def iter_qrels_rows(
    run_pairs: Iterable[Tuple[str, str]],
    gold_wiki_ids_by_qid: Dict[str, Set[str]],
    *,
    missing_qid_policy: str = "zero",  # "zero" | "skip" | "error"
) -> Iterator[Tuple[str, str, int]]:
    """
    Yields (query_id, doc_id, relevance) in streaming.
    """
    for qid, docid in run_pairs:
        gold_set = gold_wiki_ids_by_qid.get(qid, None)

        if gold_set is None:
            if missing_qid_policy == "skip":
                continue
            if missing_qid_policy == "error":
                raise KeyError(f"query_id not found in datasets gold map: {qid}")
            # default: "zero"
            yield qid, docid, 0
            continue

        wid = wikipedia_id_from_doc_id(docid)
        rel = 1 if (wid and wid in gold_set) else 0
        yield qid, docid, rel


# -----------------------------
# Output writing (streaming)
# -----------------------------
def write_qrels_csv(rows: Iterable[Tuple[str, str, int]], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "doc_id", "relevance"])
        for qid, docid, rel in rows:
            w.writerow([qid, docid, int(rel)])


def write_qrels_txt(rows: Iterable[Tuple[str, str, int]], out_path: str, sep: str = "\t") -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for qid, docid, rel in rows:
            f.write(f"{qid}{sep}{docid}{sep}{int(rel)}\n")


def write_qrels_json(rows: Iterable[Tuple[str, str, int]], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("[\n")
        first = True
        for qid, docid, rel in rows:
            obj = {"query_id": qid, "doc_id": docid, "relevance": int(rel)}
            if not first:
                f.write(",\n")
            f.write(json.dumps(obj, ensure_ascii=False))
            first = False
        f.write("\n]\n")


# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build qrels (query_id, doc_id, relevance) from a retrieval run file and KILT dataset(s).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--datasets", type=str, nargs="+", required=True, help="Path(s) to KILT .jsonl dataset(s).")
    parser.add_argument("--max_examples", type=int, default=None, help="Limit examples per dataset (debug).")
    parser.add_argument(
        "--prefix_with_dataset",
        action="store_true",
        help="If multiple datasets, prefix query_id with dataset basename to avoid collisions (must match run).",
    )

    parser.add_argument("--run", type=str, required=True, help="Run file path produced by build_retrieval_run.py.")
    parser.add_argument("--run_format", choices=["txt", "csv", "json"], default=None, help="Override run format.")
    parser.add_argument(
        "--run_txt_sep",
        type=str,
        default=None,
        help="Separator for txt run parsing. If omitted, tries to auto-detect (tab, then comma, then whitespace).",
    )

    parser.add_argument("--output", type=str, required=True, help="Output qrels path.")
    parser.add_argument("--format", choices=["txt", "csv", "json"], default="txt", help="Output format.")
    parser.add_argument("--txt_sep", type=str, default="\t", help="Separator for txt output.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if exists.")
    parser.add_argument(
        "--missing_qid_policy",
        choices=["zero", "skip", "error"],
        default="zero",
        help="What to do if a query_id from run is not present in datasets.",
    )

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if os.path.exists(args.output) and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output} (use --overwrite)")

    # 1) Load gold wiki ids by qid
    gold_map = load_kilt_gold_wikipedia_ids(
        datasets=args.datasets,
        max_examples=args.max_examples,
        prefix_with_dataset=args.prefix_with_dataset,
    )
    if not gold_map:
        raise RuntimeError("No gold data loaded from dataset(s).")

    # 2) Stream run pairs -> stream qrels rows -> write
    pairs = iter_run_pairs(args.run, run_format=args.run_format, run_txt_sep=args.run_txt_sep)
    rows = iter_qrels_rows(pairs, gold_map, missing_qid_policy=args.missing_qid_policy)

    if args.format == "csv":
        write_qrels_csv(rows, args.output)
    elif args.format == "json":
        write_qrels_json(rows, args.output)
    elif args.format == "txt":
        write_qrels_txt(rows, args.output, sep=args.txt_sep)
    else:
        raise ValueError(f"Unknown output format: {args.format}")

    print(f"Saved qrels -> {args.output}")


if __name__ == "__main__":
    main()