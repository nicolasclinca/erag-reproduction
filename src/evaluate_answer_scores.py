"""
evaluate_answer_scores.py

Dato uno o più dataset KILT (.jsonl), calcola uno score per ogni coppia (query, answer)
come se "answer" fosse il documento da valutare con il retriever.

Output CSV con colonne:
  dataset, query_id, answer, score, retriever

Retrievers supportati:
  - bm25        (BM25 locale calcolato sugli answer come collezione; tokenizzazione semplice)
  - contriever  (cosine similarity tra embedding query e embedding answer)
  - tct         (inner product tra embedding query e embedding answer con TCT-ColBERT v2)
  - bge         (inner product tra embedding normalizzati query/answer con BGE base v1.5)
  - dpr         (inner product tra DPR question encoder e DPR ctx encoder)

Uso CLI (esempio):
python evaluate_answer_scores.py \
  --datasets ../data/nq-train-kilt.jsonl ../data/fever-train-kilt.jsonl \
  --output ../out/answer_scores.csv \
  --overwrite \
  --batch_size 64

Nota:
- Per BM25, lo score dipende dalle statistiche della "collezione". Qui, per semplicità,
  la collezione è l'insieme degli answer del dataset (non l'intera Wikipedia).
"""

from __future__ import annotations

import os
import re
import gc
import json
import csv
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel  # type: ignore
from transformers import DPRQuestionEncoder, DPRContextEncoder  # type: ignore
from transformers import BertModel, BertTokenizer  # type: ignore

from contriever_encoder import ContrieverEncoder
from query_encoders import MODEL_BGE, MODEL_TCT, MODEL_DPR_Q


MODEL_DPR_CTX = "facebook/dpr-ctx_encoder-multiset-base"


# -----------------------------
# Utils
# -----------------------------
_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def _as_device(device: Optional[str]) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _safe_empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _dataset_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


# -----------------------------
# KILT loading (query, answers)
# -----------------------------
@dataclass
class DatasetData:
    name: str
    path: str
    qids: List[str]
    queries: List[str]
    pair_qidx: List[int]     # len = #pairs
    answers: List[str]       # len = #pairs


def load_kilt_query_answers(
    path: str,
    max_examples: Optional[int] = None,
) -> DatasetData:
    """
    Carica un dataset KILT jsonl e produce:
      - qids[i], queries[i] per record
      - pair_qidx[j], answers[j] per ogni answer nel record
    """
    name = _dataset_name(path)
    qids: List[str] = []
    queries: List[str] = []
    pair_qidx: List[int] = []
    answers: List[str] = []

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

            q = (rec.get("input") or "").strip()
            if not q:
                continue

            qid = str(rec.get("id", i))
            qidx = len(qids)
            qids.append(qid)
            queries.append(q)

            outs = rec.get("output") or []
            if not isinstance(outs, list):
                continue

            for out in outs:
                if not isinstance(out, dict):
                    continue
                ans = out.get("answer")
                if ans is None:
                    continue
                ans = str(ans).strip()
                if not ans:
                    continue
                pair_qidx.append(qidx)
                answers.append(ans)

    return DatasetData(
        name=name,
        path=path,
        qids=qids,
        queries=queries,
        pair_qidx=pair_qidx,
        answers=answers,
    )


# -----------------------------
# BM25 (local) scorer
# -----------------------------
class LocalBM25:
    """
    BM25 "locale" sugli answer del dataset come collezione.

    Tokenizzazione: regex \w+ lowercase.
    IDF: log(1 + (N - df + 0.5)/(df + 0.5))
    """

    def __init__(self, k1: float = 0.9, b: float = 0.4):
        self.k1 = float(k1)
        self.b = float(b)
        self.N: int = 0
        self.avgdl: float = 0.0
        self.idf: Dict[str, float] = {}

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return _WORD_RE.findall((text or "").lower())

    def build(self, documents: List[str]) -> None:
        df: Dict[str, int] = {}
        total_len = 0
        N = 0

        for doc in documents:
            toks = self.tokenize(doc)
            if not toks:
                # consideriamo comunque il doc, per N/avgdl (dl=0)
                N += 1
                continue

            N += 1
            total_len += len(toks)
            for t in set(toks):
                df[t] = df.get(t, 0) + 1

        self.N = int(N)
        self.avgdl = (float(total_len) / float(N)) if N > 0 else 0.0

        idf: Dict[str, float] = {}
        for t, f in df.items():
            # BM25 standard-ish
            idf[t] = math.log(1.0 + (self.N - f + 0.5) / (f + 0.5))
        self.idf = idf

    def score(self, query: str, document: str) -> float:
        if self.N <= 0:
            return 0.0

        q_toks = self.tokenize(query)
        if not q_toks:
            return 0.0
        d_toks = self.tokenize(document)

        qtf = Counter(q_toks)
        tf = Counter(d_toks)
        dl = float(len(d_toks))

        score = 0.0
        for term, qf in qtf.items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            f = float(tf.get(term, 0))
            if f <= 0.0:
                continue

            denom = f + self.k1 * (1.0 - self.b + self.b * (dl / max(1e-9, self.avgdl)))
            score += idf * (f * (self.k1 + 1.0) / max(1e-9, denom)) * float(qf)

        return float(score)


# -----------------------------
# Dense scorers (query/doc encoders)
# -----------------------------
class _HFClsEncoder:
    """
    Encoder HF generico su AutoModel:
    - embedding = last_hidden_state[:,0,:]
    - optional normalize
    """

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        max_length: int = 256,
        normalize: bool = False,
        amp_dtype: torch.dtype = torch.float16,
    ):
        self.device = device
        self.max_length = int(max_length)
        self.normalize = bool(normalize)
        self.amp_dtype = amp_dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.D = int(getattr(getattr(self.model, "config", None), "hidden_size", 0) or 0)
        if self.D <= 0:
            self.D = 768

        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    @torch.inference_mode()
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, self.D), dtype=np.float32)

        bs = max(1, int(batch_size))
        chunks: List[np.ndarray] = []

        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            if self.device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
                    out = self.model(**enc)
            else:
                out = self.model(**enc)

            x = out.last_hidden_state[:, 0, :].float()
            if self.normalize:
                x = F.normalize(x, p=2, dim=1)

            chunks.append(x.cpu().numpy().astype(np.float32, copy=False))

        return np.vstack(chunks)


class _DprQEncoder:
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        max_length: int = 256,
        amp_dtype: torch.dtype = torch.float16,
    ):
        self.device = device
        self.max_length = int(max_length)
        self.amp_dtype = amp_dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = DPRQuestionEncoder.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.D = int(self.model.config.hidden_size)

        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    @torch.inference_mode()
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, self.D), dtype=np.float32)

        bs = max(1, int(batch_size))
        chunks: List[np.ndarray] = []

        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            if self.device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
                    out = self.model(**enc)
            else:
                out = self.model(**enc)

            x = out.pooler_output.float()
            chunks.append(x.cpu().numpy().astype(np.float32, copy=False))

        return np.vstack(chunks)


class _DprCtxEncoder:
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        max_length: int = 256,
        amp_dtype: torch.dtype = torch.float16,
    ):
        self.device = device
        self.max_length = int(max_length)
        self.amp_dtype = amp_dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = DPRContextEncoder.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.D = int(self.model.config.hidden_size)

        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    @torch.inference_mode()
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, self.D), dtype=np.float32)

        bs = max(1, int(batch_size))
        chunks: List[np.ndarray] = []

        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            if self.device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
                    out = self.model(**enc)
            else:
                out = self.model(**enc)

            x = out.pooler_output.float()
            chunks.append(x.cpu().numpy().astype(np.float32, copy=False))

        return np.vstack(chunks)


class _TctColBertEncoder:
    """
    Replica della logica PySerini-style (come query_encoders.py), ma parametrizzabile:
    - role: "Q" oppure "D"
    - input: "[CLS] [<role>] " + text + "[MASK]" * 36
    - max_length=36, add_special_tokens=False
    - embedding: mean(last_hidden_state[:, 4:, :], dim=1)
    """

    def __init__(
        self,
        model_name: str,
        role: str,
        device: torch.device,
        amp_dtype: torch.dtype = torch.float16,
    ):
        self.device = device
        self.role = str(role).strip().upper()
        if self.role not in ("Q", "D"):
            raise ValueError("role must be 'Q' or 'D'")
        self.amp_dtype = amp_dtype

        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)

        self.D = int(self.model.config.hidden_size)
        self.max_length = 36
        self._mask_suffix = "[MASK]" * self.max_length

        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    @torch.inference_mode()
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, self.D), dtype=np.float32)

        bs = max(1, int(batch_size))
        chunks: List[np.ndarray] = []

        prefix = f"[CLS] [{self.role}] "

        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            inp = [(prefix + t + self._mask_suffix) for t in batch]

            enc = self.tokenizer(
                inp,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            if self.device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
                    out = self.model(**enc)
            else:
                out = self.model(**enc)

            x = out.last_hidden_state[:, 4:, :].float().mean(dim=1)
            chunks.append(x.cpu().numpy().astype(np.float32, copy=False))

        return np.vstack(chunks)


# -----------------------------
# Scoring loops
# -----------------------------
def _write_dense_scores(
    writer: csv.writer,
    dataset: DatasetData,
    retriever_name: str,
    query_emb: np.ndarray,          # (num_queries, D)
    doc_encoder,                    # .encode(List[str], batch_size) -> np.ndarray (B, D)
    batch_size: int,
    score_precision: int,
) -> None:
    fmt = f"{{:.{int(score_precision)}f}}"

    answers = dataset.answers
    pair_qidx = dataset.pair_qidx
    qids = dataset.qids

    for i in range(0, len(answers), batch_size):
        a_batch = answers[i : i + batch_size]
        qidx_batch = pair_qidx[i : i + batch_size]

        A = doc_encoder.encode(a_batch, batch_size=batch_size)  # (B, D)
        Q = query_emb[np.array(qidx_batch, dtype=np.int64)]      # (B, D)

        scores = np.sum(Q * A, axis=1).astype(np.float32)

        for j in range(len(a_batch)):
            qidx = int(qidx_batch[j])
            writer.writerow([
                dataset.name,
                qids[qidx],
                a_batch[j],
                fmt.format(float(scores[j])),
                retriever_name,
            ])


def _write_bm25_scores(
    writer: csv.writer,
    dataset: DatasetData,
    bm25: LocalBM25,
    score_precision: int,
) -> None:
    fmt = f"{{:.{int(score_precision)}f}}"
    qids = dataset.qids
    queries = dataset.queries

    for qidx, ans in zip(dataset.pair_qidx, dataset.answers):
        q = queries[int(qidx)]
        s = bm25.score(q, ans)
        writer.writerow([
            dataset.name,
            qids[int(qidx)],
            ans,
            fmt.format(float(s)),
            "bm25",
        ])


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute retriever score for each (query, answer) pair, output as CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--datasets", type=str, nargs="+", required=True, help="Path(s) to KILT .jsonl dataset(s).")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if exists.")
    parser.add_argument("--max_examples", type=int, default=None, help="Limit examples per dataset (debug).")

    parser.add_argument(
        "--retrievers",
        type=str,
        nargs="+",
        default=["bm25", "contriever", "tct", "bge", "dpr"],
        choices=["bm25", "contriever", "tct", "bge", "dpr"],
        help="Which retrievers to evaluate.",
    )

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for encoding answers.")
    parser.add_argument("--query_batch_size", type=int, default=64, help="Batch size for encoding queries.")
    parser.add_argument("--score_precision", type=int, default=6, help="Decimals in score column.")

    parser.add_argument("--device", type=str, default=None, help="Torch device (e.g., 'cuda', 'cpu', 'cuda:0').")
    parser.add_argument("--max_length", type=int, default=256, help="Max length for HF encoders (BGE/DPR).")

    # BM25 params
    parser.add_argument("--bm25_k1", type=float, default=0.9, help="BM25 k1 (local).")
    parser.add_argument("--bm25_b", type=float, default=0.4, help="BM25 b (local).")

    args = parser.parse_args()

    out_path = args.output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if os.path.exists(out_path) and not args.overwrite:
        raise FileExistsError(f"Output exists: {out_path} (use --overwrite)")

    # Load datasets once
    datasets: List[DatasetData] = []
    for p in args.datasets:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Dataset not found: {p}")
        ds = load_kilt_query_answers(p, max_examples=args.max_examples)
        print(f"[{ds.name}] loaded queries={len(ds.queries)} pairs(query-answer)={len(ds.answers)}")
        datasets.append(ds)

    if not datasets:
        raise RuntimeError("No datasets loaded.")
    if all(len(ds.answers) == 0 for ds in datasets):
        raise RuntimeError("No (query, answer) pairs found in all datasets (empty outputs?).")

    device = _as_device(args.device)
    print(f"Device: {device}")

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "query_id", "answer", "score", "retriever"])

        selected = [r.lower().strip() for r in (args.retrievers or [])]

        # --- BM25 (no models)
        if "bm25" in selected:
            for ds in datasets:
                print(f"[bm25] building stats on answers for dataset={ds.name} ...")
                bm25 = LocalBM25(k1=args.bm25_k1, b=args.bm25_b)
                bm25.build(ds.answers)
                print(f"[bm25] scoring pairs for dataset={ds.name} ...")
                _write_bm25_scores(writer, ds, bm25, score_precision=args.score_precision)

        # --- Contriever
        if "contriever" in selected:
            print("[contriever] loading model ...")
            contr = ContrieverEncoder(device=device)
            for ds in datasets:
                print(f"[contriever] encoding queries dataset={ds.name} ...")
                Q = contr.encode(ds.queries, batch_size=args.query_batch_size)  # normalized
                print(f"[contriever] scoring pairs dataset={ds.name} ...")
                # use same encoder for docs (answers)
                _write_dense_scores(
                    writer=writer,
                    dataset=ds,
                    retriever_name="contriever",
                    query_emb=Q,
                    doc_encoder=contr,
                    batch_size=args.batch_size,
                    score_precision=args.score_precision,
                )
            del contr
            gc.collect()
            _safe_empty_cuda_cache()

        # --- TCT
        if "tct" in selected:
            print("[tct] loading model ...")
            tct_q = _TctColBertEncoder(model_name=MODEL_TCT, role="Q", device=device)
            tct_d = _TctColBertEncoder(model_name=MODEL_TCT, role="D", device=device)
            for ds in datasets:
                print(f"[tct] encoding queries dataset={ds.name} ...")
                Q = tct_q.encode(ds.queries, batch_size=args.query_batch_size)
                print(f"[tct] scoring pairs dataset={ds.name} ...")
                _write_dense_scores(
                    writer=writer,
                    dataset=ds,
                    retriever_name="tct",
                    query_emb=Q,
                    doc_encoder=tct_d,
                    batch_size=args.batch_size,
                    score_precision=args.score_precision,
                )
            del tct_q, tct_d
            gc.collect()
            _safe_empty_cuda_cache()

        # --- BGE
        if "bge" in selected:
            print("[bge] loading model ...")
            bge = _HFClsEncoder(
                model_name=MODEL_BGE,
                device=device,
                max_length=args.max_length,
                normalize=True,
            )
            for ds in datasets:
                print(f"[bge] encoding queries dataset={ds.name} ...")
                Q = bge.encode(ds.queries, batch_size=args.query_batch_size)
                print(f"[bge] scoring pairs dataset={ds.name} ...")
                _write_dense_scores(
                    writer=writer,
                    dataset=ds,
                    retriever_name="bge",
                    query_emb=Q,
                    doc_encoder=bge,
                    batch_size=args.batch_size,
                    score_precision=args.score_precision,
                )
            del bge
            gc.collect()
            _safe_empty_cuda_cache()

        # --- DPR
        if "dpr" in selected:
            print("[dpr] loading models ...")
            dpr_q = _DprQEncoder(
                model_name=MODEL_DPR_Q,
                device=device,
                max_length=args.max_length,
            )
            dpr_d = _DprCtxEncoder(
                model_name=MODEL_DPR_CTX,
                device=device,
                max_length=args.max_length,
            )
            for ds in datasets:
                print(f"[dpr] encoding queries dataset={ds.name} ...")
                Q = dpr_q.encode(ds.queries, batch_size=args.query_batch_size)
                print(f"[dpr] scoring pairs dataset={ds.name} ...")
                _write_dense_scores(
                    writer=writer,
                    dataset=ds,
                    retriever_name="dpr",
                    query_emb=Q,
                    doc_encoder=dpr_d,
                    batch_size=args.batch_size,
                    score_precision=args.score_precision,
                )
            del dpr_q, dpr_d
            gc.collect()
            _safe_empty_cuda_cache()

    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()