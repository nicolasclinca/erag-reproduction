"""
evaluate_answer_scores.py

Dato uno o più dataset KILT (.jsonl), calcola uno score per ogni coppia (query, answer)
come se "answer" fosse il documento da valutare con il retriever.

Output CSV con colonne:
  dataset, query_id, answer, score, retriever

Retrievers supportati:
  - bm25
      BM25 con statistiche del corpus intero da indice Lucene/PySerini:
      df/N/avgdl dall'indice; tf/dl calcolati sull'answer analizzato con lo stesso analyzer.

  - contriever
      Score = -L2^2 (higher=better), coerente con contriever_retriever.py quando return_cosine=False.
      Opzionalmente può simulare anche IVF+OPQ+PQ (se fornisci --contriever_faiss_index):
        - applica la chain di transform (es. OPQ) a query/doc
        - assegna doc alla lista (coarse centroid) più vicina
        - PQ-encoda e ricostruisce doc approssimato (by_residual supportato)
        - calcola -||q' - doc_hat||^2

  - tct
      inner product tra embedding query e embedding answer (TCT-ColBERT v2, PySerini-style).

  - bge
      inner product tra embedding normalizzati query/answer (BGE base v1.5).

  - dpr
      inner product tra DPR question encoder e DPR ctx encoder.

Esempio:
python evaluate_answer_scores.py \
  --datasets ../data/nq-dev-kilt.jsonl ../data/fever-dev-kilt.jsonl \
  --output ../out/answer_scores.csv \
  --retrievers bm25 contriever tct bge dpr \
  --bm25_index_dir ../indexes/bm25_index \
  --contriever_faiss_index ../indexes/contriever_index/ivfpq_opq_contriever.faiss \
  --overwrite
"""

from __future__ import annotations

import os
import gc
import json
import csv
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel  # type: ignore
from transformers import DPRQuestionEncoder, DPRContextEncoder  # type: ignore
from transformers import BertModel, BertTokenizer  # type: ignore

import faiss  # type: ignore
from pyserini.search.lucene import LuceneSearcher  # type: ignore
from pyserini.index.lucene import LuceneIndexReader as IndexReader  # type: ignore

from contriever_encoder import ContrieverEncoder
from query_encoders import MODEL_BGE, MODEL_TCT, MODEL_DPR_Q


MODEL_DPR_CTX = "facebook/dpr-ctx_encoder-multiset-base"


# -----------------------------
# Utils
# -----------------------------
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


def load_kilt_query_answers(path: str, max_examples: Optional[int] = None) -> DatasetData:
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
# BM25 scorer using index stats
# -----------------------------
class IndexBM25Scorer:
    """
    BM25 per "testo arbitrario" usando:
      - df, N, avgdl dal corpus indicizzato (IndexReader.stats + get_term_counts)
      - tf e dl calcolati sul testo dell'answer, tokenizzato con lo stesso analyzer Lucene

    Formula:
      idf(t) = log(1 + (N - df + 0.5)/(df + 0.5))
      score = sum_t idf(t) * ((tf*(k1+1)) / (tf + k1*(1-b + b*dl/avgdl))) * qtf
    """

    def __init__(self, index_dir: str, k1: float = 0.9, b: float = 0.4):
        if IndexReader is None:
            raise ImportError(
                "IndexReader not available in this pyserini version. "
                "Install/upgrade pyserini or disable bm25."
            )

        self.index_dir = index_dir
        self.k1 = float(k1)
        self.b = float(b)

        self.searcher = LuceneSearcher(index_dir)
        self.reader = IndexReader(index_dir)  # type: ignore[operator]

        st = self.reader.stats() or {}
        self.N = int(st.get("documents") or 0)
        total_terms = float(st.get("total_terms") or 0.0)
        self.avgdl = (total_terms / self.N) if (self.N > 0 and total_terms > 0) else 1.0

        self._df_cache: Dict[str, int] = {}

        if self.N <= 0:
            raise RuntimeError(f"IndexReader.stats() returned documents={self.N}. Index dir ok? {index_dir}")

    def analyze(self, text: str) -> List[str]:
        try:
            return self.searcher.analyze(text or "")
        except Exception:
            return []

    def _get_term_counts(self, term: str) -> int:
        """
        Ritorna df(term) dal corpus indicizzato (best effort across pyserini versions).
        """
        t = str(term)
        cached = self._df_cache.get(t)
        if cached is not None:
            return cached

        df = 0
        try:
            # spesso: (df, cf)
            df, _cf = self.reader.get_term_counts(t)
            df = int(df or 0)
        except TypeError:
            # alcune versioni accettano analyzer=
            try:
                df, _cf = self.reader.get_term_counts(t, analyzer=None)
                df = int(df or 0)
            except Exception:
                df = 0
        except Exception:
            df = 0

        self._df_cache[t] = int(df)
        return int(df)

    def _idf(self, df: int) -> float:
        if df <= 0:
            return 0.0
        return math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))

    def score_from_analyzed(self, qtf: Counter, doc_terms: List[str]) -> float:
        if not qtf:
            return 0.0
        tf = Counter(doc_terms)
        dl = float(len(doc_terms))
        if dl <= 0.0:
            return 0.0

        score = 0.0
        for t, qf in qtf.items():
            df = self._get_term_counts(t)
            if df <= 0:
                continue
            idf = self._idf(df)

            f = float(tf.get(t, 0))
            if f <= 0.0:
                continue

            denom = f + self.k1 * (1.0 - self.b + self.b * (dl / max(1e-9, self.avgdl)))
            score += idf * (f * (self.k1 + 1.0) / max(1e-9, denom)) * float(qf)

        return float(score)


# -----------------------------
# Contriever: -L2^2 scoring, optional IVF+OPQ+PQ simulation
# -----------------------------
def _faiss_downcast_index(index: faiss.Index) -> faiss.Index:
    """
    faiss.read_index spesso ritorna un tipo base (faiss.Index). downcast_index lo specializza.
    """
    try:
        if hasattr(faiss, "downcast_index"):
            return faiss.downcast_index(index)
    except Exception:
        pass
    return index


def _faiss_unwrap_idmap(index: faiss.Index) -> faiss.Index:
    core = _faiss_downcast_index(index)
    try:
        while isinstance(core, (faiss.IndexIDMap, faiss.IndexIDMap2)) and hasattr(core, "index"):
            core = _faiss_downcast_index(core.index)
    except Exception:
        pass
    return core


def _faiss_extract_ivf(index: faiss.Index) -> Optional[faiss.IndexIVF]:
    """
    Estrae la parte IVF se possibile (best effort).
    """
    idx = _faiss_downcast_index(index)
    try:
        if hasattr(faiss, "extract_index_ivf"):
            ivf = faiss.extract_index_ivf(idx)
            ivf = _faiss_downcast_index(ivf)
            return ivf  # type: ignore[return-value]
    except Exception:
        return None
    return None


class ContrieverIvfOpqPqScorer:
    """
    Dato un indice FAISS (tipicamente IndexPreTransform(OPQ) + IndexIVFPQ),
    calcola score ~ come farebbe FAISS (distanza L2^2) per un vettore doc arbitrario,
    simulando la quantizzazione:
      1) applica chain di VectorTransform (es. OPQ)
      2) assegna il doc al centroid più vicino del coarse quantizer
      3) (se by_residual) PQ-encoda il residual e ricostruisce doc_approx = centroid + residual_hat
         altrimenti PQ-encoda direttamente doc
      4) dist2 = ||q' - doc_approx||^2
      5) score = -dist2

    Nota: questa è una simulazione "standalone" (non usa nprobe né ricerca top-k).
    """

    def __init__(self, index_path: str):
        self.index_path = index_path

        idx = faiss.read_index(index_path)
        idx = _faiss_unwrap_idmap(idx)
        idx = _faiss_downcast_index(idx)

        self.pre: Optional[faiss.IndexPreTransform] = None
        core: faiss.Index = idx

        # Unwrap PreTransform (OPQ, ecc.)
        try:
            if isinstance(core, faiss.IndexPreTransform) and hasattr(core, "index"):
                self.pre = core
                core = _faiss_unwrap_idmap(core.index)
                core = _faiss_downcast_index(core)
        except Exception:
            self.pre = None

        # Estrai IVF e verifica che sia IVFPQ
        ivf = _faiss_extract_ivf(core)
        if ivf is None:
            core_dc = _faiss_downcast_index(core)
            # può già essere IVFPQ ma extract non disponibile
            if isinstance(core_dc, faiss.IndexIVFPQ):
                ivf = core_dc  # type: ignore[assignment]
            else:
                raise RuntimeError(
                    "ContrieverIvfOpqPqScorer expects an IVF index (IndexIVFPQ). "
                    f"Got: {type(core_dc)} from {index_path}"
                )

        ivf = _faiss_downcast_index(ivf)
        if not isinstance(ivf, faiss.IndexIVFPQ):
            raise RuntimeError(
                "ContrieverIvfOpqPqScorer expects IndexIVFPQ. "
                f"Got: {type(ivf)} from {index_path}"
            )

        self.ivfpq: faiss.IndexIVFPQ = ivf
        self.d = int(getattr(self.ivfpq, "d", 0) or 0)
        if self.d <= 0:
            raise RuntimeError("Invalid FAISS index dimension.")

        self.by_residual = bool(getattr(self.ivfpq, "by_residual", True))
        self.quantizer = _faiss_downcast_index(getattr(self.ivfpq, "quantizer", None))
        self.pq = getattr(self.ivfpq, "pq", None)
        if self.quantizer is None or self.pq is None:
            raise RuntimeError("IndexIVFPQ missing quantizer/pq attributes.")

        self._centroid_cache: Dict[int, np.ndarray] = {}

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Applica la chain di VectorTransform (OPQ etc.) se presente.
        x: float32 (B, d)
        """
        X = np.ascontiguousarray(x.astype(np.float32, copy=False))
        if self.pre is None:
            return X

        # Best effort: applica self.pre.chain.at(i).apply_py
        try:
            chain = getattr(self.pre, "chain", None)
            if chain is None:
                return X

            # VectorTransformVector: size() e at(i)
            size = None
            if hasattr(chain, "size"):
                size = int(chain.size())
            elif hasattr(chain, "__len__"):
                size = int(len(chain))  # type: ignore[arg-type]

            if size is None:
                return X

            for i in range(size):
                vt = chain.at(i)
                X = vt.apply_py(np.ascontiguousarray(X, dtype=np.float32))
            return np.ascontiguousarray(X.astype(np.float32, copy=False))
        except Exception:
            # fallback: nessun transform
            return X

    def _get_centroids(self, list_ids: np.ndarray) -> np.ndarray:
        """
        list_ids: int64 (B,)
        ritorna centroids: float32 (B, d)
        """
        B = int(list_ids.shape[0])
        C = np.empty((B, self.d), dtype=np.float32)

        uniq = list(dict.fromkeys([int(x) for x in list_ids.tolist() if int(x) >= 0]))
        missing = [lid for lid in uniq if lid not in self._centroid_cache]

        for lid in missing:
            try:
                c = self.quantizer.reconstruct(int(lid))
                c = np.asarray(c, dtype=np.float32).reshape(-1)
                if c.shape[0] != self.d:
                    c = np.zeros((self.d,), dtype=np.float32)
                self._centroid_cache[lid] = c
            except Exception:
                self._centroid_cache[lid] = np.zeros((self.d,), dtype=np.float32)

        z = np.zeros((self.d,), dtype=np.float32)
        for i in range(B):
            lid = int(list_ids[i])
            C[i] = self._centroid_cache.get(lid, z)

        return C

    def reconstruct_doc_approx(self, X_doc_t: np.ndarray) -> np.ndarray:
        """
        X_doc_t: float32 (B, d) nello spazio trasformato (OPQ)
        ritorna X_hat: float32 (B, d) approssimazione PQ (e residual) come nell'indice
        """
        Xd = np.ascontiguousarray(X_doc_t.astype(np.float32, copy=False))

        # assegna ogni doc alla lista più vicina (coarse quantizer)
        _D, I = self.quantizer.search(Xd, 1)
        list_ids = I.reshape(-1).astype(np.int64)

        if self.by_residual:
            centroids = self._get_centroids(list_ids)  # (B, d)
            residual = np.ascontiguousarray((Xd - centroids).astype(np.float32, copy=False))
            codes = self.pq.compute_codes(residual)
            residual_hat = self.pq.decode(codes)
            residual_hat = np.ascontiguousarray(residual_hat.astype(np.float32, copy=False))
            return centroids + residual_hat

        codes = self.pq.compute_codes(Xd)
        Xhat = self.pq.decode(codes)
        return np.ascontiguousarray(Xhat.astype(np.float32, copy=False))

    @staticmethod
    def neg_l2sq_scores(Q_t: np.ndarray, X_hat: np.ndarray) -> np.ndarray:
        """
        Q_t e X_hat: (B, d) float32
        ritorna score: -||Q - X||^2 (distanza L2 al quadrato)
        """
        diff = Q_t - X_hat
        dist2 = np.sum(diff * diff, axis=1).astype(np.float32)
        return (-dist2).astype(np.float32)


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
    def __init__(self, model_name: str, device: torch.device, max_length: int = 256, amp_dtype: torch.dtype = torch.float16):
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
            enc = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            if self.device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
                    out = self.model(**enc)
            else:
                out = self.model(**enc)
            chunks.append(out.pooler_output.float().cpu().numpy().astype(np.float32, copy=False))
        return np.vstack(chunks)


class _DprCtxEncoder:
    def __init__(self, model_name: str, device: torch.device, max_length: int = 256, amp_dtype: torch.dtype = torch.float16):
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
            enc = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            if self.device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
                    out = self.model(**enc)
            else:
                out = self.model(**enc)
            chunks.append(out.pooler_output.float().cpu().numpy().astype(np.float32, copy=False))
        return np.vstack(chunks)


class _TctColBertEncoder:
    """
    Replica della logica PySerini-style (come query_encoders.py), ma parametrizzabile:
    - role: "Q" oppure "D"
    - input: "[CLS] [<role>] " + text + "[MASK]" * 36
    - max_length=36, add_special_tokens=False
    - embedding: mean(last_hidden_state[:, 4:, :], dim=1)
    """

    def __init__(self, model_name: str, role: str, device: torch.device, amp_dtype: torch.dtype = torch.float16):
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

            enc = self.tokenizer(inp, return_tensors="pt", truncation=True, max_length=self.max_length, add_special_tokens=False)
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
    query_emb: np.ndarray,
    doc_encoder,
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

        A = doc_encoder.encode(a_batch, batch_size=batch_size)
        Q = query_emb[np.array(qidx_batch, dtype=np.int64)]

        scores = np.sum(Q * A, axis=1).astype(np.float32)

        for j in range(len(a_batch)):
            qidx = int(qidx_batch[j])
            writer.writerow([dataset.name, qids[qidx], a_batch[j], fmt.format(float(scores[j])), retriever_name])


def _write_bm25_scores_with_index_stats(
    writer: csv.writer,
    dataset: DatasetData,
    bm25: IndexBM25Scorer,
    score_precision: int,
) -> None:
    fmt = f"{{:.{int(score_precision)}f}}"
    qids = dataset.qids

    # pre-analizza le query una volta sola
    q_qtf: List[Counter] = []
    for q in dataset.queries:
        q_qtf.append(Counter(bm25.analyze(q)))

    for qidx, ans in zip(dataset.pair_qidx, dataset.answers):
        qidx_i = int(qidx)
        doc_terms = bm25.analyze(ans)
        s = bm25.score_from_analyzed(q_qtf[qidx_i], doc_terms)
        writer.writerow([dataset.name, qids[qidx_i], ans, fmt.format(float(s)), "bm25"])


def _write_contriever_neg_l2sq_scores(
    writer: csv.writer,
    dataset: DatasetData,
    contriever: ContrieverEncoder,
    answer_batch_size: int,
    query_batch_size: int,
    score_precision: int,
    faiss_scorer: Optional[ContrieverIvfOpqPqScorer] = None,
) -> None:
    """
    Score = -L2^2 (higher=better).

    Se faiss_scorer non è None:
      - applica transform (OPQ) alle query e ai doc
      - ricostruisce doc approssimato via IVFPQ (PQ su residual)
      - score = -||q' - doc_hat||^2
    """
    fmt = f"{{:.{int(score_precision)}f}}"
    qids = dataset.qids

    Q = contriever.encode(dataset.queries, batch_size=max(1, int(query_batch_size))).astype(np.float32, copy=False)
    Q_t = faiss_scorer.transform(Q) if faiss_scorer is not None else Q

    if faiss_scorer is not None and int(Q_t.shape[1]) != int(faiss_scorer.d):
        raise RuntimeError(f"Contriever FAISS dim mismatch: Q_t dim={Q_t.shape[1]} vs index_dim={faiss_scorer.d}")

    answers = dataset.answers
    pair_qidx = dataset.pair_qidx

    bs = max(1, int(answer_batch_size))
    for i in range(0, len(answers), bs):
        a_batch = answers[i : i + bs]
        qidx_batch = pair_qidx[i : i + bs]

        A = contriever.encode(a_batch, batch_size=bs).astype(np.float32, copy=False)
        Q_sel = Q_t[np.array(qidx_batch, dtype=np.int64)]

        if faiss_scorer is not None:
            A_t = faiss_scorer.transform(A)
            A_hat = faiss_scorer.reconstruct_doc_approx(A_t)
            scores = faiss_scorer.neg_l2sq_scores(Q_sel, A_hat)
        else:
            diff = Q_sel - A
            dist2 = np.sum(diff * diff, axis=1).astype(np.float32)
            scores = (-dist2).astype(np.float32)

        for j in range(len(a_batch)):
            qidx = int(qidx_batch[j])
            writer.writerow([dataset.name, qids[qidx], a_batch[j], fmt.format(float(scores[j])), "contriever"])


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

    # BM25 args (index stats)
    parser.add_argument("--bm25_index_dir", type=str, default=None, help="Directory indice BM25 (PySerini/Lucene)")
    parser.add_argument("--bm25_k1", type=float, default=0.9, help="BM25 k1.")
    parser.add_argument("--bm25_b", type=float, default=0.4, help="BM25 b.")

    # Contriever IVF+OPQ+PQ simulation (optional)
    parser.add_argument(
        "--contriever_faiss_index",
        type=str,
        default=None,
        help="Path indice FAISS (IVF+OPQ+PQ) per simulare anche la quantizzazione nello score contriever (-L2^2).",
    )

    args = parser.parse_args()

    out_path = args.output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if os.path.exists(out_path) and not args.overwrite:
        raise FileExistsError(f"Output exists: {out_path} (use --overwrite)")

    selected = [r.lower().strip() for r in (args.retrievers or [])]
    if "bm25" in selected and not args.bm25_index_dir:
        raise ValueError("--bm25_index_dir is required when 'bm25' is selected in --retrievers")

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

        # --- BM25 (index stats)
        if "bm25" in selected:
            print(f"[bm25] loading index stats from: {args.bm25_index_dir}")
            bm25 = IndexBM25Scorer(index_dir=args.bm25_index_dir, k1=args.bm25_k1, b=args.bm25_b)
            print(f"[bm25] N={bm25.N} avgdl={bm25.avgdl:.4f}")

            for ds in datasets:
                print(f"[bm25] scoring pairs for dataset={ds.name} ...")
                _write_bm25_scores_with_index_stats(writer, ds, bm25, score_precision=args.score_precision)

            del bm25
            gc.collect()

        # --- Contriever (-L2^2), optional IVF+OPQ+PQ simulation
        if "contriever" in selected:
            faiss_scorer: Optional[ContrieverIvfOpqPqScorer] = None
            if args.contriever_faiss_index:
                print(f"[contriever] loading FAISS index for IVF+OPQ+PQ simulation: {args.contriever_faiss_index}")
                faiss_scorer = ContrieverIvfOpqPqScorer(args.contriever_faiss_index)
                print(f"[contriever] FAISS index dim={faiss_scorer.d} by_residual={faiss_scorer.by_residual}")

            print("[contriever] loading encoder ...")
            contr = ContrieverEncoder(device=device)

            for ds in datasets:
                print(f"[contriever] scoring (-L2^2) dataset={ds.name} ...")
                _write_contriever_neg_l2sq_scores(
                    writer=writer,
                    dataset=ds,
                    contriever=contr,
                    answer_batch_size=args.batch_size,
                    query_batch_size=args.query_batch_size,
                    score_precision=args.score_precision,
                    faiss_scorer=faiss_scorer,
                )

            del contr, faiss_scorer
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
            bge = _HFClsEncoder(model_name=MODEL_BGE, device=device, max_length=args.max_length, normalize=True)

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
            dpr_q = _DprQEncoder(model_name=MODEL_DPR_Q, device=device, max_length=args.max_length)
            dpr_d = _DprCtxEncoder(model_name=MODEL_DPR_CTX, device=device, max_length=args.max_length)

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