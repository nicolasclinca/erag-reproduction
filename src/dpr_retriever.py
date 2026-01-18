"""
dpr_retriever.py
Dense retrieval DPR su indici FAISS sharded (PySerini-style) salvati come:

indexes/wiki-dpr-118m/
  part_0/
    index
    docid
  part_1/
    index
    docid
  ...

Questo retriever:
- embedd-a le query con il DPR question encoder (HuggingFace)
- cerca in tutti gli shard FAISS e fa il merge per ottenere il top-k globale
- converte docid -> "contents" usando un docstore Lucene (es. un indice BM25 con storeRaw)

Dettagli implementativi:
- gli indici FAISS vengono PRELOADATI (aperti) una sola volta all'avvio (con mmap opzionale)
- durante lo scan shard-by-shard NON converte rid -> docid
- accumula solo (score, shard_idx, rid) e converte in docid solo dopo il merge top-k
- carica in RAM il file docid solo per gli shard che compaiono nel top-k (cache LRU dedicata)

Uso CLI (docstore richiesto):
Il docstore deve essere già costruito con PySerini (serve storeRaw):
python -m pyserini.index.lucene \
  -collection JsonCollection \
  -input data/collection \
  -index indexes/wiki_docstore_lucene \
  -generator DefaultLuceneDocumentGenerator \
  -threads 8 \
  -storeRaw

Singola query:
python dpr_retriever.py \
  --dpr_index_root_dir ../indexes/wiki-dpr-118m \
  --docstore_index_dir ../indexes/wiki_docstore_lucene \
  --query "When did Apollo 11 land?" \
  --k 5 --threads 8 --max_loaded_docid_shards 16

Note:
- se l'indice BM25 è disponibile e contiene storeRaw, può essere usato come docstore_index_dir:
  es. --docstore_index_dir ../indexes/bm25_index.
"""

from __future__ import annotations

import os
import re
import json
import heapq
import argparse
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict, defaultdict

import numpy as np
import torch

try:
    import faiss  # type: ignore
except Exception as e:
    raise RuntimeError("faiss non disponibile. Installa faiss-cpu o faiss-gpu.") from e

from transformers import AutoTokenizer, DPRQuestionEncoder
from pyserini.search.lucene import LuceneSearcher


# ============================
# Default
# ============================
QUERY_ENCODER_NAME = "facebook/dpr-question_encoder-multiset-base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_QUERY_LENGTH = 256


# ============================
# Helpers: shard listing/sorting
# ============================
_PART_RE = re.compile(r"^part_(\d+)$")


def _list_part_dirs(root: str) -> List[str]:
    if not os.path.isdir(root):
        raise ValueError(f"Index root dir not found: {root}")

    parts: List[Tuple[int, str]] = []
    for name in os.listdir(root):
        m = _PART_RE.match(name)
        if m:
            parts.append((int(m.group(1)), os.path.join(root, name)))

    parts.sort(key=lambda x: x[0])
    return [p for _, p in parts]


# ============================
# Query encoder (HF DPR question encoder)
# ============================
class DPRQueryEncoderHF:
    """
    Wrapper minimale per DPR question encoder.
    Ritorna embedding float32 (B, D).
    """

    def __init__(
        self,
        model_name: str = QUERY_ENCODER_NAME,
        device= DEVICE,
        max_length: int = MAX_QUERY_LENGTH,
    ):
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = DPRQuestionEncoder.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Performance knobs (safe)
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        # dim
        with torch.inference_mode():
            dummy = self.tokenizer(["hello"], return_tensors="pt", padding=True, truncation=True)
            dummy = {k: v.to(self.device) for k, v in dummy.items()}
            out = self.model(**dummy)
            self.D = int(out.pooler_output.shape[-1])

    @torch.inference_mode()
    def encode(self, queries: List[str], batch_size: int = 32) -> np.ndarray:
        if not queries:
            return np.empty((0, self.D), dtype=np.float32)

        vecs: List[np.ndarray] = []
        for i in range(0, len(queries), batch_size):
            batch = queries[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc)
            x = out.pooler_output  # (B, D)
            vecs.append(x.detach().float().cpu().numpy().astype(np.float32, copy=False))

        return np.vstack(vecs)


# ============================
# LRU cache helpers (for docid shards)
# ============================
class _LRUCache:
    def __init__(self, max_loaded: int = 1):
        self.max_loaded = max(1, int(max_loaded))
        self._cache: "OrderedDict[str, object]" = OrderedDict()

    def get(self, key: str):
        v = self._cache.get(key)
        if v is not None:
            self._cache.move_to_end(key)
        return v

    def put(self, key: str, value) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_loaded:
            _, ev = self._cache.popitem(last=False)
            del ev

    def clear(self) -> None:
        self._cache.clear()


# ============================
# DPR Sharded Searcher
# ============================
class DPRShardedSearcher:
    """
    Oggetto "searcher" da passare a dpr_retrieve/dpr_batch_retrieve.

    - index_root_dir: directory contenente part_0..part_N
    - docstore_index_dir: path a indice Lucene con storeRaw (docid -> raw json -> contents)
    - preload indici: self.indexes è allineato a self.part_dirs (shard_idx)
    """

    def __init__(
        self,
        index_root_dir: str,
        docstore_index_dir: str,
        query_encoder_name: str = None,
        device=None,
        max_query_length: int = None,
        max_loaded_docid_shards: int = 16,
        faiss_threads: Optional[int] = None,
        mmap: bool = True,
    ):
        self.index_root_dir = index_root_dir
        self.part_dirs = _list_part_dirs(index_root_dir)
        if not self.part_dirs:
            raise ValueError(f"Nessuno shard part_* trovato in: {index_root_dir}")

        # Docstore Lucene
        self.docstore = LuceneSearcher(docstore_index_dir)

        # Query encoder
        self.encoder = DPRQueryEncoderHF(
            model_name=query_encoder_name,
            device=device,
            max_length=max_query_length,
        )

        self.mmap = bool(mmap)

        # Cache docid: carica docid solo per shard presenti nel top-k
        self.docid_cache = _LRUCache(max_loaded=max_loaded_docid_shards)

        if faiss_threads is not None:
            try:
                faiss.omp_set_num_threads(int(faiss_threads))
            except Exception:
                pass

        # PRELOAD: apre tutti gli indici FAISS una sola volta
        self.indexes: List["faiss.Index"] = []
        for shard_dir in self.part_dirs:
            self.indexes.append(self._load_index(shard_dir))

    # ---------- Index loading ----------
    def _load_index(self, shard_dir: str) -> "faiss.Index":
        index_path = os.path.join(shard_dir, "index")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Missing FAISS index file: {index_path}")

        idx = None
        if self.mmap and hasattr(faiss, "IO_FLAG_MMAP"):
            try:
                idx = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
            except Exception:
                idx = None
        if idx is None:
            idx = faiss.read_index(index_path)

        return idx

    # ---------- DocID loading ----------
    def _load_docids(self, shard_dir: str) -> List[str]:
        docid_path = os.path.join(shard_dir, "docid")
        if not os.path.exists(docid_path):
            raise FileNotFoundError(f"Missing docid file: {docid_path}")
        with open(docid_path, "r", encoding="utf-8") as f:
            docids = [line.rstrip("\n") for line in f]
        if docids and docids[-1] == "":
            docids = docids[:-1]
        return docids

    def _get_docids(self, shard_dir: str) -> List[str]:
        cached = self.docid_cache.get(shard_dir)
        if cached is not None:
            return cached  # type: ignore[return-value]
        docids = self._load_docids(shard_dir)
        self.docid_cache.put(shard_dir, docids)
        return docids

    # ---------- Retrieval core ----------
    def search_docids_batch(
        self,
        queries: List[str],
        k: int = 50,
        encode_batch_size: int = 32,
        faiss_threads: Optional[int] = None,
    ) -> List[List[Tuple[str, float]]]:
        """
        Ritorna, per ogni query, una lista di (docid, score) top-k globali.

        - Mantiene per ogni query un min-heap (dimensione k) di tuple (score, shard_idx, rid)
        - score viene normalizzato in "higher is better" (se metrica distanza: score=-distance)
        - dopo il merge risolve rid->docid solo per gli shard effettivamente usati
        """
        if not queries:
            return []
        if k <= 0:
            return [[] for _ in queries]

        if faiss_threads is not None:
            try:
                faiss.omp_set_num_threads(int(faiss_threads))
            except Exception:
                pass

        Q = self.encoder.encode(queries, batch_size=encode_batch_size)
        if Q.ndim != 2:
            raise RuntimeError("Query embeddings shape non valida.")
        qn, qd = Q.shape
        Qc = np.ascontiguousarray(Q, dtype=np.float32)

        # heaps[i] è un min-heap di (score, shard_idx, rid) per la query i
        heaps: List[List[Tuple[float, int, int]]] = [[] for _ in range(qn)]

        for shard_idx, shard_dir in enumerate(self.part_dirs):
            shard_index = self.indexes[shard_idx]

            try:
                d_index = int(shard_index.d)
                if d_index != qd:
                    raise RuntimeError(
                        f"Dim mismatch: query_dim={qd} vs index_dim={d_index} in {shard_dir}"
                    )
            except Exception:
                pass

            scores, ids = shard_index.search(Qc, int(k))  # shapes: (qn, k)

            # Normalizzo score per merge: higher is better
            metric_type = getattr(shard_index, "metric_type", faiss.METRIC_INNER_PRODUCT)
            try:
                metric_type = int(metric_type)
            except Exception:
                metric_type = faiss.METRIC_INNER_PRODUCT
            is_distance_metric = metric_type != faiss.METRIC_INNER_PRODUCT

            for i in range(qn):
                h = heaps[i]
                for j in range(k):
                    rid = int(ids[i, j])
                    if rid < 0:
                        continue

                    s = float(scores[i, j])
                    if is_distance_metric:
                        s = -s

                    item = (s, shard_idx, rid)
                    if len(h) < k:
                        heapq.heappush(h, item)
                    else:
                        # se migliore del peggiore nel heap, sostituisci
                        if s > h[0][0]:
                            heapq.heapreplace(h, item)

        # Ordina i top-k per query in ordine decrescente e identifica shard necessari
        top_per_query: List[List[Tuple[float, int, int]]] = []
        needed_shards: Dict[int, List[int]] = defaultdict(list)

        for i in range(qn):
            items_sorted = sorted(heaps[i], key=lambda x: x[0], reverse=True)
            top_per_query.append(items_sorted)
            for _s, sh, rid in items_sorted:
                needed_shards[sh].append(rid)

        # Carica docids solo per shard effettivamente usati
        docids_by_shard: Dict[int, List[str]] = {}
        for sh in needed_shards.keys():
            shard_dir = self.part_dirs[sh]
            docids_by_shard[sh] = self._get_docids(shard_dir)

        # Ricostruisce output: (docid, score)
        out: List[List[Tuple[str, float]]] = []
        for i in range(qn):
            res_i: List[Tuple[str, float]] = []
            for s, sh, rid in top_per_query[i]:
                docids = docids_by_shard.get(sh, [])
                if 0 <= rid < len(docids):
                    res_i.append((docids[rid], s))
                else:
                    res_i.append(("", s))
            out.append(res_i)

        return out

    def docids_to_contents(self, docids: List[str]) -> List[str]:
        """
        Converte docid -> contents usando il docstore Lucene (storeRaw).
        """
        out: List[str] = []
        for did in docids:
            if not did:
                out.append("")
                continue
            try:
                doc = self.docstore.doc(did)
                if doc is None:
                    out.append("")
                    continue
                js = json.loads(doc.raw())
                out.append(js.get("contents") or js.get("text") or "")
            except Exception:
                out.append("")
        return out


# ============================
# API
# ============================
def dpr_retrieve(query: str, searcher: DPRShardedSearcher, k: int = 50) -> List[str]:
    """
    DPR retrieval per singola query.
    Ritorna: lista di 'contents' (stringhe), top-k.
    """
    if searcher is None:
        raise ValueError("A DPRShardedSearcher instance must be provided.")

    scored = searcher.search_docids_batch([query], k=k)
    docids = [d for d, _s in scored[0]] if scored else []
    return searcher.docids_to_contents(docids)


def dpr_batch_retrieve(
    queries: List[str],
    searcher: DPRShardedSearcher,
    k: int = 50,
    batch_size: int = 256,
    threads: int = 8,
    encode_batch_size: int = 32,
) -> Dict[str, List[str]]:
    """
    DPR batch retrieval.
    Ritorna: {query: [doc1, doc2, ..., dock]} dove i doc sono i 'contents'.

    threads: usato per faiss.omp_set_num_threads (CPU).
    """
    if searcher is None:
        raise ValueError("A DPRShardedSearcher instance must be provided.")
    if not queries:
        return {}

    results: Dict[str, List[str]] = {}

    for i in range(0, len(queries), batch_size):
        batch_q = queries[i : i + batch_size]

        scored_lists = searcher.search_docids_batch(
            batch_q,
            k=k,
            encode_batch_size=encode_batch_size,
            faiss_threads=threads,
        )

        # 1) estrai docids top-k per query
        per_query_docids: List[List[str]] = []
        all_docids: List[str] = []
        for scored in scored_lists:
            docids = [d for d, _s in scored if d]
            per_query_docids.append(docids)
            all_docids.extend(docids)

        # 2) fetch contents in modo deduplicato (meno chiamate Lucene)
        unique_docids = list(dict.fromkeys(all_docids))  # preserve order
        did2cont = dict(zip(unique_docids, searcher.docids_to_contents(unique_docids)))

        # 3) ricostruisci risultati
        for q, docids in zip(batch_q, per_query_docids):
            results[q] = [did2cont.get(d, "") for d in docids]

    return results


# ============================
# CLI
# ============================
def main():
    parser = argparse.ArgumentParser(description="DPR retrieval (FAISS sharded) + Lucene docstore for contents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dpr_index_root_dir", required=True, help="Dir con part_0..part_N (FAISS)")
    parser.add_argument("--docstore_index_dir", required=True, 
                        help="Indice Lucene con storeRaw per docid->contents (es. bm25_index)")
    parser.add_argument("--query", required=True, help="Singola query")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--threads", type=int, default=8, help="FAISS omp threads (CPU)")
    parser.add_argument("--max_loaded_docid_shards", type=int, default=16,
        help="Docid cache size (LRU). Opzione A: shard docid caricati on-demand.")
    parser.add_argument("--mmap", dest="mmap", action="store_true", help="Usa FAISS IO_FLAG_MMAP (default)")
    parser.add_argument("--no_mmap", dest="mmap", action="store_false", help="Disabilita mmap")
    parser.set_defaults(mmap=True)

    args = parser.parse_args()

    searcher = DPRShardedSearcher(
        index_root_dir=args.dpr_index_root_dir,
        docstore_index_dir=args.docstore_index_dir,
        max_loaded_docid_shards=args.max_loaded_docid_shards,
        faiss_threads=args.threads,
        mmap=args.mmap,
    )
    docs = dpr_retrieve(args.query, searcher=searcher, k=args.k)
    print(json.dumps({args.query: docs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()