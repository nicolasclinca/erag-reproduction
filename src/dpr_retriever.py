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
- embedd-a le query con il DPR question encoder
- cerca in tutti gli shard e merge-a i risultati per ottenere il top-k globale
- converte docid -> "contents" usando un docstore Lucene (es. l'indice BM25 con storeRaw)

Uso CLI:

Il docstore deve essere già costruito con PySerini:
python -m pyserini.index.lucene \
  -collection JsonCollection \
  -input data/collection \
  -index indexes/wiki_docstore_lucene \
  -generator DefaultLuceneDocumentGenerator \
  -threads 8 \
  -storeRaw

Singola query:
python dpr_retriever.py --dpr_index_root_dir ../indexes/wiki-dpr-118m \
    --docstore_index_dir indexes/wiki_docstore_lucene \
    --query "When did Apollo 11 land?" \
    --k 5 --threads 8 --max_loaded_shards 8 --mmap

Note: se l'indice bm25 è disponibile, può essere usato come docstore_index_dir
(es. --docstore_index_dir ../indexes/bm25_index).
"""

from __future__ import annotations

import os
import re
import json
import heapq
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

import numpy as np
import torch

try:
    import faiss  # type: ignore
except Exception as e:
    raise RuntimeError(
        "faiss non disponibile. Installa faiss-cpu o faiss-gpu."
    ) from e

from transformers import AutoTokenizer, DPRQuestionEncoder
from pyserini.search.lucene import LuceneSearcher


# ============================
# Helpers: shard listing/sorting
# ============================
_PART_RE = re.compile(r"^part_(\d+)$")


def _list_part_dirs(root: str) -> List[str]:
    if not os.path.isdir(root):
        raise ValueError(f"Index root dir not found: {root}")

    parts = []
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

    def __init__(self, model_name: str, device: Optional[torch.device] = None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
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
    def encode(self, queries: List[str], batch_size: int = 32, max_length: int = 256) -> np.ndarray:
        if not queries:
            return np.empty((0, self.D), dtype=np.float32)

        vecs = []
        for i in range(0, len(queries), batch_size):
            batch = queries[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc)
            # DPR usa pooler_output come embedding query
            x = out.pooler_output  # (B, D)
            vecs.append(x.detach().float().cpu().numpy().astype(np.float32, copy=False))

        return np.vstack(vecs)


# ============================
# FAISS shard cache (LRU)
# ============================
@dataclass
class _ShardData:
    index: "faiss.Index"
    docids: List[str]


class _LRUShardCache:
    """
    Cache LRU per shard. Utile perché:
    - non puoi caricare 120 shard contemporaneamente (anche con PQ restano pesanti)
    - però vuoi evitare di ricaricare continuamente lo stesso shard in alcuni workflow

    Nota: se usi IO_FLAG_MMAP, l'indice è mmappato e l'overhead RAM è minore, ma
    tenere 120 mmap aperti può comunque essere indesiderabile.
    """

    def __init__(self, max_loaded: int = 1):
        self.max_loaded = max(1, int(max_loaded))
        self._cache: "OrderedDict[str, _ShardData]" = OrderedDict()

    def get(self, shard_dir: str) -> Optional[_ShardData]:
        v = self._cache.get(shard_dir)
        if v is not None:
            self._cache.move_to_end(shard_dir)
        return v

    def put(self, shard_dir: str, data: _ShardData) -> None:
        self._cache[shard_dir] = data
        self._cache.move_to_end(shard_dir)
        while len(self._cache) > self.max_loaded:
            _, ev = self._cache.popitem(last=False)
            # release references
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
    - docstore: LuceneSearcher usato per docid -> raw json -> contents
    """

    def __init__(
        self,
        index_root_dir: str,
        docstore: LuceneSearcher,
        query_encoder_name: str = "facebook/dpr-question_encoder-multiset-base",
        device: Optional[torch.device] = None,
        max_loaded_shards: int = 1,
        faiss_threads: Optional[int] = None,
        mmap: bool = True,
    ):
        if docstore is None:
            raise ValueError("docstore (LuceneSearcher) è obbligatorio per restituire i contents.")

        self.index_root_dir = index_root_dir
        self.part_dirs = _list_part_dirs(index_root_dir)
        if not self.part_dirs:
            raise ValueError(f"Nessuno shard part_* trovato in: {index_root_dir}")

        self.docstore = docstore
        self.encoder = DPRQueryEncoderHF(query_encoder_name, device=device)

        self.cache = _LRUShardCache(max_loaded=max_loaded_shards)
        self.mmap = bool(mmap)

        if faiss_threads is not None:
            try:
                faiss.omp_set_num_threads(int(faiss_threads))
            except Exception:
                pass

    def _load_shard(self, shard_dir: str) -> _ShardData:
        index_path = os.path.join(shard_dir, "index")
        docid_path = os.path.join(shard_dir, "docid")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Missing FAISS index file: {index_path}")
        if not os.path.exists(docid_path):
            raise FileNotFoundError(f"Missing docid file: {docid_path}")

        # Load FAISS index (try mmap if available)
        idx = None
        if self.mmap and hasattr(faiss, "IO_FLAG_MMAP"):
            try:
                idx = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
            except Exception:
                idx = None
        if idx is None:
            idx = faiss.read_index(index_path)

        # Load docids list
        with open(docid_path, "r", encoding="utf-8") as f:
            docids = [line.strip() for line in f if line.strip()]

        return _ShardData(index=idx, docids=docids)

    def _get_shard(self, shard_dir: str) -> _ShardData:
        cached = self.cache.get(shard_dir)
        if cached is not None:
            return cached
        data = self._load_shard(shard_dir)
        self.cache.put(shard_dir, data)
        return data


    def search_docids_batch(
        self,
        queries: List[str],
        k: int = 50,
        encode_batch_size: int = 32,
        max_query_len: int = 256,
        faiss_threads: Optional[int] = None,
    ) -> List[List[Tuple[str, float]]]:
        """
        Ritorna, per ogni query, una lista di (docid, score) top-k globali.

        Per fare un merge corretto tra shard usando heapq.nlargest, convertiamo tutte le
        metriche di distanza in uno score "higher-is-better" usando score = -distance.
        """
        if not queries:
            return []

        if faiss_threads is not None:
            try:
                faiss.omp_set_num_threads(int(faiss_threads))
            except Exception:
                pass

        Q = self.encoder.encode(queries, batch_size=encode_batch_size, max_length=max_query_len)
        if Q.ndim != 2:
            raise RuntimeError("Query embeddings shape non valida.")
        qn, qd = Q.shape

        # candidates[i] = lista di (merge_score, docid) per la query i
        candidates: List[List[Tuple[float, str]]] = [[] for _ in range(qn)]

        Qc = np.ascontiguousarray(Q, dtype=np.float32)

        for shard_dir in self.part_dirs:
            shard = self._get_shard(shard_dir)
            index = shard.index
            docids = shard.docids

            # Sanity check dimension
            try:
                d_index = int(index.d)
                if d_index != qd:
                    raise RuntimeError(
                        f"Dim mismatch: query_dim={qd} vs index_dim={d_index} in {shard_dir}"
                    )
            except Exception:
                pass

            scores, ids = index.search(Qc, int(k))  # shapes: (qn, k)

            # Determina se lo score è una distanza (lower is better) o similarità (higher is better)
            # In FAISS: solo METRIC_INNER_PRODUCT è "higher-is-better", le altre sono distanze.
            metric_type = int(getattr(index, "metric_type", faiss.METRIC_INNER_PRODUCT))
            is_distance_metric = (metric_type != faiss.METRIC_INNER_PRODUCT)

            for i in range(qn):
                for j in range(k):
                    rid = int(ids[i, j])
                    if rid < 0:
                        continue
                    if rid >= len(docids):
                        continue

                    s = float(scores[i, j])
                    if is_distance_metric:
                        s = -s  # converto distanza in score "higher-is-better" per merge corretto

                    candidates[i].append((s, docids[rid]))

        # merge top-k per query (ora sempre higher-is-better)
        out: List[List[Tuple[str, float]]] = []
        for i in range(qn):
            top = heapq.nlargest(k, candidates[i], key=lambda x: x[0])
            out.append([(docid, score) for score, docid in top])

        return out

    def docids_to_contents(self, docids: List[str]) -> List[str]:
        """
        Converte docid -> contents usando il docstore Lucene (storeRaw).
        """
        out: List[str] = []
        for did in docids:
            try:
                doc = self.docstore.doc(did)
                if doc is None:
                    out.append("")
                    continue
                raw = doc.raw()
                js = json.loads(raw)
                out.append(js.get("contents", ""))
            except Exception:
                out.append("")
        return out


# ============================
# API
# ============================
def create_dpr_searcher(
    dpr_index_root_dir: str,
    docstore_index_dir: str,
    query_encoder_name: str = "facebook/dpr-question_encoder-multiset-base",
    device: Optional[str] = None,
    max_loaded_shards: int = 1,
    faiss_threads: Optional[int] = None,
    mmap: bool = True,
) -> DPRShardedSearcher:
    """
    Factory simile a create_bm25_searcher().
    docstore_index_dir: indice Lucene con storeRaw.
    """
    lucene = LuceneSearcher(docstore_index_dir)

    dev = None
    if device is not None:
        device = device.lower().strip()
        if device == "cpu":
            dev = torch.device("cpu")
        elif device == "cuda":
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            dev = torch.device(device)

    return DPRShardedSearcher(
        index_root_dir=dpr_index_root_dir,
        docstore=lucene,
        query_encoder_name=query_encoder_name,
        device=dev,
        max_loaded_shards=max_loaded_shards,
        faiss_threads=faiss_threads,
        mmap=mmap,
    )


def dpr_retrieve(query: str, searcher: DPRShardedSearcher = None, k: int = 50) -> List[str]:
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
    searcher: DPRShardedSearcher = None,
    k: int = 50,
    batch_size: int = 256,
    threads: int = 8,
    encode_batch_size: int = 32,
    max_query_len: int = 256,
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
            max_query_len=max_query_len,
            faiss_threads=threads,
        )

        # 1) estrai docids top-k per query
        per_query_docids: List[List[str]] = []
        all_docids = []
        for scored in scored_lists:
            docids = [d for d, _s in scored]
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
    parser = argparse.ArgumentParser(
        description="DPR retrieval (FAISS sharded) + Lucene docstore for contents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dpr_index_root_dir", required=True, help="Dir con part_0..part_N (FAISS)")
    parser.add_argument("--docstore_index_dir", required=True, help="Indice Lucene con storeRaw per docid->contents (es. bm25_index)")
    parser.add_argument("--query", required=True, help="Singola query")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--threads", type=int, default=8, help="FAISS omp threads")
    parser.add_argument("--max_loaded_shards", type=int, default=8, help="Shard cache size (LRU)")
    parser.add_argument("--mmap", action="store_true", help="Prova a usare faiss IO_FLAG_MMAP")
    args = parser.parse_args()

    searcher = create_dpr_searcher(
        dpr_index_root_dir=args.dpr_index_root_dir,
        docstore_index_dir=args.docstore_index_dir,
        max_loaded_shards=args.max_loaded_shards,
        faiss_threads=args.threads,
        mmap=args.mmap,
    )
    docs = dpr_retrieve(args.query, searcher=searcher, k=args.k)
    print(json.dumps({args.query: docs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()