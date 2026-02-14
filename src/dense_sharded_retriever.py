"""
dense_sharded_retriever.py
Generic dense retrieval over *sharded* FAISS indexes in PySerini style, with rid->docid mapping
and a Lucene docstore for docid->contents.

Expected index structure (as for your DPR/BGE/TCT indexes):

indexes/<name>/
  part_0/
    index   # FAISS shard
    docid   # mapping rid -> docid (one line per vector)
  part_1/
    index
    docid
  ...

Main features:
- Loads ALL FAISS shards only once (optional mmap).
- For a batch of queries:
  1) encodes queries using a query encoder (build_query_encoder from query_encoders.py)
  2) searches each shard (top per_shard_k)
  3) merges results into a global top-k (per query) via heap
  4) resolves rid -> docid using shard docid files (loaded on-demand with an LRU cache)
  5) resolves docid -> contents using a Lucene docstore (storeRaw)

Note on per_shard_k (accuracy):
- If you only retrieve k per shard, the global top-k may be approximate.
- Setting per_shard_k > k reduces the risk of “miss”.
- Default: per_shard_k = k*4 (capped at 1000).

For the Lucene docstore:
python -m pyserini.index.lucene \
  -collection JsonCollection \
  -input ../data/collection \
  -index ../indexes/wiki_docstore_lucene \
  -generator DefaultLuceneDocumentGenerator \
  -threads 8 \
  -storeRaw

CLI usage:
python dense_sharded_retriever.py \
  --index_root_dir ../indexes/wiki-bge-118m \
  --docstore_index_dir ../indexes/wiki_docstore_lucene \
  --encoder_type bge \
  --query "When did Apollo 11 land?" \
  --k 50
"""

from __future__ import annotations

import os
import re
import json
import heapq
import argparse
from typing import Dict, List, Optional, Tuple, Protocol, TypedDict
from collections import OrderedDict

import numpy as np
import faiss  # type: ignore
from pyserini.search.lucene import LuceneSearcher  # type: ignore

from query_encoders import build_query_encoder


# ============================
# Shard listing/sorting
# ============================
_PART_RE = re.compile(r"^part_(\d+)$")


def _list_part_dirs(root: str) -> List[str]:
    """Lists and sorts part_0..part_N numerically."""
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
# Query encoder protocol
# ============================
class QueryEncoder(Protocol):
    """
    Minimal required interface:
    - .D
    - .encode(List[str], batch_size) -> np.ndarray float32 (B, D)
    """

    D: int

    def encode(self, queries: List[str], batch_size: int = 32) -> np.ndarray:
        ...


# ============================
# LRU cache (docid files)
# ============================
class _LRUCache:
    """
    Minimal LRU cache to load into RAM the docids of only a few shards at a time.
    Useful because docid files can be large.
    """

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
# FAISS helpers
# ============================
def _safe_omp_set_threads(n: Optional[int]) -> None:
    """Sets FAISS threads (best effort)."""
    if n is None:
        return
    try:
        faiss.omp_set_num_threads(int(n))
    except Exception:
        pass


def _read_faiss_index(path: str, mmap: bool) -> "faiss.Index":
    """
    Loads a FAISS index.
    If mmap=True, tries IO_FLAG_MMAP, otherwise falls back to standard loading.
    """
    if mmap and hasattr(faiss, "IO_FLAG_MMAP"):
        try:
            return faiss.read_index(path, faiss.IO_FLAG_MMAP)
        except Exception:
            pass
    return faiss.read_index(path)


def _assert_inner_product_metric(indexes: List["faiss.Index"], part_dirs: List[str]) -> None:
    """
    Consistency check: all shards should have metric_type = INNER_PRODUCT.
    If not available or not exposed, do not block.
    """
    try:
        expected = int(getattr(indexes[0], "metric_type", faiss.METRIC_INNER_PRODUCT))
        if expected != int(faiss.METRIC_INNER_PRODUCT):
            raise RuntimeError(
                f"Shard {part_dirs[0]}: metric_type={expected} (expected INNER_PRODUCT). "
                "The logic for the correct metric has been removed; restore it."
            )

        for idx, d in zip(indexes[1:], part_dirs[1:]):
            mt = int(getattr(idx, "metric_type", expected))
            if mt != expected:
                raise RuntimeError(f"Non-uniform FAISS metric across shards: {d} metric_type={mt}, expected {expected}")
    except AttributeError:
        # Some wrappers may not expose metric_type: do not block
        return


# ============================
# Sharded FAISS Searcher
# ============================
class RetrievedDoc(TypedDict):
    doc_id: str
    score: float
    contents: str


class ShardedFaissSearcher:
    """
    Generic searcher over PySerini-style FAISS shards, using an external QueryEncoder.

    Parameters
    ---------
    index_root_dir:
        Root with part_0..part_N
    query_encoder:
        Encoder with .encode(...) -> float32 embeddings (B, D)
    docstore_index_dir:
        Lucene index with storeRaw for docid -> raw json -> contents.
        If None, you can use search_docids_batch() but not docids_to_contents().
    max_loaded_docid_shards:
        LRU cache size for docids (shard docid files)
    mmap:
        If True, tries faiss.read_index(..., IO_FLAG_MMAP)
    default_per_shard_k:
        If None, per_shard_k is decided per call (k*4 capped)
    assert_inner_product:
        If True, checks that shards are INNER_PRODUCT.
    """

    def __init__(
        self,
        index_root_dir: str,
        query_encoder: QueryEncoder,
        docstore_index_dir: Optional[str] = None,
        max_loaded_docid_shards: int = 16,
        faiss_threads: Optional[int] = None,
        mmap: bool = True,
        default_per_shard_k: Optional[int] = None,
        assert_inner_product: bool = True,
    ):
        self.index_root_dir = index_root_dir
        self.part_dirs = _list_part_dirs(index_root_dir)
        if not self.part_dirs:
            raise ValueError(f"No part_* shard found in: {index_root_dir}")

        self.query_encoder = query_encoder
        self.mmap = bool(mmap)
        self.default_per_shard_k = default_per_shard_k

        self.docstore: Optional[LuceneSearcher] = LuceneSearcher(docstore_index_dir) if docstore_index_dir else None
        self.docid_cache = _LRUCache(max_loaded=max_loaded_docid_shards)

        _safe_omp_set_threads(faiss_threads)

        # PRELOAD: load all FAISS shards once
        self.indexes: List["faiss.Index"] = []
        for shard_dir in self.part_dirs:
            self.indexes.append(self._load_index(shard_dir))

        # Metric check (optional)
        if assert_inner_product:
            _assert_inner_product_metric(self.indexes, self.part_dirs)

    # ---------- Index loading ----------
    def _load_index(self, shard_dir: str) -> "faiss.Index":
        index_path = os.path.join(shard_dir, "index")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Missing FAISS index file: {index_path}")
        return _read_faiss_index(index_path, mmap=self.mmap)

    # ---------- DocID loading ----------
    def _load_docids(self, shard_dir: str) -> List[str]:
        docid_path = os.path.join(shard_dir, "docid")
        if not os.path.exists(docid_path):
            raise FileNotFoundError(f"Missing docid file: {docid_path}")

        # splitlines() avoids the case of an empty last line
        with open(docid_path, "r", encoding="utf-8") as f:
            return f.read().splitlines()

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
        per_shard_k: Optional[int] = None,
    ) -> List[List[Tuple[str, float]]]:
        """
        Returns, for each query, a list of (docid, score) with global top-k.

        Implementation:
        - encodes all queries (B, D)
        - for each shard: search top per_shard_k
        - global merge per query with a heap (min-heap of size k)
        - resolves rid->docid by loading only the docid files of shards actually present in the top-k
        """
        if not queries:
            return []
        if k <= 0:
            return [[] for _ in queries]

        _safe_omp_set_threads(faiss_threads)

        # Decide per_shard_k
        if per_shard_k is None:
            if self.default_per_shard_k is not None:
                per_shard_k = int(self.default_per_shard_k)
            else:
                per_shard_k = int(min(1000, max(k, k * 4)))
        per_shard_k = max(int(per_shard_k), k)

        # Encode queries
        Q = self.query_encoder.encode(queries, batch_size=encode_batch_size)
        if not isinstance(Q, np.ndarray) or Q.ndim != 2:
            raise RuntimeError("Invalid query embeddings shape.")
        qn, qd = Q.shape
        Qc = np.ascontiguousarray(Q, dtype=np.float32)

        # heaps[i] is a min-heap of (score, shard_idx, rid), size <= k
        # score: INNER_PRODUCT => “higher is better”
        heaps: List[List[Tuple[float, int, int]]] = [[] for _ in range(qn)]

        for shard_idx, shard_dir in enumerate(self.part_dirs):
            shard_index = self.indexes[shard_idx]

            # dimensional check: mismatch = wrong encoder or wrong index
            if int(getattr(shard_index, "d", -1)) != int(qd):
                raise RuntimeError(
                    f"Dim mismatch: query_dim={qd} vs index_dim={int(getattr(shard_index, 'd', -1))} in {shard_dir}"
                )

            raw_scores, ids = shard_index.search(Qc, int(per_shard_k))  # (qn, per_shard_k)

            for i in range(qn):
                h = heaps[i]
                for j in range(per_shard_k):
                    rid = int(ids[i, j])
                    if rid < 0:
                        continue

                    s = float(raw_scores[i, j])
                    item = (s, shard_idx, rid)

                    if len(h) < k:
                        heapq.heappush(h, item)
                    else:
                        if s > h[0][0]:
                            heapq.heapreplace(h, item)

        # Determine needed shards (lazy load docids)
        needed_shards = {sh for h in heaps for (_s, sh, _rid) in h}
        docids_by_shard: Dict[int, List[str]] = {
            sh: self._get_docids(self.part_dirs[sh]) for sh in needed_shards
        }

        # Build per-query output: (docid, score) sorted desc
        out: List[List[Tuple[str, float]]] = []
        for i in range(qn):
            items_sorted = sorted(heaps[i], key=lambda x: x[0], reverse=True)
            res_i: List[Tuple[str, float]] = []
            for s, sh, rid in items_sorted:
                docids = docids_by_shard.get(sh, [])
                did = docids[rid] if 0 <= rid < len(docids) else ""
                res_i.append((did, float(s)))
            out.append(res_i)

        return out

    def docids_to_contents(self, docids: List[str]) -> List[str]:
        """
        Converts docid -> contents using the Lucene docstore (storeRaw).
        """
        if self.docstore is None:
            raise RuntimeError("docstore not configured: pass docstore_index_dir to use docids_to_contents().")

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
    
    def batch_retrieve(
        self,
        queries: List[str],
        k: int = 50,
        batch_size: int = 256,
        faiss_threads: Optional[int] = 8,
        encode_batch_size: int = 32,
        per_shard_k: Optional[int] = None,
    ) -> Dict[str, List[RetrievedDoc]]:
        """
        Returns:
            {query: [{"doc_id": str, "score": float, "contents": str}, ...]}

        score is the FAISS inner product (higher=better).
        """
        if not queries:
            return {}

        if self.docstore is None:
            raise RuntimeError(
                "docstore not configured: pass docstore_index_dir to use batch_retrieve()."
            )

        results: Dict[str, List[RetrievedDoc]] = {}

        for i in range(0, len(queries), batch_size):
            batch_q = queries[i : i + batch_size]

            scored_lists = self.search_docids_batch(
                batch_q,
                k=k,
                encode_batch_size=encode_batch_size,
                faiss_threads=faiss_threads,
                per_shard_k=per_shard_k,
            )  # List[List[Tuple[docid, score]]]

            # collect docids per query and global dedup (to minimize Lucene lookups)
            per_query_pairs: List[List[Tuple[str, float]]] = []
            all_docids: List[str] = []

            for scored in scored_lists:
                pairs = [(d, float(s)) for d, s in scored if d]
                per_query_pairs.append(pairs)
                all_docids.extend([d for d, _s in pairs])

            unique_docids = list(dict.fromkeys(all_docids))  # preserves order
            did2cont = dict(zip(unique_docids, self.docids_to_contents(unique_docids)))

            for q, pairs in zip(batch_q, per_query_pairs):
                results[q] = [
                    {"doc_id": d, "score": float(s), "contents": did2cont.get(d, "")}
                    for d, s in pairs
                ]

        return results


# ============================
# Thin functional API
# ============================
def dense_sharded_batch_retrieve(
    queries: List[str],
    searcher: ShardedFaissSearcher,
    k: int = 50,
    batch_size: int = 256,
    threads: int = 8,
    encode_batch_size: int = 32,
    per_shard_k: Optional[int] = None,
) -> Dict[str, List[RetrievedDoc]]:
    """
    Functional wrapper: {query: [{"doc_id","score","contents"}...]}.
    """
    if searcher is None:
        raise ValueError("A ShardedFaissSearcher instance must be provided.")
    return searcher.batch_retrieve(
        queries=queries,
        k=k,
        batch_size=batch_size,
        faiss_threads=threads,
        encode_batch_size=encode_batch_size,
        per_shard_k=per_shard_k,
    )


def dense_sharded_retrieve(
    query: str,
    searcher: ShardedFaissSearcher,
    k: int = 50,
    per_shard_k: Optional[int] = None,
) -> List[RetrievedDoc]:
    if searcher is None:
        raise ValueError("A ShardedFaissSearcher instance must be provided.")
    out = searcher.batch_retrieve(
        queries=[query],
        k=k,
        batch_size=1,
        faiss_threads=None,
        encode_batch_size=1,
        per_shard_k=per_shard_k,
    )
    return out.get(query, [])


# ============================
# CLI
# ============================
def main():
    parser = argparse.ArgumentParser(
        description="Generic dense retrieval over sharded FAISS + Lucene docstore.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--index_root_dir", required=True, help="Dir with part_0..part_N (FAISS shards)")
    parser.add_argument("--docstore_index_dir", required=True,
        help="Lucene index with storeRaw for docid->contents (e.g., wiki_docstore_lucene or bm25_index)")
    parser.add_argument("--encoder_type", choices=["dpr", "bge", "tct"], required=True)

    parser.add_argument("--query", required=True, help="Single query")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--per_shard_k", type=int, default=None, help="How many results per shard before merging.")
    parser.add_argument("--threads", type=int, default=8, help="FAISS omp threads (CPU)")
    parser.add_argument("--encode_batch_size", type=int, default=32, help="Batch size for query encoding")
    parser.add_argument("--max_loaded_docid_shards", type=int, default=16, help="Docid LRU cache size")
    parser.add_argument("--mmap", dest="mmap", action="store_true", help="Use FAISS IO_FLAG_MMAP (default)")
    parser.add_argument("--no_mmap", dest="mmap", action="store_false", help="Disable mmap")
    parser.set_defaults(mmap=True)

    parser.add_argument("--no_assert_inner_product", dest="assert_inner_product", action="store_false",
        help="Disables the metric_type==INNER_PRODUCT check on shards.")
    parser.set_defaults(assert_inner_product=True)

    args = parser.parse_args()

    query_encoder = build_query_encoder(encoder_type=args.encoder_type)

    searcher = ShardedFaissSearcher(
        index_root_dir=args.index_root_dir,
        query_encoder=query_encoder,
        docstore_index_dir=args.docstore_index_dir,
        max_loaded_docid_shards=args.max_loaded_docid_shards,
        faiss_threads=args.threads,
        mmap=args.mmap,
        assert_inner_product=args.assert_inner_product,
    )

    docs = dense_sharded_retrieve(args.query, searcher=searcher, k=args.k, per_shard_k=args.per_shard_k)
    print(json.dumps({args.query: docs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()