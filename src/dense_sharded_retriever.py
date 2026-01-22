"""
dense_sharded_retriever.py
Dense retrieval generico su indici FAISS *sharded* in stile PySerini, con mapping rid->docid
e docstore Lucene per docid->contents.

Struttura indice attesa (come per i tuoi indici DPR/BGE/TCT):

indexes/<name>/
  part_0/
    index   # shard FAISS
    docid   # mapping rid -> docid (una riga per vettore)
  part_1/
    index
    docid
  ...

Caratteristiche principali:
- Carica TUTTI gli shard FAISS una sola volta (opzionale mmap).
- Per un batch di query:
  1) encoda le query usando un query encoder (build_query_encoder da query_encoders.py)
  2) fa search su ogni shard (top per_shard_k)
  3) fonde i risultati in un top-k globale (per query) tramite heap
  4) risolve rid -> docid usando i file docid degli shard (caricati on-demand con LRU cache)
  5) risolve docid -> contents usando un docstore Lucene (storeRaw)

Nota su per_shard_k (accuratezza):
- Se cerchi solo k per shard, il top-k globale può risultare approssimato.
- Impostare per_shard_k > k riduce il rischio di “miss”.
- Default: per_shard_k = k*4 (capped a 1000).

Uso CLI:
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
    """Lista e ordina part_0..part_N in modo numerico."""
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
    Interfaccia minima richiesta:
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
    Cache LRU minimale per caricare in RAM i docid di pochi shard alla volta.
    Utile perché i docid file possono essere grandi.
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
    """Imposta threads FAISS (best effort)."""
    if n is None:
        return
    try:
        faiss.omp_set_num_threads(int(n))
    except Exception:
        pass


def _read_faiss_index(path: str, mmap: bool) -> "faiss.Index":
    """
    Carica un indice FAISS.
    Se mmap=True prova IO_FLAG_MMAP, altrimenti fallback standard.
    """
    if mmap and hasattr(faiss, "IO_FLAG_MMAP"):
        try:
            return faiss.read_index(path, faiss.IO_FLAG_MMAP)
        except Exception:
            pass
    return faiss.read_index(path)


def _assert_inner_product_metric(indexes: List["faiss.Index"], part_dirs: List[str]) -> None:
    """
    Controllo di coerenza: tutti gli shard dovrebbero avere metric_type = INNER_PRODUCT.
    Se non disponibile o non esposto, non blocca.
    """
    try:
        expected = int(getattr(indexes[0], "metric_type", faiss.METRIC_INNER_PRODUCT))
        if expected != int(faiss.METRIC_INNER_PRODUCT):
            raise RuntimeError(
                f"Shard {part_dirs[0]}: metric_type={expected} (atteso INNER_PRODUCT). "
                "La logica per la metrica corretta è stata rimossa; ripristinala."
            )

        for idx, d in zip(indexes[1:], part_dirs[1:]):
            mt = int(getattr(idx, "metric_type", expected))
            if mt != expected:
                raise RuntimeError(f"Metrica FAISS non uniforme tra shard: {d} metric_type={mt}, atteso {expected}")
    except AttributeError:
        # Alcuni wrapper potrebbero non esporre metric_type: non bloccare
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
    Searcher generico su shard FAISS in stile PySerini, usando un QueryEncoder esterno.

    Parametri
    ---------
    index_root_dir:
        Root con part_0..part_N
    query_encoder:
        Encoder con .encode(...) -> embedding float32 (B, D)
    docstore_index_dir:
        Indice Lucene con storeRaw per docid -> raw json -> contents.
        Se None, puoi usare search_docids_batch() ma non docids_to_contents().
    max_loaded_docid_shards:
        Dimensione cache LRU per docid (shard docid file)
    mmap:
        Se True tenta faiss.read_index(..., IO_FLAG_MMAP)
    default_per_shard_k:
        Se None, per_shard_k viene deciso per chiamata (k*4 capped)
    assert_inner_product:
        Se True, verifica che gli shard siano INNER_PRODUCT.
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
            raise ValueError(f"Nessuno shard part_* trovato in: {index_root_dir}")

        self.query_encoder = query_encoder
        self.mmap = bool(mmap)
        self.default_per_shard_k = default_per_shard_k

        self.docstore: Optional[LuceneSearcher] = LuceneSearcher(docstore_index_dir) if docstore_index_dir else None
        self.docid_cache = _LRUCache(max_loaded=max_loaded_docid_shards)

        _safe_omp_set_threads(faiss_threads)

        # PRELOAD: carica una volta tutti gli shard FAISS
        self.indexes: List["faiss.Index"] = []
        for shard_dir in self.part_dirs:
            self.indexes.append(self._load_index(shard_dir))

        # Check metrica (opzionale)
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

        # splitlines() evita il caso di ultima riga vuota
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
        Restituisce, per ogni query, una lista di (docid, score) con top-k globale.

        Implementazione:
        - encoda tutte le query (B, D)
        - per ogni shard: search top per_shard_k
        - merge globale per query con heap (min-heap di dimensione k)
        - risolve rid->docid caricando solo i docid file degli shard effettivamente presenti nel top-k
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
            raise RuntimeError("Query embeddings shape non valida.")
        qn, qd = Q.shape
        Qc = np.ascontiguousarray(Q, dtype=np.float32)

        # heaps[i] è una min-heap di (score, shard_idx, rid), size <= k
        # score: INNER_PRODUCT => “higher is better”
        heaps: List[List[Tuple[float, int, int]]] = [[] for _ in range(qn)]

        for shard_idx, shard_dir in enumerate(self.part_dirs):
            shard_index = self.indexes[shard_idx]

            # check dimensionale: mismatch = encoder sbagliato o indice sbagliato
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

        # Determina gli shard necessari (lazy load docids)
        needed_shards = {sh for h in heaps for (_s, sh, _rid) in h}
        docids_by_shard: Dict[int, List[str]] = {
            sh: self._get_docids(self.part_dirs[sh]) for sh in needed_shards
        }

        # Build output per query: (docid, score) ordinati desc
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
        Converte docid -> contents usando il docstore Lucene (storeRaw).
        """
        if self.docstore is None:
            raise RuntimeError("docstore non configurato: passa docstore_index_dir per usare docids_to_contents().")

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

        score è l'inner product FAISS (higher=better).
        """
        if not queries:
            return {}

        if self.docstore is None:
            raise RuntimeError(
                "docstore non configurato: passa docstore_index_dir per usare batch_retrieve()."
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

            # raccogli docid per query e dedup globale (per minimizzare lookup Lucene)
            per_query_pairs: List[List[Tuple[str, float]]] = []
            all_docids: List[str] = []

            for scored in scored_lists:
                pairs = [(d, float(s)) for d, s in scored if d]
                per_query_pairs.append(pairs)
                all_docids.extend([d for d, _s in pairs])

            unique_docids = list(dict.fromkeys(all_docids))  # preserva ordine
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
    Wrapper funzionale: {query: [{"doc_id","score","contents"}...]}.
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
    parser.add_argument("--index_root_dir", required=True, help="Dir con part_0..part_N (FAISS shards)")
    parser.add_argument("--docstore_index_dir", required=True,
        help="Indice Lucene con storeRaw per docid->contents (es. wiki_docstore_lucene o bm25_index)")
    parser.add_argument("--encoder_type", choices=["dpr", "bge", "tct"], required=True)

    parser.add_argument("--query", required=True, help="Singola query")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--per_shard_k", type=int, default=None, help="Quanti risultati per shard prima del merge.")
    parser.add_argument("--threads", type=int, default=8, help="FAISS omp threads (CPU)")
    parser.add_argument("--encode_batch_size", type=int, default=32, help="Batch size per query encoding")
    parser.add_argument("--max_loaded_docid_shards", type=int, default=16, help="Docid LRU cache size")
    parser.add_argument("--mmap", dest="mmap", action="store_true", help="Use FAISS IO_FLAG_MMAP (default)")
    parser.add_argument("--no_mmap", dest="mmap", action="store_false", help="Disable mmap")
    parser.set_defaults(mmap=True)

    parser.add_argument("--no_assert_inner_product", dest="assert_inner_product", action="store_false",
        help="Disabilita il controllo metric_type==INNER_PRODUCT sugli shard.")
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