"""
contriever_retriever.py
Dense retrieval su collezione JSONL preprocessata (id, contents) + indice FAISS OPQ+IVF-PQ.
Supporto offsets binari per random access (consigliato).

Uso CLI:
python contriever_retriever.py --faiss_index ./index_out_full/ivfpq_opq_contriever.faiss \
    --collection ./data/collection/wikipedia_passages.jsonl \
    --offsets ./index_out_full/collection_offsets.u64.bin \
    --query "When did Apollo 11 land?" --k 5
"""

import os
import json
import argparse
from typing import List, Dict, Optional

import numpy as np
import faiss
import mmap, struct

from contriever_encoder import ContrieverEncoder




# --------- JSONL collection with optional offsets ---------
class JsonlCollection:
    def __init__(self, path: str, offsets_path: Optional[str] = None, in_memory: bool = False):
        self.path = path
        self.in_memory = in_memory
        self._docs = None
        self._f = None
        self._off_f = None
        self._off_mm = None
        self._off_count = 0

        if in_memory:
            # carica tutto (solo mini-run)
            docs = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        rec = json.loads(line); docs.append(rec.get("contents",""))
                    except Exception: docs.append("")
            self._docs = docs
        elif offsets_path and os.path.exists(offsets_path):
            # apri e mappa gli offsets, e tieni aperto anche l'handle del file JSONL
            self._off_f = open(offsets_path, "rb")
            self._off_mm = mmap.mmap(self._off_f.fileno(), 0, access=mmap.ACCESS_READ)
            self._off_count = len(self._off_mm) // 8  # uint64 per riga
            self._f = open(self.path, "rb")

    def _offset_at(self, i: int) -> int:
        return struct.unpack_from("<Q", self._off_mm, i * 8)[0]

    def get_many(self, ids: List[int]) -> List[str]:
        if self.in_memory:
            return [self._docs[i] if 0 <= i < len(self._docs) else "" for i in ids]

        out = []
        if self._off_mm is None:
            # fallback lento senza offsets: apri/chiudi il file ad ogni chiamata
            with open(self.path, "rb") as f:
                max_id = max(ids) if ids else -1
                wanted = set(ids)
                got = {}
                for i, line in enumerate(f):
                    if i in wanted:
                        try:
                            rec = json.loads(line.decode("utf-8")); got[i] = rec.get("contents","")
                        except Exception: got[i] = ""
                    if i >= max_id: break
                for i in ids: out.append(got.get(i, ""))
            return out

        if self._f is None or self._f.closed:
            self._f = open(self.path, "rb")

        f = self._f
        for i in ids:
            if i < 0 or i >= self._off_count:
                out.append(""); continue
            f.seek(self._offset_at(i))
            line = f.readline()
            try:
                rec = json.loads(line.decode("utf-8")); out.append(rec.get("contents",""))
            except Exception:
                out.append("")
        return out
    
    def close(self) -> None:
        # chiudi in ordine: mmap -> file offsets -> file collection
        if self._off_mm is not None:
            try: self._off_mm.close()
            except Exception: pass
            self._off_mm = None
        if self._off_f is not None:
            try: self._off_f.close()
            except Exception: pass
            self._off_f = None
        if self._f is not None:
            try: self._f.close()
            except Exception: pass
            self._f = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# --------- Helper ---------
def _set_nprobe_deep(index: faiss.Index, nprobe: Optional[int]) -> None:
    if nprobe is None:
        return
    try:
        core = index
        # Unwrap IDMap
        if isinstance(core, (faiss.IndexIDMap, faiss.IndexIDMap2)) and hasattr(core, "index"):
            core = core.index
        # Unwrap PreTransform (OPQ)
        if isinstance(core, faiss.IndexPreTransform) and hasattr(core, "index"):
            core = core.index
        # Now core should be IVF-like
        if hasattr(core, "nprobe"):
            core.nprobe = int(nprobe)
    except Exception:
        pass

# --------- Dense Retriever ---------
class DenseRetriever:
    def __init__(self, index_path: str, collection_path: str, offsets_path: Optional[str] = None,
                 nprobe: Optional[int] = None, in_memory: bool = False):
        self.index = faiss.read_index(index_path)
        _set_nprobe_deep(self.index, nprobe)

        self.collection = JsonlCollection(collection_path, offsets_path=offsets_path, in_memory=in_memory)
        self.encoder = ContrieverEncoder()

    def dense_retrieve(self, query: str, k: int = 50, return_cosine: bool = True) -> Dict:
        q = self.encoder.encode([query], batch_size=1).astype(np.float32)
        D, I = self.index.search(q, k)  # L2 su vettori normalizzati
        ids = [int(x) for x in I[0]]
        dists = [float(x) for x in D[0]]
        docs = self.collection.get_many(ids)
        out = {"ids": ids, "distances_l2": dists, "documents": docs}
        if return_cosine:
            out["approx_cosine"] = [1.0 - 0.5 * d for d in dists]
        return out

    def batch_dense_retrieve(self, queries: List[str], k: int = 50, batch_size: int = 8, return_cosine: bool = True) -> List[Dict]:
        Q = self.encoder.encode(queries, batch_size=min(64, max(1, batch_size))).astype(np.float32)
        D, I = self.index.search(Q, k)
        
        # fetch in blocco
        unique_ids = sorted(set(int(x) for row in I for x in row))
        id2doc = dict(zip(unique_ids, self.collection.get_many(unique_ids)))
        
        results = []
        for i in range(len(queries)):
            ids = [int(x) for x in I[i]]
            dists = [float(x) for x in D[i]]
            docs = [id2doc[j] for j in ids]
            obj = {"ids": ids, "distances_l2": dists, "documents": docs}
            if return_cosine:
                obj["approx_cosine"] = [1.0 - 0.5 * d for d in dists]
            results.append(obj)
        
        return results
    
    def contriever_batch_retrieve(
        self,
        queries: List[str],
        k: int = 50,
        batch_size: int = 256,
        return_cosine: bool = False,
    ) -> Dict[str, List[str]]:
        """
        Restituisce {query: [doc1, doc2, ..., dock]} usando Contriever in batch.
        """
        out: Dict[str, List[str]] = {}
        for i in range(0, len(queries), batch_size):
            batch_q = queries[i : i + batch_size]
            results = self.batch_dense_retrieve(batch_q, k=k, return_cosine=return_cosine)
            for q, r in zip(batch_q, results):
                out[q] = r["documents"]
        return out


# --------- CLI ---------
def main():
    parser = argparse.ArgumentParser(description="Dense retrieval su JSONL + FAISS (no docstore, no memmap).",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--faiss_index", required=True, help="Path indice FAISS (.faiss)")
    parser.add_argument("--collection", required=True, help="Path JSONL preprocessato (id, contents)")
    parser.add_argument("--offsets", default=None, help="Offsets binari uint64 (opzionale, consigliato)")
    parser.add_argument("--query", required=True, help="Query text")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--nprobe", type=int, default=64)
    parser.add_argument("--in_memory", action="store_true", help="Carica l'intera collezione in RAM (solo mini-run)")
    args = parser.parse_args()

    retr = DenseRetriever(index_path=args.faiss_index,
                          collection_path=args.collection,
                          offsets_path=args.offsets,
                          nprobe=args.nprobe,
                          in_memory=args.in_memory)
    results = retr.dense_retrieve(args.query, k=args.k, return_cosine=True)
    print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()