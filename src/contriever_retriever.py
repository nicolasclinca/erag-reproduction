"""
contriever_retriever.py
Dense retrieval over a preprocessed JSONL collection (id, contents) + FAISS OPQ+IVF-PQ index.

doc_id alignment:
- The FAISS index built by build_contriever_indexes.py uses as FAISS internal ID
  the row index (0-based) of the JSONL file (id_scheme = "line_index").
- To be consistent with BM25/DPR/BGE/TCT and with the original collection, this retriever
  maps FAISS_ID (line index) -> original doc_id (rec["id"], e.g., "40885965_20")
  by reading the corresponding line from the JSONL (preferably via binary offsets).

CLI usage:
python contriever_retriever.py --faiss_index ./index_out_full/ivfpq_opq_contriever.faiss \
    --collection ../data/collection/wikipedia_passages.jsonl \
    --offsets ./index_out_full/collection_offsets.u64.bin \
    --query "When did Apollo 11 land?" --k 5
"""

import os
import json
import argparse
from typing import List, Dict, Optional, TypedDict

import numpy as np
import faiss  # type: ignore
import mmap
import struct

from contriever_encoder import ContrieverEncoder


class RetrievedDoc(TypedDict):
    doc_id: str
    score: float
    contents: str


# --------- JSONL collection with optional offsets ---------
class JsonlCollection:
    """
    Random access (or best-effort) to a JSONL file where each line is a record:
      {"id": "...", "contents": "..."}  (or compatible)

    When used with Contriever:
    - FAISS returns as ID the JSONL line index (line index).
    - Here we resolve line index -> record (original doc_id + contents).

    offsets_path (recommended):
    - little-endian uint64 binary file, one offset per line (built with build_offsets.py)
    """

    def __init__(self, path: str, offsets_path: Optional[str] = None, in_memory: bool = False):
        self.path = path
        self.in_memory = in_memory

        self._docs: Optional[List[Dict[str, str]]] = None  # in-memory list of records {doc_id, contents}
        self._f = None

        self._off_f = None
        self._off_mm = None
        self._off_count = 0

        if in_memory:
            # load everything (mini-run / debug only)
            docs: List[Dict[str, str]] = []
            with open(path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if not line.strip():
                        docs.append({"doc_id": "", "contents": ""})
                        continue
                    try:
                        rec = json.loads(line)
                        doc_id = str(rec.get("id", i))
                        contents = rec.get("contents") or rec.get("text") or ""
                        docs.append({"doc_id": doc_id, "contents": str(contents) if contents is not None else ""})
                    except Exception:
                        docs.append({"doc_id": "", "contents": ""})
            self._docs = docs

        elif offsets_path and os.path.exists(offsets_path):
            # open and mmap offsets, and also keep the JSONL file handle open
            self._off_f = open(offsets_path, "rb")
            self._off_mm = mmap.mmap(self._off_f.fileno(), 0, access=mmap.ACCESS_READ)
            self._off_count = len(self._off_mm) // 8  # uint64 per line
            self._f = open(self.path, "rb")

    def _offset_at(self, i: int) -> int:
        return struct.unpack_from("<Q", self._off_mm, i * 8)[0]

    def get_many_records(self, ids: List[int]) -> List[Dict[str, str]]:
        """
        Returns a list of records in the same order as ids:
          [{"doc_id": <original_id>, "contents": <text>}, ...]
        """
        if not ids:
            return []

        if self.in_memory:
            assert self._docs is not None
            out: List[Dict[str, str]] = []
            for i in ids:
                if 0 <= i < len(self._docs):
                    out.append(self._docs[i])
                else:
                    out.append({"doc_id": "", "contents": ""})
            return out

        out: List[Dict[str, str]] = []

        if self._off_mm is None:
            # slow fallback without offsets: scan the file up to max_id
            max_id = max(ids)
            wanted = set(ids)
            got: Dict[int, Dict[str, str]] = {}

            with open(self.path, "rb") as f:
                for i, line in enumerate(f):
                    if i in wanted:
                        try:
                            rec = json.loads(line.decode("utf-8"))
                            doc_id = str(rec.get("id", i))
                            contents = rec.get("contents") or rec.get("text") or ""
                            got[i] = {"doc_id": doc_id, "contents": str(contents) if contents is not None else ""}
                        except Exception:
                            got[i] = {"doc_id": "", "contents": ""}
                    if i >= max_id:
                        break

            for i in ids:
                out.append(got.get(i, {"doc_id": "", "contents": ""}))
            return out

        # offsets available
        if self._f is None or self._f.closed:
            self._f = open(self.path, "rb")

        f = self._f
        for i in ids:
            if i < 0 or i >= self._off_count:
                out.append({"doc_id": "", "contents": ""})
                continue

            f.seek(self._offset_at(i))
            line = f.readline()
            try:
                rec = json.loads(line.decode("utf-8"))
                doc_id = str(rec.get("id", i))
                contents = rec.get("contents") or rec.get("text") or ""
                out.append({"doc_id": doc_id, "contents": str(contents) if contents is not None else ""})
            except Exception:
                out.append({"doc_id": "", "contents": ""})

        return out

    def get_many(self, ids: List[int]) -> List[str]:
        """
        Backward-compatible: returns only the contents (same order as ids).
        """
        recs = self.get_many_records(ids)
        return [r.get("contents", "") for r in recs]

    def close(self) -> None:
        # close in order: mmap -> offsets file -> collection file
        if self._off_mm is not None:
            try:
                self._off_mm.close()
            except Exception:
                pass
            self._off_mm = None

        if self._off_f is not None:
            try:
                self._off_f.close()
            except Exception:
                pass
            self._off_f = None

        if self._f is not None:
            try:
                self._f.close()
            except Exception:
                pass
            self._f = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# --------- Helper ---------
def _set_nprobe_deep(index: faiss.Index, nprobe: Optional[int]) -> None:
    """
    Sets nprobe also when the index is wrapped in IDMap2 and/or IndexPreTransform (OPQ).
    """
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
    def __init__(
        self,
        index_path: str,
        collection_path: str,
        offsets_path: Optional[str] = None,
        nprobe: Optional[int] = None,
        in_memory: bool = False,
    ):
        self.index = faiss.read_index(index_path)
        _set_nprobe_deep(self.index, nprobe)

        self.collection = JsonlCollection(collection_path, offsets_path=offsets_path, in_memory=in_memory)
        self.encoder = ContrieverEncoder()

    def dense_retrieve(self, query: str, k: int = 50, return_cosine: bool = True) -> Dict:
        """
        Returns a "debug-friendly" object with:
          - faiss_ids: line index in the JSONL (int)
          - doc_ids: original rec["id"] (string)
          - distances_l2
          - documents (contents)
          - approx_cosine (optional)
        """
        q = self.encoder.encode([query], batch_size=1).astype(np.float32)
        D, I = self.index.search(q, k)  # L2 on normalized vectors

        faiss_ids = [int(x) for x in I[0]]
        dists = [float(x) for x in D[0]]

        valid_ids = [i for i in faiss_ids if i >= 0]
        recs = self.collection.get_many_records(valid_ids)
        id2rec = dict(zip(valid_ids, recs))

        doc_ids: List[str] = []
        docs: List[str] = []
        for fid in faiss_ids:
            if fid < 0:
                doc_ids.append("")
                docs.append("")
                continue
            r = id2rec.get(fid, {"doc_id": str(fid), "contents": ""})
            doc_ids.append(r.get("doc_id", ""))
            docs.append(r.get("contents", ""))

        out = {
            "faiss_ids": faiss_ids,
            "doc_ids": doc_ids,
            "distances_l2": dists,
            "documents": docs,
        }
        if return_cosine:
            out["approx_cosine"] = [1.0 - 0.5 * d for d in dists]
        return out

    def batch_dense_retrieve(
        self,
        queries: List[str],
        k: int = 50,
        batch_size: int = 8,
        return_cosine: bool = True,
    ) -> List[Dict]:
        """
        Batch retrieval:
        - Encodes the queries
        - Searches on FAISS
        - Resolves FAISS_ID (line index) -> original doc_id and contents via JsonlCollection

        Returns a list of dicts (one per query), same fields as dense_retrieve().
        """
        if not queries:
            return []

        # batch_size here is intended as "query batch size"; for the encoder we cap it at 64
        Q = self.encoder.encode(queries, batch_size=min(64, max(1, batch_size))).astype(np.float32)
        D, I = self.index.search(Q, k)

        # bulk fetch: dedup all requested IDs (valid only)
        unique_ids = sorted(set(int(x) for row in I for x in row if int(x) >= 0))
        recs = self.collection.get_many_records(unique_ids)
        id2rec = dict(zip(unique_ids, recs))

        results: List[Dict] = []
        for i in range(len(queries)):
            faiss_ids = [int(x) for x in I[i]]
            dists = [float(x) for x in D[i]]

            doc_ids: List[str] = []
            docs: List[str] = []
            for fid in faiss_ids:
                if fid < 0:
                    doc_ids.append("")
                    docs.append("")
                    continue
                r = id2rec.get(fid, {"doc_id": str(fid), "contents": ""})
                doc_ids.append(r.get("doc_id", ""))
                docs.append(r.get("contents", ""))

            obj = {
                "faiss_ids": faiss_ids,
                "doc_ids": doc_ids,
                "distances_l2": dists,
                "documents": docs,
            }
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
    ) -> Dict[str, List[RetrievedDoc]]:
        """
        API used by build_retrieval_run.py / dataset_builder.py

        Returns:
            {query: [{"doc_id": str, "score": float, "contents": str}, ...]}

        doc_id:
          - original doc_id from the collection (rec["id"], e.g., "40885965_20")

        score:
          - if return_cosine=True uses approx_cosine (higher=better)
          - otherwise uses -L2 (higher=better and preserves ranking)
        """
        out: Dict[str, List[RetrievedDoc]] = {}

        for i in range(0, len(queries), batch_size):
            batch_q = queries[i : i + batch_size]

            dense_results = self.batch_dense_retrieve(
                batch_q, k=k, batch_size=batch_size, return_cosine=return_cosine
            )

            for q, r in zip(batch_q, dense_results):
                faiss_ids = r.get("faiss_ids", [])
                doc_ids = r.get("doc_ids", [])
                docs = r.get("documents", [])
                dists = r.get("distances_l2", [])
                cos = r.get("approx_cosine", None)

                per_q: List[RetrievedDoc] = []
                n = min(len(faiss_ids), len(doc_ids), len(docs))

                for j in range(n):
                    fid = int(faiss_ids[j])
                    if fid < 0:
                        continue

                    did = str(doc_ids[j] or "").strip()
                    if not did:
                        # fallback: if the line has no id, use fid (not ideal, but avoids empty strings)
                        did = str(fid)

                    if return_cosine and cos is not None and j < len(cos):
                        score = float(cos[j])
                    else:
                        score = -float(dists[j]) if j < len(dists) else 0.0

                    per_q.append({"doc_id": did, "score": score, "contents": docs[j] or ""})

                out[q] = per_q

        return out


# --------- CLI ---------
def main():
    parser = argparse.ArgumentParser(
        description="Dense retrieval over JSONL + FAISS (Contriever). doc_id aligned to the JSONL 'id' field.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--faiss_index", required=True, help="FAISS index path (.faiss)")
    parser.add_argument("--collection", required=True, help="Preprocessed JSONL path (id, contents)")
    parser.add_argument("--offsets", default=None, help="Binary uint64 offsets (optional, recommended)")
    parser.add_argument("--query", required=True, help="Query text")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--nprobe", type=int, default=64)
    parser.add_argument("--in_memory", action="store_true", help="Load the entire collection into RAM (mini-run only)")
    args = parser.parse_args()

    retr = DenseRetriever(
        index_path=args.faiss_index,
        collection_path=args.collection,
        offsets_path=args.offsets,
        nprobe=args.nprobe,
        in_memory=args.in_memory,
    )

    results = retr.dense_retrieve(args.query, k=args.k, return_cosine=True)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()