"""
contriever_retriever.py
Dense retrieval su collezione JSONL preprocessata (id, contents) + indice FAISS OPQ+IVF-PQ.
Supporto offsets binari per random access (consigliato).

Uso CLI:
  python retrieval.py --index ./index_out_full/ivfpq_opq_contriever.faiss \
                      --collection ./data/collection/wikipedia_passages.jsonl \
                      --offsets ./index_out_full/collection_offsets.u64.bin \
                      --query "When did Apollo 11 land?" --k 5
"""

import os
import json
import argparse
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import faiss
from transformers import AutoTokenizer, AutoModel
import mmap, struct

MODEL_NAME = "facebook/contriever"
MAX_LENGTH = 200
DTYPE = torch.float16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------- JSONL collection with optional offsets ---------
class JsonlCollection:
    def __init__(self, path: str, offsets_path: Optional[str] = None, in_memory: bool = False):
        self.path = path
        self.in_memory = in_memory
        self._docs = None
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
            self._off_f = open(offsets_path, "rb")
            self._off_mm = mmap.mmap(self._off_f.fileno(), 0, access=mmap.ACCESS_READ)
            self._off_count = len(self._off_mm) // 8  # uint64 per riga

    def _offset_at(self, i: int) -> int:
        return struct.unpack_from("<Q", self._off_mm, i * 8)[0]

    def get_many(self, ids: List[int]) -> List[str]:
        if self.in_memory:
            return [self._docs[i] if 0 <= i < len(self._docs) else "" for i in ids]

        out = []
        with open(self.path, "rb") as f:
            if self._off_mm is None:
                # fallback lento senza offsets
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
            # con offsets (veloce)
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

# --------- Contriever Encoder (query) ---------
class ContrieverEncoder:
    def __init__(self, model_name=MODEL_NAME, device=DEVICE, max_length=MAX_LENGTH, dtype=DTYPE):
        self.device = device; self.max_length = max_length; self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(device); self.model.eval()
        torch.backends.cuda.matmul.allow_tf32 = True
        try: torch.set_float32_matmul_precision("high")
        except Exception: pass

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        vecs = []
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
                inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
                with torch.amp.autocast(device_type='cuda', dtype=self.dtype, enabled=(self.device.type=="cuda")):
                    x = self.model(**inputs).last_hidden_state
                    mask = inputs["attention_mask"].to(x.dtype).unsqueeze(-1)
                    summed = (x * mask).sum(dim=1)
                    lengths = mask.sum(dim=1).clamp(min=1e-6)
                    mean = summed / lengths
                    norm = F.normalize(mean, p=2, dim=1)
                vecs.append(norm.float().cpu())
        return torch.cat(vecs, dim=0).contiguous().numpy() if vecs else np.empty((0, self.model.config.hidden_size), dtype=np.float32)

# --------- Dense Retriever ---------
class DenseRetriever:
    def __init__(self, index_path: str, collection_path: str, offsets_path: Optional[str] = None,
                 nprobe: Optional[int] = None, model_name=MODEL_NAME, device=DEVICE, max_length=MAX_LENGTH, dtype=DTYPE,
                 in_memory: bool = False):
        self.index = faiss.read_index(index_path)
        try:
            if hasattr(self.index, "nprobe") and nprobe is not None:
                self.index.nprobe = nprobe
            elif hasattr(self.index, "index") and hasattr(self.index.index, "nprobe") and nprobe is not None:
                self.index.index.nprobe = nprobe
        except Exception: pass

        self.collection = JsonlCollection(collection_path, offsets_path=offsets_path, in_memory=in_memory)
        self.encoder = ContrieverEncoder(model_name=model_name, device=device, max_length=max_length, dtype=dtype)

    def dense_retrieve(self, query: str, k: int = 5, return_cosine: bool = True) -> Dict:
        q = self.encoder.encode([query], batch_size=1).astype(np.float32)
        D, I = self.index.search(q, k)  # L2 su vettori normalizzati
        ids = [int(x) for x in I[0]]
        dists = [float(x) for x in D[0]]
        docs = self.collection.get_many(ids)
        out = {"ids": ids, "distances_l2": dists, "documents": docs}
        if return_cosine:
            out["cosine_sim"] = [1.0 - 0.5 * d for d in dists]
        return out

    def batch_dense_retrieve(self, queries: List[str], k: int = 5, return_cosine: bool = True) -> List[Dict]:
        Q = self.encoder.encode(queries, batch_size=min(64, max(1, 8))).astype(np.float32)
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
                obj["cosine_sim"] = [1.0 - 0.5 * d for d in dists]
            results.append(obj)
        return results

# --------- CLI ---------
def main():
    ap = argparse.ArgumentParser(description="Dense retrieval su JSONL + FAISS (no docstore, no memmap).")
    ap.add_argument("--index", required=True, help="Path indice FAISS (.faiss)")
    ap.add_argument("--collection", required=True, help="Path JSONL preprocessato (id, contents)")
    ap.add_argument("--offsets", default=None, help="Offsets binari uint64 (opzionale, consigliato)")
    ap.add_argument("--query", required=True, help="Query text")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--nprobe", type=int, default=None)
    ap.add_argument("--in_memory", action="store_true", help="Carica l'intera collezione in RAM (solo mini-run)")
    args = ap.parse_args()

    retr = DenseRetriever(index_path=args.index,
                          collection_path=args.collection,
                          offsets_path=args.offsets,
                          nprobe=args.nprobe,
                          in_memory=args.in_memory)
    res = retr.dense_retrieve(args.query, k=args.k, return_cosine=True)
    print(json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()