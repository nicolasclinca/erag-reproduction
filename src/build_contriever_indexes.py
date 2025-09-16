"""
build_indexes.py
Costruisce un indice OPQ+IVF-PQ (Contriever) partendo da una collezione preprocessata JSONL
(id, contents). Include un tuning iniziale di (batch_size, flush_every) su un subset di 
documenti, quindi prosegue automaticamente con la combinazione migliore.

Requisiti:
  pip install faiss-cpu transformers

Uso CLI:
  python build_indexes.py --input_jsonl ../data/collection/wikipedia_passages.jsonl \
                          --out_dir ./index_out_full \
                          --tune --tune_docs 100000 \
                          --train_size 1500000 \
                          --resume
"""

import os
import json
import time
import argparse
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import faiss
from transformers import AutoTokenizer, AutoModel


# ---------------- Config default ----------------
MODEL_NAME = "facebook/contriever"
D = 768
MAX_LENGTH = 200
DTYPE = torch.float16
SEED = 42

# Tuning (subset)
TUNE_CANDIDATES: List[Tuple[int,int]] = [
    (12, 8192), (16, 8192), (16, 4096),
    (24, 8192), (24, 4096),
    (32, 8192), (32, 4096),
    (64, 4096), (64, 8192),
]
TUNE_DOCS = 100_000  # modificabile via CLI

# Encoder knobs (valori di fallback, verranno rimpiazzati dal tuning)
BATCH_SIZE = 16
FLUSH_EVERY = 8192

# IVF-PQ knobs full-scale
NLIST = 65536
M = 64
NBITS = 8
NPROBE = 64

# Training
TRAIN_SIZE = 1_500_000
TRAIN_BLOCK_DOCS = 50_000

# Add streaming
ADD_BLOCK = 250_000

# Output
INDEX_FILENAME = "ivfpq_opq_contriever.faiss"
META_FILENAME = "index_meta.json"

# ---------------- Utils ----------------
def log(msg): print(msg, flush=True)

def set_env():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")

def set_determinism(seed=SEED):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def iter_jsonl(path: str, start_line: int = 0) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start_line: continue
            if not line.strip(): continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def read_contents_list(path: str, n: int) -> List[str]:
    out = []
    for rec in iter_jsonl(path):
        c = rec.get("contents", "").strip()
        if not c: continue
        out.append(c)
        if len(out) >= n: break
    return out

# ---------------- Encoder con prefetch + pooling efficiente ----------------
import threading, queue

class TokenizePrefetcher:
    def __init__(self, tokenizer, device, batch_size, max_length, prefetch_batches=8):
        self.tokenizer = tokenizer; self.device = device
        self.batch_size = batch_size; self.max_length = max_length
        self.q = queue.Queue(maxsize=prefetch_batches)
        self.stop = object()

    def _producer(self, docs_iter):
        batch = []
        for doc in docs_iter:
            batch.append(doc)
            if len(batch) >= self.batch_size:
                inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
                if self.device.type == "cuda":
                    inputs = {k: v.pin_memory() for k,v in inputs.items()}
                self.q.put(inputs); batch=[]
        if batch:
            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            if self.device.type == "cuda":
                inputs = {k: v.pin_memory() for k,v in inputs.items()}
            self.q.put(inputs)
        self.q.put(self.stop)

    def start(self, docs_iterable):
        t = threading.Thread(target=self._producer, args=(iter(docs_iterable),), daemon=True)
        t.start()

    def next(self):
        x = self.q.get()
        return None if x is self.stop else x

class ContrieverEncoder:
    def __init__(self, model_name=MODEL_NAME, device=None, max_length=MAX_LENGTH, dtype=DTYPE):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(self.device); self.model.eval()
        self.max_length = max_length; self.dtype = dtype
        torch.backends.cuda.matmul.allow_tf32 = True
        try: torch.set_float32_matmul_precision("high")
        except Exception: pass

    def encode_with(self, docs: List[str], batch_size: int, flush_every: int) -> np.ndarray:
        if not docs: return np.empty((0, D), dtype=np.float32)
        gpu_blocks, cpu_blocks = [], []
        pf = TokenizePrefetcher(self.tokenizer, self.device, batch_size, self.max_length,
                                prefetch_batches=8 if batch_size<=16 else 4)
        pf.start(docs)
        with torch.inference_mode():
            while True:
                inputs_cpu = pf.next()
                if inputs_cpu is None: break
                inputs = {k: v.to(self.device, non_blocking=True) if self.device.type=="cuda" else v.to(self.device)
                          for k,v in inputs_cpu.items()}
                with torch.amp.autocast(device_type='cuda', dtype=self.dtype, enabled=(self.device.type=="cuda")):
                    x = self.model(**inputs).last_hidden_state
                    mask = inputs["attention_mask"].to(x.dtype).unsqueeze(-1)
                    sum_emb = (x * mask).sum(dim=1)
                    lengths = mask.sum(dim=1).clamp(min=1e-6)
                    mean = sum_emb / lengths
                    norm = F.normalize(mean, p=2, dim=1).to(torch.float16)
                gpu_blocks.append(norm)
                if len(gpu_blocks) >= flush_every:
                    big = torch.cat(gpu_blocks, dim=0); cpu_blocks.append(big.float().cpu()); gpu_blocks=[]
        if gpu_blocks:
            big = torch.cat(gpu_blocks, dim=0); cpu_blocks.append(big.float().cpu())
        return torch.cat(cpu_blocks, dim=0).contiguous().numpy()

# ---------------- Tuning bs/flush ----------------
def bench_candidate(encoder: ContrieverEncoder, docs: List[str], bs: int, fe: int) -> Tuple[float, float, bool]:
    """Ritorna (docs_per_s, peak_gb, ok). ok=False se OOM/error."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        _ = encoder.encode_with(docs, batch_size=bs, flush_every=fe)  # embeddings scartati
        dt = time.time() - t0
        docs_s = len(docs) / max(1e-9, dt)
        peak = (torch.cuda.max_memory_allocated()/(1024**3)) if torch.cuda.is_available() else 0.0
        return docs_s, peak, True
    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return 0.0, float("inf"), False
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            return 0.0, float("inf"), False
        raise

def autotune_bs_flush(encoder: ContrieverEncoder, input_jsonl: str, tune_docs: int, candidates: List[Tuple[int,int]]) -> Tuple[int,int]:
    log(f"[tune] Sampling {tune_docs} docs for tuning...")
    sample = read_contents_list(input_jsonl, tune_docs)
    if len(sample) == 0:
        raise RuntimeError("Nessun documento nella collezione per il tuning.")
    log(f"[tune] Testing {len(candidates)} candidates...")
    rows = []
    for (bs, fe) in candidates:
        docs_s, peak, ok = bench_candidate(encoder, sample, bs, fe)
        status = "OK" if ok else "OOM"
        rows.append((docs_s, peak, ok, bs, fe))
        log(f"[tune] bs={bs:>3}, flush={fe:>5} -> {docs_s:6.1f} docs/s | peak {peak:4.2f} GB | {status}")
    ok_rows = [r for r in rows if r[2]]
    if not ok_rows:
        log("[tune] Nessuna combinazione valida. Uso fallback bs=12, flush=8192.")
        return 12, 8192
    ok_rows.sort(key=lambda r: (r[0], -r[1]), reverse=True)  # max docs/s, poi min peak
    best_docs, best_peak, _, best_bs, best_fe = ok_rows[0]
    log(f"[tune] Best -> bs={best_bs}, flush={best_fe} | {best_docs:.1f} docs/s (peak {best_peak:.2f} GB)")
    return best_bs, best_fe

# ---------------- Training set ----------------
def build_training_matrix(input_jsonl: str, encoder: ContrieverEncoder, train_size: int,
                          block_docs: int, bs: int, fe: int) -> np.ndarray:
    log(f"[train] Building training set in RAM: target={train_size:,} | bs={bs}, flush={fe}")
    blocks = []; total = 0; t0 = time.time(); buf = []
    for rec in iter_jsonl(input_jsonl):
        c = rec.get("contents", "").strip()
        if not c: continue
        buf.append(c)
        if len(buf) >= block_docs:
            xb = encoder.encode_with(buf, batch_size=bs, flush_every=fe)
            take = min(train_size - total, xb.shape[0])
            if take > 0:
                blocks.append(xb[:take].astype(np.float32, copy=False)); total += take
            buf = []
            if total >= train_size: break
            if total % 100_000 == 0:
                log(f"[train] Encoded {total:,}/{train_size:,} in {(time.time()-t0)/60:.1f} min")
    if buf and total < train_size:
        xb = encoder.encode_with(buf, batch_size=bs, flush_every=fe)
        take = min(train_size - total, xb.shape[0])
        blocks.append(xb[:take].astype(np.float32, copy=False)); total += take
    X = np.vstack(blocks) if blocks else np.empty((0, D), dtype=np.float32)
    log(f"[train] Done: {X.shape[0]:,} vectors | {(time.time()-t0)/60:.1f} min")
    return X

# ---------------- Build/Load index ----------------
def build_or_load_index(train_vectors: np.ndarray, out_dir: str, nlist: int, m: int, nbits: int, nprobe: int,
                        index_filename: str = INDEX_FILENAME, resume: bool = True) -> faiss.IndexIDMap2:
    os.makedirs(out_dir, exist_ok=True)
    index_path = os.path.join(out_dir, index_filename)
    if resume and os.path.exists(index_path):
        idx = faiss.read_index(index_path)
        if not isinstance(idx, (faiss.IndexIDMap, faiss.IndexIDMap2)):
            idx = faiss.IndexIDMap2(idx)
        try:
            if hasattr(idx, "nprobe"): idx.nprobe = nprobe
            elif hasattr(idx, "index") and hasattr(idx.index, "nprobe"): idx.index.nprobe = nprobe
        except Exception: pass
        log(f"[faiss] Loaded existing index: {index_path}")
        return idx

    log("[faiss] Building OPQ + IVF-PQ (IndexPreTransform) + IDMap2 ...")
    quantizer = faiss.IndexFlatL2(D)
    ivfpq = faiss.IndexIVFPQ(quantizer, D, nlist, m, nbits); ivfpq.nprobe = nprobe
    opq = faiss.OPQMatrix(D, m)
    pre = faiss.IndexPreTransform(opq, ivfpq)
    idx = faiss.IndexIDMap2(pre)

    log(f"[faiss] Training on {train_vectors.shape[0]:,} vectors ...")
    t0 = time.time(); idx.train(train_vectors.astype(np.float32, copy=False))
    log(f"[faiss] Training done in {(time.time()-t0)/60:.1f} min")
    faiss.write_index(idx, index_path); log(f"[save] Index saved: {index_path}")
    return idx

# ---------------- Add streaming ----------------
def add_streaming(input_jsonl: str, out_dir: str, index: faiss.IndexIDMap2,
                  encoder: ContrieverEncoder, add_block: int, index_filename: str,
                  bs: int, fe: int) -> int:
    index_path = os.path.join(out_dir, index_filename)
    try:
        next_id = index.ntotal
    except Exception:
        next_id = 0
    log(f"[add] Resume at next_id={next_id:,} | bs={bs}, flush={fe}")

    total_added = 0; t0 = time.time(); buf = []
    it = iter_jsonl(input_jsonl, start_line=next_id)
    for rec in it:
        c = rec.get("contents", "").strip()
        if not c: continue
        buf.append(c)
        if len(buf) >= add_block:
            xb = encoder.encode_with(buf, batch_size=bs, flush_every=fe)
            ids = (np.arange(len(buf), dtype=np.int64) + next_id)
            index.add_with_ids(np.ascontiguousarray(xb, dtype=np.float32), ids)
            next_id += len(buf); total_added += len(buf); buf = []
            faiss.write_index(index, index_path)
            elapsed = time.time() - t0; rate = total_added / max(1e-9, elapsed)
            log(f"[add] Added {total_added:,} | {rate:.1f} docs/s | elapsed {elapsed/60:.1f} min")
    if buf:
        xb = encoder.encode_with(buf, batch_size=bs, flush_every=fe)
        ids = (np.arange(len(buf), dtype=np.int64) + next_id)
        index.add_with_ids(np.ascontiguousarray(xb, dtype=np.float32), ids)
        total_added += len(buf); next_id += len(buf)
        faiss.write_index(index, index_path)

    elapsed = time.time() - t0; rate = total_added / max(1e-9, elapsed)
    log(f"[add] Done. Added {total_added:,} | {rate:.1f} docs/s | elapsed {elapsed/60:.1f} min")
    return total_added

# ---------------- Meta ----------------
def write_meta(out_dir: str, input_jsonl: str, index_filename: str, best_bs: int, best_fe: int):
    meta = {
        "collection_path": os.path.abspath(input_jsonl),
        "index_path": os.path.abspath(os.path.join(out_dir, index_filename)),
        "tuned_batch_size": best_bs,
        "tuned_flush_every": best_fe,
        "d": D, "m": M, "nbits": NBITS, "nlist": NLIST, "nprobe": NPROBE,
        "encoder": MODEL_NAME, "max_length": MAX_LENGTH,
    }
    with open(os.path.join(out_dir, META_FILENAME), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Build OPQ+IVF-PQ index from preprocessed JSONL, with bs/flush tuning.")
    parser.add_argument("--input_jsonl", required=True, help="File JSONL preprocessato (id, contents).")
    parser.add_argument("--out_dir", type=str, default="./index_out_full", help="Directory output (indice + meta).")
    parser.add_argument("--tune", action="store_true", help="Esegui tuning bs/flush su un subset prima del build.")
    parser.add_argument("--tune_docs", type=int, default=TUNE_DOCS, help="#docs per tuning.")
    parser.add_argument("--train_size", type=int, default=TRAIN_SIZE)
    parser.add_argument("--train_block_docs", type=int, default=TRAIN_BLOCK_DOCS)
    parser.add_argument("--add_block", type=int, default=ADD_BLOCK)
    parser.add_argument("--nlist", type=int, default=NLIST)
    parser.add_argument("--nprobe", type=int, default=NPROBE)
    parser.add_argument("--m", type=int, default=M)
    parser.add_argument("--nbits", type=int, default=NBITS)
    parser.add_argument("--resume", action="store_true")
    # fallback manuale se si vuole saltare il tuning:
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--flush_every", type=int, default=FLUSH_EVERY)
    args = parser.parse_args()

    set_env(); set_determinism(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[env] device={device} | out_dir={args.out_dir}")

    # Encoder (unico, riusato per tuning/train/add)
    encoder = ContrieverEncoder(MODEL_NAME, device, MAX_LENGTH, DTYPE)

    # 1) Tuning bs/flush (opzionale)
    best_bs, best_fe = args.batch_size, args.flush_every
    if args.tune:
        best_bs, best_fe = autotune_bs_flush(encoder, args.input_jsonl, tune_docs=args.tune_docs, candidates=TUNE_CANDIDATES)

    # 2) Training set in RAM con la migliore combinazione
    X_train = build_training_matrix(args.input_jsonl, encoder, args.train_size,
                                    block_docs=args.train_block_docs, bs=best_bs, fe=best_fe)

    # 3) Build/Load index
    index = build_or_load_index(X_train, args.out_dir, args.nlist, args.m, args.nbits, args.nprobe,
                                index_filename=INDEX_FILENAME, resume=args.resume)

    # 4) FAISS threads per add
    try:
        faiss.omp_set_num_threads(os.cpu_count() or 4)
        log(f"[faiss] Using {os.cpu_count()} CPU threads")
    except Exception:
        pass

    # 5) Add streaming con la combinazione migliore
    added = add_streaming(args.input_jsonl, args.out_dir, index, encoder, args.add_block, INDEX_FILENAME,
                          bs=best_bs, fe=best_fe)
    log(f"[done] ntotal={index.ntotal:,} | added_now={added:,}")
    faiss.write_index(index, os.path.join(args.out_dir, INDEX_FILENAME))

    # 6) Meta
    write_meta(args.out_dir, args.input_jsonl, INDEX_FILENAME, best_bs, best_fe)
    log("[save] Meta written.")

if __name__ == "__main__":
    main()