"""
build_contriever_indexes.py
Costruisce un indice OPQ+IVF-PQ (Contriever) partendo da una collezione preprocessata JSONL
(id, contents). Include un tuning iniziale di batch_size su un subset di documenti, quindi 
prosegue automaticamente con la combinazione migliore.

Requisiti:
pip install torch faiss-cpu transformers

Uso CLI:
python build_contriever_indexes.py --collection ../data/collection/wikipedia_passages.jsonl \
    --faiss_index_dir ./index_out_full \
    --tune --tune_docs 100000 \
    --train_size 3000000 \
    --resume
"""

import os
import json
import time
import argparse
from typing import Iterator, List

import numpy as np
import torch
import faiss

from contriever_encoder import (ContrieverEncoder, MODEL_NAME, MAX_LENGTH, DTYPE)


# ---------------- Config default ----------------
SEED = 42

# Tuning (subset)
TUNE_CANDIDATES: List[int] = [32, 64, 80, 96, 112, 128, 144]
TUNE_DOCS = 100_000  # modificabile via CLI

# Encoder knobs (valori di fallback, verranno rimpiazzati dal tuning)
BATCH_SIZE = 16

# IVF-PQ knobs full-scale
NLIST = 65536
M = 64
NBITS = 8
NPROBE = 64

# Training
TRAIN_SIZE = 3_000_000
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


# ---------------- Tuning bs ----------------
def autotune_batch_size(encoder: ContrieverEncoder, collection: str, tune_docs: int, candidates: List[int]) -> int:
    log(f"[tune] Sampling {tune_docs} docs for tuning...")
    sample = read_contents_list(collection, tune_docs)
    if len(sample) == 0:
        raise RuntimeError("Nessun documento nella collezione per il tuning.")
    log(f"[tune] Testing {len(candidates)} candidates...")

    rows = []
    for bs in candidates:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            _ = encoder.encode(sample, batch_size=bs, prefetch=True)  # embeddings scartati
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.time() - t0
            docs_s = len(sample) / max(1e-9, dt)
            peak = (torch.cuda.max_memory_allocated() / (1024**3)) if torch.cuda.is_available() else 0.0
            ok = True
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            docs_s, peak, ok = 0.0, float("inf"), False
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                docs_s, peak, ok = 0.0, float("inf"), False
            else:
                raise

        status = "OK" if ok else "OOM"
        rows.append((docs_s, peak, ok, bs))
        log(f"[tune] bs={bs:>3} -> {docs_s:6.1f} docs/s | peak {peak:4.2f} GB | {status}")

    ok_rows = [r for r in rows if r[2]]
    if not ok_rows:
        fallback = 64 if 64 in candidates else min(candidates)
        log(f"[tune] Nessuna combinazione valida. Uso fallback bs={fallback}.")
        return fallback

    # Seleziona: max docs/s, a parità di throughput preferisci minor peak memory
    best_docs, best_peak, _, best_bs = max(ok_rows, key=lambda r: (r[0], -r[1]))
    log(f"[tune] Best -> bs={best_bs} | {best_docs:.1f} docs/s (peak {best_peak:.2f} GB)")
    return best_bs

# ---------------- Training set ----------------
def build_training_matrix(collection: str, encoder: ContrieverEncoder, train_size: int,
                          block_docs: int, bs: int) -> np.ndarray:
    log(f"[train] Building training set in RAM: target={train_size:,} | bs={bs}")
    X = np.empty((train_size, encoder.D), dtype=np.float32)
    total, t0, buf = 0, time.time(), []
    for rec in iter_jsonl(collection):
        c = rec.get("contents", "").strip()
        if not c: 
            continue
        buf.append(c)
        if len(buf) >= block_docs:
            xb = encoder.encode(buf, batch_size=bs, prefetch=True)
            take = min(train_size - total, xb.shape[0])
            if take > 0:
                X[total:total+take] = xb[:take]
                total += take
            buf = []
            if total >= train_size:
                break
            if total and total % 100_000 == 0:
                log(f"[train] Encoded {total:,}/{train_size:,} in {(time.time()-t0)/60:.1f} min")
    if buf and total < train_size:
        xb = encoder.encode(buf, batch_size=bs, prefetch=True)
        take = min(train_size - total, xb.shape[0])
        X[total:total+take] = xb[:take]
        total += take
    X = X[:total]
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
            if hasattr(idx, "nprobe"):
                idx.nprobe = nprobe
            elif hasattr(idx, "index"):
                inner = idx.index
                if hasattr(inner, "nprobe"):
                    inner.nprobe = nprobe
                elif hasattr(inner, "index") and hasattr(inner.index, "nprobe"):
                    inner.index.nprobe = nprobe
        except Exception:
            pass
        log(f"[faiss] Loaded existing index: {index_path}")
        return idx

    log("[faiss] Building OPQ + IVF-PQ (IndexPreTransform) + IDMap2 ...")
    d = int(train_vectors.shape[1])
    quantizer = faiss.IndexFlatL2(d)
    ivfpq = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits); ivfpq.nprobe = nprobe
    opq = faiss.OPQMatrix(d, m)
    pre = faiss.IndexPreTransform(opq, ivfpq)
    idx = faiss.IndexIDMap2(pre)

    log(f"[faiss] Training on {train_vectors.shape[0]:,} vectors ...")
    t0 = time.time(); idx.train(train_vectors.astype(np.float32, copy=False))
    log(f"[faiss] Training done in {(time.time()-t0)/60:.1f} min")
    faiss.write_index(idx, index_path); log(f"[save] Index saved: {index_path}")
    return idx

# ---------------- Helpers ----------------
def _progress_path(out_dir: str, fname: str = "add_progress.json") -> str:
    return os.path.join(out_dir, fname)

def _load_progress(out_dir: str) -> dict:
    p = _progress_path(out_dir)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"last_line": -1, "ntotal": 0}

def _save_progress(out_dir: str, last_line: int, ntotal: int) -> None:
    p = _progress_path(out_dir)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"last_line": int(last_line), "ntotal": int(ntotal)}, f)
    os.replace(tmp, p)

# ---------------- Add streaming ----------------
def add_streaming(collection: str, out_dir: str, index: faiss.IndexIDMap2,
                  encoder: ContrieverEncoder, add_block: int, index_filename: str,
                  bs: int, checkpoint_every: int = 0, resume: bool = True) -> int:
    """
    Aggiunge documenti all'indice in streaming usando come ID FAISS l'indice di riga (0-based)
    del file JSONL.

    Resume: se resume=True, riparte dalla riga last_line+1 salvata in add_progress.json.
    Se resume=True ma il progress file non esiste e l'indice contiene già vettori,
    viene sollevata un'eccezione per evitare duplicati.
    """
    index_path = os.path.join(out_dir, index_filename)
    os.makedirs(out_dir, exist_ok=True)

    prog_path = _progress_path(out_dir)
    prog_exists = os.path.exists(prog_path)

    # Determina start_line in modo robusto
    if resume:
        if prog_exists:
            prog = _load_progress(out_dir)
            start_line = int(prog.get("last_line", -1)) + 1
        else:
            # Niente progress file: se l'indice non è vuoto, interrompi per evitare duplicati
            if getattr(index, "ntotal", 0) > 0:
                raise RuntimeError(
                    "Resume richiesto ma non esiste alcun progress file. "
                    "Per riprendere in sicurezza usa il progress file oppure cancella l'indice "
                    "o lancia con --resume=False per ricostruire da zero."
                )
            start_line = 0
    else:
        # Non si supporta l'append senza progress: se l'indice non è vuoto, interrompi
        if getattr(index, "ntotal", 0) > 0:
            raise RuntimeError(
                "Indice non vuoto e resume=False: per evitare duplicati interrompo. "
                "Usa --resume o cancella l'indice e ricostruisci."
            )
        start_line = 0

    log(f"[add] Start from line={start_line} | bs={bs} | ckpt_every={checkpoint_every} | resume={resume}")

    total_added = 0
    t0 = time.time()
    buf_docs: List[str] = []
    buf_ids: List[int] = []
    block_idx = 0
    last_added_line = start_line - 1

    with open(collection, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start_line:
                continue
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            c = rec.get("contents", "")
            if not isinstance(c, str):
                c = str(c) if c is not None else ""
            c = c.strip()
            if not c:
                continue

            # ID FAISS = indice di riga
            rid = int(i)
            buf_docs.append(c)
            buf_ids.append(rid)
            last_added_line = i

            if len(buf_docs) >= add_block:
                xb = encoder.encode(buf_docs, batch_size=bs, prefetch=True)
                n = xb.shape[0]
                if n:
                    ids = np.asarray(buf_ids[:n], dtype=np.int64)
                    index.add_with_ids(np.ascontiguousarray(xb, dtype=np.float32), ids)
                    total_added += n
                    block_idx += 1
                    # checkpoint opzionale
                    if checkpoint_every > 0 and (block_idx % checkpoint_every == 0):
                        faiss.write_index(index, index_path)
                        _save_progress(out_dir, last_added_line, int(index.ntotal))
                        elapsed = time.time() - t0
                        rate = total_added / max(1e-9, elapsed)
                        log(f"[add][ckpt] Block {block_idx} | Added {total_added:,} | {rate:.1f} docs/s | saved -> {index_path}")
                buf_docs.clear()
                buf_ids.clear()

                elapsed = time.time() - t0
                rate = total_added / max(1e-9, elapsed)
                log(f"[add] Added {total_added:,} | {rate:.1f} docs/s | elapsed {elapsed/60:.1f} min")

    # Flush finale
    if buf_docs:
        xb = encoder.encode(buf_docs, batch_size=bs, prefetch=True)
        n = xb.shape[0]
        if n:
            ids = np.asarray(buf_ids[:n], dtype=np.int64)
            index.add_with_ids(np.ascontiguousarray(xb, dtype=np.float32), ids)
            total_added += n
            block_idx += 1
            if checkpoint_every > 0 and (block_idx % checkpoint_every == 0):
                faiss.write_index(index, index_path)
                _save_progress(out_dir, last_added_line, int(index.ntotal))
                elapsed = time.time() - t0
                rate = total_added / max(1e-9, elapsed)
                log(f"[add][ckpt] Block {block_idx} | Added {total_added:,} | {rate:.1f} docs/s | saved -> {index_path}")

    # Salvataggio finale su disco
    faiss.write_index(index, index_path)
    _save_progress(out_dir, last_added_line, int(index.ntotal))
    elapsed = time.time() - t0
    rate = total_added / max(1e-9, elapsed)
    log(f"[add] Done. Added {total_added:,} | {rate:.1f} docs/s | elapsed {elapsed/60:.1f} min | last_line={last_added_line}")
    return total_added

# ---------------- Meta ----------------
def write_meta(out_dir: str, collection: str, index_filename: str, best_bs: int, d: int,
               nlist: int, nprobe: int, m: int, nbits: int, meta_filename: str = META_FILENAME):
    meta = {
        "collection_path": os.path.abspath(collection),
        "index_path": os.path.abspath(os.path.join(out_dir, index_filename)),
        "tuned_batch_size": best_bs,
        "m": m, "nbits": nbits, "nlist": nlist, "nprobe": nprobe,
        "dim": d, "encoder": MODEL_NAME, "max_length": MAX_LENGTH,
        "id_scheme": "line_index",
        "progress_file": os.path.abspath(_progress_path(out_dir)),
    }
    with open(os.path.join(out_dir, meta_filename), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def get_index_params_from_faiss(index: faiss.Index) -> dict:
    """
    Estrae nlist, nprobe, M, nbits dal vero indice FAISS anche quando è wrappato
    da IDMap2 e/o IndexPreTransform (OPQ).
    """
    core = index
    # Unwrap IDMap
    if isinstance(core, (faiss.IndexIDMap, faiss.IndexIDMap2)) and hasattr(core, "index"):
        core = core.index
    # Unwrap PreTransform (OPQ + IVF-PQ)
    if isinstance(core, faiss.IndexPreTransform) and hasattr(core, "index"):
        core = core.index

    params = {
        "nlist": int(getattr(core, "nlist", 0)),
        "nprobe": int(getattr(core, "nprobe", 0)),
        "m": None,
        "nbits": None,
    }
    # IVFPQ espone .pq con M e nbits
    if hasattr(core, "pq"):
        params["m"] = int(getattr(core.pq, "M", 0))
        params["nbits"] = int(getattr(core.pq, "nbits", 0))
    elif hasattr(core, "M"):
        params["m"] = int(getattr(core, "M", 0))
    return params

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Build OPQ+IVF-PQ index from preprocessed JSONL, with bs tuning.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--collection", required=True, help="File JSONL preprocessato (id, contents).")
    parser.add_argument("--faiss_index_dir", type=str, default="./index_out_full", 
                        help="Directory output (indice + meta).")
    parser.add_argument("--index_filename", type=str, default=INDEX_FILENAME,
                        help="Nome del file indice FAISS salvato in faiss_index_dir.")
    parser.add_argument("--meta_filename", type=str, default=META_FILENAME,
                        help="Nome del file metadata JSON salvato in faiss_index_dir.")
    parser.add_argument("--tune", action="store_true", 
                        help="Esegui tuning bs su un subset prima del build.")
    parser.add_argument("--tune_docs", type=int, default=TUNE_DOCS, help="#docs per tuning.")
    parser.add_argument("--train_size", type=int, default=TRAIN_SIZE)
    parser.add_argument("--train_block_docs", type=int, default=TRAIN_BLOCK_DOCS)
    parser.add_argument("--add_block", type=int, default=ADD_BLOCK)
    parser.add_argument("--nlist", type=int, default=NLIST)
    parser.add_argument("--nprobe", type=int, default=NPROBE)
    parser.add_argument("--m", type=int, default=M)
    parser.add_argument("--nbits", type=int, default=NBITS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint_every", type=int, default=0,
                        help="Scrivi un checkpoint su disco ogni N blocchi. 0 = solo al termine.")
    # fallback manuale se si vuole saltare il tuning:
    parser.add_argument("--encode_batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    set_env(); set_determinism(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[env] device={device} | out_dir={args.faiss_index_dir}")

    # Encoder (unico, riusato per tuning/train/add)
    encoder = ContrieverEncoder(MODEL_NAME, device, MAX_LENGTH, DTYPE)

    # 1) FAISS threads per add
    try:
        faiss.omp_set_num_threads(os.cpu_count() or 4)
        log(f"[faiss] Using {os.cpu_count()} CPU threads")
    except Exception:
        pass

    # 2) Tuning bs (opzionale)
    best_bs = args.encode_batch_size
    if args.tune:
        best_bs = autotune_batch_size(encoder, args.collection, tune_docs=args.tune_docs, candidates=TUNE_CANDIDATES)

    # 3) Costruisci o carica indice
    index_path = os.path.join(args.faiss_index_dir, args.index_filename)
    index_exists = args.resume and os.path.exists(index_path)

    if index_exists:
        # Carica direttamente l’indice
        index = build_or_load_index(np.empty((0, encoder.D), dtype=np.float32),
                                    args.faiss_index_dir, args.nlist, args.m, args.nbits, args.nprobe,
                                    index_filename=args.index_filename, resume=True)
    else:
        # Costruisci training set e indice da zero
        X_train = build_training_matrix(args.collection, encoder, args.train_size,
                                        block_docs=args.train_block_docs, bs=best_bs)
        index = build_or_load_index(X_train, args.faiss_index_dir, args.nlist, args.m, args.nbits,
                                    args.nprobe, index_filename=args.index_filename, resume=args.resume)

    # 4) Add streaming con checkpoint opzionali (IDs = line index)
    added = add_streaming(args.collection, args.faiss_index_dir, index, encoder, args.add_block,
                          args.index_filename, bs=best_bs, checkpoint_every=args.checkpoint_every, 
                          resume=args.resume)
    log(f"[done] ntotal={index.ntotal:,} | added_now={added:,}")

    # 5) Meta
    idx_params = get_index_params_from_faiss(index)
    write_meta(args.faiss_index_dir, args.collection, args.index_filename, best_bs, encoder.D,
               nlist=idx_params["nlist"], nprobe=idx_params["nprobe"],
               m=idx_params["m"], nbits=idx_params["nbits"],
               meta_filename=args.meta_filename)
    
if __name__ == "__main__":
    main()