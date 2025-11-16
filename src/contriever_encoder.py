"""
contriever_encoder.py
Encoder Contriever centralizzato per indicizzazione e retrieval.

- Default condivisi (MODEL_NAME, MAX_LENGTH, DTYPE)
- Mean pooling + L2 normalization (float32)
- Prefetch tokenization opzionale per throughput elevato (GPU-friendly)
"""

from typing import List, Optional
import threading, queue

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# Default condivisi
MODEL_NAME = "facebook/contriever"
MAX_LENGTH = 200
DTYPE = torch.float16  # usato per autocast su GPU; pooling/normalizzazione in float32

class TokenizePrefetcher:
    def __init__(self, tokenizer, device, batch_size, max_length, prefetch_batches=8):
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.q = queue.Queue(maxsize=prefetch_batches)
        self._stop = object()

    def _producer(self, docs_iter):
        batch = []
        for doc in docs_iter:
            batch.append(doc)
            if len(batch) >= self.batch_size:
                inputs = self.tokenizer(
                    batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
                )
                if self.device.type == "cuda":
                    inputs = {k: v.pin_memory() for k, v in inputs.items()}
                self.q.put(inputs)
                batch = []
        if batch:
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
            )
            if self.device.type == "cuda":
                inputs = {k: v.pin_memory() for k, v in inputs.items()}
            self.q.put(inputs)
        self.q.put(self._stop)

    def start(self, docs_iterable):
        t = threading.Thread(target=self._producer, args=(iter(docs_iterable),), daemon=True)
        t.start()

    def next(self):
        x = self.q.get()
        return None if x is self._stop else x


class ContrieverEncoder:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[torch.device] = None,
        max_length: int = MAX_LENGTH,
        dtype: torch.dtype = DTYPE,
        normalize: bool = True,
    ):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.max_length = max_length
        self.dtype = dtype
        self.normalize = normalize

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.D = int(self.model.config.hidden_size)

        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    def _forward_and_pool(self, inputs: dict) -> torch.Tensor:
        """
        Forward + mean pooling + L2 normalize (in float32).
        Ritorna tensore [B, D] su device corrente.
        """
        with torch.amp.autocast(device_type="cuda", dtype=self.dtype, enabled=(self.device.type == "cuda")):
            out = self.model(**inputs).last_hidden_state
        x = out.float()  # pooling e normalizzazione in float32 per consistenza
        mask = inputs["attention_mask"].to(x.dtype).unsqueeze(-1)
        mean = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        if self.normalize:
            mean = F.normalize(mean, p=2, dim=1)
        return mean  # [B, D] float32

    def encode(self, texts: List[str], batch_size: int = 64, prefetch: bool = False) -> np.ndarray:
        """
        Encoda una lista di testi in embedding Contriever normalizzati.
        - batch_size: dimensione batch
        - prefetch: se True, usa prefetch tokenization (consigliato per build)
        Ritorna np.ndarray float32 di shape [N, D]
        """
        if not texts:
            return np.empty((0, self.D), dtype=np.float32)

        vecs: List[np.ndarray] = []

        if prefetch:
            pf = TokenizePrefetcher(
                self.tokenizer, self.device, batch_size, self.max_length,
                prefetch_batches=8 if batch_size <= 16 else 4,
            )
            pf.start(texts)
            with torch.inference_mode():
                while True:
                    inputs_cpu = pf.next()
                    if inputs_cpu is None:
                        break
                    inputs = {
                        k: v.to(self.device, non_blocking=(self.device.type == "cuda"))
                        for k, v in inputs_cpu.items()
                    }
                    pooled = self._forward_and_pool(inputs)       # [B, D] float32 (device)
                    vecs.append(pooled.cpu().numpy().astype(np.float32, copy=False))
        else:
            with torch.inference_mode():
                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]
                    inputs = self.tokenizer(
                        batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
                    )
                    inputs = {
                        k: v.to(self.device, non_blocking=(self.device.type == "cuda"))
                        for k, v in inputs.items()
                    }
                    pooled = self._forward_and_pool(inputs)       # [B, D] float32 (device)
                    vecs.append(pooled.cpu().numpy().astype(np.float32, copy=False))

        return np.concatenate(vecs, axis=0) if vecs else np.empty((0, self.D), dtype=np.float32)
    