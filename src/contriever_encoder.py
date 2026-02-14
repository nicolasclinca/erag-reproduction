"""
contriever_encoder.py
Centralized Contriever encoder for indexing and retrieval.

- Shared defaults (MODEL_NAME, MAX_LENGTH, DTYPE)
- Mean pooling + L2 normalization (float32)
- Optional tokenization prefetch for high throughput (GPU-friendly)
"""

from typing import List
import threading, queue

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel  # type: ignore


# Shared defaults
MODEL_NAME = "facebook/contriever"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 200
DTYPE = torch.float16  # used for autocast on GPU; pooling/normalization in float32

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
        device=DEVICE,
        max_length: int = MAX_LENGTH,
        dtype: torch.dtype = DTYPE,
        normalize: bool = True,
    ):
        self.device = device
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
        Returns a tensor [B, D] on the current device.
        """
        with torch.amp.autocast(device_type="cuda", dtype=self.dtype, enabled=(self.device.type == "cuda")):
            out = self.model(**inputs).last_hidden_state
        x = out.float()  # pooling and normalization in float32 for consistency
        mask = inputs["attention_mask"].to(x.dtype).unsqueeze(-1)
        mean = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        if self.normalize:
            mean = F.normalize(mean, p=2, dim=1)
        return mean  # [B, D] float32

    def encode(self, texts: List[str], batch_size: int = 64, prefetch: bool = False) -> np.ndarray:
        """
        Encodes a list of texts into normalized Contriever embeddings.
        - batch_size: batch size
        - prefetch: if True, uses tokenization prefetch (recommended for build)
        Returns a float32 np.ndarray of shape [N, D]
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