"""
query_encoders.py

Indexes created with:
- BAAI/bge-base-en-v1.5
- castorini/tct_colbert-v2-hnp-msmarco
- facebook/dpr-ctx_encoder-multiset-base

For DPR queries we use:
- facebook/dpr-question_encoder-multiset-base
"""

from __future__ import annotations

from typing import List, Optional, Literal, Union, Type

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, DPRQuestionEncoder  # type: ignore
from transformers import BertModel, BertTokenizer  # type: ignore


EncoderType = Literal["dpr", "bge", "tct"]

MODEL_BGE = "BAAI/bge-base-en-v1.5"
MODEL_TCT = "castorini/tct_colbert-v2-hnp-msmarco"
MODEL_DPR_Q = "facebook/dpr-question_encoder-multiset-base"


def _as_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device if isinstance(device, torch.device) else torch.device(device)


VectorSource = Literal["cls", "pooler"]


class HFQueryEncoder:
    """
    Generic HF encoder:
    - loads tokenizer + model (configurable class)
    - batching + tokenization
    - forward
    - extracts embeddings with:
        - vector_source="cls": last_hidden_state[:,0,:]
        - vector_source="pooler": pooler_output
    - optional L2 normalize
    - optional query_prefix

    Exposes:
      - .D
      - .encode(List[str], batch_size) -> np.ndarray float32 (B, D)
    """

    def __init__(
        self,
        model_name: str,
        model_cls: Type[torch.nn.Module],
        vector_source: VectorSource,
        device: Optional[Union[str, torch.device]] = None,
        max_length: int = 256,
        normalize: bool = False,
        query_prefix: str = "",
        amp_dtype: torch.dtype = torch.float16,
    ):
        self.device = _as_device(device)
        self.max_length = int(max_length)
        self.normalize = bool(normalize)
        self.query_prefix = query_prefix or ""
        self.vector_source = vector_source
        self.amp_dtype = amp_dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = model_cls.from_pretrained(model_name).to(self.device)  # type: ignore[attr-defined]
        self.model.eval()

        # perf knobs
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        # infer dim robustly
        self.D = int(getattr(getattr(self.model, "config", None), "hidden_size", 0) or 0)
        if self.D <= 0:
            with torch.inference_mode():
                dummy = self.tokenizer(["hello"], return_tensors="pt", padding=True, truncation=True)
                dummy = {k: v.to(self.device) for k, v in dummy.items()}
                out = self.model(**dummy)
                if self.vector_source == "pooler":
                    x = out.pooler_output
                else:
                    x = out.last_hidden_state[:, 0, :]
                self.D = int(x.shape[-1])

    def _extract(self, out) -> torch.Tensor:
        if self.vector_source == "pooler":
            return out.pooler_output  # (B, D)
        # cls
        return out.last_hidden_state[:, 0, :]  # (B, D)

    @torch.inference_mode()
    def encode(self, queries: List[str], batch_size: int = 32) -> np.ndarray:
        if not queries:
            return np.empty((0, self.D), dtype=np.float32)

        bs = max(1, int(batch_size))
        out_chunks: List[np.ndarray] = []

        texts = [(self.query_prefix + q) for q in queries]

        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            if self.device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
                    out = self.model(**enc)
            else:
                out = self.model(**enc)

            x = self._extract(out).float()  # float32 for stability
            if self.normalize:
                x = F.normalize(x, p=2, dim=1)

            out_chunks.append(x.cpu().numpy().astype(np.float32, copy=False))

        return np.vstack(out_chunks)


class TctColBertPyseriniQueryEncoder:
    """
    Replica of the PySerini logic (TctColBertQueryEncoder):
      - input: "[CLS] [Q] " + query + "[MASK]" * 36
      - tokenizer: add_special_tokens=False, truncation=True, max_length=36
      - embedding: mean(outputs.last_hidden_state[:, 4:, :], dim=1)
      - NO L2 normalization
    """

    def __init__(
        self,
        model_name: str = MODEL_TCT,
        device: Optional[Union[str, torch.device]] = None,
        amp_dtype: torch.dtype = torch.float16,
    ):
        self.device = _as_device(device)
        self.amp_dtype = amp_dtype

        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(
            model_name, clean_up_tokenization_spaces=True
        )
        self.D = int(self.model.config.hidden_size)

        self.max_length = 36

        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    @torch.inference_mode()
    def encode(self, queries: List[str], batch_size: int = 32) -> np.ndarray:
        if not queries:
            return np.empty((0, self.D), dtype=np.float32)

        bs = max(1, int(batch_size))
        out_chunks: List[np.ndarray] = []

        mask_suffix = "[MASK]" * self.max_length

        for i in range(0, len(queries), bs):
            batch_q = queries[i : i + bs]
            texts = [("[CLS] [Q] " + q + mask_suffix) for q in batch_q]

            enc = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            if self.device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
                    out = self.model(**enc)
            else:
                out = self.model(**enc)

            x = out.last_hidden_state[:, 4:, :].float().mean(dim=1)  # (B, D)
            out_chunks.append(x.cpu().numpy().astype(np.float32, copy=False))

        return np.vstack(out_chunks)


def build_query_encoder(
    encoder_type: EncoderType,
    device: Optional[Union[str, torch.device]] = None,
    max_length: int = 256,
):
    et = str(encoder_type).lower().strip()
    if et not in ("dpr", "bge", "tct"):
        raise ValueError("encoder_type must be one of: dpr, bge, tct")

    if et == "dpr":
        return HFQueryEncoder(
            model_name=MODEL_DPR_Q,
            model_cls=DPRQuestionEncoder,
            vector_source="pooler",
            device=device,
            max_length=max_length,
            normalize=False,
            query_prefix="",
        )

    if et == "bge":
        return HFQueryEncoder(
            model_name=MODEL_BGE,
            model_cls=AutoModel,
            vector_source="cls",
            device=device,
            max_length=max_length,
            normalize=True,
            query_prefix="",
        )

    # tct
    return TctColBertPyseriniQueryEncoder(
        model_name=MODEL_TCT,
        device=device,
    )