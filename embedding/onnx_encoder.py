"""Shared CPU text encoder: gte-modernbert-base, fp32 ONNX export, run with onnxruntime.

Used by the indexer, the query path, and (via an MTEB wrapper in eval/) the apps evaluation.

Why onnxruntime directly rather than sentence-transformers' ONNX backend: that backend needs
optimum-onnx, which requires transformers<4.58 and conflicts with this environment (transformers 5.x).

Why fp32 rather than the int8 export: int8 roughly halved the gap between matched and mismatched
query/solution similarities on our sanity pairs, while fp32 matched the PyTorch model exactly.

Why max_length=1024 by default: on AppsRetrieval it truncates 2.2% of queries and costs 0.0014 NDCG@10
versus 2048, while cutting estimated CPU encode time from ~129 to ~78 min.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

DEFAULT_MODEL = "Alibaba-NLP/gte-modernbert-base"
DEFAULT_ONNX_FILE = "onnx/model.onnx"
DEFAULT_MAX_LENGTH = 1024
DEFAULT_BATCH_SIZE = 4  # fastest fp32 setting measured on CPU; also keeps peak RAM around 3 GB


class OnnxEncoder:
    """Tokenize -> onnxruntime forward -> CLS pooling -> L2 normalize.

    Matches the model's sentence-transformers configuration (CLS pooling, cosine similarity,
    no query/document prompts), so embeddings are interchangeable with the PyTorch model's.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        onnx_file: str = DEFAULT_ONNX_FILE,
        max_length: int = DEFAULT_MAX_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_threads: int | None = None,
        revision: str | None = None,
    ):
        self.model_name = model_name
        self.onnx_file = onnx_file
        self.max_length = max_length
        self.batch_size = batch_size
        self.revision = revision

        self._check_pooling_is_cls()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        if self.tokenizer.padding_side != "right":
            # CLS pooling reads position 0, which is only the [CLS] token under right padding.
            raise ValueError(f"expected right padding, tokenizer uses {self.tokenizer.padding_side!r}")

        self._onnx_path = Path(hf_hub_download(model_name, onnx_file, revision=revision))
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if num_threads is not None:
            opts.intra_op_num_threads = num_threads
        self.session = ort.InferenceSession(str(self._onnx_path), opts, providers=["CPUExecutionProvider"])
        self.dim = int(self.session.get_outputs()[0].shape[-1])
        self._onnx_sha256: str | None = None

    def _check_pooling_is_cls(self) -> None:
        with open(hf_hub_download(self.model_name, "1_Pooling/config.json", revision=self.revision)) as f:
            pooling = json.load(f)
        if not pooling.get("pooling_mode_cls_token"):
            raise ValueError(f"{self.model_name} does not use CLS pooling; OnnxEncoder only implements CLS")

    def token_lengths(self, texts: list[str], truncate: bool = True) -> list[int]:
        """Token counts including special tokens; capped at max_length when truncate=True."""
        encoded = self.tokenizer(
            texts, truncation=truncate, max_length=self.max_length if truncate else None
        )["input_ids"]
        return [len(ids) for ids in encoded]

    def encode(
        self,
        texts: list[str],
        batch_size: int | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        """Embed texts as L2-normalized float32 vectors of shape (len(texts), dim), in input order.

        progress, if given, is called as progress(done, total) after each batch.
        """
        texts = list(texts)
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        batch_size = batch_size or self.batch_size

        input_ids = self.tokenizer(texts, truncation=True, max_length=self.max_length)["input_ids"]
        # Longest first, so each batch pads to similar lengths (attention cost grows with padded length).
        order = sorted(range(len(texts)), key=lambda i: len(input_ids[i]), reverse=True)
        pad_id = self.tokenizer.pad_token_id

        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            width = len(input_ids[idx[0]])  # longest in this batch, since sorted descending
            ids = np.full((len(idx), width), pad_id, dtype=np.int64)
            mask = np.zeros((len(idx), width), dtype=np.int64)
            for row, i in enumerate(idx):
                ids[row, : len(input_ids[i])] = input_ids[i]
                mask[row, : len(input_ids[i])] = 1
            hidden = self.session.run(["last_hidden_state"], {"input_ids": ids, "attention_mask": mask})[0]
            out[idx] = hidden[:, 0]  # CLS pooling
            if progress is not None:
                progress(start + len(idx), len(texts))

        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)

    def fingerprint(self) -> dict:
        """Everything that determines the embeddings. An index built under a different fingerprint
        is not comparable with this encoder's query embeddings."""
        if self._onnx_sha256 is None:
            digest = hashlib.sha256()
            with open(self._onnx_path, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    digest.update(block)
            self._onnx_sha256 = digest.hexdigest()
        return {
            "model_name": self.model_name,
            "onnx_file": self.onnx_file,
            "onnx_sha256": self._onnx_sha256,
            "max_length": self.max_length,
            "pooling": "cls",
            "normalized": True,
            "dim": self.dim,
        }
