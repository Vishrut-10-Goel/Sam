"""Sanity-check and time gte-modernbert-base's int8 ONNX export on CPU (onnxruntime), then extrapolate
the full AppsRetrieval run time.

Uses onnxruntime directly: sentence-transformers' ONNX backend needs optimum-onnx, which requires
transformers<4.58 and so conflicts with this environment (transformers 5.x).

Draws the same random sanity pairs and timing samples as bench_model.py (same seed and call order),
so results compare directly with the PyTorch runs.

Usage: python bench_onnx.py [onnx_file] [max_seq_length] [batch_size]
"""
import functools
import os
import sys
import time

import numpy as np
import onnxruntime as ort
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

print = functools.partial(print, flush=True)  # show progress live, even when output is piped

MODEL_NAME = "Alibaba-NLP/gte-modernbert-base"
ONNX_FILE = sys.argv[1] if len(sys.argv) > 1 else "onnx/model_int8.onnx"
MAX_LEN = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
BATCH_SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 8
DATASET = "CoIR-Retrieval/apps"
SAMPLE = {"queries": 150, "corpus": 300}
N_SANITY = 20
rng = np.random.default_rng(0)


class OnnxEncoder:
    """Tokenize -> onnxruntime forward -> CLS pooling -> L2 normalize (matches the model's ST config)."""

    def __init__(self, model_name: str, onnx_file: str, max_len: int):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            hf_hub_download(model_name, onnx_file), opts, providers=["CPUExecutionProvider"]
        )

    def encode(self, texts, batch_size: int = BATCH_SIZE) -> np.ndarray:
        # Sort by length so each batch pads to similar lengths (attention cost grows with padded length).
        lengths = [len(ids) for ids in self.tokenizer(texts, truncation=True, max_length=self.max_len)["input_ids"]]
        order = np.argsort(lengths)[::-1]
        out = np.empty((len(texts), 768), dtype=np.float32)
        for start in range(0, len(texts), batch_size):
            idx = order[start:start + batch_size]
            enc = self.tokenizer(
                [texts[i] for i in idx], truncation=True, max_length=self.max_len,
                padding=True, return_tensors="np",
            )
            hidden = self.session.run(
                ["last_hidden_state"],
                {"input_ids": enc["input_ids"].astype(np.int64), "attention_mask": enc["attention_mask"].astype(np.int64)},
            )[0]
            out[idx] = hidden[:, 0]  # CLS pooling
        return out / np.linalg.norm(out, axis=1, keepdims=True)


if __name__ == "__main__":
    print(f"model: {MODEL_NAME} [{ONNX_FILE}]   max_seq_length: {MAX_LEN}   batch size: {BATCH_SIZE}")
    print(f"onnxruntime {ort.__version__} on CPU ({os.cpu_count()} logical cores)")
    t0 = time.perf_counter()
    model = OnnxEncoder(MODEL_NAME, ONNX_FILE, MAX_LEN)
    print(f"model load: {time.perf_counter() - t0:.0f} s")

    qrels = load_dataset(DATASET, "default", split="test")
    queries_ds = load_dataset(DATASET, "queries", split="queries")
    corpus_ds = load_dataset(DATASET, "corpus", split="corpus")
    test_qids = set(qrels["query-id"])
    texts = {
        "queries": [q["text"] for q in queries_ds if q["_id"] in test_qids],
        "corpus": list(corpus_ds["text"]),
    }
    query_text = dict(zip(queries_ds["_id"], queries_ds["text"]))
    doc_text = dict(zip(corpus_ds["_id"], corpus_ds["text"]))

    # Token lengths, capped at MAX_LEN, for token-weighted extrapolation.
    lengths = {
        name: np.minimum([len(ids) for ids in model.tokenizer(t, add_special_tokens=True)["input_ids"]], MAX_LEN)
        for name, t in texts.items()
    }

    # Sanity check (same pairs as bench_model.py): each query should score highest against its own solution.
    sanity_rows = qrels.select(rng.choice(len(qrels), N_SANITY, replace=False).tolist())
    q = model.encode([query_text[r["query-id"]] for r in sanity_rows])
    d = model.encode([doc_text[r["corpus-id"]] for r in sanity_rows])
    sims = q @ d.T
    print(f"sanity: mean diag sim {np.diag(sims).mean():.3f} vs off-diag {sims[~np.eye(N_SANITY, dtype=bool)].mean():.3f}; "
          f"top-1 correct {int((sims.argmax(1) == np.arange(N_SANITY)).sum())}/{N_SANITY}")

    total_est = 0.0
    for name, all_texts in texts.items():
        idx = rng.choice(len(all_texts), SAMPLE[name], replace=False)
        t0 = time.perf_counter()
        model.encode([all_texts[i] for i in idx])
        dt = time.perf_counter() - t0
        # Extrapolate by token count (sample tokens -> full-set tokens), which tracks cost better than text count.
        est = dt * lengths[name].sum() / lengths[name][idx].sum()
        total_est += est
        print(f"{name}: {SAMPLE[name]} texts ({lengths[name][idx].sum():,} tok) in {dt:.1f} s -> full set est. {est / 60:.1f} min")

    print(f"\nESTIMATED full encode time: {total_est / 60:.1f} min (+ ~1-2 min load/scoring)")
