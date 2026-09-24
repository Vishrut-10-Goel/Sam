"""Sanity-check and time an embedding model on AppsRetrieval samples, then extrapolate the full-run time.

Runs on the GPU (fp16) when the environment's PyTorch can see one (venv-gpu), otherwise on CPU (venv).

Usage: python bench_model.py <model_name> [max_seq_length] [batch_size]
"""
import functools
import sys
import time

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

print = functools.partial(print, flush=True)  # show progress live, even when output is piped

MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "Alibaba-NLP/gte-modernbert-base"
MAX_LEN = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 4 GB VRAM (GTX 1650): small batches keep 2048-token inputs within memory.
BATCH_SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else (8 if DEVICE == "cuda" else 16)
DATASET = "CoIR-Retrieval/apps"
SAMPLE = {"queries": 150, "corpus": 300}
N_SANITY = 20
rng = np.random.default_rng(0)

if DEVICE == "cuda":
    print(f"device: cuda ({torch.cuda.get_device_name(0)}, fp16)   batch size: {BATCH_SIZE}")
else:
    print(f"device: cpu (fp32, {torch.get_num_threads()} threads)   batch size: {BATCH_SIZE}")
print(f"model: {MODEL_NAME}   max_seq_length: {MAX_LEN}")
t0 = time.perf_counter()
model = SentenceTransformer(MODEL_NAME, device=DEVICE)
if DEVICE == "cuda":
    model.half()  # Turing has fast fp16 but no bf16; also halves VRAM use
model.max_seq_length = MAX_LEN
print(f"model load: {time.perf_counter() - t0:.0f} s")
print(f"modules: {model}")

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

# Token lengths under this model's own tokenizer, capped at MAX_LEN, for token-weighted extrapolation.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
lengths = {
    name: np.minimum([len(ids) for ids in tokenizer(t, add_special_tokens=True)["input_ids"]], MAX_LEN)
    for name, t in texts.items()
}
print(f"longest test query: {lengths['queries'].max():,} tokens (limit {MAX_LEN})")

# Sanity check: each query should score highest against its own labelled solution among N_SANITY candidates.
sanity_rows = qrels.select(rng.choice(len(qrels), N_SANITY, replace=False).tolist())
q = model.encode([query_text[r["query-id"]] for r in sanity_rows], batch_size=BATCH_SIZE, normalize_embeddings=True)
d = model.encode([doc_text[r["corpus-id"]] for r in sanity_rows], batch_size=BATCH_SIZE, normalize_embeddings=True)
sims = q @ d.T
print(f"sanity: mean diag sim {np.diag(sims).mean():.3f} vs off-diag {sims[~np.eye(N_SANITY, dtype=bool)].mean():.3f}; "
      f"top-1 correct {int((sims.argmax(1) == np.arange(N_SANITY)).sum())}/{N_SANITY}")

total_est = 0.0
for name, all_texts in texts.items():
    idx = rng.choice(len(all_texts), SAMPLE[name], replace=False)
    t0 = time.perf_counter()
    model.encode([all_texts[i] for i in idx], batch_size=BATCH_SIZE)  # returns numpy, so GPU work is complete
    dt = time.perf_counter() - t0
    # Extrapolate by token count (sample tokens -> full-set tokens), which tracks cost better than text count.
    est = dt * lengths[name].sum() / lengths[name][idx].sum()
    total_est += est
    print(f"{name}: {SAMPLE[name]} texts ({lengths[name][idx].sum():,} tok) in {dt:.1f} s -> full set est. {est / 60:.1f} min")

print(f"\nESTIMATED full encode time: {total_est / 60:.1f} min (+ ~1-2 min load/scoring)")
if DEVICE == "cuda":
    print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB of "
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
