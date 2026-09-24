"""Time jina-embeddings-v2-base-code on a random sample and extrapolate to the full AppsRetrieval run."""
import time

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

import jina_compat  # noqa: F401  (must run before the model loads)

MODEL_NAME = "jinaai/jina-embeddings-v2-base-code"
DATASET = "CoIR-Retrieval/apps"
MAX_LEN = 2048
SAMPLE = {"queries": 150, "corpus": 300}
rng = np.random.default_rng(0)

print(f"torch threads: {torch.get_num_threads()}")
t0 = time.perf_counter()
model = SentenceTransformer(MODEL_NAME, device="cpu", trust_remote_code=True)
model.max_seq_length = MAX_LEN
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
pairs = [(query_text[r["query-id"]], doc_text[r["corpus-id"]]) for r in qrels.select(range(20))]

# Warm-up so one-time setup cost doesn't skew the timing.
model.encode(texts["queries"][:8], batch_size=8)

# Sanity check that the model loaded correctly under the shim: each query should score
# highest against its own labelled solution among a small candidate set.
q = model.encode([p[0] for p in pairs], normalize_embeddings=True)
d = model.encode([p[1] for p in pairs], normalize_embeddings=True)
sims = q @ d.T
print(f"sanity: mean diag sim {np.diag(sims).mean():.3f} vs off-diag {sims[~np.eye(20, dtype=bool)].mean():.3f}; "
      f"top-1 correct {int((sims.argmax(1) == np.arange(20)).sum())}/20")

total_est = 0.0
for name, all_texts in texts.items():
    lengths = np.minimum(np.load(f"lengths_{name}.npy"), MAX_LEN)
    idx = rng.choice(len(all_texts), SAMPLE[name], replace=False)
    t0 = time.perf_counter()
    model.encode([all_texts[i] for i in idx], batch_size=16)
    dt = time.perf_counter() - t0
    # Extrapolate by token count (sample tokens -> full-set tokens), which tracks cost better than text count.
    est = dt * lengths.sum() / lengths[idx].sum()
    total_est += est
    print(f"{name}: {SAMPLE[name]} texts ({lengths[idx].sum():,} tok) in {dt:.1f} s -> full set est. {est / 60:.1f} min")

print(f"\nESTIMATED full encode time: {total_est / 60:.1f} min (+ ~1 min load/scoring)")
