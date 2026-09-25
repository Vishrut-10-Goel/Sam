"""Encode AppsRetrieval train and test queries once, for offline experiments (e.g. pseudo-relevance feedback).

Train (5,000 queries) and test (3,765) are disjoint, with one relevant solution each in the same 8,765-document
corpus, so a second-pass method can be tuned on train and measured once on test. Vectors come from the shared
encoder, exactly as the retriever embeds queries; the encoder's fingerprint is saved with them so they are never
mixed with an index built by a different encoder.

Run from the repository root:  python experiments/encode_apps_queries.py
Writes experiments/cache/apps_queries_<split>.npz (ids, embeddings, fingerprint JSON), gitignored.
~1 h per split on a 12-thread CPU.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from embedding.onnx_encoder import OnnxEncoder  # noqa: E402
from loaders.apps import load_apps_queries  # noqa: E402

OUT = Path("experiments/cache")


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    encoder = OnnxEncoder()
    fingerprint = json.dumps(encoder.fingerprint(), sort_keys=True)
    for split in ("train", "test"):
        path = OUT / f"apps_queries_{split}.npz"
        if path.exists():
            log(f"{path} exists, skipping")
            continue
        queries, _ = load_apps_queries(split)
        ids = list(queries)
        t0 = last = time.perf_counter()

        def progress(done, total):
            global last
            now = time.perf_counter()
            if now - last >= 60 or done == total:
                last = now
                log(f"  {split}: {done:,}/{total:,} ({(now - t0) / 60:.1f} min)")

        log(f"{split}: encoding {len(ids):,} queries")
        emb = encoder.encode([queries[q] for q in ids], progress=progress)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, ids=np.array(ids), embeddings=emb, fingerprint=np.array(fingerprint))
        tmp.replace(path)
        log(f"{split}: saved {emb.shape} to {path} in {(time.perf_counter() - t0) / 60:.1f} min")
    log("done")
