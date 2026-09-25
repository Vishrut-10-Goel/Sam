"""Throughput probe: cross-encoders on real AppsRetrieval (query, candidate) pairs, CPU fp32."""
import sys
import time

import torch
from sentence_transformers import CrossEncoder

sys.path.insert(0, ".")
import loaders.apps  # noqa: E402

queries, qrels = loaders.apps.load_apps_queries()
docs = {d.id: d.text for d in loaders.apps.load_apps()}
qids = list(queries)[:8]
pairs = [(queries[q], docs[d]) for q in qids for d in list(docs)[:8]]  # 64 pairs, realistic lengths

for name, max_len in [("cross-encoder/ms-marco-MiniLM-L-6-v2", 512),
                      ("Alibaba-NLP/gte-reranker-modernbert-base", 512),
                      ("Alibaba-NLP/gte-reranker-modernbert-base", 1024)]:
    t0 = time.perf_counter()
    model = CrossEncoder(name, device="cpu", max_length=max_len, model_kwargs={"dtype": torch.float32})
    dtype = next(model.model.parameters()).dtype
    load = time.perf_counter() - t0
    model.predict(pairs[:8], batch_size=8)  # warm-up
    t0 = time.perf_counter()
    model.predict(pairs, batch_size=16)
    per_pair = (time.perf_counter() - t0) / len(pairs)
    print(f"{name} max_len={max_len} ({dtype}, load {load:.0f} s): {per_pair * 1000:.0f} ms/pair -> "
          f"200 q x top-50: {200 * 50 * per_pair / 60:.0f} min; full 3,765 q x top-50: {3765 * 50 * per_pair / 3600:.1f} h; "
          f"x top-20: {3765 * 20 * per_pair / 3600:.1f} h", flush=True)
