"""Two-pass test on the first N AppsRetrieval test queries: first stage = our encoder + prebuilt apps index (top 50),
second stage = a cross-encoder re-scoring the top K candidates. Saves all candidates and scores (rerank_test.json)
so fusion variants can be computed without re-running models. Metrics via eval/metrics.py (MTEB conventions).
"""
import json
import sys
import time
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder

sys.path.insert(0, ".")
import loaders.apps  # noqa: E402
from embedding.onnx_encoder import OnnxEncoder  # noqa: E402
from eval.metrics import evaluate  # noqa: E402
from index import load_index  # noqa: E402
from retrieval import Retriever  # noqa: E402

N, DEPTH = 200, 50
RERANKERS = [("cross-encoder/ms-marco-MiniLM-L-6-v2", 512, 50), ("Alibaba-NLP/gte-reranker-modernbert-base", 512, 10)]
OUT = Path(__file__).parent / "rerank_test.json"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


queries, qrels = loaders.apps.load_apps_queries()
docs = {d.id: d.text for d in loaders.apps.load_apps()}
qids = list(queries)[:N]

t0 = time.perf_counter()
enc = OnnxEncoder()
index = load_index("indexes/apps", encoder=enc)
first = Retriever(index, enc).search_many([queries[q] for q in qids], top_k=DEPTH)
log(f"first stage: {N} queries, top {DEPTH}, {time.perf_counter() - t0:.0f} s")
cands = {q: [(r.doc_id, r.score) for r in res] for q, res in zip(qids, first)}

committed = json.load(open("apps_pipeline_results.json"))["per_query"]
base = evaluate({q: dict(c) for q, c in cands.items()}, qrels)
same = all(abs(base["per_query"][q]["ndcg"] - committed[q]["ndcg"]) < 1e-9 for q in qids)
recall = sum(any(d in qrels[q] for d, _ in cands[q]) for q in qids) / N
log(f"baseline on these {N}: NDCG@10 {base['ndcg_at_10']:.5f} MRR@10 {base['mrr_at_10']:.5f} "
    f"(matches committed per-query results: {same}); recall@{DEPTH} {recall:.3f}")

out = {"n": N, "depth": DEPTH, "qids": qids, "candidates": cands,
       "baseline": {k: base[k] for k in ("ndcg_at_10", "mrr_at_10")}, "recall_at_depth": recall, "rerankers": {}}
for name, max_len, k in RERANKERS:
    t0 = time.perf_counter()
    model = CrossEncoder(name, device="cpu", max_length=max_len, model_kwargs={"dtype": torch.float32})
    pairs = [(queries[q], docs[d]) for q in qids for d, _ in cands[q][:k]]
    log(f"{name}: scoring {len(pairs):,} pairs (top {k}, max_length {max_len})")
    scores, done, chunk = [], 0, 400
    for i in range(0, len(pairs), chunk):
        scores.extend(float(s) for s in model.predict(pairs[i:i + chunk], batch_size=16))
        done += len(pairs[i:i + chunk])
        el = time.perf_counter() - t0
        log(f"  {done:,}/{len(pairs):,} pairs, {el / 60:.1f} min, eta {el / done * (len(pairs) - done) / 60:.1f} min")
    per_q, it = {}, iter(scores)
    for q in qids:
        per_q[q] = [next(it) for _ in cands[q][:k]]
    out["rerankers"][name] = {"max_length": max_len, "k": k, "scores": per_q,
                              "seconds": round(time.perf_counter() - t0, 1), "pairs": len(pairs)}
    OUT.write_text(json.dumps(out), encoding="utf-8")
log("done")
