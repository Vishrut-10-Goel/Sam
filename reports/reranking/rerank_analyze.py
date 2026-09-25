"""Score re-ranking variants from rerank_test.json with eval/metrics.py (MTEB conventions)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import loaders.apps  # noqa: E402
from eval.metrics import evaluate  # noqa: E402

data = json.loads((Path(__file__).parent / "rerank_test.json").read_text(encoding="utf-8"))
_, qrels = loaders.apps.load_apps_queries()
qids, cands = data["qids"], data["candidates"]


def run_from_orders(orders):
    """Strictly decreasing synthetic scores, so the metric sees exactly this order (no tie-breaking)."""
    return {q: {d: float(len(o) - i) for i, d in enumerate(o)} for q, o in orders.items()}


def variant(scores_by_q, k, mode, rrf_k=60):
    orders = {}
    for q in qids:
        first = [d for d, _ in cands[q]]
        head, tail = first[:k], first[k:]
        ce = scores_by_q[q]
        if mode == "rerank":
            head = [d for _, d in sorted(zip(ce, head), key=lambda x: -x[0])]
        elif mode == "rrf":
            ce_rank = {d: r for r, (_, d) in enumerate(sorted(zip(ce, head), key=lambda x: -x[0]))}
            fused = {d: 1 / (rrf_k + i + 1) + 1 / (rrf_k + ce_rank[d] + 1) for i, d in enumerate(head)}
            head = sorted(head, key=lambda d: -fused[d])
        orders[q] = head + tail
    return evaluate(run_from_orders(orders), qrels)


base = evaluate({q: dict(c) for q, c in cands.items()}, qrels)
print(f"{len(qids)} queries, first stage top {data['depth']}: recall@{data['depth']} {data['recall_at_depth']:.3f}")
print(f"{'variant':52} NDCG@10  MRR@10   better/worse queries (NDCG@10)")
print(f"{'baseline (first stage only)':52} {base['ndcg_at_10']:.5f}  {base['mrr_at_10']:.5f}")
for name, r in data["rerankers"].items():
    short = name.split("/")[-1]
    for mode in ("rerank", "rrf"):
        v = variant(r["scores"], r["k"], mode)
        better = sum(v["per_query"][q]["ndcg"] > base["per_query"][q]["ndcg"] + 1e-12 for q in qids)
        worse = sum(v["per_query"][q]["ndcg"] < base["per_query"][q]["ndcg"] - 1e-12 for q in qids)
        label = f"{short} top-{r['k']} {'re-rank only' if mode == 'rerank' else 'RRF fusion (k=60)'}"
        print(f"{label:52} {v['ndcg_at_10']:.5f}  {v['mrr_at_10']:.5f}   {better:>3} / {worse:<3}"
              + (f"   [{r['pairs']:,} pairs in {r['seconds'] / 60:.1f} min]" if mode == "rerank" else ""))
