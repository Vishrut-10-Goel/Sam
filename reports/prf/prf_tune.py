"""Pseudo-relevance feedback (PRF) on AppsRetrieval: tune on train, measure once on test.

Second pass (Rocchio-style, vectors only, no re-encoding): take the first stage's top k documents, average their
embeddings into a centroid c, and re-score every document with  q + alpha * c  (cosine against the unit-length
document vectors; the query's norm does not change a query's ranking, so no re-normalization is needed).
alpha = 0 is the submitted first stage.

Inputs (no model load): the cached query vectors from experiments/encode_apps_queries.py and the committed
indexes/apps embeddings. Metrics: eval/metrics.py (MTEB's definitions and tie-breaking) over each query's top 100.

Protocol, fixed before any test measurement:
  1. Check the cached vectors reproduce the first stage: test NDCG@10 must be 0.57545 (the submission).
  2. Grid over k and alpha on the TRAIN split only (5,000 queries, disjoint from test); pick the best NDCG@10.
  3. Measure that one configuration on TEST, once.
  4. Keep it only if test NDCG@10 improves by at least 0.005 over 0.57545 AND more test queries improve than
     get worse. Otherwise the submission stays first-stage only.

Run from the repository root (writes reports/prf/prf_results.json):
  python reports/prf/prf_tune.py train   steps 1-2
  python reports/prf/prf_tune.py test    step 3-4, reading the train choice; refuses to run a second time
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from eval.metrics import evaluate  # noqa: E402
from loaders.apps import load_apps_queries  # noqa: E402

HERE = Path(__file__).parent
KS = [1, 2, 3, 5, 10, 20]
ALPHAS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
POOL = 100  # documents per query handed to the metric (NDCG@10 / MRR@10 only look at the top 10)
MIN_GAIN = 0.005
SUBMITTED_NDCG = 0.57545


def load_split(split, doc_ids):
    z = np.load(f"experiments/cache/apps_queries_{split}.npz")
    _, qrels = load_apps_queries(split)
    ids = [str(q) for q in z["ids"]]
    return ids, z["embeddings"].astype(np.float32), {q: qrels[q] for q in ids}


def run_from_scores(ids, scores, doc_ids):
    top = np.argpartition(-scores, POOL, axis=1)[:, :POOL]
    return {q: {doc_ids[j]: float(scores[i, j]) for j in top[i]} for i, q in enumerate(ids)}


def prf_scores(Q, E, first_stage, k, alpha):
    top = np.argpartition(-first_stage, k - 1, axis=1)[:, :k]
    centroid = E[top].mean(axis=1)                     # (n_queries, dim)
    return first_stage + alpha * (centroid @ E.T)      # == (Q + alpha * centroid) @ E.T


def main(phase):
    results_path = HERE / "prf_results.json"
    E = np.load("indexes/apps/embeddings-000001.npy").astype(np.float32)
    rows = [json.loads(line) for line in open("indexes/apps/chunks-000001.jsonl", encoding="utf-8")]
    doc_ids = [r["doc_id"] for r in rows]  # one chunk per apps document
    out = {"protocol": __doc__.split("Protocol, fixed before any test measurement:")[1].split("Run from")[0].strip()}
    if phase == "test":
        out = json.loads(results_path.read_text(encoding="utf-8"))
        if "test_prf" in out:
            raise SystemExit("test was already measured once; not measuring again")

    ids_te, Q_te, qrels_te = load_split("test", doc_ids)
    S_te = Q_te @ E.T
    base_te = evaluate(run_from_scores(ids_te, S_te, doc_ids), qrels_te)
    print(f"test first stage (check): NDCG@10 {base_te['ndcg_at_10']:.5f}  MRR@10 {base_te['mrr_at_10']:.5f}")
    assert round(base_te["ndcg_at_10"], 5) == SUBMITTED_NDCG, "cached vectors do not reproduce the submission"

    if phase == "test":
        best = {"k": out["chosen"]["k"], "alpha": out["chosen"]["alpha"]}
        return measure_test(out, best, ids_te, Q_te, qrels_te, S_te, base_te, E, doc_ids, results_path)

    ids_tr, Q_tr, qrels_tr = load_split("train", doc_ids)
    S_tr = Q_tr @ E.T
    base_tr = evaluate(run_from_scores(ids_tr, S_tr, doc_ids), qrels_tr)
    print(f"train first stage: NDCG@10 {base_tr['ndcg_at_10']:.5f}  MRR@10 {base_tr['mrr_at_10']:.5f}")
    grid = []
    for k in KS:
        for alpha in ALPHAS:
            m = evaluate(run_from_scores(ids_tr, prf_scores(Q_tr, E, S_tr, k, alpha), doc_ids), qrels_tr)
            grid.append({"k": k, "alpha": alpha, "ndcg_at_10": round(m["ndcg_at_10"], 5), "mrr_at_10": round(m["mrr_at_10"], 5)})
            print(f"  train k={k:<2} alpha={alpha:<4}  NDCG@10 {m['ndcg_at_10']:.5f}  MRR@10 {m['mrr_at_10']:.5f}")
    best = max(grid, key=lambda g: g["ndcg_at_10"])
    print(f"best on train: k={best['k']} alpha={best['alpha']}  NDCG@10 {best['ndcg_at_10']:.5f} "
          f"(first stage {base_tr['ndcg_at_10']:.5f})")
    out.update({
        "train_first_stage": {"ndcg_at_10": round(base_tr["ndcg_at_10"], 5), "mrr_at_10": round(base_tr["mrr_at_10"], 5)},
        "train_grid": grid,
        "chosen": {"k": best["k"], "alpha": best["alpha"]},
    })
    results_path.write_text(json.dumps(out, indent=1), encoding="utf-8")


def measure_test(out, best, ids_te, Q_te, qrels_te, S_te, base_te, E, doc_ids, results_path):
    """The one test measurement."""
    m_te = evaluate(run_from_scores(ids_te, prf_scores(Q_te, E, S_te, best["k"], best["alpha"]), doc_ids), qrels_te)
    better = sum(m_te["per_query"][q]["ndcg"] > base_te["per_query"][q]["ndcg"] + 1e-12 for q in ids_te)
    worse = sum(m_te["per_query"][q]["ndcg"] < base_te["per_query"][q]["ndcg"] - 1e-12 for q in ids_te)
    gain = m_te["ndcg_at_10"] - base_te["ndcg_at_10"]
    keep = gain >= MIN_GAIN and better > worse
    print(f"TEST with k={best['k']} alpha={best['alpha']}: NDCG@10 {m_te['ndcg_at_10']:.5f} ({gain:+.5f})  "
          f"MRR@10 {m_te['mrr_at_10']:.5f} ({m_te['mrr_at_10'] - base_te['mrr_at_10']:+.5f})  "
          f"queries better/worse {better}/{worse}  -> {'KEEP' if keep else 'REJECT'}")

    out.update({
        "test_first_stage": {"ndcg_at_10": round(base_te["ndcg_at_10"], 5), "mrr_at_10": round(base_te["mrr_at_10"], 5)},
        "test_prf": {"ndcg_at_10": round(m_te["ndcg_at_10"], 5), "mrr_at_10": round(m_te["mrr_at_10"], 5),
                     "queries_better": better, "queries_worse": worse},
        "decision": "keep" if keep else "reject",
    })
    results_path.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1:] not in (["train"], ["test"]):
        raise SystemExit("usage: prf_tune.py train|test")
    main(sys.argv[1])
