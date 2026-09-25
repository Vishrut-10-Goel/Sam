"""Tests for eval/ (step 5): metrics against pytrec_eval and MTEB's MRR, and the apps pipeline end to end.

The pipeline test uses an oracle fake encoder (each query embeds to its relevant document's vector) over the
real apps corpus, so it exercises loader -> index -> retriever -> metrics without loading the model.

Run from the project root:  python -m tests.test_eval
(Also collectable by pytest if it is installed.)
"""
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pytrec_eval
from mteb._evaluators.retrieval_metrics import mrr as mteb_mrr

import loaders.apps
from eval import apps_pipeline
from eval.metrics import evaluate, mrr_at_k, ndcg_at_k, ranked
from index.storage import MANIFEST
from tests.test_index import FakeEncoder


def _random_case(seed: int):
    rng = np.random.default_rng(seed)
    docs = [f"d{i}" for i in range(40)]
    run, qrels = {}, {}
    for q in range(30):
        # Scores from a small set of values, so exact ties are common.
        run[f"q{q}"] = {d: float(rng.integers(0, 6)) / 5 for d in rng.choice(docs, 25, replace=False)}
        rel_docs = rng.choice(docs, rng.integers(1, 5), replace=False)
        qrels[f"q{q}"] = {d: int(rng.integers(0, 4)) for d in rel_docs}  # graded, including 0
        if not any(qrels[f"q{q}"].values()):
            qrels[f"q{q}"][rel_docs[0]] = 1
    return run, qrels


def test_ndcg_matches_pytrec_eval():
    for seed in range(5):
        run, qrels = _random_case(seed)
        reference = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"}).evaluate(run)
        ours = evaluate(run, qrels, k=10)["per_query"]
        for q in run:
            assert abs(ours[q]["ndcg"] - reference[q]["ndcg_cut_10"]) < 1e-9, (seed, q, ours[q], reference[q])


def test_mrr_matches_mteb():
    for seed in range(5):
        run, qrels = _random_case(seed)
        reference = mteb_mrr(qrels, run, [10])["MRR@10"]
        ours = evaluate(run, qrels, k=10)["per_query"]
        assert [ours[q]["mrr"] for q in run] == reference, seed


def test_ties_break_by_doc_id_descending():
    assert ranked({"d1": 0.5, "d9": 0.5, "d5": 0.7}) == ["d5", "d9", "d1"]
    assert ranked({"d10": 0.5, "d9": 0.5}) == ["d9", "d10"]  # string order, as MTEB/pytrec_eval
    assert mrr_at_k(["d9", "d1"], {"d1": 1}, 10) == 0.5
    assert ndcg_at_k(["d9", "d1"], {"d1": 1}, 1) == 0.0


class OracleEncoder(FakeEncoder):
    """Embeds each apps test query as its relevant document's text: a perfect retriever."""

    def __init__(self):
        super().__init__()
        queries, qrels = loaders.apps.load_apps_queries()
        docs = {d.id: d.text for d in loaders.apps.load_apps()}
        self.as_doc = {queries[q]: docs[next(iter(rel))] for q, rel in qrels.items()}

    def encode(self, texts, batch_size=None, progress=None):
        return super().encode([self.as_doc.get(t, t) for t in texts], batch_size, progress)


def test_apps_pipeline_end_to_end_with_oracle():
    enc = OracleEncoder()
    docs = list(loaders.apps.load_apps())
    counts = Counter(d.text for d in docs)
    text = {d.id: d.text for d in docs}
    _, qrels = loaders.apps.load_apps_queries()

    with tempfile.TemporaryDirectory() as tmp:
        index_dir = Path(tmp, "apps")
        result = apps_pipeline.run(index_dir, limit=300, encoder=enc)
        assert (index_dir / MANIFEST).exists() and result["n_documents"] == result["n_chunks"] == 8765
        assert result["n_queries"] == 300 and len(result["per_query"]) == 300
        for q, m in result["per_query"].items():
            rel = next(iter(qrels[q]))
            if counts[text[rel]] == 1:
                assert m["ndcg"] == m["mrr"] == 1.0, (q, m)
            else:  # an identical duplicate ties with it; the higher doc id ranks first
                assert m["mrr"] in (1.0, 0.5), (q, m)

        # Second run reuses the saved index: only the queries get embedded.
        enc.encoded.clear()
        again = apps_pipeline.run(index_dir, limit=5, encoder=enc)
        assert len(enc.encoded) == 5 and again["n_queries"] == 5
        assert apps_pipeline.compare_with_mteb(again, Path(tmp, "none.json")) is None  # --limit: not comparable


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
