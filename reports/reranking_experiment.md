# Two-pass retrieval: cross-encoder re-ranking on AppsRetrieval

**Result: re-ranking hurts on this task.** Both cross-encoders tested make the ranking worse than the first stage
alone, whether they replace its order or are fused with it. We did not run it on the full test set, and the
submission (`appsretrieval_results.json`, first stage only) is unchanged.

## Setup

- **First stage:** our encoder (gte-modernbert-base, fp32 ONNX, 1024 tokens) with the prebuilt `indexes/apps`
  index, top 50 candidates per query. This is exactly the submitted system.
- **Queries:** the first 200 AppsRetrieval test queries. They are easier than average: the first stage scores
  NDCG@10 0.67357 on them against 0.57545 on all 3,765, so every variant is compared on these same 200.
  The first stage reproduces the committed per-query scores (`apps_pipeline_results.json`) exactly on them.
- **Second stage:** a cross-encoder scores each (problem statement, candidate solution) pair, CPU fp32 via
  sentence-transformers `CrossEncoder`, pairs truncated to 512 tokens.
  - `cross-encoder/ms-marco-MiniLM-L-6-v2`: the usual default; 6 layers, trained on MS MARCO web search. Top 50.
  - `Alibaba-NLP/gte-reranker-modernbert-base`: same ModernBERT family as our encoder, trained with code data,
    loads natively (no `trust_remote_code`). Top 10 only: at ~0.7 s per pair a deeper run was not feasible.
  - `jinaai/jina-reranker-v2-base-multilingual` was excluded: it needs `trust_remote_code`.
- **Variants:** *re-rank only* (candidates reordered by the cross-encoder) and *RRF fusion* (reciprocal rank
  fusion of the first-stage and cross-encoder ranks, k = 60; no tuned weights, so nothing was fitted to these
  queries). Candidates below the re-ranked depth keep their first-stage order.
- **Metrics:** `eval/metrics.py` (MTEB's conventions; tested against pytrec_eval and MTEB's MRR).

## Results

| Variant | NDCG@10 | MRR@10 | Queries better / worse (NDCG@10) |
|---|---|---|---|
| **First stage only (submitted system)** | **0.67357** | **0.63171** | — |
| MiniLM-L-6, re-rank top 50 | 0.14836 | 0.10650 | 10 / 149 |
| MiniLM-L-6, RRF fusion | 0.40573 | 0.32780 | 19 / 105 |
| gte-reranker-modernbert, re-rank top 10 | 0.50657 | 0.41442 | 25 / 89 |
| gte-reranker-modernbert, RRF fusion | 0.62105 | 0.56208 | 22 / 50 |

The first stage's recall@50 on these queries is 0.945, so there was room to improve: the correct solution is
almost always among the candidates. The cross-encoders move it down, not up. With 50–149 queries worse against
10–25 better, this is not sampling noise.

**This matches the reranker's own model card.** The gte-modernbert model card's CoIR table reports apps NDCG@10
of **57.54 for the encoder** (gte-modernbert-base) and **47.57 for the reranker** (gte-reranker-modernbert-base).

## Why MiniLM fails

It is not a bug in how we call it. On a normal web-search pair it behaves correctly:
"How many people live in Berlin?" scores +8.76 against a Berlin population passage and −11.25 against an unrelated
one. On apps, among the 189 queries whose correct solution is in the top 50, it puts that solution at
**median rank 17 of 50**, where the first stage puts it at **median rank 1** (a random order would give ~25).
MS MARCO teaches it to match a question with a passage that states the answer in words; here the query is a long
competitive-programming statement (story, input/output spec, examples) and the document is uncommented Python.
Nothing in that pairing looks like its training data, and truncating the pair to 512 tokens also cuts most of the
problem statement.

Our encoder, by contrast, already does well on exactly this benchmark (its model card reports apps), and when the
right solution is in the top 50 it is usually already first. An off-the-shelf re-ranker has to be
better than that at the one decision that matters, and neither is.

## Speed (12-thread CPU, fp32)

| Model | Seconds per pair | 200 queries | Full test set, top 50 | Full test set, top 10 |
|---|---|---|---|---|
| MiniLM-L-6 (512 tokens) | ~0.14 | 10,000 pairs, ~23 min of scoring | ~7.3 h | ~1.5 h |
| gte-reranker-modernbert (512 tokens) | ~0.71 | 2,000 pairs (top 10), 23.7 min | ~37 h | ~7.4 h |
| gte-reranker-modernbert (1,024 tokens, probe only) | ~1.9 | — | ~97 h | ~20 h |

The MiniLM run's log shows 45.3 min of wall time, but that includes a 22-minute stall (22:50 → 23:12) during which
no pairs were scored (the machine was busy or suspended); the scoring rate before and after it matches the
~0.14 s/pair measured beforehand.

## What this means for "multiple passes"

A second pass has to add information the first pass lacks. A generic cross-encoder does not, on this task. Options
that remain, not yet tested: pseudo-relevance feedback (a second retrieval with the query moved toward its top
results; needs a held-out split to tune on), or fusing a second embedding model trained on code.

## Files

- [`reranking/rerank_probe.py`](reranking/rerank_probe.py): throughput probe for the candidate cross-encoders.
- [`reranking/rerank_test.py`](reranking/rerank_test.py): first stage + both re-rankers on the first 200 test
  queries; writes every candidate list and score to `rerank_test.json`.
- [`reranking/rerank_analyze.py`](reranking/rerank_analyze.py): computes the table above from the saved scores.
- [`reranking/rerank_test.json`](reranking/rerank_test.json): all candidates (top 50 per query, first-stage
  scores) and cross-encoder scores.
- [`reranking/rerank_test_log.txt`](reranking/rerank_test_log.txt): the run's log, including the stall.

Run the scripts from the repository root (e.g. `python reports/reranking/rerank_analyze.py`); they read and write
`rerank_test.json` next to themselves.
