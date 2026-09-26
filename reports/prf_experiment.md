# Pseudo-relevance feedback on AppsRetrieval (it does not help)

**Result: rejected. Tuned on the train split, the best PRF setting scores NDCG@10 0.57536 on test against the
submitted first stage's 0.57545 (−0.00009; MRR@10 0.52795 vs 0.52798; 119 queries better, 118 worse). On train,
every one of the 54 settings tried was worse than no feedback. The submission stays first-stage only.**

## What was tried

A second retrieval pass that needs no re-encoding (Rocchio-style feedback on the vectors): take the first stage's
top *k* solutions, average their embeddings into a centroid *c*, and re-score every solution against
*q + α·c*. α = 0 is the submitted system.

## Protocol (fixed before the test measurement)

1. The cached query vectors must reproduce the submission: test NDCG@10 0.57545. They do (and MRR@10 0.52798).
2. Grid search on the **train** split only: 5,000 queries, disjoint from test (different problems and solutions),
   searching the same 8,765-solution corpus. k ∈ {1, 2, 3, 5, 10, 20}, α ∈ {0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0,
   1.5, 2.0}; pick the best NDCG@10.
3. Measure that one setting on **test**, once. The script refuses to measure test a second time.
4. Keep it only if test NDCG@10 improves by at least 0.005 and more test queries improve than get worse.

Metrics are MTEB's definitions and tie-breaking (`eval/metrics.py`). Vectors come from
`experiments/encode_apps_queries.py` (the same encoder as the submission) and the committed `indexes/apps`.

## Results

**Train** (first stage: NDCG@10 0.71098, MRR@10 0.67488), NDCG@10:

| k \ α | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|---|---|---|
| 1 | 0.70968 | 0.70755 | 0.70491 | 0.70034 | 0.69334 | 0.68194 | 0.67048 |
| 2 | **0.71021** | 0.70928 | 0.70797 | 0.70609 | 0.70303 | 0.69773 | 0.69300 |
| 3 | 0.70878 | 0.70726 | 0.70447 | 0.70153 | 0.69706 | 0.68290 | 0.66084 |
| 5 | 0.70852 | 0.70632 | 0.70290 | 0.69980 | 0.69129 | 0.67135 | 0.63286 |
| 10 | 0.70856 | 0.70702 | 0.70197 | 0.69702 | 0.68686 | 0.65735 | 0.60089 |
| 20 | 0.70774 | 0.70598 | 0.70064 | 0.69548 | 0.68353 | 0.64843 | 0.57423 |

(α = 0.75 and 1.5 are in [`prf/prf_results.json`](prf/prf_results.json); they follow the same trend.) Every cell is
below the first stage, and the loss grows with both k and α: feedback is monotonically harmful here. The "best"
setting (k = 2, α = 0.05) is simply the one that changes the ranking least.

**Test**, the one measurement:

| | NDCG@10 | MRR@10 | Queries better / worse (NDCG@10) |
|---|---|---|---|
| First stage (submitted) | 0.57545 | 0.52798 | |
| PRF, k = 2, α = 0.05 | 0.57536 (−0.00009) | 0.52795 (−0.00003) | 119 / 118 |

**Decision: reject** (needed ≥ +0.005).

## Why it does not help (interpretation, not measured)

- **One relevant solution per query.** PRF helps when a query has many relevant documents and the first stage finds
  some of them: their centroid pulls in the rest. Here each problem has exactly one relevant solution, so the other
  top-k documents are, by construction, solutions to *other* problems.
- **Near neighbours are near-miss problems.** The top hits for a problem are solutions to similar problems (all
  interval problems, all subset problems; see the "subsets with duplicates" example in the demo survey, where the
  plain-subsets solution outranks the right one). Averaging them moves the query toward the neighbourhood, not
  toward the one right answer, which is what separated it from its neighbours in the first place.
- **When the first stage is already right, feedback can only dilute it;** when it is wrong, feedback reinforces the
  wrong neighbourhood. Both effects push the same way, which matches the monotone losses on train.

Together with the [cross-encoder re-ranking experiment](reranking_experiment.md), this leaves the first stage as the
best system we have measured on AppsRetrieval.

## Files

- [`prf/prf_tune.py`](prf/prf_tune.py): the script (`train` phase, then `test` phase).
- [`prf/prf_results.json`](prf/prf_results.json): the full train grid, the chosen setting and the test measurement.
- [`prf/prf_train_log.txt`](prf/prf_train_log.txt), [`prf/prf_test_log.txt`](prf/prf_test_log.txt): run output.
