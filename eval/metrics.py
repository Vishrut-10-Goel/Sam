"""NDCG@k and MRR@k, computed the same way MTEB computes them, so pipeline scores are directly comparable.

Ranking: documents sorted by score descending, ties broken by doc id descending (string order). This matches
MTEB's MRR and pytrec_eval (which MTEB uses for NDCG). Ties are real on AppsRetrieval: 11 solution texts appear
twice in the corpus and embed identically, and 20 relevant documents are among them.

NDCG gain is the relevance level (not 2^rel - 1) with a log2(rank + 1) discount, as in trec_eval's ndcg_cut.
MRR@k counts documents with relevance > 0 as relevant.
"""
from __future__ import annotations

import math
from typing import Mapping

Run = Mapping[str, Mapping[str, float]]    # query_id -> {doc_id: score}
Qrels = Mapping[str, Mapping[str, int]]    # query_id -> {doc_id: relevance}


def ranked(doc_scores: Mapping[str, float]) -> list[str]:
    """Doc ids in rank order: score descending, ties by doc id descending."""
    return [doc_id for doc_id, _ in sorted(doc_scores.items(), key=lambda item: (item[1], item[0]), reverse=True)]


def ndcg_at_k(ranking: list[str], relevance: Mapping[str, int], k: int) -> float:
    dcg = sum(max(relevance.get(doc_id, 0), 0) / math.log2(rank + 2) for rank, doc_id in enumerate(ranking[:k]))
    ideal = sorted((r for r in relevance.values() if r > 0), reverse=True)[:k]
    idcg = sum(r / math.log2(rank + 2) for rank, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(ranking: list[str], relevance: Mapping[str, int], k: int) -> float:
    for rank, doc_id in enumerate(ranking[:k]):
        if relevance.get(doc_id, 0) > 0:
            return 1.0 / (rank + 1)
    return 0.0


def evaluate(run: Run, qrels: Qrels, k: int = 10) -> dict:
    """Mean NDCG@k and MRR@k over the run's queries, plus each query's values."""
    per_query = {}
    for query_id, doc_scores in run.items():
        ranking = ranked(doc_scores)
        per_query[query_id] = {"ndcg": ndcg_at_k(ranking, qrels[query_id], k), "mrr": mrr_at_k(ranking, qrels[query_id], k)}
    n = len(per_query)
    return {
        f"ndcg_at_{k}": sum(q["ndcg"] for q in per_query.values()) / n if n else 0.0,
        f"mrr_at_{k}": sum(q["mrr"] for q in per_query.values()) / n if n else 0.0,
        "per_query": per_query,
    }
