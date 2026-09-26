"""Query -> ranked documents: embed the query, score every chunk, group chunk scores into document scores.

Scores are cosine similarities (index embeddings and query embeddings are both L2-normalized, so a dot
product). A document's score aggregates its chunks' scores:
  max   the best-matching chunk (default): a file is as relevant as its most relevant part, and it is not
        penalized or boosted for its length
  mean  the average over its chunks
  sum   the total over its chunks (favours long files with many related parts)
For unchunked indexes (apps: one chunk per document) all three give the chunk score.

File kinds (loaders.directory.file_kind): on plain-language questions, docs (prose, like the question) and tests
(which repeat the implementation's vocabulary) tend to outrank the implementation. kind_penalty subtracts a fixed
amount from test and docs files' scores before ranking; code_only drops them. DocResult.score stays the raw cosine
similarity (what the evaluation uses); DocResult.rank_score is the score the results were ordered by (similarity
minus the penalty). Apps documents (ids like d123) always classify as code, so neither option affects apps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from index import Index
from loaders.directory import file_kind

AGGREGATES = ("max", "mean", "sum")
DEFAULT_TOP_K = 10
DEFAULT_CHUNKS_PER_DOC = 3
QUERY_BLOCK = 256  # queries scored per matrix product; bounds memory at QUERY_BLOCK x n_chunks floats


@dataclass(frozen=True)
class ChunkHit:
    chunk_id: str
    start_line: int
    end_line: int
    score: float


@dataclass(frozen=True)
class DocResult:
    doc_id: str
    score: float
    chunks: list[ChunkHit]  # the document's best chunks, best first (at most chunks_per_doc)
    metadata: dict
    rank_score: float  # what the ranking used: score minus the kind penalty (== score for code files)
    kind: str  # "code", "test" or "docs" (loaders.directory.file_kind)


class Retriever:
    def __init__(
        self,
        index: Index,
        encoder,
        aggregate: str = "max",
        chunks_per_doc: int = DEFAULT_CHUNKS_PER_DOC,
        kind_penalty: float = 0.0,
        code_only: bool = False,
    ):
        if aggregate not in AGGREGATES:
            raise ValueError(f"unknown aggregate {aggregate!r}; expected one of {AGGREGATES}")
        if kind_penalty < 0:
            raise ValueError(f"kind_penalty must be >= 0, got {kind_penalty}")
        index.check_compatible(encoder)
        self.index = index
        self.encoder = encoder
        self.aggregate = aggregate
        self.chunks_per_doc = chunks_per_doc

        # Group embedding rows by document: order rows so each document's rows are contiguous, then
        # reduce each segment. Documents without chunks never appear in results.
        doc_ids = [d for d, entry in index.documents.items() if entry.chunk_ids]
        position = {d: i for i, d in enumerate(doc_ids)}
        row_doc = np.array([position[row.doc_id] for row in index.chunks], dtype=np.int64)
        self._order = np.argsort(row_doc, kind="stable")
        counts = np.bincount(row_doc, minlength=len(doc_ids))
        self._starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
        self._counts = counts
        self._doc_ids = doc_ids
        self._embeddings = index.embeddings[self._order]  # rows in grouped order

        # Per-document score adjustment by file kind: 0 for code, -kind_penalty for tests/docs (-inf: dropped).
        self._kinds = [file_kind(d) for d in doc_ids]
        is_code = np.array([k == "code" for k in self._kinds], dtype=bool)
        other = -np.inf if code_only else -float(kind_penalty)
        self._adjust = np.where(is_code, 0.0, other).astype(np.float32)

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[DocResult]:
        return self.search_many([query], top_k)[0]

    def search_many(
        self, queries: list[str], top_k: int = DEFAULT_TOP_K, progress: Callable[[int, int], None] | None = None
    ) -> list[list[DocResult]]:
        """Embeds all queries in one encoder call (length-sorted batching), then scores them in blocks."""
        if not queries:
            return []
        return self.search_embeddings(self.encoder.encode(list(queries), progress=progress), top_k)

    def search_embeddings(self, query_embeddings: np.ndarray, top_k: int = DEFAULT_TOP_K) -> list[list[DocResult]]:
        query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
        results = []
        for start in range(0, len(query_embeddings), QUERY_BLOCK):
            chunk_scores = query_embeddings[start:start + QUERY_BLOCK] @ self._embeddings.T  # (q, rows)
            doc_scores = self._aggregate(chunk_scores)
            ranking_scores = doc_scores + self._adjust
            for q in range(len(chunk_scores)):
                results.append(self._top_docs(chunk_scores[q], doc_scores[q], ranking_scores[q], top_k))
        return results

    def _aggregate(self, chunk_scores: np.ndarray) -> np.ndarray:
        if not self._doc_ids:
            return np.empty((len(chunk_scores), 0), dtype=np.float32)
        if self.aggregate == "max":
            return np.maximum.reduceat(chunk_scores, self._starts, axis=1)
        sums = np.add.reduceat(chunk_scores, self._starts, axis=1)
        return sums / self._counts if self.aggregate == "mean" else sums

    def _top_docs(
        self, chunk_scores: np.ndarray, doc_scores: np.ndarray, ranking_scores: np.ndarray, top_k: int
    ) -> list[DocResult]:
        """Rank by ranking_scores (kind-adjusted); report both them and the raw doc_scores."""
        k = min(top_k, int(np.isfinite(ranking_scores).sum()))
        if k <= 0:
            return []
        top = np.argpartition(-ranking_scores, k - 1)[:k]
        top = top[np.argsort(-ranking_scores[top], kind="stable")]
        out = []
        for d in top:
            first, n = self._starts[d], self._counts[d]
            seg = chunk_scores[first:first + n]
            best = np.argsort(-seg, kind="stable")[: self.chunks_per_doc]
            hits = []
            for j in best:
                row = self.index.chunks[self._order[first + j]]
                hits.append(ChunkHit(row.chunk_id, row.start_line, row.end_line, float(seg[j])))
            doc_id = self._doc_ids[d]
            out.append(DocResult(doc_id, float(doc_scores[d]), hits, self.index.documents[doc_id].metadata,
                                 float(ranking_scores[d]), self._kinds[d]))
        return out
