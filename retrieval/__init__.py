"""Search an index: query embedding, chunk scoring, chunk -> document grouping, and snippets from the source."""
from retrieval.search import AGGREGATES, ChunkHit, DocResult, Retriever
from retrieval.snippets import Snippet, SnippetReader

__all__ = ["AGGREGATES", "ChunkHit", "DocResult", "Retriever", "Snippet", "SnippetReader"]
