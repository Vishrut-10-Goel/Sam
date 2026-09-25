"""Chunkers turn a Document into the Chunks that get embedded.

Chunking is a per-loader setting (each loader module's CHUNKING names one of CHUNKERS), not a stage every
document passes through: apps solutions are embedded whole, directory files are split into windows.
"ast" (function/class-level chunks for Python, chunking.ast_chunks) is opt-in for folder indexes.
"""
from __future__ import annotations

from typing import Callable

from chunking.ast_chunks import chunk_ast
from chunking.none import chunk_none
from chunking.windows import DEFAULT_OVERLAP_TOKENS, chunk_windows
from records import Chunk, Document

Chunker = Callable[[Document], list[Chunk]]

CHUNKERS = ("none", "windows", "ast")


def make_chunker(
    name: str,
    *,
    tokenizer=None,
    max_length: int | None = None,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    header: bool = False,
) -> Chunker:
    """name: a loader's CHUNKING. "windows" needs the encoder's tokenizer and max_length, so chunks fit it.

    header (windows only): prefix each chunk's embedded text with its file path and enclosing definitions.
    """
    if name == "none":
        return chunk_none
    if name in ("windows", "ast"):
        if tokenizer is None or max_length is None:
            raise ValueError(f"{name} chunking needs the encoder's tokenizer and max_length")
        chunk = chunk_windows if name == "windows" else chunk_ast
        return lambda doc: chunk(doc, tokenizer, max_length, overlap_tokens, header=header)
    raise ValueError(f"unknown chunker {name!r}; expected one of {CHUNKERS}")
