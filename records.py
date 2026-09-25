"""Record types shared by the loaders, chunkers, index and retrieval.

Document: one retrievable unit as a loader sees it (an apps solution, a source file).
Chunk:    the unit that gets embedded. A chunk always belongs to exactly one document, and search results
          are grouped back to documents via doc_id.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


def content_hash(text: str) -> str:
    """sha256 of the text's UTF-8 bytes. The index compares these to detect changed documents."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Document:
    """id: stable across re-indexing (dataset ID, or root-relative path for files) - never positional.

    metadata always has:
      source_path   where the text came from (absolute file path, or dataset reference)
      language      lowercase language name, or None if unknown
      content_hash  content_hash(text)
    Loaders may add more keys (e.g. the apps partition).
    """

    id: str
    text: str
    metadata: dict = field(default_factory=dict)

    @classmethod
    def create(cls, id: str, text: str, source_path: str, language: str | None, **extra) -> Document:
        metadata = {"source_path": source_path, "language": language, "content_hash": content_hash(text), **extra}
        return cls(id=id, text=text, metadata=metadata)

    @property
    def content_hash(self) -> str:
        return self.metadata["content_hash"]


@dataclass(frozen=True)
class Chunk:
    """A contiguous run of whole lines from one document.

    start_line / end_line are 1-based and inclusive, so results can cite them directly.
    text is exactly those lines of the document, line endings included.
    """

    chunk_id: str
    doc_id: str
    text: str
    start_line: int
    end_line: int
