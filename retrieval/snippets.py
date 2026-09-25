"""Chunk text for search results, read from the document's source.

The index stores chunk positions, not text, so snippets come from the source: the file on disk for
directory indexes, the dataset for apps. The source is checked against the content hash recorded at indexing
time; if it changed, its line numbers may no longer match the index, so the snippet is reported stale
rather than showing the wrong lines.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from chunking.lines import split_lines
from index import Index
from records import content_hash


@dataclass(frozen=True)
class Snippet:
    text: str | None
    status: str  # "ok", "stale" (source changed since indexing: re-index), or "missing" (source unavailable)


class SnippetReader:
    """Reads and caches document sources for one index."""

    def __init__(self, index: Index):
        self.index = index
        self._texts: dict[str, str | None] = {}
        self._apps: dict[str, str] | None = None

    def _apps_text(self, doc_id: str) -> str | None:
        if self._apps is None:
            from loaders.apps import load_apps  # lazy: loads the dataset, only needed for apps indexes
            self._apps = {d.id: d.text for d in load_apps(self.index.source.get("revision"))}
        return self._apps.get(doc_id)

    def _read(self, doc_id: str) -> str | None:
        if doc_id not in self._texts:
            if self.index.source.get("kind") == "apps":
                text = self._apps_text(doc_id)
            else:
                try:
                    text = Path(self.index.documents[doc_id].metadata["source_path"]).read_bytes().decode("utf-8-sig")
                except (OSError, UnicodeDecodeError, KeyError):
                    text = None
            self._texts[doc_id] = text
        return self._texts[doc_id]

    def snippet(self, doc_id: str, start_line: int, end_line: int) -> Snippet:
        text = self._read(doc_id)
        if text is None:
            return Snippet(None, "missing")
        if content_hash(text) != self.index.documents[doc_id].content_hash:
            return Snippet(None, "stale")
        return Snippet("".join(split_lines(text)[start_line - 1:end_line]), "ok")
