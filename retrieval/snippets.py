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
    """Reads and caches document sources for one index.

    File sources are re-read whenever their size or modification time changes, so a long-lived reader (the
    interactive CLI) reports an edit made mid-session as stale instead of serving the cached old text.
    """

    def __init__(self, index: Index):
        self.index = index
        self._files: dict[str, tuple[tuple[int, int], str | None]] = {}  # doc_id -> (stat stamp, text)
        self._apps: dict[str, str] | None = None

    def _apps_text(self, doc_id: str) -> str | None:
        if self._apps is None:
            from loaders.apps import load_apps  # lazy: loads the dataset, only needed for apps indexes
            self._apps = {d.id: d.text for d in load_apps(self.index.source.get("revision"))}
        return self._apps.get(doc_id)

    def _read(self, doc_id: str) -> str | None:
        if self.index.source.get("kind") == "apps":
            return self._apps_text(doc_id)
        try:
            path = Path(self.index.documents[doc_id].metadata["source_path"])
            st = path.stat()
            stamp = (st.st_size, st.st_mtime_ns)
        except (OSError, KeyError):
            self._files.pop(doc_id, None)
            return None
        cached = self._files.get(doc_id)
        if cached is None or cached[0] != stamp:
            try:
                text = path.read_bytes().decode("utf-8-sig")
            except (OSError, UnicodeDecodeError):
                text = None
            cached = self._files[doc_id] = (stamp, text)
        return cached[1]

    def snippet(self, doc_id: str, start_line: int, end_line: int) -> Snippet:
        text = self._read(doc_id)
        if text is None:
            return Snippet(None, "missing")
        if content_hash(text) != self.index.documents[doc_id].content_hash:
            return Snippet(None, "stale")
        return Snippet("".join(split_lines(text)[start_line - 1:end_line]), "ok")
