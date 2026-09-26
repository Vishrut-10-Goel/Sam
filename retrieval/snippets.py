"""Chunk text for search results, read from the document's source.

The index stores chunk positions, not text, so snippets come from the source: the file on disk for
directory indexes, the dataset for apps. The source is checked against the content hash recorded at indexing
time; if it changed, its line numbers may no longer match the index, so the snippet is reported stale
rather than showing the wrong lines.
"""
from __future__ import annotations

import re
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


# Snippet focus: which lines of a long chunk to show first. A folder chunk runs to ~100 lines and a result shows a
# dozen, so opening at the chunk's first line often hides the part that matched. This is display only: it never
# changes ranking. It is lexical, so it needs no extra model call: query words are matched against the chunk's
# identifiers split into parts (resolve_redirects -> resolve, redirects; maxRetries -> max, retries).
_IDENT = re.compile(r"[A-Za-z0-9]+")
_PART = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|[0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by can could do does done for from how i if in into is it its me my of on or should "
    "so that the their there this to was we what when where which who why will with would you your "
    "implement implemented implementation work works".split())
_DEFINITION = re.compile(
    r"^\s*(?:@|(?:async\s+)?def\s|class\s|(?:export\s+)?(?:async\s+)?function\b|[\w.]+\s*=\s*(?:async\s+)?function\b"
    r"|(?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?\()")
_SNAP_BACK = 6  # how far above the first matching line to look for the enclosing definition


def _terms(text: str) -> set[str]:
    """Lower-cased identifier parts, cut to 5 characters so redirect/redirects/redirected match."""
    parts = (p.lower() for word in _IDENT.findall(text) for p in _PART.findall(word))
    return {p[:5] for p in parts if len(p) >= 3 and p not in _STOPWORDS}


def focus_offset(lines: list[str], query: str, budget: int) -> int:
    """0-based index of the first of `budget` lines to show: the window covering most of the query's words.

    Each query word is weighted by 1 / (number of chunk lines containing it), so a window scores the share of
    each word's occurrences it holds and covering several different words beats repeating one. Matches on
    definition lines (def, class, function) count double. Ties go to the earliest window, and 0 is returned
    unless a later window scores strictly higher than the first; the chosen window is then moved up to the
    definition just above its first match, if there is one within a few lines.
    """
    if len(lines) <= budget:
        return 0
    query_terms = _terms(query)
    if not query_terms:
        return 0
    hits = [_terms(line) & query_terms for line in lines]
    df: dict[str, int] = {}
    for h in hits:
        for t in h:
            df[t] = df.get(t, 0) + 1
    line_score = [sum(1 / df[t] for t in h) * (2 if h and _DEFINITION.match(line) else 1)
                  for h, line in zip(hits, lines)]
    window = sum(line_score[:budget])
    best, best_start = window, 0
    for start in range(1, len(lines) - budget + 1):
        window += line_score[start + budget - 1] - line_score[start - 1]
        if window > best + 1e-9:
            best, best_start = window, start
    if best_start == 0:
        return 0
    first = next(i for i in range(best_start, best_start + budget) if line_score[i] > 0)
    start = first
    for i in range(first, max(first - _SNAP_BACK, 0) - 1, -1):
        if _DEFINITION.match(lines[i]):
            start = i
            break
    # A decorator line belongs to the definition below it: include the whole stack.
    while start > 0 and lines[start - 1].lstrip().startswith("@"):
        start -= 1
    return min(start, len(lines) - budget)
