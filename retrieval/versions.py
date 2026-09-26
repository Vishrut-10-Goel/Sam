"""Search a versioned index (index.versions): one result per file, not one per version of it.

Near-identical versions of a file score almost the same, so ranking revisions directly would fill the top 10 with
the same code several times. Revisions are grouped by path the way chunks are grouped into documents: every
revision is scored, and a file ranks by its *primary* revision -
  all versions:      its best-scoring revision (ties, e.g. identical chunks, go to the newest version)
  one version (sha): the revision that commit contains (files absent from that commit are left out)
Nothing is dropped silently: each result lists every revision of the file with its score and commits, and whether
it contains the primary match's lines verbatim (so a reader can see which versions hold the answer, and where).
That check compares text, so it does not depend on where chunk boundaries fall in each revision.
"""
from __future__ import annotations

from dataclasses import dataclass

from index.versions import is_versioned
from retrieval.search import DocResult, Retriever
from retrieval.snippets import Snippet, SnippetReader


@dataclass(frozen=True)
class RevisionHit:
    doc_id: str                 # "<path>@<hash12>"
    commits: list[str]          # shas of the indexed commits holding this exact file, oldest first
    score: float                # ranking score (similarity minus any kind penalty)
    similarity: float
    start_line: int             # this revision's best chunk
    end_line: int
    primary: bool
    contains_match: bool | None  # holds the primary match's lines verbatim (None: text unavailable)
    match_line: int | None       # where those lines start in this revision, if it holds them


@dataclass(frozen=True)
class VersionedResult:
    path: str
    kind: str
    primary: DocResult           # the revision the file is ranked (and its snippet shown) by
    snippet: Snippet
    revisions: list[RevisionHit]  # primary first, then the others by score


def _commit_order(index) -> dict[str, int]:
    return {c["sha"]: i for i, c in enumerate(index.source.get("commits", []))}


def search_versions(retriever: Retriever, reader: SnippetReader, query: str, top_k: int,
                    version: str | None = None) -> list[VersionedResult]:
    """version: a full sha from index.versions.resolve_version, or None for all versions."""
    index = retriever.index
    if not is_versioned(index):
        raise ValueError("not a versioned index")
    order = _commit_order(index)
    ranked = retriever.search(query, top_k=len(retriever.doc_ids))  # every revision, best first
    groups: dict[str, list[DocResult]] = {}
    for r in ranked:
        groups.setdefault(r.metadata["path"], []).append(r)

    def newest(r: DocResult) -> int:
        return max(order.get(s, -1) for s in r.metadata["versions"])

    for path in groups:  # best first; equal scores (identical chunks) newest first
        groups[path].sort(key=lambda r: (-r.rank_score, -newest(r)))

    def primary_of(revisions: list[DocResult]) -> DocResult | None:
        if version is None:
            return revisions[0]
        return next((r for r in revisions if version in r.metadata["versions"]), None)

    chosen = [(p, rs, prim) for p, rs in groups.items() if (prim := primary_of(rs)) is not None]
    chosen.sort(key=lambda g: -g[2].rank_score)  # stable: ties keep the retriever's order

    out = []
    for path, revisions, prim in chosen[:top_k]:
        best = prim.chunks[0]
        snippet = reader.snippet(prim.doc_id, best.start_line, best.end_line)
        matched = snippet.text if snippet.status == "ok" else None
        hits = []
        for r in [prim] + [r for r in revisions if r is not prim]:
            contains = line = None
            if r is prim:
                contains, line = (True, best.start_line) if matched else (None, None)
            elif matched:
                text = reader.text(r.doc_id)
                if text is not None:
                    at = text.find(matched)
                    contains = at >= 0
                    line = text.count("\n", 0, at) + 1 if contains else None
            hits.append(RevisionHit(
                r.doc_id, sorted(r.metadata["versions"], key=lambda s: order.get(s, 0)), r.rank_score, r.score,
                r.chunks[0].start_line, r.chunks[0].end_line, r is prim, contains, line))
        out.append(VersionedResult(path, prim.kind, prim, snippet, hits))
    return out

