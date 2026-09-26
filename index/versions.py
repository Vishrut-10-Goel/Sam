"""Versioned index: one index over several commits of a git repository (retrieval across versions).

The unit stored is a *revision*: a file's content at some commit, keyed by (path, content hash), with index id
"<path>@<hash12>". A file that is identical in several commits is one revision listing all of them
(metadata["versions"]), so its chunks are embedded once. Adding commits keeps the ones already indexed and embeds
only revisions not seen before: the same content-hash reuse as index.update_index, applied across commits.

Commits are read from a temporary detached git worktree, checked out commit by commit and removed at the end, so
the repository's own working tree (and anything uncommitted in it) is never touched. Files are loaded with the
same rules as a folder index (loaders.directory) and chunked under their plain path, so each chunk embeds exactly
what a folder index of that commit would embed; only the chunk and document ids gain the "@<hash12>" suffix.

The result is an ordinary Index (saved and loaded by index.storage) with source
{"kind": "git", "root": <repo>, "commits": [{sha, short, date, subject, tags}, ...oldest first]}.
Snippets are read back from git objects (retrieval.snippets), so they never go stale.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np

import loaders.directory
from index.index import ChunkRow, DocEntry, Index, IndexMismatchError, ProgressCallback, _chunker
from records import Document

HASH_CHARS = 12


class GitError(RuntimeError):
    pass


def git(repo: Path, *args: str, text: bool = True) -> str | bytes:
    """Run git in repo; raise GitError with git's message on failure."""
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=False)
    if r.returncode != 0:
        raise GitError(f"git {' '.join(args)}: {r.stderr.decode('utf-8', 'replace').strip()}")
    return r.stdout.decode("utf-8", "replace") if text else r.stdout


def text_key(text: str) -> str:
    """Identifies an embedded text (a chunk's header + lines): equal keys, equal vectors."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def revision_id(path: str, content_hash: str) -> str:
    return f"{path}@{content_hash[:HASH_CHARS]}"


def is_versioned(index: Index) -> bool:
    return index.source.get("kind") == "git"


def commit_info(repo: Path, rev: str) -> dict:
    """{sha, short, date, subject, tags} for a commit-ish (sha, tag, branch, HEAD~3...)."""
    sha = git(repo, "rev-parse", "--verify", f"{rev}^{{commit}}").strip()
    date, subject = git(repo, "log", "-1", "--format=%cI%x00%s", sha).rstrip("\n").split("\x00", 1)
    tags = sorted(t for t in git(repo, "tag", "--points-at", sha).split() if t)
    return {"sha": sha, "short": sha[:7], "date": date, "subject": subject, "tags": tags}


def last_commits(repo: Path, n: int, rev: str = "HEAD") -> list[str]:
    """The last n commits on rev's first-parent line, oldest first."""
    return list(reversed(git(repo, "rev-list", "--first-parent", f"--max-count={n}", rev).split()))


def version_label(info: dict) -> str:
    """How a commit is shown: its first tag if it has one, else its short sha."""
    return info["tags"][0] if info.get("tags") else info["short"]


def resolve_version(index: Index, version: str) -> str:
    """The full sha of an indexed commit named by sha prefix (4+ chars) or tag. Raises ValueError if none/ambiguous."""
    commits = index.source.get("commits", [])
    v = version.strip()
    hits = [c["sha"] for c in commits if v in c.get("tags", []) or (len(v) >= 4 and c["sha"].startswith(v.lower()))]
    if len(set(hits)) != 1:
        known = ", ".join(version_label(c) for c in commits)
        raise ValueError(f"{version!r} {'is ambiguous' if hits else 'is not an indexed version'} (indexed: {known})")
    return hits[0]


def _load_commit(worktree: Path, sha: str, on_skip) -> list[Document]:
    git(worktree, "checkout", "--quiet", "--force", "--detach", sha)
    # A worktree's .git is a file ("gitdir: ..."), not a directory, so the loader's .git rule does not skip it.
    return [d for d in loaders.directory.load_directory(worktree, on_skip=on_skip) if d.id != ".git"]


def index_versions(
    repo: str | Path,
    commits: list[str],
    encoder,
    chunking: dict,
    existing: Index | None = None,
    progress: ProgressCallback | None = None,
    note: Callable[[str], None] = print,
) -> tuple[Index, dict]:
    """Index each commit (sha, tag or other commit-ish) of repo, adding to `existing` if given.

    Returns the new Index and stats: per commit {commit, files, revisions_reused, revisions_new, chunks_embedded,
    seconds}, plus totals and chunks_if_separate (what one full build per indexed commit would embed).
    """
    repo = Path(repo).resolve()
    git(repo, "rev-parse", "--git-dir")  # fails clearly if this is not a git repository
    if existing is not None:
        existing.check_compatible(encoder, chunking)
        if not is_versioned(existing):
            raise IndexMismatchError("the existing index is not a versioned (git) index")
    infos = {c["sha"]: c for c in (existing.source.get("commits", []) if existing else [])}
    wanted = []
    for rev in commits:
        info = commit_info(repo, rev)
        if info["sha"] not in infos and info["sha"] not in {w["sha"] for w in wanted}:
            wanted.append(info)
    wanted.sort(key=lambda c: c["date"])

    matrix = existing.embeddings if existing else np.empty((0, encoder.dim), dtype=np.float32)
    rows = list(existing.chunks) if existing else []
    entries = {d: DocEntry(e.content_hash, list(e.chunk_ids), {**e.metadata, "versions": list(e.metadata["versions"]),
                                                              "chunk_hashes": list(e.metadata.get("chunk_hashes", []))})
               for d, e in (existing.documents.items() if existing else [])}
    # Chunk-level reuse: a changed file's chunks above its first edit keep their text (and header), so they reuse the
    # vector already computed for that text. known maps a hash of the embedded text to a row holding its vector; the
    # hashes are stored per revision ("chunk_hashes", aligned with chunk_ids) so this survives save/load.
    row_of = {r.chunk_id: i for i, r in enumerate(rows)}
    known: dict[str, int] = {}
    for e in entries.values():
        for chunk_id, key in zip(e.chunk_ids, e.metadata["chunk_hashes"]):
            known.setdefault(key, row_of[chunk_id])
    chunker = _chunker(chunking, encoder)
    per_commit = []
    start_all = time.perf_counter()
    if wanted:
        tmp = Path(tempfile.mkdtemp(prefix="prism-versions-"))
        worktree = tmp / "wt"
        git(repo, "worktree", "add", "--detach", "--force", str(worktree), wanted[0]["sha"])
        try:
            for n, info in enumerate(wanted, 1):
                t0 = time.perf_counter()
                docs = _load_commit(worktree, info["sha"], None)
                new_docs, reused = [], 0
                for doc in docs:
                    rid = revision_id(doc.id, doc.content_hash)
                    if rid in entries:
                        entries[rid].metadata["versions"].append(info["sha"])
                        reused += 1
                    else:
                        new_docs.append(doc)
                # Chunk under the plain path (headers name the file, exactly as in a folder index), then give the
                # chunks the revision id. Only texts never embedded before go to the encoder.
                chunks = [c for doc in new_docs for c in chunker(doc)]
                keys = [text_key(c.embed_text) for c in chunks]
                text_of = dict(zip(keys, (c.embed_text for c in chunks)))
                to_embed = list(dict.fromkeys(k for k in keys if k not in known))
                fresh = (np.asarray(encoder.encode([text_of[k] for k in to_embed], progress=progress), dtype=np.float32)
                         if to_embed else np.empty((0, encoder.dim), dtype=np.float32))
                fresh_row = {k: i for i, k in enumerate(to_embed)}
                block = np.stack([fresh[fresh_row[k]] if k in fresh_row else matrix[known[k]] for k in keys])                     if keys else np.empty((0, encoder.dim), dtype=np.float32)
                base = len(rows)
                for i, k in enumerate(keys):
                    known.setdefault(k, base + i)
                matrix = np.concatenate([matrix, block])
                hashes = {d.id: d.content_hash for d in new_docs}  # one commit: paths are unique
                for doc in new_docs:
                    rid = revision_id(doc.id, doc.content_hash)
                    entries[rid] = DocEntry(doc.content_hash, [], {
                        "path": doc.id, "language": doc.metadata.get("language"), "versions": [info["sha"]],
                        "source_path": f"{info['sha'][:HASH_CHARS]}:{doc.id}", "chunk_hashes": []})
                for c, k in zip(chunks, keys):
                    rid = revision_id(c.doc_id, hashes[c.doc_id])
                    row = ChunkRow(f"{rid}#L{c.start_line}-L{c.end_line}", rid, c.start_line, c.end_line)
                    rows.append(row)
                    entries[rid].chunk_ids.append(row.chunk_id)
                    entries[rid].metadata["chunk_hashes"].append(k)
                stat = {"commit": info["sha"], "label": version_label(info), "files": len(docs),
                        "revisions_reused": reused, "revisions_new": len(new_docs), "chunks_new": len(chunks),
                        "chunks_embedded": len(to_embed), "chunks_copied": len(chunks) - len(to_embed),
                        "seconds": round(time.perf_counter() - t0, 1)}
                per_commit.append(stat)
                note(f"[{n}/{len(wanted)}] {stat['label']} ({info['date'][:10]}): {len(docs)} files, "
                     f"{reused} unchanged since an indexed version, {len(new_docs)} new or changed "
                     f"({len(chunks)} chunks: {len(to_embed)} embedded, {stat['chunks_copied']} identical to an "
                     f"indexed chunk) in {stat['seconds']} s")
        finally:
            try:
                git(repo, "worktree", "remove", "--force", str(worktree))
            except GitError:
                pass
            shutil.rmtree(tmp, ignore_errors=True)
            try:
                git(repo, "worktree", "prune")
            except GitError:
                pass
    else:
        note("all requested commits are already indexed")

    all_commits = sorted([*infos.values(), *wanted], key=lambda c: c["date"])
    source = {"kind": "git", "root": str(repo), "commits": all_commits}
    index = Index(np.ascontiguousarray(matrix, dtype=np.float32), rows, entries, encoder.fingerprint(),
                  dict(chunking), source)
    stats = {"commits_added": per_commit, "seconds": round(time.perf_counter() - start_all, 1),
             **version_stats(index)}
    return index, stats


def version_stats(index: Index) -> dict:
    """Totals for a versioned index: how much one-build-per-commit would have embedded, and what was stored."""
    per_commit_chunks: dict[str, int] = {}
    for entry in index.documents.values():
        for sha in entry.metadata["versions"]:
            per_commit_chunks[sha] = per_commit_chunks.get(sha, 0) + len(entry.chunk_ids)
    separate = sum(per_commit_chunks.values())
    return {"commits": len(index.source.get("commits", [])), "revisions": len(index.documents),
            "chunks": len(index.chunks), "chunks_if_separate": separate,
            "chunks_shared_pct": round(100 * (1 - len(index.chunks) / separate), 1) if separate else 0.0}
