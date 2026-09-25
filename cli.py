"""Command-line interface for the code-retrieval system.

  python cli.py index <path> [--out DIR] [--rebuild]
      Index a folder of source files. If DIR already holds an index for this encoder, it is updated
      incrementally: only new and changed files are re-embedded, deleted files are dropped.
      Default DIR: indexes/<folder name>-<hash8>, the hash of the folder's absolute path, so folders that
      share a name never share an index.
  python cli.py query "<text>" --index DIR [--top-k N] [--json]
      Search an index (a folder index, or indexes/apps). Snippets are read from the source and flagged
      "stale" if it changed since indexing (re-run `index` to refresh).
  python cli.py eval-apps [--limit N] [--index DIR] ...
      Real-pipeline AppsRetrieval evaluation (see eval/apps_pipeline.py).

Results go to stdout; progress and summaries go to stderr, so --json output can be piped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import loaders.directory
from index import IndexMismatchError, build_index, chunking_config, load_index, save_index, update_index
from index.storage import MANIFEST

SNIPPET_LINES = 12  # lines of the best chunk shown per text result


def log(*args) -> None:
    print(*args, file=sys.stderr, flush=True)


def _make_encoder():
    from embedding.onnx_encoder import OnnxEncoder  # imported lazily: loading onnxruntime takes a moment
    return OnnxEncoder()


def default_index_dir(root: Path) -> Path:
    """indexes/<name>-<hash8>. normcase: Windows paths are case-insensitive, so D:\Repo and d:\repo are one folder."""
    digest = hashlib.sha256(os.path.normcase(str(root.resolve())).encode("utf-8")).hexdigest()[:8]
    return Path("indexes") / f"{root.resolve().name}-{digest}"


def _progress(label: str):
    t0 = last = time.perf_counter()

    def report(done: int, total: int) -> None:
        nonlocal last
        now = time.perf_counter()
        if now - last >= 10 or done == total:
            last = now
            log(f"  embedded {done:,}/{total:,} {label} ({(now - t0) / 60:.1f} min)")

    return report


def cmd_index(args) -> int:
    root = Path(args.path).resolve()
    if not root.is_dir():
        log(f"error: {root} is not a directory")
        return 2
    out = (args.out or default_index_dir(root)).resolve()
    start = time.perf_counter()
    encoder = _make_encoder()

    skipped: Counter = Counter()
    # An index stored inside the folder it indexes must not index itself.
    docs = [d for d in loaders.directory.load_directory(root, on_skip=lambda _, reason: skipped.update([reason]))
            if out not in Path(d.metadata["source_path"]).parents]
    log(f"{root}: {len(docs):,} files" + (f"; skipped {dict(skipped)}" if skipped else ""))
    chunking = chunking_config(loaders.directory.CHUNKING, encoder)
    source = {"kind": "directory", "root": str(root)}

    if (out / MANIFEST).exists() and not args.rebuild:
        try:
            index = load_index(out)
            index.check_compatible(encoder, chunking)
        except IndexMismatchError as e:
            log(f"error: {e}\n(re-run with --rebuild to rebuild {out} from scratch)")
            return 2
        if index.source.get("root") not in (None, str(root)):
            log(f"note: {out} was built from {index.source['root']}; updating it to {root}")
        index, stats = update_index(index, docs, encoder, source, _progress("chunks"))
        log(f"updated: {stats.added} added, {stats.changed} changed, {stats.deleted} deleted, "
            f"{stats.unchanged} unchanged; {stats.chunks_embedded} chunks embedded")
    else:
        index = build_index(docs, encoder, chunking, source, _progress("chunks"))
        log(f"built: {len(index.documents):,} files, {len(index.chunks):,} chunks")
    save_index(index, out)
    log(f"saved {out} in {time.perf_counter() - start:.1f} s")
    return 0


def cmd_query(args) -> int:
    from retrieval import Retriever, SnippetReader

    encoder = _make_encoder()
    try:
        index = load_index(args.index, encoder=encoder)
    except FileNotFoundError:
        log(f"error: no index at {args.index} (build one with: python cli.py index <path>)")
        return 2
    except IndexMismatchError as e:
        log(f"error: {e}")
        return 2
    results = Retriever(index, encoder).search(args.text, top_k=args.top_k)
    reader = SnippetReader(index)

    rows = []
    for rank, r in enumerate(results, 1):
        best = r.chunks[0]
        snippet = reader.snippet(r.doc_id, best.start_line, best.end_line)
        rows.append({
            "rank": rank,
            "doc_id": r.doc_id,
            "score": round(r.score, 6),
            "source_path": r.metadata.get("source_path"),
            "language": r.metadata.get("language"),
            "chunks": [{"chunk_id": c.chunk_id, "start_line": c.start_line, "end_line": c.end_line,
                        "score": round(c.score, 6)} for c in r.chunks],
            "snippet": {"status": snippet.status, "text": snippet.text},
        })

    if args.json:
        print(json.dumps(rows, indent=1, ensure_ascii=False))
        return 0
    if not rows:
        print("no results")
    for row in rows:
        best = row["chunks"][0]
        print(f"{row['rank']:>2}. {row['doc_id']}:{best['start_line']}-{best['end_line']}   score {row['score']:.4f}")
        status, text = row["snippet"]["status"], row["snippet"]["text"]
        if status != "ok":
            print(f"    [{status}: source {'changed since indexing; re-run index' if status == 'stale' else 'not found'}]")
            continue
        lines = text.splitlines()
        for i, line in enumerate(lines[:SNIPPET_LINES]):
            print(f"    {best['start_line'] + i:>5} | {line}")
        if len(lines) > SNIPPET_LINES:
            print(f"          ... {len(lines) - SNIPPET_LINES} more lines")
        print()
    return 0


def cmd_eval_apps(args, rest: list[str]) -> int:
    from eval import apps_pipeline
    argv = (["--limit", str(args.limit)] if args.limit is not None else []) + rest
    return apps_pipeline.main(argv)


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to a legacy code page; source files routinely contain characters outside it.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(prog="cli.py", description="Code retrieval: index folders, search, evaluate.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("index", help="index (or incrementally update the index of) a folder of source files")
    p.add_argument("path", help="folder to index")
    p.add_argument("--out", type=Path, help="index directory (default: indexes/<folder name>-<hash8>)")
    p.add_argument("--rebuild", action="store_true", help="rebuild from scratch instead of updating")

    p = sub.add_parser("query", help="search an index")
    p.add_argument("text", help="natural-language or code query")
    p.add_argument("--index", type=Path, required=True, help="index directory")
    p.add_argument("--top-k", type=int, default=10, help="number of results (default 10)")
    p.add_argument("--json", action="store_true", help="print results as JSON")

    p = sub.add_parser("eval-apps", help="real-pipeline AppsRetrieval evaluation (NDCG@10, MRR@10)",
                       description="Other options (--index, --output, --mteb-results) pass through to "
                                   "eval/apps_pipeline.py.")
    p.add_argument("--limit", type=int, default=None, help="evaluate only the first N test queries")

    args, rest = parser.parse_known_args(argv)
    if args.command == "eval-apps":
        return cmd_eval_apps(args, rest)
    if rest:
        parser.error(f"unrecognized arguments: {' '.join(rest)}")
    if args.command == "query" and args.top_k < 1:
        parser.error("--top-k must be >= 1")
    return cmd_index(args) if args.command == "index" else cmd_query(args)


if __name__ == "__main__":
    sys.exit(main())
