"""Command-line interface for the code-retrieval system.

  python cli.py index <path> [--out DIR] [--rebuild] [--source-only] [--no-header]
      Index a folder of source files. If DIR already holds an index for this encoder, it is updated
      incrementally: only new and changed files are re-embedded, deleted files are dropped.
      Each chunk is embedded with a header naming its file and enclosing classes/functions (--no-header to
      turn off); --source-only skips test and docs files (see loaders.directory.file_kind).
      Default DIR: indexes/<folder name>-<hash8>, the hash of the folder's absolute path, so folders that
      share a name never share an index.
  python cli.py query "<text>" --index DIR [--top-k N] [--json] [--code-only | --kind-penalty X]
  python cli.py query --index DIR --interactive [--top-k N] [--json] [--code-only | --kind-penalty X]
      Search an index (a folder index, or indexes/apps). Snippets are read from the source and flagged
      "stale" if it changed since indexing (re-run `index` to refresh). --interactive loads the model once
      and answers queries at a prompt with no reload; it picks up a re-saved index automatically.
      Test and docs files rank below code by default (--kind-penalty, default 0.05; 0 turns it off);
      --code-only leaves them out. Neither affects the apps index, whose documents are all code.
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
# Subtracted from test and docs files' scores before ranking. On the pre-registered Scrapy questions
# (reports/scrapy_retrieval_check.md) any value from 0.05 up took strict top-1 from 5/10 to 8/10.
DEFAULT_KIND_PENALTY = 0.05


def log(*args) -> None:
    print(*args, file=sys.stderr, flush=True)


# The encoder's model (embedding.onnx_encoder.DEFAULT_MODEL) and the apps dataset (loaders.apps.DATASET) as
# Hugging Face cache directory names. Kept as strings so this check needs no Hugging Face import.
_MODEL_CACHE_NAME = "models--Alibaba-NLP--gte-modernbert-base"
_APPS_CACHE_NAME = "CoIR-Retrieval___apps"


def use_offline_hub_if_cached(needs_apps_dataset: bool) -> bool:
    """Set HF_HUB_OFFLINE=1 when everything this command downloads is already cached. Returns whether it did.

    Offline, Hugging Face libraries make no network calls: no "unauthenticated requests" warning in demos and
    a faster start. It is only set when the model's files (and, for apps commands, the dataset) are cached, so
    a first run can still download them; an explicit HF_HUB_OFFLINE in the environment always wins. Must run
    before huggingface_hub is imported, which reads the variable once at import.
    """
    if "HF_HUB_OFFLINE" in os.environ or "huggingface_hub" in sys.modules:
        return False
    hf_home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    snapshots = Path(os.environ.get("HF_HUB_CACHE") or hf_home / "hub") / _MODEL_CACHE_NAME / "snapshots"
    if not (any(snapshots.glob("*/onnx/model.onnx")) and any(snapshots.glob("*/tokenizer.json"))):
        return False
    if needs_apps_dataset:
        datasets_cache = Path(os.environ.get("HF_DATASETS_CACHE") or hf_home / "datasets")
        if not (datasets_cache / _APPS_CACHE_NAME).is_dir():
            return False
    os.environ["HF_HUB_OFFLINE"] = "1"
    return True


def _index_is_apps(index_dir: Path) -> bool:
    try:
        return json.loads((index_dir / MANIFEST).read_text(encoding="utf-8")).get("source", {}).get("kind") == "apps"
    except (OSError, ValueError):
        return False


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
    kinds = {"code"} if args.source_only else None
    docs = [d for d in loaders.directory.load_directory(root, kinds=kinds,
                                                        on_skip=lambda _, reason: skipped.update([reason]))
            if out not in Path(d.metadata["source_path"]).parents]
    log(f"{root}: {len(docs):,} files" + (f"; skipped {dict(skipped)}" if skipped else ""))
    chunking = chunking_config(loaders.directory.CHUNKING, encoder, header=not args.no_header)
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


class QuerySession:
    """A loaded encoder + index + retriever, reused across queries (the model loads once).

    Before each query it checks manifest.json: if the index was re-saved (e.g. `cli.py index` run in another
    terminal after an edit), the new index is loaded without reloading the model. Snippets are re-read from
    files that changed on disk, so edits made mid-session show up as stale until re-indexed.
    """

    def __init__(self, index_dir: Path, encoder, kind_penalty: float = DEFAULT_KIND_PENALTY, code_only: bool = False):
        self.index_dir = Path(index_dir)
        self.encoder = encoder
        self.retriever_options = {"kind_penalty": kind_penalty, "code_only": code_only}
        self._stamp = None
        self._load()

    def _manifest_stamp(self):
        st = (self.index_dir / MANIFEST).stat()
        return st.st_size, st.st_mtime_ns

    def _load(self) -> None:
        from retrieval import Retriever, SnippetReader
        stamp = self._manifest_stamp()
        index = load_index(self.index_dir, encoder=self.encoder)
        retriever = Retriever(index, self.encoder, **self.retriever_options)
        self.index, self.retriever, self.reader = index, retriever, SnippetReader(index)
        self._stamp = stamp

    def refresh(self, force: bool = False) -> None:
        """Reload the index if it was re-saved since it was loaded. On failure, keep serving the loaded one."""
        try:
            if not force and self._manifest_stamp() == self._stamp:
                return
            self._load()
            log(f"(index reloaded: {len(self.index.documents):,} files, {len(self.index.chunks):,} chunks)")
        except (OSError, ValueError) as e:  # mid-save, or rebuilt with another encoder (IndexMismatchError)
            log(f"(index at {self.index_dir} could not be reloaded, still using the loaded one: {e})")

    def search(self, text: str, top_k: int) -> list[dict]:
        rows = []
        for rank, r in enumerate(self.retriever.search(text, top_k=top_k), 1):
            best = r.chunks[0]
            snippet = self.reader.snippet(r.doc_id, best.start_line, best.end_line)
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
        return rows


def print_rows(rows: list[dict], as_json: bool) -> None:
    if as_json:
        print(json.dumps(rows, indent=1, ensure_ascii=False), flush=True)
        return
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
    sys.stdout.flush()


INTERACTIVE_HELP = """\
Type a query and press Enter. Commands:
  :paste      multi-line query (e.g. a whole problem statement); end it with a line containing only "."
  :k N        show N results per query (now {top_k})
  :json       toggle JSON output (now {json})
  :reload     reload the index from disk (also automatic when it is re-saved)
  :help       this help
  :q          quit (or end of input: Ctrl+Z Enter on Windows, Ctrl+D elsewhere)"""


def _read_line(prompt: str) -> str | None:
    """One line from stdin without its newline, or None at end of input. The prompt goes to stderr."""
    print(prompt, end="", file=sys.stderr, flush=True)
    line = sys.stdin.readline()
    return None if line == "" else line.rstrip("\r\n")


def run_interactive(session: QuerySession, top_k: int, as_json: bool, first: str | None) -> int:
    log(INTERACTIVE_HELP.format(top_k=top_k, json="on" if as_json else "off"))
    pending = [first] if first else []
    while True:
        try:
            text = pending.pop() if pending else _read_line("\nquery> ")
        except KeyboardInterrupt:
            log("")
            return 0
        if text is None or text.strip() in (":q", ":quit", ":exit"):
            return 0
        cmd = text.strip()
        if not cmd:
            continue
        if cmd == ":help":
            log(INTERACTIVE_HELP.format(top_k=top_k, json="on" if as_json else "off"))
            continue
        if cmd == ":json":
            as_json = not as_json
            log(f"(JSON output {'on' if as_json else 'off'})")
            continue
        if cmd == ":reload":
            session.refresh(force=True)
            continue
        if cmd == ":k" or cmd.startswith(":k "):
            try:
                top_k = int(cmd[2:])
                if top_k < 1:
                    raise ValueError
                log(f"(showing {top_k} results)")
            except ValueError:
                log("usage: :k N   (N >= 1)")
            continue
        if cmd == ":paste":
            lines = []
            while (line := _read_line("... ")) is not None and line.strip() != ".":
                lines.append(line)
            text = "\n".join(lines)
            if not text.strip():
                continue
        elif cmd.startswith(":"):
            log(f"unknown command {cmd.split()[0]!r}; :help lists commands")
            continue

        try:
            session.refresh()
            t0 = time.perf_counter()
            rows = session.search(text, top_k)
        except KeyboardInterrupt:
            log("(interrupted)")
            continue
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print_rows(rows, as_json)
        log(f"({len(rows)} results in {elapsed_ms:.0f} ms)")


def cmd_query(args) -> int:
    if not args.interactive and args.text is None:
        log("error: a query text is required (or use --interactive)")
        return 2
    t0 = time.perf_counter()
    encoder = _make_encoder()
    try:
        session = QuerySession(args.index, encoder, kind_penalty=args.kind_penalty, code_only=args.code_only)
    except FileNotFoundError:
        log(f"error: no index at {args.index} (build one with: python cli.py index <path>)")
        return 2
    except IndexMismatchError as e:
        log(f"error: {e}")
        return 2
    if args.interactive:
        log(f"model and index loaded in {time.perf_counter() - t0:.1f} s: "
            f"{len(session.index.documents):,} files, {len(session.index.chunks):,} chunks")
        return run_interactive(session, args.top_k, args.json, args.text)
    print_rows(session.search(args.text, args.top_k), args.json)
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
    p.add_argument("--source-only", action="store_true", help="skip test and docs files (index code only)")
    p.add_argument("--no-header", action="store_true",
                   help="embed chunks without the file path / class / function header")

    p = sub.add_parser("query", help="search an index")
    p.add_argument("text", nargs="?", help="natural-language or code query (optional with --interactive)")
    p.add_argument("--index", type=Path, required=True, help="index directory")
    p.add_argument("--interactive", "-i", action="store_true",
                   help="load the model once, then answer queries typed at a prompt (:help for commands)")
    p.add_argument("--top-k", type=int, default=10, help="number of results (default 10)")
    p.add_argument("--json", action="store_true", help="print results as JSON")
    kind = p.add_mutually_exclusive_group()
    kind.add_argument("--code-only", action="store_true", help="leave test and docs files out of the results")
    kind.add_argument("--kind-penalty", type=float, default=DEFAULT_KIND_PENALTY,
                      help=f"score subtracted from test and docs files before ranking (default {DEFAULT_KIND_PENALTY}; "
                           "0 ranks them like code)")

    p = sub.add_parser("eval-apps", help="real-pipeline AppsRetrieval evaluation (NDCG@10, MRR@10)",
                       description="Other options (--index, --output, --mteb-results) pass through to "
                                   "eval/apps_pipeline.py.")
    p.add_argument("--limit", type=int, default=None, help="evaluate only the first N test queries")

    args, rest = parser.parse_known_args(argv)
    use_offline_hub_if_cached(needs_apps_dataset=args.command == "eval-apps"
                              or (args.command == "query" and _index_is_apps(args.index)))
    if args.command == "eval-apps":
        return cmd_eval_apps(args, rest)
    if rest:
        parser.error(f"unrecognized arguments: {' '.join(rest)}")
    if args.command == "query" and args.top_k < 1:
        parser.error("--top-k must be >= 1")
    if args.command == "query" and args.kind_penalty < 0:
        parser.error("--kind-penalty must be >= 0")
    return cmd_index(args) if args.command == "index" else cmd_query(args)


if __name__ == "__main__":
    sys.exit(main())
