"""A folder of source files as Documents, one per file.

Document IDs are the file's path relative to the indexed root with forward slashes (e.g. "src/app/main.py"),
on every OS. They depend only on the path, never on walk order, so they stay stable across re-indexing and
the index's incremental update can match documents by ID and detect changes by content hash.

Skipped: SKIP_DIRS, anything matched by a .gitignore in the tree, symlinks, files over max_file_bytes,
binaries, files that are not valid UTF-8, empty / whitespace-only files, and minified files.

file_kind() classifies a path as "code", "test" or "docs". Retrieval uses it to down-weight or filter tests and
docs, which otherwise outrank the implementation on plain-language questions; `kinds=` restricts loading to
some kinds (`cli.py index --source-only`).
"""
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Callable, Collection, Iterator

import pathspec

from records import Document

CHUNKING = "windows"

SKIP_DIRS = frozenset({".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build"})
MAX_FILE_BYTES = 1_000_000
BINARY_SNIFF_BYTES = 8192  # a NUL byte in the first 8 KiB marks a file as binary (git uses the same rule)
# Minified: a ".min." name, or long lines throughout. Hand-written code rarely averages 300+ chars per line.
MINIFIED_MIN_CHARS = 2000
MINIFIED_MEAN_LINE_CHARS = 300

LANGUAGES = {
    ".py": "python", ".pyi": "python", ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".jsx": "javascript", ".ts": "typescript", ".tsx": "typescript", ".java": "java", ".kt": "kotlin",
    ".scala": "scala", ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp",
    ".cs": "csharp", ".go": "go", ".rs": "rust", ".rb": "ruby", ".php": "php", ".swift": "swift",
    ".m": "objective-c", ".dart": "dart", ".lua": "lua", ".r": "r", ".pl": "perl", ".sh": "shell",
    ".bash": "shell", ".ps1": "powershell", ".sql": "sql", ".html": "html", ".css": "css", ".scss": "scss",
    ".vue": "vue", ".md": "markdown", ".rst": "rst", ".json": "json", ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml", ".xml": "xml",
}

FILE_KINDS = ("code", "test", "docs")
_TEST_DIRS = frozenset({"test", "tests", "testing", "__tests__", "spec", "specs"})
_DOCS_DIRS = frozenset({"doc", "docs", "documentation"})
_DOCS_EXTENSIONS = frozenset({".md", ".rst", ".txt", ".adoc", ".rdoc"})
_DOCS_STEMS = frozenset({"readme", "changelog", "changes", "history", "license", "licence", "contributing",
                         "authors", "notice", "news"})


def file_kind(doc_id: str) -> str:
    """"test", "docs" or "code" for a root-relative forward-slash path. Decided from the path alone.

    test: under a tests/ (test, testing, __tests__, spec) directory, or named like a test file (test_x.py,
          x_test.py/go, x.test.js, x.spec.ts, conftest.py). Checked first: tests/README.md is a test file.
    docs: under a docs/ directory, a prose extension (.md, .rst, .txt, ...), or README/CHANGELOG/LICENSE/...
    code: everything else, including configuration.
    """
    parts = doc_id.lower().split("/")
    name = parts[-1]
    stem = name.split(".", 1)[0]
    suffix = "." + name.rsplit(".", 1)[-1] if "." in name else ""
    if any(d in _TEST_DIRS or d.startswith("tests_") for d in parts[:-1]):
        return "test"
    if (name.startswith("test_") or name == "conftest.py" or stem.endswith("_test")
            or ".test." in name or ".spec." in name):
        return "test"
    if any(d in _DOCS_DIRS for d in parts[:-1]) or suffix in _DOCS_EXTENSIONS or stem in _DOCS_STEMS:
        return "docs"
    return "code"


# on_skip(doc_id, reason) is called for every skipped file (not for pruned directories' contents).
SkipCallback = Callable[[str, str], None]


class _GitIgnores:
    """The .gitignore files found so far, keyed by the root-relative directory that holds them.

    For a path, the .gitignores of its ancestor directories are checked from the root down, and the deepest
    one with a matching pattern decides (so a nested file's "!pattern" can re-include what a parent ignored).
    Relies on insertion order: os.walk (top-down) adds a parent's .gitignore before its children's.
    """

    def __init__(self):
        self._specs: dict[PurePosixPath, pathspec.GitIgnoreSpec] = {}

    def add(self, rel_dir: PurePosixPath, gitignore: Path) -> None:
        try:
            lines = gitignore.read_text(encoding="utf-8-sig").splitlines()
        except (OSError, UnicodeDecodeError):
            return
        self._specs[rel_dir] = pathspec.GitIgnoreSpec.from_lines(lines)

    def ignored(self, rel_path: PurePosixPath, is_dir: bool) -> bool:
        ignored = False
        for rel_dir, spec in self._specs.items():
            if rel_dir != PurePosixPath(".") and rel_dir not in rel_path.parents:
                continue
            sub = rel_path.relative_to(rel_dir).as_posix() + ("/" if is_dir else "")
            include = spec.check_file(sub).include
            if include is not None:
                ignored = include
        return ignored


def _skip_reason(path: Path, name: str, max_file_bytes: int) -> tuple[str | None, str | None]:
    """(reason, None) if the file should be skipped, else (None, decoded text)."""
    if path.is_symlink():
        return "symlink", None
    try:
        if path.stat().st_size > max_file_bytes:
            return "too_large", None
        data = path.read_bytes()
    except OSError:
        return "unreadable", None
    if b"\0" in data[:BINARY_SNIFF_BYTES]:
        return "binary", None
    try:
        text = data.decode("utf-8-sig")  # strict; also drops a leading BOM so line 1 starts with real content
    except UnicodeDecodeError:
        return "not_utf8", None
    if not text.strip():
        return "empty", None
    if ".min." in name.lower():
        return "minified", None
    if len(text) >= MINIFIED_MIN_CHARS and len(text) / (text.count("\n") + 1) > MINIFIED_MEAN_LINE_CHARS:
        return "minified", None
    return None, text


def load_directory(
    root: str | os.PathLike,
    *,
    max_file_bytes: int = MAX_FILE_BYTES,
    skip_dirs: frozenset[str] = SKIP_DIRS,
    respect_gitignore: bool = True,
    kinds: Collection[str] | None = None,
    on_skip: SkipCallback | None = None,
) -> Iterator[Document]:
    """Yield one Document per eligible file under root, in sorted path order.

    kinds: if given, only files whose file_kind() is in it (others are skipped with reason "kind:<kind>").
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    gitignores = _GitIgnores()

    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        here = Path(dirpath)
        rel_dir = PurePosixPath(here.relative_to(root).as_posix())
        if respect_gitignore and ".gitignore" in filenames:
            gitignores.add(rel_dir, here / ".gitignore")

        # Prune in place so os.walk never descends into skipped directories.
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in skip_dirs
            and not (here / d).is_symlink()
            and not (respect_gitignore and gitignores.ignored(rel_dir / d, is_dir=True))
        )

        for name in sorted(filenames):
            rel_path = rel_dir / name
            doc_id = rel_path.as_posix()
            if respect_gitignore and gitignores.ignored(rel_path, is_dir=False):
                reason, text = "gitignored", None
            elif kinds is not None and (kind := file_kind(doc_id)) not in kinds:
                reason, text = f"kind:{kind}", None
            else:
                reason, text = _skip_reason(here / name, name, max_file_bytes)
            if reason is not None:
                if on_skip is not None:
                    on_skip(doc_id, reason)
                continue
            yield Document.create(
                id=doc_id,
                text=text,
                source_path=str(here / name),
                language=LANGUAGES.get(Path(name).suffix.lower()),
            )
