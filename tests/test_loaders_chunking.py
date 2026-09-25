"""Tests for the record types, loaders and chunkers (step 2). Loads only the tokenizer, not the model.

Run from the project root:  python -m tests.test_loaders_chunking
(Also collectable by pytest if it is installed.)
"""
import tempfile
from pathlib import Path

from transformers import AutoTokenizer

import loaders.apps
import loaders.directory
from chunking import make_chunker
from chunking.lines import split_lines
from chunking.windows import chunk_windows
from embedding.onnx_encoder import DEFAULT_MODEL
from loaders.apps import load_apps
from loaders.directory import load_directory
from records import Document, content_hash

_tokenizer = None


def _tok():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    return _tokenizer


def _write(root: Path, rel: str, content: str | bytes) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_bytes(content.encode("utf-8"))


def _load(root: Path, **kwargs) -> tuple[dict[str, Document], dict[str, str]]:
    skipped = {}
    docs = {d.id: d for d in load_directory(root, on_skip=lambda i, r: skipped.__setitem__(i, r), **kwargs)}
    return docs, skipped


# ---- records / lines ----

def test_document_create_sets_metadata():
    doc = Document.create("a.py", "x = 1\n", source_path="/abs/a.py", language="python", extra=1)
    assert doc.metadata == {"source_path": "/abs/a.py", "language": "python",
                            "content_hash": content_hash("x = 1\n"), "extra": 1}
    assert doc.content_hash == content_hash("x = 1\n") != content_hash("x = 2\n")


def test_split_lines():
    assert split_lines("") == []
    assert split_lines("a") == ["a"]
    assert split_lines("a\nb\n") == ["a\n", "b\n"]
    assert split_lines("a\r\nb") == ["a\r\n", "b"]
    assert split_lines("a\fb c\n") == ["a\fb c\n"]  # only \n breaks lines
    for text in ["", "a", "a\n\n", "\n", "x\r\ny\n\nz"]:
        assert "".join(split_lines(text)) == text


# ---- directory loader ----

def test_directory_ids_are_root_relative_posix_and_stable():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root, "src/pkg/mod.py", "def f():\n    return 1\n")
        _write(root, "main.py", "print('hi')\n")
        docs, _ = _load(root)
        assert set(docs) == {"main.py", "src/pkg/mod.py"}
        doc = docs["src/pkg/mod.py"]
        assert doc.metadata["language"] == "python"
        assert Path(doc.metadata["source_path"]) == (root / "src/pkg/mod.py").resolve()
        assert doc.content_hash == content_hash(doc.text)

        # Adding a file that sorts first must not change any existing ID or hash.
        _write(root, "aaa.py", "x = 1\n")
        docs2, _ = _load(root)
        assert set(docs2) == set(docs) | {"aaa.py"}
        for doc_id, d in docs.items():
            assert docs2[doc_id].content_hash == d.content_hash
        # Loading via a relative/unresolved spelling of the root gives the same IDs.
        docs3, _ = _load(root / "src" / "..")
        assert set(docs3) == set(docs2)


def test_directory_skips():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root, "keep.py", "x = 1\n")
        for d in [".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build"]:
            _write(root, f"{d}/inner.py", "x = 1\n")
            _write(root, f"sub/{d}/inner.py", "x = 1\n")  # skipped at any depth
        _write(root, "image.png", b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR")
        _write(root, "latin1.py", "caf\xe9 = 1\n".encode("latin-1"))
        _write(root, "big.py", "x = 1\n" * 2000)
        _write(root, "empty.py", "  \n\n")
        _write(root, "lib.min.js", "var a=1;\n")
        _write(root, "bundle.js", ("var a=1;" * 100 + "\n") * 5)  # 800-char lines
        docs, skipped = _load(root, max_file_bytes=10_000)
        assert set(docs) == {"keep.py"}, set(docs)
        assert skipped == {"image.png": "binary", "latin1.py": "not_utf8", "big.py": "too_large",
                           "empty.py": "empty", "lib.min.js": "minified", "bundle.js": "minified"}, skipped


def test_directory_bom_is_stripped():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root, "bom.py", b"\xef\xbb\xbfx = 1\n")
        docs, _ = _load(root)
        assert docs["bom.py"].text == "x = 1\n"


def test_directory_respects_gitignore():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root, ".gitignore", "*.log\nsecret/\ngenerated_*.py\n")
        _write(root, "app.py", "x = 1\n")
        _write(root, "run.log", "log line\n")
        _write(root, "secret/key.py", "k = 1\n")
        _write(root, "generated_a.py", "x = 1\n")
        _write(root, "pkg/.gitignore", "!keep.log\nlocal.py\n")  # nested file: re-include + extra ignore
        _write(root, "pkg/keep.log", "kept\n")
        _write(root, "pkg/other.log", "dropped\n")
        _write(root, "pkg/local.py", "x = 1\n")
        _write(root, "pkg/mod.py", "x = 1\n")
        _write(root, "local.py", "x = 1\n")  # pkg/.gitignore does not apply outside pkg/
        docs, skipped = _load(root)
        assert set(docs) == {".gitignore", "app.py", "pkg/.gitignore", "pkg/keep.log", "pkg/mod.py", "local.py"}, set(docs)
        assert skipped == {"run.log": "gitignored", "generated_a.py": "gitignored",
                           "pkg/other.log": "gitignored", "pkg/local.py": "gitignored"}, skipped  # secret/ pruned whole

        docs_all, _ = _load(root, respect_gitignore=False)
        assert {"run.log", "secret/key.py", "pkg/local.py"} <= set(docs_all)


# ---- chunking ----

def _check_windows(doc: Document, chunks, max_length: int) -> None:
    tok = _tok()
    lines = split_lines(doc.text)
    assert chunks[0].start_line == 1 and chunks[-1].end_line == len(lines)
    for c in chunks:
        assert c.doc_id == doc.id and c.chunk_id == f"{doc.id}#L{c.start_line}-L{c.end_line}"
        assert c.text == "".join(lines[c.start_line - 1:c.end_line])
        if c.end_line > c.start_line:  # multi-line chunks must fit without truncation
            assert len(tok(c.text)["input_ids"]) <= max_length, (c.chunk_id, len(tok(c.text)["input_ids"]))
    for a, b in zip(chunks, chunks[1:]):
        assert a.start_line < b.start_line <= a.end_line + 1  # progress, no gaps
        assert b.end_line > a.end_line  # no window repeats the previous one's end


def test_windows_cover_document_with_overlap():
    body = "".join(f"def f{i}(x):\n    return x * {i} + {i * 7}  # step {i}\n\n" for i in range(60))
    doc = Document.create("m.py", body, source_path="m.py", language="python")
    chunks = chunk_windows(doc, _tok(), max_length=64, overlap_tokens=16)
    assert len(chunks) > 5
    _check_windows(doc, chunks, 64)
    assert any(b.start_line <= a.end_line for a, b in zip(chunks, chunks[1:])), "expected overlapping windows"
    # Deterministic: same input, same chunk IDs.
    assert [c.chunk_id for c in chunk_windows(doc, _tok(), 64, 16)] == [c.chunk_id for c in chunks]


def test_windows_zero_overlap_and_crlf():
    body = "".join(f"line {i} of the file\r\n" for i in range(100))
    doc = Document.create("w.txt", body, source_path="w.txt", language=None)
    chunks = chunk_windows(doc, _tok(), max_length=48, overlap_tokens=0)
    _check_windows(doc, chunks, 48)
    assert all(b.start_line == a.end_line + 1 for a, b in zip(chunks, chunks[1:]))


def test_windows_long_line_gets_its_own_chunk():
    long_line = "x = [" + ", ".join(str(i) for i in range(400)) + "]\n"
    body = "a = 1\n" * 5 + long_line + "b = 2\n" * 5
    doc = Document.create("l.py", body, source_path="l.py", language="python")
    chunks = chunk_windows(doc, _tok(), max_length=64, overlap_tokens=16)
    _check_windows(doc, chunks, 64)
    assert any(c.start_line == c.end_line == 6 for c in chunks)


def test_windows_small_doc_is_one_chunk_and_empty_is_none():
    doc = Document.create("s.py", "x = 1\ny = 2", source_path="s.py", language="python")
    [chunk] = chunk_windows(doc, _tok(), max_length=1024)
    assert (chunk.start_line, chunk.end_line, chunk.text) == (1, 2, doc.text)
    assert chunk_windows(Document.create("e", "", source_path="e", language=None), _tok(), 1024) == []


def test_chunking_is_per_loader():
    assert loaders.apps.CHUNKING == "none" and loaders.directory.CHUNKING == "windows"
    doc = Document.create("d42", "a\nb\nc\n", source_path="x", language="python")
    [chunk] = make_chunker("none")(doc)
    assert (chunk.chunk_id, chunk.doc_id, chunk.text, chunk.start_line, chunk.end_line) == ("d42", "d42", doc.text, 1, 3)
    windows = make_chunker("windows", tokenizer=_tok(), max_length=1024)
    assert [c.chunk_id for c in windows(doc)] == ["d42#L1-L3"]


# ---- apps loader ----

def test_apps_loader_uses_dataset_ids():
    docs = list(load_apps())
    assert len(docs) == 8765
    ids = [d.id for d in docs]
    assert len(set(ids)) == len(ids) and "d1" in ids and "d5001" in ids and "d8765" in ids
    d = docs[0]
    assert d.metadata["language"] == "python" and d.metadata["partition"] in {"train", "test"}
    assert d.content_hash == content_hash(d.text)


def test_apps_text_matches_what_mteb_encodes():
    from datasets import load_dataset
    from mteb._create_dataloaders import _corpus_to_dict

    raw = load_dataset(loaders.apps.DATASET, "corpus", split="corpus", revision=loaders.apps.REVISION)
    ours = {d.id: d.text for d in load_apps()}
    mismatched = [row["_id"] for row in raw
                  if ours[row["_id"]] != _corpus_to_dict({"id": row["_id"], "title": row["title"], "text": row["text"]})["text"]]
    assert not mismatched, mismatched[:5]
    assert sum(ours[row["_id"]] != row["text"] for row in raw) > 0  # stripping really changes some texts


def test_apps_queries_and_qrels():
    queries, qrels = loaders.apps.load_apps_queries()
    assert len(queries) == len(qrels) == 3765
    assert list(queries)[0] == "q5001" and qrels["q5001"] == {"d5001": 1}
    assert all(len(rels) == 1 for rels in qrels.values())


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
