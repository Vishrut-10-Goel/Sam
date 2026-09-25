"""Tests for the opt-in AST chunker (chunking/ast_chunks.py): mechanics only, not retrieval quality.

Run from the project root:  python -m tests.test_ast_chunks
(Also collectable by pytest if it is installed.)
"""
import tempfile
from pathlib import Path

from chunking import make_chunker
from chunking.ast_chunks import MIN_TOKENS, TARGET_TOKENS, chunk_ast
from chunking.lines import split_lines
from chunking.windows import chunk_windows
from index import chunking_config
from records import Document
from tests.test_index import FakeEncoder, _write
from tests.test_loaders_chunking import _tok


def _n(text: str) -> int:
    return len(_tok()(text, add_special_tokens=False)["input_ids"])


def _check_cover(doc: Document, chunks, allow_overlap: bool = False) -> None:
    """Chunks are whole lines, exact text, in order, covering every line; no gaps (and no overlap unless allowed)."""
    lines = split_lines(doc.text)
    assert chunks[0].start_line == 1 and chunks[-1].end_line == len(lines), (chunks[0], chunks[-1])
    for c in chunks:
        assert c.text == "".join(lines[c.start_line - 1:c.end_line])
        assert c.chunk_id == f"{doc.id}#L{c.start_line}-L{c.end_line}"
    for a, b in zip(chunks, chunks[1:]):
        assert b.start_line > a.start_line and b.end_line > a.end_line
        if allow_overlap:
            assert b.start_line <= a.end_line + 1
        else:
            assert b.start_line == a.end_line + 1, (a.chunk_id, b.chunk_id)


def _fn(name: str, body_lines: int) -> str:
    return f"def {name}(x):\n" + "".join(f"    x = x * {i} + {i + 1}  # step {i} of {name}\n" for i in range(body_lines)) + "    return x\n\n\n"


def test_small_functions_are_merged_whole():
    src = '"""Module docstring."""\nimport os\nimport sys\n\n\n' + "".join(_fn(f"f{i}", 3) for i in range(30))
    doc = Document.create("pkg/small.py", src, "x", "python")
    chunks = chunk_ast(doc, _tok(), 1024)
    _check_cover(doc, chunks)
    assert 1 < len(chunks) < 30  # merged, not one chunk per 5-line function
    for c in chunks:
        assert _n(c.text) <= TARGET_TOKENS
        # Merged chunks start at a function boundary (or the top of the file), never mid-function.
        assert c.start_line == 1 or c.text.startswith("def ")
    assert all(_n(c.text) >= MIN_TOKENS for c in chunks[:-1])  # only the last may stay small


def test_big_class_splits_by_method_and_keeps_decorators_and_comments():
    methods = "".join(
        f"    # helper comment for m{i}\n    @staticmethod\n    def m{i}(x):\n"
        + "".join(f"        x = x + {j}  # method m{i} line {j}\n" for j in range(12)) + "        return x\n\n"
        for i in range(12))
    src = "import os\n\n\nclass Big:\n    \"\"\"A big class.\"\"\"\n    LIMIT = 3\n\n" + methods
    doc = Document.create("pkg/big.py", src, "x", "python")
    chunks = chunk_ast(doc, _tok(), 1024)
    _check_cover(doc, chunks)
    assert len(chunks) > 2 and all(_n(c.text) <= TARGET_TOKENS for c in chunks)
    for c in chunks[1:]:
        # A method chunk starts at its comment block, so the comment and decorator stay with the method.
        assert c.text.lstrip().startswith("# helper comment for m"), c.text[:60]


def test_oversized_function_falls_back_to_windows():
    doc = Document.create("pkg/long.py", "import os\n\n\n" + _fn("huge", 300), "x", "python")
    chunks = chunk_ast(doc, _tok(), 1024)
    _check_cover(doc, chunks, allow_overlap=True)
    assert len(chunks) > 3 and all(_n(c.text) <= TARGET_TOKENS for c in chunks)
    assert any(b.start_line <= a.end_line for a, b in zip(chunks, chunks[1:]))  # windows overlap inside the unit


def test_non_python_and_unparsable_use_line_windows():
    for doc_id, text in [("docs/guide.md", "# Title\n" + "Some text here.\n" * 400),
                         ("broken.py", "def f(:\n" + "    pass\n" * 400)]:
        doc = Document.create(doc_id, text, "x", None)
        assert chunk_ast(doc, _tok(), 256, 16) == chunk_windows(doc, _tok(), 256, 16)
    assert chunk_ast(Document.create("empty.py", "", "x", "python"), _tok(), 1024) == []


def test_headers_and_token_budget():
    src = "import os\n\n\nclass Big:\n" + "".join(
        f"    def m{i}(self):\n" + "".join(f"        v = {j}\n" for j in range(30)) + "        return v\n\n" for i in range(10))
    doc = Document.create("pkg/hdr.py", src, "x", "python")
    chunks = chunk_ast(doc, _tok(), 1024, header=True)
    _check_cover(doc, chunks)
    for c in chunks:
        assert c.header.startswith("# pkg/hdr.py\n") and c.embed_text == c.header + c.text
        assert len(_tok()(c.embed_text)["input_ids"]) <= 1024
    assert any("Big.m" in c.header for c in chunks)
    assert all(c.header == "" for c in chunk_ast(doc, _tok(), 1024))


def test_config_and_make_chunker():
    enc = FakeEncoder()
    config = chunking_config("ast", enc, header=True)
    assert config == {"name": "ast", "max_length": enc.max_length, "overlap_tokens": 128, "header": True,
                      "target_tokens": TARGET_TOKENS, "min_tokens": MIN_TOKENS}
    doc = Document.create("a.py", _fn("f", 3), "x", "python")
    assert make_chunker("ast", tokenizer=_tok(), max_length=1024)(doc) == chunk_ast(doc, _tok(), 1024)


def test_cli_chunking_flag_default_unchanged():
    import json
    from tests.test_cli import _run
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp, "repo")
        _write(root, "pkg/mod.py", "".join(_fn(f"f{i}", 3) for i in range(20)))
        assert _run("index", root, "--out", Path(tmp, "a"))[0] == 0
        assert json.loads(Path(tmp, "a", "manifest.json").read_text(encoding="utf-8"))["chunking"]["name"] == "windows"
        assert _run("index", root, "--out", Path(tmp, "b"), "--chunking", "ast")[0] == 0
        manifest = json.loads(Path(tmp, "b", "manifest.json").read_text(encoding="utf-8"))
        assert manifest["chunking"]["name"] == "ast" and len(manifest["documents"]["pkg/mod.py"]["chunk_ids"]) >= 1
        code, _, err = _run("index", root, "--out", Path(tmp, "b"))  # windows default vs an ast index: rebuild
        assert code == 2 and "--rebuild" in err


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
