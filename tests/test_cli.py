"""Tests for cli.py (step 6), with the fake encoder from tests.test_index in place of the model.

Covers the P1 flow: index a folder, change it, re-index incrementally, and see stale snippets before re-indexing.

Run from the project root:  python -m tests.test_cli
(Also collectable by pytest if it is installed.)
"""
import contextlib
import io
import json
import tempfile
from pathlib import Path

import cli
from tests.test_index import FakeEncoder


def _run(*argv, encoder=None) -> tuple[int, str, str]:
    old = cli._make_encoder
    cli._make_encoder = lambda: encoder or FakeEncoder()
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main([str(a) for a in argv])
    finally:
        cli._make_encoder = old
    return code, out.getvalue(), err.getvalue()


def _repo(root: Path) -> None:
    (root / "pkg").mkdir(parents=True)
    for i in range(3):
        (root / f"pkg/mod{i}.py").write_text(
            "".join(f"def f{i}_{j}(x):\n    return x * {j}\n\n" for j in range(40)), encoding="utf-8", newline="")
    (root / "README.md").write_text("# demo\n", encoding="utf-8", newline="")


def test_index_query_update_stale_cycle():
    with tempfile.TemporaryDirectory() as tmp:
        root, out = Path(tmp, "repo"), Path(tmp, "idx")
        _repo(root)
        code, _, err = _run("index", root, "--out", out)
        assert code == 0 and "built: 4 files" in err, err

        # FakeEncoder embeds identical text identically, so querying a function's exact text finds it.
        query = "def f1_5(x):\n    return x * 5\n"
        code, stdout, _ = _run("query", query, "--index", out, "--top-k", 2, "--json")
        rows = json.loads(stdout)
        assert code == 0 and len(rows) == 2 and rows[0]["rank"] == 1
        assert {"doc_id", "score", "source_path", "language", "chunks", "snippet"} <= rows[0].keys()
        assert rows[0]["snippet"]["status"] == "ok" and rows[0]["snippet"]["text"]

        # Change one file, add one, delete one. Before re-indexing, the changed file's snippets are stale.
        (root / "pkg/mod1.py").write_text("def g():\n    return 1\n", encoding="utf-8", newline="")
        (root / "pkg/new.py").write_text("def h():\n    return 2\n", encoding="utf-8", newline="")
        (root / "pkg/mod2.py").unlink()
        code, stdout, _ = _run("query", "def g():\n    return 1\n", "--index", out, "--top-k", 10, "--json")
        status = {r["doc_id"]: r["snippet"]["status"] for r in json.loads(stdout)}
        assert status["pkg/mod1.py"] == "stale" and status["pkg/mod2.py"] == "missing", status

        # Incremental update: only the changed and new files are embedded.
        enc = FakeEncoder()
        code, _, err = _run("index", root, "--out", out, encoder=enc)
        assert code == 0 and "1 added, 1 changed, 1 deleted, 2 unchanged" in err, err
        assert sorted(enc.encoded) == ["def g():\n    return 1\n", "def h():\n    return 2\n"]

        code, stdout, _ = _run("query", "def g():\n    return 1\n", "--index", out)
        assert code == 0 and stdout.startswith(" 1. pkg/mod1.py:1-2") and "stale" not in stdout, stdout
        assert "pkg/mod2.py" not in stdout


def test_default_out_and_index_inside_root_excludes_itself():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp, "repo")
        _repo(root)
        inner = root / ".codeindex"
        for _ in range(2):  # the second run must not pick up the first run's manifest/chunk files
            code, _, err = _run("index", root, "--out", inner)
            assert code == 0, err
        manifest = json.loads((inner / "manifest.json").read_text(encoding="utf-8"))
        assert sorted(manifest["documents"]) == ["README.md", "pkg/mod0.py", "pkg/mod1.py", "pkg/mod2.py"]

        cwd = Path.cwd()
        try:
            import os
            os.chdir(tmp)
            code, _, err = _run("index", "repo")
            assert code == 0 and Path(tmp, "indexes", "repo", "manifest.json").exists(), err
        finally:
            os.chdir(cwd)


def test_errors():
    with tempfile.TemporaryDirectory() as tmp:
        root, out = Path(tmp, "repo"), Path(tmp, "idx")
        _repo(root)
        code, _, err = _run("query", "x", "--index", Path(tmp, "nope"))
        assert code == 2 and "no index" in err
        code, _, err = _run("index", Path(tmp, "missing"))
        assert code == 2 and "not a directory" in err

        assert _run("index", root, "--out", out)[0] == 0
        other = FakeEncoder(name="other-model")
        code, _, err = _run("index", root, "--out", out, encoder=other)
        assert code == 2 and "--rebuild" in err, err
        code, _, err = _run("query", "x", "--index", out, encoder=other)
        assert code == 2 and "differs" in err, err
        code, _, err = _run("index", root, "--out", out, "--rebuild", encoder=other)
        assert code == 0 and "built" in err, err


def test_eval_apps_passes_arguments_through():
    from eval import apps_pipeline
    seen = []
    old = apps_pipeline.main
    apps_pipeline.main = lambda argv: seen.append(argv) or 0
    try:
        assert _run("eval-apps", "--limit", 7)[0] == 0
        assert _run("eval-apps", "--index", "x", "--output", "y.json")[0] == 0
    finally:
        apps_pipeline.main = old
    assert seen == [["--limit", "7"], ["--index", "x", "--output", "y.json"]]


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
