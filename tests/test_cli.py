"""Tests for cli.py (step 6), with the fake encoder from tests.test_index in place of the model.

Covers the P1 flow: index a folder, change it, re-index incrementally, and see stale snippets before re-indexing,
both one query at a time and in one --interactive session (model loaded once).

Run from the project root:  python -m tests.test_cli
(Also collectable by pytest if it is installed.)
"""
import contextlib
import io
import json
import os
import sys
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
            [made] = Path(tmp, "indexes").iterdir()
            assert code == 0 and (made / "manifest.json").exists(), err
            assert made.name.startswith("repo-") and len(made.name) == len("repo-") + 8, made.name
        finally:
            os.chdir(cwd)


def test_default_index_dir_distinguishes_same_named_folders():
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp, "x", "repo"), Path(tmp, "y", "repo")
        a.mkdir(parents=True), b.mkdir(parents=True)
        assert cli.default_index_dir(a) != cli.default_index_dir(b)
        assert cli.default_index_dir(a) == cli.default_index_dir(Path(tmp, "x", ".", "repo"))
        assert cli.default_index_dir(a).name.startswith("repo-")
        # Windows paths are case-insensitive: the same folder spelled differently maps to the same index.
        if os.name == "nt":
            assert cli.default_index_dir(Path(str(a).upper())) == cli.default_index_dir(a)


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



class ScriptedStdin:
    """stdin for interactive tests: yields lines; a callable item runs (e.g. edits a file) and is skipped."""

    def __init__(self, items):
        self.items = list(items)

    def readline(self) -> str:
        while self.items:
            item = self.items.pop(0)
            if callable(item):
                item()
                continue
            return item + "\n"
        return ""  # end of input


def _run_interactive(index_dir: Path, items, encoder=None) -> tuple[int, str, str, int]:
    """Run `query --index DIR --interactive` on scripted input. Returns (code, stdout, stderr, encoders made)."""
    made = []

    def make():
        made.append(1)
        return encoder or FakeEncoder()

    old_make, old_stdin = cli._make_encoder, sys.stdin
    cli._make_encoder, sys.stdin = make, ScriptedStdin(items)
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(["query", "--index", str(index_dir), "--interactive"])
    finally:
        cli._make_encoder, sys.stdin = old_make, old_stdin
    return code, out.getvalue(), err.getvalue(), len(made)


def test_interactive_commands_and_single_model_load():
    with tempfile.TemporaryDirectory() as tmp:
        root, out = Path(tmp, "repo"), Path(tmp, "idx")
        _repo(root)
        assert _run("index", root, "--out", out)[0] == 0
        code, stdout, stderr, made = _run_interactive(out, [
            "def f1_5(x):",
            ":k 1",
            "return x * 5",
            ":json",
            "def f2_3(x):",
            ":json",
            ":paste",
            "def f0_7(x):",
            "    return x * 7",
            ".",
            ":k zero",
            ":bogus",
            "",
            ":q",
            "never reached",
        ])
        assert code == 0 and made == 1, (code, made)  # one model load for the whole session
        assert stderr.count("results in") == 4, stderr  # four queries answered
        assert "(showing 1 results)" in stderr and "usage: :k N" in stderr and "unknown command ':bogus'" in stderr
        # The :json query printed exactly one result as JSON; the :paste query after it printed as text.
        json_start = stdout.index("[\n")
        rows = json.loads(stdout[json_start:stdout.index("\n]", json_start) + 2])
        assert len(rows) == 1 and rows[0]["rank"] == 1
        assert " 1. " in stdout.split("\n]", 1)[1]
        assert "never reached" not in stdout + stderr


def test_interactive_sees_edits_and_reindex_without_reloading_model():
    with tempfile.TemporaryDirectory() as tmp:
        root, out = Path(tmp, "repo"), Path(tmp, "idx")
        _repo(root)
        assert _run("index", root, "--out", out)[0] == 0
        new_text = "def g():\n    return 1"  # no trailing newline: :paste joins lines without one

        def edit():
            (root / "pkg/mod1.py").write_text(new_text, encoding="utf-8", newline="")

        def reindex():  # as if `cli.py index` ran in another terminal
            assert _run("index", root, "--out", out)[0] == 0

        code, stdout, stderr, made = _run_interactive(out, [
            ":k 4",
            edit,
            ":paste", "def g():", "    return 1", ".",   # before re-indexing: mod1 is stale
            reindex,
            ":paste", "def g():", "    return 1", ".",   # after: index reloaded, mod1 fresh and on top
        ])
        assert code == 0 and made == 1, (code, made)
        assert "[stale: source changed since indexing; re-run index]" in stdout, stdout
        assert stderr.count("(index reloaded") == 1, stderr
        last = stdout[stdout.rindex(" 1. "):]
        assert last.startswith(" 1. pkg/mod1.py:1-2") and "stale" not in last, last


def test_interactive_first_query_and_end_of_input():
    with tempfile.TemporaryDirectory() as tmp:
        root, out = Path(tmp, "repo"), Path(tmp, "idx")
        _repo(root)
        assert _run("index", root, "--out", out)[0] == 0
        old_make, old_stdin = cli._make_encoder, sys.stdin
        cli._make_encoder, sys.stdin = FakeEncoder, ScriptedStdin([])  # EOF straight after the first query
        buf_out, buf_err = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                code = cli.main(["query", "README", "--index", str(out), "-i", "--top-k", "2"])
        finally:
            cli._make_encoder, sys.stdin = old_make, old_stdin
        assert code == 0 and buf_err.getvalue().count("results in") == 1
        assert _run("query", "--index", out)[0] == 2  # no text and not interactive

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
