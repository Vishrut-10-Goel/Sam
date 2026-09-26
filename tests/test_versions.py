"""Tests for versioned indexes (index/versions.py, retrieval/versions.py, cli.py index-versions / query --version).

A small git repository with three commits, indexed with the fake encoder from tests.test_index (no model load):
embeddings are reused across commits for unchanged files, the repository's working tree is never touched, search
returns one result per file listing which versions hold the match, and one version can be searched on its own.

Run from the project root:  python -m tests.test_versions
(Also collectable by pytest if it is installed.)
"""
import contextlib
import io
import json
import subprocess
import tempfile
from pathlib import Path

import cli
from index import chunking_config, load_index, save_index
from index.versions import index_versions, last_commits, resolve_version, revision_id
from tests.test_index import FakeEncoder


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          check=True, capture_output=True, text=True).stdout


def _funcs(prefix: str, n: int) -> str:
    return "".join(f"def {prefix}_{j}(x):\n    return x * {j}\n\n" for j in range(n))


def _repo(root: Path, autocrlf: str = "false") -> list[str]:
    """Three commits (tagged v1, v2, v3); returns their shas, oldest first.

    v1: pkg/a.py, pkg/b.py, tests/test_a.py
    v2: pkg/b.py rewritten, pkg/c.py added
    v3: a function appended to the end of pkg/a.py (its first lines unchanged)
    """
    root.mkdir(parents=True)
    _git(root, "init", "-q")
    _git(root, "config", "core.autocrlf", autocrlf)
    files = {"pkg/a.py": _funcs("alpha", 60), "pkg/b.py": _funcs("beta", 20), "tests/test_a.py": _funcs("test", 3)}
    shas = []
    for n, change in enumerate([{}, {"pkg/b.py": _funcs("gamma", 20), "pkg/c.py": _funcs("delta", 5)},
                                {"pkg/a.py": _funcs("alpha", 60) + "def omega(x):\n    return -x\n"}], 1):
        files.update(change)
        for rel, text in files.items():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(text, encoding="utf-8", newline="")
        _git(root, "add", "-A")
        _git(root, "commit", "-q", "-m", f"commit {n}")
        _git(root, "tag", f"v{n}")
        shas.append(_git(root, "rev-parse", "HEAD").strip())
    return shas


def _chunking(enc):
    return chunking_config("windows", enc, header=False)  # no header: a chunk's text is exactly what is embedded


def test_reuse_across_commits_and_working_tree_untouched():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp, "repo")
        shas = _repo(repo)
        (repo / "pkg/a.py").write_text("uncommitted edit\n", encoding="utf-8")  # must survive indexing
        status_before = _git(repo, "status", "--porcelain")
        enc = FakeEncoder()
        index, stats = index_versions(repo, ["v1", "v2", "v3"], enc, _chunking(enc), note=lambda _: None)

        per = {c["label"]: c for c in stats["commits_added"]}
        assert (per["v1"]["revisions_new"], per["v1"]["revisions_reused"]) == (3, 0), per["v1"]
        assert (per["v2"]["revisions_new"], per["v2"]["revisions_reused"]) == (2, 2), per["v2"]  # b changed, c new
        assert (per["v3"]["revisions_new"], per["v3"]["revisions_reused"]) == (1, 3), per["v3"]  # a changed
        # Every distinct chunk text is embedded exactly once. v3 changed only the end of a.py, so the new revision's
        # earlier chunks are identical to v1's and copy their vectors instead of being embedded again.
        assert len(enc.encoded) == len(set(enc.encoded)) == sum(c["chunks_embedded"] for c in per.values())
        assert per["v3"]["chunks_copied"] > 0 and per["v3"]["chunks_embedded"] < per["v3"]["chunks_new"]
        assert len(enc.encoded) < len(index.chunks) == stats["chunks"]
        rows = {r.chunk_id: i for i, r in enumerate(index.chunks)}
        by_key: dict = {}
        for e in index.documents.values():  # rows with the same embedded text hold the same vector
            assert len(e.metadata["chunk_hashes"]) == len(e.chunk_ids)
            for chunk_id, key in zip(e.chunk_ids, e.metadata["chunk_hashes"]):
                by_key.setdefault(key, []).append(index.embeddings[rows[chunk_id]])
        assert len(by_key) == len(enc.encoded)
        assert all((v == vs[0]).all() for vs in by_key.values() for v in vs)
        assert set(map(tuple, (vs[0] for vs in by_key.values()))) == {tuple(enc.vector(t)) for t in enc.encoded}
        assert stats["chunks_if_separate"] > stats["chunks"] and stats["revisions"] == 6
        assert [c["sha"] for c in index.source["commits"]] == shas and index.source["kind"] == "git"
        b1 = index.documents[revision_id("pkg/b.py", _hash(repo, "v1", "pkg/b.py"))]
        assert b1.metadata["versions"] == [shas[0]] and b1.metadata["path"] == "pkg/b.py"
        a1 = index.documents[revision_id("pkg/a.py", _hash(repo, "v1", "pkg/a.py"))]
        assert a1.metadata["versions"] == shas[:2]

        # The repository is exactly as it was: same HEAD, the uncommitted edit intact, no worktree left behind.
        assert _git(repo, "rev-parse", "HEAD").strip() == shas[-1]
        assert _git(repo, "status", "--porcelain") == status_before
        assert (repo / "pkg/a.py").read_text(encoding="utf-8") == "uncommitted edit\n"
        assert len(_git(repo, "worktree", "list").splitlines()) == 1

        # Adding a commit later keeps the old ones and embeds only what is new.
        enc2 = FakeEncoder()
        first, _ = index_versions(repo, ["v1", "v2"], enc2, _chunking(enc2), note=lambda _: None)
        n_first = len(enc2.encoded)
        more, stats2 = index_versions(repo, ["v3", "v1"], enc2, _chunking(enc2), existing=first, note=lambda _: None)
        assert [c["label"] for c in stats2["commits_added"]] == ["v3"]  # v1 is already there
        assert len(enc2.encoded) - n_first == stats2["commits_added"][0]["chunks_embedded"] > 0
        assert sorted(more.documents) == sorted(index.documents) and stats2["commits"] == 3
        assert last_commits(repo, 2) == shas[1:]


def _hash(repo: Path, rev: str, path: str) -> str:
    from records import content_hash
    return content_hash(subprocess.run(["git", "-C", str(repo), "cat-file", "--filters", f"{rev}:{path}"],
                                       check=True, capture_output=True).stdout.decode("utf-8"))


def _session(tmp: Path, autocrlf: str):
    repo = tmp / "repo"
    shas = _repo(repo, autocrlf)
    enc = FakeEncoder()
    index, _ = index_versions(repo, ["v1", "v2", "v3"], enc, _chunking(enc), note=lambda _: None)
    save_index(index, tmp / "idx")
    return cli.QuerySession(tmp / "idx", enc, kind_penalty=0.0), load_index(tmp / "idx"), shas


def test_search_groups_versions_and_filters_one_version():
    for autocrlf in ("false", "true"):  # snippets are read from git with the same line endings as a checkout
        with tempfile.TemporaryDirectory() as tmp:
            session, index, shas = _session(Path(tmp), autocrlf)
            a_rev1 = next(d for d in index.documents if d.startswith("pkg/a.py@") and shas[0] in index.documents[d].metadata["versions"])
            first_chunk = next(r for r in index.chunks if r.doc_id == a_rev1)
            query = "".join(session.reader.snippet(a_rev1, first_chunk.start_line, first_chunk.end_line).text)

            rows = session.search(query, top_k=10)
            paths = [r["doc_id"] for r in rows]
            assert len(paths) == len(set(paths)) == 4, paths  # one result per file, not per version
            top = rows[0]
            assert top["doc_id"] == "pkg/a.py" and top["snippet"]["status"] == "ok", (autocrlf, top["snippet"]["status"])
            assert abs(top["similarity"] - 1.0) < 1e-5
            revs = top["versions"]
            # v3 changed a.py only at the end, so its first chunk is identical to v1's and scores the same: the tie
            # goes to the newest version, and v1/v2's revision is listed as holding the same lines, at the same place.
            assert len(revs) == 2 and revs[0]["primary"] and [c["label"] for c in revs[0]["commits"]] == ["v3"]
            assert revs[1]["revision"] == a_rev1 and [c["label"] for c in revs[1]["commits"]] == ["v1", "v2"]
            assert revs[1]["contains_match"] is True and revs[1]["match_line"] == first_chunk.start_line
            assert revs[0]["score"] == revs[1]["score"]

            # b.py was rewritten in v2: searching v1's text, v2's revision does not hold those lines.
            b_rev1 = next(d for d in index.documents if d.startswith("pkg/b.py@") and shas[0] in index.documents[d].metadata["versions"])
            b_chunk = next(r for r in index.chunks if r.doc_id == b_rev1)
            b_query = session.reader.snippet(b_rev1, b_chunk.start_line, b_chunk.end_line).text
            b_row = next(r for r in session.search(b_query, top_k=10) if r["doc_id"] == "pkg/b.py")
            assert b_row["revision"] == b_rev1 and b_row["versions"][1]["contains_match"] is False

            # One version: that version's revision is the result, files it lacks are left out.
            v1_rows = session.search(query, top_k=10, version="v1")
            assert "pkg/c.py" not in {r["doc_id"] for r in v1_rows} and v1_rows[0]["version"] == shas[0]
            v2_b = next(r for r in session.search(b_query, top_k=10, version=shas[1][:8]) if r["doc_id"] == "pkg/b.py")
            assert shas[1] in [c["sha"] for c in v2_b["versions"][0]["commits"]] and v2_b["revision"] != b_rev1
            assert resolve_version(index, "v3") == shas[2]
            for bad in ("v9", "zz"):
                try:
                    session.search(query, top_k=3, version=bad)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"version {bad!r} accepted")


def _run(*argv):
    old = cli._make_encoder
    cli._make_encoder = lambda: FakeEncoder()
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main([str(a) for a in argv])
    finally:
        cli._make_encoder = old
    return code, out.getvalue(), err.getvalue()


def test_cli_index_versions_and_query():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        repo = tmp / "repo"
        _repo(repo)
        out = tmp / "vidx"
        code, _, err = _run("index-versions", repo, "--commits", "v1", "v2", "--out", out)
        assert code == 0 and "[2/2] v2" in err and "2 versions" in err, err
        code, _, err = _run("index-versions", repo, "--last", "1", "--out", out)  # adds v3 (HEAD), keeps the rest
        assert code == 0 and "[1/1] v3" in err and "3 versions" in err, err
        code, stdout, _ = _run("query", "def alpha_3(x):", "--index", out, "--top-k", 2)
        assert code == 0 and "matched in:" in stdout, stdout
        code, stdout, _ = _run("query", "def alpha_3(x):", "--index", out, "--top-k", 5, "--json", "--version", "v2")
        rows = json.loads(stdout)
        assert code == 0 and rows and all(r["version"] for r in rows) and "pkg/c.py" in {r["doc_id"] for r in rows}
        assert _run("query", "x", "--index", out, "--version", "nope")[0] == 2
        assert _run("index-versions", tmp / "missing", "--last", "1")[0] == 2
        assert _run("index-versions", repo, "--commits", "no-such-rev", "--out", tmp / "v2")[0] == 2
        # --version on an index without versions is an error, not ignored.
        code, _, _ = _run("index", repo, "--out", tmp / "plain")
        assert code == 0 and _run("query", "x", "--index", tmp / "plain", "--version", "v1")[0] == 2


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
