"""Tests for web/server.py with the fake encoder from tests.test_index (no model load).

Covers the browser flow: list indexes, index a folder (streamed progress), search, edit -> stale -> re-index
(incremental counts) -> fresh, code-only, zip upload (including a zip entry that tries to escape the upload folder),
and error responses.

Run from the project root:  python -m tests.test_web
(Also collectable by pytest if it is installed.)
"""
import io
import json
import tempfile
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from tests.test_index import FakeEncoder
from web.server import Server, create_app


def _client(tmp: Path):
    server = Server(tmp / "indexes", tmp / "uploads", encoder_factory=FakeEncoder)
    return TestClient(create_app(server)), server


def _repo(root: Path) -> None:
    (root / "pkg").mkdir(parents=True)
    (root / "tests").mkdir()
    for i in range(3):
        (root / f"pkg/mod{i}.py").write_text(
            "".join(f"def f{i}_{j}(x):\n    return x * {j}\n\n" for j in range(40)), encoding="utf-8", newline="")
    (root / "tests/test_mod.py").write_text("def test_x():\n    assert True\n", encoding="utf-8", newline="")


def _events(response) -> list[dict]:
    assert response.status_code == 200, response.text
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def test_index_search_stale_reindex_cycle():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        client, _ = _client(tmp)
        assert client.get("/api/indexes").json()["indexes"] == []
        assert "Code retrieval" in client.get("/").text

        root = tmp / "repo"
        _repo(root)
        events = _events(client.post("/api/index", json={"path": str(root)}))
        kinds = [e["event"] for e in events]
        assert kinds[-1] == "done" and "progress" in kinds and "log" in kinds, kinds
        done = events[-1]
        assert done["result"]["mode"] == "built" and done["result"]["files"] == 4
        name = done["info"]["name"]
        listed = client.get("/api/indexes").json()["indexes"]
        assert [i["name"] for i in listed] == [name]
        info = listed[0]
        assert info["documents"] == 4 and info["chunking"] == "windows" and info["header"] is True
        assert info["source"] == str(root.resolve()) and info["reindexable"] is True
        assert client.get(f"/api/indexes/{name}").json()["load_ms"] is not None  # loaded now
        assert client.get(f"/api/indexes/{name}").json()["load_ms"] is None  # already loaded

        query = {"index": name, "query": "def f1_5(x):\n    return x * 5\n", "top_k": 10}
        data = client.post("/api/query", json=query).json()
        assert data["query_ms"] >= 0 and data["index"]["name"] == name
        top = data["results"][0]
        assert top["rank"] == 1 and top["hits"][0]["version"] is None
        assert top["hits"][0]["snippet"]["status"] == "ok" and top["hits"][0]["snippet"]["text"]
        assert isinstance(top["hits"][0]["snippet"]["match_lines"], list)
        assert "tests/test_mod.py" in {r["doc_id"] for r in data["results"]}
        only = client.post("/api/query", json={**query, "code_only": True}).json()["results"]
        assert "tests/test_mod.py" not in {r["doc_id"] for r in only}

        # Edit a file: its results are stale until re-indexed; re-indexing reports the incremental counts.
        (root / "pkg/mod1.py").write_text("def g():\n    return 1\n", encoding="utf-8", newline="")
        stale = client.post("/api/query", json=query).json()["results"]
        status = {r["doc_id"]: r["hits"][0]["snippet"]["status"] for r in stale}
        assert status["pkg/mod1.py"] == "stale", status
        done = _events(client.post("/api/index", json={"index": name}))[-1]
        assert done["event"] == "done", done
        r = done["result"]
        assert (r["mode"], r["added"], r["changed"], r["deleted"], r["unchanged"]) == ("updated", 0, 1, 0, 3), r
        assert r["chunks_embedded"] == 1 and r["seconds"] >= 0
        fresh = client.post("/api/query", json=query).json()["results"]
        status = {r["doc_id"]: r["hits"][0]["snippet"]["status"] for r in fresh}
        assert status["pkg/mod1.py"] == "ok" and set(status.values()) == {"ok"}, status


def test_zip_upload_and_unsafe_zip():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        client, server = _client(tmp)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:  # one top-level folder: unpacked from inside it
            z.writestr("proj/app.py", "def main():\n    return 42\n")
            z.writestr("proj/lib/util.py", "def helper():\n    return 1\n")
        files = {"file": ("proj.zip", buf.getvalue(), "application/zip")}
        events = _events(client.post("/api/index", files=files, data={"name": "proj"}))
        assert events[-1]["event"] == "done", events
        assert (tmp / "uploads/proj/app.py").is_file() and (tmp / "uploads/proj/lib/util.py").is_file()
        assert events[-1]["result"]["files"] == 2

        bad = io.BytesIO()
        with zipfile.ZipFile(bad, "w") as z:
            z.writestr("ok.py", "x = 1\n")
            z.writestr("../escaped.py", "x = 2\n")
        events = _events(client.post("/api/index", files={"file": ("bad.zip", bad.getvalue())}, data={"name": "bad"}))
        assert events[-1]["event"] == "error" and "outside" in events[-1]["message"], events
        assert not (tmp / "escaped.py").exists() and not (tmp / "uploads/bad").exists()
        assert not server._indexing.locked()  # a failed job releases the lock


def test_errors():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        client, _ = _client(tmp)
        assert client.post("/api/query", json={"index": "nope", "query": "x"}).status_code == 404
        assert client.post("/api/query", json={"index": "nope", "query": ""}).status_code == 422
        assert client.post("/api/query", json={"index": "../x", "query": "x"}).status_code == 404
        assert client.post("/api/index", json={"path": str(tmp / "missing")}).status_code == 400
        assert client.post("/api/index", json={}).status_code == 400
        assert client.post("/api/index", json={"index": "nope"}).status_code == 404
        root = tmp / "repo"
        _repo(root)
        assert client.post("/api/index", json={"path": str(root), "chunking": "bogus"}).status_code == 400
        # A prebuilt (non-folder) index cannot be re-indexed from the page.
        apps = tmp / "indexes/apps"
        apps.mkdir(parents=True)
        (apps / "manifest.json").write_text(json.dumps({"source": {"kind": "apps"}, "chunking": {"name": "none"},
                                                        "n_documents": 1, "n_chunks": 1}), encoding="utf-8")
        listed = client.get("/api/indexes").json()["indexes"]
        assert listed[0]["reindexable"] is False
        assert client.post("/api/index", json={"index": "apps"}).status_code == 400


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
