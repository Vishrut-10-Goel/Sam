"""Tests for index/ (step 3): build, save/load, fingerprint check, manifest, incremental update, atomic saves.

Uses a fake encoder (real tokenizer, deterministic hash-based vectors) so the tests can count exactly which
texts get embedded, without loading the model.

Run from the project root:  python -m tests.test_index
(Also collectable by pytest if it is installed.)
"""
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from chunking import DEFAULT_OVERLAP_TOKENS
from embedding.onnx_encoder import DEFAULT_MODEL
from index import IndexMismatchError, build_index, chunking_config, load_index, save_index, update_index
from index.storage import MANIFEST
from loaders.directory import CHUNKING, load_directory
from records import Document

_tokenizer = None


class FakeEncoder:
    """Same interface as OnnxEncoder. Embeds each text as a unit vector seeded by the text's hash."""

    dim = 8

    def __init__(self, max_length: int = 256, name: str = "fake"):
        global _tokenizer
        if _tokenizer is None:
            _tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
        self.tokenizer = _tokenizer
        self.max_length = max_length
        self.name = name
        self.encoded: list[str] = []  # every text ever passed to encode()

    def encode(self, texts, batch_size=None, progress=None):
        self.encoded.extend(texts)
        out = np.stack([self.vector(t) for t in texts]) if texts else np.empty((0, self.dim), np.float32)
        if progress is not None:
            progress(len(texts), len(texts))
        return out

    def vector(self, text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
        v = np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def fingerprint(self) -> dict:
        return {"model_name": self.name, "max_length": self.max_length, "dim": self.dim}


def _write(root: Path, rel: str, text: str) -> None:
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_bytes(text.encode("utf-8"))


def _repo(root: Path) -> None:
    for i in range(4):
        _write(root, f"pkg/mod{i}.py", "".join(f"def f{i}_{j}(x):\n    return x + {j}\n\n" for j in range(40)))
    _write(root, "README.md", "# demo\n\nA small repo.\n")


def _build(root: Path, enc: FakeEncoder):
    return build_index(load_directory(root), enc, chunking_config(CHUNKING, enc), source={"kind": "directory"})


def _by_chunk_id(index) -> dict:
    return {row.chunk_id: index.embeddings[i] for i, row in enumerate(index.chunks)}


def _assert_same(a, b) -> None:
    ea, eb = _by_chunk_id(a), _by_chunk_id(b)
    assert ea.keys() == eb.keys(), (sorted(ea.keys() ^ eb.keys()))[:5]
    for k in ea:
        assert np.array_equal(ea[k], eb[k]), k
    assert {d: (e.content_hash, e.chunk_ids) for d, e in a.documents.items()} == \
           {d: (e.content_hash, e.chunk_ids) for d, e in b.documents.items()}


def test_build_manifest_and_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        root, out = Path(tmp, "repo"), Path(tmp, "idx")
        _repo(root)
        enc = FakeEncoder()
        index = _build(root, enc)
        assert set(index.documents) == {"README.md", "pkg/mod0.py", "pkg/mod1.py", "pkg/mod2.py", "pkg/mod3.py"}
        assert len(index.documents["pkg/mod0.py"].chunk_ids) > 1  # 256-token windows split the modules
        assert index.chunking == {"name": "windows", "max_length": 256, "overlap_tokens": DEFAULT_OVERLAP_TOKENS}
        for i, row in enumerate(index.chunks):  # row i really embeds that chunk
            assert row.chunk_id in index.documents[row.doc_id].chunk_ids

        save_index(index, out)
        manifest = json.loads((out / MANIFEST).read_text(encoding="utf-8"))
        assert manifest["documents"]["README.md"]["content_hash"] == index.documents["README.md"].content_hash
        assert manifest["documents"]["README.md"]["chunk_ids"] == ["README.md#L1-L3"]
        loaded = load_index(out, encoder=enc)
        _assert_same(index, loaded)
        assert loaded.fingerprint == enc.fingerprint() and loaded.source == {"kind": "directory"}
        assert loaded.chunks == index.chunks


def test_fingerprint_and_chunking_mismatch_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        root, out = Path(tmp, "repo"), Path(tmp, "idx")
        _repo(root)
        enc = FakeEncoder()
        save_index(_build(root, enc), out)
        for other in [FakeEncoder(name="other"), FakeEncoder(max_length=512)]:
            for fn in [lambda: load_index(out, encoder=other),
                       lambda: update_index(load_index(out), load_directory(root), other)]:
                try:
                    fn()
                except IndexMismatchError:
                    pass
                else:
                    raise AssertionError("mismatched encoder was accepted")
        index = load_index(out)  # loading without an encoder is allowed (inspection)
        try:
            index.check_compatible(enc, chunking={"name": "none"})
        except IndexMismatchError:
            pass
        else:
            raise AssertionError("mismatched chunking was accepted")


def test_incremental_update_embeds_only_changes_and_matches_rebuild():
    with tempfile.TemporaryDirectory() as tmp:
        root, out = Path(tmp, "repo"), Path(tmp, "idx")
        _repo(root)
        enc = FakeEncoder()
        save_index(_build(root, enc), out)

        # No changes: nothing embedded, identical index.
        enc.encoded.clear()
        same, stats = update_index(load_index(out), load_directory(root), enc)
        assert enc.encoded == [] and stats.unchanged == 5 and stats.chunks_embedded == 0
        _assert_same(same, load_index(out))

        # Change one file, add one, delete one.
        _write(root, "pkg/mod1.py", "def changed():\n    return 42\n")
        _write(root, "pkg/new.py", "def added():\n    return 7\n")
        (root / "pkg/mod3.py").unlink()
        enc.encoded.clear()
        updated, stats = update_index(load_index(out, encoder=enc), load_directory(root), enc)
        assert (stats.added, stats.changed, stats.deleted, stats.unchanged) == (1, 1, 1, 3), stats
        assert sorted(enc.encoded) == sorted(["def changed():\n    return 42\n", "def added():\n    return 7\n"])
        assert stats.chunks_embedded == 2
        assert "pkg/mod3.py" not in updated.documents
        assert not any(r.doc_id == "pkg/mod3.py" for r in updated.chunks)

        _assert_same(updated, _build(root, FakeEncoder()))  # same result as a full rebuild
        save_index(updated, out)
        _assert_same(load_index(out), updated)


def test_saves_are_atomic_and_clean_up():
    with tempfile.TemporaryDirectory() as tmp:
        root, out = Path(tmp, "repo"), Path(tmp, "idx")
        _repo(root)
        enc = FakeEncoder()
        index = _build(root, enc)
        save_index(index, out)
        first = json.loads((out / MANIFEST).read_text(encoding="utf-8"))

        # Simulate a save that crashed after writing the next generation's data (and a temp file), before
        # the manifest was replaced: the committed index must still load unchanged.
        (out / "embeddings-000002.npy").write_bytes(b"partial")
        (out / ".manifest.json.tmp-999").write_bytes(b"{")
        _assert_same(load_index(out), index)

        # The next save commits a new generation and removes everything else.
        save_index(index, out)
        second = json.loads((out / MANIFEST).read_text(encoding="utf-8"))
        assert second["generation"] == first["generation"] + 1
        names = sorted(p.name for p in out.iterdir())
        assert names == sorted([MANIFEST, *(f["name"] for f in second["files"].values())]), names
        _assert_same(load_index(out), index)


def test_corruption_detected():
    with tempfile.TemporaryDirectory() as tmp:
        root, out = Path(tmp, "repo"), Path(tmp, "idx")
        _repo(root)
        save_index(_build(root, FakeEncoder()), out)
        manifest = json.loads((out / MANIFEST).read_text(encoding="utf-8"))
        chunks_file = out / manifest["files"]["chunks"]["name"]
        chunks_file.write_bytes(chunks_file.read_bytes().replace(b"\n", b"\r\n"))  # e.g. git autocrlf
        try:
            load_index(out)
        except ValueError as e:
            assert "checksum" in str(e)
        else:
            raise AssertionError("corrupted chunk table was accepted")


def test_duplicate_ids_and_none_chunking():
    enc = FakeEncoder()
    docs = [Document.create("d1", "print(1)\n", "x", "python"), Document.create("d1", "print(2)\n", "x", "python")]
    try:
        build_index(docs, enc, chunking_config("none", enc))
    except ValueError as e:
        assert "duplicate" in str(e)
    else:
        raise AssertionError("duplicate document ids were accepted")

    docs = [Document.create(f"d{i}", f"print({i})\n" * 200, "x", "python") for i in range(3)]
    index = build_index(docs, enc, chunking_config("none", enc))
    assert index.chunking == {"name": "none"}
    assert [r.chunk_id for r in index.chunks] == ["d0", "d1", "d2"]  # one chunk per doc, even past max_length
    assert np.array_equal(index.embeddings[1], enc.vector(docs[1].text))



def test_header_config_is_recorded_and_embedded():
    enc = FakeEncoder()
    assert chunking_config(CHUNKING, enc) == {"name": "windows", "max_length": 256, "overlap_tokens": DEFAULT_OVERLAP_TOKENS}
    assert chunking_config(CHUNKING, enc, header=True)["header"] is True
    assert chunking_config("none", enc, header=True) == {"name": "none"}
    with tempfile.TemporaryDirectory() as tmp:
        root, out = Path(tmp, "repo"), Path(tmp, "idx")
        _repo(root)
        index = build_index(load_directory(root), enc, chunking_config(CHUNKING, enc, header=True))
        assert all(t.startswith("# ") for t in enc.encoded)  # every embedded text carries its header
        row = index.chunks[0]
        assert np.array_equal(index.embeddings[0], enc.vector(enc.encoded[0]))
        save_index(index, out)
        assert load_index(out).chunking["header"] is True
        try:  # an index built with headers is not updated as if it had none
            load_index(out).check_compatible(enc, chunking_config(CHUNKING, enc))
        except IndexMismatchError:
            pass
        else:
            raise AssertionError("header/no-header chunking mismatch accepted")

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
