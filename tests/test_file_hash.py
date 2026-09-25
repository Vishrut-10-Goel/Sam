"""Tests for the cross-process file hash cache used by the encoder fingerprint (no model load).

Run from the project root:  python -m tests.test_file_hash
(Also collectable by pytest if it is installed.)
"""
import hashlib
import json
import os
import tempfile
from pathlib import Path

from embedding.onnx_encoder import file_sha256


def test_hash_is_correct_and_cached():
    with tempfile.TemporaryDirectory() as tmp:
        f, cache = Path(tmp, "model.bin"), Path(tmp, "cache")
        f.write_bytes(b"weights v1" * 1000)
        expected = hashlib.sha256(f.read_bytes()).hexdigest()
        assert file_sha256(f, cache) == expected

        # Prove the second call uses the cache: plant a marker under the same size+mtime stamp.
        cache_file = cache / "sha256_cache.json"
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        data[str(f.resolve())]["sha256"] = "cached-marker"
        cache_file.write_text(json.dumps(data), encoding="utf-8")
        assert file_sha256(f, cache) == "cached-marker"


def test_changed_file_is_rehashed():
    with tempfile.TemporaryDirectory() as tmp:
        f, cache = Path(tmp, "model.bin"), Path(tmp, "cache")
        f.write_bytes(b"a" * 100)
        first = file_sha256(f, cache)
        f.write_bytes(b"b" * 100)  # same size, new content
        st = f.stat()
        os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))  # guarantee a new mtime
        assert file_sha256(f, cache) == hashlib.sha256(b"b" * 100).hexdigest() != first


def test_corrupt_or_unwritable_cache_still_hashes():
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp, "model.bin")
        f.write_bytes(b"x")
        cache = Path(tmp, "cache")
        cache.mkdir()
        (cache / "sha256_cache.json").write_text("{not json", encoding="utf-8")
        assert file_sha256(f, cache) == hashlib.sha256(b"x").hexdigest()
        blocker = Path(tmp, "blocker")
        blocker.write_text("a file, so the cache dir cannot be created under it", encoding="utf-8")
        assert file_sha256(f, blocker / "cache") == hashlib.sha256(b"x").hexdigest()


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
