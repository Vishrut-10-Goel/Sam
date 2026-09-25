"""Save/load an Index as a directory, with crash-safe (atomic) updates.

Layout:
  manifest.json            format version, encoder fingerprint, chunking, source, per-document entries
                           (doc_id -> content hash, chunk ids, metadata), and the data files it refers to
  embeddings-<gen>.npy     (n_chunks, dim) float32
  chunks-<gen>.jsonl       one row per embedding: chunk_id, doc_id, start_line, end_line

manifest.json is the commit point. A save writes the new generation's data files first, then replaces
manifest.json in one os.replace (atomic on the same volume, Windows included), then deletes the previous
generation's files. A crash at any moment leaves manifest.json pointing at a complete set of files: the old
generation or the new one, never a mix. Leftover files from an interrupted save are removed by the next save.

Data files carry sha256 checksums in the manifest, verified on load.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from index.index import ChunkRow, DocEntry, Index

FORMAT_VERSION = 1
MANIFEST = "manifest.json"
_DATA_FILE = re.compile(r"^(embeddings-\d+\.npy|chunks-\d+\.jsonl)$")
_TEMP_FILE = re.compile(r"^\..+\.tmp-\d+$")


def _atomic_write(path: Path, data: bytes) -> None:
    """Write via a temp file in the same directory + os.replace, so path never holds a partial file."""
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_manifest(path: Path) -> dict:
    with open(path / MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{path}: index format {manifest.get('format_version')}, expected {FORMAT_VERSION}; rebuild it")
    return manifest


def _next_generation(path: Path) -> int:
    try:
        return _read_manifest(path)["generation"] + 1
    except (OSError, ValueError, KeyError):
        # No readable manifest: start above any leftover data files so nothing is overwritten while in use.
        gens = [int(m.group(1)) for p in path.iterdir() if (m := re.search(r"-(\d+)\.", p.name))]
        return max(gens, default=0) + 1


def save_index(index: Index, path: str | os.PathLike) -> None:
    index.validate()
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    gen = _next_generation(path)

    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(index.embeddings, dtype=np.float32), allow_pickle=False)
    embeddings_bytes = buf.getvalue()
    chunks_bytes = "".join(
        json.dumps({"chunk_id": r.chunk_id, "doc_id": r.doc_id, "start_line": r.start_line, "end_line": r.end_line},
                   ensure_ascii=False) + "\n"
        for r in index.chunks
    ).encode("utf-8")
    files = {
        "embeddings": {"name": f"embeddings-{gen:06d}.npy", "sha256": _sha256(embeddings_bytes)},
        "chunks": {"name": f"chunks-{gen:06d}.jsonl", "sha256": _sha256(chunks_bytes)},
    }
    _atomic_write(path / files["embeddings"]["name"], embeddings_bytes)
    _atomic_write(path / files["chunks"]["name"], chunks_bytes)

    manifest = {
        "format_version": FORMAT_VERSION,
        "generation": gen,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "files": files,
        "fingerprint": index.fingerprint,
        "chunking": index.chunking,
        "source": index.source,
        "n_documents": len(index.documents),
        "n_chunks": len(index.chunks),
        "dim": index.dim,
        "documents": {
            doc_id: {"content_hash": e.content_hash, "chunk_ids": e.chunk_ids, "metadata": e.metadata}
            for doc_id, e in sorted(index.documents.items())
        },
    }
    _atomic_write(path / MANIFEST, json.dumps(manifest, indent=1, ensure_ascii=False).encode("utf-8"))

    # Committed. Remove other generations' data files and temp files left by interrupted saves.
    keep = {f["name"] for f in files.values()}
    for p in path.iterdir():
        if (_DATA_FILE.match(p.name) and p.name not in keep) or _TEMP_FILE.match(p.name):
            try:
                p.unlink()
            except OSError:
                pass  # e.g. still open elsewhere on Windows; the next save retries


def _read_checked(path: Path, entry: dict) -> bytes:
    data = (path / entry["name"]).read_bytes()
    if _sha256(data) != entry["sha256"]:
        raise ValueError(f"{path / entry['name']}: checksum mismatch; the index is corrupted, rebuild it")
    return data


def load_index(path: str | os.PathLike, encoder=None) -> Index:
    """Load an index. If encoder is given, also check it matches the one that built the index."""
    path = Path(path)
    manifest = _read_manifest(path)
    files = manifest["files"]
    embeddings = np.load(io.BytesIO(_read_checked(path, files["embeddings"])), allow_pickle=False)
    chunks = [ChunkRow(**json.loads(line)) for line in
              _read_checked(path, files["chunks"]).decode("utf-8").splitlines() if line]
    documents = {doc_id: DocEntry(e["content_hash"], e["chunk_ids"], e["metadata"])
                 for doc_id, e in manifest["documents"].items()}
    index = Index(embeddings, chunks, documents, manifest["fingerprint"], manifest["chunking"], manifest["source"])
    if encoder is not None:
        index.check_compatible(encoder)
    return index
