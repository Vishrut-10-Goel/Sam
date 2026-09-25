"""In-memory index: chunk embeddings plus the manifest that maps documents to their chunks.

build_index embeds every chunk; update_index re-embeds only documents whose content hash changed (or that are
new), reuses the stored rows of unchanged documents, and drops deleted ones. Both need an encoder with the same
fingerprint as the index, since embeddings from different encoders are not comparable.

The index stores chunk positions (doc_id, start/end line), not chunk text: the document source is
authoritative, and the manifest's content hashes tell whether it still matches what was indexed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from chunking import DEFAULT_OVERLAP_TOKENS, make_chunker
from records import Chunk, Document

ProgressCallback = Callable[[int, int], None]


class IndexMismatchError(ValueError):
    """The index was built with a different encoder or chunking configuration than the one supplied."""


@dataclass(frozen=True)
class ChunkRow:
    """One row of the embedding matrix: which lines of which document it embeds."""

    chunk_id: str
    doc_id: str
    start_line: int
    end_line: int

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> ChunkRow:
        return cls(chunk.chunk_id, chunk.doc_id, chunk.start_line, chunk.end_line)


@dataclass
class DocEntry:
    """Manifest entry for one document."""

    content_hash: str
    chunk_ids: list[str]
    metadata: dict = field(default_factory=dict)


@dataclass
class Index:
    embeddings: np.ndarray       # (n_chunks, dim) float32, L2-normalized, row i embeds chunks[i]
    chunks: list[ChunkRow]
    documents: dict[str, DocEntry]
    fingerprint: dict            # the encoder's fingerprint()
    chunking: dict               # {"name": ..., plus the chunker's settings}
    source: dict = field(default_factory=dict)  # where the documents came from; informational

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Check the embedding matrix, chunk table and manifest agree with each other."""
        if self.embeddings.dtype != np.float32 or self.embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2-D float32, got {self.embeddings.dtype} {self.embeddings.shape}")
        if len(self.chunks) != self.embeddings.shape[0]:
            raise ValueError(f"{len(self.chunks)} chunk rows but {self.embeddings.shape[0]} embeddings")
        if self.chunks and self.embeddings.shape[1] != self.fingerprint.get("dim"):
            raise ValueError(f"embedding dim {self.embeddings.shape[1]} != encoder dim {self.fingerprint.get('dim')}")
        by_doc: dict[str, list[str]] = {}
        for row in self.chunks:
            by_doc.setdefault(row.doc_id, []).append(row.chunk_id)
        if set(by_doc) - set(self.documents):
            raise ValueError(f"chunks reference unknown documents: {sorted(set(by_doc) - set(self.documents))[:5]}")
        for doc_id, entry in self.documents.items():
            if by_doc.get(doc_id, []) != entry.chunk_ids:
                raise ValueError(f"manifest chunk ids for {doc_id!r} do not match the chunk table")

    @property
    def dim(self) -> int:
        return self.embeddings.shape[1]

    def check_compatible(self, encoder, chunking: dict | None = None) -> None:
        """Raise IndexMismatchError unless encoder (and chunking, if given) match what built this index."""
        fingerprint = encoder.fingerprint()
        if fingerprint != self.fingerprint:
            diff = sorted(k for k in fingerprint.keys() | self.fingerprint.keys()
                          if fingerprint.get(k) != self.fingerprint.get(k))
            raise IndexMismatchError(f"encoder differs from the one that built this index (fields: {diff}); rebuild the index")
        if chunking is not None and chunking != self.chunking:
            raise IndexMismatchError(f"chunking {chunking} differs from the index's {self.chunking}; rebuild the index")


@dataclass(frozen=True)
class UpdateStats:
    added: int
    changed: int
    deleted: int
    unchanged: int
    chunks_embedded: int


def chunking_config(name: str, encoder, overlap_tokens: int = DEFAULT_OVERLAP_TOKENS, header: bool = False) -> dict:
    """The full chunking configuration for a loader's CHUNKING name, as recorded in the manifest.

    header is recorded only when on, so indexes built before headers existed keep their exact configuration.
    For "ast" the chunk-size targets are recorded too, so changing them forces a rebuild rather than mixing sizes.
    """
    if name == "none":
        return {"name": "none"}
    config = {"name": name, "max_length": encoder.max_length, "overlap_tokens": overlap_tokens}
    if name == "ast":
        from chunking.ast_chunks import MIN_TOKENS, TARGET_TOKENS
        config.update(target_tokens=TARGET_TOKENS, min_tokens=MIN_TOKENS)
    if header:
        config["header"] = True
    return config


def _chunker(chunking: dict, encoder):
    return make_chunker(chunking["name"], tokenizer=encoder.tokenizer, max_length=chunking.get("max_length"),
                        overlap_tokens=chunking.get("overlap_tokens", DEFAULT_OVERLAP_TOKENS),
                        header=chunking.get("header", False))


def _chunk_and_embed(docs: list[Document], chunking: dict, encoder, progress: ProgressCallback | None):
    chunker = _chunker(chunking, encoder)
    chunks = [chunk for doc in docs for chunk in chunker(doc)]
    if chunks:
        # One encode call for everything, so the encoder can sort all texts by length (minimal padding).
        embeddings = encoder.encode([c.embed_text for c in chunks], progress=progress)
    else:
        embeddings = np.empty((0, encoder.dim), dtype=np.float32)
    return chunks, np.asarray(embeddings, dtype=np.float32)


def _unique(documents: Iterable[Document]) -> dict[str, Document]:
    docs: dict[str, Document] = {}
    for doc in documents:
        if doc.id in docs:
            raise ValueError(f"duplicate document id {doc.id!r}")
        docs[doc.id] = doc
    return docs


def build_index(
    documents: Iterable[Document],
    encoder,
    chunking: dict,
    source: dict | None = None,
    progress: ProgressCallback | None = None,
) -> Index:
    """chunking: from chunking_config(loader.CHUNKING, encoder)."""
    docs = _unique(documents)
    chunks, embeddings = _chunk_and_embed(list(docs.values()), chunking, encoder, progress)
    entries = {doc_id: DocEntry(doc.content_hash, [], dict(doc.metadata)) for doc_id, doc in docs.items()}
    for chunk in chunks:
        entries[chunk.doc_id].chunk_ids.append(chunk.chunk_id)
    return Index(embeddings, [ChunkRow.from_chunk(c) for c in chunks], entries,
                 encoder.fingerprint(), dict(chunking), dict(source or {}))


def update_index(
    index: Index,
    documents: Iterable[Document],
    encoder,
    source: dict | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[Index, UpdateStats]:
    """A new Index for the current documents, re-embedding only new and changed ones.

    documents is the complete current set: any indexed document not in it is removed.
    The input index is not modified.
    """
    index.check_compatible(encoder)
    docs = _unique(documents)

    unchanged = {d for d, doc in docs.items() if d in index.documents
                 and index.documents[d].content_hash == doc.content_hash}
    to_embed = [doc for d, doc in docs.items() if d not in unchanged]
    new_chunks, new_embeddings = _chunk_and_embed(to_embed, index.chunking, encoder, progress)

    keep_rows = [i for i, row in enumerate(index.chunks) if row.doc_id in unchanged]
    embeddings = np.concatenate([index.embeddings[keep_rows], new_embeddings]) if keep_rows else new_embeddings
    rows = [index.chunks[i] for i in keep_rows] + [ChunkRow.from_chunk(c) for c in new_chunks]

    entries: dict[str, DocEntry] = {}
    for doc_id, doc in docs.items():
        # Metadata is refreshed even for unchanged content (e.g. the source path of a moved index root).
        chunk_ids = list(index.documents[doc_id].chunk_ids) if doc_id in unchanged else []
        entries[doc_id] = DocEntry(doc.content_hash, chunk_ids, dict(doc.metadata))
    for chunk in new_chunks:
        entries[chunk.doc_id].chunk_ids.append(chunk.chunk_id)

    stats = UpdateStats(
        added=sum(d not in index.documents for d in docs),
        changed=sum(d in index.documents and d not in unchanged for d in docs),
        deleted=sum(d not in docs for d in index.documents),
        unchanged=len(unchanged),
        chunks_embedded=len(new_chunks),
    )
    new_index = Index(np.ascontiguousarray(embeddings, dtype=np.float32), rows, entries, index.fingerprint,
                      dict(index.chunking), dict(source) if source is not None else dict(index.source))
    return new_index, stats
