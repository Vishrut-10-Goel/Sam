"""Chunk-embedding index: build, incremental update, and atomic save/load. See index.index and index.storage."""
from index.index import (
    ChunkRow,
    DocEntry,
    Index,
    IndexMismatchError,
    UpdateStats,
    build_index,
    chunking_config,
    update_index,
)
from index.storage import load_index, save_index

__all__ = [
    "ChunkRow", "DocEntry", "Index", "IndexMismatchError", "UpdateStats",
    "build_index", "chunking_config", "update_index", "load_index", "save_index",
]
