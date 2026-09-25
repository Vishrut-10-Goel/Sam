"""AppsRetrieval corpus (CoIR-Retrieval/apps) as Documents.

Documents keep the dataset's own IDs (d1 ... d8765), since the relevance labels reference them. No chunking:
the benchmark treats each solution as one unit and labels point at whole documents.

The whole corpus is loaded (train and test partitions, 8,765 solutions), matching what MTEB's AppsRetrieval
searches over; the partition is kept in metadata.
"""
from __future__ import annotations

from typing import Iterator

from datasets import load_dataset

from records import Document

CHUNKING = "none"

DATASET = "CoIR-Retrieval/apps"
# Same revision MTEB's AppsRetrieval task pins, so pipeline results are comparable with MTEB's.
REVISION = "f22508f96b7a36c2415181ed8bb76f76e04ae2d5"


def load_apps(revision: str = REVISION) -> Iterator[Document]:
    corpus = load_dataset(DATASET, "corpus", split="corpus", revision=revision)
    for row in corpus:
        yield Document.create(
            id=row["_id"],
            text=row["text"],
            source_path=f"hf://datasets/{DATASET}@{revision}#{row['_id']}",
            language=row["language"].lower(),
            partition=row["partition"],
        )
