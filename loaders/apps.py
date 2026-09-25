"""AppsRetrieval (CoIR-Retrieval/apps): the corpus as Documents, plus the test queries and relevance labels.

Documents keep the dataset's own IDs (d1 ... d8765), since the relevance labels reference them. No chunking:
the benchmark treats each solution as one unit and labels point at whole documents.

The whole corpus is loaded (train and test partitions, 8,765 solutions), matching what MTEB's AppsRetrieval
searches over; the partition is kept in metadata.

Document text is built exactly as MTEB builds it for encoding: "{title} {text}" stripped, or the stripped text
when there is no title (apps has none). 3,687 of the 8,765 raw solutions carry leading/trailing whitespace
that this strips, so using the raw text would embed different strings than MTEB and keep the pipeline from
reproducing MTEB's score.
"""
from __future__ import annotations

from typing import Iterator

from datasets import load_dataset

from records import Document

CHUNKING = "none"

DATASET = "CoIR-Retrieval/apps"
# Same revision MTEB's AppsRetrieval task pins, so pipeline results are comparable with MTEB's.
REVISION = "f22508f96b7a36c2415181ed8bb76f76e04ae2d5"


def document_text(title: str | None, text: str) -> str:
    """Mirror of mteb._create_dataloaders._corpus_to_dict (mteb 2.21)."""
    return (title + " " + text).strip() if title else text.strip()


def load_apps(revision: str = REVISION) -> Iterator[Document]:
    corpus = load_dataset(DATASET, "corpus", split="corpus", revision=revision)
    for row in corpus:
        yield Document.create(
            id=row["_id"],
            text=document_text(row["title"], row["text"]),
            source_path=f"hf://datasets/{DATASET}@{revision}#{row['_id']}",
            language=row["language"].lower(),
            partition=row["partition"],
        )


def load_apps_queries(
    split: str = "test", revision: str = REVISION
) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """(queries, qrels) for a split: query_id -> text (qrels order), query_id -> {doc_id: relevance}.

    Query text is used as-is, as MTEB does (AppsRetrieval has no query instructions).
    """
    qrels_ds = load_dataset(DATASET, "default", split=split, revision=revision)
    all_queries = load_dataset(DATASET, "queries", split="queries", revision=revision)
    text = dict(zip(all_queries["_id"], all_queries["text"]))
    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_ds:
        qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    return {qid: text[qid] for qid in qrels}, qrels
