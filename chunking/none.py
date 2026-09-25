"""No chunking: the whole document is one chunk, with chunk_id == doc_id.

Used for apps, where relevance labels point at whole solutions. Anything past the encoder's max length is
truncated by the encoder, exactly as in the MTEB evaluation.
"""
from __future__ import annotations

from chunking.lines import split_lines
from records import Chunk, Document


def chunk_none(doc: Document) -> list[Chunk]:
    return [Chunk(chunk_id=doc.id, doc_id=doc.id, text=doc.text, start_line=1,
                  end_line=max(len(split_lines(doc.text)), 1))]
