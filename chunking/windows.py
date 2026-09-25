"""Token-budgeted, line-aligned, overlapping windows.

Each chunk is a run of whole lines whose token count (under the encoder's own tokenizer, plus its special
tokens) fits the encoder's max length, so nothing in a chunk is truncated and results can cite exact lines.
Consecutive chunks share up to overlap_tokens worth of lines, so code near a boundary appears whole in at
least one chunk.

Only exception: a single line longer than the whole budget becomes a chunk on its own, and the encoder
truncates it (splitting it would break the line-number citation).
"""
from __future__ import annotations

from bisect import bisect_right

from chunking.lines import split_lines
from records import Chunk, Document

DEFAULT_OVERLAP_TOKENS = 128


def _line_token_counts(lines: list[str], tokenizer) -> list[int]:
    """Tokens per line, from one tokenization of the whole text (each token counted on the line where it starts).

    Tokenizing the whole text rather than line by line keeps merges across line breaks (e.g. "\\n" plus the
    next line's indentation as one token) counted the way the encoder will actually see them.
    """
    starts, pos = [], 0
    for line in lines:
        starts.append(pos)
        pos += len(line)
    offsets = tokenizer("".join(lines), add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    counts = [0] * len(lines)
    for start, _ in offsets:
        counts[bisect_right(starts, start) - 1] += 1
    return counts


def chunk_windows(doc: Document, tokenizer, max_length: int, overlap_tokens: int = DEFAULT_OVERLAP_TOKENS) -> list[Chunk]:
    lines = split_lines(doc.text)
    if not lines:
        return []
    budget = max_length - tokenizer.num_special_tokens_to_add()
    if not 0 <= overlap_tokens < budget:
        raise ValueError(f"overlap_tokens must be in [0, {budget}), got {overlap_tokens}")
    counts = _line_token_counts(lines, tokenizer)

    def n_tokens(first: int, stop: int) -> int:
        return len(tokenizer("".join(lines[first:stop]), add_special_tokens=False)["input_ids"])

    chunks = []
    start = 0
    while True:
        # Greedily take lines while the estimated count fits.
        stop, total = start + 1, counts[start]
        while stop < len(lines) and total + counts[stop] <= budget:
            total += counts[stop]
            stop += 1
        # The estimate can be off by a token or two at the window's edges (merges across its boundaries),
        # so check the real count and drop trailing lines until it fits.
        while stop - start > 1 and n_tokens(start, stop) > budget:
            stop -= 1

        chunks.append(Chunk(
            chunk_id=f"{doc.id}#L{start + 1}-L{stop}",
            doc_id=doc.id,
            text="".join(lines[start:stop]),
            start_line=start + 1,
            end_line=stop,
        ))
        if stop == len(lines):
            return chunks

        # Next window starts early enough to repeat up to overlap_tokens of trailing lines, but always
        # at least one line later than this one (so the loop makes progress), and never with so much
        # overlap that line `stop` would not fit after it (which would repeat this window's end).
        max_overlap = min(overlap_tokens, budget - counts[stop])
        next_start, overlap = stop, 0
        while next_start - 1 > start and overlap + counts[next_start - 1] <= max_overlap:
            next_start -= 1
            overlap += counts[next_start]
        start = next_start
