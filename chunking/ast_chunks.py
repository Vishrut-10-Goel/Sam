"""Function/class-level chunks for Python, from the ast. Opt-in: `cli.py index --chunking ast`.

Line windows (chunking.windows) pack every chunk to the encoder's limit, so one vector often blends imports and
several functions. Here the split points come from the code's structure instead:

1. Units: each top-level function or class is a unit, and each run of other module-level statements (imports,
   constants, `if __name__ == ...`) is one unit. A unit starts at its first decorator, or at the comment block
   directly above it, so leading comments stay with the definition they describe.
2. A class larger than TARGET_TOKENS is split into its head (class line, docstring, attributes) and one unit per
   method, the same way.
3. Adjacent small units are merged while the chunk is under MIN_TOKENS, never beyond TARGET_TOKENS, so tiny
   functions do not each become a chunk of their own.
4. A unit still larger than TARGET_TOKENS (a very long function) falls back to line windows of TARGET_TOKENS
   with FALLBACK_OVERLAP_TOKENS of overlap.

Non-Python files, and Python files that do not parse, use the regular line windows (chunk_windows) unchanged.
Chunks keep the same contract as everywhere else: whole lines, exact text, 1-based inclusive line numbers,
optional context header (header=True) embedded in front of the text but never cited.
"""
from __future__ import annotations

import ast

from chunking.lines import split_lines
from chunking.windows import (
    _HEADER_SLACK, DEFAULT_OVERLAP_TOKENS, HEADER_TOKENS, chunk_windows, context_header, python_definitions,
)
from records import Chunk, Document

TARGET_TOKENS = 500           # a chunk never exceeds this (or the encoder's budget, if smaller), except single
                              # lines longer than it
MIN_TOKENS = 200              # adjacent units are merged until a chunk reaches this
FALLBACK_OVERLAP_TOKENS = 64  # overlap for line windows inside an oversized unit

_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _start_line(node: ast.stmt, lines: list[str], floor: int) -> int:
    """First line of node: its first decorator, extended up over a directly preceding comment block (not past floor)."""
    first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    while first - 1 > floor and lines[first - 2].lstrip().startswith("#"):
        first -= 1
    return first


def _segments(body: list[ast.stmt], lines: list[str], first: int, last: int) -> list[tuple[int, int, ast.stmt | None]]:
    """Split lines first..last into contiguous (start, end, node) segments, one per definition in body and one per
    run of other statements (node None). Together they cover first..last exactly."""
    starts: list[tuple[int, ast.stmt | None]] = []
    previous_was_definition = True
    for node in body:
        is_definition = isinstance(node, _DEFINITIONS)
        if is_definition or previous_was_definition:
            floor = starts[-1][0] if starts else first - 1
            starts.append((max(_start_line(node, lines, floor), first), node if is_definition else None))
        previous_was_definition = is_definition
    if not starts:
        return [(first, last, None)]
    starts[0] = (first, starts[0][1])  # anything before the first statement (module docstring gap, etc.)
    ends = [s for s, _ in starts[1:]] + [last + 1]
    return [(s, e - 1, n) for (s, n), e in zip(starts, ends) if e - 1 >= s]


def _units(tree: ast.Module, lines: list[str], n_tokens, target: int) -> list[tuple[int, int]]:
    """(first, last) line ranges of the units, in order, covering the whole file."""
    units: list[tuple[int, int]] = []
    for start, end, node in _segments(tree.body, lines, 1, len(lines)):
        if isinstance(node, ast.ClassDef) and n_tokens(start, end) > target and node.body:
            body_first = _start_line(node.body[0], lines, floor=start)  # includes a first method's decorators
            head_end = body_first - 1
            # Split the class body into head (class line .. first method) and one segment per method.
            inner = _segments(node.body, lines, body_first, end)
            if inner and inner[0][2] is None:  # docstring/attributes before the first method join the head
                head_end = inner[0][1]
                inner = inner[1:]
            units.append((start, head_end))
            units.extend((s, e) for s, e, _ in inner)
        else:
            units.append((start, end))
    return [(s, e) for s, e in units if e >= s]


def _merge(units: list[tuple[int, int]], n_tokens, target: int, minimum: int) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in units:
        if merged:
            prev_start, prev_end = merged[-1]
            if n_tokens(prev_start, prev_end) < minimum and n_tokens(prev_start, end) <= target:
                merged[-1] = (prev_start, end)
                continue
        merged.append((start, end))
    return merged


def chunk_ast(
    doc: Document, tokenizer, max_length: int, overlap_tokens: int = DEFAULT_OVERLAP_TOKENS, header: bool = False
) -> list[Chunk]:
    """AST chunks for Python; chunk_windows(doc, tokenizer, max_length, overlap_tokens, header) for anything else."""
    if not doc.id.endswith((".py", ".pyi")):
        return chunk_windows(doc, tokenizer, max_length, overlap_tokens, header=header)
    try:
        tree = ast.parse(doc.text)
    except (SyntaxError, ValueError):
        return chunk_windows(doc, tokenizer, max_length, overlap_tokens, header=header)
    lines = split_lines(doc.text)
    if not lines:
        return []

    cache: dict[tuple[int, int], int] = {}

    def n_tokens(first: int, last: int) -> int:
        if (first, last) not in cache:
            cache[first, last] = len(tokenizer("".join(lines[first - 1:last]), add_special_tokens=False)["input_ids"])
        return cache[first, last]

    definitions = python_definitions(doc.text) if header else []
    specials = tokenizer.num_special_tokens_to_add()
    # What fits the encoder once special tokens (and the header, if on) are accounted for.
    budget = max_length - specials - (HEADER_TOKENS + _HEADER_SLACK if header else 0)
    target = min(TARGET_TOKENS, budget)
    minimum = min(MIN_TOKENS, target)
    overlap = min(FALLBACK_OVERLAP_TOKENS, target // 4)
    chunks: list[Chunk] = []
    for first, last in _merge(_units(tree, lines, n_tokens, target), n_tokens, target, minimum):
        if n_tokens(first, last) <= target:
            ranges = [(first, last)]
        else:  # oversized unit: line windows of `target` tokens inside it
            sub = Document(id=doc.id, text="".join(lines[first - 1:last]))
            ranges = [(first + c.start_line - 1, first + c.end_line - 1)
                      for c in chunk_windows(sub, tokenizer, target + specials, overlap)]
        for s, e in ranges:
            chunks.append(Chunk(
                chunk_id=f"{doc.id}#L{s}-L{e}",
                doc_id=doc.id,
                text="".join(lines[s - 1:e]),
                start_line=s,
                end_line=e,
                header=context_header(doc.id, definitions, s, e, tokenizer) if header else "",
            ))
    return chunks
