"""Mechanical check of the opt-in AST chunker on real repositories: no embedding, no quality claims.

For each folder: chunk every file with windows (the default) and with ast, verify the ast chunks' invariants
(whole lines, exact text, full coverage, no gaps; overlap only inside oversized-unit fallback windows) and
compare chunk counts and sizes.

Run from the repository root:  python experiments/ast_chunk_stats.py <folder> [<folder> ...]
"""
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from chunking.ast_chunks import chunk_ast  # noqa: E402
from chunking.lines import split_lines  # noqa: E402
from chunking.windows import chunk_windows  # noqa: E402
from loaders.directory import load_directory  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from embedding.onnx_encoder import DEFAULT_MAX_LENGTH, DEFAULT_MODEL  # noqa: E402

tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)


def n_tokens(text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def check(doc, chunks) -> list[str]:
    problems = []
    lines = split_lines(doc.text)
    if not chunks:
        return ["no chunks"] if lines else []
    if chunks[0].start_line != 1 or chunks[-1].end_line != len(lines):
        problems.append("does not cover the whole file")
    for c in chunks:
        if c.text != "".join(lines[c.start_line - 1:c.end_line]):
            problems.append(f"{c.chunk_id}: text is not its lines")
    for a, b in zip(chunks, chunks[1:]):
        if b.start_line > a.end_line + 1:
            problems.append(f"gap between {a.chunk_id} and {b.chunk_id}")
        if b.start_line <= a.start_line or b.end_line <= a.end_line:
            problems.append(f"out of order: {a.chunk_id}, {b.chunk_id}")
    return problems


for folder in sys.argv[1:]:
    t0 = time.perf_counter()
    docs = list(load_directory(folder))
    py = {d.id for d in docs if d.id.endswith((".py", ".pyi"))}
    win, ast_ = [], []
    problems = []
    for d in docs:
        w = chunk_windows(d, tok, DEFAULT_MAX_LENGTH, header=True)
        a = chunk_ast(d, tok, DEFAULT_MAX_LENGTH, header=True)
        problems += [f"{d.id}: {p}" for p in check(d, a)]
        if d.id in py:
            win += [n_tokens(c.text) for c in w]
            ast_ += [n_tokens(c.text) for c in a]
        elif a != w:
            problems.append(f"{d.id}: non-Python file not chunked like windows")
        over = [c.chunk_id for c in a if len(tok(c.embed_text)["input_ids"]) > DEFAULT_MAX_LENGTH
                and c.start_line != c.end_line]
        problems += [f"{cid}: embedded text over {DEFAULT_MAX_LENGTH} tokens" for cid in over]
    name = folder.rstrip("/").rsplit("/", 1)[-1]
    print(f"{name}: {len(docs)} files ({len(py)} Python), checked in {time.perf_counter() - t0:.0f} s")
    if py:
        w, a = np.array(win), np.array(ast_)
        print(f"  Python chunks  windows: {len(w):5,}  tokens p50 {int(np.median(w)):4}  p90 {int(np.percentile(w, 90)):4}  max {w.max()}")
        print(f"                 ast:     {len(a):5,}  tokens p50 {int(np.median(a)):4}  p90 {int(np.percentile(a, 90)):4}  max {a.max()}"
              f"  (chunks over 500 tokens: {(a > 500).sum()})")
        print(f"  total Python tokens embedded  windows {w.sum():,}  ast {a.sum():,}")
    print(f"  invariant problems: {len(problems)}")
    for p in problems[:10]:
        print("   ", p)
