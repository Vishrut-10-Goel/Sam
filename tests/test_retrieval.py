"""Tests for retrieval/ (step 4): scoring, chunk -> document grouping, top-k, snippets.

Uses hand-built indexes with known scores, and the fake encoder from tests.test_index for end-to-end checks.

Run from the project root:  python -m tests.test_retrieval
(Also collectable by pytest if it is installed.)
"""
import tempfile
from pathlib import Path

import numpy as np

import retrieval.search
from index import ChunkRow, DocEntry, Index, IndexMismatchError, build_index, chunking_config
from loaders.directory import CHUNKING, load_directory
from retrieval import Retriever, SnippetReader
from tests.test_index import FakeEncoder

DIM = 4
FINGERPRINT = {"model_name": "stub", "dim": DIM}


class StubEncoder:
    """Returns preset query vectors."""

    dim = DIM

    def __init__(self, vectors: dict[str, np.ndarray]):
        self.vectors = vectors

    def encode(self, texts, batch_size=None, progress=None):
        return np.stack([self.vectors[t] for t in texts]).astype(np.float32)

    def fingerprint(self):
        return FINGERPRINT


def _vec(score: float) -> np.ndarray:
    """Unit vector whose dot product with e0 is score."""
    return np.array([score, np.sqrt(1 - score ** 2), 0, 0], dtype=np.float32)


def _grouping_index() -> Index:
    # Doc a: one chunk (0.9). Doc b: three chunks (0.6, 0.5, 0.7). Doc c: one chunk (0.3). Doc e: no chunks.
    # Rows deliberately interleave documents, as an incremental update can leave them.
    rows = [("a", "a#1", 0.9), ("b", "b#1", 0.6), ("c", "c#1", 0.3), ("b", "b#2", 0.5), ("b", "b#3", 0.7)]
    chunks = [ChunkRow(cid, doc, i + 1, i + 1) for i, (doc, cid, _) in enumerate(rows)]
    embeddings = np.stack([_vec(s) for _, _, s in rows])
    documents = {d: DocEntry(f"hash-{d}", [cid for doc, cid, _ in rows if doc == d], {"source_path": d})
                 for d in ["a", "b", "c", "e"]}
    return Index(embeddings, chunks, documents, FINGERPRINT, {"name": "windows"})


E0 = np.array([1, 0, 0, 0], dtype=np.float32)


def _ranking(results):
    return [(r.doc_id, round(r.score, 4)) for r in results]


def test_max_grouping_and_chunk_hits():
    retriever = Retriever(_grouping_index(), StubEncoder({"q": E0}))
    results = retriever.search("q")
    assert _ranking(results) == [("a", 0.9), ("b", 0.7), ("c", 0.3)]  # e has no chunks: never returned
    b = results[1]
    assert [(h.chunk_id, round(h.score, 4)) for h in b.chunks] == [("b#3", 0.7), ("b#1", 0.6), ("b#2", 0.5)]
    assert (b.chunks[0].start_line, b.chunks[0].end_line) == (5, 5)
    assert b.metadata == {"source_path": "b"}


def test_mean_and_sum_grouping():
    index, enc = _grouping_index(), StubEncoder({"q": E0})
    assert _ranking(Retriever(index, enc, aggregate="mean").search("q")) == [("a", 0.9), ("b", 0.6), ("c", 0.3)]
    assert _ranking(Retriever(index, enc, aggregate="sum").search("q")) == [("b", 1.8), ("a", 0.9), ("c", 0.3)]
    try:
        Retriever(index, enc, aggregate="median")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown aggregate accepted")


def test_top_k_and_chunks_per_doc():
    retriever = Retriever(_grouping_index(), StubEncoder({"q": E0}), chunks_per_doc=1)
    assert [r.doc_id for r in retriever.search("q", top_k=1)] == ["a"]
    assert [r.doc_id for r in retriever.search("q", top_k=100)] == ["a", "b", "c"]
    assert retriever.search("q", top_k=0) == []
    assert all(len(r.chunks) == 1 for r in retriever.search("q"))


def test_search_many_matches_single_queries_across_blocks():
    rng = np.random.default_rng(0)
    queries = {f"q{i}": (v := rng.standard_normal(DIM).astype(np.float32)) / np.linalg.norm(v) for i in range(7)}
    retriever = Retriever(_grouping_index(), StubEncoder(queries))
    singles = [_ranking(retriever.search(q)) for q in queries]
    old = retrieval.search.QUERY_BLOCK
    try:
        retrieval.search.QUERY_BLOCK = 3  # force several blocks
        assert [_ranking(r) for r in retriever.search_many(list(queries))] == singles
    finally:
        retrieval.search.QUERY_BLOCK = old
    assert retriever.search_many([]) == []


def test_incompatible_encoder_rejected():
    class Other(StubEncoder):
        def fingerprint(self):
            return {"model_name": "other", "dim": DIM}
    try:
        Retriever(_grouping_index(), Other({}))
    except IndexMismatchError:
        pass
    else:
        raise AssertionError("mismatched encoder accepted")


def test_end_to_end_directory_search_and_snippets():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for i in range(3):
            (root / f"m{i}.py").write_text("".join(f"def f{i}_{j}(x):\n    return x * {j}\n\n" for j in range(40)),
                                           encoding="utf-8", newline="")
        enc = FakeEncoder()
        index = build_index(load_directory(root), enc, chunking_config(CHUNKING, enc))
        retriever = Retriever(index, enc)
        reader = SnippetReader(index)

        # A chunk's exact text embeds to the same fake vector, so it must be the top hit with score 1.
        target = index.chunks[len(index.chunks) // 2]
        text = "".join((root / target.doc_id).read_text(encoding="utf-8").splitlines(keepends=True)
                       [target.start_line - 1:target.end_line])
        [top, *_] = retriever.search(text, top_k=3)
        assert top.doc_id == target.doc_id and top.chunks[0].chunk_id == target.chunk_id
        assert abs(top.score - 1.0) < 1e-5

        hit = top.chunks[0]
        snip = reader.snippet(top.doc_id, hit.start_line, hit.end_line)
        assert snip.status == "ok" and snip.text == text

        # Source changed since indexing: stale, not wrong lines. Deleted: missing.
        (root / "m0.py").write_text("changed\n", encoding="utf-8")
        assert SnippetReader(index).snippet("m0.py", 1, 1).status == "stale"
        (root / "m1.py").unlink()
        assert SnippetReader(index).snippet("m1.py", 1, 1).status == "missing"



def _kinds_index() -> Index:
    # A doc and a test outscore the code file on raw similarity.
    rows = [("docs/guide.rst", 0.9), ("tests/test_retry.py", 0.8), ("pkg/retry.py", 0.78), ("README.md", 0.5)]
    chunks = [ChunkRow(f"{d}#L1-L1", d, 1, 1) for d, _ in rows]
    documents = {d: DocEntry(f"hash-{d}", [f"{d}#L1-L1"], {}) for d, _ in rows}
    return Index(np.stack([_vec(s) for _, s in rows]), chunks, documents, FINGERPRINT, {"name": "windows"})


def test_kind_penalty_and_code_only():
    enc = StubEncoder({"q": E0})
    raw = Retriever(_kinds_index(), enc).search("q")
    assert [r.doc_id for r in raw] == ["docs/guide.rst", "tests/test_retry.py", "pkg/retry.py", "README.md"]
    weighted = Retriever(_kinds_index(), enc, kind_penalty=0.15).search("q")
    assert [r.doc_id for r in weighted][:2] == ["pkg/retry.py", "docs/guide.rst"]
    assert _ranking(weighted)[0] == ("pkg/retry.py", 0.78)  # reported scores stay raw cosine
    only = Retriever(_kinds_index(), enc, code_only=True).search("q", top_k=10)
    assert [r.doc_id for r in only] == ["pkg/retry.py"]
    try:
        Retriever(_kinds_index(), enc, kind_penalty=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative penalty accepted")


def test_kind_options_do_not_affect_apps_ids():
    enc = StubEncoder({"q": E0})
    index = _grouping_index()  # ids a, b, c: classify as code, like apps ids d1..d8765
    base = _ranking(Retriever(index, enc).search("q"))
    assert _ranking(Retriever(index, enc, kind_penalty=0.5).search("q")) == base
    assert _ranking(Retriever(index, enc, code_only=True).search("q")) == base

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
