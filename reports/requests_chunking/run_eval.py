"""Windows vs AST chunking on the ten pre-registered psf/requests questions (reports/requests_ground_truth.json).

Same grading as the Scrapy check (reports/scrapy_retrieval_check.md):
  file rank    rank of the first result whose file is a pre-registered correct file
  chunk rank   rank of the first result whose file is correct AND whose best chunk overlaps a correct span
  strict top-1 chunk rank == 1;  file in top 3 / top 10;  MRR (file) = mean of 1 / file rank (0 if not in the top 10)
Retrieval uses the CLI defaults (kind penalty 0.05, max grouping), top 10. The model is loaded once.

Run from the repository root, after building the two indexes (see index_log.txt):
  python reports/requests_chunking/run_eval.py indexes/eval-requests-windows indexes/eval-requests-ast
Writes reports/requests_chunking/results.json.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
from cli import DEFAULT_KIND_PENALTY, _make_encoder  # noqa: E402
from index import load_index  # noqa: E402
from retrieval import Retriever  # noqa: E402

HERE = Path(__file__).parent
TRUTH = json.loads((HERE.parent / "requests_ground_truth.json").read_text(encoding="utf-8"))["queries"]


def grade(results, truth):
    spans = {}
    for path, a, b in truth["correct"]:
        spans.setdefault(path, []).append((a, b))
    file_rank = chunk_rank = None
    for rank, r in enumerate(results, 1):
        if r.doc_id not in spans:
            continue
        file_rank = file_rank or rank
        best = r.chunks[0]
        if chunk_rank is None and any(best.start_line <= b and best.end_line >= a for a, b in spans[r.doc_id]):
            chunk_rank = rank
    return file_rank, chunk_rank


def main(index_dirs):
    encoder = _make_encoder()
    out = {}
    for index_dir in index_dirs:
        index = load_index(Path(index_dir), encoder=encoder)
        retriever = Retriever(index, encoder, kind_penalty=DEFAULT_KIND_PENALTY)
        per_query = []
        for t in TRUTH:
            t0 = time.perf_counter()
            results = retriever.search(t["q"], top_k=10)
            ms = (time.perf_counter() - t0) * 1000
            file_rank, chunk_rank = grade(results, t)
            per_query.append({
                "q": t["q"], "file": file_rank, "chunk": chunk_rank, "ms": round(ms, 1),
                "top10": [{"doc_id": r.doc_id, "lines": [r.chunks[0].start_line, r.chunks[0].end_line],
                           "score": round(r.rank_score, 4), "similarity": round(r.score, 4), "kind": r.kind}
                          for r in results],
            })
        lines = [row.end_line - row.start_line + 1 for row in index.chunks]
        lines.sort()
        n = len(per_query)
        out[Path(index_dir).name] = {
            "chunking": index.chunking,
            "n_documents": len(index.documents),
            "n_chunks": len(index.chunks),
            "chunk_lines_p50": lines[len(lines) // 2],
            "strict_top1": sum(q["chunk"] == 1 for q in per_query),
            "file_top1": sum(q["file"] == 1 for q in per_query),
            "file_top3": sum(q["file"] is not None and q["file"] <= 3 for q in per_query),
            "file_top10": sum(q["file"] is not None for q in per_query),
            "mrr_file": round(sum(1 / q["file"] for q in per_query if q["file"]) / n, 3),
            "per_query": per_query,
        }
        s = out[Path(index_dir).name]
        print(f"{Path(index_dir).name}: {s['n_chunks']} chunks (median {s['chunk_lines_p50']} lines); strict top-1 "
              f"{s['strict_top1']}/{n}, file top-1 {s['file_top1']}/{n}, top-3 {s['file_top3']}/{n}, "
              f"top-10 {s['file_top10']}/{n}, MRR {s['mrr_file']:.3f}")
        for i, q in enumerate(per_query, 1):
            top = q["top10"][0]
            print(f"  Q{i:<2} file {q['file']!s:>4} chunk {q['chunk']!s:>4}  top: {top['doc_id']}:"
                  f"{top['lines'][0]}-{top['lines'][1]}  ({q['ms']:.0f} ms)")
    (HERE / "results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
