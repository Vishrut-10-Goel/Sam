"""Real-pipeline AppsRetrieval evaluation: loader -> index -> retriever -> NDCG@10 / MRR@10, computed here.

Unlike eval/run_apps_cpu.py (MTEB drives the encoder and scores), this runs the system's own code path end
to end, so it tests the pipeline itself. With chunking off (apps' setting) it should reproduce the MTEB
number exactly (MTEB reports 5 decimals): same encoder, same document texts (loaders.apps mirrors MTEB's text
preparation), same scoring (cosine) and the same metric definitions and tie-breaking (eval/metrics.py).
A full run compares against the MTEB results JSON and exits 1 on a mismatch.

The index at --index is reused: loaded, checked against the encoder, and incrementally updated (nothing is
re-embedded if the corpus is unchanged). If none exists it is built and saved first (~40 min on CPU), so
--limit only makes a run quick once an index exists.

Run from the project root:
  python -m eval.apps_pipeline                 full test set (3,765 queries, ~40 min of query encoding)
  python -m eval.apps_pipeline --limit 50      first 50 test queries (smoke test)
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path

import loaders.apps
from eval.metrics import evaluate
from index import build_index, chunking_config, load_index, save_index, update_index
from index.storage import MANIFEST
from retrieval import Retriever

print = functools.partial(print, flush=True)  # show progress live when output goes to a log file

DEFAULT_INDEX = Path("indexes/apps")
DEFAULT_MTEB_RESULTS = Path("appsretrieval_results.json")
DEFAULT_OUTPUT = Path("apps_pipeline_results.json")
K = 10
# Retrieve deeper than K so the top K are decided by the metric's tie-breaking, not by where the retriever's
# top-k cut happened to fall among tied scores.
RETRIEVE_DEPTH = 100
LOG_EVERY_SECONDS = 60


def _progress(label: str):
    t0 = last = time.perf_counter()

    def report(done: int, total: int) -> None:
        nonlocal last
        now = time.perf_counter()
        if now - last >= LOG_EVERY_SECONDS or done == total:
            last = now
            elapsed = now - t0
            print(f"    {done:,}/{total:,} {label} ({done / total:.0%})  elapsed {elapsed / 60:.1f} min  "
                  f"eta <= {elapsed / done * (total - done) / 60:.1f} min")

    return report


def prepare_index(index_dir: Path, encoder):
    """Load + incrementally update the apps index at index_dir, or build it; save if anything changed."""
    docs = list(loaders.apps.load_apps())
    chunking = chunking_config(loaders.apps.CHUNKING, encoder)
    source = {"kind": "apps", "dataset": loaders.apps.DATASET, "revision": loaders.apps.REVISION}
    if (index_dir / MANIFEST).exists():
        index = load_index(index_dir)
        index.check_compatible(encoder, chunking)
        index, stats = update_index(index, docs, encoder, source, _progress("chunks"))
        print(f"index {index_dir}: {stats.unchanged:,} unchanged, {stats.added} added, {stats.changed} changed, "
              f"{stats.deleted} deleted")
        if stats.added or stats.changed or stats.deleted:
            save_index(index, index_dir)
    else:
        print(f"index {index_dir}: not found, building ({len(docs):,} documents)")
        index = build_index(docs, encoder, chunking, source, _progress("chunks"))
        save_index(index, index_dir)
    return index


def run(index_dir: Path = DEFAULT_INDEX, limit: int | None = None, encoder=None) -> dict:
    """Evaluate the pipeline on the AppsRetrieval test queries (the first `limit` of them, if given)."""
    if encoder is None:
        from embedding.onnx_encoder import OnnxEncoder
        encoder = OnnxEncoder()
    start = time.perf_counter()
    index = prepare_index(Path(index_dir), encoder)
    t_index = time.perf_counter() - start

    queries, qrels = loaders.apps.load_apps_queries()
    query_ids = list(queries)[:limit] if limit is not None else list(queries)
    print(f"querying: {len(query_ids):,} of {len(queries):,} test queries")
    retriever = Retriever(index, encoder)
    t0 = time.perf_counter()
    results = retriever.search_many([queries[q] for q in query_ids], top_k=RETRIEVE_DEPTH,
                                    progress=_progress("queries"))
    t_query = time.perf_counter() - t0

    run_scores = {q: {r.doc_id: r.score for r in res} for q, res in zip(query_ids, results)}
    metrics = evaluate(run_scores, qrels, k=K)
    return {
        "task": "AppsRetrieval", "split": "test",
        "n_queries": len(query_ids), "limit": limit,
        f"ndcg_at_{K}": metrics[f"ndcg_at_{K}"], f"mrr_at_{K}": metrics[f"mrr_at_{K}"],
        "index": str(index_dir), "n_documents": len(index.documents), "n_chunks": len(index.chunks),
        "encoder": index.fingerprint, "chunking": index.chunking,
        "time_index_s": round(t_index, 1), "time_query_s": round(t_query, 1),
        "per_query": metrics["per_query"],
    }


def compare_with_mteb(result: dict, mteb_path: Path) -> bool | None:
    """True/False if the MTEB number is reproduced (to MTEB's 5-decimal rounding); None if not comparable."""
    if result["limit"] is not None:
        print("(--limit run: not comparable with the full-set MTEB result)")
        return None
    if not mteb_path.exists():
        print(f"(no MTEB results at {mteb_path} to compare against)")
        return None
    mteb = json.loads(mteb_path.read_text(encoding="utf-8"))["scores"]["test"][0]
    ok = True
    for key in (f"ndcg_at_{K}", f"mrr_at_{K}"):
        ours, theirs = round(result[key], 5), round(mteb[key], 5)
        match = abs(ours - theirs) < 5e-6
        ok &= match
        print(f"  {key}: pipeline {ours:.5f}   MTEB {theirs:.5f}   {'MATCH' if match else f'MISMATCH ({ours - theirs:+.5f})'}")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N test queries")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX, help=f"index directory (default {DEFAULT_INDEX})")
    parser.add_argument("--mteb-results", type=Path, default=DEFAULT_MTEB_RESULTS,
                        help=f"MTEB results JSON to compare a full run against (default {DEFAULT_MTEB_RESULTS})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"results JSON (default {DEFAULT_OUTPUT})")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")

    start = time.perf_counter()
    result = run(args.index, args.limit)
    args.output.write_text(json.dumps(result, indent=1), encoding="utf-8")

    print("\n" + "=" * 50)
    print(f"AppsRetrieval (test), real pipeline: {result['n_queries']:,} queries")
    print(f"NDCG@{K}: {result[f'ndcg_at_{K}']:.5f}")
    print(f"MRR@{K}:  {result[f'mrr_at_{K}']:.5f}")
    matched = compare_with_mteb(result, args.mteb_results)
    print(f"Wall time: {(time.perf_counter() - start) / 60:.1f} min   results: {args.output}")
    print("=" * 50)
    return 1 if matched is False else 0


if __name__ == "__main__":
    sys.exit(main())
