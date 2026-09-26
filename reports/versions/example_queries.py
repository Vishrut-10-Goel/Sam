"""Example queries on the versioned requests index (reports/versions_check.md). No ground truth: this records what
search across versions returns, with and without grouping revisions by file.

For each query:
  grouped    what cli.py / the web page show: one result per file, the matched version, the versions holding the
             same lines, and those whose lines differ
  ungrouped  the top 10 revisions ranked directly (no grouping), to show how near-identical versions crowd it
Then one query searched in two single versions, whose best-matching file differs.

Run from the repository root, after building indexes/requests-versions (measure_builds.sh):
  python reports/versions/example_queries.py > reports/versions/example_queries.txt
Writes reports/versions/example_queries.json.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
import cli  # noqa: E402

INDEX = Path("indexes/requests-versions")
QUERIES = [
    "how are redirects handled",
    "how do I configure retries",
    "where is HTTP basic authentication implemented",
    "parse the parameters of a Content-Type header",
    "how are proxy settings read from the environment",
    "where does it raise an exception for 4xx and 5xx responses",
]
PER_VERSION = ("how are proxy settings read from the environment", ["v2.33.0", "v2.34.0"])


def labels(commits):
    return ", ".join(c["label"] for c in commits)


def main():
    encoder = cli._make_encoder()
    session = cli.QuerySession(INDEX, encoder)
    names = {c["sha"]: (c["tags"][0] if c["tags"] else c["short"]) for c in session.index.source["commits"]}
    out = {"index": str(INDEX), "versions": [names[c["sha"]] for c in session.index.source["commits"]], "queries": []}
    for q in QUERIES:
        t0 = time.perf_counter()
        rows = session.search(q, top_k=5)
        ms = (time.perf_counter() - t0) * 1000
        raw = session.retriever.search(q, top_k=10)  # revisions ranked directly, no grouping
        ungrouped = [{"revision": r.doc_id, "path": r.metadata["path"], "score": round(r.rank_score, 4),
                      "versions": [names[s] for s in r.metadata["versions"]]} for r in raw]
        entry = {"query": q, "ms": round(ms, 1), "grouped": [], "ungrouped_top10": ungrouped,
                 "ungrouped_distinct_files": len({u["path"] for u in ungrouped})}
        print(f"\n### {q!r}  ({ms:.0f} ms)")
        print("grouped (one result per file):")
        for r in rows:
            prim, others = r["versions"][0], r["versions"][1:]
            g = {"path": r["doc_id"], "lines": [r["chunks"][0]["start_line"], r["chunks"][0]["end_line"]],
                 "score": r["score"], "matched_in": [c["label"] for c in prim["commits"]],
                 "same_lines_in": [[c["label"] for c in h["commits"]] + [h["match_line"]] for h in others if h["contains_match"]],
                 "lines_differ_in": [[c["label"] for c in h["commits"]] + [h["score"]] for h in others if h["contains_match"] is False]}
            entry["grouped"].append(g)
            print(f"  {r['rank']}. {r['doc_id']}:{g['lines'][0]}-{g['lines'][1]}  score {r['score']:.4f}")
            for line in cli._version_lines(r):
                print(f"       {line}")
        print(f"ungrouped top 10 (revisions ranked directly): {entry['ungrouped_distinct_files']} distinct files")
        for u in ungrouped:
            print(f"     {u['path']:<36} {u['score']:.4f}  [{', '.join(u['versions'])}]")
        out["queries"].append(entry)

    q, versions = PER_VERSION
    print(f"\n### {q!r} in single versions")
    out["per_version"] = {"query": q, "results": {}}
    for v in versions:
        out["per_version"]["results"][v] = []
        print(f"  version {v}:")
        for r in session.search(q, top_k=2, version=v):
            s = r["snippet"]
            at = s["focus_line"] - r["chunks"][0]["start_line"]
            shown = s["text"].splitlines()[at:at + 8]
            out["per_version"]["results"][v].append({"path": r["doc_id"], "revision": r["revision"],
                                                     "score": r["score"], "focus_line": s["focus_line"], "lines": shown})
            print(f"   {r['rank']}. {r['doc_id']} ({r['revision']})  score {r['score']:.4f}")
            for i, line in enumerate(shown):
                print(f"      {s['focus_line'] + i:>5} | {line}")
    (Path(__file__).parent / "example_queries.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
