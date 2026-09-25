# Plan

## Goal

A general code-retrieval system. The apps dataset is one supported input among others; it must also index an
arbitrary folder of source files. Encoder is `embedding/onnx_encoder.py` (fp32 ONNX, max 1024, CPU).

## Hackathon submission goals

**P0 — Retrieval accuracy.** Screened competitively on NDCG@10 and MRR over the CoIR AppsRetrieval test
split, submitted as an MTEB results JSON attached to a GitHub release tagged `PRISM_GENAI_HACKATHON_Y2026`.

**P1 — Retrieval across versions.** Codebases change constantly. The system must rebuild indexes and caches
for any version or change in reasonable time. Evaluated hands-on, not by the JSON. This is what index/'s
incremental update and the stale-snippet check serve.

**Bonus — Evolutionary retrieval.** Retrieve across all versions of the snippets, not just the current one.
Hard because near-identical versions are difficult to rank against each other. Not started.

## Layout

```
loaders/     apps.py, directory.py  -> Document
chunking/    none (apps), windows (directory); AST later
embedding/   onnx_encoder.py        [done]
index/       build, save/load, manifest, incremental update, encoder fingerprint
retrieval/   query embed, score, chunk -> document grouping
eval/        MTEB AbsEncoder wrapper [done] + real-pipeline apps eval
cli.py       index <path>, query "<text>" --index [--top-k] [--json], eval-apps [--limit]
```

Work proceeds one step at a time, stopping for review after each step.

## Step 2 — loaders + record types only  [done]

Record types:

- `Document(id, text, metadata)` — metadata: source path, language, content hash
- `Chunk(chunk_id, doc_id, text, start_line, end_line)`

`loaders/apps.py`: loads the CoIR-Retrieval/apps corpus, emits Documents with the dataset's own IDs (d5001 etc).
No chunking — the benchmark treats one solution as one unit and relevance labels point at whole documents.

`loaders/directory.py`: walks a folder, emits one Document per source file.

- IDs: path relative to the indexed root, always forward slashes (this is Windows). IDs must be STABLE across
  re-indexing — never derived from list position or iteration order, since step 3's incremental updates
  depend on that.
- Skip: .git, node_modules, venv, \_\_pycache\_\_, dist, build; binaries; minified files; anything that fails to
  decode as UTF-8; files over a size threshold. Respect .gitignore if present.
- Content hash per document (step 3 uses it to detect changes).
- Index all text files rather than an extension allow-list, since an allow-list would silently drop file types
  a general system should handle.

`chunking/`: token-based windows sized to the encoder's 1024 limit, aligned to line boundaries so results can
cite line numbers, with overlap. Chunking is a per-loader setting — off for apps, on for directory — not a
stage everything passes through.

## Step 3 — index/  [done]

Build from Documents + Chunks, save/load, the encoder fingerprint check, the manifest (doc_id -> content hash +
chunk ids), and incremental update. Atomic writes.

The prebuilt apps index is committed (about 27 MB: 8,765 x 768 float32), so the index directory is an
exception to the `*.npy` gitignore rule.
Index files keep their generation-numbered names. Git stores content, not names, so every committed rebuild
with new embeddings adds ~27 MB to history regardless of filenames: commit an index rebuild only when the
corpus or encoder actually changes, never intermediate development rebuilds.

## Steps 4–6

Step 4 (retrieval/), step 5 (real-pipeline apps eval, eval/apps_pipeline.py) and step 6 (cli.py, including
`query --interactive`) done. The real pipeline reproduces MTEB exactly (0.57545 / 0.52798).

## Next: retrieval quality on real code

The Scrapy check (reports/scrapy_retrieval_check.md) gave a strict top-1 of 5/10: docs and tests outrank code on
plain-language questions, 1024-token windows blend several functions into one vector, and vocabulary gaps miss.
Changes for folder indexes only (apps stays unchunked, so P0 is unaffected):

1. [done] File-kind weighting: tests and docs rank below code (penalty 0.05 by default), `--code-only` drops
   them, `index --source-only` skips them. Strict top-1 5/10 -> 8/10, MRR 0.633 -> 0.850.
2. [done] Context header embedded with each chunk: file path + overlapping Python classes/functions (ast).
   Top-1 unchanged (8/10), right file in top 3 for 10/10, MRR 0.850 -> 0.867, ~5% more chunks.
3. **Next: AST chunking.** Function/class-level chunks (Python `ast`; tree-sitter for other languages),
   ~200-500 tokens, falling back to line windows for oversized units and non-code files. Targets the two
   remaining #3 answers (Q6, Q10), coarse ~90-line citations, and indexing cost (43 min for Scrapy).
4. Measure every change against the same pre-registered Scrapy queries before keeping it; add fresh queries
   too, since the current ten have now been used to choose the weighting penalty.
