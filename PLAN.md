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

Step 4 (retrieval/) and step 5 (real-pipeline apps eval, eval/apps_pipeline.py) done. Step 6: cli.py, per the layout above.
