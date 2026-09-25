# Plan

## Goal

A general code-retrieval system. The apps dataset is one supported input among others; it must also index an
arbitrary folder of source files. Encoder is `embedding/onnx_encoder.py` (fp32 ONNX, max 1024, CPU).

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

## Step 3 — index/

Build from Documents + Chunks, save/load, the encoder fingerprint check, the manifest (doc_id -> content hash +
chunk ids), and incremental update. Atomic writes.

The prebuilt apps index is committed (about 27 MB: 8,765 x 768 float32), so the index directory is an
exception to the `*.npy` gitignore rule.

## Steps 4–6

retrieval/, the real-pipeline apps eval, and cli.py, per the layout above.
