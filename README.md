# Code Retrieval — PRISM GenAI Hackathon 2026

> Release tag: **`PRISM_GENAI_HACKATHON_Y2026`**

Text-to-code retrieval: given a natural-language programming problem, retrieve the code solution that solves it.
Evaluated on the MTEB **AppsRetrieval** task (CoIR benchmark, `CoIR-Retrieval/apps`).

A general, CPU-only code-retrieval system built on **gte-modernbert-base**, exported to fp32 ONNX and run with
onnxruntime. It indexes the AppsRetrieval corpus or any folder of source files, keeps indexes current with
incremental updates as code changes, and answers queries from a command-line tool with file paths, line ranges and
snippets. On the AppsRetrieval test split it scores **NDCG@10 0.57545 / MRR@10 0.52798 on CPU** (MTEB), against
0.0660 / 0.0558 for the all-MiniLM-L6-v2 baseline. That matches the 0.5754 NDCG@10 its model card reports.

## Problem

- **Queries:** full competitive-programming problem statements (story, input/output spec, examples). Mean ≈ 500 tokens, longest ≈ 1,700.
- **Corpus:** 8,765 Python solutions; each of the 3,765 test queries has exactly one relevant solution.
- **Metrics:** NDCG@10 (MTEB main score) and MRR@10.

## Hackathon goals

See [PLAN.md](PLAN.md) for the full plan.

- **P0 — Retrieval accuracy:** NDCG@10 and MRR@10 on the AppsRetrieval test split, submitted as the MTEB results
  JSON (`appsretrieval_results.json`, produced by `python -m eval.run_apps_cpu`).
- **P1 — Retrieval across versions:** incremental index updates (only new or changed files are re-embedded) and
  stale-snippet detection (results from files that changed since indexing are flagged, not shown with wrong lines).
- **Bonus — Evolutionary retrieval** across all versions of a snippet: not started.

## Approach

### Model

- **gte-modernbert-base** (`Alibaba-NLP/gte-modernbert-base`): a ModernBERT encoder with an 8k-token context,
  CLS pooling and cosine similarity. See Results for the comparison.
- **Max length 1024 tokens.** This truncates 82 test queries (2.2%) and costs 0.0014 NDCG@10 compared with 2048, but
  cuts the estimated CPU encode time from ~129 to ~78 min.
- `jinaai/jina-embeddings-v2-base-code` was dropped: its custom model code is incompatible with transformers 5.x.

### Encoder (`embedding/onnx_encoder.py`)

- **fp32 ONNX export run directly with onnxruntime** on CPU. sentence-transformers' ONNX backend needs optimum-onnx,
  which requires transformers < 4.58 and conflicts with transformers 5.x.
- **Why fp32 and not int8:** the int8 export roughly halved the gap between matched and mismatched query/solution
  similarities on our sanity pairs. fp32 matches the PyTorch model.
- **Pipeline:** tokenize, run onnxruntime, take the CLS token, L2-normalize.
- **Batching:** batch size 4 (the fastest fp32 setting measured). Texts are sorted longest first, so each batch pads to
  similar lengths.
- **Parity test:** matches the sentence-transformers PyTorch fp32 model to within 1 − cos = 1.1e-12, including an input
  truncated at 1024 tokens.
- **Fingerprint:** model, ONNX file sha256, max length, pooling and dimension. Every index records it, and an index is
  never queried or updated with a different encoder. The ONNX file's hash is cached across runs, keyed by file size
  and modification time.

### Pipeline

`loaders/` (Documents) → `chunking/` (Chunks) → `index/` (embeddings + manifest) → `retrieval/` (ranked documents
with line ranges) → `cli.py`.

- **Loaders**
  - `loaders/apps.py` yields the 8,765 corpus solutions under their dataset IDs (`d1`…). The text is built exactly as
    MTEB builds it for encoding (`"{title} {text}".strip()`). 3,687 raw solutions carry surrounding whitespace, so
    this is required to reproduce MTEB's number.
  - `loaders/directory.py` yields one Document per text file. IDs are root-relative forward-slash paths, stable across
    re-indexing.
  - It skips `.git`, `node_modules`, `venv`, `.venv`, `__pycache__`, `dist` and `build`, anything a `.gitignore`
    excludes (nested files and `!` negation are supported), symlinks, files over 1 MB, binaries, non-UTF-8 files,
    empty files and minified files.
- **Chunking** is set per loader.
  - Apps solutions are embedded whole: relevance labels point at whole solutions.
  - Folder files are split into windows of whole lines that fit the encoder's 1024 tokens (special tokens included),
    measured with its own tokenizer, with 128 tokens of overlap. Results can therefore cite exact line numbers.
- **Index** (`index/`)
  - Stores the embedding matrix, a row-aligned chunk table (`chunk_id, doc_id, start_line, end_line`), and a manifest
    mapping `doc_id` to content hash, chunk ids and metadata.
  - **Incremental update:** documents whose content hash is unchanged keep their stored embeddings. Only new or changed
    documents are re-embedded, and deleted ones are dropped. The result is identical to a full rebuild.
  - **Atomic saves:** data files are written under generation-numbered names, then `manifest.json` is swapped in with
    `os.replace`. A crash leaves either the old index or the new one, never a mix.
  - Data files are checksummed in the manifest and verified on load.
- **Retrieval** (`retrieval/`)
  - Cosine similarity between the query and every chunk.
  - Chunk scores are grouped into document scores by **max**: a file is as relevant as its best-matching part. `mean`
    and `sum` are also available.
  - Snippets are read from the source and checked against the indexed content hash. A file that changed since indexing
    is reported **stale**, and a deleted one **missing**, rather than showing lines that may have moved.

### Evaluation

- **`eval/run_apps_cpu.py`:** the encoder wrapped as an MTEB `AbsEncoder`. It runs `mteb.evaluate` on AppsRetrieval
  and writes the submission JSON.
- **`eval/apps_pipeline.py`:** the system's own path end to end (loader → index → retriever), with NDCG@10 and
  MRR@10 computed in `eval/metrics.py`.
  - The metrics follow MTEB's conventions, including tie-breaking by doc id: 11 solution texts appear twice in the
    corpus and tie exactly.
  - Our NDCG is tested against pytrec_eval, and our MRR against MTEB's own implementation.
  - A full run compares itself with the MTEB JSON to 5 decimals.

## Results

AppsRetrieval test split (3,765 queries, 8,765-solution corpus). CPU runs are on a 12-thread CPU.

| Model | Max tokens | Hardware / runtime | NDCG@10 | MRR@10 | Wall time |
|---|---|---|---|---|---|
| sentence-transformers/all-MiniLM-L6-v2 (baseline) | 256 | CPU, PyTorch fp32 | 0.0660 | 0.0558 | 7.3 min |
| Alibaba-NLP/gte-modernbert-base | 2048 | GPU (GTX 1650), PyTorch fp16 | 0.5770 | 0.5293 | 58.1 min |
| Alibaba-NLP/gte-modernbert-base | 1024 | GPU (GTX 1650), PyTorch fp16 | 0.5756 | 0.5281 | 55.2 min |
| **Alibaba-NLP/gte-modernbert-base (submission)** | **1024** | **CPU, onnxruntime fp32** | **0.57545** | **0.52798** | **78.4 min** |
| Same, through the real pipeline (`eval.apps_pipeline`) | 1024 | CPU, onnxruntime fp32 | _pending_ | _pending_ | _pending_ |

- gte-modernbert-base's 0.5770 at 2048 tokens is in line with the 0.5754 NDCG@10 on `apps` reported in its model
  card.
- The CPU submission's other MTEB metrics: NDCG@1 0.44037, Recall@10 0.72669, Recall@100 0.91687.
- **CPU encode time, ONNX vs PyTorch** (timing sample at 1024 tokens, batch size 4, extrapolated to the full encode):
  onnxruntime fp32 ~77.6 min (`bench_onnx_fp32_1024.log`) vs PyTorch fp32 ~97.1 min: onnxruntime is ~20% faster,
  with identical sanity-check similarities (0.725 matched vs 0.454 mismatched, 17/20 top-1).
- **Building the prebuilt apps index** (`python -m index.build_apps`): _pending_.

**P1: retrieval across versions** (the real model, indexing this repository's code):

| Operation | Time |
|---|---|
| Initial `cli.py index` | _pending_ |
| Re-index after editing one file | _pending_ |
| `cli.py query` (end to end, including model load) | _pending_ |

## Repository structure

| Path | Purpose |
|---|---|
| `cli.py` | Command-line tool: `index`, `query`, `eval-apps` |
| `records.py` | `Document` and `Chunk` record types |
| `loaders/apps.py` | AppsRetrieval corpus as Documents (MTEB-identical text), test queries and relevance labels |
| `loaders/directory.py` | A folder of source files as Documents (skip rules, `.gitignore`, stable path IDs) |
| `chunking/` | `none` (whole document) and `windows` (line-aligned, token-budgeted, overlapping) chunkers |
| `embedding/onnx_encoder.py` | Shared CPU encoder: gte-modernbert-base fp32 ONNX via onnxruntime (CLS pooling, max 1024 tokens) |
| `index/` | Index build, incremental update, fingerprint check, atomic save/load; `build_apps.py` builds `indexes/apps` |
| `retrieval/` | Query embedding, cosine scoring, chunk → document grouping, snippets with stale detection |
| `eval/run_apps_cpu.py` | **Submission:** MTEB evaluation of the CPU encoder, writes `appsretrieval_results.json` |
| `eval/apps_pipeline.py`, `eval/metrics.py` | Real-pipeline AppsRetrieval evaluation and MTEB-compatible NDCG / MRR |
| `indexes/apps/` | Prebuilt AppsRetrieval index (committed; folder indexes built with `cli.py index` stay local) |
| `tests/` | Tests: `test_onnx_parity` (loads the model), and `test_loaders_chunking`, `test_index`, `test_retrieval`, `test_eval`, `test_cli`, `test_file_hash` (tokenizer only) |
| `bench_onnx.py` | CPU timing + sanity check for the int8 / fp32 ONNX exports (results in `bench_onnx*.log`) |
| `bench_model.py` | Sanity check + timing estimate for a candidate model with PyTorch; GPU fp16 if available, else CPU fp32 |
| `baseline.py` | MiniLM baseline wrapped as an MTEB `AbsEncoder`, evaluated on AppsRetrieval |
| `baseline_v2.py` | gte-modernbert-base with PyTorch as an MTEB `AbsEncoder` (max 2048 tokens); GPU fp16 if available, else CPU fp32 |
| `explore_apps.py` | Prints dataset structure and example query → code pairs |
| `token_stats.py` | Token-length statistics for queries and corpus |
| `check_models.py` | Read-only metadata check of candidate embedding models |
| `jina_compat.py` | Partial transformers 5 shim for jina-embeddings-v2 (not used; see Notes) |
| `PLAN.md` | Plan: goals, layout, step specs |

## Setup

Requires Python 3.11 (developed on Windows 11). The submission runs on CPU only.

### CPU environment (`venv`, used for the submission)

```powershell
python -m venv venv
venv\Scripts\Activate.ps1          # macOS/Linux: source venv/bin/activate

# CPU-only PyTorch first, so nothing pulls in the CUDA build
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Optional GPU environment (`venv-gpu`, for model experimentation only)

A separate environment with CUDA-enabled PyTorch, used to evaluate large models quickly. Needs an NVIDIA GPU
and driver; developed on a GTX 1650 (4 GB VRAM) with driver 581.83.

```powershell
python -m venv venv-gpu
venv-gpu\Scripts\Activate.ps1

pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

python -c "import torch; print(torch.cuda.is_available())"   # should print True
```

## How to run

With the CPU environment (`venv`) active, from the repository root:

### Command-line tool

```powershell
# Index a folder (default index dir: indexes/<folder name>-<hash8>). Re-running it updates the index
# incrementally: only new and changed files are re-embedded. --rebuild starts from scratch.
python cli.py index D:\path\to\repo
python cli.py index D:\path\to\repo --out indexes\myrepo

# Search an index: ranked files with line ranges and the best-matching snippet
# (flagged stale if the file changed since indexing)
python cli.py query "parse a config file and merge defaults" --index indexes\myrepo
python cli.py query "parse a config file and merge defaults" --index indexes\myrepo --top-k 5 --json

# Interactive search: loads the model once, then answers each query with no startup cost.
# :paste for multi-line queries, :k N, :json, :help, :q. A re-saved index (e.g. `cli.py index` run in another
# terminal after an edit) is picked up automatically, and edited files show as stale until re-indexed.
python cli.py query --index indexes\myrepo --interactive

# Search the prebuilt AppsRetrieval index
python cli.py query "count the ways to climb n stairs taking 1 or 2 steps" --index indexes\apps

# Real-pipeline AppsRetrieval evaluation: NDCG@10 / MRR@10, compared with the MTEB JSON on a full run
python cli.py eval-apps --limit 50      # smoke test: first 50 test queries (needs indexes/apps; else builds it, ~40 min)
python cli.py eval-apps                 # full test set (~40 min of query encoding)
```

### Submission and indexes

```powershell
# P0 submission: MTEB evaluation of the CPU encoder -> appsretrieval_results.json (~80 min)
python -m eval.run_apps_cpu

# Build or incrementally update the prebuilt apps index in indexes/apps (~40 min from scratch)
python -m index.build_apps
```

### Docker

The image is CPU-only. The model, tokenizer and AppsRetrieval dataset are downloaded at build time, so
containers run offline. The entrypoint is `cli.py`.

```powershell
docker build -t prism-retrieval .

# Index a folder mounted at /data/repo, keeping indexes in a named volume so they persist between runs
docker run --rm -v D:\path\to\repo:/data/repo:ro -v prism-indexes:/app/indexes prism-retrieval index /data/repo --out indexes/myrepo

# Interactive search (-it for the prompt). Mount the folder too, so snippets can be read and checked for staleness.
docker run --rm -it -v D:\path\to\repo:/data/repo:ro -v prism-indexes:/app/indexes prism-retrieval query --index indexes/myrepo --interactive

# One-shot search of the prebuilt apps index shipped in the image
docker run --rm prism-retrieval query "count the ways to climb n stairs taking 1 or 2 steps" --index indexes/apps

# Real-pipeline evaluation, writing results to a mounted folder
docker run --rm -v ${PWD}/out:/out prism-retrieval eval-apps --limit 50 --output /out/apps_pipeline_results.json

# P0 submission: MTEB evaluation (~80 min), results JSON written to the mounted folder
docker run --rm -v ${PWD}/out:/out --entrypoint python prism-retrieval -m eval.run_apps_cpu /out/appsretrieval_results.json
```

The ONNX encoder peaks at about 3 GB of RAM during long encodes. Give Docker Desktop at least 4 GB.

### Tests

```powershell
python -m tests.test_loaders_chunking
python -m tests.test_index
python -m tests.test_retrieval
python -m tests.test_eval
python -m tests.test_cli
python -m tests.test_file_hash
python -m tests.test_onnx_parity        # loads the ONNX and PyTorch models (~3 GB RAM)
```

### Exploration and model selection

```powershell
python explore_apps.py                                        # dataset structure and examples
python baseline.py                                            # MiniLM baseline (~7 min on CPU)
python bench_model.py Alibaba-NLP/gte-modernbert-base 1024 4  # PyTorch timing estimate: <model> [max_len] [batch]
python bench_onnx.py onnx/model.onnx 1024 4                    # ONNX timing estimate: [onnx_file] [max_len] [batch]
```

With the GPU environment (`venv-gpu`) active:

```powershell
# Timing estimate on the GPU (batch size 8 was fastest on a 4 GB GTX 1650)
python bench_model.py Alibaba-NLP/gte-modernbert-base 2048 8

# gte-modernbert-base full evaluation with PyTorch: [max_seq_length], default 2048
# (~58 min at 2048 / ~55 min at 1024 on a GTX 1650; many hours with PyTorch on CPU)
python baseline_v2.py
python baseline_v2.py 1024
```

Models and datasets download from HuggingFace on first run and are cached in `~/.cache/huggingface`.

## Notes

- `jinaai/jina-embeddings-v2-base-code` was evaluated but not used: its custom model code is incompatible with
  transformers 5.x.
- gte-modernbert-base's checkpoint stores fp16 weights, and transformers 5 loads the stored dtype by default.
  PyTorch CPU code here loads with `dtype=torch.float32` explicitly and asserts it.
- Committed index files keep their generation-numbered names. Git stores content, not names, so each committed
  rebuild with new embeddings adds ~27 MB to history: commit an index rebuild only when the corpus or encoder changes.

## Team

<!-- TODO: team members -->

## License

MIT. See [LICENSE](LICENSE).
