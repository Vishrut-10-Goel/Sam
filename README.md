# Code Retrieval — PRISM GenAI Hackathon 2026

> Release tag: **`PRISM_GENAI_HACKATHON_Y2026`**

Text-to-code retrieval: given a natural-language programming problem, retrieve the code solution that solves it.
Evaluated on the MTEB **AppsRetrieval** task (CoIR benchmark, `CoIR-Retrieval/apps`).

<!-- TODO: one-paragraph summary of the final approach and headline result. -->

## Problem

- **Queries:** full competitive-programming problem statements (story, input/output spec, examples). Mean ≈ 500 tokens, longest ≈ 1,700.
- **Corpus:** 8,765 Python solutions; each of the 3,765 test queries has exactly one relevant solution.
- **Metrics:** NDCG@10 (MTEB main score) and MRR@10.

## Approach

<!-- TODO: describe the final model / pipeline. -->

- Embedding models wrapped as MTEB `AbsEncoder` subclasses and evaluated with `mteb.evaluate`.
- CPU-only: no GPU required.

## Results

AppsRetrieval test split (3,765 queries, 8,765-solution corpus).

| Model | Max tokens | Hardware | NDCG@10 | MRR@10 | Wall time |
|---|---|---|---|---|---|
| sentence-transformers/all-MiniLM-L6-v2 (baseline) | 256 | CPU, fp32 | 0.0660 | 0.0558 | 7.3 min |
| Alibaba-NLP/gte-modernbert-base | 2048 | GPU (GTX 1650), fp16 | 0.5770 | 0.5293 | 58.1 min |
| Alibaba-NLP/gte-modernbert-base | 1024 | GPU (GTX 1650), fp16 | 0.5756 | 0.5281 | 55.2 min |

gte-modernbert-base's 0.5770 matches the 0.5754 NDCG@10 on `apps` reported in its model card.
Capping at 1024 tokens truncates 82 test queries (2.2%) and costs only 0.0014 NDCG@10, while cutting
estimated CPU encode time substantially (see `bench_onnx*.log`).

## Repository structure

| Path | Purpose |
|---|---|
| `embedding/onnx_encoder.py` | Shared CPU encoder: gte-modernbert-base fp32 ONNX via onnxruntime (CLS pooling, max 1024 tokens) |
| `tests/test_onnx_parity.py` | Parity test: ONNX encoder vs sentence-transformers PyTorch fp32 (`python -m tests.test_onnx_parity`) |
| `bench_onnx.py` | CPU timing + sanity check for the int8 / fp32 ONNX exports (results in `bench_onnx*.log`) |
| `baseline.py` | MiniLM baseline wrapped as an MTEB `AbsEncoder`, evaluated on AppsRetrieval |
| `baseline_v2.py` | gte-modernbert-base wrapped as an MTEB `AbsEncoder` (max 2048 tokens), evaluated on AppsRetrieval; uses the GPU in fp16 if available, else CPU |
| `explore_apps.py` | Prints dataset structure and example query → code pairs |
| `token_stats.py` | Token-length statistics for queries and corpus |
| `bench_model.py` | Sanity check + timing estimate for a candidate model; uses the GPU in fp16 if available, else CPU |
| `check_models.py` | Read-only metadata check of candidate embedding models |
| `jina_compat.py` | Partial transformers 5 shim for jina-embeddings-v2 (not used; see Notes) |

<!-- TODO: add the final (CPU) submission script once it exists. -->

## Setup

Requires Python 3.11 (developed on Windows 11). The submission runs on CPU only.

### CPU environment (`venv`, used for the submission)

```powershell
python -m venv venv
venv\Scripts\Activate.ps1          # macOS/Linux: source venv/bin/activate

# CPU-only PyTorch first, so nothing pulls in the CUDA build
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install mteb sentence-transformers rank_bm25 datasets onnxruntime pathspec
```

### Optional GPU environment (`venv-gpu`, for model experimentation only)

A separate environment with CUDA-enabled PyTorch, used to evaluate large models quickly. Needs an NVIDIA GPU
and driver; developed on a GTX 1650 (4 GB VRAM) with driver 581.83.

```powershell
python -m venv venv-gpu
venv-gpu\Scripts\Activate.ps1

pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
pip install mteb sentence-transformers rank_bm25 datasets

python -c "import torch; print(torch.cuda.is_available())"   # should print True
```

<!-- TODO: replace with `pip install -r requirements.txt` once versions are pinned. -->

## How to run

With the CPU environment (`venv`) active:

```powershell
# Explore the dataset
python explore_apps.py

# Baseline evaluation (~7 min on CPU)
python baseline.py

# Sanity-check + timing estimate for a candidate model: <model> [max_seq_length] [batch_size]
python bench_model.py Alibaba-NLP/gte-modernbert-base 2048
```

With the GPU environment (`venv-gpu`) active:

```powershell
# Timing estimate on the GPU (batch size 8 was fastest on a 4 GB GTX 1650)
python bench_model.py Alibaba-NLP/gte-modernbert-base 2048 8

# gte-modernbert-base full evaluation: [max_seq_length], default 2048
# (~58 min at 2048 / ~55 min at 1024 on a GTX 1650; many hours with PyTorch on CPU)
python baseline_v2.py
python baseline_v2.py 1024
```

<!-- TODO: command for the final (CPU) submission's full evaluation. -->

Models and datasets download from HuggingFace on first run and are cached in `~/.cache/huggingface`.

## Notes

- `jinaai/jina-embeddings-v2-base-code` was evaluated but not used: its custom model code is incompatible with transformers 5.x.

## Team

<!-- TODO: team members -->

## License

MIT. See [LICENSE](LICENSE).
