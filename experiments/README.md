# Experiments

Model selection and benchmarking scripts used to choose the encoder. They are not part of the retrieval system
(`cli.py`, `loaders/`, `chunking/`, `embedding/`, `index/`, `retrieval/`, `eval/`), and nothing imports them.

Run them from the repository root, e.g. `python experiments/bench_onnx.py onnx/model.onnx 1024 4`.
`logs/` holds the benchmark outputs the main README cites.

| Script | Purpose |
|---|---|
| `explore_apps.py` | Prints the AppsRetrieval dataset structure and example query → code pairs |
| `token_stats.py` | Token-length statistics for queries and corpus (writes `lengths_*.npy`) |
| `check_models.py` | Read-only metadata check of candidate embedding models |
| `baseline.py` | all-MiniLM-L6-v2 baseline as an MTEB `AbsEncoder` on AppsRetrieval |
| `baseline_v2.py` | gte-modernbert-base with PyTorch as an MTEB `AbsEncoder`; GPU fp16 if available, else CPU fp32 |
| `bench_model.py` | Sanity check + timing estimate for a candidate model with PyTorch |
| `bench_onnx.py` | Sanity check + timing estimate for the int8 / fp32 ONNX exports |
| `bench_jina.py`, `jina_compat.py` | jina-embeddings-v2-base-code attempt (not used: incompatible with transformers 5.x) |
