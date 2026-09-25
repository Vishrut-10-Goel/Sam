"""CPU submission run: the shared ONNX encoder wrapped as an MTEB AbsEncoder, evaluated on AppsRetrieval.

Uses embedding/onnx_encoder.py (gte-modernbert-base, fp32 ONNX, onnxruntime, CPU only) at max length 1024
and writes the TaskResult JSON that MTEB produces.

Run from the project root:  python -m eval.run_apps_cpu [output_json]   (default appsretrieval_results.json)
Expect ~80 min on a 12-thread CPU (see experiments/logs/bench_onnx_fp32_1024.log).
"""
import functools
import sys
import time
from pathlib import Path

import mteb
import numpy as np
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta

from embedding.onnx_encoder import DEFAULT_MODEL, OnnxEncoder

print = functools.partial(print, flush=True)  # show progress live when output goes to a log file

TASK_NAME = "AppsRetrieval"
MAX_LENGTH = 1024
OUTPUT_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("appsretrieval_results.json")
LOG_EVERY_SECONDS = 60


class OnnxMtebEncoder(AbsEncoder):
    def __init__(self, max_length: int = MAX_LENGTH):
        self.encoder = OnnxEncoder(max_length=max_length)
        self.mteb_model_meta = ModelMeta.create_empty(
            overwrites=dict(name=DEFAULT_MODEL, revision=None, loader=type(self))
        )

    def encode(self, inputs, *, task_metadata, hf_split, hf_subset, prompt_type=None, **kwargs):
        # Encode everything in one call so OnnxEncoder sorts all texts by length (minimal padding).
        texts = [text for batch in inputs for text in batch["text"]]
        label = prompt_type.value if prompt_type is not None else "texts"
        print(f"  encoding {len(texts):,} {label} ...")
        t0 = time.perf_counter()
        last_log = t0

        def progress(done: int, total: int) -> None:
            # Texts are processed longest first, so the ETA from the rate so far is pessimistic.
            nonlocal last_log
            now = time.perf_counter()
            if now - last_log >= LOG_EVERY_SECONDS or done == total:
                last_log = now
                elapsed = now - t0
                eta = elapsed / done * (total - done)
                print(f"    {done:,}/{total:,} {label} ({done / total:.0%})  "
                      f"elapsed {elapsed / 60:.1f} min  eta <= {eta / 60:.1f} min")

        embeddings = self.encoder.encode(texts, progress=progress)
        print(f"  done: {len(texts):,} {label} in {(time.perf_counter() - t0) / 60:.1f} min")
        return np.asarray(embeddings, dtype=np.float32)


if __name__ == "__main__":
    start = time.perf_counter()
    model = OnnxMtebEncoder()
    fp = model.encoder.fingerprint()
    print(f"encoder: {fp['model_name']} [{fp['onnx_file']}]   max_length: {fp['max_length']}   "
          f"batch size: {model.encoder.batch_size}   onnx sha256: {fp['onnx_sha256'][:16]}...")
    print(f"output: {OUTPUT_PATH.resolve()}")

    task = mteb.get_task(TASK_NAME)
    # cache=None: always recompute rather than reuse stored results from an earlier run.
    result = mteb.evaluate(model, task, cache=None, show_progress_bar=False)
    elapsed = time.perf_counter() - start

    task_result = result.task_results[0]
    # Write to a temp file first so an interrupted write never leaves a truncated results file.
    tmp_path = OUTPUT_PATH.with_suffix(".json.tmp")
    task_result.to_disk(tmp_path)
    tmp_path.replace(OUTPUT_PATH)

    scores = task_result.scores["test"][0]
    print("\n" + "=" * 50)
    print(f"Model: {DEFAULT_MODEL} (fp32 ONNX, CPU)   max_length: {MAX_LENGTH}")
    print(f"Task:  {TASK_NAME} (test)")
    print(f"NDCG@10: {scores['ndcg_at_10']:.4f}   (GPU fp16 at 1024: 0.5756)")
    print(f"MRR@10:  {scores['mrr_at_10']:.4f}   (GPU fp16 at 1024: 0.5281)")
    print(f"Wall time: {elapsed / 60:.1f} min ({elapsed:.0f} s)")
    print(f"Results written to {OUTPUT_PATH}")
    print("=" * 50)
