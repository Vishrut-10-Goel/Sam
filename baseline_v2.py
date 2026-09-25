"""gte-modernbert-base wrapped as an MTEB AbsEncoder, evaluated on AppsRetrieval.

Runs on the GPU (fp16) when the environment's PyTorch can see one (venv-gpu), otherwise on CPU (venv).

Usage: python baseline_v2.py [max_seq_length]   (default 2048)
"""
import functools
import sys
import time

import mteb
import numpy as np
import torch
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta
from sentence_transformers import SentenceTransformer

print = functools.partial(print, flush=True)  # show progress live, even when output is piped

MODEL_NAME = "Alibaba-NLP/gte-modernbert-base"
TASK_NAME = "AppsRetrieval"
# 2048: longest test query is 1,620 tokens, so no query is truncated. 1024 truncates 82 queries (2.2%).
MAX_SEQ_LENGTH = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8  # fastest measured on the 4 GB GTX 1650 (batch 32 was ~40% slower)


class GteModernBertEncoder(AbsEncoder):
    def __init__(self, model_name: str = MODEL_NAME, device: str = DEVICE):
        # Load in fp32 explicitly: transformers 5 otherwise loads the checkpoint's stored dtype (fp16 for
        # gte-modernbert-base), which would silently run the CPU path in fp16.
        self.model = SentenceTransformer(model_name, device=device, model_kwargs={"dtype": torch.float32})
        if device == "cuda":
            self.model.half()  # Turing has fast fp16 but no bf16
        dtype = next(self.model.parameters()).dtype
        expected = torch.float16 if device == "cuda" else torch.float32
        assert dtype == expected, f"model loaded as {dtype}, expected {expected}"
        self.model.max_seq_length = MAX_SEQ_LENGTH
        self.mteb_model_meta = ModelMeta.create_empty(
            overwrites=dict(name=model_name, revision=None, loader=type(self))
        )

    def encode(self, inputs, *, task_metadata, hf_split, hf_subset, prompt_type=None, **kwargs):
        # Encode everything in one call so sentence-transformers sorts all texts by length,
        # minimising padding (attention cost grows with the padded length).
        texts = [text for batch in inputs for text in batch["text"]]
        label = prompt_type.value if prompt_type is not None else "texts"
        print(f"  encoding {len(texts):,} {label} ...")
        t0 = time.perf_counter()
        embeddings = self.model.encode(
            texts, batch_size=BATCH_SIZE, convert_to_numpy=True, normalize_embeddings=True
        )
        print(f"  done: {len(texts):,} {label} in {(time.perf_counter() - t0) / 60:.1f} min")
        return np.asarray(embeddings, dtype=np.float32)


if __name__ == "__main__":
    start = time.perf_counter()
    print(f"model: {MODEL_NAME}   device: {DEVICE}   batch size: {BATCH_SIZE}   max_seq_length: {MAX_SEQ_LENGTH}")
    model = GteModernBertEncoder()
    task = mteb.get_task(TASK_NAME)
    # cache=None: always recompute rather than reuse stored results from an earlier run.
    result = mteb.evaluate(model, task, cache=None, show_progress_bar=False)
    elapsed = time.perf_counter() - start

    scores = result.task_results[0].scores["test"][0]
    print("\n" + "=" * 50)
    print(f"Model: {MODEL_NAME}   max_seq_length: {MAX_SEQ_LENGTH}")
    print(f"Task:  {TASK_NAME} (test)")
    print(f"NDCG@10: {scores['ndcg_at_10']:.4f}   (MiniLM baseline 0.0660)")
    print(f"MRR@10:  {scores['mrr_at_10']:.4f}   (MiniLM baseline 0.0558)")
    print(f"Wall time: {elapsed / 60:.1f} min ({elapsed:.0f} s)")
    print("=" * 50)
