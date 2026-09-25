"""Baseline: all-MiniLM-L6-v2 wrapped as an MTEB AbsEncoder, evaluated on AppsRetrieval (CPU)."""
import time

import mteb
import numpy as np
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TASK_NAME = "AppsRetrieval"


class MiniLMEncoder(AbsEncoder):
    def __init__(self, model_name: str = MODEL_NAME, device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)
        self.mteb_model_meta = ModelMeta.create_empty(
            overwrites=dict(name=model_name, revision=None, loader=type(self))
        )

    def encode(self, inputs, *, task_metadata, hf_split, hf_subset, prompt_type=None, **kwargs):
        # Queries and code documents are encoded identically; MiniLM has no prompts.
        embeddings = [
            self.model.encode(batch["text"], convert_to_numpy=True, normalize_embeddings=True)
            for batch in inputs
        ]
        return np.concatenate(embeddings, axis=0)


if __name__ == "__main__":
    start = time.perf_counter()
    model = MiniLMEncoder()
    task = mteb.get_task(TASK_NAME)
    # cache=None: always recompute rather than reuse stored results from an earlier run.
    result = mteb.evaluate(model, task, cache=None, encode_kwargs={"batch_size": 64})
    elapsed = time.perf_counter() - start

    scores = result.task_results[0].scores["test"][0]
    print("\n" + "=" * 50)
    print(f"Model: {MODEL_NAME}")
    print(f"Task:  {TASK_NAME} (test)")
    print(f"NDCG@10: {scores['ndcg_at_10']:.4f}")
    print(f"MRR@10:  {scores['mrr_at_10']:.4f}")
    print(f"Wall time: {elapsed / 60:.1f} min ({elapsed:.0f} s)")
    print("=" * 50)
