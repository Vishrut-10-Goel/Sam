"""Parity test: OnnxEncoder (fp32 ONNX, onnxruntime) vs the sentence-transformers PyTorch model.

Both run on CPU in fp32 with the same max_length, so embeddings should agree to within numerical noise.

Run from the project root:  python -m tests.test_onnx_parity
(Also collectable by pytest if it is installed.)
"""
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from embedding.onnx_encoder import DEFAULT_MAX_LENGTH, DEFAULT_MODEL, OnnxEncoder

MIN_COSINE = 0.999

CODE_SNIPPET = '''def two_sum(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        if target - x in seen:
            return [seen[target - x], i]
        seen[x] = i
    return []
'''

PROBLEM_STATEMENT = """Anton has the integer x. He is interested what positive integer, which doesn't exceed x,
has the maximum sum of digits. Your task is to help Anton and to find the integer that interests him.
If there are several such integers, determine the biggest of them.

-----Input-----
The first line contains the positive integer x (1 <= x <= 10^18) - the integer which Anton has.

-----Output-----
Print the positive integer which doesn't exceed x and has the maximum sum of digits."""

TEXTS = {
    "short query": "find the index of two numbers that add up to a target",
    "code snippet": CODE_SNIPPET,
    "problem statement": PROBLEM_STATEMENT,
    "unicode + symbols": "Déjà vu: ≤ ≥ → λx. x² — 数组求和 🚀",
    "single word": "binary",
    # Well over 1024 tokens, so this checks truncation matches too.
    "long (truncated)": "\n".join(f"def f{i}(a, b):\n    return a * {i} + b - {i}" for i in range(400)),
}

_models = {}


def _load():
    if not _models:
        onnx = OnnxEncoder()
        # The checkpoint is stored in fp16 and transformers 5 loads the stored dtype by default,
        # so force fp32: the reference must use the same precision as the fp32 ONNX export.
        torch_model = SentenceTransformer(DEFAULT_MODEL, device="cpu", model_kwargs={"dtype": torch.float32})
        torch_model.max_seq_length = DEFAULT_MAX_LENGTH
        _models.update(onnx=onnx, torch=torch_model)
    return _models["onnx"], _models["torch"]


def test_long_text_is_actually_truncated():
    onnx, _ = _load()
    full = onnx.token_lengths([TEXTS["long (truncated)"]], truncate=False)[0]
    assert full > DEFAULT_MAX_LENGTH, f"long text is only {full} tokens; test would not exercise truncation"


def _cosines(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def test_reference_model_is_fp32():
    _, torch_model = _load()
    dtype = next(torch_model.parameters()).dtype
    assert dtype == torch.float32, f"PyTorch reference loaded as {dtype}; parity would compare against the wrong precision"


def test_parity_with_sentence_transformers():
    onnx, torch_model = _load()
    names, texts = list(TEXTS), list(TEXTS.values())
    a = onnx.encode(texts)
    b = torch_model.encode(texts, batch_size=4, convert_to_numpy=True)
    cosines = _cosines(a, b)  # explicit, in float64, so neither side's normalization can skew it
    for name, cos in zip(names, cosines):
        print(f"  {name:20s} cosine {cos:.8f}  (1 - cos = {1 - cos:.1e})")
    assert cosines.min() > MIN_COSINE, f"min cosine {cosines.min():.6f} <= {MIN_COSINE}"


def test_batching_does_not_change_embeddings():
    # Padding differs between batch sizes; the attention mask must make it irrelevant.
    onnx, _ = _load()
    texts = list(TEXTS.values())
    one_by_one = onnx.encode(texts, batch_size=1)
    batched = onnx.encode(texts, batch_size=4)
    cosines = _cosines(one_by_one, batched)
    assert cosines.min() > 0.99999, f"batching changed embeddings: min cosine {cosines.min():.7f}"


def test_output_shape_order_and_norm():
    onnx, _ = _load()
    texts = list(TEXTS.values())
    emb = onnx.encode(texts)
    assert emb.shape == (len(texts), onnx.dim) and emb.dtype == np.float32
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)
    # Input order is preserved even though encode() sorts by length internally.
    reversed_emb = onnx.encode(texts[::-1])
    assert np.allclose(emb, reversed_emb[::-1], atol=1e-5)
    assert onnx.encode([]).shape == (0, onnx.dim)


def test_fingerprint_is_stable():
    onnx, _ = _load()
    fp = onnx.fingerprint()
    assert fp == onnx.fingerprint()
    assert fp["max_length"] == DEFAULT_MAX_LENGTH and fp["pooling"] == "cls" and len(fp["onnx_sha256"]) == 64


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
