"""Token-length statistics for AppsRetrieval under the jina-embeddings-v2-base-code tokenizer."""
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "jinaai/jina-embeddings-v2-base-code"
DATASET = "CoIR-Retrieval/apps"
THRESHOLDS = [256, 512, 1024, 2048, 4096, 8192]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

test_qids = set(load_dataset(DATASET, "default", split="test")["query-id"])
queries = [q["text"] for q in load_dataset(DATASET, "queries", split="queries") if q["_id"] in test_qids]
corpus = [d["text"] for d in load_dataset(DATASET, "corpus", split="corpus")]


def token_lengths(texts):
    return np.array([len(ids) for ids in tokenizer(texts, add_special_tokens=True)["input_ids"]])


for label, texts in [("Test queries", queries), ("Corpus (code)", corpus)]:
    lengths = token_lengths(texts)
    print(f"\n=== {label}: {len(lengths):,} texts ===")
    print(f"longest: {lengths.max():,} tokens   mean: {lengths.mean():,.0f}   total: {lengths.sum():,}")
    pct = np.percentile(lengths, [50, 75, 90, 95, 99])
    print("percentiles:  " + "  ".join(f"p{p}={v:,.0f}" for p, v in zip([50, 75, 90, 95, 99], pct)))
    for t in THRESHOLDS:
        n = int((lengths > t).sum())
        print(f"  > {t:>5} tokens: {n:>5,} ({n / len(lengths):6.2%})")
    np.save(f"lengths_{'queries' if label.startswith('Test') else 'corpus'}.npy", lengths)
