"""Explore the CoIR-Retrieval/apps dataset: structure plus two query -> code examples."""
from datasets import load_dataset

NAME = "CoIR-Retrieval/apps"
N_EXAMPLES = 2


def describe(label, ds):
    print(f"{label}: {ds.num_rows:,} rows | columns: {ds.column_names}")


qrels = load_dataset(NAME, "default", split="test")
queries = load_dataset(NAME, "queries", split="queries")
corpus = load_dataset(NAME, "corpus", split="corpus")

print("=" * 80)
print("STRUCTURE")
print("=" * 80)
describe("qrels (test)", qrels)
describe("queries", queries)
describe("corpus", corpus)
print(f"\nunique test queries: {len(set(qrels['query-id'])):,}")
print(f"score values in test qrels: {sorted(set(qrels['score']))}")

query_by_id = {row["_id"]: row for row in queries}
doc_by_id = {row["_id"]: row for row in corpus}

for i, rel in enumerate(qrels.select(range(N_EXAMPLES)), 1):
    q = query_by_id[rel["query-id"]]
    d = doc_by_id[rel["corpus-id"]]
    print("\n" + "=" * 80)
    print(f"EXAMPLE {i}   query-id={rel['query-id']}  corpus-id={rel['corpus-id']}  score={rel['score']}")
    print("=" * 80)
    print("--- QUERY ---")
    print(q["text"])
    print("\n--- RELEVANT CODE ---")
    if d.get("title"):
        print(f"[title] {d['title']}")
    print(d["text"])
