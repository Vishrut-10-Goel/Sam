"""Read-only check of candidate embedding models: native transformers support, context length, size.

Downloads only small JSON metadata files (config.json, sentence_bert_config.json), never weights.
"""
import json

import transformers
from huggingface_hub import HfApi, hf_hub_download

CANDIDATES = [
    "ibm-granite/granite-embedding-small-english-r2",
    "ibm-granite/granite-embedding-english-r2",
    "Alibaba-NLP/gte-modernbert-base",
    "nomic-ai/modernbert-embed-base",
    "Qwen/Qwen3-Embedding-0.6B",
    "BAAI/bge-base-en-v1.5",
    # Known code models, included to confirm whether they need remote code:
    "nomic-ai/CodeRankEmbed",
    "Salesforce/SFR-Embedding-Code-400M_R",
]

api = HfApi()


def load_json(repo, name):
    try:
        with open(hf_hub_download(repo, name), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


for repo in CANDIDATES:
    try:
        info = api.model_info(repo, expand=["safetensors", "cardData", "tags"])
    except Exception as e:
        print(f"\n{repo}: NOT FOUND ({type(e).__name__})")
        continue
    cfg = load_json(repo, "config.json") or {}
    st_cfg = load_json(repo, "sentence_bert_config.json") or {}
    archs = cfg.get("architectures") or []
    native = all(hasattr(transformers, a) for a in archs) if archs else None
    params = info.safetensors.total if info.safetensors else None
    print(f"\n{repo}")
    print(f"  model_type={cfg.get('model_type')}  architectures={archs}")
    print(f"  native in transformers {transformers.__version__}: {native}   auto_map (remote code): {bool(cfg.get('auto_map'))}")
    print(f"  params: {params / 1e6:.0f}M" if params else "  params: unknown")
    print(f"  max_position_embeddings={cfg.get('max_position_embeddings')}  ST max_seq_length={st_cfg.get('max_seq_length')}")
    print(f"  hidden={cfg.get('hidden_size')} layers={cfg.get('num_hidden_layers')}  license={(info.card_data or {}).get('license')}")
    code_tags = [t for t in (info.tags or []) if "code" in t.lower()]
    print(f"  code-related tags: {code_tags}")
