"""Compatibility shim so jina-embeddings-v2 remote code loads under transformers 5.x.

Jina's modeling_bert.py imports `find_pruneable_heads_and_indices`, which transformers 5 removed.
It is only used by `prune_heads()` (a training-time utility), never during inference, so a stub that
fails loudly if called is safe. Import this module before loading the model.

transformers 5 also dropped the `is_decoder` / `add_cross_attention` defaults from PretrainedConfig
(and ignores them as load-time kwargs), so restore the transformers 4 defaults (plain encoder) as
class attributes. Instances that set these explicitly still override them.
"""
import transformers.pytorch_utils as pytorch_utils
from transformers import PretrainedConfig

for _name in ("is_decoder", "add_cross_attention"):
    if not hasattr(PretrainedConfig, _name):
        setattr(PretrainedConfig, _name, False)

if not hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):

    def find_pruneable_heads_and_indices(*args, **kwargs):
        raise NotImplementedError("Head pruning is not supported under the transformers 5 compatibility shim.")

    pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
