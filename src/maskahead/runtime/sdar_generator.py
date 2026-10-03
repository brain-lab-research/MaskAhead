"""SDAR block decoding with the existing BitSieve session lifecycle."""

from __future__ import annotations

import torch
from torch.nn import functional as F
from transformers.cache_utils import Cache

from ..cache import PackedKVCache
from .dream_generator import DreamBitSieveGenerator
from .sdar_adapter import patch_sdar


class _SDARPackedKVCache(PackedKVCache, Cache):
    """Satisfy SDAR's explicit transformers.Cache type check."""


class _SDARGenerationUtils:
    @staticmethod
    def get_num_transfer_tokens(block, steps):
        return torch.tensor([block // steps + (i < block % steps) for i in range(steps)])

    @staticmethod
    def sample_with_temperature_topk_topp(logits, temperature=0.0, top_k=0, top_p=1.0):
        if temperature != 0 or top_k or top_p != 1.0:
            raise ValueError("SDAR comparison requires greedy decoding")
        probs = F.softmax(logits, dim=-1)
        tokens = probs.argmax(dim=-1)
        return tokens, probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def _select_transfer_index(strategy, mask, tokens, confidence, transfer, step,
                               threshold, eb_threshold, force_accept=False):
        if strategy != "low_confidence_dynamic":
            raise ValueError(strategy)
        if force_accept:
            return mask.clone()
        scores = confidence.masked_fill(~mask, -torch.inf)
        minimum = int(transfer[step])
        selected = (scores > threshold) & mask
        for row in range(scores.shape[0]):
            if int(selected[row].sum()) < minimum:
                k = min(minimum, int(mask[row].sum()))
                selected[row] = False
                if k:
                    selected[row, scores[row].topk(k).indices] = True
        return selected

    @staticmethod
    def _resolve_stopping_ids(ids):
        return [ids] if isinstance(ids, int) else list(ids or [])

    @staticmethod
    def _should_stop(x, prompt_len, stop_ids):
        return any(bool((x[:, prompt_len:] == sid).any()) for sid in stop_ids)


class SDARBitSieveGenerator(DreamBitSieveGenerator):
    def _new_cache(self, batch_size):
        return _SDARPackedKVCache(
            num_layers=self.num_layers, batch_size=batch_size,
            num_kv_heads=self.num_kv_heads, head_dim=self.head_dim,
            max_tokens=self.config.max_cache_tokens, quant=self.config.quant,
            device=self.device, compute_dtype=self.compute_dtype,
            backend=self.config.backend,
        )

    def __init__(self, model, tokenizer, config):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.config = config
        cfg = model.config
        self.num_layers = int(cfg.num_hidden_layers)
        self.num_q_heads = int(cfg.num_attention_heads)
        self.num_kv_heads = int(cfg.num_key_value_heads)
        self.head_dim = int(cfg.head_dim)
        self.device = next(model.parameters()).device
        self.compute_dtype = next(model.parameters()).dtype
        self.config.validate(head_dim=self.head_dim, num_layers=self.num_layers)
        self.gu = _SDARGenerationUtils
        self.mask_token_id = int(tokenizer.convert_tokens_to_ids("<|MASK|>"))
        if self.mask_token_id == tokenizer.unk_token_id:
            raise ValueError("SDAR mask token absent")
        self.stop_token_id = [151645, 151643]
        self.patch = patch_sdar(model, None)
