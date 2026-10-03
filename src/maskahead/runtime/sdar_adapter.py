"""Patch SDAR attention to route its block KV reads through BitSieve."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import torch

from ..cache import PackedKVCache
from .session import BitSieveSession


def _forward(self, hidden_states, position_embeddings, attention_mask=None,
             past_key_value=None, cache_position=None, **kwargs):
    session = getattr(self, "_bitsieve_session", None)
    original = self._bitsieve_original_forward
    if session is None or not isinstance(past_key_value, PackedKVCache):
        return original(hidden_states=hidden_states, position_embeddings=position_embeddings,
                        attention_mask=attention_mask, past_key_value=past_key_value,
                        cache_position=cache_position, **kwargs)
    store_kv = bool(kwargs.get("store_kv", False))
    if store_kv != session.commit_current:
        raise RuntimeError("SDAR store_kv and BitSieve commit state disagree")
    input_shape = hidden_states.shape[:-1]
    shape = (*input_shape, -1, self.head_dim)
    q = self.q_norm(self.q_proj(hidden_states).view(shape)).transpose(1, 2)
    k = self.k_norm(self.k_proj(hidden_states).view(shape)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(shape).transpose(1, 2)
    rope = sys.modules[type(self).__module__].apply_rotary_pos_emb
    q, k = rope(q, k, *position_embeddings)
    session.set_rope(*position_embeddings)
    out = session.attend(self.layer_idx, q, k, v)
    session.stage_if_committing(self.layer_idx, k, v)
    out = out.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    return self.o_proj(out), None


@dataclass
class SDARPatchHandle:
    model: torch.nn.Module
    modules: list[torch.nn.Module]

    def set_session(self, session: BitSieveSession | None) -> None:
        for module in self.modules:
            module._bitsieve_session = session

    def unpatch(self) -> None:
        for module in self.modules:
            module.forward = module._bitsieve_original_forward
            del module._bitsieve_original_forward
            del module._bitsieve_session


def patch_sdar(model, session=None) -> SDARPatchHandle:
    modules = [m for m in model.modules() if type(m).__name__ == "SDARAttention"]
    if not modules:
        raise RuntimeError("no SDARAttention modules found")
    for module in modules:
        module._bitsieve_original_forward = module.forward
        module.forward = types.MethodType(_forward, module)
        module._bitsieve_session = session
    return SDARPatchHandle(model, modules)
