"""Route DreamReasoner's attention through a BitSieve session.

Dream's decoding loop has the same shape as Fast-dLLM-v2's -- prefill the
block-aligned prompt, denoise a block against the cache, then one forward with
``store_kv=True`` that commits the finished block -- so the session's contract
carries over unchanged. Three things differ and the patch has to honour them:

* ``q_norm`` / ``k_norm``: Dream normalizes each head of Q and K *before* RoPE.
  Skipping them would leave the selector ranking on vectors the model never
  uses, and the mismatch would look like a weak selector rather than a bug.
* ``store_kv`` is the commit flag, where Fast-dLLM has ``update_past_key_values``.
* the forward returns ``(output, weights)``, not a bare tensor.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any

import torch

from ..cache import PackedKVCache
from .session import BitSieveSession


def _rope_fn(module: torch.nn.Module):
    return sys.modules[type(module).__module__].apply_rotary_pos_emb


def is_dream_attention(module: torch.nn.Module) -> bool:
    """Dream's attention block, identified by what the patch actually needs."""
    return (
        type(module).__name__ == "DreamAttention"
        and all(hasattr(module, a) for a in ("q_proj", "k_proj", "v_proj", "o_proj"))
        and all(hasattr(module, a) for a in ("q_norm", "k_norm"))
        and hasattr(module, "layer_idx")
    )


def _sparse_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    past_key_values=None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Any,
):
    session: BitSieveSession | None = getattr(self, "_bitsieve_session", None)
    original = getattr(self, "_bitsieve_original_forward")
    if session is None or not isinstance(past_key_values, PackedKVCache):
        return original(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )

    # The generator sets commit_current; store_kv says the same thing from the
    # model's side. If they ever disagree the block boundary has slipped, and a
    # silently mis-staged block would corrupt the cache for every later block.
    store_kv = bool(kwargs.get("store_kv", False))
    if store_kv != session.commit_current:
        raise RuntimeError(
            f"store_kv={store_kv} but the session has commit_current="
            f"{session.commit_current}; the block lifecycle is out of step"
        )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    q, k = _rope_fn(self)(q, k, cos, sin)

    session.set_rope(cos, sin)
    out = session.attend(self.layer_idx, q, k, v)
    session.stage_if_committing(self.layer_idx, k, v)
    out = out.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    return self.o_proj(out), None


@dataclass(slots=True)
class DreamPatchHandle:
    model: torch.nn.Module
    modules: list[torch.nn.Module]

    def set_session(self, session: BitSieveSession | None) -> None:
        for module in self.modules:
            setattr(module, "_bitsieve_session", session)

    def unpatch(self) -> None:
        unpatch_dream(self.model)


def patch_dream(model: torch.nn.Module, session: BitSieveSession | None) -> DreamPatchHandle:
    modules = [m for m in model.modules() if is_dream_attention(m)]
    if not modules:
        raise RuntimeError(
            "no DreamAttention modules found; is this a Dream checkpoint?"
        )
    for module in modules:
        if not hasattr(module, "_bitsieve_original_forward"):
            setattr(module, "_bitsieve_original_forward", module.forward)
            module.forward = types.MethodType(_sparse_forward, module)
        setattr(module, "_bitsieve_session", session)
    setattr(model, "_bitsieve_patch_modules", modules)
    return DreamPatchHandle(model=model, modules=modules)


def unpatch_dream(model: torch.nn.Module) -> None:
    for module in getattr(model, "_bitsieve_patch_modules", []):
        original = getattr(module, "_bitsieve_original_forward", None)
        if original is not None:
            module.forward = original
            delattr(module, "_bitsieve_original_forward")
        if hasattr(module, "_bitsieve_session"):
            delattr(module, "_bitsieve_session")
    if hasattr(model, "_bitsieve_patch_modules"):
        delattr(model, "_bitsieve_patch_modules")
