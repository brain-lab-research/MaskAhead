from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any

import torch

from ..cache import PackedKVCache
from .session import BitSieveSession


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (
        query * cos + _rotate_half(query) * sin,
        key * cos + _rotate_half(key) * sin,
    )


def _is_fast_dllm_attention(module: torch.nn.Module) -> bool:
    return all(
        hasattr(module, name)
        for name in (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "layer_idx",
            "head_dim",
            "num_key_value_groups",
        )
    )


def _validate_generation_attention_mask(
    attention_mask: torch.Tensor | None,
    *,
    expected_queries: int,
    expected_keys: int,
) -> None:
    if attention_mask is None:
        return
    if attention_mask.shape[-1] != expected_keys:
        raise RuntimeError(
            f"unexpected attention-mask key length {attention_mask.shape[-1]} != {expected_keys}"
        )
    if attention_mask.ndim >= 2 and attention_mask.shape[-2] not in (1, expected_queries):
        raise RuntimeError(
            f"unexpected attention-mask query length {attention_mask.shape[-2]}"
        )
    if attention_mask.dtype == torch.bool:
        visible = bool(attention_mask.all().item())
    else:
        visible = bool((attention_mask == 0).all().item())
    if not visible:
        raise RuntimeError(
            "BitSieve received a nontrivial attention mask. The workshop kernels "
            "support Fast-dLLM block generation, not arbitrary masked attention."
        )


def _sparse_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_value=None,
    cache_position: torch.LongTensor | None = None,
    update_past_key_values: bool = False,
    block_past_key_values=None,
    replace_position: int | None = None,
    **kwargs: Any,
):
    session: BitSieveSession | None = getattr(self, "_bitsieve_session", None)
    original = getattr(self, "_bitsieve_original_forward")
    if session is None or not isinstance(past_key_value, PackedKVCache):
        return original(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            update_past_key_values=update_past_key_values,
            block_past_key_values=block_past_key_values,
            replace_position=replace_position,
            **kwargs,
        )

    input_shape = hidden_states.shape[:-1]
    q = self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim).transpose(1, 2)
    k = self.k_proj(hidden_states).view(*input_shape, -1, self.head_dim).transpose(1, 2)
    v = self.v_proj(hidden_states).view(*input_shape, -1, self.head_dim).transpose(1, 2)
    cos, sin = position_embeddings
    q, k = _apply_rope(q, k, cos, sin)

    if session.config.validate_attention_masks:
        _validate_generation_attention_mask(
            attention_mask,
            expected_queries=q.shape[2],
            expected_keys=session.old_cache_len + k.shape[2],
        )


    if block_past_key_values is not None:
        if len(block_past_key_values) <= self.layer_idx:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = block_past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        else:
            if replace_position is None:
                raise RuntimeError("replace_position is required when updating a block cache")
            block_k, block_v = block_past_key_values[self.layer_idx]
            end = replace_position + k.shape[2]
            block_k[:, :, replace_position:end, :].copy_(k)
            block_v[:, :, replace_position:end, :].copy_(v)
            k, v = block_k, block_v

    session.set_rope(cos, sin)
    out = session.attend(self.layer_idx, q, k, v)
    session.stage_if_committing(self.layer_idx, k, v)
    out = out.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    return self.o_proj(out)


@dataclass(slots=True)
class PatchHandle:
    model: torch.nn.Module
    modules: list[torch.nn.Module]

    def set_session(self, session: BitSieveSession | None) -> None:
        for module in self.modules:
            setattr(module, "_bitsieve_session", session)

    def unpatch(self) -> None:
        unpatch_fast_dllm(self.model)


def patch_fast_dllm(
    model: torch.nn.Module,
    session: BitSieveSession | None = None,
) -> PatchHandle:
    modules: list[torch.nn.Module] = []
    for module in model.modules():
        if not _is_fast_dllm_attention(module):
            continue
        if not hasattr(module, "_bitsieve_original_forward"):
            setattr(module, "_bitsieve_original_forward", module.forward)
            module.forward = types.MethodType(_sparse_forward, module)
        setattr(module, "_bitsieve_session", session)
        modules.append(module)
    if not modules:
        raise TypeError(
            "no Fast-dLLM v2 attention modules were found; load "
            "Efficient-Large-Model/Fast_dLLM_v2_7B with trust_remote_code=True"
        )
    setattr(model, "_bitsieve_patch_modules", modules)
    return PatchHandle(model=model, modules=modules)


def set_bitsieve_session(model: torch.nn.Module, session: BitSieveSession | None) -> None:
    modules = getattr(model, "_bitsieve_patch_modules", None)
    if modules is None:
        raise RuntimeError("model is not patched")
    for module in modules:
        setattr(module, "_bitsieve_session", session)


def unpatch_fast_dllm(model: torch.nn.Module) -> None:
    modules = getattr(model, "_bitsieve_patch_modules", [])
    for module in modules:
        original = getattr(module, "_bitsieve_original_forward", None)
        if original is not None:
            module.forward = original
            delattr(module, "_bitsieve_original_forward")
        if hasattr(module, "_bitsieve_session"):
            delattr(module, "_bitsieve_session")
    if hasattr(model, "_bitsieve_patch_modules"):
        delattr(model, "_bitsieve_patch_modules")
