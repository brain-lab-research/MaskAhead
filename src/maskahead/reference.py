from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class PackedKeys:
    payload: torch.Tensor
    scale: torch.Tensor
    zero: torch.Tensor
    bits: int
    token_group: int
    tokens: int


@dataclass(slots=True)
class PackedValues:
    payload: torch.Tensor
    scale: torch.Tensor
    zero: torch.Tensor
    bits: int
    channel_group: int
    tokens: int


def values_per_byte(bits: int) -> int:
    if bits not in (2, 4):
        raise ValueError(f"packed bit width must be 2 or 4, got {bits}")
    return 8 // bits


def pack_last_dim(q: torch.Tensor, bits: int) -> torch.Tensor:
    if q.dtype != torch.uint8:
        q = q.to(torch.uint8)
    vpb = values_per_byte(bits)
    if q.shape[-1] % vpb:
        raise ValueError(f"last dimension {q.shape[-1]} is not divisible by {vpb}")
    shape = (*q.shape[:-1], q.shape[-1] // vpb, vpb)
    x = q.reshape(shape).to(torch.int16)
    shifts = torch.arange(vpb, device=q.device, dtype=torch.int16) * bits
    packed = torch.sum(x << shifts, dim=-1)
    return packed.to(torch.uint8)


def unpack_last_dim(payload: torch.Tensor, bits: int, unpacked_size: int) -> torch.Tensor:
    vpb = values_per_byte(bits)
    if unpacked_size != payload.shape[-1] * vpb:
        raise ValueError("unpacked_size does not match payload shape")
    shifts = torch.arange(vpb, device=payload.device, dtype=torch.int16) * bits
    x = payload.to(torch.int16).unsqueeze(-1)
    q = (x >> shifts) & ((1 << bits) - 1)
    return q.reshape(*payload.shape[:-1], unpacked_size).to(torch.uint8)


def quantize_keys(
    key: torch.Tensor,
    *,
    bits: int = 4,
    token_group: int = 32,
    param_dtype: torch.dtype = torch.float16,
) -> PackedKeys:
    if key.ndim != 4:
        raise ValueError("key must have shape [B, Hkv, T, D]")
    if bits not in (2, 4):
        raise ValueError("quantize_keys supports 2- or 4-bit payloads")
    b, h, t, d = key.shape
    if t % token_group:
        raise ValueError(f"T={t} must be divisible by token_group={token_group}")
    if token_group % values_per_byte(bits):
        raise ValueError("token_group is incompatible with bit width")

    ng = t // token_group

    x = key.reshape(b, h, ng, token_group, d).permute(0, 1, 2, 4, 3)
    mn = x.amin(dim=-1, keepdim=True)
    mx = x.amax(dim=-1, keepdim=True)
    scale = ((mx - mn) / float((1 << bits) - 1)).clamp_min(1e-8)
    q = torch.round((x - mn) / scale).clamp_(0, (1 << bits) - 1).to(torch.uint8)
    payload = pack_last_dim(q, bits).contiguous()
    return PackedKeys(
        payload=payload,
        scale=scale.squeeze(-1).to(param_dtype).contiguous(),
        zero=mn.squeeze(-1).to(param_dtype).contiguous(),
        bits=bits,
        token_group=token_group,
        tokens=t,
    )


def simulate_key_quantization(
    key: torch.Tensor,
    *,
    bits: int,
    token_group: int = 32,
    param_dtype: torch.dtype = torch.float16,
    dtype: torch.dtype | None = None,
    allow_ragged: bool = False,
) -> torch.Tensor:
    """KIVI key quantization round-tripped back to float, for ANY bit width.

    Returns exactly the values a b-bit packed cache would hand the selector, and
    is bit-exact with quantize_keys + dequantize_keys wherever those exist - the
    grouping, the dtype the min/max and scale are computed in, and the float16
    storage of scale/zero all matter, because near a bin edge any of them moves a
    value a whole quantization step.

    It differs from the packed path only in skipping the bit-packing, which is
    why it is not limited to the 2 and 4 bits the kernels can address: 3 bits
    straddle byte boundaries and have no packed path here, but their quantization
    grid is perfectly well defined.

    `allow_ragged` quantizes a trailing partial group as its own short group,
    for callers holding a live prefix whose length is not a multiple of
    token_group. Off by default so the strict divisibility the packed path
    demands stays the checked behaviour.

    That makes this the honest way to measure how selection quality degrades with
    precision - the selector sees the numbers it would really see, and the memory
    claim (bits x entries) is arithmetic rather than something the measurement
    has to demonstrate. It is NOT a substitute for the packed path in a speed or
    memory measurement: nothing here is actually stored compactly.
    """
    if key.ndim != 4:
        raise ValueError("key must have shape [B, Hkv, T, D]")
    if not 1 <= bits <= 16:
        raise ValueError(f"bits must be between 1 and 16, got {bits}")
    if bits == 16:
        return key if dtype is None else key.to(dtype)
    b, h, t, d = key.shape
    if t % token_group:
        if not allow_ragged:
            raise ValueError(f"T={t} must be divisible by token_group={token_group}")
        head = (t // token_group) * token_group
        parts = []
        if head:
            parts.append(
                simulate_key_quantization(
                    key[:, :, :head, :], bits=bits, token_group=token_group,
                    param_dtype=param_dtype, dtype=dtype,
                )
            )
        tail = key[:, :, head:, :]
        parts.append(
            simulate_key_quantization(
                tail, bits=bits, token_group=tail.shape[2],
                param_dtype=param_dtype, dtype=dtype,
            )
        )
        return torch.cat(parts, dim=2)

    ng = t // token_group
    x = key.reshape(b, h, ng, token_group, d).permute(0, 1, 2, 4, 3)
    mn = x.amin(dim=-1, keepdim=True)
    mx = x.amax(dim=-1, keepdim=True)
    levels = float((1 << bits) - 1)
    scale = ((mx - mn) / levels).clamp_min(1e-8)
    q = torch.round((x - mn) / scale).clamp_(0, levels)
    # scale and zero live in param_dtype in the real cache; rounding them here
    # too is what makes this bit-exact rather than merely close.
    x = q.float() * scale.to(param_dtype).float() + mn.to(param_dtype).float()
    x = x.permute(0, 1, 2, 4, 3).reshape(b, h, t, d)
    return x.to(dtype if dtype is not None else key.dtype)


def dequantize_keys(packed: PackedKeys, *, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    q = unpack_last_dim(packed.payload, packed.bits, packed.token_group)
    x = q.to(torch.float32) * packed.scale.float().unsqueeze(-1)
    x = x + packed.zero.float().unsqueeze(-1)

    x = x.permute(0, 1, 2, 4, 3).reshape(
        packed.payload.shape[0],
        packed.payload.shape[1],
        packed.tokens,
        packed.scale.shape[-1],
    )
    return x.to(dtype)


def quantize_values(
    value: torch.Tensor,
    *,
    bits: int = 4,
    channel_group: int = 32,
    param_dtype: torch.dtype = torch.float16,
) -> PackedValues:
    if value.ndim != 4:
        raise ValueError("value must have shape [B, Hkv, T, D]")
    if bits not in (2, 4):
        raise ValueError("quantize_values supports 2- or 4-bit payloads")
    b, h, t, d = value.shape
    if d % channel_group:
        raise ValueError(f"D={d} must be divisible by channel_group={channel_group}")
    if channel_group % values_per_byte(bits):
        raise ValueError("channel_group is incompatible with bit width")

    ng = d // channel_group
    x = value.reshape(b, h, t, ng, channel_group)
    mn = x.amin(dim=-1, keepdim=True)
    mx = x.amax(dim=-1, keepdim=True)
    scale = ((mx - mn) / float((1 << bits) - 1)).clamp_min(1e-8)
    q = torch.round((x - mn) / scale).clamp_(0, (1 << bits) - 1).to(torch.uint8)
    payload = pack_last_dim(q, bits).contiguous()
    return PackedValues(
        payload=payload,
        scale=scale.squeeze(-1).to(param_dtype).contiguous(),
        zero=mn.squeeze(-1).to(param_dtype).contiguous(),
        bits=bits,
        channel_group=channel_group,
        tokens=t,
    )


def simulate_value_quantization(
    value: torch.Tensor,
    *,
    bits: int,
    channel_group: int = 32,
    param_dtype: torch.dtype = torch.float16,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Value quantization round-tripped back to float, the twin of
    ``simulate_key_quantization``.

    Values group along channels, not tokens, so there is no ragged case: the
    head dimension is fixed and must divide ``channel_group``. Like the key
    version this rounds scale and zero through ``param_dtype``, which is what
    makes it bit-exact with ``quantize_values`` + ``dequantize_values`` rather
    than merely close, and it skips only the bit-packing - so it is not limited
    to the 2 and 4 bits the kernels can address.

    Training reads old-cache values through this, so what a student learns to
    tolerate is the grid the packed cache actually hands it.
    """
    if value.ndim != 4:
        raise ValueError("value must have shape [B, Hkv, T, D]")
    if not 1 <= bits <= 16:
        raise ValueError(f"bits must be between 1 and 16, got {bits}")
    if bits == 16:
        return value if dtype is None else value.to(dtype)
    b, h, t, d = value.shape
    if d % channel_group:
        raise ValueError(f"D={d} must be divisible by channel_group={channel_group}")

    ng = d // channel_group
    x = value.reshape(b, h, t, ng, channel_group)
    mn = x.amin(dim=-1, keepdim=True)
    mx = x.amax(dim=-1, keepdim=True)
    levels = float((1 << bits) - 1)
    scale = ((mx - mn) / levels).clamp_min(1e-8)
    q = torch.round((x - mn) / scale).clamp_(0, levels)
    x = q.float() * scale.to(param_dtype).float() + mn.to(param_dtype).float()
    x = x.reshape(b, h, t, d)
    return x.to(dtype if dtype is not None else value.dtype)


def dequantize_values(
    packed: PackedValues,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    q = unpack_last_dim(packed.payload, packed.bits, packed.channel_group)
    x = q.to(torch.float32) * packed.scale.float().unsqueeze(-1)
    x = x + packed.zero.float().unsqueeze(-1)
    b, h, t, ng, cg = x.shape
    return x.reshape(b, h, t, ng * cg).to(dtype)


def repeat_kv(x: torch.Tensor, num_q_heads: int) -> torch.Tensor:
    if x.shape[1] == num_q_heads:
        return x
    if num_q_heads % x.shape[1]:
        raise ValueError("number of query heads must be divisible by KV heads")
    return x.repeat_interleave(num_q_heads // x.shape[1], dim=1)


def dense_attention_reference(
    query: torch.Tensor,
    old_key: torch.Tensor,
    old_value: torch.Tensor,
    current_key: torch.Tensor,
    current_value: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    key = torch.cat([old_key, current_key], dim=2)
    value = torch.cat([old_value, current_value], dim=2)
    key = repeat_kv(key, query.shape[1])
    value = repeat_kv(value, query.shape[1])
    if scale is None:
        scale = query.shape[-1] ** -0.5
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scale
    prob = torch.softmax(scores, dim=-1)
    return torch.matmul(prob, value.float()).to(query.dtype)


def compact_attention_reference(
    query: torch.Tensor,
    selected_key: torch.Tensor,
    selected_value: torch.Tensor,
    current_key: torch.Tensor,
    current_value: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    return dense_attention_reference(
        query,
        selected_key,
        selected_value,
        current_key,
        current_value,
        scale=scale,
    )


def selector_importance_reference(
    query: torch.Tensor,
    old_key: torch.Tensor,
    *,
    query_indices: list[int],
    current_key: torch.Tensor | None = None,
    domain: str = "prefix",
    score_kind: str = "softmax",
    scale: float | None = None,
    live_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """``live_mask`` [B, Hkv, N]: False marks entries evicted from the cache.

    They are removed before the softmax, not after: an entry that no longer
    exists must not contribute to the normalizer, or every surviving alpha is
    scaled down by mass that went nowhere.
    """
    if not query_indices:
        raise ValueError("query_indices cannot be empty")
    b, hq, _, d = query.shape
    hkv = old_key.shape[1]
    if hq % hkv:
        raise ValueError("Hq must be divisible by Hkv")
    g = hq // hkv
    q = query[:, :, query_indices, :].reshape(b, hkv, g, len(query_indices), d)
    logits_old = torch.einsum("bhgmd,bhnd->bhgmn", q.float(), old_key.float())
    logits_old = logits_old * (scale if scale is not None else d**-0.5)
    if live_mask is not None:
        logits_old = logits_old.masked_fill(
            ~live_mask[:, :, None, None, :], float("-inf")
        )

    if score_kind == "raw":
        return logits_old.mean(dim=(2, 3))
    if score_kind != "softmax":
        raise ValueError(f"unknown score_kind={score_kind}")

    if domain == "prefix":
        prob_old = torch.softmax(logits_old, dim=-1)
    elif domain == "full":
        if current_key is None:
            raise ValueError("current_key is required for domain='full'")
        logits_cur = torch.einsum("bhgmd,bhnd->bhgmn", q.float(), current_key.float())
        logits_cur = logits_cur * (scale if scale is not None else d**-0.5)
        denom = torch.logsumexp(torch.cat([logits_old, logits_cur], dim=-1), dim=-1)
        prob_old = torch.exp(logits_old - denom.unsqueeze(-1))
    else:
        raise ValueError(f"unknown selector domain={domain}")
    return prob_old.mean(dim=(2, 3))


def select_topk_reference(
    query: torch.Tensor,
    old_key: torch.Tensor,
    *,
    query_indices: list[int],
    topk: int,
    current_key: torch.Tensor | None = None,
    domain: str = "prefix",
    score_kind: str = "softmax",
    sort_indices: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    importance = selector_importance_reference(
        query,
        old_key,
        query_indices=query_indices,
        current_key=current_key,
        domain=domain,
        score_kind=score_kind,
    )
    k = min(topk, old_key.shape[2])
    values, indices = torch.topk(importance, k=k, dim=-1, largest=True, sorted=False)
    if sort_indices:
        indices, order = torch.sort(indices, dim=-1)
        values = torch.gather(values, -1, order)
    return indices, values


def gather_per_kv_head(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if x.shape[:2] != indices.shape[:2]:
        raise ValueError("batch/head dimensions do not match")
    gather_idx = indices.unsqueeze(-1).expand(*indices.shape, x.shape[-1])
    return torch.gather(x, 2, gather_idx)


def sdpa_compact(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    try:
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=query.shape[1] != key.shape[1],
        )
    except (RuntimeError, TypeError):
        return F.scaled_dot_product_attention(
            query,
            repeat_kv(key, query.shape[1]),
            repeat_kv(value, query.shape[1]),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
