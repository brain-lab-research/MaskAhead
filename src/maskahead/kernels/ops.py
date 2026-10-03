from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from ..cache import LayerCacheView
from ..reference import (
    dense_attention_reference,
    gather_per_kv_head,
    selector_importance_reference,
)
from . import triton_ops

Backend = Literal["auto", "triton", "torch"]


class KernelUnavailableError(RuntimeError):
    pass


def triton_available() -> bool:
    return bool(triton_ops.TRITON_AVAILABLE and torch.cuda.is_available())


def _require_contiguous(x: torch.Tensor, name: str) -> torch.Tensor:
    return x if x.is_contiguous() else x.contiguous()


def _dummy(device: torch.device, dtype: torch.dtype = torch.float16) -> torch.Tensor:

    return torch.empty((1,), device=device, dtype=dtype)


def _strides(x: torch.Tensor | None, rank: int) -> tuple[int, ...]:
    if x is None:
        return (0,) * rank
    if x.ndim != rank:
        raise ValueError(f"expected rank {rank}, got {x.ndim}")
    return tuple(int(v) for v in x.stride())


def _choose_backend(backend: Backend, tensor: torch.Tensor) -> Backend:
    if backend == "torch":
        return "torch"
    if backend == "triton":
        if tensor.device.type != "cuda" or not triton_ops.TRITON_AVAILABLE:
            raise KernelUnavailableError("Triton backend requested but CUDA/Triton is unavailable")
        return "triton"
    if tensor.device.type == "cuda" and triton_ops.TRITON_AVAILABLE:
        return "triton"
    return "torch"


_TRITON_DOT_MIN_DIM = 16


def _dot_row_tile(num_rows: int) -> int:
    if num_rows <= 0:
        raise ValueError("tl.dot kernels require at least one logical row")
    return _TRITON_DOT_MIN_DIM

def _materialize_view(view: LayerCacheView, *, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    from ..reference import PackedKeys, PackedValues, dequantize_keys, dequantize_values

    k_parts: list[torch.Tensor] = []
    v_parts: list[torch.Tensor] = []
    if view.k_bits == 16:
        if view.k_fp is None:
            raise ValueError("missing full-precision K")
        k_parts.append(view.k_fp.to(dtype))
    else:
        if view.quantized_length:
            if view.k_q is None or view.k_scale is None or view.k_zero is None:
                raise ValueError("missing packed K")
            k_parts.append(
                dequantize_keys(
                    PackedKeys(
                        view.k_q,
                        view.k_scale,
                        view.k_zero,
                        view.k_bits,
                        view.key_token_group,
                        view.quantized_length,
                    ),
                    dtype=dtype,
                )
            )
        if view.residual_length:
            if view.k_residual is None:
                raise ValueError("missing residual K")
            k_parts.append(view.k_residual.to(dtype))

    if view.v_bits == 16:
        if view.v_fp is None:
            raise ValueError("missing full-precision V")
        v_parts.append(view.v_fp.to(dtype))
    else:
        if view.quantized_length:
            if view.v_q is None or view.v_scale is None or view.v_zero is None:
                raise ValueError("missing packed V")
            v_parts.append(
                dequantize_values(
                    PackedValues(
                        view.v_q,
                        view.v_scale,
                        view.v_zero,
                        view.v_bits,
                        view.value_channel_group,
                        view.quantized_length,
                    ),
                    dtype=dtype,
                )
            )
        if view.residual_length:
            if view.v_residual is None:
                raise ValueError("missing residual V")
            v_parts.append(view.v_residual.to(dtype))

    if not k_parts or not v_parts:
        shape = (0,)
        raise ValueError(f"cannot materialize empty view: {shape}")
    return (
        k_parts[0] if len(k_parts) == 1 else torch.cat(k_parts, dim=2),
        v_parts[0] if len(v_parts) == 1 else torch.cat(v_parts, dim=2),
    )


def dense_packed_attention(
    query: torch.Tensor,
    view: LayerCacheView,
    current_key: torch.Tensor,
    current_value: torch.Tensor,
    *,
    scaling: float | None = None,
    backend: Backend = "auto",
    kernel_variant: str = "blocked",
) -> torch.Tensor:
    if query.ndim != 4 or current_key.ndim != 4 or current_value.ndim != 4:
        raise ValueError("Q/K/V must be rank-4 [B,H,T,D]")
    if current_key.shape != current_value.shape:
        raise ValueError("current K and V shapes differ")
    b, hq, tq, d = query.shape
    if current_key.shape[0] != b or current_key.shape[-1] != d:
        raise ValueError("current K shape is incompatible with Q")
    hkv = current_key.shape[1]
    if hq % hkv:
        raise ValueError("Hq must be divisible by Hkv")
    if view.length == 0:
        from ..reference import sdpa_compact

        return sdpa_compact(query, current_key, current_value)

    chosen = _choose_backend(backend, query)
    if chosen == "torch":
        old_k, old_v = _materialize_view(view, dtype=query.dtype)
        return dense_attention_reference(
            query,
            old_k,
            old_v,
            current_key,
            current_value,
            scale=scaling,
        )

    q = _require_contiguous(query, "query")
    kc = _require_contiguous(current_key, "current_key")
    vc = _require_contiguous(current_value, "current_value")
    out = torch.empty_like(q)
    device = q.device


    if view.k_bits == 16 and view.v_bits == 16:
        n_quant, n_residual = view.length, 0
    else:
        n_quant, n_residual = view.quantized_length, view.residual_length

    kq = view.k_q if view.k_q is not None else _dummy(device, torch.uint8)
    ks = view.k_scale if view.k_scale is not None else _dummy(device)
    kz = view.k_zero if view.k_zero is not None else _dummy(device)
    vq = view.v_q if view.v_q is not None else _dummy(device, torch.uint8)
    vs = view.v_scale if view.v_scale is not None else _dummy(device)
    vz = view.v_zero if view.v_zero is not None else _dummy(device)
    kfp = view.k_fp if view.k_fp is not None else _dummy(device, q.dtype)
    vfp = view.v_fp if view.v_fp is not None else _dummy(device, q.dtype)
    kr = view.k_residual if view.k_residual is not None else _dummy(device, q.dtype)
    vr = view.v_residual if view.v_residual is not None else _dummy(device, q.dtype)

    sq = _strides(q, 4)
    skq = _strides(view.k_q, 5)
    sks = _strides(view.k_scale, 4)
    svq = _strides(view.v_q, 5)
    svs = _strides(view.v_scale, 4)
    skf = _strides(view.k_fp, 4)
    svf = _strides(view.v_fp, 4)
    skr = _strides(view.k_residual, 4)
    svr = _strides(view.v_residual, 4)
    skc = _strides(kc, 4)
    svc = _strides(vc, 4)
    so = _strides(out, 4)

    block_d = triton_ops.triton.next_power_of_2(d)
    block_n = 32
    if kernel_variant == "blocked":
        block_m = _dot_row_tile(tq)
        grid = (b * hq, triton_ops.triton.cdiv(tq, block_m))
        kernel = triton_ops.dense_packed_attention_blocked_kernel
    elif kernel_variant == "legacy":
        block_m = 1
        grid = (b * hq, tq)
        kernel = triton_ops.dense_packed_attention_kernel
    else:
        raise ValueError(f"unknown dense kernel variant: {kernel_variant}")
    kernel[grid](
        q,
        kq,
        ks,
        kz,
        vq,
        vs,
        vz,
        kfp,
        vfp,
        kr,
        vr,
        kc,
        vc,
        out,
        n_quant,
        n_residual,
        current_key.shape[2],
        *sq,
        *skq,
        *sks,
        *svq,
        *svs,
        *skf,
        *svf,
        *skr,
        *svr,
        *skc,
        *svc,
        *so,
        float(scaling if scaling is not None else d**-0.5),
        H_Q=hq,
        H_KV=hkv,
        T_Q=tq,
        HEAD_DIM=d,
        BLOCK_D=block_d,
        BLOCK_N=block_n,
        KEY_BITS=view.k_bits,
        VALUE_BITS=view.v_bits,
        KEY_GROUP=view.key_token_group,
        VALUE_GROUP=view.value_channel_group,
        **(
            {"BLOCK_M": block_m, "USE_BF16": q.dtype == torch.bfloat16}
            if kernel_variant == "blocked"
            else {}
        ),
        num_warps=4 if kernel_variant == "legacy" else 8,
        num_stages=2,
    )
    return out


@dataclass(slots=True)
class SelectorResult:
    indices: torch.Tensor
    values: torch.Tensor
    importance: torch.Tensor | None = None


class SelectorScratch:

    def __init__(self) -> None:
        self._buffers: dict[tuple, tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def _token_capacity(tokens: int) -> int:
        if tokens <= 0:
            raise ValueError("selector scratch requires a positive token count")

        return max(256, 1 << (tokens - 1).bit_length())

    def tensors(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        batch: int,
        hkv: int,
        representatives: int,
        tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if representatives <= 0:
            raise ValueError("selector scratch requires representative rows")
        key = (device.type, device.index, dtype, batch, hkv, representatives)
        entry = self._buffers.get(key)
        if entry is None or entry[0] < tokens:
            capacity = self._token_capacity(tokens)
            logits = torch.empty(
                (batch, hkv, representatives, capacity), device=device, dtype=dtype
            )
            lse = torch.empty(
                (batch, hkv, representatives), device=device, dtype=torch.float32
            )
            importance = torch.empty(
                (batch, hkv, capacity), device=device, dtype=torch.float32
            )
            entry = (capacity, logits, lse, importance)
            self._buffers[key] = entry
        _, logits, lse, importance = entry
        return logits[..., :tokens], lse, importance[..., :tokens]


_GLOBAL_SELECTOR_SCRATCH = SelectorScratch()


def _selected_queries(query: torch.Tensor, query_indices: list[int], hkv: int) -> torch.Tensor:
    if not query_indices:
        raise ValueError("query_indices cannot be empty")
    b, hq, _, d = query.shape
    if hq % hkv:
        raise ValueError("Hq must be divisible by Hkv")


    q = query[:, :, query_indices, :]
    q = q.reshape(b, hkv, hq // hkv, len(query_indices), d)
    return q.reshape(b, hkv, -1, d).contiguous()


def selector_topk(
    query: torch.Tensor,
    view: LayerCacheView,
    *,
    query_indices: list[int],
    topk: int,
    current_key: torch.Tensor | None = None,
    domain: str = "prefix",
    score_kind: str = "softmax",
    sort_indices: bool = True,
    scaling: float | None = None,
    backend: Backend = "auto",
    live_mask: torch.Tensor | None = None,
    return_importance: bool = False,
    scratch: SelectorScratch | None = None,
    kernel_variant: str = "blocked",
    logits_dtype: torch.dtype = torch.float16,
) -> SelectorResult:
    if view.length <= 0:
        raise ValueError("selector requires a non-empty prefix")
    hkv = (
        view.k_fp.shape[1]
        if view.k_fp is not None
        else view.k_q.shape[1] if view.k_q is not None else view.k_residual.shape[1]
    )
    qrep = _selected_queries(query, query_indices, hkv)
    k = min(int(topk), view.length)
    chosen = _choose_backend(backend, query)

    if chosen == "torch":
        old_k, _ = _materialize_view(view, dtype=query.dtype)
        importance = selector_importance_reference(
            query,
            old_k,
            query_indices=query_indices,
            current_key=current_key,
            domain=domain,
            score_kind=score_kind,
            scale=scaling,
            live_mask=live_mask,
        )
        # The masked softmax already puts exact zeros on dead entries, which is
        # what importance means -- a probability. -inf is only for ranking, so
        # a dead entry can never be selected even if every live one scores 0.
        rank_on = importance if live_mask is None else importance.masked_fill(
            ~live_mask, float("-inf")
        )
        values, indices = torch.topk(rank_on, k=k, dim=-1, sorted=False)
    else:
        b, hkv, r, d = qrep.shape
        n = view.length
        scratch = scratch or _GLOBAL_SELECTOR_SCRATCH
        logits, lse, importance = scratch.tensors(
            device=query.device,
            dtype=logits_dtype,
            batch=b,
            hkv=hkv,
            representatives=r,
            tokens=n,
        )
        kq = view.k_q if view.k_q is not None else _dummy(query.device, torch.uint8)
        ks = view.k_scale if view.k_scale is not None else _dummy(query.device)
        kz = view.k_zero if view.k_zero is not None else _dummy(query.device)
        kfp = view.k_fp if view.k_fp is not None else _dummy(query.device, query.dtype)
        kr = view.k_residual if view.k_residual is not None else _dummy(query.device, query.dtype)
        if view.k_bits == 16:
            n_quant, n_res = view.length, 0
        else:
            n_quant, n_res = view.quantized_length, view.residual_length

        sq = _strides(qrep, 4)
        skq = _strides(view.k_q, 5)
        sks = _strides(view.k_scale, 4)
        skf = _strides(view.k_fp, 4)
        skr = _strides(view.k_residual, 4)
        sl = _strides(logits, 4)
        block_d = triton_ops.triton.next_power_of_2(d)
        block_n = 32
        if kernel_variant == "blocked":
            block_r = _dot_row_tile(r)
            grid = (
                b * hkv,
                triton_ops.triton.cdiv(r, block_r),
                triton_ops.triton.cdiv(n, block_n),
            )
            selector_kernel = triton_ops.selector_logits_blocked_kernel
        elif kernel_variant == "legacy":
            block_r = 1
            grid = (b * hkv * r, triton_ops.triton.cdiv(n, block_n))
            selector_kernel = triton_ops.selector_logits_kernel
        else:
            raise ValueError(f"unknown selector kernel variant: {kernel_variant}")
        selector_kernel[grid](
            qrep,
            kq,
            ks,
            kz,
            kfp,
            kr,
            logits,
            n_quant,
            n_res,
            *sq,
            *skq,
            *sks,
            *skf,
            *skr,
            *sl,
            float(scaling if scaling is not None else d**-0.5),
            H_KV=hkv,
            R=r,
            HEAD_DIM=d,
            BLOCK_D=block_d,
            BLOCK_N=block_n,
            KEY_BITS=view.k_bits,
            KEY_GROUP=view.key_token_group,
            **(
                {"BLOCK_R": block_r, "USE_BF16": qrep.dtype == torch.bfloat16}
                if kernel_variant == "blocked"
                else {}
            ),
            num_warps=4 if kernel_variant == "legacy" else 8,
            num_stages=2,
        )

        if live_mask is not None:
            # Evicted entries do not exist at inference, so they must not sit in
            # the softmax denominator either -- masking after the fact would
            # leave every live alpha scaled down by whatever the dead entries
            # still carried. [B,Hkv,n] broadcasts over the R representatives.
            logits.masked_fill_(~live_mask.unsqueeze(2), float("-inf"))

        se = _strides(lse, 3)
        if score_kind == "softmax":
            triton_ops.row_logsumexp_kernel[(b * hkv * r,)](
                logits,
                lse,
                n,
                *sl,
                *se,
                H_KV=hkv,
                R=r,
                BLOCK_N=256,
                num_warps=4,
            )
            if domain == "full":
                if current_key is None:
                    raise ValueError("current_key is required for selector domain='full'")


                cur = torch.einsum("bhrd,bhnd->bhrn", qrep.float(), current_key.float())
                cur.mul_(float(scaling if scaling is not None else d**-0.5))
                cur_lse = torch.logsumexp(cur, dim=-1)
                torch.logaddexp(lse, cur_lse, out=lse)
            elif domain != "prefix":
                raise ValueError(f"unknown selector domain: {domain}")
        elif score_kind != "raw":
            raise ValueError(f"unknown selector score: {score_kind}")

        si = _strides(importance, 3)
        triton_ops.reduce_selector_importance_kernel[
            (b * hkv, triton_ops.triton.cdiv(n, 256))
        ](
            logits,
            lse,
            importance,
            n,
            *sl,
            *se,
            *si,
            H_KV=hkv,
            R=r,
            BLOCK_N=256,
            SOFTMAX=score_kind == "softmax",
            num_warps=4,
        )
        rank_on = importance if live_mask is None else importance.masked_fill(
            ~live_mask, float("-inf")
        )
        values, indices = torch.topk(rank_on, k=k, dim=-1, sorted=False)

    if sort_indices:
        indices, order = torch.sort(indices, dim=-1)
        values = torch.gather(values, -1, order)
    return SelectorResult(
        indices=indices,
        values=values,
        importance=importance.clone() if return_importance else None,
    )


def _layer_view_head_dim(
    view: LayerCacheView,
    *,
    out_key: torch.Tensor | None = None,
    out_value: torch.Tensor | None = None,
) -> int:

    evidence: list[tuple[str, int]] = []

    def add(source: str, dim: int | None) -> None:
        if dim is not None:
            evidence.append((source, int(dim)))

    add("view.head_dim", getattr(view, "head_dim", None))
    for name, tensor in (("out_key", out_key), ("out_value", out_value)):
        if tensor is not None:
            if tensor.ndim != 4:
                raise ValueError(f"{name} must be rank 4")
            add(name, int(tensor.shape[-1]))
    for name, tensor in (
        ("k_fp", view.k_fp),
        ("v_fp", view.v_fp),
        ("k_residual", view.k_residual),
        ("v_residual", view.v_residual),
    ):
        if tensor is not None:
            if tensor.ndim != 4:
                raise ValueError(f"{name} must be rank 4")
            add(name, int(tensor.shape[-1]))
    if view.k_q is not None:
        if view.k_q.ndim != 5:
            raise ValueError("k_q must be rank 5")
        packed_width = view.key_token_group * view.k_bits // 8
        a, b = int(view.k_q.shape[-2]), int(view.k_q.shape[-1])
        if b == packed_width and a != packed_width:
            add("k_q", a)
        elif a == packed_width and b != packed_width:
            add("k_q_transposed", b)

    unique = sorted({dim for _, dim in evidence if dim > 0})
    if len(unique) != 1:
        details = ", ".join(f"{name}={dim}" for name, dim in evidence)
        raise ValueError(f"cannot infer one KV head dimension: {details or 'no evidence'}")
    return unique[0]


def gather_packed_kv(
    view: LayerCacheView,
    indices: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    backend: Backend = "auto",
    out_key: torch.Tensor | None = None,
    out_value: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if indices.ndim != 3:
        raise ValueError("indices must have shape [B,Hkv,K]")
    chosen = _choose_backend(backend, indices)
    b, hkv, k = indices.shape
    d = _layer_view_head_dim(
        view,
        out_key=out_key,
        out_value=out_value,
    )
    if out_key is None:
        out_key = torch.empty((b, hkv, k, d), device=indices.device, dtype=dtype)
    if out_value is None:
        out_value = torch.empty_like(out_key)
    if tuple(out_key.shape) != (b, hkv, k, d) or tuple(out_value.shape) != (b, hkv, k, d):
        raise ValueError("output buffer has the wrong shape")

    if chosen == "torch":
        old_k, old_v = _materialize_view(view, dtype=dtype)
        out_key.copy_(gather_per_kv_head(old_k, indices))
        out_value.copy_(gather_per_kv_head(old_v, indices))
        return out_key, out_value

    device = indices.device
    kq = view.k_q if view.k_q is not None else _dummy(device, torch.uint8)
    ks = view.k_scale if view.k_scale is not None else _dummy(device)
    kz = view.k_zero if view.k_zero is not None else _dummy(device)
    vq = view.v_q if view.v_q is not None else _dummy(device, torch.uint8)
    vs = view.v_scale if view.v_scale is not None else _dummy(device)
    vz = view.v_zero if view.v_zero is not None else _dummy(device)
    kfp = view.k_fp if view.k_fp is not None else _dummy(device, dtype)
    vfp = view.v_fp if view.v_fp is not None else _dummy(device, dtype)
    kr = view.k_residual if view.k_residual is not None else _dummy(device, dtype)
    vr = view.v_residual if view.v_residual is not None else _dummy(device, dtype)
    n_quant = view.length if (view.k_bits == 16 and view.v_bits == 16) else view.quantized_length

    si = _strides(indices, 3)
    skq = _strides(view.k_q, 5)
    sks = _strides(view.k_scale, 4)
    svq = _strides(view.v_q, 5)
    svs = _strides(view.v_scale, 4)
    skf = _strides(view.k_fp, 4)
    svf = _strides(view.v_fp, 4)
    skr = _strides(view.k_residual, 4)
    svr = _strides(view.v_residual, 4)
    sok = _strides(out_key, 4)
    sov = _strides(out_value, 4)
    block_d = triton_ops.triton.next_power_of_2(d)
    triton_ops.gather_packed_kv_kernel[(b * hkv * k,)](
        indices,
        kq,
        ks,
        kz,
        vq,
        vs,
        vz,
        kfp,
        vfp,
        kr,
        vr,
        out_key,
        out_value,
        n_quant,
        *si,
        *skq,
        *sks,
        *svq,
        *svs,
        *skf,
        *svf,
        *skr,
        *svr,
        *sok,
        *sov,
        H_KV=hkv,
        K=k,
        HEAD_DIM=d,
        BLOCK_D=block_d,
        KEY_BITS=view.k_bits,
        VALUE_BITS=view.v_bits,
        KEY_GROUP=view.key_token_group,
        VALUE_GROUP=view.value_channel_group,
        num_warps=4,
    )
    return out_key, out_value


def quantize_key_groups_into(
    source: torch.Tensor,
    out_payload: torch.Tensor,
    out_scale: torch.Tensor,
    out_zero: torch.Tensor,
    *,
    bits: int,
    token_group: int,
    backend: Backend = "auto",
) -> None:
    from ..reference import quantize_keys

    chosen = _choose_backend(backend, source)
    if chosen == "torch":
        packed = quantize_keys(
            source,
            bits=bits,
            token_group=token_group,
            param_dtype=out_scale.dtype,
        )
        out_payload.copy_(packed.payload)
        out_scale.copy_(packed.scale)
        out_zero.copy_(packed.zero)
        return
    b, h, t, d = source.shape
    ng = t // token_group
    if t % token_group or out_payload.shape[2] != ng:
        raise ValueError("invalid key quantization output geometry")
    sx = _strides(source, 4)
    sq = _strides(out_payload, 5)
    ss = _strides(out_scale, 4)
    block_d = min(128, triton_ops.triton.next_power_of_2(d))
    triton_ops.quantize_key_groups_kernel[(b * h * ng, triton_ops.triton.cdiv(d, block_d))](
        source,
        out_payload,
        out_scale,
        out_zero,
        t,
        *sx,
        *sq,
        *ss,
        H_KV=h,
        HEAD_DIM=d,
        KEY_BITS=bits,
        KEY_GROUP=token_group,
        BLOCK_D=block_d,
        num_warps=4,
    )


def quantize_values_into(
    source: torch.Tensor,
    out_payload: torch.Tensor,
    out_scale: torch.Tensor,
    out_zero: torch.Tensor,
    *,
    bits: int,
    channel_group: int,
    backend: Backend = "auto",
) -> None:
    from ..reference import quantize_values

    chosen = _choose_backend(backend, source)
    if chosen == "torch":
        packed = quantize_values(
            source,
            bits=bits,
            channel_group=channel_group,
            param_dtype=out_scale.dtype,
        )
        out_payload.copy_(packed.payload)
        out_scale.copy_(packed.scale)
        out_zero.copy_(packed.zero)
        return
    b, h, t, d = source.shape
    sx = _strides(source, 4)
    sq = _strides(out_payload, 5)
    ss = _strides(out_scale, 4)
    channel_groups = d // channel_group
    triton_ops.quantize_values_kernel[(b * h * t * channel_groups,)](
        source,
        out_payload,
        out_scale,
        out_zero,
        t,
        *sx,
        *sq,
        *ss,
        H_KV=h,
        HEAD_DIM=d,
        VALUE_BITS=bits,
        VALUE_GROUP=channel_group,
        num_warps=1,
    )


_base_dense_packed_attention = dense_packed_attention
_base_selector_topk = selector_topk
_base_gather_packed_kv = gather_packed_kv


def dense_packed_attention(
    query: torch.Tensor,
    view: LayerCacheView,
    current_key: torch.Tensor,
    current_value: torch.Tensor,
    *,
    scaling: float | None = None,
    backend: Backend = "auto",
    kernel_variant: str = "blocked",
) -> torch.Tensor:
    from .cuda_fast import maybe_fast_dense_packed_attention

    return maybe_fast_dense_packed_attention(
        _base_dense_packed_attention,
        query,
        view,
        current_key,
        current_value,
        scaling=scaling,
        backend=backend,
        kernel_variant=kernel_variant,
    )


def selector_topk(
    query: torch.Tensor,
    view: LayerCacheView,
    *,
    query_indices: list[int],
    topk: int,
    current_key: torch.Tensor | None = None,
    domain: str = "prefix",
    score_kind: str = "softmax",
    sort_indices: bool = True,
    scaling: float | None = None,
    backend: Backend = "auto",
    live_mask: torch.Tensor | None = None,
    return_importance: bool = False,
    scratch: SelectorScratch | None = None,
    kernel_variant: str = "blocked",
    logits_dtype: torch.dtype = torch.float16,
) -> SelectorResult:
    common = dict(
        query_indices=query_indices,
        topk=topk,
        current_key=current_key,
        domain=domain,
        score_kind=score_kind,
        sort_indices=sort_indices,
        scaling=scaling,
        backend=backend,
        live_mask=live_mask,
        return_importance=return_importance,
        scratch=scratch,
        kernel_variant=kernel_variant,
        logits_dtype=logits_dtype,
    )
    if live_mask is not None:
        # The hand-written CUDA selector has no notion of a live set, and it
        # accepts the argument silently rather than refusing it -- so routing an
        # eviction run through it would score dead entries as if they were still
        # there. Take the Triton path, which masks the logits before the softmax.
        return _base_selector_topk(query, view, **common)

    from .cuda_fast import maybe_fast_selector

    return maybe_fast_selector(_base_selector_topk, query, view, **common)


def gather_packed_kv(
    view: LayerCacheView,
    indices: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    backend: Backend = "auto",
    out_key: torch.Tensor | None = None,
    out_value: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from .cuda_fast import maybe_fast_gather

    return maybe_fast_gather(
        _base_gather_packed_kv,
        view,
        indices,
        dtype=dtype,
        backend=backend,
        out_key=out_key,
        out_value=out_value,
        _cuda_result_style="tuple",
    )


dense_packed_attention._bitsieve_cuda_fast = True
selector_topk._bitsieve_cuda_fast = True
gather_packed_kv._bitsieve_cuda_fast = True
