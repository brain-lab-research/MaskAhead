from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
import os
import sys
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


_FAST_MARKER = "bitsieve-cuda-v1.0"
_SEEN_LOGS: set[str] = set()


def _env_value(name: str) -> str | None:
    return os.getenv(name)


def _env_flag(name: str, default: bool) -> bool:
    value = _env_value(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _log_once(key: str, message: str) -> None:
    if not _env_flag("BITSIEVE_CUDA_VERBOSE", False) or key in _SEEN_LOGS:
        return
    _SEEN_LOGS.add(key)
    print(f"[bitsieve-cuda] {message}", file=sys.stderr, flush=True)


def _enabled() -> bool:
    return _env_flag("BITSIEVE_CUDA_FAST", True)


def _strict() -> bool:
    return _env_flag("BITSIEVE_CUDA_STRICT", False)


def _getattr_any(obj: Any, names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj and obj[name] is not None:
                return obj[name]
    return default


def _as_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return default
        return int(value.item())
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _live_partition_hint(view: Any) -> tuple[int | None, int | None, int | None]:

    total = _as_int(
        _getattr_any(view, ("total_tokens", "cache_len", "length", "seq_len"))
    )
    packed = _as_int(
        _getattr_any(
            view,
            (
                "packed_tokens",
                "num_packed_tokens",
                "quantized_tokens",
                "quantized_length",
                "packed_length",
                "grouped_tokens",
            ),
        )
    )
    residual = _as_int(
        _getattr_any(
            view,
            (
                "num_residual_tokens",
                "residual_tokens",
                "residual_len",
                "residual_length",
            ),
        )
    )

    if total is not None and packed is not None:
        residual = total - packed
    elif total is not None and residual is not None:
        packed = total - residual
    elif packed is not None and residual is not None:
        total = packed + residual

    for name, value in (("total", total), ("packed", packed), ("residual", residual)):
        if value is not None and value < 0:
            raise ValueError(f"negative {name} cache length: {value}")
    if total is not None and packed is not None and residual is not None:
        if packed + residual != total:
            raise ValueError(
                "inconsistent cache partition: "
                f"packed={packed}, residual={residual}, total={total}"
            )
    return total, packed, residual


def _supports_compute_capability(capability: tuple[int, int] | Sequence[int]) -> bool:
    try:
        major, _minor = int(capability[0]), int(capability[1])
    except (TypeError, ValueError, IndexError):
        return False
    return major >= 8


def _architecture_label(capability: tuple[int, int] | Sequence[int]) -> str:
    major, minor = int(capability[0]), int(capability[1])
    if major >= 12:
        return "blackwell"
    if major >= 10:
        return "blackwell"
    if major == 9:
        return "hopper"
    if major == 8 and minor >= 9:
        return "ada"
    if major == 8:
        return "ampere"
    return f"sm{major}{minor}"


@dataclass(frozen=True)
class _CudaDeviceProfile:
    name: str
    major: int
    minor: int
    sm_count: int
    max_threads_per_sm: int

    @property
    def family(self) -> str:


        return _architecture_label((self.major, self.minor))

    @property
    def supports_bf16_tensor_cores(self) -> bool:
        return self.major >= 8


def _device_profile(device: torch.device) -> _CudaDeviceProfile:
    index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    major, minor = torch.cuda.get_device_capability(index)
    return _CudaDeviceProfile(
        name=props.name,
        major=int(major),
        minor=int(minor),
        sm_count=int(props.multi_processor_count),
        max_threads_per_sm=int(props.max_threads_per_multi_processor),
    )


def _supports_recent_cuda(device: torch.device, dtype: torch.dtype | None = None) -> bool:
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    profile = _device_profile(device)


    if profile.major < 8:
        return False
    if dtype == torch.bfloat16 and not profile.supports_bf16_tensor_cores:
        return False
    return True


def _next_power_of_two(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _floor_power_of_two(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x.bit_length() - 1)


@dataclass(frozen=True)
class _FPKeyView:
    key: torch.Tensor
    total_tokens: int


def _normalize_fp_key_view(view: Any, *, head_dim: int) -> _FPKeyView:
    bits = _as_int(_getattr_any(view, ("k_bits", "key_bits")), 16)
    if bits is not None and bits < 16:


        total, packed, residual = _live_partition_hint(view)
        if packed != 0:
            raise ValueError("view declares a low-bit key cache with packed prefix data")
        key = _tensor_attr(
            view,
            (
                "k_residual",
                "key_residual",
                "residual_k",
                "residual_key",
                "k_fp_residual",
            ),
        )
        if key is None or key.ndim != 4 or key.shape[-1] != head_dim:
            raise ValueError("could not find the residual-only full-precision key cache")
        live = residual if residual is not None else total
        if live is None:
            live = int(key.shape[2])
        if live < 0 or live > key.shape[2]:
            raise ValueError("invalid residual-only full-precision cache length")
        return _FPKeyView(key=key, total_tokens=int(live))
    key = _tensor_attr(
        view,
        (
            "k_fp",
            "key_fp",
            "k_full",
            "key_full",
            "full_key",
            "k_cache",
            "key_cache",
            "k_q",
            "key_q",
            "k_residual",
            "key_residual",
            "residual_k",
        ),
    )
    if key is None or key.ndim != 4 or key.shape[-1] != head_dim:
        raise ValueError("could not find a rank-4 full-precision key cache")
    total = _as_int(_getattr_any(view, ("total_tokens", "cache_len", "length", "seq_len")))
    if total is None:
        total = int(key.shape[2])
    if total < 0 or total > key.shape[2]:
        raise ValueError("invalid full-precision cache length")
    return _FPKeyView(key=key, total_tokens=total)


@dataclass(frozen=True)
class _NormalizedView:
    k_q: torch.Tensor
    k_scale: torch.Tensor
    k_zero: torch.Tensor
    v_q: torch.Tensor
    v_scale: torch.Tensor
    v_zero: torch.Tensor
    k_residual: torch.Tensor
    v_residual: torch.Tensor
    k_bits: int
    v_bits: int
    key_group: int
    value_group: int
    packed_tokens: int
    residual_tokens: int
    total_tokens: int
    head_dim: int


    skb: int
    skh: int
    skg: int
    skd: int
    skp: int

    sksb: int
    sksh: int
    sksg: int
    sksd: int

    svb: int
    svh: int
    svn: int
    svg: int
    svp: int

    svsb: int
    svsh: int
    svsn: int
    svsg: int


def _tensor_attr(obj: Any, names: Sequence[str]) -> torch.Tensor | None:
    value = _getattr_any(obj, names)
    return value if isinstance(value, torch.Tensor) else None


def _metadata_strides(
    tensor: torch.Tensor, *, groups: int, head_dim: int, name: str
) -> tuple[int, int, int, int]:
    if tensor.ndim != 4:
        raise ValueError(f"{name} must be rank 4, got {tuple(tensor.shape)}")

    if groups < 0:
        raise ValueError(f"{name} received a negative live group count: {groups}")
    if tensor.shape[2] >= groups and tensor.shape[3] == head_dim:
        return tensor.stride(0), tensor.stride(1), tensor.stride(2), tensor.stride(3)
    if tensor.shape[2] == head_dim and tensor.shape[3] >= groups:
        return tensor.stride(0), tensor.stride(1), tensor.stride(3), tensor.stride(2)
    raise ValueError(
        f"cannot recognize {name} layout {tuple(tensor.shape)} for groups={groups}, D={head_dim}"
    )


def _value_metadata_strides(
    tensor: torch.Tensor, *, tokens: int, value_groups: int, name: str
) -> tuple[int, int, int, int]:
    if tensor.ndim != 4:
        raise ValueError(f"{name} must be rank 4, got {tuple(tensor.shape)}")
    if tokens < 0:
        raise ValueError(f"{name} received a negative live token count: {tokens}")
    if tensor.shape[2] >= tokens and tensor.shape[3] == value_groups:
        return tensor.stride(0), tensor.stride(1), tensor.stride(2), tensor.stride(3)
    if tensor.shape[2] == value_groups and tensor.shape[3] >= tokens:
        return tensor.stride(0), tensor.stride(1), tensor.stride(3), tensor.stride(2)
    raise ValueError(
        f"cannot recognize {name} layout {tuple(tensor.shape)} for N={tokens}, VG={value_groups}"
    )


def _infer_gather_head_dim(
    view: Any,
    *,
    out_key: torch.Tensor | None = None,
    out_value: torch.Tensor | None = None,
) -> int:

    evidence: list[tuple[str, int]] = []

    def add(source: str, value: int | None) -> None:
        if value is None:
            return
        dim = int(value)
        if dim <= 0:
            raise ValueError(f"invalid head dimension from {source}: {dim}")
        evidence.append((source, dim))

    add(
        "view.head_dim",
        _as_int(_getattr_any(view, ("head_dim", "head_size", "d_head"))),
    )

    for name, tensor in (("out_key", out_key), ("out_value", out_value)):
        if tensor is not None:
            if tensor.ndim != 4:
                raise ValueError(f"{name} must be rank 4, got {tuple(tensor.shape)}")
            add(name, int(tensor.shape[-1]))

    for name, names in (
        ("k_residual", ("k_residual", "key_residual", "residual_k", "residual_key", "k_fp_residual")),
        ("v_residual", ("v_residual", "value_residual", "residual_v", "residual_value", "v_fp_residual")),
        ("k_fp", ("k_fp", "key_fp", "full_key", "key", "keys")),
        ("v_fp", ("v_fp", "value_fp", "full_value", "value", "values")),
    ):
        tensor = _tensor_attr(view, names)
        if tensor is not None:
            if tensor.ndim != 4:
                raise ValueError(f"{name} must be rank 4, got {tuple(tensor.shape)}")
            add(name, int(tensor.shape[-1]))

    k_q = _tensor_attr(
        view,
        ("k_q", "key_q", "key_payload", "packed_k", "packed_key", "key_packed"),
    )
    k_bits = _as_int(_getattr_any(view, ("k_bits", "key_bits")))
    key_group = _as_int(
        _getattr_any(view, ("key_group", "key_token_group", "k_group_size", "group_size")),
        32,
    )
    if k_q is not None and k_q.ndim == 5 and k_bits in (2, 4) and key_group:
        packed_width = int(key_group) * int(k_bits) // 8
        axis3, axis4 = int(k_q.shape[3]), int(k_q.shape[4])
        if axis4 == packed_width and axis3 != packed_width:
            add("packed K payload", axis3)
        elif axis3 == packed_width and axis4 != packed_width:
            add("transposed packed K payload", axis4)
        elif axis3 == packed_width and axis4 == packed_width:
            raise ValueError(
                f"ambiguous packed K payload {tuple(k_q.shape)}: both trailing axes equal {packed_width}"
            )

    unique = sorted({dim for _, dim in evidence})
    if not unique:
        raise ValueError(
            "cannot infer gather head dimension from view, output buffers, residual/full tensors, or packed K payload"
        )
    if len(unique) != 1:
        details = ", ".join(f"{source}={dim}" for source, dim in evidence)
        raise ValueError(f"inconsistent gather head dimensions: {details}")
    return unique[0]


def _normalize_view(view: Any, *, head_dim: int) -> _NormalizedView:
    k_q = _tensor_attr(
        view,
        ("k_q", "key_q", "key_payload", "packed_k", "packed_key", "key_packed"),
    )
    k_scale = _tensor_attr(view, ("k_scale", "key_scale", "k_scales", "key_scales"))
    k_zero = _tensor_attr(
        view, ("k_zero", "key_zero", "k_zeros", "key_zeros", "k_min", "key_min")
    )
    v_q = _tensor_attr(
        view,
        ("v_q", "value_q", "value_payload", "packed_v", "packed_value", "value_packed"),
    )
    v_scale = _tensor_attr(view, ("v_scale", "value_scale", "v_scales", "value_scales"))
    v_zero = _tensor_attr(
        view, ("v_zero", "value_zero", "v_zeros", "value_zeros", "v_min", "value_min")
    )
    k_res = _tensor_attr(
        view,
        ("k_residual", "key_residual", "residual_k", "residual_key", "k_fp_residual"),
    )
    v_res = _tensor_attr(
        view,
        ("v_residual", "value_residual", "residual_v", "residual_value", "v_fp_residual"),
    )
    missing = [
        name
        for name, tensor in (
            ("k_q", k_q),
            ("k_scale", k_scale),
            ("k_zero", k_zero),
            ("v_q", v_q),
            ("v_scale", v_scale),
            ("v_zero", v_zero),
            ("k_residual", k_res),
            ("v_residual", v_res),
        )
        if tensor is None
    ]
    if missing:
        raise ValueError(f"packed layer view is missing: {', '.join(missing)}")
    assert k_q is not None and k_scale is not None and k_zero is not None
    assert v_q is not None and v_scale is not None and v_zero is not None
    assert k_res is not None and v_res is not None

    k_bits = _as_int(_getattr_any(view, ("k_bits", "key_bits")))
    v_bits = _as_int(_getattr_any(view, ("v_bits", "value_bits")))
    key_group = _as_int(
        _getattr_any(view, ("key_group", "key_token_group", "k_group_size", "group_size")),
        32,
    )
    value_group = _as_int(
        _getattr_any(view, ("value_group", "value_channel_group", "v_group_size")),
        32,
    )
    if key_group is None or value_group is None:
        raise ValueError("could not infer KIVI group sizes")

    declared_head_dim = _as_int(
        _getattr_any(view, ("head_dim", "head_size", "d_head"))
    )
    if declared_head_dim is not None and declared_head_dim != head_dim:
        raise ValueError(
            f"view declares head_dim={declared_head_dim}, but caller requested D={head_dim}"
        )


    if k_q.ndim != 5 or v_q.ndim != 5:
        raise ValueError(
            f"low-bit payloads must be rank 5, got K={tuple(k_q.shape)}, V={tuple(v_q.shape)}"
        )
    if k_bits is None:
        candidates = [d for d in k_q.shape[3:] if d != head_dim]
        if len(candidates) != 1:
            raise ValueError("could not infer key bit width")
        k_bits = int(candidates[0] * 8 // key_group)
    if v_bits is None:
        value_groups = head_dim // value_group
        candidates = [d for d in v_q.shape[3:] if d != value_groups]
        if len(candidates) != 1:
            raise ValueError("could not infer value bit width")
        v_bits = int(candidates[0] * 8 // value_group)

    if k_bits not in (2, 4) or v_bits not in (2, 4):
        raise ValueError(f"optimized CUDA path supports K/V 2 or 4 bits, got {k_bits}/{v_bits}")
    if key_group != 32:
        raise ValueError(f"optimized attention currently requires key group 32, got {key_group}")
    if head_dim % value_group != 0:
        raise ValueError("head_dim must be divisible by value_channel_group")

    packed_per_key = key_group * k_bits // 8

    if k_q.shape[3] == head_dim and k_q.shape[4] == packed_per_key:
        skb, skh, skg, skd, skp = k_q.stride()
    elif k_q.shape[3] == packed_per_key and k_q.shape[4] == head_dim:
        skb, skh, skg = k_q.stride(0), k_q.stride(1), k_q.stride(2)
        skd, skp = k_q.stride(4), k_q.stride(3)
    else:
        raise ValueError(
            f"cannot recognize K payload layout {tuple(k_q.shape)} for D={head_dim}, P={packed_per_key}"
        )

    packed_per_value = value_group * v_bits // 8
    value_groups = head_dim // value_group

    if v_q.shape[3] == value_groups and v_q.shape[4] == packed_per_value:
        svb, svh, svn, svg, svp = v_q.stride()
    elif v_q.shape[3] == packed_per_value and v_q.shape[4] == value_groups:
        svb, svh, svn = v_q.stride(0), v_q.stride(1), v_q.stride(2)
        svg, svp = v_q.stride(4), v_q.stride(3)
    else:
        raise ValueError(
            f"cannot recognize V payload layout {tuple(v_q.shape)} for VG={value_groups}, P={packed_per_value}"
        )


    total_tokens, packed_tokens, residual_tokens = _live_partition_hint(view)
    if packed_tokens is None:
        live_groups = _as_int(_getattr_any(view, ("num_key_groups", "packed_groups")))
        packed_tokens = (live_groups * key_group) if live_groups is not None else int(k_q.shape[2] * key_group)
    if residual_tokens is None:
        residual_tokens = int(k_res.shape[2])
    if total_tokens is None:
        total_tokens = packed_tokens + residual_tokens
    assert packed_tokens is not None and residual_tokens is not None and total_tokens is not None
    if packed_tokens % key_group:
        raise ValueError("packed token count is not key-group aligned")
    groups = packed_tokens // key_group
    if groups > k_q.shape[2] or packed_tokens > v_q.shape[2]:
        raise ValueError("live packed length exceeds payload capacity")
    if residual_tokens > k_res.shape[2] or residual_tokens > v_res.shape[2]:
        raise ValueError("live residual length exceeds residual capacity")


    sksb, sksh, sksg, sksd = _metadata_strides(
        k_scale, groups=groups, head_dim=head_dim, name="K scale"
    )
    zksb, zksh, zksg, zksd = _metadata_strides(
        k_zero, groups=groups, head_dim=head_dim, name="K zero"
    )
    if (sksb, sksh, sksg, sksd) != (zksb, zksh, zksg, zksd):


        raise ValueError("K scale and zero strides differ")
    svsb, svsh, svsn, svsg = _value_metadata_strides(
        v_scale, tokens=packed_tokens, value_groups=value_groups, name="V scale"
    )
    zvsb, zvsh, zvsn, zvsg = _value_metadata_strides(
        v_zero, tokens=packed_tokens, value_groups=value_groups, name="V zero"
    )
    if (svsb, svsh, svsn, svsg) != (zvsb, zvsh, zvsn, zvsg):
        raise ValueError("V scale and zero strides differ")

    return _NormalizedView(
        k_q=k_q,
        k_scale=k_scale,
        k_zero=k_zero,
        v_q=v_q,
        v_scale=v_scale,
        v_zero=v_zero,
        k_residual=k_res,
        v_residual=v_res,
        k_bits=k_bits,
        v_bits=v_bits,
        key_group=key_group,
        value_group=value_group,
        packed_tokens=packed_tokens,
        residual_tokens=residual_tokens,
        total_tokens=total_tokens,
        head_dim=head_dim,
        skb=skb,
        skh=skh,
        skg=skg,
        skd=skd,
        skp=skp,
        sksb=sksb,
        sksh=sksh,
        sksg=sksg,
        sksd=sksd,
        svb=svb,
        svh=svh,
        svn=svn,
        svg=svg,
        svp=svp,
        svsb=svsb,
        svsh=svsh,
        svsn=svsn,
        svsg=svsg,
    )


class _SplitWorkspacePool:

    def __init__(self) -> None:
        self._buffers: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def get(
        self,
        *,
        device: torch.device,
        batch: int,
        hkv: int,
        rows: int,
        splits: int,
        dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (device.type, device.index, batch, hkv, rows, splits, dim)
        value = self._buffers.get(key)
        if value is None:
            m = torch.empty((batch, hkv, rows, splits), device=device, dtype=torch.float32)
            l = torch.empty_like(m)
            o = torch.empty((batch, hkv, rows, splits, dim), device=device, dtype=torch.float32)
            value = (m, l, o)
            self._buffers[key] = value
        return value


class _SelectorWorkspacePool:

    def __init__(self) -> None:
        self._buffers: dict[
            tuple[Any, ...],
            tuple[int, torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}

    def get(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        batch: int,
        hkv: int,
        rows: int,
        tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stable = (device.type, device.index, dtype, batch, hkv, rows)
        existing = self._buffers.get(stable)
        if existing is None or existing[0] < tokens:
            capacity = max(256, _next_power_of_two(tokens))
            logits = torch.empty(
                (batch, hkv, rows, capacity), device=device, dtype=dtype
            )
            lse = torch.empty((batch, hkv, rows), device=device, dtype=torch.float32)
            importance = torch.empty(
                (batch, hkv, capacity), device=device, dtype=torch.float32
            )
            existing = (capacity, logits, lse, importance)
            self._buffers[stable] = existing
        _, logits, lse, importance = existing
        return logits[..., :tokens], lse, importance[..., :tokens]


_SPLIT_POOL = _SplitWorkspacePool()
_SELECTOR_POOL = _SelectorWorkspacePool()


def _choose_block_m(
    device: torch.device,
    *,
    rows: int,
    head_dim: int,
) -> int:
    override = _env_value("BITSIEVE_CUDA_BLOCK_M")
    if override:
        value = int(override)
        if value not in (16, 32):
            raise ValueError("BITSIEVE_CUDA_BLOCK_M must be 16 or 32")
        return value
    if rows <= 16 or head_dim >= 256:
        return 16
    profile = _device_profile(device)
    if profile.major >= 9:
        return 32


    return 32 if head_dim <= 64 and rows >= 32 else 16


def _choose_num_warps(*, block_m: int, head_dim: int) -> int:
    override = os.getenv("BITSIEVE_CUDA_NUM_WARPS")
    if override:
        value = int(override)
        if value not in (4, 8):
            raise ValueError("BITSIEVE_CUDA_NUM_WARPS must be 4 or 8")
        return value
    return 8 if block_m == 32 and head_dim <= 128 else 4


def _choose_splits(
    *, device: torch.device, batch: int, hkv: int, rows: int, block_m: int, groups: int
) -> int:
    if groups <= 1:
        return 1
    override = os.getenv("BITSIEVE_SPLIT_K")
    if override and int(override) > 0:
        requested = max(1, int(override))
    else:
        profile = _device_profile(device)
        qblocks = math.ceil(rows / block_m)
        base_ctas = max(1, batch * hkv * qblocks)
        target_waves = int(os.getenv("BITSIEVE_CUDA_TARGET_WAVES", "2"))
        target_waves = max(1, min(8, target_waves))
        requested = _next_power_of_two(
            max(1, math.ceil((profile.sm_count * target_waves) / base_ctas))
        )
    min_groups_per_split = max(
        1, int(os.getenv("BITSIEVE_CUDA_MIN_GROUPS_PER_SPLIT", "8"))
    )
    max_splits = max(1, int(os.getenv("BITSIEVE_CUDA_MAX_SPLITS", "32")))
    useful_max = max(1, groups // min_groups_per_split)
    return max(
        1,
        min(
            max_splits,
            _floor_power_of_two(min(requested, useful_max, groups)),
        ),
    )


if _TRITON_AVAILABLE:

    @triton.jit
    def _load_k_group(
        KQ,
        KS,
        KZ,
        b,
        h,
        group,
        offs_d,
        mask_d,
        stride_kb: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_kg: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_kp: tl.constexpr,
        stride_ksb: tl.constexpr,
        stride_ksh: tl.constexpr,
        stride_ksg: tl.constexpr,
        stride_ksd: tl.constexpr,
        K_BITS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):


        if K_BITS == 4:
            offs_p = tl.arange(0, 16)
            raw = tl.load(
                KQ
                + b * stride_kb
                + h * stride_kh
                + group * stride_kg
                + offs_d[:, None] * stride_kd
                + offs_p[None, :] * stride_kp,
                mask=mask_d[:, None],
                other=0,
            ).to(tl.int32)
            lo = raw & 15
            hi = (raw >> 4) & 15
            q_dp = tl.interleave(lo, hi)
            scale = tl.load(
                KS
                + b * stride_ksb
                + h * stride_ksh
                + group * stride_ksg
                + offs_d * stride_ksd,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)
            zero = tl.load(
                KZ
                + b * stride_ksb
                + h * stride_ksh
                + group * stride_ksg
                + offs_d * stride_ksd,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)
            return tl.trans(q_dp.to(tl.float32) * scale[:, None] + zero[:, None])
        else:


            offs_n = tl.arange(0, 32)
            byte = offs_n // 4
            shift = (offs_n % 4) * 2
            raw = tl.load(
                KQ
                + b * stride_kb
                + h * stride_kh
                + group * stride_kg
                + offs_d[None, :] * stride_kd
                + byte[:, None] * stride_kp,


                mask=mask_d[None, :],
                other=0,
            ).to(tl.int32)
            q_nd = (raw >> shift[:, None]) & 3
            scale = tl.load(
                KS
                + b * stride_ksb
                + h * stride_ksh
                + group * stride_ksg
                + offs_d * stride_ksd,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)
            zero = tl.load(
                KZ
                + b * stride_ksb
                + h * stride_ksh
                + group * stride_ksg
                + offs_d * stride_ksd,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)
            return q_nd.to(tl.float32) * scale[None, :] + zero[None, :]


    @triton.jit
    def _load_v_group(
        VQ,
        VS,
        VZ,
        b,
        h,
        token_base,
        offs_d,
        mask_d,
        valid_group,
        stride_vb: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_vn: tl.constexpr,
        stride_vg: tl.constexpr,
        stride_vp: tl.constexpr,
        stride_vsb: tl.constexpr,
        stride_vsh: tl.constexpr,
        stride_vsn: tl.constexpr,
        stride_vsg: tl.constexpr,
        V_BITS: tl.constexpr,
        VALUE_GROUP: tl.constexpr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        offs_n = tl.arange(0, 32)
        if V_BITS == 4:


            offs_pb = tl.arange(0, D // 2)
            packed_per_vg = VALUE_GROUP // 2
            vg_b = offs_pb // packed_per_vg
            p_b = offs_pb % packed_per_vg
            valid_payload = (offs_n[:, None] < 32) & (offs_pb[None, :] < D // 2) & valid_group
            raw = tl.load(
                VQ
                + b * stride_vb
                + h * stride_vh
                + (token_base + offs_n[:, None]) * stride_vn
                + vg_b[None, :] * stride_vg
                + p_b[None, :] * stride_vp,
                mask=valid_payload,
                other=0,
            ).to(tl.int32)
            lo = raw & 15
            hi = (raw >> 4) & 15
            q_nd = tl.interleave(lo, hi)
        else:
            byte = offs_d // 4
            shift = (offs_d % 4) * 2
            vg_d = offs_d // VALUE_GROUP
            p_d = (offs_d % VALUE_GROUP) // 4
            raw = tl.load(
                VQ
                + b * stride_vb
                + h * stride_vh
                + (token_base + offs_n[:, None]) * stride_vn
                + vg_d[None, :] * stride_vg
                + p_d[None, :] * stride_vp,
                mask=mask_d[None, :] & valid_group,
                other=0,
            ).to(tl.int32)
            q_nd = (raw >> shift[None, :]) & 3
        vg_d = offs_d // VALUE_GROUP
        scale = tl.load(
            VS
            + b * stride_vsb
            + h * stride_vsh
            + (token_base + offs_n[:, None]) * stride_vsn
            + vg_d[None, :] * stride_vsg,
            mask=(offs_n[:, None] < 32) & mask_d[None, :] & valid_group,
            other=0.0,
        ).to(tl.float32)
        zero = tl.load(
            VZ
            + b * stride_vsb
            + h * stride_vsh
            + (token_base + offs_n[:, None]) * stride_vsn
            + vg_d[None, :] * stride_vsg,
            mask=(offs_n[:, None] < 32) & mask_d[None, :] & valid_group,
            other=0.0,
        ).to(tl.float32)
        return q_nd.to(tl.float32) * scale + zero


    @triton.jit
    def _online_tile(
        q,
        k,
        v,
        m,
        l,
        acc,
        sm_scale,
        row_mask,
        key_mask,
        USE_BF16: tl.constexpr,
    ):
        if USE_BF16:
            q_dot = q.to(tl.bfloat16)
            k_dot = k.to(tl.bfloat16)
            v_dot = v.to(tl.bfloat16)
        else:
            q_dot = q.to(tl.float16)
            k_dot = k.to(tl.float16)
            v_dot = v.to(tl.float16)
        scores = tl.dot(q_dot, tl.trans(k_dot), out_dtype=tl.float32) * sm_scale
        scores = tl.where(row_mask[:, None] & key_mask[None, :], scores, -float("inf"))
        tile_m = tl.max(scores, axis=1)
        new_m = tl.maximum(m, tile_m)

        new_m = tl.where(row_mask, new_m, 0.0)
        alpha = tl.exp(m - new_m)
        p = tl.exp(scores - new_m[:, None])
        new_l = l * alpha + tl.sum(p, axis=1)
        if USE_BF16:
            p_dot = p.to(tl.bfloat16)
        else:
            p_dot = p.to(tl.float16)
        new_acc = acc * alpha[:, None] + tl.dot(p_dot, v_dot, out_dtype=tl.float32)
        return new_m, new_l, new_acc


    @triton.jit
    def _splitk_partial_kernel(
        Q,
        KQ,
        KS,
        KZ,
        VQ,
        VS,
        VZ,
        PM,
        PL,
        PO,
        stride_qb: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_qt: tl.constexpr,
        stride_qd: tl.constexpr,
        stride_kb: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_kg: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_kp: tl.constexpr,
        stride_ksb: tl.constexpr,
        stride_ksh: tl.constexpr,
        stride_ksg: tl.constexpr,
        stride_ksd: tl.constexpr,
        stride_vb: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_vn: tl.constexpr,
        stride_vg: tl.constexpr,
        stride_vp: tl.constexpr,
        stride_vsb: tl.constexpr,
        stride_vsh: tl.constexpr,
        stride_vsn: tl.constexpr,
        stride_vsg: tl.constexpr,
        stride_pmb: tl.constexpr,
        stride_pmh: tl.constexpr,
        stride_pmr: tl.constexpr,
        stride_pms: tl.constexpr,
        stride_pob: tl.constexpr,
        stride_poh: tl.constexpr,
        stride_por: tl.constexpr,
        stride_pos: tl.constexpr,
        stride_pod: tl.constexpr,
        NUM_GROUPS,
        sm_scale,
        H_KV: tl.constexpr,
        GQA: tl.constexpr,
        TQ: tl.constexpr,
        D: tl.constexpr,
        K_BITS: tl.constexpr,
        V_BITS: tl.constexpr,
        VALUE_GROUP: tl.constexpr,
        USE_BF16: tl.constexpr,
        SPLIT_K: tl.constexpr,
        MAX_GROUPS_PER_SPLIT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_q = tl.program_id(0)
        pid_s = tl.program_id(1)
        rows_per_kv = GQA * TQ
        qblocks = (rows_per_kv + BLOCK_M - 1) // BLOCK_M
        bh = pid_q // qblocks
        qblock = pid_q - bh * qblocks
        b = bh // H_KV
        hkv = bh - b * H_KV

        offs_m = qblock * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = offs_m < rows_per_kv
        qh_local = offs_m // TQ
        qt = offs_m - qh_local * TQ
        qh = hkv * GQA + qh_local
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        q = tl.load(
            Q
            + b * stride_qb
            + qh[:, None] * stride_qh
            + qt[:, None] * stride_qt
            + offs_d[None, :] * stride_qd,
            mask=row_mask[:, None] & mask_d[None, :],
            other=0.0,
        )

        start = (NUM_GROUPS * pid_s) // SPLIT_K
        end = (NUM_GROUPS * (pid_s + 1)) // SPLIT_K
        m = tl.where(row_mask, -float("inf"), 0.0).to(tl.float32)
        l = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
        key_mask = tl.arange(0, 32) < 32

        for it in tl.range(0, MAX_GROUPS_PER_SPLIT):
            group = start + it
            valid_group = group < end
            k = _load_k_group(
                KQ,
                KS,
                KZ,
                b,
                hkv,
                group,
                offs_d,
                mask_d & valid_group,
                stride_kb,
                stride_kh,
                stride_kg,
                stride_kd,
                stride_kp,
                stride_ksb,
                stride_ksh,
                stride_ksg,
                stride_ksd,
                K_BITS,
                BLOCK_D,
            )
            v = _load_v_group(
                VQ,
                VS,
                VZ,
                b,
                hkv,
                group * 32,
                offs_d,
                mask_d,
                valid_group,
                stride_vb,
                stride_vh,
                stride_vn,
                stride_vg,
                stride_vp,
                stride_vsb,
                stride_vsh,
                stride_vsn,
                stride_vsg,
                V_BITS,
                VALUE_GROUP,
                D,
                BLOCK_D,
            )
            active_keys = key_mask & valid_group
            m, l, acc = _online_tile(q, k, v, m, l, acc, sm_scale, row_mask, active_keys, USE_BF16)

        tl.store(
            PM
            + b * stride_pmb
            + hkv * stride_pmh
            + offs_m * stride_pmr
            + pid_s * stride_pms,
            m,
            mask=row_mask,
        )
        tl.store(
            PL
            + b * stride_pmb
            + hkv * stride_pmh
            + offs_m * stride_pmr
            + pid_s * stride_pms,
            l,
            mask=row_mask,
        )
        tl.store(
            PO
            + b * stride_pob
            + hkv * stride_poh
            + offs_m[:, None] * stride_por
            + pid_s * stride_pos
            + offs_d[None, :] * stride_pod,
            acc,
            mask=row_mask[:, None] & mask_d[None, :],
        )


    @triton.jit
    def _splitk_reduce_tail_kernel(
        Q,
        PM,
        PL,
        PO,
        KR,
        VR,
        CK,
        CV,
        OUT,
        stride_qb: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_qt: tl.constexpr,
        stride_qd: tl.constexpr,
        stride_pmb: tl.constexpr,
        stride_pmh: tl.constexpr,
        stride_pmr: tl.constexpr,
        stride_pms: tl.constexpr,
        stride_pob: tl.constexpr,
        stride_poh: tl.constexpr,
        stride_por: tl.constexpr,
        stride_pos: tl.constexpr,
        stride_pod: tl.constexpr,
        stride_krb: tl.constexpr,
        stride_krh: tl.constexpr,
        stride_krn: tl.constexpr,
        stride_krd: tl.constexpr,
        stride_vrb: tl.constexpr,
        stride_vrh: tl.constexpr,
        stride_vrn: tl.constexpr,
        stride_vrd: tl.constexpr,
        stride_ckb: tl.constexpr,
        stride_ckh: tl.constexpr,
        stride_ckn: tl.constexpr,
        stride_ckd: tl.constexpr,
        stride_cvb: tl.constexpr,
        stride_cvh: tl.constexpr,
        stride_cvn: tl.constexpr,
        stride_cvd: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_oh: tl.constexpr,
        stride_ot: tl.constexpr,
        stride_od: tl.constexpr,
        RESIDUAL_LEN,
        CURRENT_LEN,
        sm_scale,
        H_KV: tl.constexpr,
        GQA: tl.constexpr,
        TQ: tl.constexpr,
        D: tl.constexpr,
        SPLIT_K: tl.constexpr,
        USE_BF16: tl.constexpr,
        MAX_RES_TILES: tl.constexpr,
        MAX_CUR_TILES: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_q = tl.program_id(0)
        rows_per_kv = GQA * TQ
        qblocks = (rows_per_kv + BLOCK_M - 1) // BLOCK_M
        bh = pid_q // qblocks
        qblock = pid_q - bh * qblocks
        b = bh // H_KV
        hkv = bh - b * H_KV

        offs_m = qblock * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = offs_m < rows_per_kv
        qh_local = offs_m // TQ
        qt = offs_m - qh_local * TQ
        qh = hkv * GQA + qh_local
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        q = tl.load(
            Q
            + b * stride_qb
            + qh[:, None] * stride_qh
            + qt[:, None] * stride_qt
            + offs_d[None, :] * stride_qd,
            mask=row_mask[:, None] & mask_d[None, :],
            other=0.0,
        )

        offs_s = tl.arange(0, SPLIT_K)
        ms = tl.load(
            PM
            + b * stride_pmb
            + hkv * stride_pmh
            + offs_m[:, None] * stride_pmr
            + offs_s[None, :] * stride_pms,
            mask=row_mask[:, None],
            other=-float("inf"),
        ).to(tl.float32)
        ls = tl.load(
            PL
            + b * stride_pmb
            + hkv * stride_pmh
            + offs_m[:, None] * stride_pmr
            + offs_s[None, :] * stride_pms,
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        m = tl.max(ms, axis=1)
        m = tl.where(row_mask, m, 0.0)
        split_weight = tl.exp(ms - m[:, None])
        l = tl.sum(split_weight * ls, axis=1)
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
        for s in range(0, SPLIT_K):
            part = tl.load(
                PO
                + b * stride_pob
                + hkv * stride_poh
                + offs_m[:, None] * stride_por
                + s * stride_pos
                + offs_d[None, :] * stride_pod,
                mask=row_mask[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            ms_s = tl.load(
                PM
                + b * stride_pmb
                + hkv * stride_pmh
                + offs_m * stride_pmr
                + s * stride_pms,
                mask=row_mask,
                other=-float("inf"),
            ).to(tl.float32)
            weight_s = tl.exp(ms_s - m)
            acc += part * weight_s[:, None]

        offs_n = tl.arange(0, 32)
        for rt in range(0, MAX_RES_TILES):
            n = rt * 32 + offs_n
            key_mask = n < RESIDUAL_LEN
            k = tl.load(
                KR
                + b * stride_krb
                + hkv * stride_krh
                + n[:, None] * stride_krn
                + offs_d[None, :] * stride_krd,
                mask=key_mask[:, None] & mask_d[None, :],
                other=0.0,
            )
            v = tl.load(
                VR
                + b * stride_vrb
                + hkv * stride_vrh
                + n[:, None] * stride_vrn
                + offs_d[None, :] * stride_vrd,
                mask=key_mask[:, None] & mask_d[None, :],
                other=0.0,
            )
            m, l, acc = _online_tile(q, k, v, m, l, acc, sm_scale, row_mask, key_mask, USE_BF16)

        for ct in range(0, MAX_CUR_TILES):
            n = ct * 32 + offs_n
            key_mask = n < CURRENT_LEN
            k = tl.load(
                CK
                + b * stride_ckb
                + hkv * stride_ckh
                + n[:, None] * stride_ckn
                + offs_d[None, :] * stride_ckd,
                mask=key_mask[:, None] & mask_d[None, :],
                other=0.0,
            )
            v = tl.load(
                CV
                + b * stride_cvb
                + hkv * stride_cvh
                + n[:, None] * stride_cvn
                + offs_d[None, :] * stride_cvd,
                mask=key_mask[:, None] & mask_d[None, :],
                other=0.0,
            )
            m, l, acc = _online_tile(q, k, v, m, l, acc, sm_scale, row_mask, key_mask, USE_BF16)

        out = acc / tl.maximum(l[:, None], 1.0e-20)
        tl.store(
            OUT
            + b * stride_ob
            + qh[:, None] * stride_oh
            + qt[:, None] * stride_ot
            + offs_d[None, :] * stride_od,
            out,
            mask=row_mask[:, None] & mask_d[None, :],
        )


    @triton.jit
    def _selector_qk_tiled_kernel(
        QREP,
        KQ,
        KS,
        KZ,
        LOGITS,
        stride_qb: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_qr: tl.constexpr,
        stride_qd: tl.constexpr,
        stride_kb: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_kg: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_kp: tl.constexpr,
        stride_ksb: tl.constexpr,
        stride_ksh: tl.constexpr,
        stride_ksg: tl.constexpr,
        stride_ksd: tl.constexpr,
        stride_lb: tl.constexpr,
        stride_lh: tl.constexpr,
        stride_lr: tl.constexpr,
        stride_ln: tl.constexpr,
        sm_scale,
        H_KV: tl.constexpr,
        R: tl.constexpr,
        D: tl.constexpr,
        K_BITS: tl.constexpr,
        USE_BF16: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        bh = tl.program_id(0)
        group = tl.program_id(1)
        qblock = tl.program_id(2)
        b = bh // H_KV
        h = bh - b * H_KV
        offs_r = qblock * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = offs_r < R
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        q = tl.load(
            QREP
            + b * stride_qb
            + h * stride_qh
            + offs_r[:, None] * stride_qr
            + offs_d[None, :] * stride_qd,
            mask=row_mask[:, None] & mask_d[None, :],
            other=0.0,
        )
        k = _load_k_group(
            KQ,
            KS,
            KZ,
            b,
            h,
            group,
            offs_d,
            mask_d,
            stride_kb,
            stride_kh,
            stride_kg,
            stride_kd,
            stride_kp,
            stride_ksb,
            stride_ksh,
            stride_ksg,
            stride_ksd,
            K_BITS,
            BLOCK_D,
        )
        if USE_BF16:
            q_dot = q.to(tl.bfloat16)
            k_dot = k.to(tl.bfloat16)
        else:
            q_dot = q.to(tl.float16)
            k_dot = k.to(tl.float16)
        scores = tl.dot(q_dot, tl.trans(k_dot), out_dtype=tl.float32) * sm_scale
        offs_n = group * 32 + tl.arange(0, 32)
        tl.store(
            LOGITS
            + b * stride_lb
            + h * stride_lh
            + offs_r[:, None] * stride_lr
            + offs_n[None, :] * stride_ln,
            scores,
            mask=row_mask[:, None],
        )


    @triton.jit
    def _gather_kv_kernel(
        INDICES,
        KQ,
        KS,
        KZ,
        VQ,
        VS,
        VZ,
        KR,
        VR,
        OUTK,
        OUTV,
        stride_ib: tl.constexpr,
        stride_ih: tl.constexpr,
        stride_ik: tl.constexpr,
        stride_kb: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_kg: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_kp: tl.constexpr,
        stride_ksb: tl.constexpr,
        stride_ksh: tl.constexpr,
        stride_ksg: tl.constexpr,
        stride_ksd: tl.constexpr,
        stride_vb: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_vn: tl.constexpr,
        stride_vg: tl.constexpr,
        stride_vp: tl.constexpr,
        stride_vsb: tl.constexpr,
        stride_vsh: tl.constexpr,
        stride_vsn: tl.constexpr,
        stride_vsg: tl.constexpr,
        stride_krb: tl.constexpr,
        stride_krh: tl.constexpr,
        stride_krn: tl.constexpr,
        stride_krd: tl.constexpr,
        stride_vrb: tl.constexpr,
        stride_vrh: tl.constexpr,
        stride_vrn: tl.constexpr,
        stride_vrd: tl.constexpr,
        stride_okb: tl.constexpr,
        stride_okh: tl.constexpr,
        stride_okn: tl.constexpr,
        stride_okd: tl.constexpr,
        stride_ovb: tl.constexpr,
        stride_ovh: tl.constexpr,
        stride_ovn: tl.constexpr,
        stride_ovd: tl.constexpr,
        K_SELECTED,
        PACKED_TOKENS,
        H_KV: tl.constexpr,
        D: tl.constexpr,
        K_BITS: tl.constexpr,
        V_BITS: tl.constexpr,
        KEY_GROUP: tl.constexpr,
        VALUE_GROUP: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        bh = tl.program_id(0)
        kb = tl.program_id(1)
        db = tl.program_id(2)
        b = bh // H_KV
        h = bh - b * H_KV
        offs_i = kb * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_d = db * BLOCK_D + tl.arange(0, BLOCK_D)
        valid_i = offs_i < K_SELECTED
        valid_d = offs_d < D
        token = tl.load(
            INDICES + b * stride_ib + h * stride_ih + offs_i * stride_ik,
            mask=valid_i,
            other=0,
        ).to(tl.int64)
        is_packed = token < PACKED_TOKENS

        k_pack = 8 // K_BITS
        k_group = token // KEY_GROUP
        k_within = token - k_group * KEY_GROUP
        k_byte = k_within // k_pack
        k_shift = (k_within % k_pack) * K_BITS
        k_raw = tl.load(
            KQ
            + b * stride_kb
            + h * stride_kh
            + k_group[:, None] * stride_kg
            + offs_d[None, :] * stride_kd
            + k_byte[:, None] * stride_kp,
            mask=valid_i[:, None] & valid_d[None, :] & is_packed[:, None],
            other=0,
        ).to(tl.int32)
        k_q = (k_raw >> k_shift[:, None]) & ((1 << K_BITS) - 1)
        k_scale = tl.load(
            KS
            + b * stride_ksb
            + h * stride_ksh
            + k_group[:, None] * stride_ksg
            + offs_d[None, :] * stride_ksd,
            mask=valid_i[:, None] & valid_d[None, :] & is_packed[:, None],
            other=0.0,
        ).to(tl.float32)
        k_zero = tl.load(
            KZ
            + b * stride_ksb
            + h * stride_ksh
            + k_group[:, None] * stride_ksg
            + offs_d[None, :] * stride_ksd,
            mask=valid_i[:, None] & valid_d[None, :] & is_packed[:, None],
            other=0.0,
        ).to(tl.float32)
        k_deq = k_q.to(tl.float32) * k_scale + k_zero

        v_pack = 8 // V_BITS
        v_group = offs_d // VALUE_GROUP
        v_byte = (offs_d % VALUE_GROUP) // v_pack
        v_shift = ((offs_d % VALUE_GROUP) % v_pack) * V_BITS
        v_raw = tl.load(
            VQ
            + b * stride_vb
            + h * stride_vh
            + token[:, None] * stride_vn
            + v_group[None, :] * stride_vg
            + v_byte[None, :] * stride_vp,
            mask=valid_i[:, None] & valid_d[None, :] & is_packed[:, None],
            other=0,
        ).to(tl.int32)
        v_q = (v_raw >> v_shift[None, :]) & ((1 << V_BITS) - 1)
        v_scale = tl.load(
            VS
            + b * stride_vsb
            + h * stride_vsh
            + token[:, None] * stride_vsn
            + v_group[None, :] * stride_vsg,
            mask=valid_i[:, None] & valid_d[None, :] & is_packed[:, None],
            other=0.0,
        ).to(tl.float32)
        v_zero = tl.load(
            VZ
            + b * stride_vsb
            + h * stride_vsh
            + token[:, None] * stride_vsn
            + v_group[None, :] * stride_vsg,
            mask=valid_i[:, None] & valid_d[None, :] & is_packed[:, None],
            other=0.0,
        ).to(tl.float32)
        v_deq = v_q.to(tl.float32) * v_scale + v_zero

        rindex = token - PACKED_TOKENS
        k_recent = tl.load(
            KR
            + b * stride_krb
            + h * stride_krh
            + rindex[:, None] * stride_krn
            + offs_d[None, :] * stride_krd,
            mask=valid_i[:, None] & valid_d[None, :] & (~is_packed[:, None]),
            other=0.0,
        )
        v_recent = tl.load(
            VR
            + b * stride_vrb
            + h * stride_vrh
            + rindex[:, None] * stride_vrn
            + offs_d[None, :] * stride_vrd,
            mask=valid_i[:, None] & valid_d[None, :] & (~is_packed[:, None]),
            other=0.0,
        )
        k_out = tl.where(is_packed[:, None], k_deq, k_recent)
        v_out = tl.where(is_packed[:, None], v_deq, v_recent)
        mask = valid_i[:, None] & valid_d[None, :]
        tl.store(
            OUTK
            + b * stride_okb
            + h * stride_okh
            + offs_i[:, None] * stride_okn
            + offs_d[None, :] * stride_okd,
            k_out,
            mask=mask,
        )
        tl.store(
            OUTV
            + b * stride_ovb
            + h * stride_ovh
            + offs_i[:, None] * stride_ovn
            + offs_d[None, :] * stride_ovd,
            v_out,
            mask=mask,
        )


    @triton.jit
    def _selector_fp_qk_tiled_kernel(
        QREP,
        K,
        LOGITS,
        stride_qb: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_qr: tl.constexpr,
        stride_qd: tl.constexpr,
        stride_kb: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_kn: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_lb: tl.constexpr,
        stride_lh: tl.constexpr,
        stride_lr: tl.constexpr,
        stride_ln: tl.constexpr,
        N,
        sm_scale,
        H_KV: tl.constexpr,
        R: tl.constexpr,
        D: tl.constexpr,
        USE_BF16: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        bh = tl.program_id(0)
        nblock = tl.program_id(1)
        qblock = tl.program_id(2)
        b = bh // H_KV
        h = bh - b * H_KV
        offs_r = qblock * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = nblock * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        mask_r = offs_r < R
        mask_n = offs_n < N
        mask_d = offs_d < D
        q = tl.load(
            QREP
            + b * stride_qb
            + h * stride_qh
            + offs_r[:, None] * stride_qr
            + offs_d[None, :] * stride_qd,
            mask=mask_r[:, None] & mask_d[None, :],
            other=0.0,
        )
        k = tl.load(
            K
            + b * stride_kb
            + h * stride_kh
            + offs_n[:, None] * stride_kn
            + offs_d[None, :] * stride_kd,
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
        )
        if USE_BF16:
            q = q.to(tl.bfloat16)
            k = k.to(tl.bfloat16)
        else:
            q = q.to(tl.float16)
            k = k.to(tl.float16)
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale
        tl.store(
            LOGITS
            + b * stride_lb
            + h * stride_lh
            + offs_r[:, None] * stride_lr
            + offs_n[None, :] * stride_ln,
            scores,
            mask=mask_r[:, None] & mask_n[None, :],
        )


    @triton.jit
    def _row_lse_kernel(
        LOGITS,
        LSE,
        N,
        stride_lb: tl.constexpr,
        stride_lh: tl.constexpr,
        stride_lr: tl.constexpr,
        stride_ln: tl.constexpr,
        stride_eb: tl.constexpr,
        stride_eh: tl.constexpr,
        stride_er: tl.constexpr,
        H_KV: tl.constexpr,
        R: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // (H_KV * R)
        rem = pid - b * H_KV * R
        h = rem // R
        r = rem - h * R
        m_i = -float("inf")
        l_i = 0.0
        for start in tl.range(0, N, BLOCK_N):
            offs = start + tl.arange(0, BLOCK_N)
            x = tl.load(
                LOGITS
                + b * stride_lb
                + h * stride_lh
                + r * stride_lr
                + offs * stride_ln,
                mask=offs < N,
                other=-float("inf"),
            ).to(tl.float32)
            m_new = tl.maximum(m_i, tl.max(x, axis=0))
            l_i = l_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
            m_i = m_new
        tl.store(
            LSE + b * stride_eb + h * stride_eh + r * stride_er,
            m_i + tl.log(l_i),
        )


    @triton.jit
    def _importance_kernel(
        LOGITS,
        LSE,
        OUT,
        stride_lb: tl.constexpr,
        stride_lh: tl.constexpr,
        stride_lr: tl.constexpr,
        stride_ln: tl.constexpr,
        stride_lseb: tl.constexpr,
        stride_lseh: tl.constexpr,
        stride_lser: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_oh: tl.constexpr,
        stride_on: tl.constexpr,
        N,
        H_KV: tl.constexpr,
        R: tl.constexpr,
        USE_SOFTMAX: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        bh = tl.program_id(0)
        pid_n = tl.program_id(1)
        b = bh // H_KV
        h = bh - b * H_KV
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        acc = tl.zeros((BLOCK_N,), tl.float32)
        for r in tl.range(0, R):
            x = tl.load(
                LOGITS
                + b * stride_lb
                + h * stride_lh
                + r * stride_lr
                + offs_n * stride_ln,
                mask=mask_n,
                other=-float("inf") if USE_SOFTMAX else 0.0,
            ).to(tl.float32)
            if USE_SOFTMAX:
                z = tl.load(LSE + b * stride_lseb + h * stride_lseh + r * stride_lser)
                acc += tl.exp(x - z)
            else:
                acc += x
        acc /= R
        tl.store(
            OUT + b * stride_ob + h * stride_oh + offs_n * stride_on,
            acc,
            mask=mask_n,
        )


def _declares_lowbit(view: Any) -> bool:
    bits = _as_int(_getattr_any(view, ("k_bits", "key_bits")))
    payload = _tensor_attr(
        view,
        ("k_q", "key_q", "key_payload", "packed_k", "packed_key", "key_packed"),
    )
    return bits in (2, 4) or (isinstance(payload, torch.Tensor) and payload.dtype == torch.uint8)


def _strict_should_raise(view: Any, reason: str) -> bool:


    if "no packed prefix" in reason or "no packed prefix groups" in reason:
        return False
    return _strict() and _declares_lowbit(view)

def _can_fast(query: torch.Tensor, view: Any, current_key: torch.Tensor | None = None) -> tuple[bool, str]:
    if not _enabled():
        return False, "BITSIEVE_CUDA_FAST is disabled"
    if not _TRITON_AVAILABLE:
        return False, "Triton is unavailable"
    if not isinstance(query, torch.Tensor) or query.device.type != "cuda":
        return False, "query is not a CUDA tensor"
    if not _supports_recent_cuda(query.device, query.dtype):
        return False, "optimized path requires SM80+ (and Triton support for this GPU)"
    if query.ndim != 4:
        return False, f"query rank is {query.ndim}, expected 4"
    if query.dtype not in (torch.float16, torch.bfloat16):
        return False, f"unsupported query dtype {query.dtype}"
    if query.shape[-1] not in (64, 128, 256) or (query.shape[-1] & (query.shape[-1] - 1)):
        return False, f"unsupported head dimension {query.shape[-1]}"
    if current_key is not None and current_key.ndim != 4:
        return False, "current key is not rank 4"
    try:
        _total_hint, packed_hint, _residual_hint = _live_partition_hint(view)
    except Exception as exc:
        return False, str(exc)
    if packed_hint == 0:
        return False, "no packed prefix groups"
    try:
        nv = _normalize_view(view, head_dim=int(query.shape[-1]))
    except Exception as exc:
        return False, str(exc)
    if nv.packed_tokens <= 0:
        return False, "no packed prefix groups"
    if query.shape[0] != nv.k_q.shape[0]:
        return False, "batch dimension mismatch"
    return True, ""


def _can_fast_fp_selector(query: torch.Tensor, view: Any) -> tuple[bool, str]:
    if not _enabled() or not _TRITON_AVAILABLE:
        return False, "overlay or Triton unavailable"
    if not isinstance(query, torch.Tensor) or query.device.type != "cuda":
        return False, "query is not CUDA"
    if not _supports_recent_cuda(query.device, query.dtype):
        return False, "SM80+ and Triton support are required"
    if query.ndim != 4 or query.dtype not in (torch.float16, torch.bfloat16):
        return False, "unsupported query"
    try:
        fp = _normalize_fp_key_view(view, head_dim=int(query.shape[-1]))
    except Exception as exc:
        return False, str(exc)
    if fp.total_tokens <= 0:
        return False, "empty full-precision prefix"
    return True, ""

def _fast_dense_impl(
    query: torch.Tensor,
    view: Any,
    current_key: torch.Tensor,
    current_value: torch.Tensor,
    *,
    scaling: float | None,
) -> torch.Tensor:
    nv = _normalize_view(view, head_dim=int(query.shape[-1]))
    b, hq, tq, d = map(int, query.shape)
    hkv = int(current_key.shape[1])
    if hq % hkv:
        raise ValueError(f"Hq={hq} is not divisible by Hkv={hkv}")
    gqa = hq // hkv
    if current_key.shape != current_value.shape:
        raise ValueError("current K/V shape mismatch")
    if current_key.shape[0] != b or current_key.shape[-1] != d:
        raise ValueError("current K shape is incompatible with Q")
    if int(nv.k_q.shape[1]) != hkv:
        raise ValueError("packed cache Hkv differs from current K")

    rows = gqa * tq
    block_m = _choose_block_m(query.device, rows=rows, head_dim=d)
    block_d = d
    groups = nv.packed_tokens // 32
    splits = _choose_splits(
        device=query.device,
        batch=b,
        hkv=hkv,
        rows=rows,
        block_m=block_m,
        groups=groups,
    )
    max_groups_per_split = math.ceil(groups / splits)
    pm, pl, po = _SPLIT_POOL.get(
        device=query.device,
        batch=b,
        hkv=hkv,
        rows=rows,
        splits=splits,
        dim=d,
    )
    out = torch.empty_like(query)
    qblocks = triton.cdiv(rows, block_m)
    warps = _choose_num_warps(block_m=block_m, head_dim=d)
    sm_scale = float(scaling if scaling is not None else d**-0.5)

    _log_once(
        f"dense:{b}:{hq}:{tq}:{d}:{nv.k_bits}:{nv.v_bits}:{groups}:{splits}:{block_m}",
        f"dense fast path B={b} Hq/Hkv={hq}/{hkv} Tq={tq} D={d} "
        f"K{nv.k_bits}/V{nv.v_bits} groups={groups} splits={splits} BLOCK_M={block_m}",
    )
    _splitk_partial_kernel[(b * hkv * qblocks, splits)](
        query,
        nv.k_q,
        nv.k_scale,
        nv.k_zero,
        nv.v_q,
        nv.v_scale,
        nv.v_zero,
        pm,
        pl,
        po,
        *query.stride(),
        nv.skb,
        nv.skh,
        nv.skg,
        nv.skd,
        nv.skp,
        nv.sksb,
        nv.sksh,
        nv.sksg,
        nv.sksd,
        nv.svb,
        nv.svh,
        nv.svn,
        nv.svg,
        nv.svp,
        nv.svsb,
        nv.svsh,
        nv.svsn,
        nv.svsg,
        *pm.stride(),
        *po.stride(),
        groups,
        sm_scale,
        H_KV=hkv,
        GQA=gqa,
        TQ=tq,
        D=d,
        K_BITS=nv.k_bits,
        V_BITS=nv.v_bits,
        VALUE_GROUP=nv.value_group,
        USE_BF16=query.dtype == torch.bfloat16,
        SPLIT_K=splits,
        MAX_GROUPS_PER_SPLIT=max_groups_per_split,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        num_warps=warps,
        num_stages=2,
    )

    kr = nv.k_residual
    vr = nv.v_residual
    residual_len = nv.residual_tokens


    if kr.numel() == 0 or vr.numel() == 0:
        kr = current_key[:, :, :1, :]
        vr = current_value[:, :, :1, :]
    current_len = int(current_key.shape[2])
    max_res_tiles = max(1, math.ceil(max(1, int(kr.shape[2])) / 32))
    max_cur_tiles = max(1, math.ceil(max(1, current_len) / 32))
    _splitk_reduce_tail_kernel[(b * hkv * qblocks,)](
        query,
        pm,
        pl,
        po,
        kr,
        vr,
        current_key,
        current_value,
        out,
        *query.stride(),
        *pm.stride(),
        *po.stride(),
        *kr.stride(),
        *vr.stride(),
        *current_key.stride(),
        *current_value.stride(),
        *out.stride(),
        residual_len,
        current_len,
        sm_scale,
        H_KV=hkv,
        GQA=gqa,
        TQ=tq,
        D=d,
        SPLIT_K=splits,
        USE_BF16=query.dtype == torch.bfloat16,
        MAX_RES_TILES=max_res_tiles,
        MAX_CUR_TILES=max_cur_tiles,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        num_warps=warps,
        num_stages=2,
    )
    return out


def _bind_arguments(fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        return dict(inspect.signature(fn).bind_partial(*args, **kwargs).arguments)
    except Exception:
        result = dict(kwargs)
        for i, value in enumerate(args):
            result[f"_arg{i}"] = value
        return result


def _pick_named(mapping: Mapping[str, Any], names: Sequence[str]) -> Any:
    lowered = {str(k).lower(): v for k, v in mapping.items()}
    for name in names:
        if name in mapping:
            return mapping[name]
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _parse_dense_call(
    original: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[torch.Tensor, Any, torch.Tensor, torch.Tensor, float | None, str, str] | None:
    bound = _bind_arguments(original, args, kwargs)
    query = _pick_named(bound, ("query", "q"))
    view = _pick_named(bound, ("cache", "cache_view", "layer_view", "view", "old_cache"))
    ck = _pick_named(bound, ("current_key", "current_k", "k_current", "k_cur"))
    cv = _pick_named(bound, ("current_value", "current_v", "v_current", "v_cur"))
    if query is None and len(args) >= 1:
        query = args[0]
    if view is None and len(args) >= 2:
        view = args[1]
    if ck is None and len(args) >= 3:
        ck = args[2]
    if cv is None and len(args) >= 4:
        cv = args[3]
    if not all(isinstance(x, torch.Tensor) for x in (query, ck, cv)) or view is None:
        return None
    scaling = _pick_named(bound, ("scaling", "scale", "sm_scale"))
    backend = str(_pick_named(bound, ("backend",)) or "auto")
    variant = str(_pick_named(bound, ("kernel_variant", "variant")) or "blocked")
    return query, view, ck, cv, (None if scaling is None else float(scaling)), backend, variant


def _repeat_kv_for_gqa(x: torch.Tensor, hq: int) -> torch.Tensor:

    hkv = int(x.shape[1])
    if hq == hkv:
        return x
    if hq % hkv:
        raise ValueError(f"query heads ({hq}) are not divisible by KV heads ({hkv})")
    repeat = hq // hkv
    return (
        x[:, :, None, :, :]
        .expand(x.shape[0], hkv, repeat, x.shape[2], x.shape[3])
        .reshape(x.shape[0], hq, x.shape[2], x.shape[3])
    )


def _residual_only_dense_attention(
    query: torch.Tensor,
    view: Any,
    current_key: torch.Tensor,
    current_value: torch.Tensor,
    *,
    scaling: float | None,
) -> torch.Tensor:

    total, packed, residual = _live_partition_hint(view)
    if packed not in (None, 0):
        raise ValueError("residual-only attention was called with packed prefix data")

    k_bits = int(getattr(view, "k_bits", 16))
    v_bits = int(getattr(view, "v_bits", 16))
    old_k = _tensor_attr(
        view,
        ("k_fp", "key_fp", "full_key", "key_full")
        if k_bits == 16
        else ("k_residual", "key_residual", "residual_k", "residual_key", "k_fp_residual"),
    )
    old_v = _tensor_attr(
        view,
        ("v_fp", "value_fp", "full_value", "value_full")
        if v_bits == 16
        else ("v_residual", "value_residual", "residual_v", "residual_value", "v_fp_residual"),
    )
    if old_k is None or old_v is None or old_k.ndim != 4 or old_v.ndim != 4:
        fields = {
            name: None if getattr(view, name, None) is None else tuple(getattr(view, name).shape)
            for name in ("k_fp", "v_fp", "k_residual", "v_residual")
        }
        raise ValueError(
            "zero-packed cache is missing a compatible rank-4 K/V pair; "
            f"k_bits={k_bits}, v_bits={v_bits}, fields={fields}"
        )


    k_live = int(total if k_bits == 16 and total is not None else (residual or old_k.shape[2]))
    v_live = int(total if v_bits == 16 and total is not None else (residual or old_v.shape[2]))
    if k_live != v_live:
        raise ValueError(f"zero-packed K/V live lengths differ: K={k_live}, V={v_live}")
    live = k_live
    if live < 0 or live > old_k.shape[2] or live > old_v.shape[2]:
        raise ValueError("invalid zero-packed cache length")

    old_k = old_k[:, :, :live, :]
    old_v = old_v[:, :, :live, :]
    key = current_key if live == 0 else torch.cat((old_k, current_key), dim=2)
    value = current_value if live == 0 else torch.cat((old_v, current_value), dim=2)

    sdpa_kwargs: dict[str, Any] = {
        "attn_mask": None,
        "dropout_p": 0.0,
        "is_causal": False,
    }
    if scaling is not None:
        sdpa_kwargs["scale"] = float(scaling)
    try:
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            enable_gqa=query.shape[1] != key.shape[1],
            **sdpa_kwargs,
        )
    except TypeError:

        sdpa_kwargs.pop("scale", None)
        q = query if scaling is None else query * (float(scaling) * query.shape[-1] ** 0.5)
        return F.scaled_dot_product_attention(
            q,
            _repeat_kv_for_gqa(key, query.shape[1]),
            _repeat_kv_for_gqa(value, query.shape[1]),
            **sdpa_kwargs,
        )


def maybe_fast_dense_packed_attention(
    original: Callable[..., torch.Tensor], *args: Any, **kwargs: Any
) -> torch.Tensor:
    parsed = _parse_dense_call(original, args, kwargs)
    if parsed is None:
        return original(*args, **kwargs)
    query, view, ck, cv, scaling, backend, variant = parsed
    requested = backend.lower() in {"auto", "triton"} and variant.lower() in {
        "blocked",
        "cuda",
        "cuda_fast",
        "portable",
        "splitk",

        "hopper",
        "hopper_fast",
    }
    if not requested:
        return original(*args, **kwargs)


    if int(getattr(view, "k_bits", 16)) == 16 and int(getattr(view, "v_bits", 16)) == 16:
        return original(*args, **kwargs)

    try:
        _total_hint, packed_hint, _residual_hint = _live_partition_hint(view)
    except Exception as exc:
        if _strict_should_raise(view, str(exc)):
            raise RuntimeError(f"CUDA packed-attention fast path unavailable: {exc}") from exc
        return original(*args, **kwargs)
    if packed_hint == 0:
        _log_once(
            "dense:residual-only",
            "using native SDPA for a low-bit cache with zero packed prefix groups",
        )
        return _residual_only_dense_attention(
            query,
            view,
            ck,
            cv,
            scaling=scaling,
        )
    ok, reason = _can_fast(query, view, ck)
    if not ok:
        if _strict_should_raise(view, reason):
            raise RuntimeError(f"CUDA packed-attention fast path unavailable: {reason}")
        return original(*args, **kwargs)
    return _fast_dense_impl(query, view, ck, cv, scaling=scaling)


def _parse_gather_call(
    original: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any] | None:
    bound = _bind_arguments(original, args, kwargs)
    view = _pick_named(bound, ("cache", "cache_view", "layer_view", "view", "old_cache"))
    indices = _pick_named(bound, ("indices", "selected_indices", "topk_indices", "keep_indices"))
    if view is None:
        for value in args:
            if not isinstance(value, torch.Tensor):
                try:

                    if _tensor_attr(value, ("k_q", "key_payload", "packed_k")) is not None:
                        view = value
                        break
                except Exception:
                    pass
    if indices is None:
        for value in args:
            if isinstance(value, torch.Tensor) and value.ndim == 3 and value.dtype in (
                torch.int32,
                torch.int64,
            ):
                indices = value
                break
    if view is None or not isinstance(indices, torch.Tensor):
        return None
    dtype_value = _pick_named(bound, ("dtype", "output_dtype", "compute_dtype"))
    if isinstance(dtype_value, torch.dtype):
        dtype = dtype_value
    else:
        residual = _tensor_attr(view, ("k_residual", "key_residual", "residual_k"))
        dtype = residual.dtype if residual is not None else torch.bfloat16
    out_key = _pick_named(bound, ("out_key", "output_key", "key_out"))
    out_value = _pick_named(bound, ("out_value", "output_value", "value_out"))
    if out_key is not None and not isinstance(out_key, torch.Tensor):
        out_key = None
    if out_value is not None and not isinstance(out_value, torch.Tensor):
        out_value = None
    if isinstance(out_key, torch.Tensor):
        dtype = out_key.dtype
    backend = str(_pick_named(bound, ("backend",)) or "auto")
    return {
        "view": view,
        "indices": indices,
        "dtype": dtype,
        "backend": backend,
        "out_key": out_key,
        "out_value": out_value,
    }


def _make_gather_result(
    original: Callable[..., Any], key: torch.Tensor, value: torch.Tensor, style: str
) -> Any:
    if style == "tuple":
        return key, value
    for obj in original.__globals__.values():
        if not inspect.isclass(obj):
            continue
        annotations = getattr(obj, "__annotations__", {})
        for kname, vname in (("key", "value"), ("keys", "values"), ("k", "v")):
            if kname in annotations and vname in annotations:
                try:
                    return obj(**{kname: key, vname: value})
                except Exception:
                    pass
    return SimpleNamespace(key=key, value=value, keys=key, values=value, k=key, v=value)


def _fast_gather_impl(
    *,
    view: Any,
    indices: torch.Tensor,
    dtype: torch.dtype,
    original: Callable[..., Any],
    result_style: str,
    out_key: torch.Tensor | None = None,
    out_value: torch.Tensor | None = None,
) -> Any:
    d = _infer_gather_head_dim(
        view,
        out_key=out_key,
        out_value=out_value,
    )
    nv = _normalize_view(view, head_dim=d)
    if indices.device.type != "cuda" or not _supports_recent_cuda(indices.device):
        raise ValueError("supported CUDA indices required")
    b, hkv, k = map(int, indices.shape)
    expected = (b, hkv, k, d)
    if out_key is None:
        outk = torch.empty(expected, device=indices.device, dtype=dtype)
    else:
        if tuple(out_key.shape) != expected or out_key.device != indices.device:
            raise ValueError(f"out_key must have shape/device {expected}/{indices.device}")
        outk = out_key
    if out_value is None:
        outv = torch.empty_like(outk)
    else:
        if tuple(out_value.shape) != expected or out_value.device != indices.device:
            raise ValueError(f"out_value must have shape/device {expected}/{indices.device}")
        if out_value.dtype != outk.dtype:
            raise ValueError("out_key and out_value dtypes must match")
        outv = out_value

    if nv.packed_tokens == 0:


        idx = indices if indices.dtype == torch.long else indices.to(torch.long)
        gather_idx = idx.unsqueeze(-1).expand(b, hkv, k, d)
        live_k = nv.k_residual[:, :, : nv.residual_tokens, :]
        live_v = nv.v_residual[:, :, : nv.residual_tokens, :]
        outk.copy_(torch.gather(live_k, 2, gather_idx))
        outv.copy_(torch.gather(live_v, 2, gather_idx))
        return _make_gather_result(original, outk, outv, result_style)

    kr, vr = nv.k_residual, nv.v_residual
    if kr.numel() == 0 or vr.numel() == 0:

        kr = outk[:, :, :1, :]
        vr = outv[:, :, :1, :]
    block_k = 16
    block_d = 128 if d >= 128 else 64
    _log_once(
        f"gather:{b}:{hkv}:{k}:{d}:{nv.k_bits}:{nv.v_bits}",
        f"gather fast path B={b} Hkv={hkv} k={k} D={d} K{nv.k_bits}/V{nv.v_bits}",
    )
    _gather_kv_kernel[
        (b * hkv, triton.cdiv(k, block_k), triton.cdiv(d, block_d))
    ](
        indices,
        nv.k_q,
        nv.k_scale,
        nv.k_zero,
        nv.v_q,
        nv.v_scale,
        nv.v_zero,
        kr,
        vr,
        outk,
        outv,
        *indices.stride(),
        nv.skb,
        nv.skh,
        nv.skg,
        nv.skd,
        nv.skp,
        nv.sksb,
        nv.sksh,
        nv.sksg,
        nv.sksd,
        nv.svb,
        nv.svh,
        nv.svn,
        nv.svg,
        nv.svp,
        nv.svsb,
        nv.svsh,
        nv.svsn,
        nv.svsg,
        *kr.stride(),
        *vr.stride(),
        *outk.stride(),
        *outv.stride(),
        k,
        nv.packed_tokens,
        H_KV=hkv,
        D=d,
        K_BITS=nv.k_bits,
        V_BITS=nv.v_bits,
        KEY_GROUP=nv.key_group,
        VALUE_GROUP=nv.value_group,
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    return _make_gather_result(original, outk, outv, result_style)


def maybe_fast_gather(
    original: Callable[..., Any], *args: Any, _cuda_result_style: str = "object", **kwargs: Any
) -> Any:
    parsed = _parse_gather_call(original, args, kwargs)
    if parsed is None or parsed.pop("backend").lower() not in {"auto", "triton"}:
        return original(*args, **kwargs)
    view = parsed["view"]
    indices = parsed["indices"]
    if not _enabled() or not _TRITON_AVAILABLE or indices.device.type != "cuda" or not _supports_recent_cuda(indices.device):
        if _strict_should_raise(view, "optimized CUDA gather unavailable"):
            raise RuntimeError("CUDA gather fast path unavailable")
        return original(*args, **kwargs)
    try:
        return _fast_gather_impl(
            original=original,
            result_style=_cuda_result_style,
            **parsed,
        )
    except Exception as exc:
        if _strict_should_raise(view, str(exc)):
            raise
        return original(*args, **kwargs)

def _representative_indices(
    *, tq: int, mode: str, query_indices: Any, uniform_queries: int | None
) -> torch.Tensor:
    if isinstance(query_indices, torch.Tensor):
        return query_indices.to(dtype=torch.long)
    if query_indices is not None:
        return torch.as_tensor(list(query_indices), dtype=torch.long)
    mode = mode.lower()
    if mode in {"all", "all_mean"}:
        return torch.arange(tq, dtype=torch.long)
    if mode == "middle":
        return torch.tensor([tq // 2], dtype=torch.long)
    n = max(1, min(tq, int(uniform_queries or 5)))
    return torch.linspace(0, tq - 1, n).round().to(torch.long).unique(sorted=True)


def _make_selection_result(
    original: Callable[..., Any],
    indices: torch.Tensor,
    values: torch.Tensor,
    importance: torch.Tensor | None = None,
) -> Any:

    for obj in original.__globals__.values():
        if not inspect.isclass(obj):
            continue
        annotations = getattr(obj, "__annotations__", {})
        if "indices" in annotations and "values" in annotations:
            kwargs = {"indices": indices, "values": values}
            if "importance" in annotations:
                kwargs["importance"] = importance
            try:
                return obj(**kwargs)
            except Exception:
                pass
    return SimpleNamespace(indices=indices, values=values, importance=importance)


def _fast_selector_fp_impl(
    *,
    query: torch.Tensor,
    view: Any,
    current_key: torch.Tensor | None,
    query_indices: Any,
    mode: str,
    uniform_queries: int | None,
    topk: int,
    domain: str,
    score: str,
    scaling: float | None,
    sort_indices: bool,
    logits_dtype: torch.dtype,
    return_importance: bool,
    original: Callable[..., Any],
) -> Any:
    fp = _normalize_fp_key_view(view, head_dim=int(query.shape[-1]))
    b, hq, tq, d = map(int, query.shape)
    hkv = int(fp.key.shape[1])
    if hq % hkv:
        raise ValueError("query heads must be divisible by KV heads")
    gqa = hq // hkv
    idx = _representative_indices(
        tq=tq, mode=mode, query_indices=query_indices, uniform_queries=uniform_queries
    ).to(device=query.device)
    idx = idx[(idx >= 0) & (idx < tq)]
    if idx.numel() == 0:
        raise ValueError("selector has no representative query positions")
    qrep = (
        query.reshape(b, hkv, gqa, tq, d)
        .index_select(3, idx)
        .reshape(b, hkv, -1, d)
        .contiguous()
    )
    rows = int(qrep.shape[2])
    n = fp.total_tokens
    logits, lse, importance = _SELECTOR_POOL.get(
        device=query.device,
        dtype=logits_dtype,
        batch=b,
        hkv=hkv,
        rows=rows,
        tokens=n,
    )
    block_m = _choose_block_m(query.device, rows=rows, head_dim=d)
    block_n = int(os.getenv("BITSIEVE_CUDA_SELECTOR_BLOCK_N", "128"))
    if block_n not in (64, 128, 256):
        raise ValueError("BITSIEVE_CUDA_SELECTOR_BLOCK_N must be 64, 128, or 256")
    sm_scale = float(scaling if scaling is not None else d**-0.5)
    _log_once(
        f"selector-fp:{b}:{hkv}:{rows}:{n}:{block_m}",
        f"FP selector B={b} Hkv={hkv} rows={rows} N={n} BLOCK_M={block_m} BLOCK_N={block_n}",
    )
    _selector_fp_qk_tiled_kernel[
        (b * hkv, triton.cdiv(n, block_n), triton.cdiv(rows, block_m))
    ](
        qrep,
        fp.key,
        logits,
        *qrep.stride(),
        *fp.key.stride(),
        *logits.stride(),
        n,
        sm_scale,
        H_KV=hkv,
        R=rows,
        D=d,
        USE_BF16=query.dtype == torch.bfloat16,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=d,
        num_warps=_choose_num_warps(block_m=block_m, head_dim=d),
        num_stages=3,
    )
    score_mode = score.lower()
    use_softmax = score_mode not in {"raw", "logit", "logits"}
    if use_softmax:
        _row_lse_kernel[(b * hkv * rows,)](
            logits,
            lse,
            n,
            *logits.stride(),
            *lse.stride(),
            H_KV=hkv,
            R=rows,
            BLOCK_N=256,
            num_warps=4,
        )
        if domain.lower() == "full" and current_key is not None and current_key.numel():
            current_scores = torch.matmul(
                qrep, current_key.to(qrep.dtype).transpose(-1, -2)
            ).mul_(sm_scale)
            current_lse = torch.logsumexp(current_scores.float(), dim=-1)
            torch.logaddexp(lse, current_lse, out=lse)
        elif domain.lower() != "prefix":
            raise ValueError(f"unknown selector domain: {domain}")
    _importance_kernel[(b * hkv, triton.cdiv(n, 256))](
        logits,
        lse,
        importance,
        *logits.stride(),
        *lse.stride(),
        *importance.stride(),
        n,
        H_KV=hkv,
        R=rows,
        USE_SOFTMAX=use_softmax,
        BLOCK_N=256,
        num_warps=4,
    )
    k = min(max(1, int(topk)), n)
    values, selected = torch.topk(importance, k, dim=-1, sorted=False)
    if sort_indices:
        order = torch.argsort(selected, dim=-1)
        selected = torch.gather(selected, -1, order)
        values = torch.gather(values, -1, order)
    return _make_selection_result(
        original,
        selected,
        values,
        importance.clone() if return_importance else None,
    )

def _fast_selector_impl(
    *,
    query: torch.Tensor,
    view: Any,
    current_key: torch.Tensor | None,
    query_indices: Any,
    mode: str,
    uniform_queries: int | None,
    topk: int,
    domain: str,
    score: str,
    scaling: float | None,
    sort_indices: bool,
    logits_dtype: torch.dtype,
    return_importance: bool,
    original: Callable[..., Any],
) -> Any:
    nv = _normalize_view(view, head_dim=int(query.shape[-1]))
    b, hq, tq, d = map(int, query.shape)
    hkv = int(nv.k_q.shape[1])
    if hq % hkv:
        raise ValueError("query heads must be divisible by KV heads")
    gqa = hq // hkv
    idx = _representative_indices(
        tq=tq, mode=mode, query_indices=query_indices, uniform_queries=uniform_queries
    ).to(device=query.device)
    idx = idx[(idx >= 0) & (idx < tq)]
    if idx.numel() == 0:
        raise ValueError("selector has no representative query positions")
    m = int(idx.numel())


    qrep = (
        query.reshape(b, hkv, gqa, tq, d)
        .index_select(3, idx)
        .reshape(b, hkv, gqa * m, d)
        .contiguous()
    )
    rows = int(qrep.shape[2])
    n_packed = nv.packed_tokens
    n_total = nv.packed_tokens + nv.residual_tokens
    logits, lse, importance = _SELECTOR_POOL.get(
        device=query.device,
        dtype=logits_dtype,
        batch=b,
        hkv=hkv,
        rows=rows,
        tokens=n_total,
    )
    groups = n_packed // 32
    block_m = _choose_block_m(query.device, rows=rows, head_dim=d)
    block_d = d
    sm_scale = float(scaling if scaling is not None else d**-0.5)
    warps = _choose_num_warps(block_m=block_m, head_dim=d)
    _log_once(
        f"selector-low:{b}:{hkv}:{rows}:{n_total}:{nv.k_bits}:{block_m}",
        f"low-bit selector B={b} Hkv={hkv} rows={rows} N={n_total} K{nv.k_bits} BLOCK_M={block_m}",
    )
    _selector_qk_tiled_kernel[(b * hkv, groups, triton.cdiv(rows, block_m))](
        qrep,
        nv.k_q,
        nv.k_scale,
        nv.k_zero,
        logits,
        *qrep.stride(),
        nv.skb,
        nv.skh,
        nv.skg,
        nv.skd,
        nv.skp,
        nv.sksb,
        nv.sksh,
        nv.sksg,
        nv.sksd,
        *logits.stride(),
        sm_scale,
        H_KV=hkv,
        R=rows,
        D=d,
        K_BITS=nv.k_bits,
        USE_BF16=query.dtype == torch.bfloat16,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        num_warps=warps,
        num_stages=2,
    )
    if nv.residual_tokens:
        kr = nv.k_residual[:, :, : nv.residual_tokens, :].to(qrep.dtype)
        logits[..., n_packed:n_total].copy_(
            torch.matmul(qrep, kr.transpose(-1, -2)).mul_(sm_scale).to(logits_dtype)
        )

    score_mode = score.lower()
    use_softmax = score_mode not in {"raw", "logit", "logits"}
    if use_softmax:
        _row_lse_kernel[(b * hkv * rows,)](
            logits,
            lse,
            n_total,
            *logits.stride(),
            *lse.stride(),
            H_KV=hkv,
            R=rows,
            BLOCK_N=256,
            num_warps=4,
        )
        if domain.lower() == "full" and current_key is not None and current_key.numel():
            current_scores = torch.matmul(
                qrep, current_key.to(qrep.dtype).transpose(-1, -2)
            ).mul_(sm_scale)
            current_lse = torch.logsumexp(current_scores.float(), dim=-1)
            torch.logaddexp(lse, current_lse, out=lse)
        elif domain.lower() != "prefix":
            raise ValueError(f"unknown selector domain: {domain}")
    _importance_kernel[(b * hkv, triton.cdiv(n_total, 256))](
        logits,
        lse,
        importance,
        *logits.stride(),
        *lse.stride(),
        *importance.stride(),
        n_total,
        H_KV=hkv,
        R=rows,
        USE_SOFTMAX=use_softmax,
        BLOCK_N=256,
        num_warps=4,
    )
    k = min(max(1, int(topk)), n_total)
    values, selected = torch.topk(importance, k, dim=-1, sorted=False)
    if sort_indices:
        order = torch.argsort(selected, dim=-1)
        selected = torch.gather(selected, -1, order)
        values = torch.gather(values, -1, order)
    return _make_selection_result(
        original,
        selected,
        values,
        importance.clone() if return_importance else None,
    )


# Every selector argument the fast path understands. _parse_selector_call picks
# these out by name and ignores whatever is left, so an option added to
# selector_topk would be honoured on the Triton path and silently dropped here --
# correct-looking numbers computed against the wrong contract. Anything outside
# this set makes the fast path decline instead of guess.
_KNOWN_SELECTOR_ARGS = frozenset(
    {
        "query", "q", "queries",
        "cache", "cache_view", "layer_view", "view", "old_cache",
        "current_key", "current_k", "k_current", "k_cur",
        "query_indices", "selector_queries", "representative_indices",
        "representative_query_indices",
        "mode", "selector_mode",
        "uniform_queries", "uniform_n", "num_selector_queries",
        "topk", "top_k", "k", "selected_k", "budget",
        "domain", "selector_domain",
        "score", "score_mode", "selector_score",
        "scaling", "scale", "sm_scale",
        "sort_indices", "sorted_indices",
        "logits_dtype", "selector_logits_dtype",
        "return_importance",
        "backend",
        "kernel_variant", "selector_kernel_variant", "variant",
        "scratch",
    }
)


def _unsupported_selector_args(bound: Mapping[str, Any]) -> list[str]:
    return sorted(
        name
        for name, value in bound.items()
        if str(name).lower() not in _KNOWN_SELECTOR_ARGS and value is not None
    )


def _parse_selector_call(
    original: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any] | None:
    bound = _bind_arguments(original, args, kwargs)
    unsupported = _unsupported_selector_args(bound)
    if unsupported:
        return None
    query = _pick_named(bound, ("query", "q", "queries"))
    view = _pick_named(bound, ("cache", "cache_view", "layer_view", "view", "old_cache"))
    current_key = _pick_named(bound, ("current_key", "current_k", "k_current", "k_cur"))
    if query is None and args and isinstance(args[0], torch.Tensor):
        query = args[0]
    if view is None:
        for value in args[1:]:
            if not isinstance(value, torch.Tensor):
                try:
                    _normalize_view(value, head_dim=int(query.shape[-1]))
                    view = value
                    break
                except Exception:
                    pass
    if not isinstance(query, torch.Tensor) or view is None:
        return None
    query_indices = _pick_named(
        bound,
        (
            "query_indices",
            "selector_queries",
            "representative_indices",
            "representative_query_indices",
        ),
    )
    mode = str(_pick_named(bound, ("mode", "selector_mode")) or "uniform")
    uniform_queries = _as_int(
        _pick_named(bound, ("uniform_queries", "uniform_n", "num_selector_queries")), 5
    )
    topk = _as_int(_pick_named(bound, ("topk", "top_k", "k", "selected_k", "budget")))
    if topk is None:
        return None
    domain = str(_pick_named(bound, ("domain", "selector_domain")) or "prefix")
    score = str(_pick_named(bound, ("score", "score_mode", "selector_score")) or "softmax")
    scaling = _pick_named(bound, ("scaling", "scale", "sm_scale"))
    sort_indices = bool(_pick_named(bound, ("sort_indices", "sorted_indices")) if _pick_named(bound, ("sort_indices", "sorted_indices")) is not None else True)
    dtype_value = _pick_named(bound, ("logits_dtype", "selector_logits_dtype"))
    if isinstance(dtype_value, torch.dtype):
        logits_dtype = dtype_value
    elif str(dtype_value).lower() in {"float32", "fp32", "torch.float32"}:
        logits_dtype = torch.float32
    else:
        logits_dtype = torch.float16
    return_importance = bool(_pick_named(bound, ("return_importance",)) or False)
    backend = str(_pick_named(bound, ("backend",)) or "auto")
    variant = str(_pick_named(bound, ("kernel_variant", "selector_kernel_variant", "variant")) or "blocked")
    return {
        "query": query,
        "view": view,
        "current_key": current_key if isinstance(current_key, torch.Tensor) else None,
        "query_indices": query_indices,
        "mode": mode,
        "uniform_queries": uniform_queries,
        "topk": topk,
        "domain": domain,
        "score": score,
        "scaling": None if scaling is None else float(scaling),
        "sort_indices": sort_indices,
        "logits_dtype": logits_dtype,
        "return_importance": return_importance,
        "backend": backend,
        "variant": variant,
    }


def maybe_fast_selector(original: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    parsed = _parse_selector_call(original, args, kwargs)
    if parsed is None:
        return original(*args, **kwargs)
    requested = parsed.pop("backend").lower() in {"auto", "triton"} and parsed.pop("variant").lower() in {
        "blocked",
        "cuda",
        "cuda_fast",
        "portable",
        "splitk",
        "hopper",
        "hopper_fast",
        "tiled",
    }
    if not requested:
        return original(*args, **kwargs)
    ok, reason = _can_fast(parsed["query"], parsed["view"], parsed["current_key"])
    if ok:
        return _fast_selector_impl(original=original, **parsed)
    fp_ok, fp_reason = _can_fast_fp_selector(parsed["query"], parsed["view"])
    if fp_ok:
        return _fast_selector_fp_impl(original=original, **parsed)
    if _strict_should_raise(parsed["view"], reason):
        raise RuntimeError(f"CUDA selector fast path unavailable: {reason}")
    return original(*args, **kwargs)


def install_info() -> dict[str, Any]:
    profile = None
    if torch.cuda.is_available():
        try:
            profile = _device_profile(torch.device("cuda", torch.cuda.current_device()))
        except Exception:
            profile = None
    eligible = (
        False
        if profile is None
        else _supports_recent_cuda(torch.device("cuda", torch.cuda.current_device()))
    )
    return {
        "marker": _FAST_MARKER,
        "enabled": _enabled(),
        "strict": _strict(),
        "triton_available": _TRITON_AVAILABLE,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": None if profile is None else profile.name,
        "capability": None if profile is None else (profile.major, profile.minor),
        "architecture_family": None if profile is None else profile.family,
        "sm_count": None if profile is None else profile.sm_count,


        "eligible": eligible,
        "optimized_supported": eligible,
    }
