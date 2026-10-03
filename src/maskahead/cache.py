from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .config import QuantizationConfig
from .reference import (
    PackedKeys,
    PackedValues,
    dequantize_keys,
    dequantize_values,
    gather_per_kv_head,
    values_per_byte,
)


def _dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


@dataclass(slots=True)
class AppendPlan:
    old_length: int
    new_tokens: int
    old_residual: int
    quantize_from_residual: int
    new_residual: int
    staged_layers: set[int]


@dataclass(slots=True)
class LayerCacheView:
    k_bits: int
    v_bits: int
    length: int
    quantized_length: int
    residual_length: int
    key_token_group: int
    value_channel_group: int
    k_q: torch.Tensor | None
    k_scale: torch.Tensor | None
    k_zero: torch.Tensor | None
    v_q: torch.Tensor | None
    v_scale: torch.Tensor | None
    v_zero: torch.Tensor | None
    k_fp: torch.Tensor | None
    v_fp: torch.Tensor | None
    k_residual: torch.Tensor | None
    v_residual: torch.Tensor | None


    head_dim: int | None = None
    key_mode: str = "channel"
    k_chan: torch.Tensor | None = None       # [B, H, D] per-channel key divisor (key_mode="token")


class PackedKVCache:

    def __init__(
        self,
        *,
        num_layers: int,
        batch_size: int,
        num_kv_heads: int,
        head_dim: int,
        max_tokens: int,
        quant: QuantizationConfig,
        device: torch.device | str,
        compute_dtype: torch.dtype = torch.bfloat16,
        backend: str = "auto",
    ) -> None:
        quant.validate(head_dim=head_dim)
        if num_layers <= 0 or batch_size <= 0 or num_kv_heads <= 0 or head_dim <= 0:
            raise ValueError("all cache dimensions must be positive")
        self.num_layers = int(num_layers)
        self.batch_size = int(batch_size)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.max_tokens = int(max_tokens)
        self.quant = quant
        self.device = torch.device(device)
        self.compute_dtype = compute_dtype
        self.param_dtype = _dtype_from_name(quant.param_dtype)
        if backend not in ("auto", "triton", "torch"):
            raise ValueError(f"unknown cache backend: {backend}")
        self.backend = backend

        self.length = 0
        self.quantized_length = 0
        self.residual_length = 0
        self._append_plan: AppendPlan | None = None

        # Storage grows on demand (see _grow) instead of holding max_tokens from the
        # start, and shrinks after an eviction compacts it: what is allocated is what
        # the cache really holds, not the configured ceiling.
        gk = quant.key_token_group
        self.capacity = min(self.max_tokens, max(gk, (1024 // gk) * gk))
        self.dropped = 0                     # tokens removed by compact(): logical = physical + dropped
        self.stage = False                   # True: new entries stay bf16 (residual) until rebuild()
        self.peak_used_bytes = 0             # max over the run of the bytes the entries occupy
        self.peak_allocated_bytes = 0        # max over the run of the storage torch holds
        self.memory_trace: list[list[int]] | None = None   # [tokens seen, used, allocated] per change
        # quant.rotation == 'hadamard': orthonormal H [D, D]; stored K and V are x @ H
        self.rotation: torch.Tensor | None = None
        if quant.rotation == 'hadamard' and (quant.k_bits < 16 or quant.v_bits < 16):
            h = torch.ones(1, 1)
            while h.shape[0] < head_dim:
                h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
            self.rotation = (h / head_dim**0.5).to(device=self.device, dtype=compute_dtype)
        # quant.key_bias == 'mean': per-layer [B, H, 1, D] key centre, set by the first write
        self.key_center: list[torch.Tensor | None] | None = (
            [None] * self.num_layers if quant.key_bias == 'mean' and quant.k_bits < 16 else None)
        l, b, h, d, n = (
            self.num_layers,
            self.batch_size,
            self.num_kv_heads,
            self.head_dim,
            self.capacity,
        )
        gv = quant.value_channel_group
        residual_capacity = quant.residual_tokens + gk - 1
        self.residual_capacity = max(1, residual_capacity)

        self.k_chan: torch.Tensor | None = None
        self.k_fp: torch.Tensor | None = None
        self.v_fp: torch.Tensor | None = None
        self.k_q: torch.Tensor | None = None
        self.k_scale: torch.Tensor | None = None
        self.k_zero: torch.Tensor | None = None
        self.v_q: torch.Tensor | None = None
        self.v_scale: torch.Tensor | None = None
        self.v_zero: torch.Tensor | None = None
        self.k_residual: torch.Tensor | None = None
        self.v_residual: torch.Tensor | None = None

        if quant.k_bits == 16:
            self.k_fp = torch.empty((l, b, h, n, d), device=self.device, dtype=compute_dtype)
        elif quant.key_mode == "token":
            k_vpb = values_per_byte(quant.k_bits)
            gv_ = quant.value_channel_group
            self.k_q = torch.empty((l, b, h, n, d // gv_, gv_ // k_vpb), device=self.device, dtype=torch.uint8)
            self.k_scale = torch.empty((l, b, h, n, d // gv_), device=self.device, dtype=self.param_dtype)
            self.k_zero = torch.empty_like(self.k_scale)
            self.k_residual = torch.empty((l, b, h, self.residual_capacity, d), device=self.device, dtype=compute_dtype)
            self.k_chan = torch.ones((l, b, h, d), device=self.device, dtype=torch.float32)
            self._k_chan_set = [False] * l
        else:
            k_vpb = values_per_byte(quant.k_bits)
            max_groups = math.ceil(n / gk)
            self.k_q = torch.empty(
                (l, b, h, max_groups, d, gk // k_vpb),
                device=self.device,
                dtype=torch.uint8,
            )
            self.k_scale = torch.empty(
                (l, b, h, max_groups, d), device=self.device, dtype=self.param_dtype
            )
            self.k_zero = torch.empty_like(self.k_scale)
            self.k_residual = torch.empty(
                (l, b, h, self.residual_capacity, d),
                device=self.device,
                dtype=compute_dtype,
            )

        if quant.v_bits == 16:
            self.v_fp = torch.empty((l, b, h, n, d), device=self.device, dtype=compute_dtype)
        else:
            v_vpb = values_per_byte(quant.v_bits)
            value_groups = d // gv
            self.v_q = torch.empty(
                (l, b, h, n, value_groups, gv // v_vpb),
                device=self.device,
                dtype=torch.uint8,
            )
            self.v_scale = torch.empty(
                (l, b, h, n, value_groups), device=self.device, dtype=self.param_dtype
            )
            self.v_zero = torch.empty_like(self.v_scale)
            self.v_residual = torch.empty(
                (l, b, h, self.residual_capacity, d),
                device=self.device,
                dtype=compute_dtype,
            )

    def __len__(self) -> int:


        return self.num_layers if self.length > 0 else 0

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Logical length (next position). Physical storage holds self.length entries."""
        return self.length + self.dropped

    _TOKEN_AXIS = {"k_fp": 3, "v_fp": 3, "v_q": 3, "v_scale": 3, "v_zero": 3}
    _GROUP_AXIS = {"k_q": 3, "k_scale": 3, "k_zero": 3}

    def _resize(self, cap: int) -> None:
        """Reallocate token storage to cap tokens, keeping the first self.length."""
        gk = self.quant.key_token_group
        cap = min(self.max_tokens, -(-cap // gk) * gk)
        keep_t, keep_g = self.length, -(-self.quantized_length // gk)
        for name, axis in list(self._TOKEN_AXIS.items()) + list(self._GROUP_AXIS.items()):
            old = getattr(self, name)
            if old is None:
                continue
            shape = list(old.shape)
            is_group = name in self._GROUP_AXIS and self.quant.key_mode != "token"
            shape[axis] = -(-cap // gk) if is_group else cap
            new = torch.empty(shape, device=old.device, dtype=old.dtype)
            n = keep_g if is_group else keep_t
            if n:
                new.narrow(axis, 0, n).copy_(old.narrow(axis, 0, n))
            setattr(self, name, new)
        self.capacity = cap

    def _grow_residual(self, needed: int) -> None:
        cap = max(needed, int(self.residual_capacity * 1.5), 64)
        for name in ("k_residual", "v_residual"):
            old = getattr(self, name)
            if old is None:
                continue
            shape = list(old.shape)
            shape[3] = cap
            new = torch.empty(shape, device=old.device, dtype=old.dtype)
            if self.residual_length:
                new[:, :, :, : self.residual_length].copy_(old[:, :, :, : self.residual_length])
            setattr(self, name, new)
        self.residual_capacity = cap

    def rebuild(self, groups_before: torch.Tensor, encode: torch.Tensor, groups_after: torch.Tensor,
                stage: torch.Tensor) -> torch.Tensor:
        """New physical layout [groups_before | encoded | groups_after] + bf16 staging.

        All index tensors are [L, B, H, n], padded with -1 (a padded slot is dead: it
        holds zeros or a copy and must be masked by the caller).
          groups_before / groups_after: old key-group ids copied as they are (codes and
            scales of keys and values moved bit-exactly; no re-quantization);
          encode: old physical slots quantized into fresh groups (first quantization of
            staged bf16 entries, or re-packing of a sparse group);
          stage: old physical slots kept in bf16 (the new residual).
        Returns remap [L, B, H, old_length]: the new slot of every old slot, -1 if dropped.
        Packed (k_bits, v_bits < 16) caches only.
        """
        from .kernels.ops import quantize_key_groups_into, quantize_values_into

        if self._append_plan is not None:
            raise RuntimeError("rebuild during an append")
        gk = self.quant.key_token_group
        L, B, H, D = self.num_layers, self.batch_size, self.num_kv_heads, self.head_dim
        dev = self.device
        old_len, old_q = self.length, self.quantized_length
        g1, m, g3, st = groups_before.shape[-1], encode.shape[-1], groups_after.shape[-1], stage.shape[-1]
        g2 = -(-m // gk)
        ql = (g1 + g2 + g3) * gk
        new_len = ql + st
        cap = new_len + 1024
        ng_cap = -(-cap // gk)
        kq = torch.empty((L, B, H, ng_cap, D, self.k_q.shape[-1]), device=dev, dtype=self.k_q.dtype)
        ks = torch.empty((L, B, H, ng_cap, D), device=dev, dtype=self.k_scale.dtype)
        kz = torch.empty_like(ks)
        vq = torch.empty((L, B, H, cap, *self.v_q.shape[4:]), device=dev, dtype=self.v_q.dtype)
        vs = torch.empty((L, B, H, cap, self.v_scale.shape[-1]), device=dev, dtype=self.v_scale.dtype)
        vz = torch.empty_like(vs)
        rcap = max(st + 256, 64)
        kr = torch.empty((L, B, H, rcap, D), device=dev, dtype=self.k_residual.dtype)
        vr = torch.empty_like(kr)
        remap = torch.full((L, B, H, old_len + 1), -1, device=dev, dtype=torch.int64)   # last: sink for pads
        ar_g = torch.arange(gk, device=dev)

        def copy_groups(l, gid, at):
            if gid.shape[-1] == 0:
                return
            safe = gid.clamp_min(0)
            n = safe.shape[-1]
            e = safe.view(B, H, n, 1, 1).expand(-1, -1, -1, D, kq.shape[-1])
            kq[l, :, :, at:at + n] = self.k_q[l].gather(2, e)
            e2 = safe.view(B, H, n, 1).expand(-1, -1, -1, D)
            ks[l, :, :, at:at + n] = self.k_scale[l].gather(2, e2)
            kz[l, :, :, at:at + n] = self.k_zero[l].gather(2, e2)
            tok = (safe.unsqueeze(-1) * gk + ar_g).reshape(B, H, n * gk)
            t0 = at * gk
            ev = tok.view(B, H, n * gk, 1, 1).expand(-1, -1, -1, *vq.shape[4:])
            vq[l, :, :, t0:t0 + n * gk] = self.v_q[l].gather(2, ev)
            ev2 = tok.view(B, H, n * gk, 1).expand(-1, -1, -1, vs.shape[-1])
            vs[l, :, :, t0:t0 + n * gk] = self.v_scale[l].gather(2, ev2)
            vz[l, :, :, t0:t0 + n * gk] = self.v_zero[l].gather(2, ev2)
            new_slot = ((torch.arange(n, device=dev) + at).view(1, 1, n, 1) * gk + ar_g).expand(B, H, n, gk)
            valid = (gid >= 0).unsqueeze(-1).expand(-1, -1, -1, gk)
            idx = torch.where(valid, tok.view(B, H, n, gk), old_len)
            remap[l].scatter_(-1, idx.reshape(B, H, -1), new_slot.reshape(B, H, -1))

        for l in range(L):
            copy_groups(l, groups_before[l], 0)
            copy_groups(l, groups_after[l], g1 + g2)
            if m or st:
                k, v = self.dequantize_layer(l)
            if m:
                safe = encode[l].clamp_min(0)
                src_k = k.gather(2, safe.unsqueeze(-1).expand(-1, -1, -1, D))
                src_v = v.gather(2, safe.unsqueeze(-1).expand(-1, -1, -1, D))
                dead = (encode[l] < 0).unsqueeze(-1)
                pad = g2 * gk - m
                src_k = torch.nn.functional.pad(src_k.masked_fill(dead, 0), (0, 0, 0, pad))
                src_v = torch.nn.functional.pad(src_v.masked_fill(dead, 0), (0, 0, 0, pad))
                quantize_key_groups_into(src_k.contiguous(), kq[l, :, :, g1:g1 + g2], ks[l, :, :, g1:g1 + g2],
                                         kz[l, :, :, g1:g1 + g2], bits=self.quant.k_bits, token_group=gk,
                                         backend=self.backend)
                quantize_values_into(src_v.contiguous(), vq[l, :, :, g1 * gk:(g1 + g2) * gk],
                                     vs[l, :, :, g1 * gk:(g1 + g2) * gk], vz[l, :, :, g1 * gk:(g1 + g2) * gk],
                                     bits=self.quant.v_bits, channel_group=self.quant.value_channel_group,
                                     backend=self.backend)
                slot = torch.arange(m, device=dev).view(1, 1, m).expand(B, H, m) + g1 * gk
                remap[l].scatter_(-1, torch.where(encode[l] >= 0, encode[l], old_len), slot)
            if st:
                safe = stage[l].clamp_min(0)
                dead = (stage[l] < 0).unsqueeze(-1)
                kr[l, :, :, :st] = k.gather(2, safe.unsqueeze(-1).expand(-1, -1, -1, D)).masked_fill(dead, 0)
                vr[l, :, :, :st] = v.gather(2, safe.unsqueeze(-1).expand(-1, -1, -1, D)).masked_fill(dead, 0)
                slot = torch.arange(st, device=dev).view(1, 1, st).expand(B, H, st) + ql
                remap[l].scatter_(-1, torch.where(stage[l] >= 0, stage[l], old_len), slot)
            if m or st:
                del k, v
        self.k_q, self.k_scale, self.k_zero, self.v_q, self.v_scale, self.v_zero = kq, ks, kz, vq, vs, vz
        self.k_residual, self.v_residual = kr, vr
        self.capacity, self.residual_capacity = cap, rcap
        self.length, self.quantized_length, self.residual_length = new_len, ql, st
        self.dropped += old_len - new_len
        self._note_peak()
        return remap[..., :old_len]

    def _grow(self, needed: int) -> None:
        if needed > self.max_tokens:
            raise RuntimeError(f"KV cache capacity exceeded: {needed}>{self.max_tokens}")
        self._resize(max(needed, int(self.capacity * 1.5)))

    def compact(self, keep: torch.Tensor) -> None:
        """Keep only entries keep[l, b, h, :] (physical indices) of every layer, re-staged in
        that order. Exact for bf16 caches (its remaining caller); a packed cache would be
        re-quantized, so eviction uses rebuild() there. Logical positions are unchanged.
        """
        if self._append_plan is not None:
            raise RuntimeError("compact during an append")
        c = int(keep.shape[-1])
        old = self.length
        if c > old:
            raise ValueError("compact cannot grow the cache")
        gk = self.quant.key_token_group
        survivors = []
        for layer_idx in range(self.num_layers):
            k, v = self.dequantize_layer(layer_idx)
            idx = keep[layer_idx].to(torch.int64).unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
            survivors.append((k.gather(2, idx), v.gather(2, idx)))
            del k, v
        self.length = self.quantized_length = self.residual_length = 0
        if self.capacity > 2 * max(c, gk) + 1024:
            self._resize(c + 1024)
        self.begin_append(c)
        for layer_idx, (k, v) in enumerate(survivors):
            self.stage_layer(layer_idx, k, v)
        self.commit_append()
        self.dropped += old - c

    def get_max_cache_shape(self) -> int:
        return self.max_tokens

    def reset(self) -> None:
        if self.key_center is not None:
            self.key_center = [None] * self.num_layers
        self.dropped = 0
        self.length = 0
        self.quantized_length = 0
        self.residual_length = 0
        self._append_plan = None

    @property
    def append_in_progress(self) -> bool:
        return self._append_plan is not None

    def begin_append(self, new_tokens: int) -> AppendPlan:
        if self._append_plan is not None:
            raise RuntimeError("another cache append is already in progress")
        if new_tokens <= 0:
            raise ValueError("new_tokens must be positive")
        if self.length + new_tokens > self.capacity:
            self._grow(self.length + new_tokens)

        if self.quant.k_bits == 16 and self.quant.v_bits == 16:
            move = 0
            new_residual = 0
        else:
            total_tail = self.residual_length + new_tokens
            if self.quant.key_mode == "token" and self.quant.k_bits < 16:
                move = total_tail            # per-token keys: no token groups to wait for
            elif self.stage:
                move = 0                     # quantize-once: the staged tail waits for rebuild()
            else:
                excess = max(0, total_tail - self.quant.residual_tokens)
                move = (excess // self.quant.key_token_group) * self.quant.key_token_group
            new_residual = total_tail - move
            if new_residual >= self.residual_capacity:
                self._grow_residual(new_residual + 1)

        plan = AppendPlan(
            old_length=self.length,
            new_tokens=int(new_tokens),
            old_residual=self.residual_length,
            quantize_from_residual=move,
            new_residual=new_residual,
            staged_layers=set(),
        )
        self._append_plan = plan
        return plan

    def abort_append(self) -> None:
        self._append_plan = None

    def stage_layer(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor) -> None:
        plan = self._append_plan
        if plan is None:
            raise RuntimeError("begin_append must be called before stage_layer")
        if layer_idx in plan.staged_layers:
            raise RuntimeError(f"layer {layer_idx} was staged twice")
        expected = (self.batch_size, self.num_kv_heads, plan.new_tokens, self.head_dim)
        if tuple(key.shape) != expected or tuple(value.shape) != expected:
            raise ValueError(
                f"append tensor shape mismatch: expected {expected}, got K={tuple(key.shape)}, "
                f"V={tuple(value.shape)}"
            )
        key = key.to(self.compute_dtype)
        value = value.to(self.compute_dtype)
        if self.key_center is not None:
            if self.key_center[layer_idx] is None:
                self.key_center[layer_idx] = key.float().mean(dim=2, keepdim=True).to(self.compute_dtype)
            key = key - self.key_center[layer_idx]
        if self.rotation is not None:
            key = key @ self.rotation
            value = value @ self.rotation

        start = plan.old_length
        end = start + plan.new_tokens
        if self.quant.k_bits == 16:
            assert self.k_fp is not None
            self.k_fp[layer_idx, :, :, start:end, :].copy_(key)
        if self.quant.v_bits == 16:
            assert self.v_fp is not None
            self.v_fp[layer_idx, :, :, start:end, :].copy_(value)

        if self.quant.k_bits < 16 or self.quant.v_bits < 16:
            old_r = plan.old_residual
            if old_r:
                k_parts = []
                v_parts = []
                if self.quant.k_bits < 16:
                    assert self.k_residual is not None
                    k_parts.append(self.k_residual[layer_idx, :, :, :old_r, :])
                if self.quant.v_bits < 16:
                    assert self.v_residual is not None
                    v_parts.append(self.v_residual[layer_idx, :, :, :old_r, :])
            else:
                k_parts = []
                v_parts = []

            if self.quant.k_bits < 16:
                k_parts.append(key)
                k_tail = k_parts[0] if len(k_parts) == 1 else torch.cat(k_parts, dim=2)
            else:
                k_tail = None
            if self.quant.v_bits < 16:
                v_parts.append(value)
                v_tail = v_parts[0] if len(v_parts) == 1 else torch.cat(v_parts, dim=2)
            else:
                v_tail = None

            move = plan.quantize_from_residual
            if move:
                group_start = self.quantized_length // self.quant.key_token_group
                group_count = move // self.quant.key_token_group
                if k_tail is not None and self.quant.key_mode == "token":
                    from .kernels.ops import quantize_values_into

                    kt = k_tail[:, :, :move, :]
                    if not self._k_chan_set[layer_idx]:
                        # c_i = sqrt(max_t |K_t,i|) over the first append (the prefill), fixed after
                        self.k_chan[layer_idx] = kt.abs().amax(dim=2).float().clamp_min(1e-4).sqrt()
                        self._k_chan_set[layer_idx] = True
                    kn = (kt.float() / self.k_chan[layer_idx].unsqueeze(2)).to(self.compute_dtype)
                    t0 = self.quantized_length
                    quantize_values_into(kn.contiguous(), self.k_q[layer_idx, :, :, t0:t0 + move],
                                         self.k_scale[layer_idx, :, :, t0:t0 + move],
                                         self.k_zero[layer_idx, :, :, t0:t0 + move], bits=self.quant.k_bits,
                                         channel_group=self.quant.value_channel_group, backend=self.backend)
                elif k_tail is not None:
                    assert self.k_q is not None and self.k_scale is not None and self.k_zero is not None
                    q_out = self.k_q[
                        layer_idx, :, :, group_start : group_start + group_count, :, :
                    ]
                    s_out = self.k_scale[
                        layer_idx, :, :, group_start : group_start + group_count, :
                    ]
                    z_out = self.k_zero[
                        layer_idx, :, :, group_start : group_start + group_count, :
                    ]
                    from .kernels.ops import quantize_key_groups_into

                    quantize_key_groups_into(
                        k_tail[:, :, :move, :].contiguous(),
                        q_out,
                        s_out,
                        z_out,
                        bits=self.quant.k_bits,
                        token_group=self.quant.key_token_group,
                        backend=self.backend,
                    )
                if v_tail is not None:
                    assert self.v_q is not None and self.v_scale is not None and self.v_zero is not None
                    v_start = self.quantized_length
                    v_end = v_start + move
                    qv_out = self.v_q[layer_idx, :, :, v_start:v_end, :, :]
                    sv_out = self.v_scale[layer_idx, :, :, v_start:v_end, :]
                    zv_out = self.v_zero[layer_idx, :, :, v_start:v_end, :]
                    from .kernels.ops import quantize_values_into

                    quantize_values_into(
                        v_tail[:, :, :move, :].contiguous(),
                        qv_out,
                        sv_out,
                        zv_out,
                        bits=self.quant.v_bits,
                        channel_group=self.quant.value_channel_group,
                        backend=self.backend,
                    )

            remain = plan.new_residual
            if k_tail is not None:
                assert self.k_residual is not None
                self.k_residual[layer_idx, :, :, :remain, :].copy_(k_tail[:, :, move:, :])
            if v_tail is not None:
                assert self.v_residual is not None
                self.v_residual[layer_idx, :, :, :remain, :].copy_(v_tail[:, :, move:, :])

        plan.staged_layers.add(layer_idx)

    def commit_append(self) -> None:
        plan = self._append_plan
        if plan is None:
            raise RuntimeError("no append is in progress")
        expected = set(range(self.num_layers))
        if plan.staged_layers != expected:
            missing = sorted(expected - plan.staged_layers)
            raise RuntimeError(f"cannot commit: layers not staged: {missing}")
        self.length += plan.new_tokens
        if self.quant.k_bits < 16 or self.quant.v_bits < 16:
            self.quantized_length += plan.quantize_from_residual
            self.residual_length = plan.new_residual
        self._append_plan = None
        self._note_peak()
        if self.quantized_length + self.residual_length != self.length and (
            self.quant.k_bits < 16 or self.quant.v_bits < 16
        ):
            raise AssertionError("cache length invariant failed")

    def load_dynamic_cache(self, dynamic_cache) -> None:
        if self.length != 0 or self._append_plan is not None:
            raise RuntimeError("load_dynamic_cache requires an empty cache")
        if dynamic_cache is None:
            return
        length = int(dynamic_cache.get_seq_length())
        if length == 0:
            return
        if length > self.max_tokens:
            raise RuntimeError(f"prefill cache length {length} exceeds capacity {self.max_tokens}")
        self.begin_append(length)
        try:
            for layer_idx in range(self.num_layers):
                # Three cache APIs across the transformers versions this has
                # to serve: tuple indexing, the key_cache/value_cache lists,
                # and (5.x) layers[i].keys / .values.
                key = value = None
                try:
                    key, value = dynamic_cache[layer_idx]
                except Exception:
                    if hasattr(dynamic_cache, "key_cache"):
                        key = dynamic_cache.key_cache[layer_idx]
                        value = dynamic_cache.value_cache[layer_idx]
                    else:
                        layer = dynamic_cache.layers[layer_idx]
                        key, value = layer.keys, layer.values
                if key is None or value is None:
                    raise TypeError(
                        f"cannot read layer {layer_idx} out of {type(dynamic_cache).__name__}"
                    )
                self.stage_layer(
                    layer_idx,
                    key[..., :length, :].contiguous(),
                    value[..., :length, :].contiguous(),
                )
            self.commit_append()
        except Exception:
            self.abort_append()
            raise

    def layer_view(self, layer_idx: int) -> LayerCacheView:
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(layer_idx)
        # token-mode keys are laid out per token, like values
        qgroups = (self.quantized_length if self.quant.key_mode == "token"
                   else self.quantized_length // self.quant.key_token_group)
        return LayerCacheView(
            k_bits=self.quant.k_bits,
            v_bits=self.quant.v_bits,
            length=self.length,
            quantized_length=self.quantized_length,
            residual_length=self.residual_length,
            key_token_group=self.quant.key_token_group,
            value_channel_group=self.quant.value_channel_group,
            k_q=None if self.k_q is None else self.k_q[layer_idx, :, :, :qgroups, :, :],
            k_scale=(
                None if self.k_scale is None else self.k_scale[layer_idx, :, :, :qgroups, :]
            ),
            k_zero=None if self.k_zero is None else self.k_zero[layer_idx, :, :, :qgroups, :],
            v_q=(
                None
                if self.v_q is None
                else self.v_q[layer_idx, :, :, : self.quantized_length, :, :]
            ),
            v_scale=(
                None
                if self.v_scale is None
                else self.v_scale[layer_idx, :, :, : self.quantized_length, :]
            ),
            v_zero=(
                None
                if self.v_zero is None
                else self.v_zero[layer_idx, :, :, : self.quantized_length, :]
            ),
            k_fp=None if self.k_fp is None else self.k_fp[layer_idx, :, :, : self.length, :],
            v_fp=None if self.v_fp is None else self.v_fp[layer_idx, :, :, : self.length, :],
            k_residual=(
                None
                if self.k_residual is None
                else self.k_residual[layer_idx, :, :, : self.residual_length, :]
            ),
            v_residual=(
                None
                if self.v_residual is None
                else self.v_residual[layer_idx, :, :, : self.residual_length, :]
            ),
            head_dim=self.head_dim,
            key_mode=self.quant.key_mode,
            k_chan=None if self.k_chan is None else self.k_chan[layer_idx],
        )

    def compact_tokens(self, keep: torch.Tensor) -> None:
        """Keep entries keep[l, b, h, :] (physical indices) by moving their codes: exact for
        per-token layouts (token-mode keys, per-token values, bf16), no re-quantization."""
        if self._append_plan is not None or self.residual_length:
            raise RuntimeError("compact_tokens needs a fully quantized cache")
        c = int(keep.shape[-1])
        for name in ("k_q", "k_scale", "k_zero", "v_q", "v_scale", "v_zero", "k_fp", "v_fp"):
            t = getattr(self, name)
            if t is None:
                continue
            idx = keep.to(torch.int64)
            idx = idx.view(*idx.shape, *([1] * (t.dim() - 4))).expand(*idx.shape, *t.shape[4:])
            t[:, :, :, :c] = t.gather(3, idx)
        old = self.length
        self.length = c
        self.quantized_length = c if (self.quant.k_bits < 16 or self.quant.v_bits < 16) else 0
        self.dropped += old - c
        if self.capacity > 2 * c + 2048:
            self._resize(c + 1024)
        self._note_peak()

    def dequantize_layer(
        self, layer_idx: int, *, dtype: torch.dtype | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = dtype or self.compute_dtype
        view = self.layer_view(layer_idx)
        if (self.device.type == "cuda" and self.backend != "torch" and view.k_bits in (2, 4)
                and view.v_bits in (2, 4) and view.length and self.head_dim in (64, 128)
                and (self.quant.key_token_group in (8, 16, 32) or self.quant.key_mode == "token")):
            from .kernels.splitk import Prefix, dequantize_prefix

            return dequantize_prefix(Prefix.from_view(view), self.batch_size, self.num_kv_heads,
                                     self.head_dim, dtype=dtype)
        if view.k_bits == 16:
            assert view.k_fp is not None
            key = view.k_fp.to(dtype)
        else:
            parts = []
            if view.quantized_length:
                assert view.k_q is not None and view.k_scale is not None and view.k_zero is not None
                parts.append(
                    dequantize_keys(
                        PackedKeys(
                            payload=view.k_q,
                            scale=view.k_scale,
                            zero=view.k_zero,
                            bits=view.k_bits,
                            token_group=view.key_token_group,
                            tokens=view.quantized_length,
                        ),
                        dtype=dtype,
                    )
                )
            if view.residual_length:
                assert view.k_residual is not None
                parts.append(view.k_residual.to(dtype))
            key = parts[0] if len(parts) == 1 else torch.cat(parts, dim=2)

        if view.v_bits == 16:
            assert view.v_fp is not None
            value = view.v_fp.to(dtype)
        else:
            parts_v = []
            if view.quantized_length:
                assert view.v_q is not None and view.v_scale is not None and view.v_zero is not None
                parts_v.append(
                    dequantize_values(
                        PackedValues(
                            payload=view.v_q,
                            scale=view.v_scale,
                            zero=view.v_zero,
                            bits=view.v_bits,
                            channel_group=view.value_channel_group,
                            tokens=view.quantized_length,
                        ),
                        dtype=dtype,
                    )
                )
            if view.residual_length:
                assert view.v_residual is not None
                parts_v.append(view.v_residual.to(dtype))
            value = parts_v[0] if len(parts_v) == 1 else torch.cat(parts_v, dim=2)
        return key, value

    def gather_layer_reference(
        self,
        layer_idx: int,
        indices: torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key, value = self.dequantize_layer(layer_idx, dtype=dtype)
        return gather_per_kv_head(key, indices), gather_per_kv_head(value, indices)

    def allocated_nbytes(self) -> int:
        """Bytes of the storage tensors (capacity, not just the used part)."""
        names = ("k_fp", "v_fp", "k_q", "k_scale", "k_zero", "v_q", "v_scale", "v_zero", "k_residual", "v_residual")
        return int(sum(t.numel() * t.element_size() for t in (getattr(self, x) for x in names) if t is not None))

    def bf16_equivalent_nbytes(self) -> int:
        """A bf16 K+V cache of every token seen (the logical length): the memory baseline."""
        return 2 * 2 * self.num_layers * self.batch_size * self.num_kv_heads * self.head_dim * self.get_seq_length()

    def _note_peak(self) -> None:
        used, alloc = self.logical_nbytes()["total"], self.allocated_nbytes()
        self.peak_used_bytes = max(self.peak_used_bytes, used)
        self.peak_allocated_bytes = max(self.peak_allocated_bytes, alloc)
        if self.memory_trace is not None:
            self.memory_trace.append([self.get_seq_length(), used, alloc])

    def logical_nbytes(self) -> dict[str, int]:
        l, b, h, d = self.num_layers, self.batch_size, self.num_kv_heads, self.head_dim
        qn = self.quantized_length
        rn = self.residual_length
        param_size = torch.tensor([], dtype=self.param_dtype).element_size()
        compute_size = torch.tensor([], dtype=self.compute_dtype).element_size()

        if self.quant.k_bits == 16:
            k_payload = l * b * h * self.length * d * compute_size
            k_meta = 0
            k_res = 0
        else:
            k_payload = l * b * h * qn * d * self.quant.k_bits // 8
            if self.quant.key_mode == "token":
                k_meta = l * b * h * qn * (d // self.quant.value_channel_group) * 2 * param_size + l * b * h * d * 4
            else:
              k_meta = (
                l
                * b
                * h
                * (qn // self.quant.key_token_group)
                * d
                * 2
                * param_size
            )
            k_res = l * b * h * rn * d * compute_size

        if self.quant.v_bits == 16:
            v_payload = l * b * h * self.length * d * compute_size
            v_meta = 0
            v_res = 0
        else:
            v_payload = l * b * h * qn * d * self.quant.v_bits // 8
            v_meta = (
                l
                * b
                * h
                * qn
                * (d // self.quant.value_channel_group)
                * 2
                * param_size
            )
            v_res = l * b * h * rn * d * compute_size

        return {
            "key_payload": int(k_payload),
            "key_metadata": int(k_meta),
            "key_residual": int(k_res),
            "value_payload": int(v_payload),
            "value_metadata": int(v_meta),
            "value_residual": int(v_res),
            "total": int(k_payload + k_meta + k_res + v_payload + v_meta + v_res),
        }
