from __future__ import annotations

from typing import Any

from dataclasses import replace, dataclass

import math

import torch
from pathlib import Path

from ..cache import LayerCacheView, PackedKVCache
from ..config import ExperimentConfig
from ..kernels.ops import (
    SelectorScratch,
    dense_packed_attention,
    gather_packed_kv,
    quantize_key_groups_into,
    quantize_values_into,
    selector_topk,
)
from ..eviction import EvictionState, live_cache_nbytes
from ..reference import sdpa_compact, simulate_key_quantization
from .trace import CoverageArmRecord, CoverageRecord, RunTrace, RuntimeTimer, SelectionRecord


def value_score(a: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """a_i * ||v_i - c||, c = sum_j a_j v_j / sum_j a_j. a [..., n], v [..., n, d] -> [..., n].

    The one scoring rule behind both decisions: selection feeds it the current
    block's mass, lookahead eviction the mass of the next blocks.
    """
    import os
    if os.environ.get("BITSIEVE_VSCORE_FP32") == "1":          # diagnostic
        v = v.float()
        c = (a.unsqueeze(-1) * v).sum(-2, keepdim=True) / a.sum(-1, keepdim=True).clamp_min(1e-12).unsqueeze(-1)
        return a * (v - c).norm(dim=-1)
    w = a.to(v.dtype)
    c = torch.einsum("...n,...nd->...d", w, v) / w.sum(-1, keepdim=True).clamp_min(1e-12)
    return a * (v - c.unsqueeze(-2)).norm(dim=-1).to(a.dtype)


def fast_path_supported(config, head_dim: int, device_type: str = "cuda"):
    """(ok, reason): can the split-K kernels (kernels/splitk.py) serve this config?

    Everything the fast path does not implement falls back to the reference path:
    diagnostics, research-only selectors and policies, semantic B, mixed 16/packed bits.
    """
    import os

    cfg, sel, ev, q = config, config.selector, config.eviction, config.quant
    if os.environ.get("BITSIEVE_FAST", "1") == "0" or device_type != "cuda":
        return False, "disabled or not cuda"
    if cfg.backend == "torch" or cfg.semantic not in ("A", "dense"):
        return False, "backend/semantic"
    if cfg.async_selector and cfg.semantic == "A" and not ev.enabled and not sel.value_aware:
        return False, "async selector"
    if cfg.collect_diagnostics or cfg.coverage_diagnostics or cfg.compact_format != "bf16":
        return False, "diagnostics/compact format"
    if cfg.attn_log_dir:
        return False, "attention recording"
    if sel.score != "softmax" or sel.domain != "prefix" or sel.mode not in ("all", "middle"):
        return False, "selector score/domain/mode"
    if sel.value_joint or sel.value_per_query or sel.value_set != "none" or sel.value_caote:
        return False, "research selector"
    if sel.value_aware and sel.value_center != "attn":
        return False, "value centre"
    if ev.enabled and (ev.policy in ("greedy_set", "oracle_future", "ema_recent_score")
                       or ev.diag_path or ev.oracle_record_dir):
        return False, "research eviction"
    if not ((q.k_bits in (2, 4) and q.v_bits in (2, 4)) or (q.k_bits == 16 and q.v_bits == 16)):
        return False, "bits"
    if q.key_token_group not in (8, 16, 32) or q.value_channel_group not in (8, 16, 32, 64, 128):
        return False, "group sizes"
    return head_dim in (64, 128), "head_dim"


# Set by the eval loop before each generation: names the oracle record file.
CURRENT_EXAMPLE: str | None = None


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


@dataclass(slots=True)
class CompactLayerState:
    key: torch.Tensor
    value: torch.Tensor
    indices: torch.Tensor | None = None
    selected_k: int = 0
    use_dense: bool = False
    ready_event: torch.cuda.Event | None = None
    pending_refs: tuple[torch.Tensor, ...] | None = None
    packed_view: LayerCacheView | None = None


class BitSieveSession:

    def __init__(
        self,
        cache: PackedKVCache,
        config: ExperimentConfig,
        *,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        compute_dtype: torch.dtype,
    ) -> None:
        config.validate(head_dim=head_dim, num_layers=cache.num_layers)
        self.cache = cache
        self.config = config
        self.num_q_heads = int(num_q_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.compute_dtype = compute_dtype
        self.trace = RunTrace()
        self.timer = RuntimeTimer(self.trace, enabled=config.profile_layers)
        self.selector_scratch = SelectorScratch()

        self.block_index = -1
        self.step_index = -1
        self.masked_positions: list[int] = []
        self.query_indices: list[int] = []
        self.commit_current = False
        self._step0_finished = False
        self._states: dict[int, CompactLayerState] = {}
        # Shadow fp16 keys, only when scoring coverage. Keys alone are enough:
        # attention weights depend on Q and K, not V. Seeded from the prefill
        # cache and extended at every block commit, so it mirrors exactly what
        # the packed cache holds.
        self._fp16_key_shadow: dict[int, torch.Tensor] = {}
        # Eviction: what the cache KEEPS, as opposed to what a block reads.
        self._eviction: EvictionState | None = None
        self._next_position = 0     # cache positions admitted so far
        self._tokens_seen = 0       # what the cache would hold with no eviction
        self._live_scratch: torch.Tensor | None = None
        self._live_scratch_v: torch.Tensor | None = None
        if config.eviction.enabled:
            self._eviction = EvictionState(
                num_layers=cache.num_layers,
                batch_size=cache.batch_size,
                num_kv_heads=self.num_kv_heads,
                capacity=config.eviction.capacity_floor,
                device=cache.device,
                max_position=config.max_cache_tokens,
            )

        # Expected Attention: discounted moments of pre-RoPE step-0 queries per
        # (layer, query head), plus the RoPE phase of the last observed block.
        self._greedy_q: dict[int, torch.Tensor] = {}
        self._la_pending: int | None = None          # capacity of a deferred eviction
        self._la_q: dict[int, torch.Tensor] = {}
        self._probe = False                           # inside a lookahead probe forward
        self._attn_log = [] if config.attn_log_dir else None   # records for config.attn_log_dir
        self._attn_log_layers = set(config.attn_log_layers or (2, 8, 14, 20, 27))
        self._la_probe_q: dict[int, torch.Tensor] = {}
        self._la_a1: dict[int, torch.Tensor] = {}      # next block's mass, live axis
        self._la_aprobe: dict[int, torch.Tensor] = {}  # summed mass of probe blocks, live axis
        self._merge_extra = 0                         # probe tokens appended to this step-0 input
        self._prefill_mode = False                    # inside a chunked-prefill forward
        self._prefill_n = 0
        self._pf_alpha: dict[int, torch.Tensor] = {}  # chunk's prefix mass per layer, live axis
        self._snapkv_score: dict[int, torch.Tensor] = {}
        self._q_span: tuple[int | torch.Tensor, int] | None = None   # question positions [start, prompt end)
        if config.quant.rotation != "none" and config.eviction.policy == "expected_attention":
            raise ValueError("quant.rotation with expected_attention is not supported")
        self._qprobe_start: int | None = None         # inside a prefill question probe (its first position)
        self._pf_qa: dict[int, torch.Tensor] = {}     # question probe's prefix mass per layer, live axis
        self._oracle_rec: dict[int, dict[int, torch.Tensor]] = {}
        self._oracle_data: dict | None = None
        self._diag_layers = (2, 8, 14, 20, 26)
        self._diag_q: dict[int, torch.Tensor] = {}
        self._diag_snaps: list[dict] = []
        self._block_key: torch.Tensor | None = None
        self._block_value: torch.Tensor | None = None
        self._rope: tuple[torch.Tensor, torch.Tensor] | None = None
        self._ea_mu: torch.Tensor | None = None      # [L, B, Hq, d]
        self._ea_m2: torch.Tensor | None = None      # [L, B, Hq, d, d]
        self._ea_n = [0] * cache.num_layers
        self._ea_phase: torch.Tensor | None = None   # [d] angle at the last position
        self._ea_freq: torch.Tensor | None = None    # [d] angle step per position

        import os
        self._phys_compact = os.environ.get("BITSIEVE_COMPACT", "1") != "0"
        if self._staging_wanted_init():
            cache.stage = True
        if config.quant.key_mode == "token" and not fast_path_supported(config, head_dim, cache.device.type)[0]:
            raise NotImplementedError("quant.key_mode='token' runs on the fast path only")
        self._selection_stream: torch.cuda.Stream | None = None
        if (
            cache.device.type == "cuda"
            and config.async_selector
            and config.semantic == "A"
            and self._eviction is None
            and not config.selector.value_aware
        ):
            # Step 0 under eviction attends over a single shared live buffer, so
            # overlapping layers on a second stream would race on it. Correctness
            # first; the overlap is an optimization, not a result.
            self._selection_stream = torch.cuda.Stream(device=cache.device)

    @property
    def old_cache_len(self) -> int:
        return self.cache.get_seq_length()

    @property
    def live_cache_len(self) -> int:
        """Entries that still exist. Equals the cache length without eviction."""
        if self._eviction is None:
            return self.old_cache_len
        return self._eviction.live

    def _admit_new_entries(self) -> None:
        """Give newly committed cache positions a slot in the live set."""
        if self._eviction is None:
            return
        length = self.cache.get_seq_length()
        if length > self._next_position:
            m = length - self._next_position              # the newest m physical slots
            self._eviction.append(
                torch.arange(self._next_position, length, device=self.cache.device),
                phys=torch.arange(self.cache.length - m, self.cache.length, device=self.cache.device),
            )
            self._mask_epoch = getattr(self, "_mask_epoch", 0) + 1
            self._tokens_seen += length - self._next_position
            self._next_position = length
        self._peak_live = max(getattr(self, "_peak_live", 0), self._eviction.live)

    def _live_positions(self, layer_idx: int) -> torch.Tensor:
        """Physical cache slots of the live entries of one layer. [B, Hkv, live].

        Token positions (recency, sinks, pins) are in EvictionState.pos; the physical
        slot changes when the cache is rebuilt after an eviction (see _evict_to).
        """
        assert self._eviction is not None
        return self._eviction.phys[layer_idx, ..., : self._eviction.live].to(torch.int64)

    def _slot_mask(self, layer_idx: int):
        """uint8 [B, Hkv, cache.length], 1 on live slots; None when every slot is live."""
        st = self._eviction
        if st is None or self.cache.length <= st.live:
            return None
        key = (getattr(self, "_mask_epoch", 0), self.cache.length, st.live)
        cached = getattr(self, "_mask_cache", None)
        if cached is None or cached[0] != key:
            cached = self._mask_cache = (key, {})
        m = cached[1].get(layer_idx)
        if m is None:
            m = torch.zeros((self.cache.batch_size, self.num_kv_heads, self.cache.length),
                            dtype=torch.uint8, device=self.cache.device)
            m.scatter_(-1, self._live_positions(layer_idx), 1)
            cached[1][layer_idx] = m
        return m

    def _to_live(self, layer_idx: int, per_slot: torch.Tensor) -> torch.Tensor:
        """A per-physical-slot quantity [B, Hkv, n] in live order [B, Hkv, live]."""
        if self._eviction is None:
            return per_slot
        return per_slot.gather(-1, self._live_positions(layer_idx))

    def _live_values(self, layer_idx: int) -> torch.Tensor:
        """Values of the admitted live entries, [B, Hkv, live, D] (compute dtype)."""
        from ..reference import gather_per_kv_head

        _, value = self.cache.dequantize_layer(layer_idx)
        return gather_per_kv_head(value, self._live_positions(layer_idx))

    def _evict_to(self, keep: torch.Tensor, *, state_done: bool = False) -> None:
        """Keep live entries keep [L, B, Hkv, C]; then bring the physical cache in line.

        eviction.compaction = "lazy" (default, packed caches): a key group that keeps at
        least repack_below of its entries is copied bit-exactly (its dead slots stay,
        masked); survivors of sparser groups and staged bf16 entries whose eviction
        decision has now been made are quantized into fresh groups (for staged entries
        the first and only quantization); entries not decided yet (the recent window) and
        a stored-but-unadmitted prefill chunk stay as they are. "full" re-quantizes every
        survivor; "none" only marks entries dead. bf16 caches are gathered exactly.
        """
        st, cfg = self._eviction, self.config.eviction
        if self._attn_log is not None:
            self._log_eviction(keep, state_done)
        if not state_done:
            st.compact(keep)
        self._mask_epoch = getattr(self, "_mask_epoch", 0) + 1
        mode = cfg.compaction if self._phys_compact else "none"
        if mode == "none":
            return
        cache, dev = self.cache, self.cache.device
        live = st.live
        tail = cache.get_seq_length() - self._next_position       # stored, not admitted yet
        phys = st.phys[..., :live].to(torch.int64)                   # [L, B, H, C]
        L, B, H = phys.shape[:3]
        tail_slots = torch.arange(cache.length - tail, cache.length, device=dev).view(1, 1, 1, -1).expand(L, B, H, tail)
        if cache.quant.k_bits >= 16 or cache.quant.v_bits >= 16:
            cache.compact(torch.cat([phys, tail_slots], -1))
            st.phys[..., :live] = torch.arange(live, device=dev, dtype=st.phys.dtype)
            return
        if cache.quant.key_mode == "token":
            # every entry owns its codes and scales: moving them is exact
            cache.compact_tokens(torch.cat([phys, tail_slots], -1))
            st.phys[..., :live] = torch.arange(live, device=dev, dtype=st.phys.dtype)
            return
        gk, ql = cache.quant.key_token_group, cache.quantized_length
        in_q = phys < ql
        ng = max(1, ql // gk)
        grp = torch.where(in_q, phys // gk, ng)                     # ng = "not in a group"
        counts = torch.zeros((L, B, H, ng + 1), dtype=torch.int32, device=dev)
        counts.scatter_add_(-1, grp, torch.ones_like(grp, dtype=torch.int32))
        counts = counts[..., :ng]
        if mode == "lazy":
            need = max(1, int(math.ceil(cfg.repack_below * gk)))
            keep_g = counts >= need
        else:
            keep_g = torch.zeros_like(counts, dtype=torch.bool)
        tail_groups = 0
        if tail and not cache.stage:                                # the chunk: whole groups at the end
            if cache.residual_length or (cache.length - tail) % gk:
                raise RuntimeError("unadmitted tail must be whole key groups")
            tail_groups = tail // gk
            keep_g[..., ng - tail_groups:] = False                  # placed after, as they are
        in_kept = in_q & keep_g.gather(-1, grp.clamp_max(ng - 1)) & (grp < ng)
        pos = st.pos[..., :live].to(torch.int64)
        w = min(cfg.recent_window, live)
        if w:
            thr = pos.topk(w, dim=-1).values.amin(-1, keepdim=True)
        else:
            thr = torch.full((L, B, H, 1), 1 << 60, device=dev, dtype=torch.int64)
        undecided = ~in_q & (pos >= thr)                            # staged, still in the recent window
        encode = ~in_kept & ~undecided
        big = 1 << 62

        def packed(sel, key):
            """Per lane: key where sel, ascending, padded with -1 to the max count over lanes."""
            n = int(sel.sum(-1).max().item()) if sel.numel() else 0
            k2 = torch.where(sel, key, big).sort(dim=-1).values[..., :n]
            return torch.where(k2 >= big, -1, k2)

        def slots_by_pos(sel):
            # sort by position, carry the slot: position * 2^24 + slot (slots < 2^24)
            k2 = packed(sel, pos * (1 << 24) + phys)
            return torch.where(k2 < 0, -1, k2 % (1 << 24))

        groups_before = packed(keep_g, torch.arange(ng, device=dev).expand(L, B, H, ng))
        enc_slots = slots_by_pos(encode)
        stage_slots = slots_by_pos(undecided)
        empty = torch.empty((L, B, H, 0), dtype=torch.int64, device=dev)
        if tail and cache.stage:
            stage_slots = torch.cat([stage_slots, tail_slots], -1)
            groups_after = empty
        elif tail:
            groups_after = torch.arange(ng - tail_groups, ng, device=dev).expand(L, B, H, -1)
        else:
            groups_after = empty
        remap = cache.rebuild(groups_before, enc_slots, groups_after, stage_slots)
        st.phys[..., :live] = remap.gather(-1, phys).to(st.phys.dtype)
        if bool((st.phys[..., :live] < 0).any()):
            raise AssertionError("rebuild lost a live entry")
        self.trace.add_counter("rebuild_encoded", int((enc_slots >= 0).sum()))
        self.trace.add_counter("rebuild_kept_groups", int((groups_before >= 0).sum()))

    def _staging_wanted_init(self) -> bool:
        return self._staging_wanted("prompt")

    def _staging_wanted(self, phase: str) -> bool:
        cfg, q = self.config.eviction, self.config.quant
        if self._eviction is None or not self._phys_compact or cfg.compaction == "none":
            return False
        if q.k_bits >= 16 or q.v_bits >= 16 or q.key_mode == "token":
            return False
        return cfg.stage_bf16 == "all" or (cfg.stage_bf16 == "gen" and phase == "gen")

    def begin_block(self, masked_positions: list[int]) -> None:
        if self._staging_wanted("gen"):
            self.cache.stage = True          # quantize-once: generated entries wait in bf16
        self._admit_new_entries()
        self.block_index += 1
        self.step_index = -1
        self.masked_positions = sorted(set(masked_positions))
        self.query_indices = self.config.selector.query_indices(
            self.masked_positions, self.config.generation.block_size
        )
        self.commit_current = False
        self._step0_finished = False


        for state in self._states.values():
            if state.ready_event is not None:
                raise RuntimeError("begin_block called with unfinished selector work")
            state.indices = None
            state.selected_k = 0
            state.use_dense = False
            state.ready_event = None
            state.pending_refs = None
            state.packed_view = None
        self.trace.add_counter("blocks", 1)

        # Whether the selector can bite at all this block. With a fixed top-k
        # budget it cannot until the prefix outgrows k: selecting 512 of 400
        # entries IS dense attention, so such a block must not be reported as
        # sparse. Counted here so a run can never claim a sparse path it never
        # took.
        if self.config.semantic == "dense":
            self.trace.add_counter("blocks_dense_engine", 1)
        else:
            budget = self.config.selector.effective_topk(self.old_cache_len)
            engages = (
                self.old_cache_len > 0
                and bool(self.query_indices)
                and budget < self.old_cache_len
            )
            self.trace.add_counter(
                "blocks_sparse" if engages else "blocks_dense_bypass", 1
            )

    def prepare_forward(
        self,
        step_index: int,
        *,
        masked_positions: list[int] | None = None,
        commit_current: bool = False,
    ) -> None:
        self.step_index = int(step_index)
        if masked_positions is not None:
            self.masked_positions = sorted(set(int(x) for x in masked_positions))
        self.commit_current = bool(commit_current)

    def _max_selected(self) -> int:
        selector = self.config.selector
        if selector.topk_percent is None:
            return min(selector.topk, self.config.max_cache_tokens)
        return min(
            self.config.max_cache_tokens,
            max(1, int(self.config.max_cache_tokens * selector.topk_percent / 100.0 + 0.999)),
        )

    def _ensure_state(self, layer_idx: int, selected_k: int) -> CompactLayerState:
        state = self._states.get(layer_idx)
        capacity = max(1, selected_k) + self.config.generation.block_size
        if state is not None and state.key.shape[2] >= capacity:
            return state
        shape = (
            self.cache.batch_size,
            self.num_kv_heads,
            capacity,
            self.head_dim,
        )
        state = CompactLayerState(
            key=torch.empty(shape, device=self.cache.device, dtype=self.compute_dtype),
            value=torch.empty(shape, device=self.cache.device, dtype=self.compute_dtype),
        )
        self._states[layer_idx] = state
        return state

    def _dense_required(self, layer_idx: int) -> bool:
        if self.config.semantic == "dense":
            return True
        if layer_idx < self.config.selector.dense_prefix_layers:
            return True
        if self._eviction is not None:
            # With eviction the cache IS the live set, so even a budget that
            # covers it must be served from a gather -- the physical dense path
            # would read entries that no longer exist.
            return self.live_cache_len == 0 or not self.query_indices
        k = self.config.selector.effective_topk(self.old_cache_len)
        return self.old_cache_len == 0 or k >= self.old_cache_len or not self.query_indices

    def _requantize_compact(self, state: CompactLayerState) -> None:
        k = state.selected_k
        if k <= 0:
            raise RuntimeError("cannot requantize an empty selection")
        bits_k = self.config.quant.k_bits
        bits_v = self.config.quant.v_bits
        group_k = self.config.quant.key_token_group
        group_v = self.config.quant.value_channel_group
        qn = (k // group_k) * group_k if bits_k < 16 or bits_v < 16 else k
        rn = k - qn
        b, h, _, d = state.key.shape
        device = state.key.device
        param_dtype = self.cache.param_dtype

        kq = ks = kz = vq = vs = vz = kfp = vfp = kr = vr = None
        if bits_k == 16:
            kfp = state.key[:, :, :k, :].clone()
        else:
            if qn:
                vpb = 8 // bits_k
                kq = torch.empty((b, h, qn // group_k, d, group_k // vpb), device=device, dtype=torch.uint8)
                ks = torch.empty((b, h, qn // group_k, d), device=device, dtype=param_dtype)
                kz = torch.empty_like(ks)
                quantize_key_groups_into(
                    state.key[:, :, :qn, :].contiguous(),
                    kq, ks, kz, bits=bits_k, token_group=group_k, backend=self.config.backend
                )
            if rn:
                kr = state.key[:, :, qn:k, :].clone()

        if bits_v == 16:
            vfp = state.value[:, :, :k, :].clone()
        else:
            if qn:
                vpb_v = 8 // bits_v
                vg = d // group_v
                vq = torch.empty((b, h, qn, vg, group_v // vpb_v), device=device, dtype=torch.uint8)
                vs = torch.empty((b, h, qn, vg), device=device, dtype=param_dtype)
                vz = torch.empty_like(vs)
                quantize_values_into(
                    state.value[:, :, :qn, :].contiguous(),
                    vq, vs, vz, bits=bits_v, channel_group=group_v, backend=self.config.backend
                )
            if rn:
                vr = state.value[:, :, qn:k, :].clone()

        state.packed_view = LayerCacheView(
            k_bits=bits_k,
            v_bits=bits_v,
            length=k,
            quantized_length=qn,
            residual_length=rn,
            key_token_group=group_k,
            value_channel_group=group_v,
            k_q=kq, k_scale=ks, k_zero=kz,
            v_q=vq, v_scale=vs, v_zero=vz,
            k_fp=kfp, v_fp=vfp,
            k_residual=kr, v_residual=vr,
            head_dim=d,
        )

    # ---- honest coverage scoring (diagnostic path only) -------------------

    def seed_fp16_key_shadow(self, layer_idx: int, key: torch.Tensor) -> None:
        """Record the prefill's fp16 keys as the coverage reference."""
        if not (self.config.coverage_diagnostics or self.config.rank_bf16_shadow
                or self.config.precision_pairwise_diagnostics):
            return
        self._fp16_key_shadow[layer_idx] = key.detach().clone()

    def _extend_fp16_key_shadow(self, layer_idx: int, key: torch.Tensor) -> None:
        if not (self.config.coverage_diagnostics or self.config.rank_bf16_shadow
                or self.config.precision_pairwise_diagnostics):
            return
        prev = self._fp16_key_shadow.get(layer_idx)
        cur = key.detach()
        self._fp16_key_shadow[layer_idx] = (
            cur.clone() if prev is None else torch.cat([prev, cur], dim=2)
        )

    @torch.no_grad()
    def _reference_importance(self, layer_idx: int, query: torch.Tensor) -> torch.Tensor | None:
        """Per-(batch, kv-head) attention mass over the prefix under exact fp16
        keys, aggregated over EVERY masked query in the block.

        This is deliberately independent of `selector.mode` and of the cache's
        bit width: it is the target the selection is trying to hit, so it must
        not inherit the candidate's handicaps. Returns [B, Hkv, N].
        """
        key = self._shadow_prefix(layer_idx)
        if key is None:
            return None
        return self._importance_from_keys(key, query)

    def _shadow_prefix(self, layer_idx: int) -> torch.Tensor | None:
        shadow = self._fp16_key_shadow.get(layer_idx)
        if shadow is None:
            return None
        n = self.old_cache_len
        if n <= 0 or shadow.shape[2] < n:
            return None
        return shadow[:, :, :n, :]

    def _importance_from_keys(
        self, key: torch.Tensor, query: torch.Tensor,
        query_indices: list[int] | None = None,
    ) -> torch.Tensor | None:
        """The scoring rule, factored out so a sweep arm is scored identically.

        An arm differs from the reference in exactly one thing - the precision of
        the keys it ranks with. Sharing this function is what makes that true;
        a second copy of the softmax-mass rule would let the two drift apart and
        the comparison would quietly stop meaning what it claims.
        """
        n = key.shape[2]
        rows = query_indices if query_indices is not None else (self.masked_positions or list(range(query.shape[2])))
        if not rows:
            return None

        b, hq, _, d = query.shape
        hkv = int(key.shape[1])
        if hq % hkv:
            return None
        g = hq // hkv
        if query_indices is None:
            idx = torch.as_tensor(rows, device=query.device, dtype=torch.long)
            q = query.reshape(b, hkv, g, query.shape[2], d).index_select(3, idx)
            q = q.reshape(b, hkv, g * idx.numel(), d).float()
        else:
            from ..kernels.ops import _selected_queries
            q = _selected_queries(query, rows, hkv).float()
        k32 = key.float()
        scale = float(self.head_dim**-0.5)

        acc = torch.zeros(b, hkv, n, device=query.device, dtype=torch.float32)
        chunk = max(1, int(self.config.coverage_query_chunk))
        total = q.shape[2]
        for start in range(0, total, chunk):
            qc = q[:, :, start : start + chunk, :]
            logits = torch.matmul(qc, k32.transpose(-1, -2)) * scale
            acc += torch.softmax(logits, dim=-1).sum(dim=2)
        return acc / float(total)

    def _key4_shadow(self, key: torch.Tensor) -> torch.Tensor:
        """K4 dequantized keys, preserving the packed cache's BF16 partial tail."""
        group = self.config.quant.key_token_group
        quantized = (key.shape[2] // group) * group
        if quantized == 0:
            return key
        prefix = simulate_key_quantization(
            key[:, :, :quantized, :], bits=4, token_group=group,
            param_dtype=self.cache.param_dtype,
        )
        return prefix if quantized == key.shape[2] else torch.cat(
            (prefix, key[:, :, quantized:, :]), dim=2
        )

    def _selection_scores_for_pair(self, layer_idx: int, key: torch.Tensor,
                                   query: torch.Tensor) -> torch.Tensor:
        """Our mass/value selector score, recomputed from the same Q and V."""
        importance = self._importance_from_keys(
            key, query, query_indices=self.query_indices,
        )
        if importance is None:
            raise RuntimeError("could not compute pairwise selector importance")
        cfg = self.config.selector
        if not cfg.value_aware:
            return importance
        _, values = self.cache.dequantize_layer(layer_idx)
        v = values[:, :, :key.shape[2], :].float()
        w = importance.float()
        live_mask = self._slot_mask(layer_idx)
        if live_mask is not None:
            live_mask = live_mask[..., :key.shape[2]].bool()
            w = w.masked_fill(~live_mask, 0.0)
            v = v.masked_fill(~live_mask.unsqueeze(-1), 0.0)
        if cfg.value_center == "attn":
            center = (w.unsqueeze(-1) * v).sum(2, keepdim=True) / w.sum(
                -1, keepdim=True
            ).clamp_min(1e-12).unsqueeze(-1)
        elif cfg.value_center == "none":
            center = torch.zeros_like(v[:, :, :1, :])
        elif live_mask is None:
            center = v.mean(2, keepdim=True)
        else:
            center = v.sum(2, keepdim=True) / live_mask.sum(
                -1, keepdim=True
            ).clamp_min(1).unsqueeze(-1)
        score = w * (v - center).norm(dim=-1)
        if cfg.value_caote:
            score = score / (1.0 - w).clamp_min(1e-6)
        return score

    def _log_pairwise_set(self, kind: str, layer_idx: int, capacity: int,
                          full_score: torch.Tensor, k4_score: torch.Tensor) -> None:
        if self._attn_log is None or self._eviction is None:
            return
        slots, positions = self._log_positions(layer_idx)
        n = positions.shape[-1]
        cap = min(int(capacity), n)
        full_idx = torch.topk(full_score[..., :n], cap, dim=-1).indices
        k4_idx = torch.topk(k4_score[..., :n], cap, dim=-1).indices
        self._attn_log.append({
            "kind": "pairwise_set", "set_kind": kind, "block": self.block_index,
            "layer": layer_idx, "capacity": cap,
            "positions": positions[0].to(torch.int32).cpu(),
            "full_keep": positions.gather(-1, full_idx)[0].to(torch.int32).cpu(),
            "k4_keep": positions.gather(-1, k4_idx)[0].to(torch.int32).cpu(),
        })

    def _log_pairwise_selection(self, layer_idx: int, query: torch.Tensor,
                                k: int) -> None:
        if not self.config.precision_pairwise_diagnostics:
            return
        if self.config.quant.k_bits != 16 or not self.config.eviction.record_only:
            raise RuntimeError("pairwise diagnostics require a BF16 recording-only cache")
        shadow, _ = self.cache.dequantize_layer(layer_idx)
        if shadow.shape[2] < self.old_cache_len:
            raise RuntimeError("pairwise diagnostics found an incomplete BF16 cache")
        shadow = shadow[:, :, :self.old_cache_len, :]
        full_score = self._selection_scores_for_pair(layer_idx, shadow, query)
        k4_shadow = self._key4_shadow(shadow)
        k4_score = self._selection_scores_for_pair(layer_idx, k4_shadow, query)
        self._log_pairwise_set("selection", layer_idx, k, full_score, k4_score)

    def _log_pairwise_eviction(self, capacity: int) -> None:
        """Compare lookahead keep sets under BF16/K4 keys for the same probe Qs."""
        if not self.config.precision_pairwise_diagnostics:
            return
        from ..reference import gather_per_kv_head

        st = self._eviction
        for layer_idx in range(st.num_layers):
            if self.config.quant.k_bits != 16 or not self.config.eviction.record_only:
                raise RuntimeError("pairwise diagnostics require a BF16 recording-only cache")
            shadow, _ = self.cache.dequantize_layer(layer_idx)
            if shadow.shape[2] < self.cache.length:
                raise RuntimeError("pairwise eviction diagnostics found an incomplete BF16 cache")
            slots = self._live_positions(layer_idx)
            key = gather_per_kv_head(shadow[:, :, :self.cache.length, :], slots)
            key4 = self._key4_shadow(key)

            def mass(q, candidate_key):
                prob = torch.softmax(torch.einsum(
                    "bhrd,bhnd->bhrn", q.float(), candidate_key.float()
                ) * self.head_dim**-0.5, dim=-1)
                return prob.mean(dim=2)

            a_full = mass(self._la_q[layer_idx], key)
            a_k4 = mass(self._la_q[layer_idx], key4)
            probe_q = self._la_probe_q.get(layer_idx)
            if isinstance(probe_q, torch.Tensor) and probe_q.numel():
                m_probe = max(1, probe_q.shape[2] // self.config.generation.block_size
                              // (self.num_q_heads // self.num_kv_heads))
                a_full = a_full + mass(probe_q, key) * m_probe
                a_k4 = a_k4 + mass(probe_q, key4) * m_probe
            if self.config.eviction.lookahead_score == "mass":
                s_full, s_k4 = a_full, a_k4
            else:
                values = self._live_values(layer_idx)
                s_full, s_k4 = value_score(a_full, values), value_score(a_k4, values)
            self._log_pairwise_set("eviction", layer_idx, capacity, s_full, s_k4)

    @torch.no_grad()
    def _record_coverage(
        self,
        layer_idx: int,
        query: torch.Tensor,
        indices: torch.Tensor,
        selected_k: int,
    ) -> None:
        ref = self._reference_importance(layer_idx, query)
        if ref is None:
            return
        k = min(int(selected_k), ref.shape[-1])
        if k <= 0:
            return
        ref_vals, ref_idx = torch.topk(ref, k, dim=-1)
        sel = indices.to(torch.long)
        # `ref` is a mean of softmax rows, so it sums to 1 over the prefix and
        # `got`/`best` are already shares of the FULL attention mass. Recording
        # both absolutes separates the loss the budget forces on any selector
        # from the loss this particular selector adds: a selector can be optimal
        # (mass 1.0) while the budget still costs most of the mass.
        best_abs = ref_vals.sum(dim=-1)
        got_abs = ref.gather(-1, sel).sum(dim=-1)
        best = best_abs.clamp_min(1e-12)
        got = got_abs
        mass = (got / best).flatten().tolist()
        # Index overlap against the same reference top-k.
        mark = torch.zeros_like(ref, dtype=torch.bool)
        mark.scatter_(-1, ref_idx, True)
        overlap = (mark.gather(-1, sel).sum(dim=-1).float() / float(k)).flatten().tolist()
        self._record_coverage_arms(layer_idx, query, ref)
        self.trace.coverage.append(
            CoverageRecord(
                block=self.block_index,
                layer=layer_idx,
                old_cache_len=self.old_cache_len,
                selected_k=k,
                selector_queries=len(self.query_indices),
                mass=[round(float(x), 5) for x in mass],
                overlap=[round(float(x), 5) for x in overlap],
                mass_abs=[round(float(x), 5) for x in got_abs.flatten().tolist()],
                ceiling_abs=[round(float(x), 5) for x in best_abs.flatten().tolist()],
            )
        )

    @torch.no_grad()
    def _record_coverage_arms(
        self, layer_idx: int, query: torch.Tensor, ref: torch.Tensor
    ) -> None:
        """Score each configured sweep arm against the same fp16 reference.

        Runs off the fp16 key shadow rather than the live cache, so one
        generation yields every arm on identical queries. A per-arm generation
        would diverge after the first block and the coverages would no longer be
        measurements of the same thing.

        Deliberately the slow path: simulate_key_quantization has no packed
        kernel behind it, which is what lets an arm use a bit width the kernels
        cannot address (3). It costs one scoring matmul per arm per layer-block
        and changes nothing about what the model generates.
        """
        arms = self.config.coverage_arms
        if not arms:
            return
        key = self._shadow_prefix(layer_idx)
        if key is None:
            return
        n = int(ref.shape[-1])
        group = int(self.config.quant.key_token_group)
        for arm in arms:
            bits = int(arm.get("bits", 16))
            if arm.get("topk") is not None:
                k = int(arm["topk"])
            else:
                k = int(n * float(arm["topk_percent"]) / 100.0)
            k = max(1, min(k, n))
            scored = self._importance_from_keys(
                simulate_key_quantization(
                    key, bits=bits, token_group=group, allow_ragged=True
                ),
                query,
            )
            if scored is None:
                continue
            sel = torch.topk(scored, k, dim=-1).indices
            ref_vals, ref_idx = torch.topk(ref, k, dim=-1)
            best_abs = ref_vals.sum(dim=-1)
            got_abs = ref.gather(-1, sel).sum(dim=-1)
            mark = torch.zeros_like(ref, dtype=torch.bool)
            mark.scatter_(-1, ref_idx, True)
            overlap = mark.gather(-1, sel).sum(dim=-1).float() / float(k)
            self.trace.coverage_arms.append(
                CoverageArmRecord(
                    arm=str(arm["name"]),
                    bits=bits,
                    block=self.block_index,
                    layer=layer_idx,
                    old_cache_len=self.old_cache_len,
                    selected_k=k,
                    mass=[round(float(x), 5) for x in (got_abs / best_abs.clamp_min(1e-12)).flatten().tolist()],
                    mass_abs=[round(float(x), 5) for x in got_abs.flatten().tolist()],
                    ceiling_abs=[round(float(x), 5) for x in best_abs.flatten().tolist()],
                    overlap=[round(float(x), 5) for x in overlap.flatten().tolist()],
                )
            )

    def compact_nbytes(self) -> int:
        """Bytes held by the gathered per-layer compact caches.

        These live outside PackedKVCache but are resident for the whole block,
        so a memory claim that counts only the packed cache understates the
        footprint - especially at short prefixes, where the compact buffers can
        outweigh what they were gathered from.
        """
        total = 0
        for state in self._states.values():
            total += state.key.numel() * state.key.element_size()
            total += state.value.numel() * state.value.element_size()
            view = state.packed_view
            if view is None:
                continue
            for t in (
                view.k_q, view.k_scale, view.k_zero,
                view.v_q, view.v_scale, view.v_zero,
                view.k_fp, view.v_fp, view.k_residual, view.v_residual,
            ):
                if t is not None:
                    total += t.numel() * t.element_size()
        return int(total)

    def _effective_budget(self) -> int:
        """Entries a block may read. Never more than the live set."""
        live = self.live_cache_len
        basis = live
        if self.config.selector.topk_basis == "seen" and self._eviction is not None:
            basis = max(live, self._tokens_seen)
        return min(self.config.selector.effective_topk(basis), live)

    def _rescore_by_value(self, layer_idx, selected, k, live_mask, query=None):
        """Re-rank the candidates by importance * ||v - v_head_mean||.

        The plain selector keeps whatever the attention mass alone points at.
        This weighs each candidate by how far its value sits from the head's
        mean value, so an entry that is attended to but carries the same thing
        the head already averages in loses its slot to one that does not.

        Also counts how much this actually changes the kept set, because a
        rescoring that reorders nothing cannot change the output either.
        """
        imp = selected.importance
        if imp is None:
            raise ValueError("value_aware selection needs the selector's importance")
        n = imp.shape[-1]
        key, value = self.cache.dequantize_layer(layer_idx)
        v = value[:, :, :n, :].to(torch.float32)
        w = imp.to(torch.float32)
        live = None if live_mask is None else live_mask[..., :n]
        if live is not None:
            w = w.masked_fill(~live, 0.0)
            # Evicted slots hold whatever the compaction left there, NaN
            # included; w = 0 does not mask it (0 * NaN = NaN), so zero the
            # values themselves or one dead slot poisons the head's centre.
            if self.config.collect_diagnostics:
                self.trace.add_counter(
                    "value_nonfinite_live", int((~torch.isfinite(v) & live.unsqueeze(-1)).sum())
                )
            v = v.masked_fill(~live.unsqueeze(-1), 0.0)
        if self.config.selector.value_center == "attn":
            center = (w.unsqueeze(-1) * v).sum(dim=2, keepdim=True) / w.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-12).unsqueeze(-1)
        elif self.config.selector.value_center == "none":
            center = torch.zeros_like(v[:, :, :1, :])
        elif live is not None:
            center = v.sum(dim=2, keepdim=True) / live.sum(
                dim=-1, keepdim=True
            ).clamp_min(1).unsqueeze(-1).to(v.dtype)
        else:
            center = v.mean(dim=2, keepdim=True)
        spread = (v - center).norm(dim=-1)
        if n > k and self.config.collect_diagnostics:
            # Is there anything to rank on? IQR/median of the spread per head
            # over the live entries; if it is flat, the rescoring degenerates
            # into the plain selector. Diagnostics only: every float() here is
            # a host sync per layer and block.
            s = spread if live is None else spread.masked_fill(~live, float("nan"))
            q = torch.nanquantile(s.flatten(0, 1), torch.tensor(
                [0.25, 0.5, 0.75], device=spread.device), dim=-1)
            ratio = torch.nan_to_num((q[2] - q[0]) / q[1].clamp_min(1e-12), nan=0.0)
            self.trace.add_counter("value_spread_heads", int(ratio.numel()))
            self.trace.add_counter("value_spread_iqr_ppm_sum", int(round(float(ratio.sum()) * 1e6)))
            self.trace.add_counter("value_spread_flat_heads", int((ratio < 0.2).sum()))
            self.trace.add_counter("value_spread_wide_heads", int((ratio > 1.0).sum()))
        score = w * spread
        sel_cfg = self.config.selector
        if sel_cfg.value_joint:
            score = self._joint_caote(query, key[:, :, :n, :], v, live)
        elif sel_cfg.value_per_query:
            score = self._per_query_caote(query, key[:, :, :n, :], v, live)
        if sel_cfg.value_set != "none" and n > k:
            chosen = self._greedy_set(query, key[:, :, :n, :], v, live, w, center, k)
            # Hand the chosen set to the common top-k below as the only finite scores.
            score = torch.full_like(w, float("-inf")).scatter(-1, chosen, 1.0)
        if self.config.selector.value_caote and not self.config.selector.value_per_query:
            # 1 / (1 - p_i): with i gone the rest renormalize, so the output
            # moves by p_i / (1 - p_i) * (o - v_i), not p_i * (o - v_i).
            score = score / (1.0 - w).clamp_min(1e-6)
        # ema_recent_score folds this block's score, not its bare mass, into the
        # eviction EMA; w is already zero off the live set.
        self._last_value_score = score
        if live_mask is not None:
            score = score.masked_fill(~live_mask[..., :n], float("-inf"))
        idx = torch.topk(score, k, dim=-1).indices
        if self.config.selector.sort_indices:
            idx = idx.sort(dim=-1).values
        old = selected.indices
        if old is not None and old.shape == idx.shape and n > k and self.config.collect_diagnostics:
            same = (
                (idx.unsqueeze(-1) == old.unsqueeze(-2)).any(dim=-1).to(torch.float32).mean()
            )
            self.trace.add_counter("value_rescore_calls", 1)
            self.trace.add_counter("value_rescore_kept_ppm", int(round(float(same) * 1e6)))
        return replace(selected, indices=idx.to(old.dtype) if old is not None else idx)

    def _query_terms(self, query, key, v, live):
        """Per-query prefix softmax p [B,Hkv,R,n], outputs o [B,Hkv,R,d], logits."""
        from ..kernels.ops import _selected_queries

        if query is None:
            raise ValueError("per-query value scores need the selector queries")
        q = _selected_queries(query, self.query_indices, self.num_kv_heads).to(torch.float32)
        logits = torch.einsum("bhrd,bhnd->bhrn", q, key.to(torch.float32)) * self.head_dim**-0.5
        if live is not None:
            logits = logits.masked_fill(~live.unsqueeze(2), float("-inf"))
        p = torch.softmax(logits, dim=-1)
        o = torch.einsum("bhrn,bhnd->bhrd", p, v)
        return q, p, o, logits

    @staticmethod
    def _dist(v, c):
        """||v_i - c_q|| for all (q, i) via the quadratic form. [B,Hkv,R,n]."""
        d2 = (
            (v * v).sum(-1).unsqueeze(2)
            - 2.0 * torch.einsum("bhrd,bhnd->bhrn", c, v)
            + (c * c).sum(-1).unsqueeze(-1)
        )
        return d2.clamp_min(0.0).sqrt()

    def _joint_caote(self, query, key, v, live):
        """Idea A: mean_q m_q p_iq ||v_i - y_q||, y_q the output of the joint
        prefix+block softmax the block really attends with. [B, Hkv, n]."""
        from ..kernels.ops import _selected_queries

        q, p, o, logits = self._query_terms(query, key, v, live)
        bk = self._block_key.to(torch.float32)                             # [B, Hkv, T, d]
        bv = self._block_value.to(torch.float32)
        lb = torch.einsum("bhrd,bhtd->bhrt", q, bk) * self.head_dim**-0.5
        lse_pre = torch.logsumexp(logits, dim=-1)
        lse_blk = torch.logsumexp(lb, dim=-1)
        m = torch.sigmoid(lse_pre - lse_blk)                               # [B, Hkv, R]
        o_blk = torch.einsum("bhrt,bhtd->bhrd", torch.softmax(lb, dim=-1), bv)
        y = m.unsqueeze(-1) * o + (1.0 - m.unsqueeze(-1)) * o_blk
        score = (m.unsqueeze(-1) * p * self._dist(v, y)).mean(dim=2)
        # Pre-check numbers: the prefix share and how much it varies inside a
        # KV group (only within-group variation can change a group's ranking).
        if not self.config.collect_diagnostics:
            if live is not None:
                score = score.masked_fill(~live, 0.0)
            return score
        self.trace.add_counter("joint_calls", 1)
        self.trace.add_counter("joint_m_mean_ppm", int(round(float(m.mean()) * 1e6)))
        cv = m.std(dim=2) / m.mean(dim=2).clamp_min(1e-6)
        self.trace.add_counter("joint_m_cv_ppm", int(round(float(cv.mean()) * 1e6)))
        shift = (y - o).norm(dim=-1).mean() / self._dist(v, o).median().clamp_min(1e-6)
        self.trace.add_counter("joint_center_shift_ppm", int(round(float(shift) * 1e6)))
        if live is not None:
            score = score.masked_fill(~live, 0.0)
        return score

    def _greedy_set(self, query, key, v, live, w, center, k):
        """Idea B: indices [B, Hkv, k] of a set chosen for low kept-set output error.

        Error of keeping S for a query with weights p and output o:
        ||sum_{i in S} p_i (v_i - o)|| / A, A = sum_{i in S} p_i -- exact, no
        linearization. Seed with top k/2 by mass, then value_set_rounds rounds,
        each adding the entries whose addition gives the lowest error (summed
        over queries for "per_query").
        """
        cfg = self.config.selector
        if cfg.value_set == "pooled":
            p = w.unsqueeze(2)                                             # [B,H,1,n]
            o = center                                                     # [B,H,1,d]
        else:
            _, p, o, _ = self._query_terms(query, key, v, live)
        if live is not None:
            p = p.masked_fill(~live.unsqueeze(2), 0.0)
        dist2 = self._dist(v, o) ** 2                                      # [B,H,R,n]
        k0 = max(1, k // 2)
        base = w if live is None else w.masked_fill(~live, float("-inf"))
        chosen = torch.topk(base, k0, dim=-1).indices                      # [B,H,k0]
        taken = torch.zeros_like(w, dtype=torch.bool).scatter(-1, chosen, True)
        if live is not None:
            taken |= ~live

        def add(idx):
            pi = p.gather(-1, idx.unsqueeze(2).expand(-1, -1, p.shape[2], -1))   # [B,H,R,m]
            vi = v.gather(2, idx.unsqueeze(-1).expand(-1, -1, -1, v.shape[-1]))  # [B,H,m,d]
            r_add = torch.einsum("bhrm,bhmd->bhrd", pi, vi) - pi.sum(-1, keepdim=True) * o
            return r_add, pi.sum(-1)

        r, A = add(chosen)
        left = k - k0
        rounds = cfg.value_set_rounds
        for t in range(rounds):
            m = left // (rounds - t)
            left -= m
            if m <= 0:
                continue
            rv = torch.einsum("bhrd,bhnd->bhrn", r, v)
            ro = (r * o).sum(-1, keepdim=True)
            num = (r * r).sum(-1, keepdim=True) + 2.0 * p * (rv - ro) + p * p * dist2
            err = (num / (A.unsqueeze(-1) + p).clamp_min(1e-12) ** 2).sum(dim=2)   # [B,H,n]
            err = err.masked_fill(taken, float("inf"))
            new = torch.topk(-err, m, dim=-1).indices
            dr, dA = add(new)
            r, A = r + dr, A + dA
            taken = taken.scatter(-1, new, True)
            chosen = torch.cat([chosen, new], dim=-1)
        self.trace.add_counter("set_calls", 1)
        return chosen

    def _per_query_caote(self, query, key, v, live):
        """mean_q p_iq * ||v_i - o_q|| (x 1/(1 - p_iq) with value_caote). [B, Hkv, n].

        p_iq is the prefix softmax of every selector query on its own -- the
        same queries, scale and live set the selector averaged into its
        importance -- and o_q = sum_j p_jq v_j that query's own output. The
        distance goes through ||v||^2 - 2 v.o_q + ||o_q||^2, one matmul against
        all outputs instead of an [R, n, d] difference tensor. v is already
        zeroed off the live set.
        """
        from ..kernels.ops import _selected_queries

        if query is None:
            raise ValueError("value_per_query needs the selector queries")
        q = _selected_queries(query, self.query_indices, self.num_kv_heads).to(torch.float32)
        logits = torch.einsum("bhrd,bhnd->bhrn", q, key.to(torch.float32)) * self.head_dim**-0.5
        if live is not None:
            logits = logits.masked_fill(~live.unsqueeze(2), float("-inf"))
        p = torch.softmax(logits, dim=-1)                                  # [B, Hkv, R, n]
        o = torch.einsum("bhrn,bhnd->bhrd", p, v)                          # [B, Hkv, R, d]
        d2 = (
            (v * v).sum(-1).unsqueeze(2)
            - 2.0 * torch.einsum("bhrd,bhnd->bhrn", o, v)
            + (o * o).sum(-1).unsqueeze(-1)
        )
        dist = d2.clamp_min(0.0).sqrt()
        weight = p / (1.0 - p).clamp_min(1e-6) if self.config.selector.value_caote else p
        score = (weight * dist).mean(dim=2)
        if live is not None:
            score = score.masked_fill(~live, 0.0)
        return score

    def _select_and_gather(
        self,
        layer_idx: int,
        query: torch.Tensor,
        current_key: torch.Tensor,
        state: CompactLayerState,
    ) -> None:
        k = self._effective_budget()
        view = self.cache.layer_view(layer_idx)
        with self.timer.region(
            "selector",
            tensor=query,
            layer=layer_idx,
            block=self.block_index,
            step=self.step_index,
            stream_name="selector" if self._selection_stream is not None else "main",
            stream=self._selection_stream,
        ):
            live_mask = self._slot_mask(layer_idx)
            if live_mask is not None:
                live_mask = live_mask.bool()
            selected = selector_topk(
                query,
                view,
                live_mask=live_mask,
                query_indices=self.query_indices,
                topk=k,
                current_key=current_key,
                domain=self.config.selector.domain,
                score_kind=self.config.selector.score,
                sort_indices=self.config.selector.sort_indices,
                scaling=self.head_dim**-0.5,
                backend=self.config.backend,
                return_importance=(
                    self.config.collect_diagnostics
                    or self._eviction is not None
                    or self.config.selector.value_aware
                ),
                scratch=self.selector_scratch,
                kernel_variant=self.config.selector_kernel_variant,
                logits_dtype=(
                    torch.float32
                    if self.config.selector_logits_dtype == "float32"
                    else torch.float16
                ),
            )
            self._log_pairwise_selection(layer_idx, query, k)
            if self.config.rank_bf16_shadow:
                shadow = self._fp16_key_shadow.get(layer_idx)
                if shadow is None or shadow.shape[2] < view.length:
                    raise RuntimeError(
                        "rank_bf16_shadow needs BF16 shadow keys covering the selector prefix"
                    )
                shadow_scores = self._importance_from_keys(
                    shadow[:, :, :view.length, :], query,
                    query_indices=self.query_indices,
                )
                if shadow_scores is None:
                    raise RuntimeError("could not compute BF16-shadow selector scores")
                rank_on = shadow_scores if live_mask is None else shadow_scores.masked_fill(
                    ~live_mask, float("-inf")
                )
                selected.indices = torch.topk(rank_on, k=k, dim=-1, sorted=False).indices
                selected.importance = shadow_scores
                selected.values = shadow_scores.gather(-1, selected.indices)
        if self._la_pending is not None and selected.importance is not None and self._eviction is not None:
            # The selector already computed this block's mass over the live set:
            # it is exactly the lookahead's horizon-1 term, no second pass needed.
            self._la_a1[layer_idx] = selected.importance.to(torch.float32).gather(
                -1, self._live_positions(layer_idx)
            )
        if self.config.eviction.oracle_record_dir and selected.importance is not None:
            # Mass per physical position; with eviction off every position is live.
            self._oracle_rec.setdefault(self.block_index, {})[layer_idx] = (
                selected.importance[0].to(torch.float16).cpu()
            )
        if self.config.selector.value_aware:
            selected = self._rescore_by_value(layer_idx, selected, k, live_mask, query=query)

        with self.timer.region(
            "gather_dequant",
            tensor=query,
            layer=layer_idx,
            block=self.block_index,
            step=self.step_index,
            stream_name="selector" if self._selection_stream is not None else "main",
            stream=self._selection_stream,
        ):
            gather_packed_kv(
                view,
                selected.indices,
                dtype=self.compute_dtype,
                backend=self.config.backend,
                out_key=state.key[:, :, :k, :],
                out_value=state.value[:, :, :k, :],
            )
        if self._eviction is not None and selected.importance is not None:
            # alpha for the live set, in live-slot order. The selector scored
            # the physical cache, so read it back at the positions this layer
            # still holds.
            signal = selected.importance
            if self.config.eviction.policy == "ema_recent_score":
                signal = self._last_value_score
            alpha = signal.gather(-1, self._live_positions(layer_idx))
            self._eviction.observe(layer_idx, alpha, self.config.eviction.decay, len(self.query_indices))

        state.indices = selected.indices
        state.selected_k = k
        if self.config.coverage_diagnostics:
            self._record_coverage(layer_idx, query, selected.indices, k)
        if self.config.compact_format == "requantized":
            with self.timer.region(
                "requantize_compact",
                tensor=query,
                layer=layer_idx,
                block=self.block_index,
                step=self.step_index,
                stream_name="selector" if self._selection_stream is not None else "main",
                stream=self._selection_stream,
            ):
                self._requantize_compact(state)
        if self.config.collect_diagnostics:

            checksum = int(selected.indices.to(torch.int64).sum().item())
            mean_score = (
                float(selected.values.float().mean().item()) if selected.values.numel() else None
            )
        else:
            checksum = 0
            mean_score = None
        self.trace.selections.append(
            SelectionRecord(
                block=self.block_index,
                layer=layer_idx,
                old_cache_len=self.old_cache_len,
                selected_k=k,
                selector_queries=list(self.query_indices),
                semantic=self.config.semantic,
                index_checksum=checksum,
                mean_score=mean_score,
            )
        )

    def _schedule_a_selection(
        self,
        layer_idx: int,
        query: torch.Tensor,
        current_key: torch.Tensor,
    ) -> None:
        k = self._effective_budget()
        state = self._ensure_state(layer_idx, k)
        if state.indices is not None or state.ready_event is not None:
            return
        if self._selection_stream is None:
            self._select_and_gather(layer_idx, query, current_key, state)
            return

        main_stream = torch.cuda.current_stream(query.device)
        q_ready = torch.cuda.Event()
        q_ready.record(main_stream)
        with torch.cuda.stream(self._selection_stream):
            self._selection_stream.wait_event(q_ready)
            self._select_and_gather(layer_idx, query, current_key, state)
            done = torch.cuda.Event()
            done.record(self._selection_stream)
        state.ready_event = done

        state.pending_refs = (query, current_key)

    def finish_step0(self) -> None:
        if self._step0_finished:
            return
        if self._attn_log is not None:
            self._log_selection()
        if self._selection_stream is not None:
            main = torch.cuda.current_stream(self.cache.device)
            for state in self._states.values():
                if state.ready_event is not None:
                    main.wait_event(state.ready_event)
                    state.ready_event = None
                    state.pending_refs = None
        self._step0_finished = True
        cfg = self.config.eviction
        if self._la_pending is not None and len(self._la_q) == self._eviction.num_layers:
            merged_done = self._merge_extra and len(self._la_aprobe) == self._eviction.num_layers
            if cfg.horizon_blocks == 1 or merged_done:
                self._lookahead_evict(self._la_pending)
                self._la_pending = None
                self._la_q, self._la_a1, self._la_aprobe = {}, {}, {}
        self._merge_extra = 0

    def _compact_attention(
        self,
        layer_idx: int,
        query: torch.Tensor,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
    ) -> torch.Tensor:
        state = self._states.get(layer_idx)
        if state is None or state.selected_k <= 0:
            raise RuntimeError(
                f"layer {layer_idx} has no selected cache; finish_step0 was not called or selection failed"
            )
        k = state.selected_k
        ncur = current_key.shape[2]
        if self.config.compact_format == "requantized":
            if state.packed_view is None:
                raise RuntimeError("requantized compact view is missing")
            with self.timer.region(
                "compact_requantized_attention",
                tensor=query,
                layer=layer_idx,
                block=self.block_index,
                step=self.step_index,
            ):
                return dense_packed_attention(
                    query,
                    state.packed_view,
                    current_key,
                    current_value,
                    scaling=self.head_dim**-0.5,
                    backend=self.config.backend,
                    kernel_variant=self.config.dense_kernel_variant,
                )
        if k + ncur > state.key.shape[2]:
            raise RuntimeError("compact cache buffer capacity exceeded")
        state.key[:, :, k : k + ncur, :].copy_(current_key.to(self.compute_dtype))
        state.value[:, :, k : k + ncur, :].copy_(current_value.to(self.compute_dtype))
        query_compute = query.to(self.compute_dtype)
        with self.timer.region(
            "compact_attention",
            tensor=query,
            layer=layer_idx,
            block=self.block_index,
            step=self.step_index,
        ):
            output = sdpa_compact(
                query_compute,
                state.key[:, :, : k + ncur, :],
                state.value[:, :, : k + ncur, :],
            )
        return output.to(query.dtype)


    # -- fast path (kernels/splitk.py) -------------------------------------
    @property
    def fast(self) -> bool:
        """Split-K kernels for every attention over the cache (see _fast_supported)."""
        f = getattr(self, "_fast", None)
        if f is None:
            f = self._fast = self._fast_supported()
        return f

    def _fast_supported(self) -> bool:
        ok, why = self._fast_check()
        import os
        if os.environ.get("BITSIEVE_FAST_DEBUG"):
            print(f"[bitsieve] fast path: {ok} ({why})", flush=True)
        return ok

    def _fast_check(self):
        if self._selection_stream is not None:
            return False, "async selector"
        if self._eviction is not None and not self._phys_compact:
            return False, "logical eviction (BITSIEVE_COMPACT=0)"
        return fast_path_supported(self.config, self.head_dim, self.cache.device.type)

    def _prefix(self, layer_idx: int):
        from ..kernels.splitk import Prefix

        pre = Prefix.from_view(self.cache.layer_view(layer_idx))
        pre.mask = self._slot_mask(layer_idx)
        return pre

    def _rows(self, positions: list[int]) -> torch.Tensor:
        """int32 tensor of block positions, cached per distinct list."""
        key = tuple(positions)
        cached = getattr(self, "_rows_cache", None)
        if cached is None or cached[0] != key:
            t = torch.tensor(key, dtype=torch.int32).to(self.cache.device, non_blocking=True)
            self._rows_cache = cached = (key, t)
        return cached[1]

    def _fast_attend(self, layer_idx, query, current_key, current_value, *, causal=0, q_pos0=0,
                     mass_rows=None):
        """Attention over [live prefix | current]; optionally the prefix mass of mass_rows."""
        from ..kernels.splitk import prefix_scores, splitk_attention

        prefix = self._prefix(layer_idx)
        if prefix.n == 0:
            out = self._current_only(query, current_key, current_value, causal, q_pos0)
            return out, None
        out, stats = splitk_attention(query, prefix, current_key, current_value, scaling=self.head_dim**-0.5,
                                      causal_block=causal, q_pos0=q_pos0, c_pos0=q_pos0,
                                      want_prefix=mass_rows is not None)
        if mass_rows is None:
            return out, None
        mass, _ = prefix_scores(query, prefix, stats, mass_rows, value=False, scaling=self.head_dim**-0.5)
        return out, mass

    def _current_only(self, query, current_key, current_value, causal, q_pos0):
        from ..kernels.splitk import Prefix, splitk_attention

        empty = Prefix()
        out, _ = splitk_attention(query, empty, current_key, current_value, scaling=self.head_dim**-0.5,
                                  causal_block=causal, q_pos0=q_pos0, c_pos0=q_pos0)
        return out

    def _fast_step0(self, layer_idx, query, current_key, current_value, dense: bool):
        """Step 0: dense attention over the live prefix, and (sparse layers) the block's
        top-k by mass or mass * ||v - c||, all from one pass over the prefix."""
        from ..kernels.splitk import prefix_scores, splitk_attention

        ev = self._eviction
        need_sel = not dense and self.config.semantic == "A" and bool(self.query_indices)
        need_mass = bool(self.query_indices) and ev is not None and ev.live > 0 and (
            self._la_pending is not None or self.config.eviction.policy in ("ema_recent", "ema_recent_value")
        )
        prefix = self._prefix(layer_idx)
        if prefix.n == 0:
            return self._current_only(query, current_key, current_value, 0, 0)
        scale = self.head_dim**-0.5
        out, stats = splitk_attention(query, prefix, current_key, current_value, scaling=scale,
                                      want_prefix=need_sel or need_mass)
        if not (need_sel or need_mass):
            return out
        rows = self._rows(self.query_indices)
        value = need_sel and self.config.selector.value_aware
        mass, score = prefix_scores(query, prefix, stats, rows, value=value, scaling=scale)
        if ev is not None:
            live_mass = self._to_live(layer_idx, mass)
            if self._la_pending is not None:
                self._la_a1[layer_idx] = live_mass
            if self.config.eviction.policy in ("ema_recent", "ema_recent_value"):
                ev.observe(layer_idx, live_mass, self.config.eviction.decay, len(self.query_indices))
        if need_sel:
            k = self._effective_budget()
            state = self._ensure_state(layer_idx, k)
            rank = score if value else mass
            if prefix.mask is not None:
                rank = rank.masked_fill(prefix.mask == 0, float("-inf"))
            idx = rank.topk(k, dim=-1).indices
            if self.config.selector.sort_indices:
                idx = idx.sort(dim=-1).values
            used = stats[3]                          # the prefix pass 1 read (bf16 if it unpacked it)
            if used.nq == 0:
                gi = idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
                torch.gather(used.kf, 2, gi, out=state.key[:, :, :k, :])
                torch.gather(used.vf, 2, gi, out=state.value[:, :, :k, :])
            else:
                gather_packed_kv(self.cache.layer_view(layer_idx), idx, dtype=self.compute_dtype,
                                 backend=self.config.backend, out_key=state.key[:, :, :k, :],
                                 out_value=state.value[:, :, :k, :])
            state.indices = idx
            state.selected_k = k
            self.trace.selections.append(SelectionRecord(
                block=self.block_index, layer=layer_idx, old_cache_len=self.old_cache_len, selected_k=k,
                selector_queries=self.query_indices, semantic=self.config.semantic, index_checksum=0,
            ))
        return out

    def _fast_compact_attention(self, layer_idx, query, current_key, current_value):
        # The k selected entries + the block: sdpa over the compact buffer (one launch
        # plus two copies) beats any Triton launch at these sizes (measured on PPU).
        return self._compact_attention(layer_idx, query, current_key, current_value)

    def _live_mass(self, layer_idx: int, query: torch.Tensor) -> torch.Tensor:
        """Mean prefix-softmax mass of this step's query_indices on the live set. [B, Hkv, live]."""
        from ..kernels.ops import _selected_queries
        from ..reference import gather_per_kv_head

        key, _ = self.cache.dequantize_layer(layer_idx)
        k = gather_per_kv_head(key.to(torch.float32), self._live_positions(layer_idx))
        q = _selected_queries(query, self.query_indices, self.num_kv_heads).to(torch.float32)
        p = torch.softmax(torch.einsum("bhrd,bhnd->bhrn", q, k) * self.head_dim**-0.5, dim=-1)
        return p.mean(dim=2)

    def _dense_attention(
        self,
        layer_idx: int,
        query: torch.Tensor,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
    ) -> torch.Tensor:
        if self.old_cache_len == 0:
            return sdpa_compact(query, current_key, current_value)
        if self._eviction is not None and self.live_cache_len < self.old_cache_len:
            # Semantic A attends densely at step 0 -- over the *cache*, and
            # under eviction the cache is the live set. Reading the packed
            # buffer wholesale would attend to evicted entries and quietly
            # undo the policy.
            return self._live_dense_attention(
                layer_idx, query, current_key, current_value
            )
        with self.timer.region(
            "dense_packed_attention",
            tensor=query,
            layer=layer_idx,
            block=self.block_index,
            step=self.step_index,
        ):
            return dense_packed_attention(
                query,
                self.cache.layer_view(layer_idx),
                current_key,
                current_value,
                scaling=self.head_dim**-0.5,
                backend=self.config.backend,
                kernel_variant=self.config.dense_kernel_variant,
            )

    def _live_dense_attention(
        self,
        layer_idx: int,
        query: torch.Tensor,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
    ) -> torch.Tensor:
        """Dense attention over the live set: gather it, then attend normally."""
        assert self._eviction is not None
        live = self._eviction.live
        ncur = current_key.shape[2]
        need = live + ncur
        shape = (self.cache.batch_size, self.num_kv_heads, need, self.head_dim)
        if self._live_scratch is None or self._live_scratch.shape[2] < need:
            self._live_scratch = torch.empty(
                shape, device=self.cache.device, dtype=self.compute_dtype
            )
            self._live_scratch_v = torch.empty_like(self._live_scratch)
        key_buf = self._live_scratch[:, :, :need, :]
        value_buf = self._live_scratch_v[:, :, :need, :]

        with self.timer.region(
            "live_gather_dequant",
            tensor=query,
            layer=layer_idx,
            block=self.block_index,
            step=self.step_index,
        ):
            gather_packed_kv(
                self.cache.layer_view(layer_idx),
                self._live_positions(layer_idx).to(torch.int32),
                dtype=self.compute_dtype,
                backend=self.config.backend,
                out_key=key_buf[:, :, :live, :],
                out_value=value_buf[:, :, :live, :],
            )
        key_buf[:, :, live:need, :].copy_(current_key.to(self.compute_dtype))
        value_buf[:, :, live:need, :].copy_(current_value.to(self.compute_dtype))
        with self.timer.region(
            "live_dense_attention",
            tensor=query,
            layer=layer_idx,
            block=self.block_index,
            step=self.step_index,
        ):
            return sdpa_compact(query.to(self.compute_dtype), key_buf, value_buf)

    def set_rope(self, cos: torch.Tensor, sin: torch.Tensor) -> None:
        """The block's RoPE tables [B, T, d], so queries can be un-rotated."""
        self._rope = (cos, sin)

    def _ea_observe(self, layer_idx: int, query: torch.Tensor) -> None:
        """Fold this block's step-0 queries into the layer's query moments."""
        if self._rope is None:
            raise RuntimeError("expected_attention needs set_rope() before attend()")
        if not self.query_indices:
            return
        cos, sin = (x.to(torch.float32) for x in self._rope)
        q = query.to(torch.float32)
        c, s = cos.unsqueeze(1), sin.unsqueeze(1)
        # RoPE is y = x*cos + rot(x)*sin with rot^2 = -1, so x = y*cos - rot(y)*sin.
        pre = q * c - _rotate_half(q) * s
        x = pre[:, :, self.query_indices, :]                               # [B, Hq, R, d]
        m1 = x.mean(dim=2)
        if self.config.eviction.ea_diag:
            m2 = (x * x).mean(dim=2)                                       # [B, Hq, d]
        else:
            m2 = torch.einsum("bhrd,bhre->bhde", x, x) / x.shape[2]
        if self._ea_mu is None:
            L = self.cache.num_layers
            self._ea_mu = torch.zeros((L, *m1.shape), dtype=torch.float32, device=q.device)
            self._ea_m2 = torch.zeros((L, *m2.shape), dtype=torch.float32, device=q.device)
        lam = self.config.eviction.ea_decay
        self._ea_mu[layer_idx].mul_(lam).add_(m1, alpha=1.0 - lam)
        self._ea_m2[layer_idx].mul_(lam).add_(m2, alpha=1.0 - lam)
        self._ea_n[layer_idx] += 1
        if layer_idx == 0 and cos.shape[1] >= 2:
            # Angles are layer-independent. The per-position step comes from the
            # last two positions exactly (every RoPE step is below pi).
            c0, s0, c1, s1 = cos[0, -2], sin[0, -2], cos[0, -1], sin[0, -1]
            self._ea_freq = torch.atan2(s1 * c0 - c1 * s0, c1 * c0 + s1 * s0)
            self._ea_phase = torch.atan2(s1, c1)

    def flush_oracle(self) -> None:
        """Write the recorded per-block mass of this generation, if recording."""
        d = self.config.eviction.oracle_record_dir
        if not d or not self._oracle_rec:
            return
        import os
        os.makedirs(d, exist_ok=True)
        name = (CURRENT_EXAMPLE or "unnamed").replace("/", "__")
        torch.save(self._oracle_rec, os.path.join(d, name + ".pt"))

    def _oracle_score(self) -> torch.Tensor:
        """Future mass of the next oracle_horizon blocks (recorded run) x ||v - c_fut||.

        Layers or blocks the record lacks (dense layers 0-1, a recorded run that
        stopped earlier) fall back to ema_recent_value's ghat * ||v - c||.
        """
        import os
        from ..reference import gather_per_kv_head

        st, cfg = self._eviction, self.config.eviction
        if self._oracle_data is None:
            name = (CURRENT_EXAMPLE or "unnamed").replace("/", "__")
            path = os.path.join(cfg.oracle_dir, name + ".pt")
            # The recording run may still be on this example: wait for its
            # file rather than silently falling back to the EMA policy.
            import time
            waited = 0.0
            while not os.path.exists(path) and waited < 3600.0:
                time.sleep(5.0)
                waited += 5.0
            if not os.path.exists(path):
                raise FileNotFoundError(f"no oracle record for {name} after {waited:.0f}s")
            time.sleep(1.0)  # let torch.save finish the write
            self._oracle_data = torch.load(path)
            self.trace.add_counter("oracle_wait_s", int(waited))
        fallback = torch.stack(
            [st.corrected(l, cfg.decay) for l in range(st.num_layers)]
        ).to(torch.float32) * self._value_spread()
        out = fallback.clone()
        blocks = [self.block_index + d for d in range(1, cfg.oracle_horizon + 1)]
        hits = 0
        for l in range(st.num_layers):
            # the record is indexed by token position; the cache by physical slot
            pos = st.pos[l, ..., : st.live].to(torch.int64)                 # [B, H, n] token positions
            fut = torch.zeros(pos.shape, dtype=torch.float32, device=pos.device)
            got = False
            for b in blocks:
                a = self._oracle_data.get(b, {}).get(l)
                if a is None:
                    continue
                a = a.to(pos.device, torch.float32)                         # [H, N_b]
                ok = pos < a.shape[-1]
                fut += torch.where(ok, a.unsqueeze(0).expand(pos.shape[0], -1, -1)
                                   .gather(-1, pos.clamp_max(a.shape[-1] - 1)), torch.zeros_like(fut))
                got = True
            if not got:
                continue
            hits += 1
            out[l] = value_score(fut, self._live_values(l))
        self.trace.add_counter("oracle_evictions", 1)
        self.trace.add_counter("oracle_layers_hit", hits)
        return out

    def _probe_attention(self, layer_idx, query, current_key, current_value, q_start=None) -> torch.Tensor:
        """Dense attention of probe queries over live prefix + current keys, and their prefix mass.

        One pass gives both: the joint softmax output the forward needs, and the
        prefix-only softmax (averaged over the probe queries, times the number
        of probe blocks) that the eviction score uses.
        """
        from ..kernels.ops import _selected_queries
        from ..reference import gather_per_kv_head

        b, hq, t, d = query.shape
        g = hq // self.num_kv_heads
        key, value = self.cache.dequantize_layer(layer_idx)
        pos = self._live_positions(layer_idx)
        k = gather_per_kv_head(key.to(torch.float32), pos)
        v = gather_per_kv_head(value.to(torch.float32), pos)
        q = _selected_queries(query, list(range(t)), self.num_kv_heads).to(torch.float32)  # [B,H,g*t,d]
        scale = self.head_dim**-0.5
        lp = torch.einsum("bhrd,bhnd->bhrn", q, k) * scale
        lc = torch.einsum("bhrd,bhtd->bhrt", q, current_key.to(torch.float32)) * scale
        pp = torch.softmax(lp, dim=-1)
        if q_start is None:
            m_probe = max(1, t // self.config.generation.block_size)
            self._la_aprobe[layer_idx] = pp.mean(dim=2) * m_probe
            if self.config.precision_pairwise_diagnostics and self._la_pending is not None:
                self._la_probe_q[layer_idx] = q.detach()
            if self._attn_log is not None and layer_idx in self._attn_log_layers:
                blk = self.config.generation.block_size
                for horizon in range(m_probe):
                    lo, hi = horizon * blk, min((horizon + 1) * blk, t)
                    # _selected_queries groups query heads by KV head, then
                    # positions; reduce each horizon independently.
                    p_h = pp[:, :, :, lo * g:hi * g]
                    self._attn_log.append({
                        "kind": "probe_mass", "block": self.block_index,
                        "horizon": horizon + 1, "layer": layer_idx,
                        "positions": self._log_positions(layer_idx)[1][0].to(torch.int32).cpu(),
                        "mass": p_h.mean(dim=2)[0].to(torch.float16).cpu(),
                    })
        else:
            # Question probe: the probe is question text + answer masks at their
            # real positions, so it attends to itself block-causally, as the
            # prompt and the answer blocks will.
            blk = self.config.generation.block_size
            pq = (torch.arange(t, device=q.device) + q_start) // blk
            lc = lc.masked_fill(~(pq.unsqueeze(0) <= pq.repeat(g).unsqueeze(1)), float("-inf"))
            if self._q_span is not None and isinstance(self._q_span[0], torch.Tensor):
                # Different HotpotQA questions begin at different positions in
                # a same-length serving batch. Exclude preceding context from
                # each row's question-conditioned ranking score.
                starts = self._q_span[0].to(q.device)
                valid = (q_start + torch.arange(t, device=q.device))[None, :] >= starts[:, None]
                valid = valid.repeat(1, g).to(pp.dtype)[:, None, :, None]
                self._pf_qa[layer_idx] = (pp * valid).sum(dim=2) / valid.sum(dim=2).clamp_min(1)
            else:
                self._pf_qa[layer_idx] = pp.mean(dim=2)
        joint = torch.softmax(torch.cat([lp, lc], dim=-1), dim=-1)
        n = k.shape[2]
        out = torch.einsum("bhrn,bhnd->bhrd", joint[..., :n], v) + torch.einsum(
            "bhrt,bhtd->bhrd", joint[..., n:], current_value.to(torch.float32)
        )
        return out.reshape(b, self.num_kv_heads, g, t, d).reshape(b, hq, t, d).to(query.dtype)

    # -- chunked prefill ---------------------------------------------------
    def set_question(self, start: int | torch.Tensor, end: int) -> None:
        """Positions [start, end) of the prompt hold the question (the tail after the context)."""
        self._q_span = (start if isinstance(start, torch.Tensor) else int(start), int(end))
        if self._eviction is not None and self.config.eviction.pin_question:
            self._eviction.pin = self._q_span

    def _question_probe_on(self) -> bool:
        cfg = self.config.eviction
        return self._eviction is not None and cfg.prefill_question_probe and self._q_span is not None

    def wants_question_probe(self, final: bool) -> bool:
        """After a stored non-final chunk: rank the prefix with a question probe?"""
        if final or not self._question_probe_on():
            return False
        cfg = self.config.eviction
        cap = cfg.capacity(self._tokens_seen)
        if cfg.prefill_capacity_percent is not None:
            cap = max(int(cfg.prefill_capacity_percent / 100.0 * self._tokens_seen), cfg.capacity_floor)
        self._qprobe_cap = cap
        return self._eviction.live > cap

    def begin_question_probe(self, start: int) -> None:
        self._qprobe_start = int(start)
        self._pf_qa = {}

    def end_question_probe(self) -> None:
        """Evict to the prefill budget by sum over the probe of a * ||v - c||."""
        self._qprobe_start = None
        if len(self._pf_qa) == self._eviction.num_layers:
            self._evict_by_probe(self._qprobe_cap, self.config.eviction.lookahead_score != "mass")
        self._pf_qa = {}

    def begin_dapq_probe(self, start: int) -> None:
        """DapQ: the pending eviction ranks on pseudo tokens placed at the next block."""
        self.begin_question_probe(start)

    def end_dapq_probe(self) -> None:
        self._qprobe_start = None
        if self._la_pending is not None and len(self._pf_qa) == self._eviction.num_layers:
            self._evict_by_probe(self._la_pending, use_value=False)     # DapQ: sum of attention
        self._la_pending = None
        self._la_q, self._la_probe_q = {}, {}
        self._la_a1, self._la_aprobe = {}, {}
        self._pf_qa = {}

    def _evict_by_probe(self, capacity: int, use_value: bool) -> None:
        """Evict to capacity by the positioned probe's mass (optionally x ||v - c||)."""
        from ..reference import gather_per_kv_head

        st, cfg = self._eviction, self.config.eviction
        if st.live <= capacity:
            return
        score = torch.empty((st.num_layers, self.cache.batch_size, self.num_kv_heads, st.live),
                            dtype=torch.float32, device=self.cache.device)
        for l in range(st.num_layers):
            a = self._pf_qa[l]
            if not use_value:
                score[l] = a
            else:
                score[l] = value_score(a, self._live_values(l))
        self._log_score(score, "dapq" if not use_value else "question_probe", capacity)
        if cfg.record_only:
            return
        before = st.live
        self._evict_to(st.survivors(cfg, capacity, None, score))
        self.trace.add_counter("evictions", 1)
        self.trace.add_counter("entries_evicted", before - st.live)

    def begin_prefill_chunk(self, n: int) -> None:
        if self._eviction is None and not self.fast:
            raise RuntimeError("chunked prefill without eviction needs the fast path")
        self._admit_new_entries()           # the previous chunk (or the first, unpatched one)
        self._prefill_mode = True
        self._prefill_n = int(n)
        self.query_indices = list(range(n))
        self._pf_alpha = {}

    def _prefill_attention(self, layer_idx, query, current_key, current_value) -> torch.Tensor:
        """Chunk attends to the live prefix and, block-causally, to itself.

        Also records the chunk's prefix-only mass (mean over its queries, live
        axis) -- the observation every eviction policy scores the prefix with.
        """
        from ..kernels.ops import _selected_queries
        from ..reference import gather_per_kv_head

        if self.config.eviction.policy == "expected_attention":
            self._ea_observe(layer_idx, query)
        b, hq, t, d = query.shape
        g = hq // self.num_kv_heads
        key, value = self.cache.dequantize_layer(layer_idx)
        pos = self._live_positions(layer_idx)
        k = gather_per_kv_head(key.to(torch.float32), pos)
        v = gather_per_kv_head(value.to(torch.float32), pos)
        q = _selected_queries(query, list(range(t)), self.num_kv_heads).to(torch.float32)   # [B,H,g*t,d]
        scale = self.head_dim**-0.5
        lp = torch.einsum("bhrd,bhnd->bhrn", q, k) * scale
        lc = torch.einsum("bhrd,bhtd->bhrt", q, current_key.to(torch.float32)) * scale
        blk = self.config.generation.block_size
        tq = torch.arange(t, device=q.device).repeat(g)          # query row -> position in chunk
        tk = torch.arange(t, device=q.device)
        allowed = (tk.unsqueeze(0) // blk) <= (tq.unsqueeze(1) // blk)                    # [g*t, t]
        lc = lc.masked_fill(~allowed, float("-inf"))
        self._pf_alpha[layer_idx] = torch.softmax(lp, dim=-1).mean(dim=2)
        joint = torch.softmax(torch.cat([lp, lc], dim=-1), dim=-1)
        n = k.shape[2]
        out = torch.einsum("bhrn,bhnd->bhrd", joint[..., :n], v) + torch.einsum(
            "bhrt,bhtd->bhrd", joint[..., n:], current_value.to(torch.float32)
        )
        return out.reshape(b, self.num_kv_heads, g, t, d).reshape(b, hq, t, d).to(query.dtype)

    def _fast_prefill_attention(self, layer_idx, query, current_key, current_value):
        """Chunk vs live prefix + itself (block-causal); the chunk's prefix mass when a
        policy consumes it (EMA policies, or lookahead evicting chunk by chunk)."""
        cfg = self.config.eviction
        if cfg.policy == "snapkv":
            # SnapKV uses the final W prompt queries to score all prompt keys,
            # including the keys in the current chunk. The prompt remains intact
            # until this one compression event.
            from ..reference import gather_per_kv_head

            b, hq, t, d = query.shape
            n_old = self.cache.length
            q0 = max(0, t - cfg.snapkv_window)
            qpos = torch.arange(q0, t, device=query.device)
            q = query[:, :, q0:, :]
            g = hq // self.num_kv_heads
            q = q.reshape(b, self.num_kv_heads, g, q.shape[2], d).reshape(
                b, self.num_kv_heads, g * q.shape[2], d
            )
            old_key, _ = self.cache.dequantize_layer(layer_idx)
            old_slots = torch.arange(n_old, device=query.device).view(1, 1, -1).expand(
                b, self.num_kv_heads, -1
            )
            old = gather_per_kv_head(old_key, old_slots).to(q.dtype)
            cur = current_key.to(q.dtype)
            old_logits = torch.matmul(q, old.transpose(-1, -2)) * (d ** -0.5)
            cur_logits = torch.matmul(q, cur.transpose(-1, -2)) * (d ** -0.5)
            key_pos = torch.arange(t, device=query.device)
            allowed = (key_pos.unsqueeze(0) // self.config.generation.block_size
                       <= qpos.unsqueeze(1) // self.config.generation.block_size)
            cur_logits = cur_logits.masked_fill(~allowed.repeat(g, 1).unsqueeze(0).unsqueeze(0), float("-inf"))
            prob = torch.softmax(torch.cat((old_logits, cur_logits), dim=-1).float(), dim=-1)
            mass = prob.mean(dim=2).to(torch.float32)
            self._snapkv_score[layer_idx] = mass
            out, _ = self._fast_attend(layer_idx, query, current_key, current_value,
                                       causal=self.config.generation.block_size, q_pos0=0)
            return out
        if cfg.policy == "expected_attention":
            self._ea_observe(layer_idx, query)
        need = cfg.policy in ("ema_recent", "ema_recent_value") or (
            cfg.policy == "lookahead" and not self._question_probe_on()
            and (cfg.prefill_capacity_percent is None or cfg.prefill_capacity_percent < 100)
        )
        t = query.shape[2]
        out, mass = self._fast_attend(layer_idx, query, current_key, current_value,
                                      causal=self.config.generation.block_size, q_pos0=0,
                                      mass_rows=self._rows(list(range(t))) if need else None)
        if mass is not None:
            self._pf_alpha[layer_idx] = self._to_live(layer_idx, mass)
        return out

    def end_prefill_chunk(self, final: bool = False) -> None:
        """Evict the old prefix to the budget, scored by this chunk; then admit the chunk.

        Scoring per policy: EMA policies fold the chunk's mass into the EMA;
        lookahead ranks on the chunk's mass x ||v - c|| (the next chunk is the
        real future of the prefix); SnapKV pools attention from the last prompt
        window; EA on its query moments; recent on position.
        On the final chunk the lookahead eviction is deferred instead, to the
        first generated block's step 0 and its mask probe.
        """
        self._prefill_mode = False
        st, cfg = self._eviction, self.config.eviction
        if st is None:                      # chunking only (no eviction): nothing to score
            self._pf_alpha = {}
            return
        pol = cfg.policy
        if pol == "expected_attention" and cfg.record_only and final:
            self._admit_new_entries()
            cap = cfg.capacity(self._tokens_seen)
            self._log_score(self._ea_score(), "expected_attention", cap)
            self._pf_alpha = {}
            return
        if pol == "snapkv" and final:
            # The latest attention calculation includes the current final
            # chunk, so make those positions part of the candidate set first.
            self._admit_new_entries()
        if pol in ("ema_recent", "ema_recent_value", "ema_recent_score", "greedy_set", "oracle_future", "lookahead"):
            for l, a in self._pf_alpha.items():
                st.observe(l, a, cfg.decay, self._prefill_n)
        cap = cfg.capacity(self._tokens_seen)
        if not final and cfg.prefill_capacity_percent is not None:
            cap = max(int(cfg.prefill_capacity_percent / 100.0 * self._tokens_seen), cfg.capacity_floor)
        qprobe = self._question_probe_on() and not final
        if st.live > cap and not (final and pol == "lookahead") and not qprobe:
            before = st.live
            spread = self._value_spread() if pol == "ema_recent_value" else None
            score = None
            if pol == "expected_attention":
                score = self._ea_score()
            elif pol == "lookahead":
                score = torch.empty((st.num_layers, self.cache.batch_size, self.num_kv_heads, st.live),
                                    dtype=torch.float32, device=self.cache.device)
                from ..reference import gather_per_kv_head
                for l in range(st.num_layers):
                    v = self._live_values(l)                   # admitted slots only
                    a = self._pf_alpha[l]
                    score[l] = a if cfg.lookahead_score == "mass" else value_score(a, v)
            elif pol == "snapkv":
                import torch.nn.functional as F
                score = torch.empty((st.num_layers, self.cache.batch_size, self.num_kv_heads, st.live),
                                    dtype=torch.float32, device=self.cache.device)
                for l, a in self._snapkv_score.items():
                    pooled = F.max_pool1d(a, kernel_size=cfg.snapkv_pool_kernel,
                                          stride=1, padding=cfg.snapkv_pool_kernel // 2)
                    score[l] = pooled[..., :st.live]
            use = cfg
            if pol in ("greedy_set", "oracle_future", "ema_recent_score"):
                import dataclasses
                use = dataclasses.replace(cfg, policy="ema_recent")
            keep = st.survivors(use, cap, spread, score)
            self._evict_to(keep)
            self.trace.add_counter("prefill_evictions", 1)
            self.trace.add_counter("entries_evicted", before - st.live)
        self._admit_new_entries()
        if final and pol == "lookahead" and st.live > cfg.capacity(self._tokens_seen):
            self._la_pending = cfg.capacity(self._tokens_seen)
            self._la_q = {}
        self._pf_alpha = {}
        if pol == "snapkv":
            self._snapkv_score = {}

    def merge_probe_tokens(self) -> int:
        """Probe tokens the generator should append to this step-0 input (merged mode)."""
        cfg = self.config.eviction
        if (
            self._la_pending is None or not cfg.lookahead_merge
            or cfg.horizon_blocks <= 1 or self._eviction is None
        ):
            return 0
        self._merge_extra = (cfg.horizon_blocks - 1) * self.config.generation.block_size
        self._la_aprobe = {}
        return self._merge_extra

    def wants_probe(self) -> bool:
        """True right after a step 0 whose deferred eviction needs a mask probe."""
        return (
            self._la_pending is not None
            and self.config.eviction.horizon_blocks > 1
            and not self.config.eviction.lookahead_merge
            and self._eviction is not None
            and len(self._la_q) == self._eviction.num_layers
        )

    def begin_probe(self) -> None:
        self._probe = True
        self._la_probe_q = {}
        self._la_aprobe = {}

    def end_probe(self) -> None:
        self._probe = False
        if self._la_pending is not None and len(self._la_probe_q) == self._eviction.num_layers:
            self._lookahead_evict(self._la_pending)
        self._la_pending = None
        self._la_q = {}
        self._la_probe_q = {}
        self._la_a1, self._la_aprobe = {}, {}

    def _lookahead_evict(self, capacity: int) -> None:
        """Evict with the mass the current block's step-0 queries give each entry.

        Called after this block's step 0, so its selection already read the
        pre-eviction live set; the block keeps reading what it selected, and
        every later block sees only the survivors.
        """
        from ..reference import gather_per_kv_head

        st, cfg = self._eviction, self.config.eviction
        if st.live <= capacity:
            return
        score = torch.empty(
            (st.num_layers, self.cache.batch_size, self.num_kv_heads, st.live),
            dtype=torch.float32, device=self.cache.device,
        )
        use_value = self.config.eviction.lookahead_score != "mass"
        for l in range(st.num_layers):
            if l in self._la_a1 and not use_value:
                v = None
            elif l in self._la_a1:
                v = self._live_values(l)
            else:
                key, _ = self.cache.dequantize_layer(l)
                if not self._phys_compact:
                    from ..reference import gather_per_kv_head
                    key = gather_per_kv_head(key, self._live_positions(l))
                v = self._live_values(l)
            if l in self._la_a1:
                a = self._la_a1[l]                   # step-0 mass, same queries
            else:                                     # dense layers on the old path
                p = torch.softmax(
                    torch.einsum("bhrd,bhnd->bhrn", self._la_q[l].to(key.dtype), key).float()
                    * self.head_dim**-0.5, dim=-1
                )
                a = p.mean(dim=2)
            if l in self._la_aprobe:
                # Probe blocks k+2..k+M: already summed per block in _probe_attention,
                # so every block of the horizon counts once (as in the oracle).
                a = a + self._la_aprobe[l]
            score[l] = a if not use_value else value_score(a, v)
        self._log_score(score, "lookahead_m1" if cfg.horizon_blocks == 1 else "lookahead", capacity)
        self._log_pairwise_eviction(capacity)
        if cfg.record_only:
            return
        before = st.live
        keep = st.survivors(cfg, capacity, None, score)
        self._evict_to(keep)
        self.trace.add_counter("evictions", 1)
        self.trace.add_counter("entries_evicted", before - st.live)

    def _greedy_score(self, capacity: int) -> torch.Tensor:
        """1 on the C - W entries greedy_set keeps besides the recent window. [L, B, Hkv, live].

        All layers at once: the live count is shared, so keys, values and the
        last block's queries stack to [L, B, H, ...]; every layer still picks
        its own set. Kept-set error per query q with weights p_q, output o_q:
        ||sum_{i in S} p_qi (v_i - o_q)||^2 / A_q^2, summed over q; each round
        adds the greedy_chunk entries with the lowest error after adding them.
        """
        from ..reference import gather_per_kv_head

        st, cfg = self._eviction, self.config.eviction
        L, n = st.num_layers, st.live
        W = min(cfg.recent_window, capacity)
        missing = [l for l in range(L) if l not in self._greedy_q]
        if missing:
            # No queries yet (first blocks): rank on the mass EMA instead.
            return torch.stack([st.corrected(l, cfg.decay) for l in range(L)]).to(torch.float32)
        ks, vs = [], []
        for l in range(L):
            key, value = self.cache.dequantize_layer(l)
            pos = self._live_positions(l)
            ks.append(gather_per_kv_head(key.to(torch.float32), pos))
            vs.append(gather_per_kv_head(value.to(torch.float32), pos))
        k, v = torch.stack(ks), torch.stack(vs)                              # [L, B, H, n, d]
        q = torch.stack([self._greedy_q[l] for l in range(L)])               # [L, B, H, R, d]
        p = torch.softmax(torch.einsum("lbhrd,lbhnd->lbhrn", q, k) * self.head_dim**-0.5, dim=-1)
        o = torch.einsum("lbhrn,lbhnd->lbhrd", p, v)
        dist2 = (
            (v * v).sum(-1).unsqueeze(3)
            - 2.0 * torch.einsum("lbhrd,lbhnd->lbhrn", o, v)
            + (o * o).sum(-1).unsqueeze(-1)
        ).clamp_min(0.0)
        recent = st.pos[..., :n].topk(W, dim=-1).indices
        S = torch.zeros((L, *v.shape[1:3], n), dtype=torch.bool, device=v.device).scatter(-1, recent, True)
        pS = p * S.unsqueeze(3)
        r = torch.einsum("lbhrn,lbhnd->lbhrd", pS, v) - pS.sum(-1, keepdim=True) * o
        A = pS.sum(-1)
        chosen = torch.zeros_like(S)
        need = capacity - W
        while need > 0:
            m = min(cfg.greedy_chunk, need)
            rv = torch.einsum("lbhrd,lbhnd->lbhrn", r, v)
            ro = (r * o).sum(-1, keepdim=True)
            num = (r * r).sum(-1, keepdim=True) + 2.0 * p * (rv - ro) + p * p * dist2
            err = (num / (A.unsqueeze(-1) + p).clamp_min(1e-12) ** 2).sum(3).masked_fill(S, float("inf"))
            new = torch.topk(-err, m, dim=-1).indices
            pn = p.gather(-1, new.unsqueeze(3).expand(-1, -1, -1, p.shape[3], -1))
            vn = v.gather(3, new.unsqueeze(-1).expand(-1, -1, -1, -1, v.shape[-1]))
            r = r + torch.einsum("lbhrm,lbhmd->lbhrd", pn, vn) - pn.sum(-1, keepdim=True) * o
            A = A + pn.sum(-1)
            S = S.scatter(-1, new, True)
            chosen = chosen.scatter(-1, new, True)
            need -= m
        self.trace.add_counter("greedy_evictions", 1)
        return chosen.to(torch.float32)

    # -- eviction diagnostic ------------------------------------------------
    def _diag_probs(self, q, key):
        logits = torch.einsum("bhrd,bhnd->bhrn", q, key) * self.head_dim**-0.5
        return torch.softmax(logits, dim=-1)

    def _diag_snapshot(self, capacity: int) -> None:
        """Shadow keep-sets of size `capacity` for the diagnostic layers."""
        import dataclasses
        from ..reference import gather_per_kv_head

        st, cfg = self._eviction, self.config.eviction
        W = min(cfg.recent_window, capacity)
        keeps = {
            "ema_mass": st.survivors(dataclasses.replace(cfg, policy="ema_recent"), capacity),
            "ema_value": st.survivors(
                dataclasses.replace(cfg, policy="ema_recent_value", value_center="attn"),
                capacity, self._value_spread(),
            ),
        }
        pos_all = st.pos[..., : st.live].to(torch.int64)
        recent = pos_all.topk(W, dim=-1).indices                          # [L, B, H, W]
        snap = {"id": len(self._diag_snaps), "block": self.block_index, "left": 4, "layers": {},
                "n": st.live, "C": capacity}
        for l in self._diag_layers:
            if l not in self._diag_q:
                continue
            key, value = self.cache.dequantize_layer(l)
            pos = self._live_positions(l)
            k = gather_per_kv_head(key.to(torch.float32), pos)
            v = gather_per_kv_head(value.to(torch.float32), pos)
            q = self._diag_q[l]
            p = self._diag_probs(q, k)                                     # [B, H, R, n]
            a = p.mean(dim=2)
            o = torch.einsum("bhrn,bhnd->bhrd", p, v)
            n = k.shape[2]
            forced = torch.zeros(a.shape, dtype=torch.bool, device=a.device).scatter(-1, recent[l], True)
            sets = {name: torch.zeros_like(forced).scatter(-1, kp[l], True) for name, kp in keeps.items()}
            # top-(C - W) by averaged CAOTE-like score on block-k queries
            oc = (a.unsqueeze(-1) * v).sum(2, keepdim=True)
            sc = (a * (v - oc).norm(dim=-1)).masked_fill(forced, float("-inf"))
            top = sc.topk(capacity - W, dim=-1).indices
            sets["caote_topk"] = forced.clone().scatter(-1, top, True)
            # greedy on the exact kept-set error, per query, chunks of 8
            S = forced.clone()
            pS = p * S.unsqueeze(2)
            r = torch.einsum("bhrn,bhnd->bhrd", pS, v) - pS.sum(-1, keepdim=True) * o
            A = pS.sum(-1)
            dist2 = self._dist(v, o) ** 2
            need = capacity - W
            while need > 0:
                m = min(8, need)
                rv = torch.einsum("bhrd,bhnd->bhrn", r, v)
                ro = (r * o).sum(-1, keepdim=True)
                num = (r * r).sum(-1, keepdim=True) + 2.0 * p * (rv - ro) + p * p * dist2
                err = (num / (A.unsqueeze(-1) + p).clamp_min(1e-12) ** 2).sum(2).masked_fill(S, float("inf"))
                new = torch.topk(-err, m, dim=-1).indices
                pn = p.gather(-1, new.unsqueeze(2).expand(-1, -1, p.shape[2], -1))
                vn = v.gather(2, new.unsqueeze(-1).expand(-1, -1, -1, v.shape[-1]))
                r = r + torch.einsum("bhrm,bhmd->bhrd", pn, vn) - pn.sum(-1, keepdim=True) * o
                A = A + pn.sum(-1)
                S = S.scatter(-1, new, True)
                need -= m
            sets["greedy8"] = S
            evicted = ~sets["ema_mass"]
            snap["layers"][l] = {
                "k": k, "v": v, "a0": a, "sets": sets,
                "A_J": float((a * evicted).sum(-1).mean()),
                "A_J_max": float((a * evicted).sum(-1).max()),
                "J_frac": float(evicted.float().mean()),
            }
        self._diag_snaps.append(snap)

    def _diag_step0(self, layer_idx: int, query: torch.Tensor) -> None:
        import json
        from ..kernels.ops import _selected_queries

        q = _selected_queries(query, self.query_indices, self.num_kv_heads).to(torch.float32)
        self._diag_q[layer_idx] = q
        rows = []
        for snap in self._diag_snaps:
            if snap["left"] <= 0 or layer_idx not in snap["layers"]:
                continue
            L = snap["layers"][layer_idx]
            delta = self.block_index - snap["block"]
            if delta < 1 or delta > 4:
                continue
            p = self._diag_probs(q, L["k"])
            o_full = torch.einsum("bhrn,bhnd->bhrd", p, L["v"])
            den = o_full.norm(dim=-1).clamp_min(1e-12)
            err = {}
            for name, S in L["sets"].items():
                ps = p * S.unsqueeze(2)
                o_s = torch.einsum("bhrn,bhnd->bhrd", ps, L["v"]) / ps.sum(-1, keepdim=True).clamp_min(1e-12)
                err[name] = float(((o_s - o_full).norm(dim=-1) / den).mean())
            a_now, a0 = p.mean(2), L["a0"]
            x, y = a_now - a_now.mean(-1, keepdim=True), a0 - a0.mean(-1, keepdim=True)
            corr = float(((x * y).sum(-1) / (x.norm(dim=-1) * y.norm(dim=-1)).clamp_min(1e-12)).mean())
            rows.append({"snap": snap["id"], "layer": layer_idx, "delta": delta, "n": snap["n"], "C": snap["C"],
                         "A_J": L["A_J"], "A_J_max": L["A_J_max"], "J_frac": L["J_frac"],
                         "corr_a0": corr, "err": err})
        if layer_idx == max(self._diag_layers):
            for snap in self._diag_snaps:
                if 1 <= self.block_index - snap["block"] <= 4:
                    snap["left"] -= 1
            self._diag_snaps = [s_ for s_ in self._diag_snaps if s_["left"] > 0]
        if rows:
            with open(self.config.eviction.diag_path, "a", encoding="utf-8") as fh:
                for r_ in rows:
                    fh.write(json.dumps(r_) + "\n")

    def _ea_score(self) -> torch.Tensor:
        """(a_hat + eps) * ||v|| for every live entry. [L, B, Hkv, live]."""
        from ..reference import gather_per_kv_head

        assert self._eviction is not None
        st, cfg = self._eviction, self.config.eviction
        d = self.head_dim
        out = torch.empty(
            (st.num_layers, self.cache.batch_size, self.num_kv_heads, st.live),
            dtype=torch.float32, device=self.cache.device,
        )
        rbar = None
        if self._ea_freq is not None:
            # Average RoPE matrix over the next ea_horizon query positions:
            # Rbar = diag(cbar) + diag(sbar) * rot, per rotary pair.
            t = torch.arange(1, cfg.ea_horizon + 1, device=self._ea_freq.device, dtype=torch.float32)
            ang = self._ea_phase.unsqueeze(0) + t.unsqueeze(1) * self._ea_freq.unsqueeze(0)
            cbar, sbar = ang.cos().mean(0), ang.sin().mean(0)
            rot = _rotate_half(torch.eye(d, device=cbar.device)).T         # rot(x) = rot @ x
            rbar = torch.diag(cbar) + torch.diag(sbar) @ rot
        g = self.num_q_heads // self.num_kv_heads
        for layer_idx in range(st.num_layers):
            key, value = self.cache.dequantize_layer(layer_idx)
            pos = self._live_positions(layer_idx)
            k = gather_per_kv_head(key.to(torch.float32), pos)             # [B, Hkv, n, d]
            kc = self.cache.key_center
            if kc is not None and kc[layer_idx] is not None:
                k = k + kc[layer_idx].float()   # EA's quadratic term needs the keys themselves
            vnorm = gather_per_kv_head(value.to(torch.float32), pos).norm(dim=-1)
            n = self._ea_n[layer_idx]
            if n == 0 or rbar is None:
                # No queries seen yet: fall back to the attention-mass EMA.
                out[layer_idx] = st.corrected(layer_idx, cfg.decay).to(torch.float32)
                continue
            corr = 1.0 - cfg.ea_decay ** n
            mu = self._ea_mu[layer_idx] / corr                             # [B, Hq, d]
            if cfg.ea_diag:
                var = (self._ea_m2[layer_idx] / corr - mu * mu).clamp_min(0.0)
                cov = torch.diag_embed(var)
            else:
                cov = self._ea_m2[layer_idx] / corr - mu.unsqueeze(-1) * mu.unsqueeze(-2)
            mu_bar = mu @ rbar.T
            cov_bar = rbar @ cov @ rbar.T
            kq = k.repeat_interleave(g, dim=1)                             # [B, Hq, n, d]
            z = (kq @ mu_bar.unsqueeze(-1)).squeeze(-1) / d**0.5
            z = z + torch.einsum("bhnd,bhde,bhne->bhn", kq, cov_bar, kq) / (2.0 * d)
            a_hat = torch.softmax(z, dim=-1)
            a_hat = a_hat.view(a_hat.shape[0], self.num_kv_heads, g, -1).mean(dim=2)
            out[layer_idx] = (a_hat + cfg.ea_eps) * vnorm
        return out

    def attend(
        self,
        layer_idx: int,
        query: torch.Tensor,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
    ) -> torch.Tensor:
        kc = self.cache.key_center
        if kc is not None and kc[layer_idx] is not None:
            # quant.key_bias = mean: the stored keys are centred; centre the block's keys
            # the same way, so every logit of a query moves by the same q.mu.
            current_key = current_key - kc[layer_idx]
        rot = self.cache.rotation
        if rot is None:
            return self._attend_entry(layer_idx, query, current_key, current_value)
        # quant.rotation = hadamard: the cache holds K H and V H; q H . k H = q . k and the
        # output of rotated values is o H, rotated back here.
        out = self._attend_entry(layer_idx, (query.to(rot.dtype) @ rot).to(query.dtype),
                                 (current_key.to(rot.dtype) @ rot).to(current_key.dtype),
                                 (current_value.to(rot.dtype) @ rot).to(current_value.dtype))
        return (out.to(rot.dtype) @ rot.transpose(0, 1)).to(out.dtype)

    def _attend_entry(
        self,
        layer_idx: int,
        query: torch.Tensor,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
    ) -> torch.Tensor:
        if self._prefill_mode:
            if self.fast:
                return self._fast_prefill_attention(layer_idx, query, current_key, current_value)
            return self._prefill_attention(layer_idx, query, current_key, current_value)
        if self._qprobe_start is not None:
            if self.fast:
                t = query.shape[2]
                out, mass = self._fast_attend(layer_idx, query, current_key, current_value,
                                              causal=self.config.generation.block_size, q_pos0=self._qprobe_start,
                                              mass_rows=self._rows(list(range(t))))
                if mass is not None:
                    self._pf_qa[layer_idx] = self._to_live(layer_idx, mass)
                return out
            return self._probe_attention(layer_idx, query, current_key, current_value, q_start=self._qprobe_start)
        if (self._attn_log is not None and not self._probe
                and layer_idx in self._attn_log_layers and self.query_indices):
            self._log_attention(layer_idx, query)
        if self._merge_extra and query.shape[2] > self._merge_extra:
            # Merged lookahead: the block's own positions run the normal step 0
            # against the block's keys only; the appended probe positions see the
            # live prefix, the block and each other.
            t0 = query.shape[2] - self._merge_extra
            out = self._attend_main(layer_idx, query[:, :, :t0], current_key[:, :, :t0], current_value[:, :, :t0])
            out_p = self._probe_attention(layer_idx, query[:, :, t0:], current_key, current_value)
            return torch.cat([out, out_p], dim=2)
        return self._attend_main(layer_idx, query, current_key, current_value)

    def _attend_main(
        self,
        layer_idx: int,
        query: torch.Tensor,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
    ) -> torch.Tensor:
        if self._probe:
            # Lookahead probe: masks of blocks k+1..k+M in one forward. Keep the
            # queries of blocks k+2.. (k+1's real ones were taken at step 0) and
            # attend densely over the live set; no selection, no EMA, no commit.
            from ..kernels.ops import _selected_queries

            B = self.config.generation.block_size
            self._la_probe_q[layer_idx] = True
            if self.fast:
                t = query.shape[2]
                # block-causal, as the model will see these blocks: block j reads blocks <= j
                out, mass = self._fast_attend(layer_idx, query, current_key, current_value,
                                              causal=B, q_pos0=0, mass_rows=self._rows(list(range(B, t))))
                if mass is not None:
                    self._la_aprobe[layer_idx] = self._to_live(layer_idx, mass) * max(1, (t - B) // B)
                return out
            out = self._dense_attention(layer_idx, query[:, :, :B], current_key, current_value)
            out_p = self._probe_attention(layer_idx, query[:, :, B:], current_key, current_value)
            return torch.cat([out, out_p], dim=2)
        if (
            self.step_index == 0
            and self._eviction is not None
            and self.config.eviction.policy == "expected_attention"
        ):
            self._ea_observe(layer_idx, query)
        if self.step_index == 0 and self._la_pending is not None and self.query_indices:
            from ..kernels.ops import _selected_queries

            self._la_q[layer_idx] = _selected_queries(
                query, self.query_indices, self.num_kv_heads
            ).to(torch.float32)
        if (
            self.step_index == 0
            and self._eviction is not None
            and self.config.eviction.policy == "greedy_set"
            and self.query_indices
        ):
            from ..kernels.ops import _selected_queries

            self._greedy_q[layer_idx] = _selected_queries(
                query, self.query_indices, self.num_kv_heads
            ).to(torch.float32)
        if (
            self.step_index == 0
            and self._eviction is not None
            and self.config.eviction.diag_path
            and layer_idx in self._diag_layers
            and self.query_indices
        ):
            self._diag_step0(layer_idx, query)
        dense = self._dense_required(layer_idx)
        self.trace.add_counter(
            "layer_steps_dense" if dense else "layer_steps_sparse", 1
        )
        if self.fast:
            if dense:
                state = self._states.get(layer_idx)
                if state is not None:
                    state.use_dense = True
            if self.step_index == 0:
                if self.config.semantic == "A":
                    self._block_value = current_value
                    self._block_key = current_key
                return self._fast_step0(layer_idx, query, current_key, current_value, dense)
            if dense:
                return self._fast_attend(layer_idx, query, current_key, current_value)[0]
            return self._fast_compact_attention(layer_idx, query, current_key, current_value)
        if dense:
            state = self._states.get(layer_idx)
            if state is not None:
                state.use_dense = True
            if (
                self.step_index == 0
                and self._eviction is not None
                and self.config.eviction.policy in ("ema_recent", "ema_recent_value")
                and self.query_indices
                and self._eviction.live
            ):
                # Dense layers run no selector, so their mass for the EMA is
                # computed here (the sparse layers get it from the selector).
                self._eviction.observe(layer_idx, self._live_mass(layer_idx, query),
                                       self.config.eviction.decay, len(self.query_indices))
            return self._dense_attention(layer_idx, query, current_key, current_value)

        if self.step_index == 0:
            if self.config.semantic == "A":
                self._block_value = current_value
                self._block_key = current_key
                self._schedule_a_selection(layer_idx, query, current_key)
                return self._dense_attention(layer_idx, query, current_key, current_value)
            if self.config.semantic == "B":
                k = self.config.selector.effective_topk(self.old_cache_len)
                state = self._ensure_state(layer_idx, k)
                if state.selected_k == 0:
                    self._select_and_gather(layer_idx, query, current_key, state)
                return self._compact_attention(
                    layer_idx, query, current_key, current_value
                )
            raise AssertionError(f"unexpected semantic {self.config.semantic}")


        return self._compact_attention(layer_idx, query, current_key, current_value)

    def stage_if_committing(
        self,
        layer_idx: int,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
    ) -> None:
        if not self.commit_current:
            return
        if not self.cache.append_in_progress:
            raise RuntimeError("commit forward ran without begin_append")
        expected = self._prefill_n if self._prefill_mode else self.config.generation.block_size
        if current_key.shape[2] != expected:
            raise RuntimeError(
                f"commit forward must materialize a full block ({expected}), got {current_key.shape[2]}"
            )
        self.cache.stage_layer(
            layer_idx,
            current_key.detach(),
            current_value.detach(),
        )
        self._extend_fp16_key_shadow(layer_idx, current_key)

    def _value_spread(self) -> torch.Tensor:
        """||v_i - vbar_head|| for every live entry. [L, B, Hkv, live].

        Computed from the cache as it is actually stored -- dequantized, so at
        4 bits this is the spread the model will really see, not the bf16 one.
        vbar is the mean over the live set only: an evicted entry must not keep
        influencing the centre it is no longer part of. With
        eviction.value_center="attn" it is weighted by the bias-corrected EMA
        ghat, so the centre is the head's output rather than its plain average.
        """
        from ..reference import gather_per_kv_head

        assert self._eviction is not None
        st = self._eviction
        out = torch.empty(
            (st.num_layers, self.cache.batch_size, self.num_kv_heads, st.live),
            dtype=torch.float32,
            device=self.cache.device,
        )
        for layer_idx in range(st.num_layers):
            _, value = self.cache.dequantize_layer(layer_idx)
            live = gather_per_kv_head(
                value.to(torch.float32), self._live_positions(layer_idx)
            )
            if self.config.eviction.value_center == "attn":
                w = st.corrected(layer_idx, self.config.eviction.decay).to(torch.float32)
                center = (w.unsqueeze(-1) * live).sum(dim=2, keepdim=True) / w.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-12).unsqueeze(-1)
            elif self.config.eviction.value_center == "none":
                center = torch.zeros_like(live[:, :, :1, :])
            else:
                center = live.mean(dim=2, keepdim=True)
            out[layer_idx] = (live - center).norm(dim=-1)
        return out

    def end_block(self) -> None:
        self.finish_step0()
        self.commit_current = False
        if self._eviction is not None:
            self._admit_new_entries()
            cfg = self.config.eviction
            if cfg.policy == "snapkv":
                # SnapKV compresses the prompt once; generated tokens then grow
                # the cache autoregressively without further eviction.
                return
            before = self._eviction.live
            if (
                cfg.diag_path
                and (self.block_index + 1) % cfg.interval_blocks == 0
                and self._eviction.live > cfg.capacity(self._tokens_seen)
            ):
                self._diag_snapshot(cfg.capacity(self._tokens_seen))
            if cfg.policy == "lookahead":
                # Defer to the next block's step 0, where its real queries exist.
                if (
                    (self.block_index + 1) % cfg.interval_blocks == 0
                    and self._eviction.live > cfg.capacity(self._tokens_seen)
                ):
                    self._la_pending = cfg.capacity(self._tokens_seen)
                    self._la_q = {}
                return
            if (cfg.record_only and cfg.policy == "expected_attention"
                    and (self.block_index + 1) % cfg.interval_blocks == 0
                    and self._eviction.live > cfg.capacity(self._tokens_seen)):
                cap = cfg.capacity(self._tokens_seen)
                self._log_score(self._ea_score(), "expected_attention", cap)
                return
            keep = self._eviction.maybe_evict(
                cfg,
                block_index=self.block_index,
                tokens_seen=self._tokens_seen,
                value_spread_fn=(
                    self._value_spread if cfg.policy == "ema_recent_value" else None
                ),
                score_fn=(
                    self._ea_score if cfg.policy == "expected_attention"
                    else self._greedy_score if cfg.policy == "greedy_set"
                    else self._oracle_score if cfg.policy == "oracle_future" else None
                ),
            )
            if keep is not None:
                self._evict_to(keep, state_done=True)
                self.trace.add_counter("evictions", 1)
                self.trace.add_counter("entries_evicted", before - self._eviction.live)

    def eviction_metrics(self) -> dict[str, Any]:
        """What the policy kept, and what that costs. {} when eviction is off."""
        if self._eviction is None:
            return {}
        st = self._eviction
        physical = self._phys_compact
        cache_bytes = self.cache.logical_nbytes()["total"] if physical else live_cache_nbytes(
            st.pos,
            live=st.live,
            head_dim=self.head_dim,
            k_bits=self.config.quant.k_bits,
            v_bits=self.config.quant.v_bits,
            key_token_group=self.config.quant.key_token_group,
            value_channel_group=self.config.quant.value_channel_group,
            param_bytes=torch.tensor([], dtype=self.cache.param_dtype).element_size(),
            compute_bytes=torch.tensor([], dtype=self.compute_dtype).element_size(),
        )
        state_bytes = st.state_nbytes()
        ea_bytes = 0
        if self._ea_mu is not None:
            ea_bytes = self._ea_mu.numel() * 4 + self._ea_m2.numel() * 4
        # What the same run would hold with no eviction: every token it ever saw.
        full = live_cache_nbytes(
            torch.arange(self._tokens_seen, device=st.pos.device, dtype=st.pos.dtype)
            .view(1, 1, 1, -1)
            .expand(st.num_layers, st.batch_size, st.num_kv_heads, self._tokens_seen)
            .contiguous(),
            live=self._tokens_seen,
            head_dim=self.head_dim,
            k_bits=self.config.quant.k_bits,
            v_bits=self.config.quant.v_bits,
            key_token_group=self.config.quant.key_token_group,
            value_channel_group=self.config.quant.value_channel_group,
            param_bytes=torch.tensor([], dtype=self.cache.param_dtype).element_size(),
            compute_bytes=torch.tensor([], dtype=self.compute_dtype).element_size(),
        ) if self._tokens_seen else 0
        total = cache_bytes + state_bytes
        return {
            "eviction_policy": self.config.eviction.policy,
            "eviction_live_entries": st.live,
            "eviction_tokens_seen": self._tokens_seen,
            "eviction_capacity": self.config.eviction.capacity(self._tokens_seen),
            "eviction_cache_bytes": cache_bytes,
            "eviction_state_bytes": state_bytes,
            # Share of (layer, head) eviction events that dropped a position < 4.
            "sink_evicted_frac": (
                getattr(st, "sink_lanes_evicted", 0) / st.lanes_evictions
                if getattr(st, "lanes_evictions", 0) else None
            ),
            # Per-head query moments of expected_attention: fixed size, not per
            # entry, and not folded into eviction_total_bytes.
            "ea_stats_bytes": ea_bytes,
            "eviction_total_bytes": total,
            # Everything the method keeps at its largest point of the run: live
            # entries at their maximum (e.g. the whole prompt before the end-of-
            # prefill compression) with their per-entry state, plus fixed state
            # (EA's query moments). The number to compare methods on.
            "eviction_peak_live_entries": getattr(self, "_peak_live", st.live),
            # With physical compaction: the measured peak of the bytes the cache entries
            # occupied (cache.peak_used_bytes), plus the per-entry state at the peak live
            # count and EA's fixed state. Without it: the old per-live-entry estimate.
            "eviction_peak_bytes": (
                self.cache.peak_used_bytes
                + int(state_bytes / max(st.live, 1) * getattr(self, "_peak_live", st.live)) + ea_bytes
                if physical else
                int((cache_bytes + state_bytes) / max(st.live, 1) * getattr(self, "_peak_live", st.live)) + ea_bytes
            ),
            "cache_peak_allocated_bytes": self.cache.peak_allocated_bytes,
            "cache_allocated_bytes": self.cache.allocated_nbytes(),
            "bf16_equiv_bytes": self.cache.bf16_equivalent_nbytes(),
            "eviction_unevicted_bytes": full,
            # The number the study reports: what the bounded cache costs against
            # the same run keeping everything, policy overhead included.
            "eviction_bytes_vs_unevicted": (total / full) if full else None,
            "eviction_state_overhead": (state_bytes / cache_bytes) if cache_bytes else None,
        }

    def finalize_trace(self) -> RunTrace:
        self.timer.finalize()
        if self._attn_log is not None:
            out = Path(self.config.attn_log_dir)
            out.mkdir(parents=True, exist_ok=True)
            name = (CURRENT_EXAMPLE or "unnamed").replace("/", "__")
            torch.save({"config": self.config.name, "layers": sorted(self._attn_log_layers),
                        "records": self._attn_log}, out / f"{name}.pt")
            self._attn_log = []
        return self.trace

    # -- attention-map logging (config.attn_log_dir) -------------------------------
    def _log_positions(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(physical slots, token positions) of the live entries of one layer. [B, Hkv, n] each."""
        n = self.cache.get_seq_length() if self._eviction is None else self._eviction.live
        if self._eviction is None:
            ar = torch.arange(self.cache.length, device=self.cache.device)
            ar = ar.view(1, 1, -1).expand(self.cache.batch_size, self.num_kv_heads, -1)
            return ar, ar
        st = self._eviction
        return (self._live_positions(layer_idx),
                st.pos[layer_idx, ..., : st.live].to(torch.int64))

    def _log_attention(self, layer_idx: int, query: torch.Tensor) -> None:
        """Per query head: mean prefix-softmax mass of the block's masked queries on every live entry."""
        from ..reference import gather_per_kv_head

        if self.cache.length == 0:
            return
        slots, pos = self._log_positions(layer_idx)
        key, _ = self.cache.dequantize_layer(layer_idx)
        k = gather_per_kv_head(key, slots)                                   # [B, Hkv, n, d] bf16
        g = self.num_q_heads // self.num_kv_heads
        rows = [i for i in self.masked_positions if 0 <= i < query.shape[2]]
        if not rows:
            return
        q = query[:, :, rows, :].to(k.dtype)                                 # [B, Hq, R, d]
        kq = k.repeat_interleave(g, dim=1)                                   # [B, Hq, n, d]
        p = torch.softmax((q @ kq.transpose(-1, -2)).float() * self.head_dim**-0.5, dim=-1)
        self._attn_log.append({
            "kind": "mass", "block": self.block_index, "layer": layer_idx,
            "step": self.step_index,
            "positions": pos[0].to(torch.int32).cpu(),                         # [Hkv, n] token positions
            "kv_mass": p.reshape(p.shape[0], self.num_kv_heads, g, p.shape[2], p.shape[3])
                         .mean(dim=(2, 3))[0].to(torch.float16).cpu(),          # [Hkv, n]
            "n_queries": len(rows),
        })

    def _log_selection(self) -> None:
        for layer_idx in sorted(self._attn_log_layers):
            state = self._states.get(layer_idx)
            if state is None or state.indices is None or state.selected_k <= 0:
                continue
            _, pos = self._log_positions(layer_idx)
            idx = state.indices[..., : state.selected_k].to(torch.int64)
            idx = idx.clamp_max(pos.shape[-1] - 1)
            self._attn_log.append({
                "kind": "selected", "block": self.block_index, "layer": layer_idx,
                "positions": pos.gather(-1, idx)[0].to(torch.int32).cpu(),     # [Hkv, k] token positions
            })

    def _log_eviction(self, keep: torch.Tensor, state_done: bool) -> None:
        st = self._eviction
        for layer_idx in sorted(self._attn_log_layers):
            if layer_idx >= st.num_layers:
                continue
            if state_done:
                kept = st.pos[layer_idx, ..., : st.live]
            else:
                kept = st.pos[layer_idx, ..., : st.live].gather(-1, keep[layer_idx].to(torch.int64))
            self._attn_log.append({
                "kind": "kept", "block": self.block_index, "layer": layer_idx,
                "seen": int(self._tokens_seen),
                "positions": kept[0].to(torch.int32).cpu(),                     # [Hkv, C] token positions
            })

    def _log_score(self, score: torch.Tensor, source: str, capacity: int) -> None:
        """Save per-layer/per-KV-head scores for offline keep-set comparisons."""
        if self._attn_log is None or self._eviction is None:
            return
        st = self._eviction
        for layer_idx in sorted(self._attn_log_layers):
            if layer_idx >= st.num_layers:
                continue
            self._attn_log.append({
                "kind": "eviction_score", "source": source, "block": self.block_index,
                "layer": layer_idx, "seen": int(self._tokens_seen), "capacity": int(capacity),
                "positions": st.pos[layer_idx, ..., :st.live][0].to(torch.int32).cpu(),
                "score": score[layer_idx, ..., :st.live][0].to(torch.float16).cpu(),
            })
