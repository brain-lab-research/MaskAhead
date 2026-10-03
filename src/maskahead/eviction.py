"""Bounded KV cache: keep a live set instead of the whole prefix.

Selection already decides what a *block* reads. Eviction decides what the cache
*keeps*, which is a different question: a prefix entry no selector has looked at
for twenty blocks is paying for itself every block it stays.

State, per (layer, KV head), one slot per live entry:

    g    fp32    EMA of the attention mass the entry has been receiving
    n    uint16  age, in blocks, since the entry entered the cache
    pos  int32   the entry's original position in the cache

At step 0 of every block the selector already computes ``alpha`` -- the softmax
mass each prefix entry receives from the block's representative queries -- so
the update costs nothing beyond the arithmetic:

    g[i]  <- lam * g[i] + (1 - lam) * alpha[i]
    n[i]  <- n[i] + 1
    ghat  =  g[i] / (1 - lam ** n[i])

The ``1 - lam**n`` correction is not optional. ``g`` starts at zero, so without
it a brand-new entry reads as near-worthless purely because it has been
observed once, and any policy with a recency window would spend that window
undoing the artifact rather than protecting genuinely useful new tokens. With
the correction a first observation yields exactly ``ghat = alpha``.

Every ``interval_blocks`` blocks, if the live set exceeds
``C = max(capacity_percent% of tokens seen, capacity_floor)``:

    recent      keep the C newest by ``pos``
    ema_recent  keep the W newest by ``pos``, plus the top ``C - W`` by ``ghat``
                among the rest
    ema_recent_score
                as ema_recent, but the EMA folds in the value-aware selector
                score ``alpha * ||v - o_t||`` of every block (o_t that block's
                attention-weighted centre) instead of ``alpha`` alone -- the
                selector's own criterion, averaged over time

Eviction compacts the state arrays; it is not a mask. Evicted entries leave the
candidate set entirely, so they are not ranked, not read, and not counted --
see ``docs/eviction.md`` for what "not counted" means against a cache whose
buffers are preallocated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch

Policy = Literal["none", "recent", "ema_recent", "ema_recent_value", "ema_recent_score", "expected_attention", "greedy_set", "oracle_future", "lookahead", "snapkv"]


@dataclass(slots=True)
class EvictionConfig:
    policy: Policy = "none"
    decay: float = 0.9              # lambda; 1.0 = accumulated sum over queries (H2O)
    recent_window: int = 128        # W
    capacity_percent: float = 5.0
    capacity_floor: int = 256
    interval_blocks: int = 4
    # Centre of ||v - centre|| for ema_recent_value. "mean": plain mean over the
    # live set. "attn": the ghat-weighted mean, i.e. the head's output averaged
    # over the blocks the EMA has seen -- the same centre as the selector's
    # value_center="attn", so evicting entry i is scored by what it moves.
    # "none": no centre, rank on ghat * ||v|| (the entry's own contribution).
    value_center: Literal["mean", "attn", "none"] = "mean"
    # CAOTE on the EMA (the paper's H2O+CAOTE): ghat normalized to sum 1 per
    # head over the live set, then scaled by 1 / (1 - ghat_norm).
    value_caote: bool = False
    # expected_attention (Devoto et al., arXiv:2510.00636): rank on
    # (a_hat + eps) * ||v||, a_hat the softmax of the log-normal expectation
    # mu.k/sqrt(d) + k^T Sigma k / (2d) under a Gaussian fit of future queries.
    # mu, Sigma: discounted moments of pre-RoPE step-0 queries, rotated by the
    # RoPE matrix averaged over the next ea_horizon positions.
    ea_horizon: int = 128
    ea_decay: float = 0.9
    ea_eps: float = 0.02
    # Keep only per-channel variances instead of the full d x d covariance:
    # 2*d floats per query head instead of d + d^2 (51.8 MB -> 0.8 MB on
    # Fast-dLLM). Rotation by Rbar still mixes each rotary pair.
    ea_diag: bool = False
    # Eviction diagnostic: at each eviction snapshot the live cache of a few
    # layers, build shadow keep-sets of the same size by several policies, and
    # for the next 4 blocks write their output error against the full snapshot
    # (JSONL at this path). The run itself keeps evicting by `policy`.
    diag_path: str | None = None
    # greedy_set: keep the W newest, fill the other C - W greedily by the exact
    # kept-set output error on the last block's step-0 queries (per query,
    # cross terms included), greedy_chunk entries per round, all layers batched.
    greedy_chunk: int = 8
    # Oracle study. oracle_record_dir: write the selector's attention mass per
    # (block, layer) to <dir>/<example>.pt (works with eviction off).
    # policy oracle_future: evict by the mass the next oracle_horizon blocks
    # gave each position in that recorded run, times ||v - c_future||.
    oracle_record_dir: str | None = None
    oracle_dir: str | None = None
    oracle_horizon: int = 4
    # Diagnostic-only eviction: compute and log scheduled decisions, but keep
    # the full cache so future attention remains an uncompressed reference.
    record_only: bool = False
    # SnapKV prompt compression: score context positions from the last W prompt
    # queries, then max-pool neighboring token scores before selecting C.
    snapkv_window: int = 32
    snapkv_pool_kernel: int = 5
    # Never evict cache positions < sink_tokens (StreamingLLM-style attention
    # sinks; on Fast-dLLM v2 the first prompt tokens take 25-40% of the prefix
    # mass from layer 5 on). They are kept on top of the W newest.
    sink_tokens: int = 0
    # lookahead: an eviction due at the end of block k waits for block k+1's
    # step 0 and ranks on the mass those real queries give each live entry,
    # times ||v - c|| (c their attention-weighted centre). lookahead_blocks > 1
    # adds probe blocks of masks further ahead (not implemented yet).
    lookahead_blocks: int = 1   # <= 0: use interval_blocks (probe exactly up to the next eviction)
    # What fills the probe blocks: "mask" (the dLLM's own input for unwritten
    # positions), "random" (uniform vocabulary tokens), or "dapq" (DapQ's F4-L28:
    # the first 4 prompt tokens + the last 28 context tokens, repeated).
    #   "dapq" is DapQ itself: no masks and no real queries, only 32 pseudo tokens
    #   (F4-L28) at the positions of the next block, ranked by their attention mass.
    #   "dapq_fill" = our lookahead with the F4-L28 tokens filling the probe blocks.
    lookahead_probe: str = "mask"
    # Lookahead eviction score: "value" = sum_d a_d * ||v - c|| (default), "mass" = sum_d a_d.
    lookahead_score: str = "value"
    # Run the probe inside the next block's step-0 forward (one forward of
    # block * M tokens) instead of a separate forward. The block itself never
    # sees the probe positions; probe queries see the block and each other.
    lookahead_merge: bool = False
    # Chunked prefill: prefill the prompt through the session in chunks of this
    # many tokens (multiple of block_size), evicting to the budget after each,
    # so the cache never holds the whole prompt. 0 = one full prefill (default).
    prefill_chunk: int = 0
    # Budget while the prompt is still being prefilled, as a percent of what
    # has been seen (floored by capacity_floor). 100 keeps the whole prompt
    # until its end and compresses once to C there (the usual long-prompt
    # protocol: the question at the end of the prompt is seen before anything
    # is dropped). None = capacity_percent, i.e. evict chunk by chunk.
    prefill_capacity_percent: float | None = None
    # Question-aware chunked prefill (needs the question span, which the eval
    # passes as generate(question_start=...); ignored when it is unknown).
    # prefill_question_probe: after every stored chunk, rank the live prefix by
    # a probe forward of the question tokens at their real positions (end of the
    # prompt) plus horizon_blocks mask blocks of the answer, nothing stored --
    # DapQ with the real question instead of pseudo tokens, plus our mask probe;
    # the score is sum a * ||v - c|| over the probe queries (lookahead_score).
    # Works with every scored policy; with the budget prefill_capacity_percent.
    prefill_question_probe: bool = False
    # Never evict the question span (like sinks), in prefill and generation.
    pin_question: bool = False
    # Physical cache after an eviction (packed caches): "lazy" = key groups keeping at
    # least repack_below of their entries are kept bit-exactly (dead slots masked), the
    # rest re-packed; "full" = every survivor re-quantized into fresh groups at every
    # eviction; "none" = logical eviction only (dead slots stay allocated).
    compaction: str = "lazy"
    # Default 1.0 = group-preserving compaction (chosen on MATH x300 + LongBench@50, 24.09):
    # a key group is kept bit-exactly only while all its entries live; the survivors of a
    # group that lost an entry are re-quantized once into a fresh group.
    repack_below: float = 1.0
    # Quantize once: entries stay bf16 until their first eviction decision ("gen": entries
    # generated after the prompt; "all": the prompt too; "none": quantize on admission).
    # Not the default: more memory (bf16 staging) and no quality gain in our runs.
    stage_bf16: str = "none"

    @property
    def enabled(self) -> bool:
        return self.policy != "none"

    def validate(self) -> None:
        if self.policy not in ("none", "recent", "ema_recent", "ema_recent_value", "ema_recent_score", "expected_attention", "greedy_set", "oracle_future", "lookahead", "snapkv"):
            raise ValueError(f"unknown eviction policy: {self.policy}")
        if self.prefill_question_probe and self.policy != "lookahead":
            raise ValueError("prefill_question_probe ranks with the lookahead score; use policy lookahead")
        if not 0.0 <= self.decay <= 1.0:
            raise ValueError(f"decay must be in [0, 1] (1 = H2O accumulated sum), got {self.decay}")
        if self.recent_window < 0:
            raise ValueError("recent_window must be non-negative")
        if self.capacity_floor <= 0:
            raise ValueError("capacity_floor must be positive")
        if not 0.0 < self.capacity_percent <= 100.0:
            raise ValueError("capacity_percent must be in (0, 100]")
        if self.interval_blocks <= 0:
            raise ValueError("interval_blocks must be positive")
        if self.snapkv_window <= 0 or self.snapkv_pool_kernel <= 0 or self.snapkv_pool_kernel % 2 == 0:
            raise ValueError("SnapKV window must be positive and pooling kernel must be positive and odd")
        if self.value_center not in ("mean", "attn", "none"):
            raise ValueError(f"unknown eviction value_center: {self.value_center}")
        if self.ea_horizon <= 0 or not 0.0 <= self.ea_decay < 1.0 or self.ea_eps < 0.0:
            raise ValueError("bad expected_attention parameters")
        if (self.policy.startswith("ema_recent") or self.policy in ("expected_attention", "greedy_set", "oracle_future", "lookahead")) and self.recent_window > self.capacity_floor:
            raise ValueError(
                f"recent_window={self.recent_window} exceeds capacity_floor="
                f"{self.capacity_floor}; the window alone would fill the budget "
                "and no entry could ever be kept on its EMA"
            )

    @property
    def horizon_blocks(self) -> int:
        return self.lookahead_blocks if self.lookahead_blocks > 0 else self.interval_blocks

    def capacity(self, tokens_seen: int) -> int:
        """C = max(capacity_percent% of everything the cache has held, floor)."""
        return max(int(self.capacity_percent / 100.0 * tokens_seen), self.capacity_floor)


class EvictionState:
    """Per (layer, batch, KV head) live-entry state.

    Every (layer, head) keeps its own live set -- different heads attend to
    different things, so forcing one shared set would make the policy as weak as
    its least selective head. The sets stay the same *size*, which is what lets
    the cache keep one length.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        batch_size: int,
        num_kv_heads: int,
        capacity: int,
        device: torch.device | str,
        max_position: int = 1 << 31,
    ) -> None:
        self.num_layers = int(num_layers)
        self.batch_size = int(batch_size)
        self.num_kv_heads = int(num_kv_heads)
        self.device = torch.device(device)
        # int32 addresses any cache this runtime can build; int64 only when a
        # caller genuinely needs it, since pos is a third of the state's size.
        self.pos_dtype = torch.int32 if max_position < (1 << 31) else torch.int64

        shape = (self.num_layers, self.batch_size, self.num_kv_heads, capacity)
        self.g = torch.zeros(shape, dtype=torch.float32, device=self.device)
        # int16, not uint16: the spec's width, but a signed type because torch's
        # uint16 supports almost no arithmetic. 4096 tokens / 32 = 128 blocks, so
        # the range is never in question.
        self.n = torch.zeros(shape, dtype=torch.int16, device=self.device)
        self.pos = torch.zeros(shape, dtype=self.pos_dtype, device=self.device)
        self.phys = torch.zeros(shape, dtype=torch.int32, device=self.device)   # physical cache slot
        self.live = 0
        self._capacity = capacity
        self.pin: tuple[int | torch.Tensor, int] | None = None  # positions [lo, hi) never evicted

    def _pinned(self, pos: torch.Tensor) -> torch.Tensor:
        lo, hi = self.pin
        if isinstance(lo, torch.Tensor):
            lo = lo.to(pos.device).view(1, -1, 1, 1)
        return (pos >= lo) & (pos < hi)

    # -- growth ------------------------------------------------------------
    def _grow_to(self, needed: int) -> None:
        if needed <= self._capacity:
            return
        new_cap = max(needed, self._capacity * 2, 1)
        pad = new_cap - self._capacity
        shape = (self.num_layers, self.batch_size, self.num_kv_heads, pad)
        self.g = torch.cat([self.g, torch.zeros(shape, dtype=self.g.dtype, device=self.device)], -1)
        self.n = torch.cat([self.n, torch.zeros(shape, dtype=self.n.dtype, device=self.device)], -1)
        self.pos = torch.cat(
            [self.pos, torch.zeros(shape, dtype=self.pos.dtype, device=self.device)], -1
        )
        self.phys = torch.cat([self.phys, torch.zeros(shape, dtype=self.phys.dtype, device=self.device)], -1)
        self._capacity = new_cap

    def append(self, positions: torch.Tensor | range | list[int], phys: torch.Tensor | None = None) -> None:
        """Admit newly committed cache entries with g = 0, n = 0 (phys: their cache slots,
        default = their positions)."""
        if isinstance(positions, range):
            positions = list(positions)
        if isinstance(positions, list):
            positions = torch.tensor(positions, dtype=self.pos_dtype, device=self.device)
        m = int(positions.numel())
        if m == 0:
            return
        self._grow_to(self.live + m)
        sl = slice(self.live, self.live + m)
        self.g[..., sl] = 0.0
        self.n[..., sl] = 0
        self.pos[..., sl] = positions.to(self.pos_dtype).view(1, 1, 1, m)
        self.phys[..., sl] = (positions if phys is None else phys).to(self.phys.dtype).view(1, 1, 1, m)
        self.live += m

    # -- update ------------------------------------------------------------
    def observe(self, layer_idx: int, alpha: torch.Tensor, decay: float, weight: float = 1.0) -> None:
        """Fold this block's attention mass into the EMA for one layer.

        decay == 1 is H2O's accumulated attention instead: g += weight * alpha,
        with weight the number of queries alpha is the mean over, so g is the
        sum of attention over every observed query.

        ``alpha``: [B, Hkv, live] -- the selector's per-entry softmax mass,
        already aligned with the live set because the candidate set *is* the
        live set.
        """
        if self.live == 0:
            return
        if alpha.shape[-1] != self.live:
            raise ValueError(
                f"alpha has {alpha.shape[-1]} entries but {self.live} are live; "
                "the selector must rank exactly the live set"
            )
        sl = slice(0, self.live)
        g = self.g[layer_idx, ..., sl]
        if decay == 1.0:
            g.add_(alpha.to(g.dtype), alpha=float(weight))
        else:
            g.mul_(decay).add_(alpha.to(g.dtype), alpha=1.0 - decay)
        self.n[layer_idx, ..., sl] += 1

    def corrected(self, layer_idx: int, decay: float) -> torch.Tensor:
        """ghat = g / (1 - lam**n), the bias-corrected EMA. [B, Hkv, live]."""
        sl = slice(0, self.live)
        g = self.g[layer_idx, ..., sl]
        n = self.n[layer_idx, ..., sl]
        if decay in (0.0, 1.0):
            return g.clone()
        denom = 1.0 - torch.pow(
            torch.tensor(decay, dtype=torch.float32, device=g.device), n.to(torch.float32)
        )
        # n == 0 means never observed: leave it at zero rather than dividing by
        # zero, so an entry admitted this block is not ranked on noise.
        return torch.where(n > 0, g / denom.clamp_min(1e-8), torch.zeros_like(g))

    # -- eviction ----------------------------------------------------------
    def survivors(
        self,
        cfg: EvictionConfig,
        capacity: int,
        value_spread: torch.Tensor | None = None,
        score: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Indices into the live axis to keep. [L, B, Hkv, capacity], sorted by pos.

        ``value_spread``: [L, B, Hkv, live] -- ||v_i - v_head_mean|| for every
        live entry, required by ``ema_recent_value`` and ignored otherwise. The
        EMA contest then ranks on ghat * spread rather than ghat alone: an entry
        is worth keeping only if it is both attended to *and* says something the
        head's average value does not already say.
        """
        sl = slice(0, self.live)
        pos = self.pos[..., sl]

        if cfg.policy == "recent":
            # StreamingLLM: the newest C, plus the first sink_tokens positions.
            rank = pos.to(torch.float32)
            if cfg.sink_tokens:
                rank = rank.masked_fill(pos < cfg.sink_tokens, float("inf"))
            if self.pin is not None:
                rank = rank.masked_fill(self._pinned(pos), float("inf"))
            keep = rank.topk(capacity, dim=-1).indices
        elif cfg.policy == "snapkv":
            if score is None or score.shape != self.g.shape[:3] + (self.live,):
                raise ValueError("policy snapkv needs a [L, B, Hkv, live] score")
            rank = score.to(torch.float32)
            # SnapKV keeps its observation window verbatim and uses pooled
            # attention to choose the remaining prompt positions.
            recent = pos.topk(min(cfg.snapkv_window, capacity), dim=-1).indices
            rank = rank.scatter(-1, recent, float("inf"))
            if cfg.sink_tokens:
                rank = rank.masked_fill(pos < cfg.sink_tokens, float("inf"))
            if self.pin is not None:
                rank = rank.masked_fill(self._pinned(pos), float("inf"))
            keep = rank.topk(capacity, dim=-1).indices
        elif cfg.policy in ("ema_recent", "ema_recent_value", "ema_recent_score", "expected_attention", "greedy_set", "oracle_future", "lookahead"):
            w = min(cfg.recent_window, capacity)
            recent = pos.topk(w, dim=-1).indices if w else None
            rest = capacity - w
            if rest > 0:
                ghat = torch.stack(
                    [self.corrected(l, cfg.decay) for l in range(self.num_layers)], dim=0
                )
                if cfg.policy in ("expected_attention", "greedy_set", "oracle_future", "lookahead"):
                    if score is None or score.shape != ghat.shape:
                        raise ValueError(f"policy {cfg.policy} needs a [L, B, Hkv, live] score")
                    ghat = score.to(ghat.dtype)
                if cfg.policy == "ema_recent_value":
                    if value_spread is None:
                        raise ValueError(
                            "policy ema_recent_value needs value_spread; the caller "
                            "must supply per-entry ||v - vbar||"
                        )
                    if value_spread.shape != ghat.shape:
                        raise ValueError(
                            f"value_spread has shape {tuple(value_spread.shape)} but "
                            f"the live EMA is {tuple(ghat.shape)}"
                        )
                    if cfg.value_caote:
                        norm = ghat / ghat.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                        ghat = norm / (1.0 - norm).clamp_min(1e-6)
                    ghat = ghat * value_spread.to(ghat.dtype)
                if cfg.sink_tokens:
                    # Sinks win their slots outright (unless already recent).
                    ghat = ghat.masked_fill(pos < cfg.sink_tokens, float("inf"))
                if self.pin is not None:
                    ghat = ghat.masked_fill(self._pinned(pos), float("inf"))
                if recent is not None:
                    # Exclude the window from the EMA contest; an entry must not
                    # be able to win a slot it already holds.
                    ghat = ghat.scatter(-1, recent, float("-inf"))
                extra = ghat.topk(rest, dim=-1).indices
                keep = torch.cat([recent, extra], dim=-1) if recent is not None else extra
            else:
                keep = recent
        else:
            raise ValueError(f"policy {cfg.policy} does not evict")

        # Keep the live arrays ordered by position so "recent" stays the tail.
        keep_pos = pos.gather(-1, keep)
        return keep.gather(-1, keep_pos.argsort(dim=-1))

    def compact(self, keep: torch.Tensor) -> None:
        """Physically drop everything outside ``keep``."""
        c = keep.shape[-1]
        self.g[..., :c] = self.g[..., : self.live].gather(-1, keep)
        self.n[..., :c] = self.n[..., : self.live].gather(-1, keep)
        self.pos[..., :c] = self.pos[..., : self.live].gather(-1, keep.to(torch.int64))
        self.phys[..., :c] = self.phys[..., : self.live].gather(-1, keep.to(torch.int64))
        self.live = c

    def maybe_evict(
        self,
        cfg: EvictionConfig,
        *,
        block_index: int,
        tokens_seen: int,
        value_spread_fn: "Callable[[], torch.Tensor] | None" = None,
        score_fn: "Callable[[], torch.Tensor] | None" = None,
    ) -> torch.Tensor | None:
        """Evict on schedule. Returns the survivor indices, or None if nothing ran."""
        if not cfg.enabled or self.live == 0:
            return None
        if cfg.policy == "snapkv":
            return None  # prompt-only compression; generated KV grows normally
        if (block_index + 1) % cfg.interval_blocks:
            return None
        capacity = cfg.capacity(tokens_seen)
        if self.live <= capacity:
            return None
        spread = None
        if cfg.policy == "ema_recent_value":
            if value_spread_fn is None:
                raise ValueError("policy ema_recent_value needs a value_spread_fn")
            spread = value_spread_fn()
        score = None
        if cfg.policy in ("expected_attention", "greedy_set", "oracle_future", "snapkv"):
            if score_fn is None:
                raise ValueError(f"policy {cfg.policy} needs a score_fn")
            score = score_fn(capacity) if cfg.policy == "greedy_set" else score_fn()
        keep = self.survivors(cfg, capacity, spread, score)
        # How many (layer, batch, head) lanes drop at least one of positions 0..3.
        pos = self.pos[..., : self.live]
        kept_sink = (pos.gather(-1, keep) < 4).sum(-1)
        had_sink = (pos < 4).sum(-1)
        self.sink_lanes_evicted = getattr(self, "sink_lanes_evicted", 0) + int((kept_sink < had_sink).sum())
        self.lanes_evictions = getattr(self, "lanes_evictions", 0) + int(had_sink.numel())
        self.compact(keep)
        return keep

    # -- accounting --------------------------------------------------------
    def state_nbytes(self) -> int:
        """Bytes the policy itself costs. The spec says to count it, so count it."""
        per_entry = (
            self.g.element_size() + self.n.element_size() + self.pos.element_size()
        )
        return self.num_layers * self.batch_size * self.num_kv_heads * self.live * per_entry


def live_cache_nbytes(
    pos: torch.Tensor,
    *,
    live: int,
    head_dim: int,
    k_bits: int,
    v_bits: int,
    key_token_group: int,
    value_channel_group: int,
    param_bytes: int = 2,
    compute_bytes: int = 2,
) -> int:
    """Bytes a cache holding exactly this live set would occupy.

    ``pos``: [L, B, Hkv, capacity] -- the live positions, of which the first
    ``live`` are valid.

    Keys are quantized per 32-token *group*, so an entry carries a reference to
    its group's scale vector rather than a scale of its own. A group therefore
    survives as long as any one of its entries does, and the metadata is counted
    per surviving group, not per surviving entry -- which is why a policy that
    keeps a contiguous tail is cheaper per entry than one that keeps a scatter.
    """
    if live <= 0:
        return 0
    l, b, h = pos.shape[0], pos.shape[1], pos.shape[2]
    live_pos = pos[..., :live]

    if k_bits >= 16:
        k_payload = l * b * h * live * head_dim * compute_bytes
        k_meta = 0
    else:
        k_payload = l * b * h * live * head_dim * k_bits // 8
        groups = torch.unique(live_pos // key_token_group, dim=-1)
        # unique() pads with repeats per row, so count distinct values per row.
        n_groups = 0
        flat = (live_pos // key_token_group).reshape(-1, live)
        for row in flat:
            n_groups += int(torch.unique(row).numel())
        k_meta = n_groups * head_dim * 2 * param_bytes

    if v_bits >= 16:
        v_payload = l * b * h * live * head_dim * compute_bytes
        v_meta = 0
    else:
        v_payload = l * b * h * live * head_dim * v_bits // 8
        v_meta = l * b * h * live * (head_dim // value_channel_group) * 2 * param_bytes

    return k_payload + k_meta + v_payload + v_meta
