"""BitSieve decoding for DreamReasoner.

Dream's own ``block_diffusion_generate`` already has the shape the session
expects -- prefill the block-aligned prompt, denoise a block against the cache,
commit it with one ``store_kv=True`` forward -- so this drives that loop with a
:class:`PackedKVCache` underneath instead of a ``DynamicCache``, and reuses
Dream's sampling and transfer rules verbatim rather than restating them.

Prefill runs with the patch *off* and a normal DynamicCache, then packs the
result, exactly as the Fast-dLLM generator does: the prompt is encoded once at
full precision and only then quantized, which is what the commit path does for
every later block too.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

import time

import torch
from transformers.cache_utils import DynamicCache

from ..cache import PackedKVCache
from ..config import ExperimentConfig
from .dream_adapter import DreamPatchHandle, patch_dream
from .session import BitSieveSession
from .trace import RunTrace


@dataclass
class DreamGenerationResult:
    sequences: torch.Tensor
    generated_ids: torch.Tensor
    texts: list[str]
    metrics: dict[str, Any] = field(default_factory=dict)
    trace: RunTrace | None = None


def _gen_utils(model: torch.nn.Module):
    """Dream's own generation helpers, from whichever remote-code module holds them."""
    for base in type(model).__mro__:
        mod = sys.modules.get(base.__module__)
        if mod is not None and hasattr(mod, "build_block_diffusion_attention_mask"):
            return mod
    raise RuntimeError(
        "could not locate Dream's generation utilities; is this a Dream checkpoint?"
    )


class _BlockCausalMask:
    """attn_mask[:, r0:r1, :c1] of the block-diffusion mask (1 where key block <= query
    block), materialized only for the requested slice. Same values and dtype (fp32) as
    generation_utils.build_block_diffusion_attention_mask."""

    def __init__(self, block: int, device) -> None:
        self.block, self.device = int(block), device

    def __getitem__(self, idx):
        _, rows, cols = idx
        r0, r1 = rows.start or 0, rows.stop
        c0, c1 = cols.start or 0, cols.stop
        qb = torch.arange(r0, r1, device=self.device) // self.block
        kb = torch.arange(c0, c1, device=self.device) // self.block
        return (kb.unsqueeze(0) <= qb.unsqueeze(1)).to(torch.float32).unsqueeze(0)


class DreamBitSieveGenerator:

    def __init__(self, model: torch.nn.Module, tokenizer, config: ExperimentConfig) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.config = config
        cfg = model.config
        self.num_layers = int(cfg.num_hidden_layers)
        self.num_q_heads = int(cfg.num_attention_heads)
        self.num_kv_heads = int(cfg.num_key_value_heads)
        self.head_dim = int(
            getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        )
        self.device = next(model.parameters()).device
        self.compute_dtype = next(model.parameters()).dtype
        self.config.validate(head_dim=self.head_dim, num_layers=self.num_layers)
        self.gu = _gen_utils(model)
        self.mask_token_id = int(getattr(cfg, "mask_token_id", config.generation.mask_token_id))
        # The checkpoint's own EOS, not the experiment config's -- that default
        # is Fast-dLLM's 151645, and with the wrong id nothing ever stops: every
        # request runs to the full token budget, burning compute and letting the
        # model ramble past an answer it had already written.
        self.stop_token_id = getattr(cfg, "eos_token_id", None)
        if self.stop_token_id is None:
            self.stop_token_id = config.generation.stop_token_id
        self.patch: DreamPatchHandle = patch_dream(model, None)

    def _new_cache(self, batch_size: int) -> PackedKVCache:
        return PackedKVCache(
            num_layers=self.num_layers,
            batch_size=batch_size,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            max_tokens=self.config.max_cache_tokens,
            quant=self.config.quant,
            device=self.device,
            compute_dtype=self.compute_dtype,
            backend=self.config.backend,
        )

    def _fast_prefill_ok(self) -> bool:
        """Chunked prefill without eviction runs on the fast path only."""
        from .session import fast_path_supported

        return fast_path_supported(self.config, self.head_dim, self.device.type)[0]

    def _fill_tokens(self, x, bs_, block, n, cur):
        """Tokens of the lookahead probe blocks (eviction.lookahead_probe; dapq_fill = F4-L28)."""
        from .generator import _probe_tokens_impl

        return _probe_tokens_impl(self.config.eviction.lookahead_probe, cur, x[:, :bs_ + block], block, n,
                                  self.mask_token_id, self.config.seed)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor) -> DreamGenerationResult:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("input_ids must be [1, sequence]; batching is not wired")
        gu = self.gu
        cfg = self.config.generation
        block = cfg.block_size
        torch.manual_seed(self.config.seed)

        x0 = input_ids.to(self.device)
        prompt_len = int(x0.shape[1])
        gen_length = cfg.max_new_tokens
        num_blocks = (prompt_len + gen_length + block - 1) // block
        total = num_blocks * block
        # Block-causal mask, built per slice on demand: the full [total, total] fp32
        # tensor is ~1 GB at 16K tokens and only block-row slices are ever read.
        attn_mask = _BlockCausalMask(block, self.device)
        position_ids = torch.arange(total, device=self.device).unsqueeze(0)

        x = torch.full((1, total), self.mask_token_id, dtype=x0.dtype, device=self.device)
        x[:, :prompt_len] = x0

        prefill_blocks = prompt_len // block
        prefill_len = prefill_blocks * block
        # Chunked prefill: only the first chunk goes through the unpatched
        # full-precision prefill, the rest is fed through the session below.
        chunk = self.config.eviction.prefill_chunk
        full_prefill_len = prefill_len
        if chunk and prefill_len > chunk and (self.config.eviction.enabled or self._fast_prefill_ok()):
            if chunk % block:
                raise ValueError("eviction.prefill_chunk must be a multiple of block_size")
            prefill_len = chunk
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        t_prefill = time.perf_counter()
        cache = self._new_cache(1)
        if self.config.memory_trace:
            cache.memory_trace = []
        prefill_keys: dict[int, torch.Tensor] = {}

        # ---- prefill, unpatched and at full precision ----------------------
        if prefill_len > 0:
            self.patch.set_session(None)
            dyn = DynamicCache()
            self.model(
                x[:, :prefill_len],
                attention_mask=attn_mask[:, :prefill_len, :prefill_len],
                position_ids=position_ids[:, :prefill_len],
                past_key_values=dyn,
                use_cache=True,
                store_kv=True,
                logits_to_keep=1,  # prefill only fills the cache
            )
            cache.load_dynamic_cache(dyn)
            # Snapshot the prefill's exact keys before the fp16 cache is dropped:
            # they are the reference the packed selection gets scored against.
            # Without this the coverage column comes back empty and says nothing
            # about the selection, which is a silent hole rather than an error.
            prefill_keys = {}
            if self.config.coverage_diagnostics:
                for layer_idx in range(self.num_layers):
                    try:
                        key, _ = dyn[layer_idx]
                    except Exception:
                        if hasattr(dyn, "key_cache"):
                            key = dyn.key_cache[layer_idx]
                        else:
                            key = dyn.layers[layer_idx].keys
                    prefill_keys[layer_idx] = key[..., :prefill_len, :].detach().clone()
            del dyn

        session = BitSieveSession(
            cache, self.config,
            num_q_heads=self.num_q_heads, num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim, compute_dtype=self.compute_dtype,
        )
        for layer_idx, key in prefill_keys.items():
            session.seed_fp16_key_shadow(layer_idx, key)
        prefill_keys.clear()
        self.patch.set_session(session)
        for s0 in range(prefill_len, full_prefill_len, chunk or block):
            e0 = min(s0 + chunk, full_prefill_len)
            session.begin_prefill_chunk(e0 - s0)
            cache.begin_append(e0 - s0)
            try:
                session.prepare_forward(0, masked_positions=[], commit_current=True)
                self.model(
                    x[:, s0:e0],
                    attention_mask=attn_mask[:, s0:e0, :e0],
                    position_ids=position_ids[:, s0:e0],
                    past_key_values=cache,
                    use_cache=True,
                    store_kv=True,
                    logits_to_keep=1,
                )
                cache.commit_append()
            except Exception:
                cache.abort_append()
                raise
            session.end_prefill_chunk(final=(e0 >= full_prefill_len))
        session.commit_current = False
        prefill_blocks = full_prefill_len // block
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        prefill_ms = (time.perf_counter() - t_prefill) * 1e3
        t_decode = time.perf_counter()

        steps = cfg.fixed_steps_per_block if cfg.schedule == "fixed" else block
        transfer = gu.get_num_transfer_tokens(block, steps)
        nfe = 1 if prefill_len > 0 else 0
        stop_ids = gu._resolve_stopping_ids(self.stop_token_id)

        stop_found = False
        loop_stopped = False
        be_ = prefill_blocks * block
        try:
            for nb in range(prefill_blocks, num_blocks):
                bs_, be_ = nb * block, (nb + 1) * block
                cur = x[:, bs_:be_].clone()
                session.begin_block(
                    (cur[0] == self.mask_token_id).nonzero(as_tuple=True)[0].tolist()
                )
                step = 0
                for step in range(steps + 1):
                    mask_index = cur == self.mask_token_id
                    if not bool(mask_index.any()):
                        break
                    session.prepare_forward(
                        step,
                        masked_positions=mask_index[0].nonzero(as_tuple=True)[0].tolist(),
                        commit_current=False,
                    )
                    logits = self.model(
                        cur,
                        attention_mask=attn_mask[:, bs_:be_, :be_],
                        position_ids=position_ids[:, bs_:be_],
                        past_key_values=cache,
                        use_cache=True,
                        store_kv=False,
                    ).logits
                    nfe += 1
                    cand, conf = gu.sample_with_temperature_topk_topp(
                        logits, temperature=cfg.temperature, top_k=0, top_p=cfg.top_p
                    )
                    cand = torch.where(mask_index, cand, cur)
                    take = gu._select_transfer_index(
                        "low_confidence_dynamic", mask_index, cand, conf, transfer, step,
                        cfg.threshold, 0.35, force_accept=(step == steps - 1),
                    )
                    cur[take] = cand[take]
                    session.finish_step0()
                    if step == 0 and session.wants_probe() and self.config.eviction.lookahead_probe == "dapq":
                        # DapQ: F4-L28 of the committed context at this block's positions.
                        ctx = x[:, :bs_]
                        pq = torch.cat([ctx[:, :4], ctx[:, -(block - 4):]], dim=1)
                        session.begin_dapq_probe(bs_)
                        try:
                            self.model(
                                pq,
                                attention_mask=attn_mask[:, bs_:be_, :be_],
                                position_ids=position_ids[:, bs_:be_],
                                past_key_values=cache,
                                use_cache=True,
                                store_kv=False,
                                logits_to_keep=1,
                            )
                        finally:
                            session.end_dapq_probe()
                        nfe += 1
                    elif step == 0 and session.wants_probe():
                        # Lookahead probe: masks of the next blocks in one forward,
                        # nothing stored. The probe's own block is the pre-step
                        # masked state, as at step 0.
                        m = self.config.eviction.horizon_blocks
                        extra = min(block * (m - 1), total - be_)
                        if extra > 0:
                            probe = torch.cat([
                                torch.where(mask_index, cur, cur),  # current block (as denoised so far)
                                self._fill_tokens(x, bs_, block, extra, cur),
                            ], dim=1)
                            probe[:, :block] = torch.where(mask_index, self.mask_token_id, probe[:, :block])
                            session.begin_probe()
                            try:
                                self.model(
                                    probe,
                                    attention_mask=attn_mask[:, bs_:be_ + extra, :be_ + extra],
                                    position_ids=position_ids[:, bs_:be_ + extra],
                                    past_key_values=cache,
                                    use_cache=True,
                                    store_kv=False,
                                    logits_to_keep=1,  # the probe only needs queries
                                )
                            finally:
                                session.end_probe()
                            nfe += 1
                        else:
                            session.end_probe()

                if bool((cur == self.mask_token_id).any()):
                    raise RuntimeError("denoising ended with unresolved mask tokens")

                # ---- commit the finished block ------------------------------
                cache.begin_append(block)
                try:
                    session.prepare_forward(step, masked_positions=[], commit_current=True)
                    self.model(
                        cur,
                        attention_mask=attn_mask[:, bs_:be_, :be_],
                        position_ids=position_ids[:, bs_:be_],
                        past_key_values=cache,
                        use_cache=True,
                        store_kv=True,
                        logits_to_keep=1,  # commit only writes the cache
                    )
                    cache.commit_append()
                except Exception:
                    cache.abort_append()
                    raise
                nfe += 1
                session.end_block()

                x[:, bs_:be_] = cur
                if gu._should_stop(x, prompt_len, stop_ids):
                    stop_found = True
                    break
                if cfg.loop_stop:
                    from .generator import _ends_in_loop
                    text = self.tokenizer.decode(x[0, prompt_len:be_].tolist(), skip_special_tokens=True)
                    if _ends_in_loop(text, cfg.loop_stop_reps):
                        loop_stopped = True
                        break
        finally:
            self.patch.set_session(None)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        decode_ms = (time.perf_counter() - t_decode) * 1e3
        out_len = min(total, prompt_len + gen_length)
        seq = x[:, :out_len]
        generated = seq[:, prompt_len:]
        # Real length: up to the first stop token, else everything committed.
        committed = generated[0, : max(0, be_ - prompt_len)]
        gen_tokens = int(committed.shape[0])
        for sid in (stop_ids if isinstance(stop_ids, (list, tuple, set)) else [stop_ids] if stop_ids is not None else []):
            hit = (committed == sid).nonzero(as_tuple=True)[0]
            if hit.numel():
                gen_tokens = min(gen_tokens, int(hit[0]) + 1)
        trace = session.finalize_trace()
        metrics: dict[str, Any] = {
            "coverage": trace.coverage_summary(),
            "nfe": nfe,
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "generated_tokens": gen_tokens,
            "tokens_per_second": gen_tokens / (decode_ms / 1e3) if decode_ms > 0 else None,
            "peak_cuda_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(self.device)) if self.device.type == "cuda" else None
            ),
            "stop_found": stop_found,
            "loop_stopped": loop_stopped,
            "cache_tokens": cache.get_seq_length(),
            "generated_tokens_per_request": int(generated.shape[1]),
            "blocks_sparse": session.trace.counters.get("blocks_sparse", 0),
            "blocks_dense_bypass": session.trace.counters.get("blocks_dense_bypass", 0),
            **cache.logical_nbytes(),
            "cache_peak_used_bytes": cache.peak_used_bytes,
            **({"memory_trace": cache.memory_trace} if cache.memory_trace is not None else {}),
            "cache_peak_allocated_bytes": cache.peak_allocated_bytes,
            "bf16_equiv_bytes": cache.bf16_equivalent_nbytes(),
            **session.eviction_metrics(),
        }
        return DreamGenerationResult(
            sequences=seq,
            generated_ids=generated,
            texts=self.tokenizer.batch_decode(generated, skip_special_tokens=True),
            metrics=metrics,
            trace=trace,
        )
