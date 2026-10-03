from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers.cache_utils import DynamicCache

from ..cache import PackedKVCache
from ..config import ExperimentConfig
from .adapter import PatchHandle, patch_fast_dllm, set_bitsieve_session
from .session import BitSieveSession
from .trace import RunTrace
from .generation_utils import (
    CudaInterval as _SharedCudaInterval,
    compact_dtype_from_name,
    sample_logits,
    select_unmask,
)


def _probe_tokens_impl(kind, current, x_t, block, n, mask_id, seed):
    """Tokens for n probe positions after the current block."""
    shape = (current.shape[0], n)
    if kind == "mask":
        return torch.full(shape, mask_id, dtype=current.dtype, device=current.device)
    if kind == "random":
        gen = torch.Generator(device="cpu").manual_seed(seed + int(x_t.shape[1]))
        return torch.randint(0, mask_id, shape, generator=gen).to(device=current.device, dtype=current.dtype)
    if kind in ("dapq", "dapq_fill"):
        # DapQ F4-L28 of the committed context, tiled over the probe.
        ctx = x_t[:, : x_t.shape[1] - block]
        pat = torch.cat([ctx[:, :4], ctx[:, -(block - 4):]], dim=1)
        reps = -(-n // pat.shape[1])
        return pat.repeat(1, reps)[:, :n].to(current.dtype)
    raise ValueError(f"unknown lookahead_probe: {kind}")


def _ends_in_loop(text: str, reps: int, pmin: int = 20, pmax: int = 1000) -> bool:
    """True if `text` ends in `reps` exact copies of one unit of pmin..pmax chars."""
    n = len(text)
    for p in range(pmin, min(pmax, n // reps) + 1):
        unit = text[n - p:]
        if all(text[n - (k + 1) * p: n - k * p] == unit for k in range(1, reps)):
            return True
    return False


@dataclass(slots=True)
class GenerationResult:
    sequences: torch.Tensor
    generated_ids: torch.Tensor
    texts: list[str]
    metrics: dict[str, Any] = field(default_factory=dict)
    trace: RunTrace | None = None


class _CudaInterval(_SharedCudaInterval):
    pass

class BitSieveGenerator:

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        config: ExperimentConfig,
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.config = config
        model_cfg = model.config
        self.num_layers = int(model_cfg.num_hidden_layers)
        self.num_q_heads = int(model_cfg.num_attention_heads)
        self.num_kv_heads = int(model_cfg.num_key_value_heads)
        self.head_dim = int(
            getattr(model_cfg, "head_dim", model_cfg.hidden_size // model_cfg.num_attention_heads)
        )
        self.device = next(model.parameters()).device
        self.compute_dtype = next(model.parameters()).dtype
        self.compact_dtype = compact_dtype_from_name(self.config.compact_dtype)
        self.config.validate(head_dim=self.head_dim, num_layers=self.num_layers)
        if hasattr(self.model, "model") and hasattr(self.model.model, "bd_size"):
            self.model.model.bd_size = self.config.generation.block_size
        self.patch: PatchHandle = patch_fast_dllm(model, None)

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

    def _probe_tokens(self, current, x_t, block, n):
        return _probe_tokens_impl(
            self.config.eviction.lookahead_probe, current, x_t, block, n,
            self.config.generation.mask_token_id, self.config.seed,
        )

    def _question_probe(self, session, cache, prompt, stored, prefix_end):
        """Probe forward, nothing stored: the question tokens not yet in the cache at
        their real positions, then masks up to horizon_blocks answer blocks."""
        block = self.config.generation.block_size
        starts = session._q_span[0]
        q0 = max(int(starts.min()) if isinstance(starts, torch.Tensor) else starts, stored)
        masks = prefix_end + block * self.config.eviction.horizon_blocks - prompt.shape[1]
        toks = torch.cat([prompt[:, q0:], torch.full((prompt.shape[0], masks), self.config.generation.mask_token_id,
                                                      dtype=prompt.dtype, device=prompt.device)], dim=1)
        pos = torch.arange(q0, q0 + toks.shape[1], device=prompt.device).unsqueeze(0)
        session.prepare_forward(0, masked_positions=[], commit_current=False)
        session.begin_question_probe(q0)
        try:
            self.model.forward(
                input_ids=toks,
                position_ids=pos,
                use_cache=True,
                past_key_values=cache,
                update_past_key_values=False,
                use_block_cache=False,
                block_size=block,
                logits_to_keep=1,
            )
        finally:
            session.end_question_probe()

    def _dapq_probe(self, session, cache, x_t, block):
        """DapQ: F4-L28 of the committed context at the current block's positions."""
        p0 = x_t.shape[1] - block
        toks = _probe_tokens_impl("dapq", x_t[:, -block:], x_t, block, block,
                                  self.config.generation.mask_token_id, self.config.seed)
        pos = torch.arange(p0, p0 + block, device=x_t.device).unsqueeze(0)
        session.begin_dapq_probe(p0)
        try:
            self.model.forward(
                input_ids=toks,
                position_ids=pos,
                use_cache=True,
                past_key_values=cache,
                update_past_key_values=False,
                use_block_cache=False,
                block_size=block,
                logits_to_keep=1,
            )
        finally:
            session.end_dapq_probe()

    def _fast_prefill_ok(self) -> bool:
        """Chunked prefill without eviction runs on the fast path only."""
        from .session import fast_path_supported

        return fast_path_supported(self.config, self.head_dim, self.device.type)[0]

    def _sample(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return sample_logits(
            logits,
            temperature=self.config.generation.temperature,
            top_p=self.config.generation.top_p,
        )

    def _select_unmask(
        self,
        confidence: torch.Tensor,
        mask: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor:
        cfg = self.config.generation
        return select_unmask(
            confidence,
            mask,
            schedule=cfg.schedule,
            threshold=cfg.threshold,
            fixed_steps_per_block=cfg.fixed_steps_per_block,
            step_index=step_index,
        )

    def _prefill(
        self,
        input_ids: torch.Tensor,
        cache: PackedKVCache,
    ) -> tuple[torch.Tensor, dict[str, float], dict[int, torch.Tensor]]:
        bsz, prompt_len = input_ids.shape
        if bsz != cache.batch_size:
            raise ValueError("batch size mismatch")
        block = self.config.generation.block_size
        metrics = {"prefill_ms": 0.0, "prefill_pack_ms": 0.0}
        prefill_keys: dict[int, torch.Tensor] = {}
        self._prefill_rest = None                 # set below when the prompt is chunked
        if prompt_len <= block:
            return input_ids, metrics, prefill_keys

        prefix_len = (prompt_len // block) * block
        chunk = self.config.eviction.prefill_chunk
        if chunk and prefix_len > chunk and (self.config.eviction.enabled or self._fast_prefill_ok()):
            # Only the first chunk goes through the unpatched prefill; the rest
            # is fed through the session chunk by chunk (see generate()).
            if chunk % block:
                raise ValueError("eviction.prefill_chunk must be a multiple of block_size")
            self._prefill_rest = (chunk, prefix_len)
            prefix_len = chunk
        set_bitsieve_session(self.model, None)
        with _CudaInterval(self.device) as timer:
            output = self.model.forward(
                input_ids=input_ids[:, :prefix_len],
                use_cache=True,
                update_past_key_values=True,
                block_size=block,
                logits_to_keep=1,  # prefill only fills the cache; no LM head over the prompt
            )
        metrics["prefill_ms"] = timer.milliseconds()
        dynamic_cache = output.past_key_values
        if not isinstance(dynamic_cache, DynamicCache):


            if not hasattr(dynamic_cache, "get_seq_length"):
                raise TypeError("Fast-dLLM prefill did not return a compatible cache")

        with _CudaInterval(self.device) as timer_pack:
            cache.load_dynamic_cache(dynamic_cache)
        metrics["prefill_pack_ms"] = timer_pack.milliseconds()

        # Snapshot the prefill's exact keys before the fp16 cache is dropped -
        # they are the reference the packed selection gets scored against.
        # Keys only (attention weights do not depend on V), and only when
        # coverage scoring is on, so a normal run pays nothing.
        if self.config.coverage_diagnostics:
            for layer_idx in range(self.num_layers):
                try:
                    key, _ = dynamic_cache[layer_idx]
                except Exception:
                    key = dynamic_cache.key_cache[layer_idx]
                prefill_keys[layer_idx] = key[..., :prefix_len, :].detach().clone()

        if prompt_len % block == 0 and self._prefill_rest is None:
            # (chunked: the prompt's last position is in the last chunk; see generate())
            next_token = output.logits[:, -1:, :].argmax(dim=-1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        del dynamic_cache, output
        return input_ids, metrics, prefill_keys

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, question_start: int | torch.Tensor | None = None) -> GenerationResult:
        """question_start: first prompt position of the question (the tail after the
        context); used by eviction.pin_question / prefill_question_probe."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[0] > 1 and self.config.generation.stop_token_id is not None:
            raise NotImplementedError(
                "batch_size > 1 requires generation.stop_token_id=null so the active "
                "batch remains fixed for trustworthy throughput measurements"
            )
        torch.manual_seed(self.config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.config.seed)

        original_input = input_ids.to(self.device)
        original_length = original_input.shape[1]
        cache = self._new_cache(original_input.shape[0])
        if self.config.memory_trace:
            cache.memory_trace = []
        work_ids, prefill_metrics, prefill_keys = self._prefill(original_input, cache)
        session = BitSieveSession(
            cache,
            self.config,
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            compute_dtype=self.compact_dtype,
        )
        for layer_idx, key in prefill_keys.items():
            session.seed_fp16_key_shadow(layer_idx, key)
        prefill_keys.clear()
        set_bitsieve_session(self.model, session)
        if question_start is not None:
            if isinstance(question_start, torch.Tensor):
                starts = question_start.to(device=self.device, dtype=torch.long).reshape(-1)
                if starts.numel() != original_input.shape[0] or bool(((starts < 0) | (starts >= original_length)).any()):
                    raise ValueError("question_start must contain one valid position per batch row")
                session.set_question(starts, original_length)
            elif 0 <= question_start < original_length:
                session.set_question(question_start, original_length)
        if self._prefill_rest is not None:
            start, end = self._prefill_rest
            chunk = self.config.eviction.prefill_chunk
            with _CudaInterval(self.device) as chunk_timer:
                for s0 in range(start, end, chunk):
                    e0 = min(s0 + chunk, end)
                    session.begin_prefill_chunk(e0 - s0)
                    cache.begin_append(e0 - s0)
                    try:
                        session.prepare_forward(0, masked_positions=[], commit_current=True)
                        out = self.model.forward(
                            input_ids=original_input[:, s0:e0],
                            use_cache=True,
                            past_key_values=cache,
                            update_past_key_values=False,
                            use_block_cache=False,
                            block_size=self.config.generation.block_size,
                            logits_to_keep=1,
                        )
                        cache.commit_append()
                        if e0 >= end and original_length % self.config.generation.block_size == 0:
                            # aligned prompt: its first new token comes from the prompt's last
                            # position, which is the last position of the last chunk
                            work_ids = torch.cat([work_ids, out.logits[:, -1:, :].argmax(dim=-1)], dim=1)
                        del out
                    except Exception:
                        cache.abort_append()
                        raise
                    session.end_prefill_chunk(final=(e0 >= end))
                    if session.wants_question_probe(final=(e0 >= end)):
                        self._question_probe(session, cache, original_input, e0, end)
            prefill_metrics["prefill_chunked_ms"] = chunk_timer.milliseconds()
            session.commit_current = False

        cfg = self.config.generation
        block = cfg.block_size
        num_blocks = math.ceil(cfg.max_new_tokens / block)
        nfe = 0
        block_steps: list[int] = []
        block_times: list[_CudaInterval] = []
        stop_found = False
        loop_stopped = False

        if self.device.type == "cuda":
            # Preserve the prefill peak before the decode-only reset. Serving
            # measurements need the maximum across both phases.
            prefill_metrics["peak_cuda_allocated_prefill_bytes"] = int(
                torch.cuda.max_memory_allocated(self.device)
            )
            prefill_metrics["peak_cuda_reserved_prefill_bytes"] = int(
                torch.cuda.max_memory_reserved(self.device)
            )
            torch.cuda.reset_peak_memory_stats(self.device)
        with _CudaInterval(self.device) as decode_timer:
            for _ in range(num_blocks):
                if work_ids.shape[1] - original_length >= cfg.max_new_tokens:
                    break
                fill = block - (work_ids.shape[1] % block)
                if fill == 0:
                    fill = block
                x_t = torch.cat(
                    [
                        work_ids,
                        torch.full(
                            (work_ids.shape[0], fill),
                            cfg.mask_token_id,
                            device=self.device,
                            dtype=torch.long,
                        ),
                    ],
                    dim=1,
                )


                if x_t.shape[1] < block:
                    raise AssertionError("current sequence shorter than one block")
                current = x_t[:, -block:]
                initial_mask = (current[0] == cfg.mask_token_id).nonzero(as_tuple=True)[0].tolist()
                session.begin_block(initial_mask)
                block_timer = _CudaInterval(self.device)
                block_timer.__enter__()
                block_cache = None
                step_index = 0
                num_small = block // cfg.small_block_size

                for small_idx in range(num_small):
                    small_start = small_idx * cfg.small_block_size
                    small_end = small_start + cfg.small_block_size
                    while True:
                        current = x_t[:, -block:]
                        whole_mask = current == cfg.mask_token_id
                        local_mask = whole_mask[:, small_start:small_end]
                        if not bool(local_mask.any()):
                            break
                        if cfg.schedule == "fixed" and step_index >= cfg.fixed_steps_per_block:


                            force_last = True
                        else:
                            force_last = False

                        masked_positions = whole_mask[0].nonzero(as_tuple=True)[0].tolist()
                        session.prepare_forward(
                            step_index,
                            masked_positions=masked_positions,
                            commit_current=False,
                        )
                        needs_full_block = block_cache is None or bool(
                            (current[:, small_start] == cfg.mask_token_id).any()
                        )
                        if cfg.use_block_cache and not needs_full_block:
                            output = self.model.forward(
                                input_ids=current[:, small_start:small_end],
                                use_cache=True,
                                past_key_values=cache,
                                update_past_key_values=False,
                                use_block_cache=True,
                                block_past_key_values=block_cache,
                                replace_position=small_start,
                                block_size=block,
                            )
                            logits = output.logits
                            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                        else:
                            extra = session.merge_probe_tokens() if step_index == 0 else 0
                            model_in = current
                            if extra:
                                model_in = torch.cat([current, self._probe_tokens(current, x_t, block, extra)], dim=1)
                            output = self.model.forward(
                                input_ids=model_in,
                                use_cache=True,
                                past_key_values=cache,
                                update_past_key_values=False,
                                use_block_cache=cfg.use_block_cache,
                                block_size=block,
                            )
                            logits = output.logits
                            if extra:
                                logits = logits[:, : current.shape[1]]
                            if cfg.use_block_cache:
                                block_cache = output.block_past_key_values
                                if extra and block_cache is not None:
                                    block_cache.crop(current.shape[1])
                            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                            logits = logits[:, small_start:small_end, :]

                        if step_index == 0:
                            session.finish_step0()
                            if session.wants_probe() and self.config.eviction.lookahead_probe == "dapq":
                                self._dapq_probe(session, cache, x_t, block)
                                nfe += 1
                            elif session.wants_probe():
                                # Lookahead: masks for the next M blocks in one forward,
                                # nothing written to the cache.
                                m = self.config.eviction.horizon_blocks
                                probe = torch.cat(
                                    [current, self._probe_tokens(current, x_t, block, block * (m - 1))], dim=1
                                )
                                session.begin_probe()
                                try:
                                    self.model.forward(
                                        input_ids=probe,
                                        use_cache=True,
                                        past_key_values=cache,
                                        update_past_key_values=False,
                                        use_block_cache=False,
                                        block_size=block,
                                        logits_to_keep=1,  # the probe only needs queries
                                    )
                                finally:
                                    session.end_probe()
                                nfe += 1
                        token, confidence = self._sample(logits)
                        select = self._select_unmask(confidence, local_mask, step_index)
                        if force_last:
                            select = local_mask
                        local = current[:, small_start:small_end]
                        local[select] = token[select]
                        step_index += 1
                        nfe += 1

                if bool((x_t[:, -block:] == cfg.mask_token_id).any()):
                    raise RuntimeError("denoising ended with unresolved mask tokens")


                cache.begin_append(block)
                try:
                    session.prepare_forward(
                        step_index,
                        masked_positions=[],
                        commit_current=True,
                    )
                    final_output = self.model.forward(
                        input_ids=x_t[:, -block:],
                        use_cache=True,
                        past_key_values=cache,
                        update_past_key_values=False,
                        use_block_cache=False,
                        block_size=block,
                        logits_to_keep=1,  # only the last position feeds next_token
                    )
                    cache.commit_append()
                except Exception:
                    cache.abort_append()
                    raise
                nfe += 1
                session.end_block()
                block_timer.__exit__(None, None, None)
                block_times.append(block_timer)
                block_steps.append(step_index)

                next_token = final_output.logits[:, -1:, :].argmax(dim=-1)
                work_ids = torch.cat([x_t, next_token], dim=1)
                generated = work_ids[:, original_length:]
                if cfg.stop_token_id is not None:
                    stop_pos = (generated[0] == cfg.stop_token_id).nonzero(as_tuple=True)[0]
                    if stop_pos.numel():
                        cut = int(stop_pos[0].item()) + 1
                        work_ids = work_ids[:, : original_length + cut]
                        stop_found = True
                        break
                if cfg.loop_stop and _ends_in_loop(
                    self.tokenizer.decode(generated[0].tolist(), skip_special_tokens=True),
                    cfg.loop_stop_reps,
                ):
                    loop_stopped = True
                    break

        decode_ms = decode_timer.milliseconds()
        trace = session.finalize_trace()
        generated_ids = work_ids[:, original_length : original_length + cfg.max_new_tokens]
        sequences = torch.cat([original_input, generated_ids], dim=1)
        texts = [
            self.tokenizer.decode(row.tolist(), skip_special_tokens=True)
            for row in generated_ids.detach().cpu()
        ]
        block_ms = [x.milliseconds(synchronize=False) for x in block_times]
        memory = cache.logical_nbytes()
        # The gathered compact caches are resident alongside the packed cache
        # for the whole block, so the honest denominator for a compression
        # claim includes them. Reporting only `packed_cache` overstates the
        # ratio, badly so at short prefixes.
        compact_bytes = session.compact_nbytes()
        resident_bytes = int(memory["total"]) + compact_bytes
        dense_equivalent = (
            self.num_layers
            * cache.batch_size
            * self.num_kv_heads
            * cache.length
            * self.head_dim
            * 2
            * torch.tensor([], dtype=self.compute_dtype).element_size()
        )
        counters = dict(trace.counters)
        sparse_blocks = float(counters.get("blocks_sparse", 0.0))
        bypass_blocks = float(counters.get("blocks_dense_bypass", 0.0))
        selectable = sparse_blocks + bypass_blocks
        sparse_layer_steps = float(counters.get("layer_steps_sparse", 0.0))
        dense_layer_steps = float(counters.get("layer_steps_dense", 0.0))
        total_layer_steps = sparse_layer_steps + dense_layer_steps
        session.flush_oracle()
        metrics: dict[str, Any] = {
            **prefill_metrics,
            "decode_ms": decode_ms,
            "requests": int(generated_ids.shape[0]),
            "generated_tokens": int(generated_ids.numel()),
            "generated_tokens_per_request": int(generated_ids.shape[1]),
            "tokens_per_second": (
                float(generated_ids.numel()) / (decode_ms / 1000.0) if decode_ms > 0 else None
            ),
            "nfe": nfe,
            "denoising_steps_per_block": block_steps,
            "time_per_output_block_ms": block_ms,
            "mean_tpob_ms": sum(block_ms) / len(block_ms) if block_ms else None,
            "packed_cache": memory,
            "compact_cache_bytes": compact_bytes,
            "resident_cache_bytes": resident_bytes,
            # measured: largest used / allocated cache storage over the run, and a bf16
            # K+V cache of every token seen (the common memory baseline)
            "cache_peak_used_bytes": cache.peak_used_bytes,
            **({"memory_trace": cache.memory_trace} if cache.memory_trace is not None else {}),
            "cache_peak_allocated_bytes": cache.peak_allocated_bytes,
            "bf16_equiv_bytes": cache.bf16_equivalent_nbytes(),
            **session.eviction_metrics(),
            "dense_cache_equivalent_bytes": int(dense_equivalent),
            "cache_compression_ratio": (
                dense_equivalent / resident_bytes if resident_bytes else None
            ),
            "packed_only_compression_ratio": (
                dense_equivalent / memory["total"] if memory["total"] else None
            ),
            # Did the sparse path actually run? A fixed top-k budget cannot
            # bite until the prefix outgrows it, so these must be read before
            # any sparse-vs-dense quality claim.
            "blocks_sparse": int(sparse_blocks),
            **(
                {
                    "value_rescore_calls": int(counters["value_rescore_calls"]),
                    "value_rescore_kept": (
                        counters["value_rescore_kept_ppm"]
                        / counters["value_rescore_calls"]
                        / 1e6
                    ),
                }
                if counters.get("value_rescore_calls")
                else {}
            ),
            **(
                {
                    "value_spread_heads": int(counters["value_spread_heads"]),
                    "value_spread_iqr_over_median": (
                        counters["value_spread_iqr_ppm_sum"]
                        / counters["value_spread_heads"]
                        / 1e6
                    ),
                    "value_spread_flat_frac": (
                        counters["value_spread_flat_heads"] / counters["value_spread_heads"]
                    ),
                    "value_spread_wide_frac": (
                        counters["value_spread_wide_heads"] / counters["value_spread_heads"]
                    ),
                }
                if counters.get("value_spread_heads")
                else {}
            ),
            **(
                {
                    # Idea-A pre-check: prefix share of the joint softmax, its
                    # spread inside a KV group, and the centre shift |y - o|
                    # relative to the median |v - o|.
                    "joint_m_mean": counters["joint_m_mean_ppm"] / counters["joint_calls"] / 1e6,
                    "joint_m_cv_in_group": counters["joint_m_cv_ppm"] / counters["joint_calls"] / 1e6,
                    "joint_center_shift": counters["joint_center_shift_ppm"] / counters["joint_calls"] / 1e6,
                }
                if counters.get("joint_calls")
                else {}
            ),
            "blocks_dense_bypass": int(bypass_blocks),
            "sparse_block_fraction": (sparse_blocks / selectable if selectable else None),
            "sparse_layer_step_fraction": (
                sparse_layer_steps / total_layer_steps if total_layer_steps else None
            ),
            "coverage": trace.coverage_summary(),
            "cache_tokens": cache.length,
            "stop_found": stop_found,
            "loop_stopped": loop_stopped,
            "semantic": self.config.semantic,
            "engine": "bitsieve",
            "compact_dtype": self.config.compact_dtype,
            "dense_kernel_variant": self.config.dense_kernel_variant,
            "selector_kernel_variant": self.config.selector_kernel_variant,
            "config_name": self.config.name,
        }
        if self.device.type == "cuda":
            metrics["peak_cuda_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated(self.device)
            )
            metrics["peak_cuda_reserved_bytes"] = int(
                torch.cuda.max_memory_reserved(self.device)
            )
            metrics["peak_cuda_allocated_total_bytes"] = max(
                metrics["peak_cuda_allocated_prefill_bytes"], metrics["peak_cuda_allocated_bytes"]
            )
            metrics["peak_cuda_reserved_total_bytes"] = max(
                metrics["peak_cuda_reserved_prefill_bytes"], metrics["peak_cuda_reserved_bytes"]
            )
        return GenerationResult(
            sequences=sequences,
            generated_ids=generated_ids,
            texts=texts,
            metrics=metrics,
            trace=trace,
        )
