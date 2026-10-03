from __future__ import annotations

import math
from typing import Any

import torch

from ..config import ExperimentConfig
from .generation_utils import CudaInterval, dynamic_cache_nbytes, sample_logits, select_unmask
from .generator import GenerationResult


class OfficialDenseGenerator:

    def __init__(self, model: torch.nn.Module, tokenizer, config: ExperimentConfig) -> None:
        if config.engine != "official" or config.semantic != "dense":
            raise ValueError("OfficialDenseGenerator requires engine=official, semantic=dense")
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device
        self.compute_dtype = next(model.parameters()).dtype
        config.validate(
            head_dim=int(
                getattr(
                    model.config,
                    "head_dim",
                    model.config.hidden_size // model.config.num_attention_heads,
                )
            ),
            num_layers=int(model.config.num_hidden_layers),
        )
        if hasattr(self.model, "model") and hasattr(self.model.model, "bd_size"):
            self.model.model.bd_size = config.generation.block_size

    def _disable_sparse_runtime(self) -> None:
        modules = getattr(self.model, "_bitsieve_patch_modules", None)
        if modules is not None:
            for module in modules:
                setattr(module, "_bitsieve_session", None)

    def _prefill(self, input_ids: torch.Tensor):
        cfg = self.config.generation
        prompt_len = input_ids.shape[1]
        metrics = {"prefill_ms": 0.0, "prefill_pack_ms": 0.0}
        if prompt_len <= cfg.block_size:
            return input_ids, None, metrics

        prefix_len = (prompt_len // cfg.block_size) * cfg.block_size
        with CudaInterval(self.device) as timer:
            output = self.model.forward(
                input_ids=input_ids[:, :prefix_len],
                use_cache=True,
                update_past_key_values=True,
                block_size=cfg.block_size,
            )
        metrics["prefill_ms"] = timer.milliseconds()
        past_key_values = output.past_key_values
        if past_key_values is None or not hasattr(past_key_values, "get_seq_length"):
            raise TypeError("native Fast-dLLM prefill did not return a compatible cache")
        if prompt_len % cfg.block_size == 0:
            next_token = output.logits[:, -1:, :].argmax(dim=-1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids, past_key_values, metrics

    def _sample(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config.generation
        return sample_logits(logits, temperature=cfg.temperature, top_p=cfg.top_p)

    def _select_unmask(
        self, confidence: torch.Tensor, mask: torch.Tensor, step_index: int
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

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor) -> GenerationResult:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[0] > 1 and self.config.generation.stop_token_id is not None:
            raise NotImplementedError(
                "batch_size > 1 requires generation.stop_token_id=null so the active "
                "batch remains fixed for performance measurement"
            )
        self._disable_sparse_runtime()
        torch.manual_seed(self.config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.config.seed)

        original_input = input_ids.to(self.device)
        original_length = original_input.shape[1]
        work_ids, past_key_values, prefill_metrics = self._prefill(original_input)
        cfg = self.config.generation
        block = cfg.block_size
        num_blocks = math.ceil(cfg.max_new_tokens / block)
        nfe = 0
        block_steps: list[int] = []
        block_timers: list[CudaInterval] = []
        stop_found = False

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        with CudaInterval(self.device) as decode_timer:
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
                current = x_t[:, -block:]
                block_cache = None
                step_index = 0
                timer = CudaInterval(self.device)
                timer.__enter__()

                for small_idx in range(block // cfg.small_block_size):
                    small_start = small_idx * cfg.small_block_size
                    small_end = small_start + cfg.small_block_size
                    while True:
                        current = x_t[:, -block:]
                        whole_mask = current == cfg.mask_token_id
                        local_mask = whole_mask[:, small_start:small_end]
                        if not bool(local_mask.any()):
                            break
                        force_last = (
                            cfg.schedule == "fixed"
                            and step_index >= cfg.fixed_steps_per_block
                        )
                        needs_full_block = block_cache is None or bool(
                            (current[:, small_start] == cfg.mask_token_id).any()
                        )
                        if cfg.use_block_cache and not needs_full_block:
                            output = self.model.forward(
                                input_ids=current[:, small_start:small_end],
                                use_cache=True,
                                past_key_values=past_key_values,
                                update_past_key_values=False,
                                use_block_cache=True,
                                block_past_key_values=block_cache,
                                replace_position=small_start,
                                block_size=block,
                            )
                            logits = torch.cat(
                                [output.logits[:, :1, :], output.logits[:, :-1, :]], dim=1
                            )
                        else:
                            output = self.model.forward(
                                input_ids=current,
                                use_cache=True,
                                past_key_values=past_key_values,
                                update_past_key_values=False,
                                use_block_cache=cfg.use_block_cache,
                                block_size=block,
                            )
                            if cfg.use_block_cache:
                                block_cache = output.block_past_key_values
                            logits = torch.cat(
                                [output.logits[:, :1, :], output.logits[:, :-1, :]], dim=1
                            )[:, small_start:small_end, :]

                        token, confidence = self._sample(logits)
                        select = self._select_unmask(confidence, local_mask, step_index)
                        if force_last:
                            select = local_mask
                        local = current[:, small_start:small_end]
                        local[select] = token[select]
                        step_index += 1
                        nfe += 1

                if bool((x_t[:, -block:] == cfg.mask_token_id).any()):
                    raise RuntimeError("native dense denoising ended with unresolved masks")

                final_output = self.model.forward(
                    input_ids=x_t[:, -block:],
                    use_cache=True,
                    past_key_values=past_key_values,
                    update_past_key_values=True,
                    use_block_cache=False,
                    block_size=block,
                )
                past_key_values = final_output.past_key_values
                nfe += 1
                timer.__exit__(None, None, None)
                block_timers.append(timer)
                block_steps.append(step_index)

                next_token = final_output.logits[:, -1:, :].argmax(dim=-1)
                work_ids = torch.cat([x_t, next_token], dim=1)
                if cfg.stop_token_id is not None:
                    generated = work_ids[:, original_length:]
                    stop_pos = (generated[0] == cfg.stop_token_id).nonzero(as_tuple=True)[0]
                    if stop_pos.numel():
                        cut = int(stop_pos[0].item()) + 1
                        work_ids = work_ids[:, : original_length + cut]
                        stop_found = True
                        break

        decode_ms = decode_timer.milliseconds()
        generated_ids = work_ids[:, original_length : original_length + cfg.max_new_tokens]
        sequences = torch.cat([original_input, generated_ids], dim=1)
        texts = [
            self.tokenizer.decode(row.tolist(), skip_special_tokens=True)
            for row in generated_ids.detach().cpu()
        ]
        block_ms = [x.milliseconds(synchronize=False) for x in block_timers]
        cache_bytes = dynamic_cache_nbytes(past_key_values)
        metrics: dict[str, Any] = {
            **prefill_metrics,
            "decode_ms": decode_ms,
            "requests": int(generated_ids.shape[0]),
            "generated_tokens": int(generated_ids.numel()),
            "generated_tokens_per_request": int(generated_ids.shape[1]),
            "tokens_per_second": (
                float(generated_ids.numel()) / (decode_ms / 1000.0)
                if decode_ms > 0
                else None
            ),
            "nfe": nfe,
            "denoising_steps_per_block": block_steps,
            "time_per_output_block_ms": block_ms,
            "mean_tpob_ms": sum(block_ms) / len(block_ms) if block_ms else None,
            "native_cache_bytes": cache_bytes,
            "dense_cache_equivalent_bytes": cache_bytes,
            "cache_compression_ratio": 1.0 if cache_bytes else None,
            "cache_tokens": (
                int(past_key_values.get_seq_length())
                if past_key_values is not None
                else 0
            ),
            "stop_found": stop_found,
            "semantic": "dense",
            "engine": "official_direct",
            "config_name": self.config.name,
        }
        if self.device.type == "cuda":
            metrics["peak_cuda_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated(self.device)
            )
            metrics["peak_cuda_reserved_bytes"] = int(
                torch.cuda.max_memory_reserved(self.device)
            )
        return GenerationResult(
            sequences=sequences,
            generated_ids=generated_ids,
            texts=texts,
            metrics=metrics,
            trace=None,
        )
