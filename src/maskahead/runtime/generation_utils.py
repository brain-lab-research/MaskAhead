from __future__ import annotations

import time

import torch


class CudaInterval:

    def __init__(self, device: torch.device) -> None:
        self.device = torch.device(device)
        self.start: torch.cuda.Event | None = None
        self.end: torch.cuda.Event | None = None
        self.cpu_start = 0.0
        self.cpu_end = 0.0

    def __enter__(self) -> "CudaInterval":
        if self.device.type == "cuda":
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()
        else:
            self.cpu_start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.device.type == "cuda":
            assert self.end is not None
            self.end.record()
        else:
            self.cpu_end = time.perf_counter()

    def milliseconds(self, *, synchronize: bool = True) -> float:
        if self.device.type == "cuda":
            assert self.start is not None and self.end is not None
            if synchronize:
                self.end.synchronize()
            return float(self.start.elapsed_time(self.end))
        return (self.cpu_end - self.cpu_start) * 1000.0


def sample_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
) -> tuple[torch.Tensor, torch.Tensor]:

    if temperature == 0.0:
        probs = torch.softmax(logits.float(), dim=-1)
        token = probs.argmax(dim=-1)
        chosen = torch.gather(probs, -1, token.unsqueeze(-1)).squeeze(-1)
        return token, chosen

    probs = torch.softmax(logits.float() / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        remove = cumulative - sorted_probs > top_p
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        denom = sorted_probs.sum(dim=-1, keepdim=True).clamp_min_(torch.finfo(torch.float32).tiny)
        sorted_probs = sorted_probs / denom
        sampled_rank = torch.multinomial(
            sorted_probs.reshape(-1, sorted_probs.shape[-1]), 1
        ).reshape(*sorted_probs.shape[:-1])
        token = torch.gather(sorted_idx, -1, sampled_rank.unsqueeze(-1)).squeeze(-1)
    else:
        token = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).reshape(
            *probs.shape[:-1]
        )
    chosen = torch.gather(probs, -1, token.unsqueeze(-1)).squeeze(-1)
    return token, chosen


def select_unmask(
    confidence: torch.Tensor,
    mask: torch.Tensor,
    *,
    schedule: str,
    threshold: float,
    fixed_steps_per_block: int,
    step_index: int,
) -> torch.Tensor:

    if confidence.shape != mask.shape:
        raise ValueError("confidence and mask shapes must match")
    score = torch.where(mask, confidence, torch.full_like(confidence, -torch.inf))
    if schedule == "threshold":
        select = score > threshold
        any_mask = mask.any(dim=-1)


        best = score.argmax(dim=-1)
        rows = torch.arange(mask.shape[0], device=mask.device)
        select[rows[any_mask], best[any_mask]] = True
        return select & mask
    if schedule != "fixed":
        raise ValueError(f"unknown unmask schedule: {schedule}")

    remaining_steps = max(1, int(fixed_steps_per_block) - int(step_index))
    remaining = mask.sum(dim=-1)
    take = torch.div(
        remaining + remaining_steps - 1,
        remaining_steps,
        rounding_mode="floor",
    ).clamp_min_(1)
    take = torch.minimum(take, remaining)
    max_take = int(take.max().item())
    if max_take == 0:
        return torch.zeros_like(mask)
    indices = torch.topk(score, k=max_take, dim=-1).indices
    active = torch.arange(max_take, device=mask.device)[None, :] < take[:, None]
    out = torch.zeros_like(mask)
    out.scatter_(1, indices, active)
    return out & mask


def compact_dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unknown compact dtype: {name}")


def dynamic_cache_nbytes(cache) -> int:

    if cache is None:
        return 0
    total = 0
    try:
        layers = len(cache)
    except Exception:
        layers = len(getattr(cache, "key_cache", []))
    for layer_idx in range(layers):
        try:
            key, value = cache[layer_idx]
        except Exception:
            key = cache.key_cache[layer_idx]
            value = cache.value_cache[layer_idx]
        total += int(key.numel() * key.element_size())
        total += int(value.numel() * value.element_size())
    return total
