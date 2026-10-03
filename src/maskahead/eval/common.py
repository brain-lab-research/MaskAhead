from __future__ import annotations

from typing import Any

import torch


def parse_dtype(name: str) -> torch.dtype:
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return aliases[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown dtype: {name}") from exc


def load_fast_dllm(
    model_id: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    revision: str | None = None,
    attn_implementation: str | None = None,
    adapter: str | None = None,
):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Transformers is required; install this project in its evaluation env") from exc

    # torch.cuda.current_device() stays 0 until something sets it - creating
    # tensors on cuda:N (N != 0) via device= or .to() does NOT move it. Every
    # packed/selector kernel launch in kernels/ops.py then runs against
    # device 0's context while the tensors it is handed live on device N,
    # which Triton reports as "Pointer argument (at 0) cannot be accessed from
    # Triton (cpu tensor?)" - a deterministic crash on every call, not a CPU
    # tensor at all. Only avoided previously by restricting visibility with
    # CUDA_VISIBLE_DEVICES so the requested index was always 0.
    # set_device also rejects an unindexed "cuda" - which is exactly what
    # quality.py, performance.py and every tests/ entry point default to, so
    # they all died here while scripts/run_experiments.py worked around it by
    # passing --device cuda:0. Normalize instead of making each caller do it.
    if device not in ("auto", "cpu") and torch.cuda.is_available():
        target = torch.device(device)
        torch.cuda.set_device(
            target.index if target.index is not None else torch.cuda.current_device()
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, revision=revision
    )
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "revision": revision,
    }
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    if device == "auto":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if device != "auto":
        model = model.to(device)

    if adapter:
        # Merged, not kept as a live PEFT wrapper: the runtime patches
        # attention module forwards by identity, and an adapter wrapper would
        # sit between them and the session. Merging folds the LoRA into the
        # base weights so decoding sees plain Linear layers.
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
        if device != "auto":
            model = model.to(device)

    model.eval()
    return model, tokenizer


def question_start(tokenizer, input_ids: torch.Tensor, suffix: str | None, skip: int = 2) -> int | None:
    """First position of `suffix` (the prompt tail after the context) in input_ids.

    The first `skip` suffix tokens are not matched (they may merge with the end of
    the context) and are counted back; None if the tail is not found.
    """
    if not suffix:
        return None
    tail = tokenizer(suffix, add_special_tokens=False).input_ids[skip:]
    if not tail:
        return None
    ids = input_ids[0].tolist()
    n = len(tail)
    for i in range(len(ids) - n, -1, -1):
        if ids[i:i + n] == tail:
            return max(0, i - skip)
    return None


def encode_prompt(
    tokenizer,
    prompt: str,
    *,
    max_input_tokens: int,
    use_chat_template: bool = True,
    device: torch.device | str,
) -> torch.Tensor:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        try:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            )
        except Exception:
            ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
    else:
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
    # transformers 4.x hands back a tensor here; 5.x hands back a BatchEncoding.
    # Unwrap rather than pin, so one prompt path serves both -- Fast-dLLM-v2
    # requires 4.53 and DreamReasoner requires 5.x.
    if not isinstance(ids, torch.Tensor):
        ids = ids["input_ids"]
    if ids.shape[1] > max_input_tokens:


        first = max_input_tokens // 2
        last = max_input_tokens - first
        ids = torch.cat([ids[:, :first], ids[:, -last:]], dim=1)
    return ids.to(device)
