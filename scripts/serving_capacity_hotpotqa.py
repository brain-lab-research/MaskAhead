#!/usr/bin/env python3
"""Matched serving capacity on varied, equal-length HotpotQA request batches.

Each batch row comes from a different HotpotQA example where possible. Long
contexts are middle-truncated to the stratum length while the complete question
tail is preserved. The exact row order is reused by every cache policy.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from maskahead.config import ExperimentConfig  # noqa: E402
from maskahead.eval.benchmarks import load_benchmark  # noqa: E402
from maskahead.eval.common import encode_prompt, question_start  # noqa: E402
import smoke_test  # noqa: E402

CONFIGS = ("dense", "mage", "ours", "ours_prefill_qprobe")


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def load_prompts(tokenizer, targets):
    examples = load_benchmark("hotpotqa", tokenizer=tokenizer, limit=200, split="test")
    pools = {target: [] for target in targets}
    for example in examples:
        ids = encode_prompt(tokenizer, example.prompt, max_input_tokens=30000,
                            use_chat_template=True, device="cpu")[0]
        start = question_start(tokenizer, ids[None, :],
                               (example.metadata or {}).get("question_suffix"))
        if start is None:
            continue
        tail = ids.numel() - start
        for target in targets:
            if ids.numel() < target or tail >= target:
                continue
            # Preserve both the document opening and the context immediately
            # preceding the entire question/chat tail.
            context = target - tail
            front = context // 2
            back = context - front
            trimmed = torch.cat([ids[:front], ids[start - back:start], ids[start:]])
            assert trimmed.numel() == target
            pools[target].append((example.example_id, trimmed, context))
    return pools


def config(name):
    raw = yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text())
    raw["generation"].update(max_new_tokens=256, stop_token_id=None, loop_stop=False)
    return ExperimentConfig.from_dict(raw)


def run_one(model, tokenizer, gen_cls, cfg, prompts, batch_size, *, warm=False):
    chosen = [prompts[i % len(prompts)] for i in range(batch_size)]
    ids = torch.stack([row[1] for row in chosen]).to("cuda:0")
    starts = torch.tensor([row[2] for row in chosen], device="cuda:0")
    try:
        gen = gen_cls(model, tokenizer, cfg)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        tick = time.perf_counter()
        out = gen.generate(ids, question_start=starts)
        torch.cuda.synchronize()
        wall = time.perf_counter() - tick
        if warm:
            return {"status": "warmup"}
        metrics = out.metrics
        generated = int(metrics["generated_tokens"])
        if generated != batch_size * 256:
            raise RuntimeError(f"generated {generated} tokens, expected {batch_size * 256}")
        return {
            "status": "ok", "batch": batch_size, "distinct_requests": len({row[0] for row in chosen}),
            "wall_s": round(wall, 3), "requests_per_s": batch_size / wall,
            "output_tokens_per_s_e2e": generated / wall,
            "output_tokens_per_s_decode": generated / (metrics["decode_ms"] / 1000),
            "prefill_s": ((metrics.get("prefill_ms") or 0) + (metrics.get("prefill_chunked_ms") or 0)) / 1000,
            "peak_allocated_gib": metrics["peak_cuda_allocated_total_bytes"] / 2**30,
            "peak_reserved_gib": metrics["peak_cuda_reserved_total_bytes"] / 2**30,
            "peak_kv_gib": metrics["cache_peak_used_bytes"] / 2**30,
            "generated_tokens": generated, "nfe": metrics.get("nfe"),
        }
    except (torch.cuda.OutOfMemoryError, torch.AcceleratorError) as exc:
        # Some PPU/CUDA allocation failures surface while recording a CUDA
        # event and are reported as AcceleratorError rather than OOMError.
        if "out of memory" not in str(exc).lower():
            raise
        return {"status": "out_of_memory", "batch": batch_size,
                "error": str(exc).splitlines()[0]}
    finally:
        # Release references before attempting a boundary trial after an OOM.
        if "out" in locals():
            del out
        if "gen" in locals():
            del gen
        del ids, starts
        gc.collect()
        torch.cuda.empty_cache()


def boundary(model, tokenizer, gen_cls, cfg, prompts, target, name, data, out_path, max_batch,
             deadline_fast=False):
    group = f"{target}/{name}"
    results = data.setdefault("trials", {})
    coarse = sorted({size for size in (1, 16, 32, 64, 128, 256, max_batch)
                     if size <= max_batch})

    def trial(size):
        key = f"{group}/{size}"
        if key not in results:
            results[key] = run_one(model, tokenizer, gen_cls, cfg, prompts, size)
            atomic_json(out_path, data)
            print(key, json.dumps(results[key]), flush=True)
        return results[key]["status"] == "ok"

    if deadline_fast and name == "ours_prefill_qprobe":
        # Reuse all completed full-generation trials. For long prompts the
        # cached 256-request run is already near device capacity, so probing
        # 1024, 640, ... again spends time without narrowing the useful edge.
        prefix = group + "/"
        observed = [(int(key[len(prefix):]), row["status"])
                    for key, row in results.items() if key.startswith(prefix)]
        safe = max(size for size, status in observed if status == "ok")
        failed = min((size for size, status in observed
                      if status == "out_of_memory" and size > safe), default=None)
        if failed is None and safe < max_batch:
            step = 8
            while safe < max_batch:
                candidate = min(max_batch, safe + step)
                if trial(candidate):
                    safe = candidate
                    step = min(step * 2, 128)
                else:
                    failed = candidate
                    break
        while failed is not None and failed - safe > 8:
            middle = max(safe + 8, min(((safe + failed) // 16) * 8, failed - 8))
            if trial(middle):
                safe = middle
            else:
                failed = middle
        data.setdefault("boundaries", {})[group] = {
            "safe_batch": safe, "first_oom_batch": failed, "resolution": 8}
        atomic_json(out_path, data)
        return

    safe, failed = 0, None
    for size in coarse:
        if trial(size):
            safe = size
        else:
            failed = size
            break
    if failed is not None:
        # Resolve the boundary within eight concurrent requests; a full integer
        # search would spend time without changing the paper's conclusion.
        while failed - safe > 8:
            middle = ((safe + failed) // 16) * 8
            middle = max(safe + 8, min(middle, failed - 8))
            if trial(middle):
                safe = middle
            else:
                failed = middle
    data.setdefault("boundaries", {})[group] = {"safe_batch": safe, "first_oom_batch": failed,
                                                 "resolution": 8}
    atomic_json(out_path, data)


def matched(model, tokenizer, gen_cls, prompts, target, data, out_path):
    limits = [data["boundaries"].get(f"{target}/{name}", {}).get("safe_batch", 0)
              for name in CONFIGS]
    if not all(limits):
        return
    common = min(limits)
    data.setdefault("matched_batches", {})[str(target)] = common
    atomic_json(out_path, data)
    for name in CONFIGS:
        key = f"{target}/{name}/{common}"
        if key not in data["trials"]:
            data["trials"][key] = run_one(model, tokenizer, gen_cls, config(name), prompts, common)
            atomic_json(out_path, data)
            print(key, json.dumps(data["trials"][key]), flush=True)
        if data["trials"][key]["status"] != "ok":
            raise RuntimeError(f"matched batch unexpectedly failed: {key}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--strata", nargs="+", type=int, default=[8192, 12288, 16384])
    ap.add_argument("--max-batch", type=int, default=1024)
    ap.add_argument("--prepare-only", action="store_true",
                    help="validate prompt strata and write metadata without loading a GPU model")
    ap.add_argument("--matched-only", action="store_true",
                    help="finish matched comparisons from previously measured boundaries")
    ap.add_argument("--deadline-fast", action="store_true",
                    help="reuse recorded OOM brackets and probe only missing long-context sizes")
    args = ap.parse_args()
    if args.max_batch < 1:
        ap.error("--max-batch must be positive")
    data = json.loads(args.out.read_text()) if args.out.exists() else {"trials": {}, "boundaries": {}}
    if args.prepare_only:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "Efficient-Large-Model/Fast_dLLM_v2_7B", trust_remote_code=True
        )
    else:
        model, tokenizer, gen_cls, _ = smoke_test.load("fastdllm", "cuda:0")
    pools = load_prompts(tokenizer, args.strata)
    data["strata"] = {str(target): {"source_examples": len(pools[target]),
                                   "prompt_tokens": target} for target in args.strata}
    data["conditions"] = {"output_tokens_per_request": 256, "stop_token_id": None,
                          "loop_stop": False, "policies": list(CONFIGS),
                          "prompt_normalization": "middle truncation; full question tail preserved"}
    quality_path = ROOT / "results/fastdllm/hotpotqa/ours_prefill_qprobe.summary.json"
    if quality_path.exists():
        quality = json.loads(quality_path.read_text())
        data["question_prefill_quality_200"] = {"num_examples": quality.get("num_examples"),
                                                 "mean_score": quality.get("mean_score")}
    atomic_json(args.out, data)
    if args.prepare_only:
        print(json.dumps(data["strata"], indent=2), flush=True)
        return
    for target in args.strata:
        prompts = pools[target]
        if not prompts:
            print(f"stratum {target}: no sufficiently long HotpotQA prompts; skipped", flush=True)
            continue
        if not args.matched_only:
            for name in CONFIGS:
                cfg = config(name)
                # A short one-block generation warms the policy-specific kernels.
                warm = config(name)
                warm.generation.max_new_tokens = 32
                run_one(model, tokenizer, gen_cls, warm, prompts, 1, warm=True)
                boundary(model, tokenizer, gen_cls, cfg, prompts, target, name,
                         data, args.out, args.max_batch, deadline_fast=args.deadline_fast)
        matched(model, tokenizer, gen_cls, prompts, target, data, args.out)
    atomic_json(args.out, data)


if __name__ == "__main__":
    main()
