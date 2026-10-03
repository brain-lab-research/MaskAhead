#!/usr/bin/env python3
"""Speed table: decode ms per forward, tokens/s, prefill time and peak memory per config.

One process on the GPU, fixed generation length (stop token and loop stop off), one warm-up
generation per case so Triton compiles every path first. For batch-throughput runs, pass an
increasing list of batch sizes; the sweep stops at the first OOM. Run it ALONE on a GPU.

    CUDA_VISIBLE_DEVICES=0 python scripts/speed_test.py --model dream --out results/speed_dream.json \\
        --configs dense mage ours ours_k4v4 --cases hotpotqa:256 --batch-sizes 1 2 4 8 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from maskahead.config import ExperimentConfig  # noqa: E402
from maskahead.eval.benchmarks import load_benchmark, uses_chat_template  # noqa: E402
from maskahead.eval.common import encode_prompt, question_start  # noqa: E402
import smoke_test  # noqa: E402  (model loading)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("fastdllm", "dream"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--cases", nargs="+", default=["math500:1024", "hotpotqa:256"])
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1])
    args = ap.parse_args()
    model, tok, gen_cls, _ = smoke_test.load(args.model, "cuda:0")
    res = {}
    batch_sizes = sorted(set(args.batch_sizes))
    if not batch_sizes or batch_sizes[0] < 1:
        ap.error("--batch-sizes must contain positive integers")
    for name in args.configs:
        raw = yaml.safe_load(open(ROOT / "configs" / f"{name}.yaml"))
        for case in args.cases:
            bench, mx = case.split(":")
            ex = load_benchmark(bench, tokenizer=tok, limit=1, split="test")[0]
            ids = encode_prompt(tok, ex.prompt, max_input_tokens=30000, use_chat_template=uses_chat_template(bench),
                                device="cuda:0")
            qstart = question_start(tok, ids, (ex.metadata or {}).get("question_suffix"))
            for batch_size in batch_sizes:
                batch_ids = ids.expand(batch_size, -1).contiguous()
                r2 = json.loads(json.dumps(raw))
                r2["generation"].update(max_new_tokens=int(mx), stop_token_id=None, loop_stop=False)
                warm = json.loads(json.dumps(r2)); warm["generation"]["max_new_tokens"] = 160
                key = f"{name}/{bench}/batch{batch_size}"
                try:
                    warm_gen = gen_cls(model, tok, ExperimentConfig.from_dict(warm))
                    if args.model == "fastdllm":
                        warm_gen.generate(batch_ids, question_start=qstart)
                    else:
                        warm_gen.generate(batch_ids)
                    gen = gen_cls(model, tok, ExperimentConfig.from_dict(r2))
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.perf_counter()
                    if args.model == "fastdllm":
                        r = gen.generate(batch_ids, question_start=qstart)
                    else:
                        r = gen.generate(batch_ids)
                    torch.cuda.synchronize(); wall = time.perf_counter() - t
                except torch.cuda.OutOfMemoryError as exc:
                    res[key] = {"prompt": int(ids.shape[1]), "batch_size": batch_size,
                                "status": "out_of_memory", "total_mem_gib": round(
                                    torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
                                "error": str(exc).splitlines()[0]}
                    print(key, json.dumps(res[key]), flush=True)
                    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                    json.dump(res, open(args.out, "w"), indent=1)
                    del batch_ids
                    torch.cuda.empty_cache()
                    break
                m = r.metrics
                dec = (m.get("decode_ms") or 0) / 1e3
                res[key] = {
                    "prompt": int(ids.shape[1]), "batch_size": batch_size,
                    "gen": m.get("generated_tokens"), "nfe": m.get("nfe"),
                    "ms_per_forward": round(dec * 1e3 / max(m.get("nfe") or 1, 1), 2),
                    "tok_s_decode": round((m.get("generated_tokens") or 0) / max(dec, 1e-9), 1),
                    "prefill_s": round(((m.get("prefill_ms") or 0) + (m.get("prefill_chunked_ms") or 0)) / 1e3, 3),
                    "wall_s": round(wall, 2), "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                    "total_mem_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
                    "cache_peak_used_bytes": m.get("cache_peak_used_bytes"), "bf16_equiv_bytes": m.get("bf16_equiv_bytes"),
                }
                print(key, json.dumps(res[key]), flush=True)
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                json.dump(res, open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
