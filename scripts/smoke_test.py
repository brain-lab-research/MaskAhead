#!/usr/bin/env python3
"""GPU generation check for one model and selected method configurations.

Loads the model once, then for each requested config (all by default) runs 2 short MATH-500
problems (selection, eviction, lookahead probe, loop stop) and 1 HotpotQA
prompt (~10K tokens: chunked prefill, prefill-time eviction). The first config
is run twice to check determinism. Prints one line per config and exits 1 on
any failure. Running all configs takes about 10-15 min on one GPU.

    python scripts/smoke_test.py --model fastdllm --configs ours dense
    python scripts/smoke_test.py --model dream --configs ours dense
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maskahead.config import ExperimentConfig  # noqa: E402
from maskahead.eval.benchmarks import load_benchmark, uses_chat_template  # noqa: E402
from maskahead.eval.common import encode_prompt  # noqa: E402
from maskahead.eval.metrics import score_prediction  # noqa: E402
from maskahead.runtime import session as session_mod  # noqa: E402

CASES = [("math500", 2, 256), ("hotpotqa", 1, 32)]
REQUIRED = ("decode_ms", "generated_tokens", "nfe")


def load(model_name: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.cuda.set_device(device)
    if model_name == "fastdllm":
        from maskahead.eval.common import load_fast_dllm
        from maskahead.runtime.generator import BitSieveGenerator

        mid = os.environ.get("FASTDLLM_MODEL", "Efficient-Large-Model/Fast_dLLM_v2_7B")
        model, tok = load_fast_dllm(mid, device=device)
        return model, tok, BitSieveGenerator, 0.95
    from maskahead.runtime.dream_generator import DreamBitSieveGenerator

    mid = os.environ.get("DREAM_MODEL", "Dream-org/DreamReasoner-8B")
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        mid, trust_remote_code=True, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device).eval()
    return model, tok, DreamBitSieveGenerator, 0.9


def run_case(gen_cls, model, tok, cfg, bench, n, max_new, device):
    raw = cfg.to_dict()
    raw["generation"]["max_new_tokens"] = max_new
    c = ExperimentConfig.from_dict(raw)
    gen = gen_cls(model, tok, c)
    rows = []
    try:
        for ex in load_benchmark(bench, tokenizer=tok, limit=n, split="test"):
            ids = encode_prompt(tok, ex.prompt, max_input_tokens=c.max_cache_tokens - max_new,
                                use_chat_template=uses_chat_template(bench), device=device)
            session_mod.CURRENT_EXAMPLE = str(ex.example_id)
            r = gen.generate(ids)
            m = r.metrics
            missing = [k for k in REQUIRED if m.get(k) is None]
            if missing:
                raise RuntimeError(f"missing metrics {missing}")
            if any(isinstance(v, float) and math.isnan(v) for v in m.values()):
                raise RuntimeError("NaN in metrics")
            rows.append({"text": r.texts[0], "score": score_prediction(bench, r.texts[0], ex.references),
                         "prompt": int(ids.shape[1]), "tok": m["generated_tokens"],
                         "ms": m["decode_ms"] + (m.get("prefill_ms") or 0) + (m.get("prefill_chunked_ms") or 0),
                         "mem": m.get("eviction_bytes_vs_unevicted")})
    finally:
        if hasattr(gen, "patch"):
            gen.patch.unpatch()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("fastdllm", "dream"), required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--configs", nargs="*", help="subset of config names (default: all)")
    args = ap.parse_args()
    model, tok, gen_cls, threshold = load(args.model, args.device)
    paths = sorted((ROOT / "configs").glob("*.yaml"))
    if args.configs:
        paths = [p for p in paths if p.stem in args.configs]
    order = sorted(paths, key=lambda p: p.stem != "ours")          # ours first, for the repeat
    failures, report, first_texts = [], [], None
    for rep, path in enumerate([order[0]] + order):
        cfg = ExperimentConfig.load(path)
        raw = cfg.to_dict()
        raw["generation"]["threshold"] = threshold
        cfg = ExperimentConfig.from_dict(raw)
        label = path.stem + (" (repeat)" if rep == 1 else "")
        t0 = time.time()
        try:
            rows = []
            for bench, n, max_new in CASES:
                rows += [(bench, r) for r in run_case(gen_cls, model, tok, cfg, bench, n, max_new, args.device)]
            texts = [r["text"] for _, r in rows]
            if rep == 0:
                first_texts = texts
            if rep == 1 and texts != first_texts:
                raise RuntimeError("repeat of the same config gave different outputs (nondeterminism)")
            desc = "  ".join(f"{b}: score {r['score']:.2f} prompt {r['prompt']} gen {r['tok']} "
                             f"{r['ms'] / 1e3:.1f}s mem {r['mem'] if r['mem'] is None else round(r['mem'], 2)}"
                             for b, r in rows)
            report.append(f"PASS {label:14s} {time.time() - t0:5.0f}s  {desc}")
        except Exception as exc:
            failures.append(label)
            report.append(f"FAIL {label:14s} {type(exc).__name__}: {exc}")
            traceback.print_exc()
        print(report[-1], flush=True)
    print("\n".join(["", f"== {args.model}: {len(report) - len(failures)}/{len(report)} passed"] + report))
    if torch.cuda.is_available():
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
