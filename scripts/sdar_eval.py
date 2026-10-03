#!/usr/bin/env python3
"""SDAR quality evaluation using the same schema as Dream and Fast-dLLM."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maskahead.config import ExperimentConfig
from maskahead.eval.benchmarks import load_benchmark, max_new_tokens_for, uses_chat_template
from maskahead.eval.common import encode_prompt
from maskahead.eval.metrics import score_prediction
from maskahead.eval.resume import load_resume_rows
from maskahead.runtime.sdar_generator import SDARBitSieveGenerator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="JetLM/SDAR-8B-Chat-b32")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-new-tokens", type=int)
    ap.add_argument("--threshold", type=float, default=0.85)
    args = ap.parse_args()

    raw = ExperimentConfig.load(Path(args.config)).to_dict()
    raw["generation"].update(block_size=32, threshold=args.threshold,
                             mask_token_id=151669, stop_token_id=151645,
                             max_new_tokens=args.max_new_tokens or max_new_tokens_for(args.benchmark),
                             temperature=0.0, top_p=1.0)
    raw["coverage_diagnostics"] = False
    cfg = ExperimentConfig.from_dict(raw)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.cuda.set_device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).to(args.device).eval()
    examples = load_benchmark(args.benchmark, tokenizer=tok, limit=args.limit, split="test")
    done, _ = load_resume_rows(out, benchmark=args.benchmark, config=cfg, model_id=args.model,
                               model_revision=None, dtype="bf16",
                               valid_ids={str(e.example_id) for e in examples})
    todo = [e for e in examples if str(e.example_id) not in done]
    print(f"SDAR {cfg.name} {args.benchmark}: {len(todo)}/{len(examples)} pending", flush=True)
    if not todo:
        return 0
    gen = SDARBitSieveGenerator(model, tok, cfg)
    try:
        with out.open("a", encoding="utf-8") as fh:
            for ex in todo:
                ids = encode_prompt(tok, ex.prompt,
                                    max_input_tokens=cfg.max_cache_tokens - cfg.generation.max_new_tokens,
                                    use_chat_template=uses_chat_template(args.benchmark), device=args.device)
                started = time.time()
                result = gen.generate(ids)
                prediction = result.texts[0]
                row = {"id": ex.example_id, "benchmark": args.benchmark,
                       "config": cfg.to_dict(), "model_id": args.model, "model_revision": None,
                       "dtype": "bf16", "prompt_tokens": int(ids.shape[1]),
                       "prediction": prediction, "references": ex.references,
                       "score": score_prediction(args.benchmark, prediction, ex.references),
                       "metadata": ex.metadata, "wall_s": time.time() - started,
                       "runtime": result.metrics}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                print(f"{ex.example_id}: score={row['score']} wall={row['wall_s']:.1f}s", flush=True)
    finally:
        gen.patch.unpatch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
