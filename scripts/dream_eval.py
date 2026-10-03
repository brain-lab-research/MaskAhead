#!/usr/bin/env python3
"""DreamReasoner-8B evaluation: one config, one benchmark, resumable.

Dream's remote code needs transformers 5.x, Fast-dLLM v2 pins 4.53.1, so the
two models run from separate environments (see README). Rows have the same
schema as maskahead.eval.quality, so scripts/report.py reads both.

    python scripts/dream_eval.py --config configs/ours.yaml --benchmark math500 \
        --max-new-tokens 8192 --output results/dream/math500/ours.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maskahead.config import ExperimentConfig  # noqa: E402
from maskahead.eval.benchmarks import load_benchmark, max_new_tokens_for, uses_chat_template  # noqa: E402
from maskahead.eval.common import encode_prompt  # noqa: E402
from maskahead.eval.metrics import score_prediction  # noqa: E402
from maskahead.eval.resume import load_resume_rows  # noqa: E402
from maskahead.runtime import session as session_mod  # noqa: E402
from maskahead.runtime.dream_generator import DreamBitSieveGenerator  # noqa: E402

MODEL_ID = "Dream-org/DreamReasoner-8B"


def build_config(args) -> ExperimentConfig:
    raw = ExperimentConfig.load(Path(args.config)).to_dict()
    gen = raw["generation"]
    gen.update(
        max_new_tokens=args.max_new_tokens or max_new_tokens_for(args.benchmark),
        block_size=args.block_size, threshold=args.threshold, top_p=1.0, temperature=0.0,
    )
    raw["coverage_diagnostics"] = False
    return ExperimentConfig.from_dict(raw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default=MODEL_ID, help="hub id or local checkpoint directory")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, help="first N examples (default: the whole split)")
    ap.add_argument("--max-new-tokens", type=int)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.9)
    args = ap.parse_args()

    cfg = build_config(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.cuda.set_device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=True, dtype=torch.bfloat16, low_cpu_mem_usage=True
        ).to(args.device).eval()
    )
    examples = load_benchmark(args.benchmark, tokenizer=tok, limit=args.limit, split="test")
    done, _ = load_resume_rows(
        out, benchmark=args.benchmark, config=cfg, model_id=args.model, model_revision=None,
        dtype="bf16", valid_ids={str(e.example_id) for e in examples},
    )
    todo = [e for e in examples if str(e.example_id) not in done]
    print(f"{cfg.name} | {args.benchmark} | {len(examples)} examples, {len(done)} done, "
          f"max_new={cfg.generation.max_new_tokens}", flush=True)
    if not todo:
        return 0

    gen = DreamBitSieveGenerator(model, tok, cfg)
    t0 = time.time()
    try:
        with out.open("a", encoding="utf-8") as fh:
            for i, ex in enumerate(todo):
                ids = encode_prompt(
                    tok, ex.prompt,
                    max_input_tokens=cfg.max_cache_tokens - cfg.generation.max_new_tokens,
                    use_chat_template=uses_chat_template(args.benchmark), device=args.device,
                )
                session_mod.CURRENT_EXAMPLE = str(ex.example_id)
                t_ex = time.time()
                r = gen.generate(ids)
                fh.write(json.dumps({
                    "id": ex.example_id,
                    "benchmark": args.benchmark,
                    "config": json.loads(json.dumps(cfg.to_dict())),
                    "model_id": args.model,
                    "model_revision": None,
                    "dtype": "bf16",
                    "prompt_tokens": int(ids.shape[1]),
                    "prediction": r.texts[0],
                    "references": ex.references,
                    "score": score_prediction(args.benchmark, r.texts[0], ex.references),
                    "metadata": ex.metadata,
                    "wall_s": time.time() - t_ex,
                    "runtime": r.metrics,
                }, ensure_ascii=False) + "\n")
                fh.flush()
                if (i + 1) % 10 == 0:
                    print(f"  {i + 1}/{len(todo)} ({(time.time() - t0) / (i + 1):.1f} s/ex)", flush=True)
    finally:
        gen.patch.unpatch()
    print(f"done {cfg.name} {args.benchmark} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
