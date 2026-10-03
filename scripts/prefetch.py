#!/usr/bin/env python3
"""Download the checkpoint(s) and every benchmark once, before starting parallel workers.

    PYTHONPATH=src $FASTDLLM_PYTHON scripts/prefetch.py --model fastdllm
    PYTHONPATH=src $DREAM_PYTHON    scripts/prefetch.py --model dream
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maskahead.eval.benchmarks import load_benchmark  # noqa: E402

BENCHMARKS = ["gsm8k", "math500", "aime2025", "hotpotqa", "musique", "narrativeqa"]
MODELS = {"fastdllm": ("FASTDLLM_MODEL", "Efficient-Large-Model/Fast_dLLM_v2_7B"),
          "dream": ("DREAM_MODEL", "Dream-org/DreamReasoner-8B")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    args = ap.parse_args()
    env, default = MODELS[args.model]
    mid = os.environ.get(env, default)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    if not Path(mid).exists():
        from huggingface_hub import snapshot_download

        print("checkpoint:", snapshot_download(mid))
    for b in BENCHMARKS:
        print(f"{b}: {len(load_benchmark(b, tokenizer=tok, split='test'))} examples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
