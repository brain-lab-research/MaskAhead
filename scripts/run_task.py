#!/usr/bin/env python3
"""Run one (model, config, benchmark) cell, resumably, and log its environment.

    python scripts/run_task.py --model dream --config configs/ours.yaml --benchmark math500 \
        --max-new-tokens 8192 [--limit 50] [--set eviction.capacity_percent=25] [--tag c25]

Output: results/<model>/<benchmark>/<config>[_<tag>].jsonl, one row per example,
plus a .env.json next to it (versions, GPU, git commit, command). Re-running the
same command continues where it stopped; a changed config is refused, not mixed.

Interpreters: FASTDLLM_PYTHON and DREAM_PYTHON (default: the current python).
Checkpoints:  FASTDLLM_MODEL (default Efficient-Large-Model/Fast_dLLM_v2_7B),
              DREAM_MODEL (default Dream-org/DreamReasoner-8B).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "fastdllm": ("FASTDLLM_PYTHON", "FASTDLLM_MODEL", "Efficient-Large-Model/Fast_dLLM_v2_7B"),
    "dream": ("DREAM_PYTHON", "DREAM_MODEL", "Dream-org/DreamReasoner-8B"),
    "sdar": ("SDAR_PYTHON", "SDAR_MODEL", "JetLM/SDAR-8B-Chat-b32"),
}


def apply_overrides(raw: dict, sets: list[str]) -> dict:
    for item in sets:
        key, _, val = item.partition("=")
        node = raw
        *path, leaf = key.split(".")
        for p in path:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(val)
    return raw


def env_info(python: str) -> dict:
    probe = (
        "import json,torch,transformers;"
        "d={'torch':torch.__version__,'transformers':transformers.__version__,"
        "'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None};"
        "\ntry:\n import triton;d['triton']=triton.__version__\nexcept Exception: pass\nprint(json.dumps(d))"
    )
    try:
        libs = json.loads(subprocess.run([python, "-c", probe], capture_output=True, text=True, check=True).stdout)
    except Exception as exc:  # recorded, not fatal
        libs = {"error": repr(exc)}
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                                text=True).stdout.strip() or None
    except Exception:
        commit = None
    return {"host": platform.node(), "git_commit": commit, **libs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--max-new-tokens", type=int)
    ap.add_argument("--limit", type=int, help="first N examples (default: the whole split)")
    ap.add_argument("--example-id-file", help="run only benchmark example IDs listed one per line")
    ap.add_argument("--niah-contexts", help="comma-separated context lengths for NIAH diagnostics")
    ap.add_argument("--niah-depths", help="comma-separated target depths for NIAH diagnostics")
    ap.add_argument("--set", action="append", default=[], help="config override, e.g. eviction.capacity_percent=25")
    ap.add_argument("--tag", help="suffix for the run name, required with --set")
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.set and not args.tag:
        ap.error("--set changes the config: give the run a --tag")

    py_env, model_env, model_default = MODELS[args.model]
    python = os.environ.get(py_env, sys.executable)
    model = os.environ.get(model_env, model_default)

    raw = yaml.safe_load(Path(args.config).read_text())
    name = Path(args.config).stem + (f"_{args.tag}" if args.tag else "")
    raw = apply_overrides(raw, args.set)
    raw["name"] = name
    out = Path(args.results) / args.model / args.benchmark / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg_path = out.with_suffix(".yaml")
    if cfg_path.exists():
        stored = yaml.safe_load(cfg_path.read_text())
        if stored != raw:
            raise ValueError(f"resolved config differs from existing run: {cfg_path}; use a new --tag")
    else:
        cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False))

    if args.model == "fastdllm":
        cmd = [python, "-m", "maskahead.eval.quality", "--config", str(cfg_path),
               "--benchmark", args.benchmark, "--model", model, "--device", args.device, "--output", str(out)]
    else:
        evaluator = "sdar_eval.py" if args.model == "sdar" else "dream_eval.py"
        cmd = [python, str(ROOT / "scripts" / evaluator), "--config", str(cfg_path),
               "--benchmark", args.benchmark, "--model", model, "--device", args.device, "--output", str(out)]
    if args.max_new_tokens:
        cmd += ["--max-new-tokens", str(args.max_new_tokens)]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.example_id_file:
        cmd += ["--example-id-file", args.example_id_file]
    if args.niah_contexts:
        cmd += ["--niah-contexts", args.niah_contexts]
    if args.niah_depths:
        cmd += ["--niah-depths", args.niah_depths]

    info = {"command": cmd, "argv": sys.argv, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            **env_info(python)}
    env_path = out.with_suffix(".env.json")
    if not env_path.exists():
        env_path.write_text(json.dumps(info, indent=2))
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
               TOKENIZERS_PARALLELISM="false")
    print(" ".join(cmd), flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, env=env, cwd=str(ROOT))
    print(f"exit {rc} after {(time.time() - t0) / 60:.1f} min: {out}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
