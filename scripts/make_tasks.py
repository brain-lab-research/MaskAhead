#!/usr/bin/env python3
"""Expand a matrix YAML into a task list for scripts/gpu_queue.sh.

Each emitted line is the argument list of scripts/run_task.py. Order: block
priority, then dataset (the matrix's dataset_order, else the key order of
`datasets`), then arm (arm_order): a dataset is finished by most arms before the
next one starts, so partial results are always complete tables for the first
datasets. Arms in a block keep their listed order when arm_order is absent.

    python scripts/make_tasks.py path/to/custom_matrix.yaml > tasks/custom.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml



def main() -> int:
    spec = yaml.safe_load(Path(sys.argv[1]).read_text())
    model = spec["model"]
    datasets = spec["datasets"]
    ds_order = spec.get("dataset_order") or list(datasets)
    arm_order = spec.get("arm_order") or []
    lines = []
    for block in spec["blocks"]:
        names = list(datasets) if block.get("datasets", "all") == "all" else block["datasets"]
        variants = block.get("variants") or {None: []}
        for cfg in block["configs"]:
            for tag, sets in variants.items():
                for bench in names:
                    d = datasets[bench] or {}
                    parts = [f"--model {model}", f"--config configs/{cfg}.yaml", f"--benchmark {bench}"]
                    if d.get("max_new_tokens"):
                        parts.append(f"--max-new-tokens {d['max_new_tokens']}")
                    limit = block.get("limit", d.get("limit"))
                    if limit:
                        parts.append(f"--limit {limit}")
                    for s in sets:
                        parts.append(f"--set {s}")
                    if tag:
                        parts.append(f"--tag {tag}")
                    drank = ds_order.index(bench) if bench in ds_order else len(ds_order)
                    arank = arm_order.index(cfg) if cfg in arm_order else len(arm_order) + block["configs"].index(cfg)
                    lines.append((block.get("priority", 0), drank, arank, " ".join(parts)))
    seen = set()
    for *_, line in sorted(lines, key=lambda x: (x[0], x[1], x[2])):
        if line not in seen:
            seen.add(line)
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
