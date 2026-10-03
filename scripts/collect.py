#!/usr/bin/env python3
"""Pack everything that has to come back from a machine into one archive.

Writes results/SUMMARY.csv (one row per cell: the numbers the paper tables need),
results/REPORT.md (scripts/report.py), results/STATUS.txt (which cells are complete,
which are partial or failed), and results_bundle.tar.gz with those files plus every
raw .jsonl/.yaml/.env.json (predictions are kept so answers can be re-graded).

    python scripts/collect.py [--results results] [--out results_bundle.tar.gz]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
import tarfile
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import report  # noqa: E402

FULL = {"gsm8k": 1319, "math500": 500, "aime2025": 30, "hotpotqa": 200, "musique": 200, "narrativeqa": 200}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "results_bundle.tar.gz"))
    args = ap.parse_args()
    root = Path(args.results)
    rows, status = [], []
    for path in sorted(root.glob("*/*/*.jsonl")):
        model, bench, run = path.parts[-3], path.parts[-2], path.stem
        R = list(report.load(path).values())
        if not R:
            continue
        m = [r.get("runtime") or {} for r in R]
        env = {}
        if path.with_suffix(".env.json").exists():
            env = json.loads(path.with_suffix(".env.json").read_text())
        row = {
            "model": model, "benchmark": bench, "run": run, "n": len(R),
            "complete": len(R) >= FULL.get(bench, len(R)),
            "score": report.mean([r["score"] for r in R]),
            "gen_tokens": report.mean([x.get("generated_tokens") for x in m]),
            "tok_per_s": report.mean([x.get("tokens_per_second") for x in m]),
            "s_per_example": report.mean([report.seconds(r) for r in R]),
            "prompt_tokens": report.mean([r.get("prompt_tokens") for r in R]),
            "cache_vs_bf16": report.mean([report.cache_vs_bf16(r) for r in R]),
            "peak_cache_vs_bf16": report.mean([report.peak_vs_bf16(r) for r in R]),
            "state_vs_bf16": report.mean([report.state_vs_bf16(r) for r in R]),
            "peak_gb": report.mean([(x.get("peak_cuda_allocated_bytes") or 0) / 2**30 or None for x in m]),
            "hit_limit": sum(1 for x in m if x.get("stop_found") is False and not x.get("loop_stopped")),
            "loop_stop": sum(1 for x in m if x.get("loop_stopped")),
            "gpu": env.get("gpu"), "git_commit": (env.get("git_commit") or "")[:10],
        }
        rows.append(row)
        status.append(f"{'OK     ' if row['complete'] else 'PARTIAL'} {model}/{bench}/{run}: "
                      f"{len(R)}/{FULL.get(bench, '?')}")
    # One table must come from one code version and one GPU type.
    for model in sorted({r["model"] for r in rows}):
        for key in ("gpu", "git_commit"):
            vals = sorted({str(r[key]) for r in rows if r["model"] == model})
            if len(vals) > 1:
                status.append(f"WARNING {model}: cells differ in {key}: {vals}")
    for failed in sorted((ROOT / "tasks").glob("*.failed")):
        status += [f"FAILED  {line}" for line in failed.read_text().splitlines() if line.strip()]
    with (root / "SUMMARY.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["model"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    buf = io.StringIO()
    with redirect_stdout(buf):
        sys.argv = ["report.py", "--results", str(root)]
        report.main()
    (root / "REPORT.md").write_text(buf.getvalue())
    (root / "STATUS.txt").write_text("\n".join(status) + "\n")
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
    except Exception:
        commit = ""
    with tarfile.open(args.out, "w:gz") as tar:
        for name in ("SUMMARY.csv", "REPORT.md", "STATUS.txt"):
            tar.add(root / name, arcname=f"results/{name}")
        for p in sorted(root.glob("*/*/*")):
            if p.suffix in (".jsonl", ".yaml") or p.name.endswith(".env.json"):
                tar.add(p, arcname=str(Path("results") / p.relative_to(root)))
        logs = root / "logs"
        if logs.exists():
            tar.add(logs, arcname="results/logs")
    print(f"{len(rows)} cells ({sum(r['complete'] for r in rows)} complete), commit {commit[:10]}")
    print(f"wrote {root / 'SUMMARY.csv'}, {root / 'REPORT.md'}, {root / 'STATUS.txt'}, {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
