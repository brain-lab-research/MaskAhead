#!/usr/bin/env python3
"""Pareto "quality vs memory" and "quality/speed vs read budget k" from results/<model>/<bench>/*.jsonl.

Memory is what a method holds, relative to a full bf16 cache of every token seen:
  - math (gsm8k, math500, aime2025): the peak over the run (KV cache + method state);
  - LongBench: the cache during generation (end of run; the peak is the whole prompt for every
    method that compresses once at the end of prefill).
Method state (eviction scores, Expected Attention statistics) is always included.

    python scripts/pareto.py --results results --model fastdllm --out figures/
writes <out>/pareto_<bench>.csv|png (and pareto_longbench.* = mean over the LongBench tasks the
runs share), and <out>/kcurve_<bench>.csv|png for the ours runs at C = 20 % (k <= C).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

MATH = {"gsm8k", "math500", "aime2025"}
PLOT = True          # export_numbers.py turns the images off (numbers only)
LONGBENCH = ["hotpotqa", "musique", "narrativeqa"]


def load(path: Path) -> dict:
    rows = {}
    for line in path.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            rows[str(r["id"])] = r
    return rows


CONFIGS = ["ours_noevict", "bitsieve_v1", "streaming", "herald", "dense", "mage", "caote", "dapq",
           "h2o", "ea", "ours"]          # longest names first: run names start with the config


def memory(r: dict, bench: str):
    """Memory held / full bf16 cache, method state (eviction scores, EA statistics) included."""
    m = r.get("runtime") or {}
    dense = m.get("bf16_equiv_bytes") or m.get("dense_cache_equivalent_bytes")
    if not dense:
        return None
    ea = m.get("ea_stats_bytes") or 0
    if bench not in MATH and m.get("eviction_total_bytes") is not None:
        return (m["eviction_total_bytes"] + ea) / dense          # cache + eviction state, end of run
    if m.get("cache_peak_used_bytes") is not None:
        return (m["cache_peak_used_bytes"] + (m.get("eviction_state_bytes") or 0) + ea) / dense
    return None


def ms_per_forward(r: dict):
    m = r.get("runtime") or {}
    if m.get("decode_ms") and m.get("nfe"):
        return m["decode_ms"] / m["nfe"]
    return None


def split_run(run: str):
    """(config, bits label or '', rest of the tag)."""
    cfg = next((c for c in CONFIGS if run == c or run.startswith(c + "_")), run.split("_")[0])
    tag = run[len(cfg) + 1:]
    bits = re.search(r"(?:^|_)k(\d)v(\d)(?:_|$)", tag)
    return cfg, (f"K{bits.group(1)}V{bits.group(2)}" if bits else ""), tag


def family(run: str) -> str:
    """One curve per method and bit width; dense with quantization = 'quantization only'."""
    cfg, bits, _ = split_run(run)
    if cfg == "dense" and bits:
        return "dense, quantization only"
    if cfg == "bitsieve_v1":
        return "bitsieve_v1 (K4V4)"
    return cfg + (" " + bits if bits else "")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def points(root: Path, model: str, bench: str, full: dict[str, int]):
    out = []
    for f in sorted((root / model / bench).glob("*.jsonl")):
        run = f.stem
        if any(t in run for t in ("x300", "hp50", "mu50", "nq50", "sanity", "f128")):
            continue
        R = load(f)
        if len(R) < 0.9 * full.get(bench, len(R)):
            continue                                   # only (nearly) complete cells
        v = list(R.values())
        out.append({"run": run, "family": family(run), "n": len(v),
                    "score": mean([r["score"] for r in v]),
                    "memory": mean([memory(r, bench) for r in v]),
                    "ms_per_forward": mean([ms_per_forward(r) for r in v])})
    return out


def plot(rows, path: Path, title: str, x="memory", xlabel="memory / full bf16 cache"):
    if not PLOT:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    fams = sorted({r["family"] for r in rows})
    for fam in fams:
        pts = sorted([(r[x], r["score"]) for r in rows if r["family"] == fam and r[x] is not None])
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", lw=1.5 if fam.startswith("ours") else 1.0,
                alpha=1.0 if fam.startswith("ours") else 0.7, label=fam)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_csv(rows, path: Path):
    if not rows:
        return
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["family"], r["memory"] or 0)):
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--model", default="fastdllm")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    root, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    full = {"gsm8k": 1319, "math500": 500, "aime2025": 30, "hotpotqa": 200, "musique": 200, "narrativeqa": 200}
    per = {}
    for bench in sorted(p.name for p in (root / args.model).iterdir() if p.is_dir()):
        rows = points(root, args.model, bench, full)
        per[bench] = rows
        write_csv(rows, out / f"pareto_{bench}.csv")
        plot(rows, out / f"pareto_{bench}.png", f"{args.model} / {bench}: quality vs memory")
    # LongBench average over the tasks every run has
    by_run = {}
    for bench in LONGBENCH:
        for r in per.get(bench, []):
            by_run.setdefault(r["run"], {})[bench] = r
    lb = []
    for run, d in by_run.items():
        if len(d) == len(LONGBENCH):
            lb.append({"run": run, "family": d[LONGBENCH[0]]["family"], "n": sum(x["n"] for x in d.values()),
                       "score": mean([x["score"] for x in d.values()]),
                       "memory": mean([x["memory"] for x in d.values()]),
                       "ms_per_forward": mean([x["ms_per_forward"] for x in d.values()])})
    write_csv(lb, out / "pareto_longbench.csv")
    plot(lb, out / "pareto_longbench.png", f"{args.model} / LongBench mean: quality vs memory")
    # k curve at C = 20 %: ours bf16 and K4V4 (runs lb_c20_k*, and lb_c20 = k 10 %)
    for bench in LONGBENCH:
        rows = []
        for r in per.get(bench, []):
            m = re.fullmatch(r"ours_lb_c20(?:_k(\d+p?\d*))?(_k4v4)?", r["run"])
            if m:
                k = float((m.group(1) or "10").replace("p", "."))
                rows.append({**r, "k_percent": k, "family": "ours" + (" K4V4" if m.group(2) else "")})
        if rows:
            write_csv(rows, out / f"kcurve_{bench}.csv")
            plot(rows, out / f"kcurve_{bench}.png", f"{bench}: read budget k (C = 20 %)", x="k_percent",
                 xlabel="k, % of tokens seen")
    print(f"wrote {out}/pareto_*.csv|png, kcurve_*.csv|png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
