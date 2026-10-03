#!/usr/bin/env python3
"""Export the numbers the paper's tables and figures are built from (no raw predictions).

    python scripts/export_numbers.py --results results --model fastdllm --out numbers/fastdllm_v2

Writes to --out:
  cells.csv    one row per (benchmark, run): n, completeness, score, memory (method state
               included; peak for math, during generation for LongBench), peak memory, method
               state, generated tokens, ms per forward, prefill seconds, hit-limit / loop-stop
               counts, and the budget / quantization / ablation knobs of the run's config;
  paired.csv   paired sign tests on the examples both runs have: every run against dense and
               against the ours run of the same budget family (k128 / lb_cXX / main);
  pareto_*.csv, kcurve_*.csv  (scripts/pareto.py) quality vs memory, quality/speed vs k.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pareto  # noqa: E402

FULL = {"gsm8k": 1319, "math500": 500, "aime2025": 30, "hotpotqa": 200, "musique": 200,
        "narrativeqa": 200, "lcc": 200}
SKIP = ("x300", "hp50", "mu50", "nq50", "sanity", "f128", "x100")


def sign_p(w: int, l: int) -> float:
    n = w + l
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n)


def peak_memory(r: dict):
    m = r.get("runtime") or {}
    dense = m.get("bf16_equiv_bytes") or m.get("dense_cache_equivalent_bytes")
    if not dense or m.get("cache_peak_used_bytes") is None:
        return None
    return (m["cache_peak_used_bytes"] + (m.get("eviction_state_bytes") or 0)
            + (m.get("ea_stats_bytes") or 0)) / dense


def state(r: dict):
    m = r.get("runtime") or {}
    dense = m.get("bf16_equiv_bytes") or m.get("dense_cache_equivalent_bytes")
    return ((m.get("eviction_state_bytes") or 0) + (m.get("ea_stats_bytes") or 0)) / dense if dense else None


def knobs(r: dict) -> dict:
    c = r.get("config") or {}
    sel, ev, q = c.get("selector") or {}, c.get("eviction") or {}, c.get("quant") or {}
    evicts = (ev.get("policy") or "none") != "none"
    return {
        "k_bits": q.get("k_bits"), "v_bits": q.get("v_bits"),
        "selection": ("none" if c.get("semantic") == "dense" else
                      ("caote" if sel.get("value_aware") else "mass") + ("/center" if sel.get("mode") == "middle" else "")),
        "k_percent": sel.get("topk_percent"), "k_floor": sel.get("topk_floor"), "k_basis": sel.get("topk_basis", "live"),
        "eviction": ev.get("policy") if evicts else "none",
        "C_percent": ev.get("capacity_percent") if evicts else None,
        "C_floor": ev.get("capacity_floor") if evicts else None,
        "horizon_M": ev.get("lookahead_blocks") if ev.get("policy") == "lookahead" else None,
        "probe": ev.get("lookahead_probe") if ev.get("policy") == "lookahead" else None,
        "evict_score": ev.get("lookahead_score") if ev.get("policy") == "lookahead" else None,
        "sinks": ev.get("sink_tokens") if evicts else None, "W": ev.get("recent_window") if evicts else None,
        "interval": ev.get("interval_blocks") if evicts else None,
        "prefill_percent": ev.get("prefill_capacity_percent") if evicts else None,
    }


def budget_family(run: str) -> str:
    _, _, tag = pareto.split_run(run)
    for key in ("lb_c05", "lb_c10", "lb_c20", "k128", "k64"):
        if tag.startswith(key):
            return key
    return "main"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--model", default="fastdllm")
    ap.add_argument("--out", default="numbers/fastdllm_v2")
    args = ap.parse_args()
    root, out = Path(args.results) / args.model, Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cells, runs = [], {}
    for bdir in sorted(p for p in root.iterdir() if p.is_dir()):
        bench = bdir.name
        for f in sorted(bdir.glob("*.jsonl")):
            run = f.stem
            if any(t in run for t in SKIP):
                continue
            R = pareto.load(f)
            if not R:
                continue
            runs[(bench, run)] = R
            v = list(R.values())
            m = [r.get("runtime") or {} for r in v]
            cfg, bits, tag = pareto.split_run(run)
            cells.append({
                "benchmark": bench, "run": run, "config": cfg, "tag": tag, "family": pareto.family(run),
                "budget": budget_family(run), "n": len(v), "complete": len(v) >= FULL.get(bench, len(v)),
                "score": mean([r["score"] for r in v]),
                "memory": mean([pareto.memory(r, bench) for r in v]),
                "peak_memory": mean([peak_memory(r) for r in v]),
                "state": mean([state(r) for r in v]),
                "gen_tokens": mean([x.get("generated_tokens") for x in m]),
                "ms_per_forward": mean([pareto.ms_per_forward(r) for r in v]),
                "prefill_s": mean([((x.get("prefill_ms") or 0) + (x.get("prefill_chunked_ms") or 0)) / 1e3 for x in m]),
                "hit_limit": sum(1 for x in m if x.get("stop_found") is False and not x.get("loop_stopped")),
                "loop_stop": sum(1 for x in m if x.get("loop_stopped")),
                **knobs(v[0]),
            })
    fields = list(cells[0]) if cells else ["benchmark"]
    with (out / "cells.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for c in cells:
            w.writerow({k: (round(x, 4) if isinstance(x, float) else x) for k, x in c.items()})
    pairs = []
    for (bench, run), R in runs.items():
        refs = [("dense", runs.get((bench, "dense")))]
        fam = budget_family(run)
        ours = {"main": "ours", "k128": "ours_k128", "k64": "ours_k64"}.get(fam, f"ours_{fam}")
        if run != ours:
            refs.append((ours, runs.get((bench, ours))))
        for name, ref in refs:
            if not ref or name == run:
                continue
            ids = R.keys() & ref.keys()
            if not ids:
                continue
            w_ = sum(R[i]["score"] > ref[i]["score"] for i in ids)
            l_ = sum(R[i]["score"] < ref[i]["score"] for i in ids)
            pairs.append({"benchmark": bench, "run": run, "vs": name, "n": len(ids),
                          "score": round(mean([R[i]["score"] for i in ids]), 4),
                          "score_vs": round(mean([ref[i]["score"] for i in ids]), 4),
                          "wins": w_, "losses": l_, "p_sign": round(sign_p(w_, l_), 4)})
    with (out / "paired.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["benchmark", "run", "vs", "n", "score", "score_vs", "wins", "losses", "p_sign"])
        w.writeheader()
        w.writerows(pairs)
    # Pareto / k-curve tables (CSV only)
    sys.argv = ["pareto.py", "--results", args.results, "--model", args.model, "--out", str(out)]
    pareto.PLOT = False
    pareto.main()
    (out / "README.md").write_text(
        f"# Numbers for the paper tables and figures ({args.model})\n\n"
        f"Exported {datetime.datetime.now():%Y-%m-%d %H:%M} by `scripts/export_numbers.py` from the raw results "
        "(not in git). `complete` = all examples of the benchmark.\n\n"
        "- `cells.csv`: one row per (benchmark, run). `memory` = memory held / full bf16 cache, method state "
        "included (peak for math; during generation for LongBench, whose peak is the whole prompt for every "
        "method that compresses at the end of prefill). `peak_memory` = the peak for every benchmark. "
        "`state` = method state alone (Expected Attention statistics, eviction scores).\n"
        "- `paired.csv`: paired sign tests, each run against dense and against `ours` of the same budget family.\n"
        "- `pareto_<bench>.csv`, `pareto_longbench.csv` (mean over HotpotQA / MuSiQue / NarrativeQA): quality vs memory "
        "per method family (curve = method at one bit width).\n"
        "- `kcurve_<bench>.csv`: ours at C = 20 % of tokens seen, read budget k = 2.5 / 5 / 10 %, bf16 and K4V4 "
        "(quality and ms per forward).\n")
    print(f"{len(cells)} cells, {len(pairs)} pairs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
