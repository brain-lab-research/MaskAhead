#!/usr/bin/env python3
"""Tables from results/<model>/<benchmark>/<run>.jsonl.

For every model and benchmark: one row per run with n, score, mean generated
tokens, decode tokens/s, seconds per example, cache size relative to a full
bf16 cache, and how many generations hit the length limit or the loop stop.
Then paired sign tests of every run against --ref (default: ours) on the
examples both have. Markdown to stdout.

    python scripts/report.py [--results results] [--ref ours]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from math import comb
from pathlib import Path


def load(path: Path) -> dict:
    rows = {}
    for line in path.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            rows[str(r["id"])] = r
    return rows


def bf16_bytes(r: dict):
    """A full bf16 cache of every token this example saw: the common memory baseline."""
    m = r.get("runtime") or {}
    if m.get("bf16_equiv_bytes"):                                    # measured by the cache itself
        return m["bf16_equiv_bytes"]
    dense = m.get("dense_cache_equivalent_bytes")
    if dense is None and m.get("key_payload") is not None:          # Dream reports payloads
        q = (r.get("config") or {}).get("quant") or {}
        dense = m["key_payload"] * 16 / q.get("k_bits", 16) + m["value_payload"] * 16 / q.get("v_bits", 16)
    return dense


def state_bytes(r: dict) -> int:
    """Memory a method keeps besides the KV entries: eviction scores/positions and, for
    Expected Attention, the per-head query mean and covariance (the method's own overhead)."""
    m = r.get("runtime") or {}
    return (m.get("eviction_state_bytes") or 0) + (m.get("ea_stats_bytes") or 0)


def state_vs_bf16(r: dict):
    """Method state (see state_bytes) / full bf16 cache."""
    dense = bf16_bytes(r)
    return state_bytes(r) / dense if dense else None


def peak_vs_bf16(r: dict):
    """Largest memory the method held during the run / full bf16 cache: the measured
    physical KV cache at its peak plus the method's state (EA statistics included)."""
    m = r.get("runtime") or {}
    dense = bf16_bytes(r)
    if m.get("cache_peak_used_bytes") and dense:
        return (m["cache_peak_used_bytes"] + state_bytes(r)) / dense
    if m.get("eviction_peak_bytes") is not None and dense:
        return m["eviction_peak_bytes"] / dense
    return cache_vs_bf16(r)                                          # no eviction: final = peak


def cache_vs_bf16(r: dict):
    """Cache (incl. method state) at the end of the run / full bf16 cache."""
    m = r.get("runtime") or {}
    total = m.get("eviction_total_bytes")
    if total is not None:
        total += m.get("ea_stats_bytes") or 0
    elif m.get("bf16_equiv_bytes") and m.get("total") is not None:  # no eviction: packed cache as held
        return m["total"] / m["bf16_equiv_bytes"]
    dense = m.get("bf16_equiv_bytes") or m.get("dense_cache_equivalent_bytes")
    if dense is None and m.get("key_payload") is not None:          # Dream reports payloads
        q = (r.get("config") or {}).get("quant") or {}
        dense = m["key_payload"] * 16 / q.get("k_bits", 16) + m["value_payload"] * 16 / q.get("v_bits", 16)
        total = total if total is not None else m.get("total")
    if total is None and m.get("resident_cache_bytes") is not None and m.get("compact_cache_bytes") is not None:
        total = m["resident_cache_bytes"] - m["compact_cache_bytes"]
    return total / dense if total and dense else None


def seconds(r: dict):
    """Wall seconds per example: wall_s if logged, else prefill + chunked prefill + decode."""
    if r.get("wall_s"):
        return r["wall_s"]
    m = r.get("runtime") or {}
    ms = (m.get("prefill_ms") or 0) + (m.get("prefill_chunked_ms") or 0) + (m.get("decode_ms") or 0)
    return ms / 1e3 if ms else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


def f(x, spec):
    return "—" if x is None else format(x, spec)


def sign_p(w: int, l: int) -> float:
    n = w + l
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(Path(__file__).resolve().parents[1] / "results"))
    ap.add_argument("--ref", default="ours")
    args = ap.parse_args()
    root = Path(args.results)
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "logs"):
        for bench_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            runs = {p.stem: load(p) for p in sorted(bench_dir.glob("*.jsonl"))}
            runs = {k: v for k, v in runs.items() if v}
            if not runs:
                continue
            print(f"\n## {model_dir.name} / {bench_dir.name}\n")
            print("| run | n | score | gen tokens | tok/s | s/example | cache vs bf16 (end) | memory vs bf16 (peak, incl. state) | method state vs bf16 | hit limit | loop stop |")
            print("|---|---|---|---|---|---|---|---|---|---|---|")
            for name, rows in runs.items():
                R = list(rows.values())
                m = [r.get("runtime") or {} for r in R]
                print(f"| {name} | {len(R)} | {f(mean([r['score'] for r in R]), '.3f')} "
                      f"| {f(mean([x.get('generated_tokens') for x in m]), '.0f')} "
                      f"| {f(mean([x.get('tokens_per_second') for x in m]), '.1f')} "
                      f"| {f(mean([seconds(r) for r in R]), '.1f')} "
                      f"| {f(mean([cache_vs_bf16(r) for r in R]), '.3f')} "
                      f"| {f(mean([peak_vs_bf16(r) for r in R]), '.3f')} "
                      f"| {f(mean([state_vs_bf16(r) for r in R]), '.3f')} "
                      f"| {sum(1 for x in m if x.get('stop_found') is False and not x.get('loop_stopped'))} "
                      f"| {sum(1 for x in m if x.get('loop_stopped'))} |")
            ref = runs.get(args.ref)
            if ref:
                print(f"\npaired vs `{args.ref}` (wins/losses of the row, two-sided sign test):\n")
                for name, rows in runs.items():
                    if name == args.ref:
                        continue
                    ids = ref.keys() & rows.keys()
                    w = sum(rows[i]["score"] > ref[i]["score"] for i in ids)
                    l = sum(rows[i]["score"] < ref[i]["score"] for i in ids)
                    print(f"- {name}: n={len(ids)}  {w}/{l}  p={sign_p(w, l):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
