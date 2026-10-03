#!/usr/bin/env python3
"""Exploratory paired future-KV-use diagnostic on full-cache trajectories.

Recorded M=4/DapQ scores are value-aware; offline historical/oracle sets are
mass-ranked. The result does not isolate a ranking criterion or query source
and must not be used as a publication figure. See the analysis README.
All scores are evaluated on matched checkpoints, candidates, and capacity.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import torch


METHODS = ("m4", "h2o", "dapq", "oracle")


def records(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)["records"]


def index(rows, kind):
    return {(int(r["block"]), int(r["layer"])): r for r in rows if r.get("kind") == kind}


def masses(rows):
    out = {}
    for r in rows:
        if r.get("kind") != "mass":
            continue
        key = (int(r["block"]), int(r["layer"]))
        if key not in out or int(r.get("step", 0)) < int(out[key].get("step", 0)):
            out[key] = r
    return out


def aligned(source, target, *, check_mass=False):
    if not torch.equal(source["positions"], target["positions"]):
        return False
    if check_mass:
        a = source.get("kv_mass", source.get("mass"))
        b = target.get("kv_mass", target.get("mass"))
        return a.shape == b.shape and torch.allclose(a.float(), b.float(), atol=0.002, rtol=0.01)
    return True


def future_mass(base, block, layer, candidates):
    """Sum the next four generated blocks on the exact decision-time candidates."""
    total = torch.zeros(candidates.shape, dtype=torch.float32)
    all_prefix = torch.zeros(candidates.shape[0], dtype=torch.float32)
    observed = 0
    for offset in range(1, 5):
        rec = base.get((block + offset, layer))
        if rec is None:
            continue
        positions = rec["positions"].to(torch.int64)
        mass = rec.get("kv_mass", rec.get("mass")).float()
        if mass.shape[0] != candidates.shape[0]:
            mass = mass.reshape(candidates.shape[0], -1, mass.shape[-1]).mean(dim=1)
        all_prefix += mass.sum(dim=-1)
        for head in range(candidates.shape[0]):
            lookup = dict(zip(positions[head].tolist(), mass[head].tolist()))
            total[head] += torch.tensor([lookup.get(p, 0.0) for p in candidates[head].tolist()])
        observed += 1
    return total, all_prefix, observed


def historical_mass(base, block, layer, candidates):
    total = torch.zeros(candidates.shape, dtype=torch.float32)
    for (b, l), rec in base.items():
        if l != layer or b > block:
            continue
        positions = rec["positions"].to(torch.int64)
        mass = rec.get("kv_mass", rec.get("mass")).float()
        if mass.shape[0] != candidates.shape[0]:
            mass = mass.reshape(candidates.shape[0], -1, mass.shape[-1]).mean(dim=1)
        for head in range(candidates.shape[0]):
            lookup = dict(zip(positions[head].tolist(), mass[head].tolist()))
            total[head] += torch.tensor([lookup.get(p, 0.0) for p in candidates[head].tolist()])
    return total


def bootstrap(rows, seed=2026, repeats=2000):
    by_example = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_example[r["example"]][r["method"]].append(r["retained_fraction"])
    examples = sorted(x for x, v in by_example.items() if all(v.get(m) for m in METHODS))
    per_example = {x: {m: sum(by_example[x][m]) / len(by_example[x][m]) for m in METHODS}
                   for x in examples}
    rng = random.Random(seed)
    summary = {"examples": len(examples), "methods": {}}
    for method in METHODS:
        vals = [per_example[x][method] for x in examples]
        draws = []
        for _ in range(repeats):
            draws.append(sum(vals[rng.randrange(len(vals))] for _ in vals) / len(vals))
        draws.sort()
        summary["methods"][method] = {"mean": sum(vals) / len(vals),
                                      "ci95": [draws[int(.025 * repeats)], draws[int(.975 * repeats)]]}
    for method in ("h2o", "dapq"):
        vals = [per_example[x]["m4"] - per_example[x][method] for x in examples]
        draws = []
        for _ in range(repeats):
            draws.append(sum(vals[rng.randrange(len(vals))] for _ in vals) / len(vals))
        draws.sort()
        summary[f"m4_minus_{method}"] = {"mean": sum(vals) / len(vals),
                                          "ci95": [draws[int(.025 * repeats)], draws[int(.975 * repeats)]]}
    return summary


def analyze(root: Path, dataset: str, suffix: str = "", supplement_suffix: str = ""):
    directories = {method: root / f"full_{method}_{dataset}{suffix}" for method in ("m4", "dapq")}
    ea = root / f"full_ea_{dataset}{suffix}"
    if ea.is_dir():
        directories["ea"] = ea
    names = sorted(set.intersection(*(set(p.name for p in directory.glob("*.pt"))
                                      for directory in directories.values())))
    rows = []
    skipped = defaultdict(int)
    for name in names:
        supplement = {method: root / f"full_{method}_{dataset}{supplement_suffix}" / name
                      for method in ("m4", "dapq")}
        use_supplement = bool(supplement_suffix) and all(path.exists() for path in supplement.values())
        if use_supplement:
            packs = {method: records(path) for method, path in supplement.items()}
        else:
            packs = {method: records(directory / name) for method, directory in directories.items()}
        base = masses(packs["m4"])
        others = {method: masses(rows_) for method, rows_ in packs.items() if method != "m4"}
        scores = {method: index(packs[method], "eviction_score") for method in ("m4", "dapq")}
        for key, ev in scores["m4"].items():
            block, layer = key
            if key not in scores["dapq"] or key not in base:
                skipped["missing_checkpoint"] += 1
                continue
            candidates = ev["positions"].to(torch.int64)
            cap = int(ev["capacity"])
            dapq_event = scores["dapq"][key]
            if (cap <= 0 or cap > candidates.shape[-1]
                    or int(dapq_event["capacity"]) != cap
                    or int(dapq_event["seen"]) != int(ev["seen"])
                    or not aligned(ev, dapq_event)):
                skipped["candidate_mismatch"] += 1
                continue
            # Recording-only mode must give the same generated trajectory under
            # every scoring policy. Otherwise use no cross-method comparison.
            comparable = all(
                (block + offset, layer) in base
                and all((block + offset, layer) in other
                        and aligned(base[(block + offset, layer)],
                                    other[(block + offset, layer)], check_mass=True)
                        for other in others.values())
                for offset in range(5)
            )
            if not comparable:
                skipped["trajectory_mismatch"] += 1
                continue
            truth, all_prefix, observed = future_mass(base, block, layer, candidates)
            if observed != 4:
                skipped["missing_future_block"] += 1
                continue
            historical = historical_mass(base, block, layer, candidates)
            method_scores = {"m4": ev["score"].float(), "h2o": historical,
                             "dapq": scores["dapq"][key]["score"].float(), "oracle": truth}
            for head in range(candidates.shape[0]):
                denom = float(all_prefix[head])
                if denom <= 0:
                    skipped["zero_mass"] += 1
                    continue
                for method in METHODS:
                    chosen = torch.topk(method_scores[method][head], cap).indices
                    rows.append({"example": name, "block": block, "layer": layer,
                                 "kv_head": head, "capacity": cap, "method": method,
                                 "retained_fraction": float(truth[head, chosen].sum()) / denom})
    return rows, dict(skipped), len(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--suffix", default="", help="suffix of recording directories, such as _nostop")
    ap.add_argument("--supplement-suffix", default="", help="use matching supplemental M=4/DapQ files when present")
    ap.add_argument("--math-supplement-suffix", default="", help="override supplement suffix for MATH-500")
    ap.add_argument("--hotpot-supplement-suffix", default="", help="override supplement suffix for HotpotQA")
    ap.add_argument("--datasets", nargs="+", default=["math30", "hotpot30"])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        supplement = (args.math_supplement_suffix if dataset == "math30"
                      else args.hotpot_supplement_suffix)
        rows, skipped, n_files = analyze(args.root, dataset, args.suffix,
                                        supplement or args.supplement_suffix)
        if not rows:
            raise RuntimeError(f"{dataset}: no comparable checkpoint; skipped={skipped}")
        with (args.out / f"{dataset}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = bootstrap(rows)
        summary.update({"files": n_files, "checkpoints": len(rows) // (len(METHODS) * 4),
                        "skipped": skipped, "metric": "retained mass / all realized prefix mass"})
        (args.out / f"{dataset}.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(dataset, json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
