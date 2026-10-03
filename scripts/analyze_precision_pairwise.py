#!/usr/bin/env python3
"""Summarize paired BF16-vs-simulated-K4 selection and eviction overlaps."""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, help="analyze_realized_attention.py output directory")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    source = args.input / "precision_set_overlap.csv"
    by_example = defaultdict(list)
    by_layer = defaultdict(list)
    with source.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["recording"], row["example"], row["set_kind"])
            val = float(row["overlap"])
            by_example[key].append(val)
            by_layer[(row["set_kind"], int(row["layer"]), int(row["kv_head"]))].append(val)

    examples = defaultdict(list)
    for (_, _, kind), values in by_example.items():
        examples[kind].append(sum(values) / len(values))

    rng = random.Random(20260926)
    summary = []
    for kind in ("selection", "eviction"):
        vals = examples.get(kind, [])
        if not vals:
            continue
        means = []
        for _ in range(10000):
            means.append(sum(rng.choice(vals) for _ in vals) / len(vals))
        means.sort()
        summary.append({
            "set_kind": kind,
            "n_examples": len(vals),
            "n_layer_head_events": sum(
                len(values) for (_recording, _example, set_kind), values in by_example.items()
                if set_kind == kind
            ),
            "mean_overlap": sum(vals) / len(vals),
            "median_example_overlap": sorted(vals)[len(vals) // 2],
            "bootstrap95_low": means[249],
            "bootstrap95_high": means[9749],
        })

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "precision_overlap_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]) if summary else ["set_kind"])
        writer.writeheader(); writer.writerows(summary)
    with (args.out / "precision_overlap_by_layer_head.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["set_kind", "layer", "kv_head", "n_cells", "mean_overlap"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for (kind, layer, head), vals in sorted(by_layer.items()):
            writer.writerow({"set_kind": kind, "layer": layer, "kv_head": head,
                             "n_cells": len(vals), "mean_overlap": sum(vals) / len(vals)})

    width, height = 640, 390
    left, top, plot_h = 90, 55, 250
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<text x="320" y="27" text-anchor="middle" font-size="17">BF16 vs simulated K4 selected-set overlap</text>']
    for tick in range(0, 101, 20):
        y = top + plot_h * (1 - tick / 100)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="600" y2="{y:.1f}" stroke="#ddd"/>')
        body.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-size="11">{tick}%</text>')
    for i, row in enumerate(summary):
        x = 190 + i * 250
        value = row["mean_overlap"]
        y = top + plot_h * (1 - value)
        body.append(f'<rect x="{x}" y="{y:.1f}" width="110" height="{plot_h*value:.1f}" fill="#3977a8"/>')
        lo = top + plot_h * (1 - row["bootstrap95_low"])
        hi = top + plot_h * (1 - row["bootstrap95_high"])
        body.append(f'<line x1="{x+55}" y1="{hi:.1f}" x2="{x+55}" y2="{lo:.1f}" stroke="#111" stroke-width="2"/>')
        body.append(f'<text x="{x+55}" y="{top+plot_h+23}" text-anchor="middle" font-size="13">{row["set_kind"]}</text>')
        body.append(f'<text x="{x+55}" y="{max(45,y-9):.1f}" text-anchor="middle" font-size="12">{value:.3f} (n={row["n_examples"]})</text>')
    body.append('</svg>')
    (args.out / "precision_overlap.svg").write_text("\n".join(body), encoding="utf-8")


if __name__ == "__main__":
    main()
