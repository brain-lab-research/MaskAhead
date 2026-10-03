#!/usr/bin/env python3
"""Offline eviction diagnostics from recording-only attention-map .pt files.

For each logged eviction score, measure how much subsequent block attention its
top-C set retains and its overlap with the top-C hindsight mass set. Outputs a
CSV at source/layer/KV-head granularity and standalone SVG heatmaps.
"""
from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import torch


def _mass_by_position(record):
    mass = record.get("kv_mass", record.get("mass"))
    if mass is None:
        return {}
    pos = record["positions"].to(torch.int64)
    mass = mass.float()
    if mass.ndim == 1:
        mass, pos = mass.unsqueeze(0), pos.unsqueeze(0)
    return [dict(zip(p.tolist(), m.tolist())) for p, m in zip(pos, mass)]


def _kv_mass(record):
    return (record["kv_mass"] if "kv_mass" in record else record["mass"]).float()


def _top_positions(pos, score, capacity):
    result = []
    for p, s in zip(pos, score):
        n = min(int(capacity), p.numel())
        ids = torch.topk(s.float(), n).indices
        result.append(set(int(x) for x in p[ids].tolist()))
    return result


def _heatmap_svg(path, values, title):
    rows, cols = len(values), max((len(r) for r in values), default=0)
    if not rows or not cols:
        return
    flat = [x for row in values for x in row if x is not None]
    lo, hi = min(flat, default=0.0), max(flat, default=1.0)
    width, height = max(600, cols * 26 + 100), max(420, rows * 18 + 100)
    cell_w = max(1, (width - 90) // cols)
    cell_h = max(1, (height - 70) // rows)
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="12" y="22" font-size="16">{title}</text>']
    for i, row in enumerate(values):
        body.append(f'<text x="4" y="{40+i*cell_h+cell_h*.7:.1f}" font-size="10">L{i}</text>')
        for j, value in enumerate(row):
            if value is None:
                continue
            norm = 0 if hi <= lo else (value - lo) / (hi - lo)
            red, green, blue = int(255 * norm), int(90 + 130 * (1 - norm)), int(255 * (1 - norm))
            x, y = 32 + j * cell_w, 32 + i * cell_h
            body.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" '
                        f'fill="rgb({red},{green},{blue})"/>')
    for j in range(cols):
        body.append(f'<text x="{32+j*cell_w+cell_w/2:.1f}" y="{height-8}" '
                    f'text-anchor="middle" font-size="9">H{j}</text>')
    body.append('</svg>')
    path.write_text('\n'.join(body), encoding='utf-8')


def _comparison_svg(path, actual, oracle, title, actual_label, oracle_label):
    """Small dependency-free line plot for paired coverage diagnostics."""
    steps = sorted(set(actual) | set(oracle))
    if not steps:
        return
    width, height, left, top = 720, 400, 64, 48
    plot_w, plot_h = width - left - 24, height - top - 58
    def xy(i, value):
        x = left + (plot_w * i / max(1, len(steps) - 1))
        y = top + plot_h * (1.0 - max(0.0, min(1.0, value)))
        return x, y
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="{left}" y="24" font-size="16">{title}</text>']
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + plot_h * (1.0 - tick)
        body.append(f'<path d="M{left} {y:.1f}H{width-24}" stroke="#ddd"/>')
        body.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-size="10">{tick:.2f}</text>')
    colors = ("#2166ac", "#b2182b")
    for vals, color, label, offset in ((actual, colors[0], actual_label, 0),
                                       (oracle, colors[1], oracle_label, 20)):
        pts = [xy(i, vals[s]) for i, s in enumerate(steps) if s in vals]
        if pts:
            body.append('<polyline fill="none" stroke="%s" stroke-width="3" points="%s"/>' %
                        (color, " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)))
            for x, y in pts:
                body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>')
        body.append(f'<path d="M{left+offset} {height-18}h22" stroke="{color}" stroke-width="3"/>')
        body.append(f'<text x="{left+offset+28}" y="{height-14}" font-size="11">{label}</text>')
    for i, step in enumerate(steps):
        x, _ = xy(i, 0)
        body.append(f'<text x="{x:.1f}" y="{top+plot_h+18}" text-anchor="middle" font-size="10">{step}</text>')
    body.extend([f'<text x="{width/2:.1f}" y="{height-2}" text-anchor="middle" font-size="11">decoder step</text>',
                 '</svg>'])
    path.write_text('\n'.join(body), encoding='utf-8')


def analyze_file(path, out_rows, pairwise_rows, selection_rows, *, skip_h2o=False, layers=None):
    pack = torch.load(path, map_location="cpu", weights_only=False)
    records = pack.get("records", [])
    if layers is not None:
        records = [r for r in records if "layer" not in r or int(r["layer"]) in layers]
    mass_records = [r for r in records if r.get("kind") == "mass"]
    score_records = [r for r in records if r.get("kind") == "eviction_score"]
    pairwise_records = [r for r in records if r.get("kind") == "pairwise_set"]
    by_block_layer = {}
    by_step_layer = defaultdict(list)
    for r in mass_records:
        key = (r["block"], r["layer"])
        # Eviction's block-level reference is the first masked step; later
        # records are retained separately for the step-specific selector oracle.
        if key not in by_block_layer or r.get("step", 0) < by_block_layer[key].get("step", 0):
            by_block_layer[key] = r
        by_step_layer[key].append(r)
    selected_records = {(r["block"], r["layer"]): r for r in records
                        if r.get("kind") == "selected"}

    # A single step-0 read set is compared with a hindsight top-C set for each
    # decoder step, using the same per-step attention mass and exact same budget.
    for (block, layer), chosen_rec in selected_records.items():
        chosen_pos = chosen_rec["positions"].to(torch.int64)
        for rec in by_step_layer.get((block, layer), []):
            pos = rec["positions"].to(torch.int64)
            mass = _kv_mass(rec)
            if mass.ndim == 1:
                mass = mass.unsqueeze(0)
            if mass.shape[0] != chosen_pos.shape[0]:
                mass = mass.reshape(chosen_pos.shape[0], -1, mass.shape[-1]).mean(1)
            cap = min(int(chosen_pos.shape[-1]), int(pos.shape[-1]))
            for kh in range(min(chosen_pos.shape[0], mass.shape[0])):
                loc = {int(p): float(a) for p, a in zip(pos[kh].tolist(), mass[kh].tolist())}
                total = sum(loc.values())
                keep = set(int(p) for p in chosen_pos[kh].tolist())
                actual = sum(loc.get(p, 0.0) for p in keep) / max(total, 1e-12)
                truth = set(sorted(loc, key=loc.get, reverse=True)[:cap])
                oracle = sum(loc.get(p, 0.0) for p in truth) / max(total, 1e-12)
                selection_rows.append({"recording": path.parent.name, "example": path.stem,
                                       "set_kind": "selection_shared_vs_step_oracle",
                                       "block": block, "layer": layer, "kv_head": kh,
                                       "step": rec.get("step", 0), "capacity": cap,
                                       "shared_retained_mass": actual,
                                       "oracle_retained_mass": oracle,
                                       "top_c_overlap": len(keep & truth) / max(1, cap)})
    for event in score_records:
        b, layer, source = event["block"], event["layer"], event["source"]
        epos, escore = event["positions"].to(torch.int64), event["score"].float()
        cap = int(event["capacity"])
        chosen = _top_positions(epos, escore, cap)
        # Evaluate against queries actually generated after this decision. The
        # hindsight comparator may only select keys available at decision time.
        horizons = [(i, by_block_layer[b + i, layer]) for i in range(1, 5)
                    if (b + i, layer) in by_block_layer]
        for h, rec in sorted(horizons, key=lambda x: x[0]):
            pos = rec["positions"].to(torch.int64)
            if "kv_mass" in rec:
                future = rec["kv_mass"].float()
            else:
                future = rec["mass"].float()
            if future.ndim == 1:
                future = future.unsqueeze(0)
            # If the stored query-head map is used, average groups of Q heads
            # into KV heads using the position array's KV-head count.
            if future.shape[0] != len(chosen):
                heads = len(chosen)
                future = future.reshape(heads, -1, future.shape[-1]).mean(1)
            for kh in range(min(len(chosen), future.shape[0])):
                loc = {int(p): float(a) for p, a in zip(pos[kh].tolist(), future[kh].tolist())}
                total = sum(loc.values())
                retained = sum(loc.get(p, 0.0) for p in chosen[kh])
                candidates = set(int(p) for p in epos[kh].tolist())
                truth = set(sorted(candidates, key=lambda p: loc.get(p, 0.0), reverse=True)[:cap])
                overlap = 0.0
                if truth:
                    overlap = len(truth.intersection(chosen[kh])) / len(truth)
                oracle_retained = sum(loc.get(p, 0.0) for p in truth) / max(total, 1e-12)
                out_rows.append({"recording": path.parent.name, "example": path.stem,
                                 "source": source, "block": b,
                                 "layer": layer, "kv_head": kh, "horizon": h,
                                 "retained_mass": retained / max(total, 1e-12),
                                 "oracle_retained_mass": oracle_retained,
                                 "coverage_gap": oracle_retained - retained / max(total, 1e-12),
                                 "top_c_overlap": overlap, "capacity": cap})

        # H2O is an offline comparator: cumulative real attention mass observed
        # up to this eviction point, restricted to the same candidate positions.
        if horizons and not skip_h2o:
            h2o = torch.zeros_like(escore)
            for rec in mass_records:
                if rec["block"] > b or rec["layer"] != layer:
                    continue
                pos = rec["positions"].to(torch.int64)
                mass = _kv_mass(rec)
                if mass.shape[0] != epos.shape[0]:
                    mass = mass.reshape(epos.shape[0], -1, mass.shape[-1]).mean(1)
                for kh in range(epos.shape[0]):
                    lookup = {int(p): float(a) for p, a in zip(pos[kh].tolist(), mass[kh].tolist())}
                    h2o[kh] += torch.tensor([lookup.get(int(p), 0.0) for p in epos[kh]])
            h2o_chosen = _top_positions(epos, h2o, cap)
            for h, rec in sorted(horizons, key=lambda x: x[0]):
                pos = rec["positions"].to(torch.int64)
                future = _kv_mass(rec)
                if future.ndim == 1:
                    future = future.unsqueeze(0)
                if future.shape[0] != len(h2o_chosen):
                    future = future.reshape(len(h2o_chosen), -1, future.shape[-1]).mean(1)
                for kh in range(min(len(h2o_chosen), future.shape[0])):
                    loc = {int(p): float(a) for p, a in zip(pos[kh].tolist(), future[kh].tolist())}
                    total = sum(loc.values())
                    retained = sum(loc.get(p, 0.0) for p in h2o_chosen[kh])
                    candidates = set(int(p) for p in epos[kh].tolist())
                    truth = set(sorted(candidates, key=lambda p: loc.get(p, 0.0), reverse=True)[:cap])
                    overlap = len(truth.intersection(h2o_chosen[kh])) / max(1, len(truth))
                    out_rows.append({"recording": path.parent.name, "example": path.stem,
                                     "source": "h2o_offline", "block": b,
                                     "layer": layer, "kv_head": kh, "horizon": h,
                                     "retained_mass": retained / max(total, 1e-12),
                                     "oracle_retained_mass": sum(loc.get(p, 0.0) for p in truth) / max(total, 1e-12),
                                     "coverage_gap": sum(loc.get(p, 0.0) for p in truth) / max(total, 1e-12) - retained / max(total, 1e-12),
                                     "top_c_overlap": overlap, "capacity": cap})

    for event in pairwise_records:
        full, k4 = event["full_keep"], event["k4_keep"]
        for kh in range(min(full.shape[0], k4.shape[0])):
            full_set = set(int(x) for x in full[kh].tolist())
            k4_set = set(int(x) for x in k4[kh].tolist())
            overlap = len(full_set & k4_set) / max(1, int(event["capacity"]))
            pairwise_rows.append({"recording": path.parent.name, "example": path.stem,
                                  "set_kind": event["set_kind"],
                                  "block": event["block"], "layer": event["layer"],
                                  "kv_head": kh, "capacity": event["capacity"],
                                  "overlap": overlap})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, nargs="+", help="one or more directories of recording .pt files")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--wait-count", type=int, default=0,
                    help="wait until this many per-example recordings exist before analysis")
    ap.add_argument("--wait-timeout", type=int, default=3600)
    ap.add_argument("--skip-h2o", action="store_true",
                    help="skip the historical-mass baseline; useful for focused horizon plots")
    ap.add_argument("--layers", type=str, default=None,
                    help="comma-separated representative layers to analyze, e.g. 0,4,...,28")
    ap.add_argument("--quality-note", type=str, default=None,
                    help="optional matched benchmark-score note placed below the panels")
    args = ap.parse_args()
    if args.wait_count:
        deadline = time.monotonic() + args.wait_timeout
        while any(len(list(d.glob("*.pt"))) < args.wait_count for d in args.input):
            if time.monotonic() >= deadline:
                counts = {str(d): len(list(d.glob("*.pt"))) for d in args.input}
                raise SystemExit(f"timed out waiting for {args.wait_count} recordings: {counts}")
            time.sleep(30)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    pairwise_rows = []
    selection_rows_raw = []
    layers = None if args.layers is None else {int(x) for x in args.layers.split(",") if x}
    input_files = sorted(path for d in args.input for path in d.glob("*.pt"))
    for path in input_files:
        analyze_file(path, rows, pairwise_rows, selection_rows_raw,
                     skip_h2o=args.skip_h2o, layers=layers)
    csv_path = args.out / "attention_metrics.csv"
    fields = ["recording", "example", "source", "block", "layer", "kv_head", "horizon",
              "retained_mass", "oracle_retained_mass", "coverage_gap", "top_c_overlap", "capacity"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row["source"]][(row["layer"], row["kv_head"])].append(row["retained_mass"])
    for source, pairs in groups.items():
        max_layer = max(layer for layer, _ in pairs)
        max_head = max(head for _, head in pairs)
        heat = [[None] * (max_head + 1) for _ in range(max_layer + 1)]
        for (layer, head), vals in pairs.items():
            heat[layer][head] = sum(vals) / len(vals)
        _heatmap_svg(args.out / f"retained_mass_{source}.svg", heat,
                     f"Retained future mass: {source}")
    pairwise_path = args.out / "precision_set_overlap.csv"
    pairwise_fields = ["recording", "example", "set_kind", "block", "layer", "kv_head", "capacity", "overlap",
                       "step", "shared_retained_mass", "oracle_retained_mass", "top_c_overlap"]
    with pairwise_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=pairwise_fields)
        writer.writeheader()
        writer.writerows(pairwise_rows)
    sel_example = defaultdict(lambda: {"actual": [], "oracle": []})
    for row in selection_rows_raw:
        if row["recording"] != "stepwise_m4_math30":
            continue
        values = sel_example[(row["example"], row["step"])]
        values["actual"].append(row["shared_retained_mass"])
        values["oracle"].append(row["oracle_retained_mass"])
    sel_by_step = defaultdict(lambda: {"actual": [], "oracle": [], "cells": 0})
    for (example, step), values in sel_example.items():
        sel_by_step[step]["actual"].append(sum(values["actual"]) / len(values["actual"]))
        sel_by_step[step]["oracle"].append(sum(values["oracle"]) / len(values["oracle"]))
        sel_by_step[step]["cells"] += len(values["actual"])
    selection_rows = [{"step": step,
                       "shared_retained_mass": sum(v["actual"]) / len(v["actual"]),
                       "oracle_retained_mass": sum(v["oracle"]) / len(v["oracle"]),
                       "examples": len(v["actual"]), "cells": v["cells"]}
                      for step, v in sorted(sel_by_step.items()) if v["actual"]]
    with (args.out / "selection_shared_vs_step_oracle.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["step", "shared_retained_mass", "oracle_retained_mass",
                                          "examples", "cells"])
        w.writeheader(); w.writerows(selection_rows)
    _comparison_svg(args.out / "selection_shared_vs_step_oracle.svg",
                    {r["step"]: r["shared_retained_mass"] for r in selection_rows},
                    {r["step"]: r["oracle_retained_mass"] for r in selection_rows},
                    "Block-shared selection vs step-specific oracle (matched C)",
                    "shared read set", "step-specific top-C oracle")
    # These inputs are record-only runs with a full prefix. The comparator uses
    # the same decision-time candidates and budget as the evaluated score.
    ev_example = defaultdict(lambda: {"actual": [], "oracle": []})
    for row in rows:
        key = (row["recording"], row["source"], row["horizon"], row["example"])
        ev_example[key]["actual"].append(row["retained_mass"])
        ev_example[key]["oracle"].append(row["oracle_retained_mass"])
    ev_by_horizon = defaultdict(lambda: {"actual": [], "oracle": [], "cells": 0})
    for (recording, source, horizon, example), values in ev_example.items():
        group = ev_by_horizon[(recording, source, horizon)]
        group["actual"].append(sum(values["actual"]) / len(values["actual"]))
        group["oracle"].append(sum(values["oracle"]) / len(values["oracle"]))
        group["cells"] += len(values["actual"])
    ev_rows = [{"recording": recording, "source": source, "horizon": h,
                "kept_retained_mass": sum(v["actual"]) / len(v["actual"]),
                "hindsight_top_c_mass": sum(v["oracle"]) / len(v["oracle"]),
                "examples": len(v["actual"]), "cells": v["cells"]}
               for (recording, source, h), v in sorted(ev_by_horizon.items()) if v["actual"]]
    with (args.out / "eviction_realized_vs_hindsight.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["recording", "source", "horizon",
                                          "kept_retained_mass", "hindsight_top_c_mass",
                                          "examples", "cells"])
        w.writeheader(); w.writerows(ev_rows)
    m4_rows = [r for r in ev_rows if r["recording"] == "full_m4_math30"
               and r["source"] == "lookahead"]
    _comparison_svg(args.out / "eviction_realized_vs_hindsight.svg",
                    {r["horizon"]: r["kept_retained_mass"] for r in m4_rows},
                    {r["horizon"]: r["hindsight_top_c_mass"] for r in m4_rows},
                    "MATH-30: realized future mass, full prefix",
                    "predicted keep set", "hindsight top-C at decision")
    left_path = args.out / "selection_shared_vs_step_oracle.svg"
    right_path = args.out / "eviction_realized_vs_hindsight.svg"
    if left_path.exists() and right_path.exists():
        left = left_path.read_text(encoding="utf-8")
        right = right_path.read_text(encoding="utf-8")
        left_body, right_body = left.split(">", 1)[1].rsplit("</svg>", 1)[0], right.split(">", 1)[1].rsplit("</svg>", 1)[0]
        left_body = left_body.replace('<rect width="100%" height="100%" fill="white"/>',
                                      '<rect width="720" height="400" fill="white"/>')
        right_body = right_body.replace('<rect width="100%" height="100%" fill="white"/>',
                                        '<rect width="720" height="400" fill="white"/>')
        composite = ['<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="440" viewBox="0 0 1440 440">',
                     f'<g transform="translate(0,0)">{left_body}</g>',
                     f'<g transform="translate(720,0)">{right_body}</g>']
        if args.quality_note:
            composite.append(f'<text x="24" y="430" font-size="12">{args.quality_note}</text>')
        composite.append('</svg>')
        (args.out / "decision_horizon_ablation.svg").write_text("\n".join(composite), encoding="utf-8")
    by_kind = defaultdict(list)
    by_cell = defaultdict(list)
    for row in pairwise_rows:
        by_kind[row["set_kind"]].append(row["overlap"])
        by_cell[(row["set_kind"], row["layer"], row["kv_head"])].append(row["overlap"])
    for kind, vals in by_kind.items():
        print(f"{kind}: mean BF16/K4 set overlap={sum(vals)/len(vals):.4f} over {len(vals)} layer/head/events")
    for kind in by_kind:
        max_layer = max(layer for set_kind, layer, _ in by_cell if set_kind == kind)
        max_head = max(head for set_kind, _, head in by_cell if set_kind == kind)
        heat = [[None] * (max_head + 1) for _ in range(max_layer + 1)]
        for (set_kind, layer, head), vals in by_cell.items():
            if set_kind == kind:
                heat[layer][head] = sum(vals) / len(vals)
        _heatmap_svg(args.out / f"precision_overlap_{kind}.svg", heat,
                     f"BF16/K4 {kind} set overlap")
    print(f"wrote {len(rows)} layer/head/horizon rows to {csv_path}")


if __name__ == "__main__":
    main()
