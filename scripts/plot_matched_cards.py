#!/usr/bin/env python3
"""Publication figures for serving capacity and the MaskAhead score choice.

Serif typography, markers and legend framing follow plots.ipynb. The blue and
terracotta palette follows the paper figures. Refuse incomplete results so a
partial queue run cannot accidentally become a paper figure.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


COLORS = ["#2F6FA5", "#79AAC9", "#C96C45", "#A85235"]
MARKERS = ["o", "s", "^", "D"]
POLICIES = ("dense", "mage", "ours", "ours_prefill_qprobe")
POLICY_LABELS = ("Dense", "MAGE", "MaskAhead", "MaskAhead + Q-prefill")

mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.size": 8.5,
    "axes.labelsize": 9.5,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.65,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "legend.fontsize": 7.0,
    "legend.frameon": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def style_axis(ax):
    ax.grid(True, linestyle=(0, (3, 2)), linewidth=0.45,
            color="#D8E4F0", alpha=0.85)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#B9C6D3")
        spine.set_linewidth(0.7)
    ax.tick_params(axis="both", which="major", color="#8296AA", labelcolor="#243A51")


def save(fig, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(out.with_suffix(suffix), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def serving_figure(path: Path, out: Path):
    data = json.loads(path.read_text())
    lengths = (8192, 12288, 16384)
    trials = data.get("trials", {})
    boundaries = data.get("boundaries", {})
    matched = data.get("matched_batches", {})
    keys = [f"{length}/{policy}" for length in lengths for policy in POLICIES]
    missing = [key for key in keys if key not in boundaries]
    if missing:
        raise RuntimeError(f"P0 boundaries incomplete: {', '.join(missing)}")
    capped = [key for key in keys if boundaries[key].get("first_oom_batch") is None
              and boundaries[key].get("safe_batch", 0) < 1024]
    if capped:
        raise RuntimeError(f"P0 boundaries still capped below 1024: {', '.join(capped)}")
    if any(str(length) not in matched for length in lengths):
        raise RuntimeError("P0 matched batch sizes are incomplete")
    for length in lengths:
        common = int(matched[str(length)])
        for policy in POLICIES:
            for size, which in ((common, "matched"),
                                (boundaries[f"{length}/{policy}"]["safe_batch"], "maximum-safe")):
                key = f"{length}/{policy}/{size}"
                trial = trials.get(key, {})
                if trial.get("status") != "ok" or trial.get("generated_tokens") != size * 256:
                    raise RuntimeError(f"P0 {which} measurement missing: {key}")

    def metric(length, policy, size, field):
        return trials[f"{length}/{policy}/{size}"][field]

    fig, axes = plt.subplots(2, 3, figsize=(7.4, 3.65), sharex=True)
    x = [length / 1024 for length in lengths]
    specs = (
        (axes[0, 0], "Largest OOM-safe batch", "Requests",
         lambda length, policy: boundaries[f"{length}/{policy}"]["safe_batch"]),
        (axes[0, 1], "Throughput: matched batch", "Output tokens / s",
         lambda length, policy: metric(length, policy, matched[str(length)], "output_tokens_per_s_e2e")),
        (axes[0, 2], "Throughput: max safe batch", "Output tokens / s",
         lambda length, policy: metric(length, policy, boundaries[f"{length}/{policy}"]["safe_batch"],
                                       "output_tokens_per_s_e2e")),
        (axes[1, 0], "Allocated: matched batch", "Allocated GiB",
         lambda length, policy: metric(length, policy, matched[str(length)], "peak_allocated_gib")),
        (axes[1, 1], "Reserved: matched batch", "Reserved GiB",
         lambda length, policy: metric(length, policy, matched[str(length)], "peak_reserved_gib")),
        (axes[1, 2], "Peak KV: matched batch", "KV GiB",
         lambda length, policy: metric(length, policy, matched[str(length)], "peak_kv_gib")),
    )
    for ax, title, ylabel, value in specs:
        style_axis(ax)
        for i, policy in enumerate(POLICIES):
            ax.plot(x, [value(length, policy) for length in lengths],
                    color=COLORS[i], marker=MARKERS[i], linewidth=1.45,
                    markersize=4.3, markerfacecolor=COLORS[i],
                    markeredgecolor="0.15", markeredgewidth=0.45,
                    alpha=0.96, label=POLICY_LABELS[i])
        ax.set_title(title, fontweight="semibold", pad=4)
        ax.set_ylabel(ylabel, labelpad=2)
        ax.set_xticks(x)
        ax.set_xlim(x[0] - 0.55, x[-1] + 0.55)
    axes[0, 0].set_yscale("log", base=2)
    axes[0, 0].set_yticks((32, 64, 128, 256, 512), ("32", "64", "128", "256", "512"))
    for ax in axes[1]:
        ax.set_xlabel("Prompt tokens (thousands)", labelpad=2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="lower center", ncol=4,
                        bbox_to_anchor=(0.5, -0.005), frameon=True, fancybox=False,
                        handlelength=1.6, columnspacing=1.0)
    legend.get_frame().set_facecolor("#FBFCFE")
    legend.get_frame().set_edgecolor("#CBD6E1")
    legend.get_frame().set_linewidth(0.6)
    fig.tight_layout(rect=(0, 0.09, 1, 1), pad=0.52, w_pad=0.7, h_pad=0.65)
    save(fig, out)


def future_figure(directory: Path, out: Path):
    path = directory / "criterion_factorization.json"
    if not path.exists():
        raise RuntimeError(f"criterion analysis missing: {path}")
    summaries = json.loads(path.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.55))
    for ax, (dataset, title, ylabel, expected) in zip(
        axes, (("math500", "MATH-500", "Accuracy (%)", 500),
               ("hotpotqa", "HotpotQA", "Token F1", 200))
    ):
        data = summaries[dataset]
        if data["examples"] != expected or len(data["scores"]) != 4:
            raise RuntimeError(f"incomplete score factorization for {dataset}")
        style_axis(ax)
        for read, label, color, marker in (
            ("mass", "Read: mass", COLORS[0], MARKERS[0]),
            ("value", "Read: value-aware", COLORS[2], MARKERS[2]),
        ):
            values = [data["scores"][f"{read}_{retention}"]["mean"]
                      for retention in ("mass", "value")]
            ax.plot((0, 1), values, label=label, color=color, linewidth=1.6,
                    marker=marker, markersize=6, markeredgecolor="0.15",
                    markeredgewidth=0.5)
        values = [point["mean"] for point in data["scores"].values()]
        span = max(values) - min(values)
        margin = max(span * 0.24, 0.18 if dataset == "math500" else 0.06)
        ax.set_ylim(min(values) - margin, max(values) + margin)
        ax.set_xlim(-0.16, 1.16)
        ax.set_xticks((0, 1), ("Mass", "Value-aware"))
        ax.set_xlabel("Retention ranking score", labelpad=3)
        ax.set_ylabel(ylabel, labelpad=2)
        ax.set_title(title, fontweight="semibold", pad=5)
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="lower center", ncol=2,
                        bbox_to_anchor=(0.5, -0.005), frameon=True, fancybox=False)
    legend.get_frame().set_facecolor("#FBFCFE")
    legend.get_frame().set_edgecolor("#CBD6E1")
    fig.tight_layout(rect=(0, 0.11, 1, 1), pad=0.55)
    save(fig, out)
    for suffix in (".png", ".pdf"):
        shutil.copyfile(out.with_suffix(suffix), out.with_name("criterion_factorization").with_suffix(suffix))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serving", type=Path, required=True)
    ap.add_argument("--future", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", choices=("all", "serving", "future"), default="all")
    args = ap.parse_args()
    if args.only in ("all", "serving"):
        serving_figure(args.serving, args.out / "serving_capacity_varied")
        print(args.out / "serving_capacity_varied.png")
    if args.only in ("all", "future"):
        future_figure(args.future, args.out / "realized_future_use")
        print(args.out / "realized_future_use.png")


if __name__ == "__main__":
    main()
