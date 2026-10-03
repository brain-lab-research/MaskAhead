#!/usr/bin/env python3
"""Paired two-factor score ablation within MaskAhead, from completed raw runs."""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
TAGS = {
    "math500": {
        "mass_mass": "ours_k128_selmass_evmass",
        "mass_value": "ours_k128_selmass",
        "value_mass": "ours_k128_evmass",
        "value_value": "ours_k128",
    },
    "hotpotqa": {
        "mass_mass": "ours_selmass_evmass",
        "mass_value": "ours_selmass",
        "value_mass": "ours_evmass",
        "value_value": "ours",
    },
}


def load_run(path: Path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"duplicate example IDs in {path}")
    if any(row.get("score") is None for row in rows):
        raise ValueError(f"missing scores in {path}")
    return by_id


def common_config(config):
    config = copy.deepcopy(config)
    config.pop("name", None)
    config["selector"].pop("value_aware", None)
    config["eviction"].pop("lookahead_score", None)
    return config


def analyze(dataset: str, repeats: int = 4000, seed: int = 2026):
    runs = {key: load_run(ROOT / "results/fastdllm" / dataset / (tag + ".jsonl"))
            for key, tag in TAGS[dataset].items()}
    ids = sorted(runs["mass_mass"])
    expected = 500 if dataset == "math500" else 200
    if len(ids) != expected or any(set(run) != set(ids) for run in runs.values()):
        raise ValueError(f"{dataset}: runs do not share the same {expected} examples")
    configs = {key: yaml.safe_load((ROOT / "results/fastdllm" / dataset /
                                   (tag + ".yaml")).read_text())
               for key, tag in TAGS[dataset].items()}
    references = [common_config(config) for config in configs.values()]
    if any(other != references[0] for other in references[1:]):
        raise ValueError(f"{dataset}: runs differ beyond read/retention ranking switches")
    for key, config in configs.items():
        selector = config["selector"]
        eviction = config["eviction"]
        if bool(selector["value_aware"]) != key.startswith("value_"):
            raise ValueError(f"{dataset}/{key}: read criterion mismatch")
        if (eviction.get("lookahead_score", "value") != "mass") != key.endswith("_value"):
            raise ValueError(f"{dataset}/{key}: retention criterion mismatch")
    scores = {key: [float(run[i]["score"]) * 100 for i in ids]
              for key, run in runs.items()}
    names = list(scores)
    comparisons = {
        "retention_gain_read_mass": ("mass_value", "mass_mass"),
        "retention_gain_read_value": ("value_value", "value_mass"),
        "read_gain_retention_mass": ("value_mass", "mass_mass"),
        "read_gain_retention_value": ("value_value", "mass_value"),
        "both_value_vs_both_mass": ("value_value", "mass_mass"),
    }
    draws = {key: [] for key in names}
    draws.update({key: [] for key in comparisons})
    rng = random.Random(seed)
    for _ in range(repeats):
        sampled = [rng.randrange(len(ids)) for _ in ids]
        means = {key: sum(scores[key][i] for i in sampled) / len(ids) for key in names}
        for key in names:
            draws[key].append(means[key])
        for key, (a, b) in comparisons.items():
            draws[key].append(means[a] - means[b])

    def estimate(key, value):
        sampled = sorted(draws[key])
        return {"mean": value,
                "ci95": [sampled[int(0.025 * repeats)], sampled[int(0.975 * repeats)]]}

    result = {"dataset": dataset, "examples": len(ids),
              "measure": "accuracy_points" if dataset == "math500" else "token_F1_points",
              "design": "MaskAhead bf16; same future-mask probe, k, C, E and examples; read and retention score vary",
              "source_tags": TAGS[dataset],
              "scores": {key: estimate(key, sum(values) / len(values))
                         for key, values in scores.items()},
              "paired_deltas": {key: estimate(key,
                 sum(scores[a][i] - scores[b][i] for i in range(len(ids))) / len(ids))
                 for key, (a, b) in comparisons.items()},
              "bootstrap": {"unit": "example", "repeats": repeats, "seed": seed}}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data = {dataset: analyze(dataset) for dataset in TAGS}
    args.out.write_text(json.dumps(data, indent=2) + "\n")
    for dataset, summary in data.items():
        print(dataset, "n=", summary["examples"],
              "both-value gain=", summary["paired_deltas"]["both_value_vs_both_mass"])


if __name__ == "__main__":
    main()
