"""CPU checks for the public evaluation and analysis entry points."""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maskahead.config import ExperimentConfig  # noqa: E402


class ReleaseChecks(unittest.TestCase):
    def test_configs_load(self) -> None:
        configs = sorted((ROOT / "configs").glob("*.yaml"))
        self.assertTrue(configs)
        for path in configs:
            with self.subTest(config=path.name):
                ExperimentConfig.load(path)

    def test_precision_analysis_accepts_realized_attention_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            recording = base / "recording"
            recording.mkdir()
            torch.save(
                {"records": [{
                    "kind": "pairwise_set", "set_kind": "selection", "block": 0,
                    "layer": 0, "capacity": 2,
                    "full_keep": torch.tensor([[1, 2]]),
                    "k4_keep": torch.tensor([[1, 3]]),
                }]},
                recording / "example.pt",
            )
            analyzed = base / "analyzed"
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "analyze_realized_attention.py"),
                 str(recording), "--out", str(analyzed)],
                check=True, capture_output=True, text=True,
            )
            with (analyzed / "precision_set_overlap.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["recording"], "recording")
            self.assertEqual(float(row["overlap"]), 0.5)
            summarized = base / "summary"
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "analyze_precision_pairwise.py"),
                 str(analyzed), "--out", str(summarized)],
                check=True, capture_output=True, text=True,
            )
            with (summarized / "precision_overlap_summary.csv").open(newline="") as handle:
                summary = next(csv.DictReader(handle))
            self.assertEqual(summary["n_examples"], "1")
            self.assertEqual(float(summary["mean_overlap"]), 0.5)


if __name__ == "__main__":
    unittest.main()
