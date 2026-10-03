from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import torch


@dataclass(slots=True)
class TimingRecord:
    name: str
    milliseconds: float
    layer: int | None = None
    block: int | None = None
    step: int | None = None
    stream: str = "main"


@dataclass(slots=True)
class SelectionRecord:
    block: int
    layer: int
    old_cache_len: int
    selected_k: int
    selector_queries: list[int]
    semantic: str
    index_checksum: int
    mean_score: float | None = None


@dataclass(slots=True)
class CoverageRecord:
    """How well one selection reproduced the true fp16 attention it approximates.

    The reference I* is always the top-k under fp16 keys ranked by ALL masked
    block queries, whatever query subset or key precision produced `selected`.
    `mass` is the share of reference attention mass the selection captured,
    relative to what I* itself captures (so 1.0 means "as good as the best
    possible selection at this budget", not "all the mass in the prefix").
    `overlap` is |selected n I*| / k.

    `mass` alone cannot explain a quality drop, and read quickly it actively
    misleads: on hotpotqa at a fixed 128-entry budget the all-query selector
    scores mass 1.0000 while losing 0.09 of the task score, because it picks
    the best available 128 entries and the mass it needed was never in reach of
    128 entries. `mass_abs` and `ceiling_abs` split those two losses apart -
    both are shares of the FULL prefix attention mass, so:

        ceiling_abs      - what the budget allows at best (budget-limited loss)
        mass_abs         - what this selector actually kept
        mass_abs/ceiling - == `mass`, the part the selector is responsible for
    """

    block: int
    layer: int
    old_cache_len: int
    selected_k: int
    selector_queries: int
    mass: list[float]
    overlap: list[float]
    mass_abs: list[float] = field(default_factory=list)
    ceiling_abs: list[float] = field(default_factory=list)


@dataclass(slots=True)
class CoverageArmRecord:
    """One sweep arm scored on the SAME queries and keys as the run's selector.

    The arms exist to compare precision against budget at equal memory: an arm at
    b bits and k entries costs b*k, so 4-bit at 4x the entries costs exactly what
    fp16 at 1x does. `mass_abs` - the share of the full prefix attention mass the
    arm's selection retains - is the comparable figure across arms, since the
    relative `mass` normalises by each arm's own budget and so hides the very
    trade being measured.
    """

    arm: str
    bits: int
    block: int
    layer: int
    old_cache_len: int
    selected_k: int
    mass: list[float]
    mass_abs: list[float]
    ceiling_abs: list[float]
    overlap: list[float]


@dataclass(slots=True)
class RunTrace:
    timings: list[TimingRecord] = field(default_factory=list)
    selections: list[SelectionRecord] = field(default_factory=list)
    coverage: list[CoverageRecord] = field(default_factory=list)
    coverage_arms: list[CoverageArmRecord] = field(default_factory=list)
    counters: dict[str, int | float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def add_counter(self, name: str, value: int | float) -> None:
        self.counters[name] = self.counters.get(name, 0) + value

    def coverage_summary(self) -> dict[str, float | int] | None:
        """Means over genuinely sparse selections only - dense-bypassed layers
        are excluded rather than scored as a free 1.0."""
        if not self.coverage:
            return None
        mass = [v for rec in self.coverage for v in rec.mass]
        overlap = [v for rec in self.coverage for v in rec.overlap]
        if not mass:
            return None
        summary: dict[str, float | int] = {
            "cells": len(mass),
            "mass_mean": sum(mass) / len(mass),
            "mass_min": min(mass),
            "overlap_mean": sum(overlap) / len(overlap),
            "overlap_min": min(overlap),
        }
        # Absent from traces written before these were recorded, so keep the
        # summary readable rather than reporting zeros as if measured.
        mass_abs = [v for rec in self.coverage for v in rec.mass_abs]
        ceiling_abs = [v for rec in self.coverage for v in rec.ceiling_abs]
        if mass_abs:
            summary["mass_abs_mean"] = sum(mass_abs) / len(mass_abs)
            summary["mass_abs_min"] = min(mass_abs)
        if ceiling_abs:
            summary["ceiling_abs_mean"] = sum(ceiling_abs) / len(ceiling_abs)
        return summary

    def coverage_arms_summary(self) -> dict[str, dict[str, float | int]] | None:
        """Per-arm means. `mass_abs` is the figure to compare across arms.

        The relative `mass` normalises by each arm's own budget, so an arm that
        keeps four times as many entries can look identical to one that keeps a
        quarter as many - which is exactly the trade the sweep exists to measure.
        """
        if not self.coverage_arms:
            return None
        out: dict[str, dict[str, float | int]] = {}
        by_arm: dict[str, list[CoverageArmRecord]] = {}
        for rec in self.coverage_arms:
            by_arm.setdefault(rec.arm, []).append(rec)
        for arm, recs in by_arm.items():
            mass = [v for r in recs for v in r.mass]
            mass_abs = [v for r in recs for v in r.mass_abs]
            ceiling = [v for r in recs for v in r.ceiling_abs]
            overlap = [v for r in recs for v in r.overlap]
            if not mass_abs:
                continue
            out[arm] = {
                "bits": recs[0].bits,
                "cells": len(mass_abs),
                "mean_selected_k": sum(r.selected_k for r in recs) / len(recs),
                "mass_abs_mean": sum(mass_abs) / len(mass_abs),
                "mass_abs_min": min(mass_abs),
                "ceiling_abs_mean": sum(ceiling) / len(ceiling),
                "mass_mean": sum(mass) / len(mass),
                "overlap_mean": sum(overlap) / len(overlap),
            }
        return out

    def to_dict(self) -> dict:
        return {
            "timings": [asdict(x) for x in self.timings],
            "selections": [asdict(x) for x in self.selections],
            "coverage": [asdict(x) for x in self.coverage],
            "coverage_arms": [asdict(x) for x in self.coverage_arms],
            "coverage_arms_summary": self.coverage_arms_summary(),
            "coverage_summary": self.coverage_summary(),
            "counters": dict(self.counters),
            "notes": list(self.notes),
        }

    def dump(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


@dataclass(slots=True)
class _PendingCudaTiming:
    name: str
    start: torch.cuda.Event
    end: torch.cuda.Event
    layer: int | None
    block: int | None
    step: int | None
    stream: str


class RuntimeTimer:

    def __init__(self, trace: RunTrace, enabled: bool = False) -> None:
        self.trace = trace
        self.enabled = enabled
        self._pending: list[_PendingCudaTiming] = []

    @contextmanager
    def region(
        self,
        name: str,
        *,
        tensor: torch.Tensor | None = None,
        layer: int | None = None,
        block: int | None = None,
        step: int | None = None,
        stream_name: str = "main",
        stream: torch.cuda.Stream | None = None,
    ) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        use_cuda = tensor is not None and tensor.device.type == "cuda"
        if use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            active = stream or torch.cuda.current_stream(tensor.device)
            start.record(active)
            try:
                yield
            finally:
                end.record(active)
                self._pending.append(
                    _PendingCudaTiming(
                        name, start, end, layer, block, step, stream_name
                    )
                )
        else:
            t0 = time.perf_counter()
            try:
                yield
            finally:
                self.trace.timings.append(
                    TimingRecord(
                        name=name,
                        milliseconds=(time.perf_counter() - t0) * 1000.0,
                        layer=layer,
                        block=block,
                        step=step,
                        stream=stream_name,
                    )
                )

    def finalize(self) -> None:
        if not self._pending:
            return
        torch.cuda.synchronize()
        for item in self._pending:
            self.trace.timings.append(
                TimingRecord(
                    name=item.name,
                    milliseconds=float(item.start.elapsed_time(item.end)),
                    layer=item.layer,
                    block=item.block,
                    step=item.step,
                    stream=item.stream,
                )
            )
        self._pending.clear()
