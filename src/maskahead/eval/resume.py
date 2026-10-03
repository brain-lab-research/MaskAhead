from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Protocol, Sequence


class _ConfigLike(Protocol):
    def to_dict(self) -> dict[str, Any]: ...


def resume_candidates(output_path: Path) -> list[Path]:

    candidates = [output_path, Path(f"{output_path}.partial")]
    candidates.extend(
        sorted(
            output_path.parent.glob(f"{output_path.name}.partial.*"),
            key=lambda path: path.stat().st_mtime_ns,
        )
    )
    seen: set[Path] = set()
    result: list[Path] = []
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen or not path.is_file() or path.stat().st_size == 0:
            continue
        seen.add(resolved)
        result.append(path)
    return result


def read_valid_jsonl(path: Path) -> list[dict[str, Any]]:

    raw_lines = path.read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_lines, start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            if index == len(raw_lines):
                print(
                    f"warning: ignoring truncated final JSONL line {index} in {path}",
                    file=sys.stderr,
                    flush=True,
                )
                break
            raise ValueError(f"invalid JSONL at {path}:{index}") from None
        if not isinstance(row, dict):
            raise ValueError(f"expected a JSON object at {path}:{index}")
        rows.append(row)
    return rows


def _drop_new_defaults(expected: dict, stored: Any, config: Any) -> dict:
    """Drop fields the stored row predates, if the live config leaves them at default.

    Options added to the config after a row was written (e.g. a new eviction
    knob) are absent from that row; they must not reject it as long as the
    current run keeps them at their default, i.e. behaves like the old code.
    A field set away from its default still makes the rows incompatible.
    """
    if not isinstance(stored, dict):
        return expected
    try:
        defaults = json.loads(json.dumps(type(config)().to_dict()))
    except Exception:
        return expected
    out = {}
    for key, val in expected.items():
        if key not in stored:
            if defaults.get(key) == val:
                continue
        elif isinstance(val, dict) and isinstance(stored.get(key), dict):
            sub_def = defaults.get(key) if isinstance(defaults.get(key), dict) else {}
            val = {k: v for k, v in val.items() if k in stored[key] or sub_def.get(k) != v}
        out[key] = val
    return out


def load_resume_rows(
    output_path: Path,
    *,
    benchmark: str,
    config: _ConfigLike,
    model_id: str,
    model_revision: str | None,
    dtype: str,
    valid_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], list[Path]]:

    # Round-trip through JSON: stored rows hold lists where the live config
    # holds tuples (coverage_arms), and () != [] would reject every row.
    expected_config = json.loads(json.dumps(config.to_dict()))
    rows_by_id: dict[str, dict[str, Any]] = {}
    sources = resume_candidates(output_path)
    for source in sources:
        for row in read_valid_jsonl(source):
            row_id = str(row.get("id", ""))
            if not row_id:
                raise ValueError(f"resume row without an id in {source}")
            if row_id not in valid_ids:
                raise ValueError(
                    f"resume row id {row_id!r} from {source} is not in the current dataset split"
                )
            checks = {
                "benchmark": (row.get("benchmark"), benchmark),
                "config": (row.get("config"), _drop_new_defaults(expected_config, row.get("config"), config)),
                "model_id": (row.get("model_id"), model_id),
                "model_revision": (row.get("model_revision"), model_revision),
                "dtype": (row.get("dtype"), dtype),
            }
            mismatches = [
                key for key, (actual, expected) in checks.items() if actual != expected
            ]
            if mismatches:
                details = ", ".join(
                    f"{key}: {checks[key][0]!r} != {checks[key][1]!r}"
                    for key in mismatches
                )
                raise ValueError(f"incompatible resume row {row_id!r} in {source}: {details}")
            rows_by_id[row_id] = row
    return rows_by_id, sources


def rewrite_existing_rows(
    output_path: Path,
    examples: Sequence[Any],
    rows_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:

    ordered = [
        rows_by_id[str(example.example_id)]
        for example in examples
        if str(example.example_id) in rows_by_id
    ]
    if not ordered:
        return []
    temporary = output_path.with_name(f".{output_path.name}.resume.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
    temporary.replace(output_path)
    return ordered
