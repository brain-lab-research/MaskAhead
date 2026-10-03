from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from tqdm import tqdm

from ..config import ExperimentConfig
from ..runtime.generator import BitSieveGenerator
from ..runtime.official_generator import OfficialDenseGenerator
from .benchmarks import load_benchmark, max_new_tokens_for, uses_chat_template
from .common import encode_prompt, load_fast_dllm, parse_dtype, question_start
from .metrics import score_prediction
from .resume import load_resume_rows, rewrite_existing_rows


def _parse_ints(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def _parse_floats(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Run quality benchmarks with the official dense or BitSieve engine.')
    p.add_argument("--config", required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--adapter", help="LoRA adapter directory to merge before evaluating")
    p.add_argument("--revision")
    p.add_argument("--attn-implementation")
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int)
    p.add_argument("--example-id-file", help="run only benchmark IDs listed one per line")
    p.add_argument(
        "--example-offset", type=int, default=0,
        help="skip this many examples from the front of the split. LongBench has "
             "no train split, so recovery training takes its examples from the "
             "front; evaluating a task that was trained on needs an offset at "
             "least as large or the score is measured on the training set.",
    )
    p.add_argument("--max-new-tokens", type=int)
    p.add_argument("--max-cache-tokens", type=int)
    p.add_argument(
        "--topk", type=int,
        help="override the selector budget with a fixed number of prefix tokens; "
             "clears topk_percent. Lets one config be swept across budgets.",
    )
    p.add_argument(
        "--topk-percent", type=float,
        help="override the selector budget with a percent of the live prefix; "
             "clears topk.",
    )
    p.add_argument("--k-bits", type=int, choices=(2, 4, 16),
                   help="override the persistent key width")
    p.add_argument("--v-bits", type=int, choices=(2, 4, 16),
                   help="override the persistent value width")
    p.add_argument("--eviction-policy", choices=("none", "recent", "ema_recent", "ema_recent_value", "ema_recent_score"),
                   help="what the cache keeps, as opposed to what a block reads")
    p.add_argument("--eviction-capacity-floor", type=int)
    p.add_argument("--eviction-capacity-percent", type=float)
    p.add_argument("--eviction-window", type=int, help="W, the protected tail")
    p.add_argument("--eviction-decay", type=float, help="lambda in the EMA")
    p.add_argument("--eviction-interval", type=int, help="blocks between evictions")
    p.add_argument(
        "--coverage", action="store_true",
        help="score the selection against an fp16 all-masked-query reference "
             "(reports attention mass retained; slower)",
    )
    p.add_argument("--output", required=True)
    p.add_argument("--plain-prompt", action="store_true")
    p.add_argument("--niah-contexts", default="8192,16384,28672")
    p.add_argument("--niah-depths", default="0,0.25,0.5,0.75,1")
    p.add_argument("--save-traces", action="store_true")
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "resume from OUTPUT or OUTPUT.partial when compatible rows exist "
            "(default: enabled; use --no-resume for a clean overwrite)"
        ),
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = ExperimentConfig.load(args.config)
    # The per-benchmark budget wins over the config default unless the caller
    # names one explicitly, so a config written for 512-token summarization
    # cannot silently truncate a 2048-token math chain of thought.
    resolved_max_new = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else max_new_tokens_for(args.benchmark)
    )
    if args.topk is not None and args.topk_percent is not None:
        raise SystemExit("--topk and --topk-percent are mutually exclusive")
    budget_override = args.topk is not None or args.topk_percent is not None
    evict_override = {
        "policy": args.eviction_policy,
        "capacity_floor": args.eviction_capacity_floor,
        "capacity_percent": args.eviction_capacity_percent,
        "recent_window": args.eviction_window,
        "decay": args.eviction_decay,
        "interval_blocks": args.eviction_interval,
    }
    evict_override = {k: v for k, v in evict_override.items() if v is not None}
    bits_override = args.k_bits is not None or args.v_bits is not None
    wants_coverage = args.coverage and cfg.semantic != "dense"
    if (
        resolved_max_new != cfg.generation.max_new_tokens
        or args.max_cache_tokens is not None
        or budget_override
        or evict_override
        or bits_override
        or wants_coverage != cfg.coverage_diagnostics
    ):
        raw = cfg.to_dict()
        raw["generation"]["max_new_tokens"] = resolved_max_new
        if args.max_cache_tokens is not None:
            raw["max_cache_tokens"] = args.max_cache_tokens
        if args.topk is not None:
            # A fixed budget and a percent budget are alternatives, not a pair:
            # leaving the other one set would silently win in effective_topk.
            raw["selector"]["topk"] = args.topk
            raw["selector"]["topk_percent"] = None
            raw["name"] = f"{raw['name']}_k{args.topk}"
        elif args.topk_percent is not None:
            raw["selector"]["topk_percent"] = args.topk_percent
            raw["name"] = f"{raw['name']}_p{args.topk_percent:g}".replace(".", "p")
        if args.k_bits is not None:
            raw["quant"]["k_bits"] = args.k_bits
        if args.v_bits is not None:
            raw["quant"]["v_bits"] = args.v_bits
        if bits_override:
            raw["name"] = f"{raw['name']}_k{raw['quant']['k_bits']}v{raw['quant']['v_bits']}"
        if evict_override:
            raw.setdefault("eviction", {}).update(evict_override)
            raw["name"] = f"{raw['name']}_ev-{raw['eviction'].get('policy', 'none')}"
        raw["coverage_diagnostics"] = wants_coverage
        cfg = ExperimentConfig.from_dict(raw)
    model, tokenizer = load_fast_dllm(
        args.model,
        dtype=parse_dtype(args.dtype),
        device=args.device,
        revision=args.revision,
        attn_implementation=args.attn_implementation,
        adapter=args.adapter,
    )
    examples = load_benchmark(
        args.benchmark,
        tokenizer=tokenizer,
        limit=(args.limit + args.example_offset) if args.limit is not None else None,
        split=args.split,
        niah_contexts=_parse_ints(args.niah_contexts),
        niah_depths=_parse_floats(args.niah_depths),
        seed=cfg.seed,
    )
    if args.example_offset:
        if len(examples) <= args.example_offset:
            raise SystemExit(
                f"--example-offset {args.example_offset} skips all "
                f"{len(examples)} examples of {args.benchmark}"
            )
        examples = examples[args.example_offset :]
        print(
            f"skipping the first {args.example_offset} examples; "
            f"scoring {len(examples)}"
        )
    if args.example_id_file:
        wanted = {line.strip() for line in Path(args.example_id_file).read_text().splitlines()
                  if line.strip()}
        examples = [example for example in examples if str(example.example_id) in wanted]
        if not examples:
            raise SystemExit(f"--example-id-file {args.example_id_file} selected no examples")
    generator = (
        BitSieveGenerator(model, tokenizer, cfg)
        if cfg.engine == "bitsieve"
        else OfficialDenseGenerator(model, tokenizer, cfg)
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace_dir = output_path.parent / f"{output_path.stem}_traces"

    example_ids = [str(example.example_id) for example in examples]
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("benchmark contains duplicate example ids; safe resume is impossible")

    resume_sources: list[Path] = []
    if args.resume:
        rows_by_id, resume_sources = load_resume_rows(
            output_path,
            benchmark=args.benchmark,
            config=cfg,
            model_id=args.model,
            model_revision=args.revision,
            dtype=args.dtype,
            valid_ids=set(example_ids),
        )
        rows = rewrite_existing_rows(output_path, examples, rows_by_id)
    else:
        rows = []

    completed_ids = {str(row["id"]) for row in rows}
    total = sum(float(row["score"]) for row in rows)
    if rows:
        print(
            f"resuming {args.benchmark}:{cfg.name} from {len(rows)}/{len(examples)} examples "
            f"using {', '.join(str(path) for path in resume_sources)}",
            file=sys.stderr,
            flush=True,
        )

    mode = "a" if rows else "w"
    with output_path.open(mode, encoding="utf-8") as handle, tqdm(
        total=len(examples),
        initial=len(rows),
        desc=f"{args.benchmark}:{cfg.name}",
    ) as progress:
        for example in examples:
            if str(example.example_id) in completed_ids:
                continue
            max_input = cfg.max_cache_tokens - cfg.generation.max_new_tokens
            if max_input <= 0:
                raise ValueError("max_cache_tokens must exceed max_new_tokens")
            input_ids = encode_prompt(
                tokenizer,
                example.prompt,
                max_input_tokens=max_input,
                # --plain-prompt forces it off; otherwise LongBench's own rule
                # decides (see NO_CHAT_TEMPLATE_TASKS).
                use_chat_template=(
                    not args.plain_prompt and uses_chat_template(args.benchmark)
                ),
                device=next(model.parameters()).device,
            )
            from ..runtime import session as _session_mod

            _session_mod.CURRENT_EXAMPLE = str(example.example_id)
            qs = question_start(tokenizer, input_ids, (example.metadata or {}).get("question_suffix"))
            result = generator.generate(input_ids, question_start=qs)
            prediction = result.texts[0]
            metrics = result.metrics
            trace = result.trace
            score = score_prediction(
                args.benchmark, prediction, example.references,
                all_classes=example.metadata.get("all_classes"),
            )
            total += score
            row = {
                "id": example.example_id,
                "benchmark": args.benchmark,
                "config": cfg.to_dict(),
                "model_id": args.model,
                "model_revision": args.revision,
                "dtype": args.dtype,
                "prompt_tokens": int(input_ids.shape[1]),
                "question_tokens": None if qs is None else int(input_ids.shape[1]) - qs,
                "prediction": prediction,
                "references": example.references,
                "score": score,
                "metadata": example.metadata,
                "runtime": metrics,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            rows.append(row)
            completed_ids.add(str(example.example_id))
            progress.update(1)
            if args.save_traces and trace is not None:
                trace.dump(trace_dir / f"{example.example_id}.json")

    summary = {
        "benchmark": args.benchmark,
        "config": cfg.to_dict(),
        "model_id": args.model,
        "model_revision": args.revision,
        "dtype": args.dtype,
        "num_examples": len(rows),
        "mean_score": total / len(rows) if rows else None,
        "mean_decode_ms": (
            sum(float(x["runtime"].get("decode_ms", 0.0)) for x in rows) / len(rows)
            if rows
            else None
        ),
        "mean_tokens_per_second": (
            sum(
                float(x["runtime"].get("tokens_per_second") or 0.0)
                for x in rows
            )
            / len(rows)
            if rows
            else None
        ),
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for source in resume_sources:
        if source != output_path:
            source.unlink(missing_ok=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
