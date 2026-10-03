from __future__ import annotations

import json
import os
import random
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(slots=True)
class BenchmarkExample:
    example_id: str
    prompt: str
    references: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


LONG_BENCH_CONFIGS = {
    "hotpotqa": "hotpotqa",
    "narrativeqa": "narrativeqa",
    "qasper": "qasper",
    "multifieldqa_en": "multifieldqa_en",
    "qmsum": "qmsum",
    "gov_report": "gov_report",
    "multi_news": "multi_news",
    "trec": "trec",
    "samsum": "samsum",
    "passage_count": "passage_count",
    "passage_retrieval_en": "passage_retrieval_en",
    "repobench-p": "repobench-p",
    "repobench_p": "repobench-p",
    "triviaqa": "triviaqa",
    "lcc": "lcc",
    "2wikimqa": "2wikimqa",
    "musique": "musique",
    # The one Chinese task wired in. The others (dureader, vcsum, multifieldqa_zh,
    # passage_retrieval_zh) are scored with jieba-segmented variants of qa_f1/rouge/
    # retrieval that this project has no dependency on, but lsht's official metric is
    # classification_score, which is plain substring matching against all_classes and
    # needs no segmentation at all - so it can be run at full official fidelity.
    #
    # It is here because it is the Chinese twin of trec, and trec is the only task whose
    # metric separates the selectors: on trec the single-middle-query selector answers
    # with the right meaning in the wrong vocabulary ("Golf course" for "Other location"),
    # because the label set is defined by few-shot demonstrations spread through the
    # prompt and one query cannot know to retain them. lsht has the same shape (24 closed
    # classes, few-shot, exact-match scoring), so it is an independent test of that
    # explanation rather than a second draw from the same task.
    "lsht": "lsht",
}
# Still not wired in: dureader, vcsum, multifieldqa_zh, passage_retrieval_zh - see above.

# THUDM/LongBench's pred.py skips the chat template on exactly these tasks:
#   if dataset not in ["trec","triviaqa","samsum","lsht","lcc","repobench-p"]:
#       prompt = build_chat(...)   # "chat models are better off without build prompts here"
# They are few-shot or raw-completion formats: the prompt ends mid-pattern ("...\nType:")
# and the model is meant to continue it. Wrapping that in a chat turn makes the model
# answer conversationally instead - it re-states the question and puts the answer on a
# second line - and since LongBench's scorer keeps only the FIRST line for
# trec/triviaqa/samsum, the answer is then thrown away and the task scores near zero.
# (Measured before this was honoured: trec scored 0.22 against a 0.52-0.79 published
# band, with predictions like 'Question: ...\nType: Other location' whose correct answer
# sat on the discarded second line. lcc/repobench-p were already exempted by hand in
# run_suite.py; this moves that rule to one place and extends it to the three tasks it
# was missing.)
NO_CHAT_TEMPLATE_TASKS = frozenset(
    {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p", "repobench_p"}
)


def uses_chat_template(benchmark: str) -> bool:
    """Whether this benchmark's prompt should be wrapped in the model's chat template."""
    return base_task(benchmark) not in NO_CHAT_TEMPLATE_TASKS


# LongBench-E: the same tasks, resampled so 0-4k / 4-8k / 8k+ contexts are evenly
# represented, and scored PER BUCKET rather than pooled. For a KV-cache selector this is
# the more informative cut - the claim is that dropping cache entries costs more as the
# prefix grows, and a single pooled average cannot show that. Same prompts, same metrics,
# same archive; only the data file and the reporting differ.
LONGBENCH_E_TASKS = (
    "2wikimqa", "gov_report", "hotpotqa", "lcc", "multi_news", "multifieldqa_en",
    "passage_count", "passage_retrieval_en", "qasper", "repobench-p", "samsum",
    "trec", "triviaqa",
)
LONGBENCH_E_BUCKETS = ((0, 4000, "0-4k"), (4000, 8000, "4-8k"), (8000, 10**9, "8k+"))

# _e shares its base task's prompt, metric, budget and chat-template rule; only the
# data file differs, so base_task() maps back for every one of those lookups.
for _task in LONGBENCH_E_TASKS:
    LONG_BENCH_CONFIGS[f"{_task}_e"] = f"{_task}_e"


def base_task(benchmark: str) -> str:
    """LongBench-E task name -> the base task whose prompt/metric/budget it reuses."""
    name = benchmark.lower()
    return name[:-2] if name.endswith("_e") else name


def length_bucket(length: int | None) -> str | None:
    """LongBench-E context bucket for a raw `length` field, or None if unknown."""
    if length is None:
        return None
    for lo, hi, label in LONGBENCH_E_BUCKETS:
        if lo <= length < hi:
            return label
    return None


def _take(dataset: Iterable, limit: int | None):
    for idx, row in enumerate(dataset):
        if limit is not None and idx >= limit:
            break
        yield idx, row


def _load_hf(
    name: str,
    config: str | None,
    split: str,
    *,
    revision: str | None = None,
):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install the project dependencies to use benchmark datasets") from exc


    kwargs = {"split": split}
    if revision:
        kwargs["revision"] = revision
    if config is None:
        return load_dataset(name, **kwargs)
    return load_dataset(name, config, **kwargs)


def load_gsm8k(limit: int | None = None, split: str = "test") -> list[BenchmarkExample]:
    ds = _load_hf(
        "openai/gsm8k",
        "main",
        split,
        revision=os.environ.get("GSM8K_REVISION"),
    )
    out = []
    for idx, row in _take(ds, limit):
        answer = str(row["answer"])
        ref = answer.split("####")[-1].strip()
        prompt = (
            "Solve the following grade-school mathematics problem. Explain the reasoning clearly, "
            "then put only the final answer inside \\boxed{...}.\n\n"
            f"Problem: {row['question']}\n\nSolution:"
        )
        out.append(BenchmarkExample(str(idx), prompt, [ref], {"raw_answer": answer}))
    return out


def competition_math_prompt(problem: str) -> str:
    """The MATH-500 prompt. Shared so training cannot drift from evaluation."""
    return (
        "Solve the following competition mathematics problem rigorously. Show the key steps and "
        "put the final answer inside \\boxed{...}.\n\n"
        f"Problem: {problem}\n\nSolution:"
    )


def load_math500(
    limit: int | None = None,
    split: str = "test",
    *,
    min_level: int | None = None,
) -> list[BenchmarkExample]:
    """MATH-500, optionally restricted to problems at or above ``min_level``.

    The full set averages short, easy problems together with hard ones, and the
    mean hides both: on DreamReasoner levels 1-3 score 0.60-0.92 with a 450-580
    token context, while level 5 scores 0.53 with a median of 1760. A method
    that only bites on long contexts is therefore measured mostly on problems
    where it cannot bite at all. `math500_l5` is that subset, 134 problems.
    """
    ds = _load_hf("HuggingFaceH4/MATH-500", None, split)
    if min_level is not None:
        ds = [r for r in ds if int(r.get("level") or 0) >= min_level]
    out = []
    for idx, row in _take(ds, limit):
        problem = row.get("problem") or row.get("question")
        answer = row.get("answer") or row.get("solution")
        prompt = competition_math_prompt(problem)
        out.append(
            BenchmarkExample(
                str(row.get("unique_id", idx)),
                prompt,
                [str(answer)],
                {k: row[k] for k in ("subject", "level") if k in row},
            )
        )
    return out


def load_aime2025(limit: int | None = None) -> list[BenchmarkExample]:
    """AIME 2025, parts I and II (30 problems), on the MATH-500 prompt.

    Integer answers, so the MATH exact-match grader applies unchanged.
    """
    out = []
    for part in ("AIME2025-I", "AIME2025-II"):
        ds = _load_hf("opencompass/AIME2025", part, "test")
        for idx, row in _take(ds, None):
            out.append(
                BenchmarkExample(
                    f"{part}/{idx}",
                    competition_math_prompt(row["question"]),
                    [str(row["answer"]).strip()],
                    {"part": part},
                )
            )
    return out[:limit] if limit is not None else out


# Verbatim from THUDM/LongBench/LongBench/config/dataset2prompt.json (English tasks only -
# see the comment by LONG_BENCH_CONFIGS for why the Chinese tasks are excluded). Kept as
# raw {context}/{input} templates rather than an f-string per task, so a diff against the
# upstream JSON stays trivial.
_LONGBENCH_OFFICIAL_PROMPTS: dict[str, str] = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, and a question. "
        "Answer the question asconcisely as you can, using a single phrase if possible. Do not "
        "provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the "
        "story asconcisely as you can, using a single phrase if possible. Do not provide any "
        "explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question as concisely as "
        "you can, using a single phrase or sentence if possible. If the question cannot be "
        "answered based on the information in the article, write \"unanswerable\". If the "
        "question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not "
        "provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the "
        "above article as concisely as you can, using a single phrase or sentence if possible. "
        "If the question cannot be answered based on the information in the article, write "
        "\"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or "
        "\"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following "
        "question based on the above text, only give me the answer and do not output any other "
        "words.\n\nQuestion: {input}\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer and do not "
        "output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the "
        "question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer and do not "
        "output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the "
        "question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the answer and do not "
        "output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the "
        "question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page summary of the "
        "report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question or instruction. "
        "Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the "
        "query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\n"
        "Answer:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all news. \n\n"
        "News:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:"
    ),
    # dataset2prompt.json, verbatim.
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "trec": (
        "Please determine the type of the question below. Here are some examples of "
        "questions.\n\n{context}\n{input}"
    ),
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the answer and do not "
        "output any other words. The following are some examples.\n\n{context}\n\n{input}"
    ),
    "samsum": (
        "Summarize the dialogue into a few short sentences. The following are some "
        "examples.\n\n{context}\n\n{input}"
    ),
    "passage_count": (
        "There are some paragraphs below sourced from Wikipedia. Some of them may be "
        "duplicates. Please carefully read these paragraphs and determine how many unique "
        "paragraphs there are after removing duplicates. In other words, how many "
        "non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final "
        "count of unique paragraphs after removing duplicates. The output format should only "
        "contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: "
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine "
        "which paragraph the abstract is from.\n\n{context}\n\nThe following is an "
        "abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is "
        "from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe "
        "answer is: "
    ),
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}


def _longbench_suffix(task: str, question: str) -> str | None:
    template = _LONGBENCH_OFFICIAL_PROMPTS.get(base_task(task))
    if template is None or "{context}" not in template:
        return None
    return template.split("{context}", 1)[1].replace("{input}", question)


def _longbench_prompt(task: str, context: str, question: str) -> str:
    template = _LONGBENCH_OFFICIAL_PROMPTS.get(base_task(task))
    if template is not None:
        return template.format(context=context, input=question)
    # Only reachable for a task added to LONG_BENCH_CONFIGS without an official template
    # above - not a normal code path, so fail loudly rather than silently scoring a task
    # against a prompt nobody chose.
    raise KeyError(
        f"no official LongBench prompt template for {task!r} - add one to "
        "_LONGBENCH_OFFICIAL_PROMPTS before wiring it into LONG_BENCH_CONFIGS"
    )


# LongBench v2 (2024): 503 multiple-choice questions over 8k-2M word contexts. The
# metric is accuracy on a single A-D letter, which is discrete by construction - it
# cannot be moved by an answer getting a word shorter, unlike the v1 F1/ROUGE families.
# Prompt is prompts/0shot.txt from THUDM/LongBench, verbatim.
LONGBENCH_V2_PROMPT = (
    "Please read the following text and answer the question below.\n"
    "\n"
    "<text>\n"
    "{context}\n"
    "</text>\n"
    "\n"
    "What is the correct answer to this question: {question}\n"
    "Choices:\n"
    "(A) {choice_A}\n"
    "(B) {choice_B}\n"
    "(C) {choice_C}\n"
    "(D) {choice_D}\n"
    "\n"
    'Format your response as follows: "The correct answer is (insert answer here)".'
)


def load_longbench_v2(
    limit: int | None = None,
    split: str = "train",
    *,
    length: str | None = None,
) -> list[BenchmarkExample]:
    """LongBench v2, optionally restricted to one of its length tiers.

    `length="short"` is worth considering for a 32k-context model: the tiers are
    short/medium/long by ORIGINAL context, whose median is ~417k characters, so medium
    and long are truncated to a fraction of themselves before the model sees them. The
    variant comparison stays valid either way (every variant gets the identical
    truncated prompt), but only the short tier measures the task as its authors meant it.
    """
    ds = _load_hf(
        "THUDM/LongBench-v2",
        None,
        split,
        revision=os.environ.get("LONGBENCH_V2_REVISION"),
    )
    out: list[BenchmarkExample] = []
    for idx, row in enumerate(ds):
        if length is not None and row.get("length") != length:
            continue
        if limit is not None and len(out) >= limit:
            break
        prompt = LONGBENCH_V2_PROMPT.format(
            context=row["context"],
            question=row["question"],
            choice_A=row["choice_A"],
            choice_B=row["choice_B"],
            choice_C=row["choice_C"],
            choice_D=row["choice_D"],
        )
        out.append(
            BenchmarkExample(
                str(row.get("_id", idx)),
                prompt,
                [str(row["answer"]).strip().upper()],
                {
                    "task": "longbench_v2",
                    "length_tier": row.get("length"),
                    "difficulty": row.get("difficulty"),
                    "domain": row.get("domain"),
                    "context_chars": len(row["context"]),
                },
            )
        )
    return out


def _longbench_archive_path() -> Path:
    override = os.environ.get("LONG_BENCH_ARCHIVE_PATH")
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"LONG_BENCH_ARCHIVE_PATH does not exist or is not a file: {path}"
            )
        return path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "LongBench loading requires huggingface-hub (normally installed with transformers)"
        ) from exc

    repo_id = os.environ.get("LONG_BENCH_REPO_ID", "zai-org/LongBench")
    revision = os.environ.get("LONG_BENCH_REVISION") or None
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename="data.zip",
            revision=revision,
        )
    )


def _interleave_length_buckets(rows: list[dict]) -> list[dict]:
    """Reorder LongBench-E rows so any prefix is balanced across length buckets.

    The _e files are laid out bucket-major - 100 rows of 0-4k, then 100 of 4-8k,
    then 100 of 8k+ - so taking the first N rows, as every other LongBench task
    does, yields nothing but short contexts and the length stratification that is
    the whole point of LongBench-E is silently lost. (Measured: the first 24 rows
    of hotpotqa_e and trec_e are 24/24 in the 0-4k bucket.)

    Round-robin interleaving fixes that while keeping two properties the rest of
    the pipeline relies on: the order is deterministic, and any prefix is a prefix
    of every longer one, so `--longbench-n` can be raised on an existing run
    without invalidating the rows already computed.

    Rows whose bucket cannot be determined keep their relative order and go last,
    rather than being dropped.
    """
    buckets: dict[str, list[dict]] = {name: [] for _, _, name in LONGBENCH_E_BUCKETS}
    unknown: list[dict] = []
    for row in rows:
        length = row.get("length")
        name = length_bucket(length) if isinstance(length, int) else None
        (buckets[name] if name in buckets else unknown).append(row)

    ordered: list[dict] = []
    ordered_buckets = [buckets[name] for _, _, name in LONGBENCH_E_BUCKETS]
    for index in range(max((len(b) for b in ordered_buckets), default=0)):
        for bucket in ordered_buckets:
            if index < len(bucket):
                ordered.append(bucket[index])
    return ordered + unknown


def load_longbench(
    task: str,
    limit: int | None = None,
    split: str = "test",
) -> list[BenchmarkExample]:
    if split != "test":
        raise ValueError(f"LongBench only provides a test split, got split={split!r}")

    config = LONG_BENCH_CONFIGS[task]
    archive = _longbench_archive_path()
    member = f"data/{config}.jsonl"

    out: list[BenchmarkExample] = []
    with zipfile.ZipFile(archive) as zf:
        try:
            raw = zf.open(member)
        except KeyError as exc:
            available = sorted(
                name for name in zf.namelist() if name.startswith("data/") and name.endswith(".jsonl")
            )
            raise RuntimeError(
                f"LongBench archive {archive} does not contain {member}; "
                f"available data files: {available[:8]}{'...' if len(available) > 8 else ''}"
            ) from exc

        with raw:
            rows = [json.loads(line.decode("utf-8")) for line in raw]

    if task.endswith("_e"):
        rows = _interleave_length_buckets(rows)

    for idx, row in enumerate(rows):
        if limit is not None and idx >= limit:
            break
        refs = row.get("answers", row.get("answer", []))
        if isinstance(refs, str):
            refs = [refs]
        prompt = _longbench_prompt(
            config, str(row.get("context", "")), str(row.get("input", ""))
        )
        out.append(
            BenchmarkExample(
                str(row.get("_id", idx)),
                prompt,
                [str(x) for x in refs],
                {
                    "task": config,
                    "length": row.get("length"),
                    "all_classes": row.get("all_classes"),
                    # the prompt after the context: instruction + question
                    "question_suffix": _longbench_suffix(config, str(row.get("input", ""))),
                },
            )
        )
    return out


def build_niah(
    tokenizer,
    *,
    context_lengths: list[int],
    depths: list[float],
    seed: int = 1234,
) -> list[BenchmarkExample]:
    rng = random.Random(seed)
    filler_sentence = (
        "In an old technical notebook, researchers discussed ordinary experiments, schedules, "
        "weather, books, and unrelated observations. "
    )
    filler_tokens = tokenizer.encode(filler_sentence, add_special_tokens=False)
    out: list[BenchmarkExample] = []
    for length in context_lengths:
        for depth in depths:
            key = str(rng.randint(1_000_000, 9_999_999))
            needle = (
                f" Important fact: the pass key is {key}. Remember that {key} is the pass key. "
            )
            needle_ids = tokenizer.encode(needle, add_special_tokens=False)
            target_filler = max(1, length - len(needle_ids) - 96)
            repeats = (target_filler + len(filler_tokens) - 1) // len(filler_tokens)
            body = (filler_tokens * repeats)[:target_filler]
            insert = min(len(body), max(0, int(round(depth * len(body)))))
            context_ids = body[:insert] + needle_ids + body[insert:]
            context = tokenizer.decode(context_ids, skip_special_tokens=True)
            prompt = (
                "There is one important pass key hidden in the text. Find it and answer with the "
                "seven-digit key only.\n\n"
                f"Text:\n{context}\n\nWhat is the pass key?\nAnswer:"
            )
            out.append(
                BenchmarkExample(
                    f"n{length}-d{depth:.2f}",
                    prompt,
                    [key],
                    {"context_length": length, "depth": depth},
                )
            )
    return out


def build_niah_hard(
    tokenizer,
    *,
    context_lengths: list[int],
    depths: list[float],
    seed: int = 1234,
) -> list[BenchmarkExample]:
    """Multi-fact NIAH: answer needs joining a case->unit and unit->code fact.

    Every context contains eight similarly formatted case/unit facts and eight
    unit/code facts. One pair answers the question; all other units and codes
    are decoys. The two target facts are placed at mirrored depths so a single
    local needle match is insufficient.
    """
    rng = random.Random(seed)
    filler_sentence = (
        "In an old technical notebook, researchers discussed ordinary experiments, "
        "schedules, weather, books, and unrelated observations. "
    )
    filler_tokens = tokenizer.encode(filler_sentence, add_special_tokens=False)
    out: list[BenchmarkExample] = []
    for length in context_lengths:
        for depth in depths:
            target = rng.randrange(8)
            cases = [f"CASE-{rng.randrange(10000, 99999)}" for _ in range(8)]
            units = [f"UNIT-{rng.randrange(10000, 99999)}" for _ in range(8)]
            codes = [str(rng.randrange(1_000_000, 9_999_999)) for _ in range(8)]
            facts: list[tuple[float, list[int]]] = []
            for i, (case, unit) in enumerate(zip(cases, units)):
                frac = depth if i == target else (i + 0.5) / 8
                text = f"Audit case {case} is assigned to operations unit {unit}. "
                facts.append((frac, tokenizer.encode(text, add_special_tokens=False)))
            for i, (unit, code) in enumerate(zip(units, codes)):
                frac = 1.0 - depth if i == target else (i + 0.5) / 8
                text = f"The sealed access code for operations unit {unit} is {code}. "
                facts.append((frac, tokenizer.encode(text, add_special_tokens=False)))

            suffix = (
                f"\n\nWhat is the sealed access code for audit case {cases[target]}? "
                "Answer with the seven-digit code only."
            )
            suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
            fact_tokens = sum(len(ids) for _, ids in facts)
            target_filler = max(1, length - fact_tokens - len(suffix_ids) - 8)
            repeats = (target_filler + len(filler_tokens) - 1) // len(filler_tokens)
            body = (filler_tokens * repeats)[:target_filler]
            inserts = sorted((int(round(frac * len(body))), ids) for frac, ids in facts)
            context_ids: list[int] = []
            cursor = 0
            for at, ids in inserts:
                at = min(len(body), max(cursor, at))
                context_ids.extend(body[cursor:at])
                context_ids.extend(ids)
                cursor = at
            context_ids.extend(body[cursor:])
            context = tokenizer.decode(context_ids, skip_special_tokens=True)
            prompt = (
                "A long operations log contains case assignments and sealed access codes. "
                "Use the case assignment to identify its unit, then find that unit's code.\n\n"
                f"Log:\n{context}{suffix}"
            )
            out.append(BenchmarkExample(
                f"nh{length}-d{depth:.2f}", prompt, [codes[target]],
                {"context_length": length, "depth": depth, "num_fact_pairs": 8},
            ))
    return out


# Per-benchmark output budgets, in one place so the multi-GPU runner and a
# direct `python -m maskahead.eval.quality` invocation cannot disagree.
# Math needs room for a full chain of thought: a truncated CoT never emits its
# \boxed{...}, and the grader then falls back to "last number in the text",
# which scores by accident rather than by reasoning.
MATH_BENCHMARKS = frozenset({
    "gsm8k", "math500", "math-500",
    "math500_l5", "math500-l5", "math500_l45", "math500-l45",
})
DEFAULT_MAX_NEW_TOKENS = 512
MAX_NEW_TOKENS = {
    "gsm8k": 2048,
    "math500": 2048,
    # The level-restricted subsets are the *hardest* problems, so they need at
    # least the full budget -- falling through to the 512-token default would
    # truncate exactly the long reasoning they exist to measure.
    "math500_l5": 2048,
    "math500-l5": 2048,
    "math500_l45": 2048,
    "math500-l45": 2048,
    "math-500": 2048,
    "niah": 64,
    "niah_hard": 64,
    "longbench_v2": 128,
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64,
}


def max_new_tokens_for(benchmark: str | None) -> int:
    """Output budget for a benchmark; 128 for the synthetic performance points."""
    if not benchmark:
        return 128
    return MAX_NEW_TOKENS.get(base_task(benchmark), DEFAULT_MAX_NEW_TOKENS)


def load_benchmark(
    name: str,
    *,
    tokenizer=None,
    limit: int | None = None,
    split: str = "test",
    niah_contexts: list[int] | None = None,
    niah_depths: list[float] | None = None,
    seed: int = 1234,
) -> list[BenchmarkExample]:
    key = name.lower()
    if key == "gsm8k":
        return load_gsm8k(limit, split)
    if key in {"math500", "math-500"}:
        return load_math500(limit, split)
    if key in {"aime2025", "aime-2025", "aime25"}:
        return load_aime2025(limit)
    if key in {"math500_l5", "math500-l5"}:
        return load_math500(limit, split, min_level=5)
    if key in {"math500_l45", "math500-l45"}:
        return load_math500(limit, split, min_level=4)
    if key in LONG_BENCH_CONFIGS:
        return load_longbench(key, limit, split)
    if key in {"longbench_v2", "longbench-v2", "lbv2"}:
        return load_longbench_v2(
            limit, split="train", length=os.environ.get("LONGBENCH_V2_LENGTH") or None
        )
    if key in {"niah", "niah_hard"}:
        if tokenizer is None:
            raise ValueError("tokenizer is required for NIAH")
        builder = build_niah if key == "niah" else build_niah_hard
        examples = builder(
            tokenizer,
            context_lengths=niah_contexts or [8192, 16384, 28672],
            depths=niah_depths or [0.0, 0.25, 0.5, 0.75, 1.0],
            seed=seed,
        )
        return examples[:limit] if limit is not None else examples
    raise ValueError(f"unknown benchmark: {name}")
