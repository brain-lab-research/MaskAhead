from __future__ import annotations

import re
import string
from difflib import SequenceMatcher
from collections import Counter
from typing import Iterable

try:
    from fuzzywuzzy import fuzz
except ImportError:  # pragma: no cover - setup installs the official dependency
    fuzz = None
try:
    from rouge import Rouge
except ImportError:  # pragma: no cover - setup installs the official dependency
    Rouge = None


_BOX_RE = re.compile(r"\\boxed\s*\{")
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def extract_last_boxed(text: str) -> str | None:
    starts = [m.start() for m in _BOX_RE.finditer(text)]
    if not starts:
        return None
    start = starts[-1]
    brace = text.find("{", start)
    depth = 0
    for pos in range(brace, len(text)):
        char = text[pos]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1 : pos]
    return None


def numeric_normalize(value: str) -> str:
    x = value.strip()
    x = x.replace("\\$", "").replace("$", "")
    x = x.replace("{,}", "").replace(",", "")
    x = x.replace("\\%", "").replace("%", "")
    x = x.strip(" {}[]()\n\t.")
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", x):
        sign = ""
        if x.startswith(("+", "-")):
            sign, x = x[0], x[1:]
        if "." in x:
            x = x.rstrip("0").rstrip(".")
        x = x.lstrip("0") or "0"
        if x.startswith("."):
            x = "0" + x
        return sign + x
    return x.lower().replace(" ", "")


def normalize_math(value: str) -> str:
    x = value.strip().lower()
    replacements = {
        "\\left": "",
        "\\right": "",
        "\\!": "",
        "\\,": "",
        "\\ ": "",
        " ": "",
        "\n": "",
        "\t": "",
    }
    for old, new in replacements.items():
        x = x.replace(old, new)
    x = x.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    x = x.replace("{,}", "").replace("\\$", "").replace("$", "")
    return x.strip(".")


def extract_math_answer(text: str) -> str:
    boxed = extract_last_boxed(text)
    if boxed is not None:
        return boxed
    cues = ["final answer", "answer is", "therefore", "thus"]
    lower = text.lower()
    tail = text
    for cue in cues:
        idx = lower.rfind(cue)
        if idx >= 0:
            tail = text[idx + len(cue) :]
            break
    nums = _NUMBER_RE.findall(tail)
    if nums:
        return nums[-1]
    nums = _NUMBER_RE.findall(text)
    return nums[-1] if nums else text.strip().splitlines()[-1] if text.strip() else ""


def math_exact_match(prediction: str, references: Iterable[str]) -> float:
    pred = extract_math_answer(prediction)
    p_math = normalize_math(pred)
    p_num = numeric_normalize(pred)
    for ref in references:
        candidates = [ref]
        boxed = extract_last_boxed(ref)
        if boxed is not None:
            candidates.append(boxed)
        for candidate in candidates:
            if p_math == normalize_math(candidate) or p_num == numeric_normalize(candidate):
                return 1.0
    return 0.0


def normalize_qa(text: str) -> str:
    # This is the exact English normalization used by THUDM/LongBench.
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def qa_f1(prediction: str, reference: str) -> float:
    p = normalize_qa(prediction).split()
    r = normalize_qa(reference).split()
    if not p or not r:
        return float(p == r)
    counts: dict[str, int] = {}
    for token in r:
        counts[token] = counts.get(token, 0) + 1
    common = 0
    for token in p:
        if counts.get(token, 0):
            common += 1
            counts[token] -= 1
    if common == 0:
        return 0.0
    precision = common / len(p)
    recall = common / len(r)
    return 2 * precision * recall / (precision + recall)


def best_qa_f1(prediction: str, references: Iterable[str]) -> float:
    return max((_official_qa_f1(prediction, x) for x in references), default=0.0)


def _official_qa_f1(prediction: str, reference: str) -> float:
    """THUDM/LongBench metrics.py::qa_f1_score, kept byte-for-byte in spirit."""
    prediction_tokens = normalize_qa(prediction).split()
    reference_tokens = normalize_qa(reference).split()
    common = Counter(prediction_tokens) & Counter(reference_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return float(scorer.score(reference, prediction)["rougeL"].fmeasure)
    except Exception:

        p = prediction.split()
        r = reference.split()
        row = [0] * (len(r) + 1)
        for a in p:
            previous = 0
            for j, b in enumerate(r, 1):
                old = row[j]
                row[j] = previous + 1 if a == b else max(row[j], row[j - 1])
                previous = old
        lcs = row[-1]
        return 0.0 if not p or not r else 2 * lcs / (len(p) + len(r))


def best_rouge_l(prediction: str, references: Iterable[str]) -> float:
    return max((_official_rouge(prediction, x) for x in references), default=0.0)


def _official_rouge(prediction: str, reference: str) -> float:
    """THUDM/LongBench metrics.py::rouge_score."""
    if Rouge is None:
        return rouge_l(prediction, reference)
    try:
        return float(Rouge().get_scores([prediction], [reference], avg=True)["rouge-l"]["f"])
    except Exception:
        return 0.0


def edit_similarity(prediction: str, reference: str) -> float:
    return SequenceMatcher(None, prediction.strip(), reference.strip()).ratio()


def best_edit_similarity(prediction: str, references: Iterable[str]) -> float:
    return max((_official_code_sim(prediction, x) for x in references), default=0.0)


def _official_code_sim(prediction: str, reference: str) -> float:
    """THUDM/LongBench metrics.py::code_sim_score."""
    for line in prediction.lstrip("\n").split("\n"):
        if "`" not in line and "#" not in line and "//" not in line:
            if fuzz is not None:
                return fuzz.ratio(line, reference) / 100.0
            return SequenceMatcher(None, line, reference).ratio()
    return 0.0


def count_score(prediction: str, reference: str) -> float:
    """THUDM/LongBench metrics.py::count_score (passage_count)."""
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    right = sum(1 for n in numbers if str(n) == str(reference))
    return right / len(numbers)


def best_count_score(prediction: str, references: Iterable[str]) -> float:
    return max((count_score(prediction, x) for x in references), default=0.0)


def retrieval_score(prediction: str, reference: str) -> float:
    """THUDM/LongBench metrics.py::retrieval_score (passage_retrieval_en). `reference`
    is the gold string itself (e.g. "Paragraph 15"), not a paraphrase - the official
    metric extracts the digits from IT too, not just from the prediction."""
    matches = re.findall(r"Paragraph (\d+)", reference)
    if not matches:
        return 0.0
    gold_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    right = sum(1 for n in numbers if str(n) == gold_id)
    return right / len(numbers)


def best_retrieval_score(prediction: str, references: Iterable[str]) -> float:
    return max((retrieval_score(prediction, x) for x in references), default=0.0)


def classification_score(prediction: str, reference: str, all_classes: list[str] | None) -> float:
    """THUDM/LongBench metrics.py::classification_score (trec). Needs the task's full
    label set (`all_classes`, carried in BenchmarkExample.metadata) - unlike every other
    LongBench metric here, it is not a function of (prediction, reference) alone."""
    if not all_classes:
        return 0.0
    em_match_list = [c for c in all_classes if c in prediction]
    for match_term in list(em_match_list):
        if match_term in reference and match_term != reference:
            em_match_list.remove(match_term)
    if reference in em_match_list:
        return 1.0 / len(em_match_list)
    return 0.0


def best_classification_score(
    prediction: str, references: Iterable[str], all_classes: list[str] | None
) -> float:
    return max((classification_score(prediction, x, all_classes) for x in references), default=0.0)


# Task -> official THUDM/LongBench metric family. English tasks plus lsht: the other
# Chinese LongBench tasks (dureader, vcsum, multifieldqa_zh, passage_retrieval_zh) use a
# jieba-segmented variant of qa_f1/rouge/retrieval that this project has no use for, but
# lsht's official metric is classification_score - substring matching against all_classes,
# with no segmentation anywhere in it - so it runs at full official fidelity.
_LONGBENCH_QA_F1 = {"narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "triviaqa"}
_LONGBENCH_ROUGE = {"gov_report", "qmsum", "multi_news", "samsum"}
_LONGBENCH_CODE_SIM = {"repobench-p", "repobench_p", "lcc"}
_LONGBENCH_CLASSIFICATION = {"trec", "lsht"}
_LONGBENCH_COUNT = {"passage_count"}
_LONGBENCH_RETRIEVAL = {"passage_retrieval_en"}
# THUDM/LongBench's own scorer() takes only the first line of the prediction for these
# four tasks before scoring (github.com/THUDM/LongBench eval.py) - they ask for a short
# direct answer, so anything after the first line is reasoning the official protocol was
# never designed to be scored on. lsht is in that list upstream too.
_LONGBENCH_FIRST_LINE_ONLY = {"trec", "triviaqa", "samsum", "lsht"}



_LBV2_PAREN = re.compile(r"The correct answer is \(([A-D])\)")
_LBV2_BARE = re.compile(r"The correct answer is ([A-D])")


def longbench_v2_choice(prediction: str) -> str | None:
    """THUDM/LongBench pred.py::extract_answer - the A-D letter, or None."""
    text = prediction.replace("*", "")
    match = _LBV2_PAREN.search(text) or _LBV2_BARE.search(text)
    return match.group(1) if match else None


def longbench_v2_accuracy(prediction: str, references: Iterable[str]) -> float:
    """1.0 when the extracted letter matches the gold letter. Discrete by construction:
    unlike the v1 F1/ROUGE families, no rewording or trimming of the answer can move it,
    so a variant cannot gain score by degrading into terser output."""
    choice = longbench_v2_choice(prediction)
    if choice is None:
        return 0.0
    return float(any(choice == str(r).strip().upper() for r in references))


def score_prediction(
    benchmark: str,
    prediction: str,
    references: list[str],
    *,
    all_classes: list[str] | None = None,
) -> float:
    # LongBench-E tasks ('<task>_e') are the same task resampled by context length -
    # same prompt, same metric.
    name = benchmark.lower()
    if name.endswith("_e"):
        name = name[:-2]
    if name in {"gsm8k", "math500", "math-500", "aime2025", "aime-2025", "aime25"}:
        return math_exact_match(prediction, references)
    if name in {"longbench_v2", "longbench-v2", "lbv2"}:
        return longbench_v2_accuracy(prediction, references)
    if name == "niah_hard":
        matches = re.findall(r"\b\d{7}\b", prediction)
        return float(bool(matches) and matches[-1] in references)
    if name == "niah":
        return float(any(normalize_qa(ref) in normalize_qa(prediction) for ref in references))

    if name in _LONGBENCH_FIRST_LINE_ONLY:
        prediction = prediction.lstrip("\n").split("\n")[0]
    if name in _LONGBENCH_ROUGE:
        return best_rouge_l(prediction, references)
    if name in _LONGBENCH_CODE_SIM:
        return best_edit_similarity(prediction, references)
    if name in _LONGBENCH_CLASSIFICATION:
        return best_classification_score(prediction, references, all_classes)
    if name in _LONGBENCH_COUNT:
        return best_count_score(prediction, references)
    if name in _LONGBENCH_RETRIEVAL:
        return best_retrieval_score(prediction, references)
    if name in _LONGBENCH_QA_F1:
        return best_qa_f1(prediction, references)
    # An unlisted task (not gsm8k/math/niah and not in the official LongBench dispatch
    # above) falls back to QA F1 rather than silently defaulting to a metric family that
    # was never validated against it - qa_f1 is the most common LongBench metric, but a
    # NEW task should be added to one of the sets above instead of relying on this.
    return best_qa_f1(prediction, references)
