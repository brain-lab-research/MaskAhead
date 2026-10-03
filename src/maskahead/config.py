from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

# One definition, in the module that implements the policy. A second copy here
# would drift the moment either side gained a field.
from .eviction import EvictionConfig  # noqa: F401  (re-exported for configs)

Semantic = Literal["dense", "A", "B"]
SelectorMode = Literal["all", "middle", "uniform"]
SelectorDomain = Literal["prefix", "full"]
SelectorScore = Literal["softmax", "raw"]
Backend = Literal["auto", "triton", "torch"]
KernelVariant = Literal["blocked", "legacy"]
SelectorLogitsDtype = Literal["float16", "float32"]
Engine = Literal["bitsieve", "official"]
CompactFormat = Literal["bf16", "requantized"]
UnmaskSchedule = Literal["threshold", "fixed"]


def _validate_bits(name: str, bits: int) -> None:
    if bits not in (2, 4, 16):
        raise ValueError(f"{name} must be 2, 4, or 16; got {bits}")


@dataclass(slots=True)
class QuantizationConfig:

    k_bits: int = 4
    v_bits: int = 4
    key_token_group: int = 32
    value_channel_group: int = 32
    residual_tokens: int = 32
    param_dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    # Key quantization: "channel" = per channel over key_token_group tokens (KIVI-style,
    # the default); "token" = ZipCache-style channel-separable: keys divided by a fixed
    # per-(layer, head, channel) c_i = sqrt(max_t |K_t,i|) from the first append (the
    # prefill), then quantized per token over channel groups like values -- so evicting
    # tokens never touches another token's codes.
    key_mode: str = "channel"
    # Key bias removal before quantization (packed keys only). "mean": the per-channel
    # mean of the keys (per layer and KV head, from the first write, i.e. the prompt) is
    # subtracted before storing, and the same vector from the current block's keys at
    # attention time; every logit of a query moves by the same q.mu, so softmax,
    # masses and outputs are unchanged while the quantizer no longer spends its range
    # on the constant part (the k_proj bias). "none": store keys as they are.
    key_bias: str = "none"
    # QuaRot-style rotation of the stored K and V: "hadamard" multiplies keys and values by
    # an orthonormal Hadamard matrix H before quantization (after key_bias removal); the
    # block's q, k, v are rotated the same way and the attention output rotated back, so
    # q.k, masses and ||v - c|| are unchanged and only the quantization error differs.
    rotation: str = "none"

    def validate(self, *, head_dim: int | None = None) -> None:
        _validate_bits("k_bits", self.k_bits)
        _validate_bits("v_bits", self.v_bits)
        if self.rotation not in ("none", "hadamard"):
            raise ValueError(f"unknown rotation: {self.rotation}")
        if self.rotation != "none" and head_dim is not None and head_dim & (head_dim - 1):
            raise ValueError("hadamard rotation needs a power-of-two head_dim")
        if self.key_bias not in ("none", "mean"):
            raise ValueError(f"unknown key_bias: {self.key_bias}")
        if self.key_mode not in ("channel", "token"):
            raise ValueError(f"unknown key_mode: {self.key_mode}")
        if self.key_token_group <= 0 or self.key_token_group % 8 != 0:
            raise ValueError("key_token_group must be a positive multiple of 8")
        if self.value_channel_group <= 0 or self.value_channel_group % 8 != 0:
            raise ValueError("value_channel_group must be a positive multiple of 8")
        if self.residual_tokens < 0:
            raise ValueError("residual_tokens must be non-negative")
        if head_dim is not None and head_dim % self.value_channel_group:
            raise ValueError(
                f"head_dim={head_dim} must be divisible by value_channel_group="
                f"{self.value_channel_group}"
            )
        if self.k_bits < 16 and self.key_token_group % (8 // self.k_bits):
            raise ValueError("key_token_group is incompatible with packed key bit width")
        if self.v_bits < 16 and self.value_channel_group % (8 // self.v_bits):
            raise ValueError("value_channel_group is incompatible with packed value bit width")


@dataclass(slots=True)
class SelectorConfig:

    mode: SelectorMode = "uniform"
    uniform_queries: int = 5
    topk: int = 512
    topk_percent: float | None = None
    # Lower bound on a percent budget: k = max(topk_floor, topk_percent% of the
    # live prefix). A bare 5% of a ~100-token math prompt is 5 entries and hides
    # the question; the floor keeps short prefixes readable while long ones
    # still scale. Ignored without topk_percent.
    topk_floor: int | None = None
    # What topk_percent is a percent of: "live" = the entries the cache holds now
    # (default), "seen" = every token the cache has seen, as the storage budget C
    # is defined -- then k and C are on one scale (k = 5% with C = 10% reads half
    # of what is stored). Always capped at the live set.
    topk_basis: str = "live"
    domain: SelectorDomain = "prefix"
    score: SelectorScore = "softmax"
    dense_prefix_layers: int = 2
    sort_indices: bool = True
    # Rank candidates by importance * ||v - v_head_mean|| instead of importance
    # alone: an entry earns a slot only if it is both attended to and says
    # something the head's average value does not already say.
    value_aware: bool = False
    # Centre of the spread. "mean": uniform mean of the head's values. "attn":
    # the importance-weighted mean, i.e. the head's attention output averaged
    # over the step-0 queries -- dropping entry i moves the output by
    # p_i * (v_i - o), so o is the centre the output error is measured from.
    # "none": no centre, the spread is ||v|| -- the entry's own contribution
    # p_i * v_i to the head output (Expected-Attention-style, observed p).
    value_center: str = "mean"
    # CAOTE's exact eviction error: scale by 1 / (1 - p_i), the renormalization
    # the surviving entries get once i is gone. Without it the score is CAOTE
    # minus that factor, which only differs for entries with large p_i.
    value_caote: bool = False
    # Per-query CAOTE: s_i = mean_q p_iq * ||v_i - o_q||, each masked query q
    # (and each query head of the GQA group) with its own output o_q, instead
    # of one output of the averaged head. Specific to block dLLMs, where step 0
    # has many queries; with a single AR query the two coincide.
    value_per_query: bool = False
    # Idea A: CAOTE in the joint softmax the block actually executes. Query q
    # gives mass m_q to the prefix and 1 - m_q to the block, its full output is
    # y_q = m_q o_q + (1 - m_q) o_blk_q, and s_i = mean_q m_q p_iq ||v_i - y_q||.
    value_joint: bool = False
    # Idea B: choose the set, not a per-entry score. Seed with the top k/2 by
    # mass, then add the rest greedily in value_set_rounds rounds, each time the
    # entries that most reduce the exact output error of the kept set,
    # ||sum_{i in S} p_i (v_i - o)|| / A. "pooled": one averaged query (alpha,
    # o). "per_query": the sum of that error squared over every query.
    value_set: str = "none"
    value_set_rounds: int = 4

    def validate(self, *, block_size: int | None = None, num_layers: int | None = None) -> None:
        if self.mode not in ("all", "middle", "uniform"):
            raise ValueError(f"unknown selector mode: {self.mode}")
        if self.value_center not in ("mean", "attn", "none"):
            raise ValueError(f"unknown value_center: {self.value_center}")
        if self.topk_basis not in ("live", "seen"):
            raise ValueError(f"unknown selector topk_basis: {self.topk_basis}")
        if self.value_per_query and not (self.value_aware and self.value_center == "attn"):
            raise ValueError("value_per_query needs value_aware with value_center='attn'")
        if self.value_set not in ("none", "pooled", "per_query"):
            raise ValueError(f"unknown value_set: {self.value_set}")
        if (self.value_joint or self.value_set != "none") and not self.value_aware:
            raise ValueError("value_joint / value_set need value_aware")
        if self.value_set_rounds <= 0:
            raise ValueError("value_set_rounds must be positive")
        if self.uniform_queries <= 0:
            raise ValueError("uniform_queries must be positive")
        if block_size is not None and self.uniform_queries > block_size:
            raise ValueError("uniform_queries cannot exceed block_size")
        if self.topk <= 0:
            raise ValueError("topk must be positive")
        if self.topk_floor is not None and self.topk_floor <= 0:
            raise ValueError("topk_floor must be positive")
        if self.topk_percent is not None and not (0.0 < self.topk_percent <= 100.0):
            raise ValueError("topk_percent must be in (0, 100]")
        if self.dense_prefix_layers < 0:
            raise ValueError("dense_prefix_layers must be non-negative")
        if num_layers is not None and self.dense_prefix_layers > num_layers:
            raise ValueError("dense_prefix_layers exceeds the number of layers")

    def effective_topk(self, old_cache_len: int) -> int:
        if old_cache_len <= 0:
            return 0
        if self.topk_percent is not None:
            return min(
                old_cache_len,
                max(
                    self.topk_floor or 1,
                    int(math.ceil(old_cache_len * self.topk_percent / 100.0)),
                ),
            )
        return min(old_cache_len, self.topk)

    def query_indices(self, masked_positions: list[int], block_size: int) -> list[int]:
        positions = sorted(set(int(x) for x in masked_positions if 0 <= int(x) < block_size))
        if not positions:
            return []
        if self.mode == "all":
            return positions
        if self.mode == "middle":
            center = (block_size - 1) / 2.0
            return [min(positions, key=lambda x: (abs(x - center), x))]

        n = min(self.uniform_queries, len(positions))
        if n == 1:
            return [positions[len(positions) // 2]]


        chosen = {
            positions[int(round(i * (len(positions) - 1) / (n - 1)))] for i in range(n)
        }
        return sorted(chosen)


@dataclass(slots=True)
class GenerationConfig:
    block_size: int = 32
    small_block_size: int = 8
    threshold: float = 0.95
    max_new_tokens: int = 512
    mask_token_id: int = 151665
    stop_token_id: int | None = 151645
    top_p: float = 0.95
    temperature: float = 0.0
    use_block_cache: bool = True
    schedule: UnmaskSchedule = "threshold"
    fixed_steps_per_block: int = 20
    # Stop a generation that ends in loop_stop_reps exact copies of a unit of
    # 20..1000 characters. Checked after every block. On 6455 past MATH-500
    # generations it caught 57% of the runaways (77% of their length saved) and
    # never fired on a generation with a correct answer.
    loop_stop: bool = False
    loop_stop_reps: int = 4

    def validate(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.small_block_size <= 0 or self.block_size % self.small_block_size:
            raise ValueError("small_block_size must be a positive divisor of block_size")
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError("threshold must be in [0, 1]")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.schedule == "fixed":
            if self.fixed_steps_per_block <= 0:
                raise ValueError("fixed_steps_per_block must be positive")
            if self.small_block_size != self.block_size:
                raise ValueError(
                    "fixed schedule requires small_block_size == block_size for controlled TPOB"
                )


@dataclass(slots=True)
class ExperimentConfig:

    name: str = "proposed_a_uniform5_k4v4"
    semantic: Semantic = "A"
    quant: QuantizationConfig = field(default_factory=QuantizationConfig)
    selector: SelectorConfig = field(default_factory=SelectorConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    eviction: EvictionConfig = field(default_factory=EvictionConfig)
    engine: Engine = "bitsieve"
    backend: Backend = "auto"
    async_selector: bool = True
    max_cache_tokens: int = 32768
    compact_dtype: Literal["float16", "bfloat16"] = "bfloat16"
    compact_format: CompactFormat = "bf16"
    seed: int = 1234
    collect_diagnostics: bool = False
    # Score each selection against the TRUE fp16 attention it was meant to
    # approximate. Keeps a shadow fp16 key cache and recomputes the reference
    # ranking from ALL masked block queries, independent of the query subset or
    # key precision the config under test actually used - so a starved or
    # quantized selector cannot grade its own homework. Diagnostic only:
    # costs memory and time, so it is off by default and never on in
    # performance or memory runs.
    coverage_diagnostics: bool = False
    # Use a BF16 shadow of the prefill keys to rank cache entries while the
    # actual cache remains quantized. Diagnostic ablation for K4V4 only.
    rank_bf16_shadow: bool = False
    # Compare selector/evict keep sets under BF16 and simulated K4 keys on the
    # same full-precision query trajectory. Diagnostic only.
    precision_pairwise_diagnostics: bool = False
    # Query rows per chunk when recomputing the fp16 reference (caps the
    # transient [rows x prefix] logit tensor on long prefixes).
    coverage_query_chunk: int = 32
    # Extra selector settings to score on the SAME queries and keys as the run's
    # own selector, for a precision-vs-budget sweep. Each entry is
    # {"name": str, "bits": int, "topk": int} or {..., "topk_percent": float}.
    #
    # They are evaluated from the fp16 key shadow, so one generation pass yields
    # every arm on identical inputs - which is the only way the arms are
    # comparable, since a per-arm generation would diverge into different
    # queries after the first block and the coverages would no longer be of the
    # same thing. Nothing here affects what the model generates.
    coverage_arms: tuple[dict[str, Any], ...] = ()
    profile_layers: bool = False
    # Record the cache's [tokens seen, used bytes, allocated bytes] after every change
    # (runtime.memory_trace): the resident-cache curve over a generation.
    memory_trace: bool = False
    # Attention maps for analysis (off by default; one file per example in this
    # directory): at step 0 of every block, for these layers, the prefix-softmax mass
    # of each query head over the live entries (with their token positions), the
    # entries each KV head selected, and what every eviction kept. Costs one extra
    # dequantize + q.k per logged layer and block: use on a few examples only.
    attn_log_dir: str | None = None
    attn_log_layers: list[int] | None = None       # None = layers 2, 8, 14, 20, 27
    dense_kernel_variant: KernelVariant = "blocked"
    selector_kernel_variant: KernelVariant = "blocked"
    selector_logits_dtype: SelectorLogitsDtype = "float16"
    validate_attention_masks: bool = False

    def validate(
        self,
        *,
        head_dim: int | None = None,
        num_layers: int | None = None,
    ) -> None:
        if self.semantic not in ("dense", "A", "B"):
            raise ValueError(f"unknown semantic: {self.semantic}")
        if self.engine not in ("bitsieve", "official"):
            raise ValueError(f"unknown engine: {self.engine}")
        if self.engine == "official" and self.semantic != "dense":
            raise ValueError("the official engine is only valid for the dense baseline")
        self.generation.validate()
        self.eviction.validate()
        if self.eviction.enabled and self.semantic == "dense":
            raise ValueError(
                "eviction needs a selector: the dense engine reads the whole "
                "prefix, so evicting from it would change what the model sees "
                "with nothing deciding what to keep"
            )
        self.quant.validate(head_dim=head_dim)
        self.selector.validate(
            block_size=self.generation.block_size,
            num_layers=num_layers,
        )
        if self.dense_kernel_variant not in ("blocked", "legacy"):
            raise ValueError(f"unknown dense_kernel_variant: {self.dense_kernel_variant}")
        if self.selector_kernel_variant not in ("blocked", "legacy"):
            raise ValueError(f"unknown selector_kernel_variant: {self.selector_kernel_variant}")
        if self.selector_logits_dtype not in ("float16", "float32"):
            raise ValueError(f"unknown selector_logits_dtype: {self.selector_logits_dtype}")
        if self.eviction.policy == "ema_recent_score" and not self.selector.value_aware:
            raise ValueError(
                "eviction policy ema_recent_score folds the value-aware selector "
                "score into its EMA; it needs selector.value_aware"
            )
        if self.compact_format not in ("bf16", "requantized"):
            raise ValueError(f"unknown compact_format: {self.compact_format}")
        if self.compact_format == "requantized" and self.engine != "bitsieve":
            raise ValueError("requantized compact cache requires the BitSieve engine")
        if self.max_cache_tokens < self.generation.block_size:
            raise ValueError("max_cache_tokens is smaller than one generation block")
        if self.coverage_query_chunk <= 0:
            raise ValueError("coverage_query_chunk must be positive")
        seen_arms: set[str] = set()
        for arm in self.coverage_arms:
            name = arm.get("name")
            if not name:
                raise ValueError("every coverage arm needs a name")
            if name in seen_arms:
                raise ValueError(f"duplicate coverage arm name {name!r}")
            seen_arms.add(name)
            bits = arm.get("bits", 16)
            if not isinstance(bits, int) or not 1 <= bits <= 16:
                raise ValueError(f"coverage arm {name!r}: bits must be 1-16, got {bits!r}")
            has_topk = arm.get("topk") is not None
            has_pct = arm.get("topk_percent") is not None
            if has_topk == has_pct:
                raise ValueError(
                    f"coverage arm {name!r}: give exactly one of topk / topk_percent"
                )
        if self.coverage_arms and not self.coverage_diagnostics:
            raise ValueError(
                "coverage_arms are scored against the fp16 reference that only "
                "coverage_diagnostics builds; enable it or drop the arms"
            )
        if self.precision_pairwise_diagnostics:
            if self.quant.k_bits != 16 or self.quant.v_bits != 16:
                raise ValueError("pairwise key precision diagnostics need a full-precision K/V reference run")
            if not self.eviction.record_only or self.eviction.policy != "lookahead":
                raise ValueError("pairwise precision diagnostics require lookahead with record_only=true")
            if (self.selector.value_set != "none" or self.selector.value_joint
                    or self.selector.value_per_query):
                raise ValueError("pairwise precision diagnostics currently support the default value-aware selector")
        if self.coverage_diagnostics and self.semantic == "dense":
            raise ValueError(
                "coverage_diagnostics needs a selector to score; it is meaningless "
                "for the dense baseline"
            )
        if self.semantic == "dense" and self.async_selector:

            self.async_selector = False

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        data = dict(raw)
        quant = QuantizationConfig(**data.pop("quant", {}))
        selector = SelectorConfig(**data.pop("selector", {}))
        generation = GenerationConfig(**data.pop("generation", {}))
        eviction = EvictionConfig(**data.pop("eviction", {}))
        cfg = cls(
            quant=quant, selector=selector, generation=generation,
            eviction=eviction, **data,
        )
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() == ".json":
            raw = json.loads(text)
        else:
            raw = yaml.safe_load(text)
        if not isinstance(raw, dict):
            raise ValueError(f"configuration in {p} must be a mapping")
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def dump(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix.lower() == ".json":
            p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        else:
            p.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
