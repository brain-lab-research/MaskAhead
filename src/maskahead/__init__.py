from .config import ExperimentConfig, GenerationConfig, QuantizationConfig, SelectorConfig

__all__ = [
    "ExperimentConfig",
    "GenerationConfig",
    "GenerationResult",
    "QuantizationConfig",
    "SelectorConfig",
    "BitSieveGenerator",
    "patch_fast_dllm",
    "unpatch_fast_dllm",
]


def __getattr__(name: str):
    if name in {"GenerationResult", "BitSieveGenerator"}:
        from .runtime.generator import GenerationResult, BitSieveGenerator

        return {"GenerationResult": GenerationResult, "BitSieveGenerator": BitSieveGenerator}[name]
    if name in {"patch_fast_dllm", "unpatch_fast_dllm"}:
        from .runtime.adapter import patch_fast_dllm, unpatch_fast_dllm

        return {"patch_fast_dllm": patch_fast_dllm, "unpatch_fast_dllm": unpatch_fast_dllm}[name]
    raise AttributeError(name)
