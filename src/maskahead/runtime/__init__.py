__all__ = [
    "GenerationResult",
    "PatchHandle",
    "BitSieveGenerator",
    "BitSieveSession",
    "patch_fast_dllm",
    "set_bitsieve_session",
    "unpatch_fast_dllm",
]


def __getattr__(name: str):
    if name in {"GenerationResult", "BitSieveGenerator"}:
        from .generator import GenerationResult, BitSieveGenerator

        return {"GenerationResult": GenerationResult, "BitSieveGenerator": BitSieveGenerator}[name]
    if name == "BitSieveSession":
        from .session import BitSieveSession

        return BitSieveSession
    if name in {"PatchHandle", "patch_fast_dllm", "set_bitsieve_session", "unpatch_fast_dllm"}:
        from .adapter import PatchHandle, patch_fast_dllm, set_bitsieve_session, unpatch_fast_dllm

        return {
            "PatchHandle": PatchHandle,
            "patch_fast_dllm": patch_fast_dllm,
            "set_bitsieve_session": set_bitsieve_session,
            "unpatch_fast_dllm": unpatch_fast_dllm,
        }[name]
    raise AttributeError(name)
