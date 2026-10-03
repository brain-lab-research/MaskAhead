#!/bin/bash
# Run a command inside the PPU PyTorch image (Alibaba PPU cluster), repo mounted.
# Inside a Slurm allocation podman injects the job's PPU cards automatically.
#   PPU_IMAGE=... launch/ppu/container.sh python scripts/smoke_test.py --model fastdllm
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
IMG="${PPU_IMAGE:?set PPU_IMAGE to the PPU pytorch image}"
exec podman run --rm --http-proxy=false --network host --ipc host \
  -v "$HOME:$HOME" -v "$ROOT:$ROOT" -w "$ROOT" \
  -e HOME="$HOME" -e HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" \
  -e PYTHONPATH="$ROOT/src" -e PYTHONNOUSERSITE=1 -e TOKENIZERS_PARALLELISM=false \
  -e FASTDLLM_PYTHON -e DREAM_PYTHON -e FASTDLLM_MODEL -e DREAM_MODEL \
  -e NCCL_SOCKET_IFNAME=bond0 -e GLOO_SOCKET_IFNAME=bond0 \
  "$IMG" "$@"
