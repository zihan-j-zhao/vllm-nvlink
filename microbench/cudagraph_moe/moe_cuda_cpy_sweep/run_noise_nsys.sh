#!/usr/bin/env bash
# nsys profile with GPU hardware metrics for run_noise.sh.
#
# Requires sudo (GPU hardware counters are root-only).
# Captures starts at the first NVTX `bench/phase/quiet` range so we skip
# the multi-minute LLM build + CUDA-graph capture and keep the trace small.
#
# Usage:
#     sudo bash run_noise_nsys.sh                                # NIXL noise
#     sudo bash run_noise_nsys.sh --noise-mode p2p               # P2P noise (CVD=0,1,2)
#     NSYS_OUT=results/foo sudo bash run_noise_nsys.sh --sweep-gbps 40,160 --iters 40

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NSYS=/usr/local/cuda-12.9/bin/nsys

if [[ "${EUID}" -ne 0 ]]; then
    echo "must run with sudo for GPU hardware metrics" >&2; exit 1
fi

mkdir -p "$SCRIPT_DIR/results"
NSYS_OUT="${NSYS_OUT:-$SCRIPT_DIR/results/nsys_noise_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$(dirname "$NSYS_OUT")"

# Default args make the trace small: 2 sweep points, modest iters.
EXTRA_ARGS=("$@")
if [[ ${#EXTRA_ARGS[@]} -eq 0 ]]; then
    EXTRA_ARGS=( --sweep-gbps 40,160 --iters 80 --warmup-iters 10 )
fi

# GPU metrics for the visible GPUs (0,1 for NIXL; add 2 for P2P).
# Conservatively trace cuda runtime + driver + nvtx + osrt.
# Preserve PATH so the conda env's `ninja` (used by torch.compile) is visible
# even under sudo's secure_path stripping.
CONDA_BIN="$(dirname "${PYTHON:-/home/wxzheng/miniforge3/envs/vllm-nvlink/bin/python}")"
export PATH="$CONDA_BIN:${PATH}"
exec "$NSYS" profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --sample=none \
    --gpu-metrics-devices=cuda-visible \
    --gpu-metrics-set=gb10x \
    --force-overwrite=true \
    --output="$NSYS_OUT" \
    bash "$SCRIPT_DIR/run_noise.sh" "${EXTRA_ARGS[@]}"
