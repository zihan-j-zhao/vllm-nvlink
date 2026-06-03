#!/usr/bin/env bash
# Capture an Nsight Systems timeline of the P2P-noise decode microbench.
#
# Captures GPU activity + CUDA API + NVTX. The NVTX ranges in decode_p2p_bench.py
# (bench/setup, bench/phase/quiet, bench/phase/noisy, bench/phase/.../iter_N,
# bench/p2p/copy_*) make it easy to focus on either phase or per-iter.
#
# Usage:
#   bash run_nsys.sh                                    # defaults
#   NSYS_OUTPUT=trace_25gbps bash run_nsys.sh --target-gbps 25
#   bash run_nsys.sh --no-p2p                           # control trace

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v nsys >/dev/null 2>&1; then
    echo "error: nsys not on PATH" >&2
    exit 1
fi

NSYS_OUTPUT="${NSYS_OUTPUT:-results/nsys_p2p_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$(dirname "$NSYS_OUTPUT")"

# Skip the heavy LLM-build + CUDA-graph capture phase in the trace by default.
# The bench prints "Phase A" / "Phase B" — those are the regions of interest.
# Tune --delay or use --capture-range / NVTX triggers if desired.
NSYS_ARGS=(
    profile
    --trace=cuda,nvtx,osrt
    --cuda-memory-usage=true
    --sample=none
    --capture-range=nvtx
    --nvtx-capture='bench/phase/quiet'
    --capture-range-end=stop-shutdown
    --force-overwrite=true
    --output="$NSYS_OUTPUT"
)

NSYS=/usr/local/cuda-12.9/bin/nsys
CONDA_ENV=/home/wxzheng/miniconda3/envs/vllm-nvlink
# nsys wraps the launcher; inside it, torchrun will spawn both ranks under the
# same nsys session, which is what we want.
exec $NSYS "${NSYS_ARGS[@]}" \
    bash "$SCRIPT_DIR/run.sh" "$@"
