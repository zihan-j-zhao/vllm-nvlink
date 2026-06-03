#!/usr/bin/env bash
# Launcher for the P2P-noise CUDA-graph decode microbench.
#
# Defaults:
#   CUDA_VISIBLE_DEVICES = 0,1,2  (2 vLLM ranks + 1 noise source)
#   NPROC                = 2      (DP=2, EP enabled)
#   P2P noise            = cuda:2 -> cuda:0 at 10 GiB/s
#
# Examples:
#   bash run.sh                                  # defaults
#   bash run.sh --target-gbps 25                 # heavier noise
#   bash run.sh --no-p2p                         # phase B without noise (control)
#   bash run.sh --p2p-src-device 3 --p2p-dst-rank 1
#   CUDA_VISIBLE_DEVICES=4,5,6 bash run.sh
#
# NOTE: noise GPU MUST be in CUDA_VISIBLE_DEVICES, otherwise the worker can't see it.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
NPROC="${NPROC:-2}"

export PYTHONPATH="${VLLM_ROOT}:${PYTHONPATH:-}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
export VLLM_USE_MODELSCOPE="${VLLM_USE_MODELSCOPE:-False}"
export VLLM_MOE_A2A_PROFILE="${VLLM_MOE_A2A_PROFILE:-0}"

# Hint to the CUDA driver about which devices may need P2P. Optional.
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-8}"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
PYTHON="${PYTHON:-/home/wxzheng/miniforge3/envs/vllm-nvlink/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="$(command -v python3)"
fi

cd "$SCRIPT_DIR"

echo "[run.sh] VLLM_ROOT            = $VLLM_ROOT"
echo "[run.sh] MODEL                = $MODEL"
echo "[run.sh] CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "[run.sh] NPROC                = $NPROC"
echo "[run.sh] PYTHON               = $PYTHON"

exec "$PYTHON" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$NPROC" \
    decode_p2p_bench.py \
    --model "$MODEL" \
    "$@"
