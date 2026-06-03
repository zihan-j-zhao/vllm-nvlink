#!/usr/bin/env bash
# Launcher for the P2P-bandwidth sweep bench.
#
# Defaults:
#   CUDA_VISIBLE_DEVICES = 0,1,2  (DP=2 on 0,1 + noise source 2)
#   NPROC                = 2
#   sweep                = 10, 20, 40, 80, 160, 320, 640 GiB/s
#   iters per phase      = 200
#
# Model is initialized once for the whole sweep.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
NPROC="${NPROC:-2}"

export PYTHONPATH="${VLLM_ROOT}:${PYTHONPATH:-}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
export VLLM_USE_MODELSCOPE="${VLLM_USE_MODELSCOPE:-False}"
export VLLM_MOE_A2A_PROFILE="${VLLM_MOE_A2A_PROFILE:-0}"
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
    decode_p2p_sweep.py \
    --model "$MODEL" \
    "$@"
