#!/usr/bin/env bash
# Sparse-burst noise bench, calibrated to production
#   xfers/step ~ 0.12   p50 transfer ~ 12 MB   avg ~ 9 GB/s
#
# Examples:
#   bash run_burst.sh                                       # NIXL bursts, defaults
#   bash run_burst.sh --noise-mode p2p                      # P2P bursts (set CVD=0,1,2)
#   bash run_burst.sh --burst-prob 0.25 --burst-mb 30       # heavier
#   bash run_burst.sh --noise-mode none                     # control
#   bash run_burst.sh --iters 5000 --burst-prob 0.12        # more samples
#
# Outputs to results/<tag>_{timeseries.json,iters.csv,summary.csv}.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC="${NPROC:-2}"

export PYTHONPATH="${VLLM_ROOT}:${PYTHONPATH:-}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
export VLLM_USE_MODELSCOPE="${VLLM_USE_MODELSCOPE:-False}"
export VLLM_MOE_A2A_PROFILE="${VLLM_MOE_A2A_PROFILE:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-8}"
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-all}"
export UCX_TLS="${UCX_TLS:-cuda_copy,cuda_ipc,rc_x,tcp}"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
PYTHON="${PYTHON:-/home/wxzheng/miniforge3/envs/vllm-nvlink/bin/python}"
if [[ ! -x "$PYTHON" ]]; then PYTHON="$(command -v python3)"; fi

cd "$SCRIPT_DIR"
echo "[run_burst] CVD=$CUDA_VISIBLE_DEVICES NPROC=$NPROC PYTHON=$PYTHON MODEL=$MODEL"

exec "$PYTHON" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$NPROC" \
    decode_burst_bench.py \
    --model "$MODEL" \
    "$@"
