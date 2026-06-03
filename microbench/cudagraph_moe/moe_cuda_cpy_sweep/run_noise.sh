#!/usr/bin/env bash
# Launcher for the unified noise (none/p2p/nixl) bandwidth sweep bench.
#
# Defaults: NIXL noise from peer (rank 1) -> victim (rank 0). 2 GPUs only.
# (P2P mode needs CUDA_VISIBLE_DEVICES=0,1,2.)
#
# Examples:
#   bash run_noise.sh                                   # NIXL noise, default sweep
#   bash run_noise.sh --noise-mode p2p                  # cudaMemcpyPeerAsync
#       (also set CUDA_VISIBLE_DEVICES=0,1,2)
#   bash run_noise.sh --noise-mode none --skip-quiet=false  # quiet only
#   bash run_noise.sh --sweep-gbps 10,40,160,640
#   bash run_noise.sh --nixl-requests-per-burst 16 --nixl-blocks-per-request 16
#
# Need a different GPU triple?
#   CUDA_VISIBLE_DEVICES=4,5 bash run_noise.sh                       # nixl only
#   CUDA_VISIBLE_DEVICES=4,5,6 bash run_noise.sh --noise-mode p2p    # p2p

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

# NIXL: prefer UCX with GPUDirect over NVLink/PCIe so transfers actually move bytes.
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-all}"
export UCX_TLS="${UCX_TLS:-cuda_copy,cuda_ipc,rc_x,tcp}"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
PYTHON="${PYTHON:-/home/wxzheng/miniforge3/envs/vllm-nvlink/bin/python}"
if [[ ! -x "$PYTHON" ]]; then PYTHON="$(command -v python3)"; fi

cd "$SCRIPT_DIR"

echo "[run_noise.sh] VLLM_ROOT            = $VLLM_ROOT"
echo "[run_noise.sh] MODEL                = $MODEL"
echo "[run_noise.sh] CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "[run_noise.sh] NPROC                = $NPROC"
echo "[run_noise.sh] PYTHON               = $PYTHON"
echo "[run_noise.sh] UCX_TLS              = $UCX_TLS"

exec "$PYTHON" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$NPROC" \
    decode_noise_sweep.py \
    --model "$MODEL" \
    "$@"
