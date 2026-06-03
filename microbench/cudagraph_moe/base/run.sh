#!/usr/bin/env bash
# Torchrun launcher for the CUDA-graph decode microbenchmark.
#
# Defaults: 2 GPUs, DP=2, EP enabled, FLASHINFER attention, triton MoE,
# allgather_reducescatter all2all. Mirrors playground/moe_a2a/start_server_cudagraph.sh
# and the decode side of playground/e2e_agrs_nixl/start_server.sh.
#
# Usage:
#     bash run.sh                              # defaults
#     bash run.sh --batch-sizes 1,4,16,64 --iters 50
#     CUDA_VISIBLE_DEVICES=4,5 NPROC=2 bash run.sh
#     MODEL=/local/path/Qwen3-30B bash run.sh --batch-sizes 8 --iters 20

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC="${NPROC:-$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")}"

export PYTHONPATH="${VLLM_ROOT}:${PYTHONPATH:-}"

# External launcher mode requires V1 multiprocessing disabled for determinism.
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
export VLLM_USE_MODELSCOPE="${VLLM_USE_MODELSCOPE:-False}"
# Disable the MoE A2A profiler that some forks enable by default.
export VLLM_MOE_A2A_PROFILE="${VLLM_MOE_A2A_PROFILE:-0}"

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
    decode_cudagraph_bench.py \
    --model "$MODEL" \
    "$@"
