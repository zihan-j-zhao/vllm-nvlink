#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

needs_kv=0
for arg in "$@"; do
  if [[ "$arg" == "--kv-transfer-mb" || "$arg" == --kv-transfer-mb=* ]]; then
    needs_kv=1
  fi
done

if [[ "$needs_kv" == 1 ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,3,4}"
  NPROC="${NPROC:-4}"
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
  NPROC="${NPROC:-2}"
fi
export PYTHONPATH="${VLLM_ROOT}:${PYTHONPATH:-}"
export VLLM_MOE_A2A_PROFILE="${VLLM_MOE_A2A_PROFILE:-0}"

PYTHON="${PYTHON:-/home/wxzheng/miniforge3/envs/vllm-nvlink/bin/python}"

cd "${SCRIPT_DIR}"
exec "${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${NPROC}" \
  fused_moe_2gpu.py "$@"
