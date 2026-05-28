#!/usr/bin/env bash
# Same as start_server_noprof.sh but with CUDA graphs ENABLED (i.e. without
# --enforce-eager). Used by the cudagraph TPOT-fluctuation experiment that
# compares baseline vs concurrent KV-transfer simulation. All other flags
# are a byte-for-byte match with start_server_noprof.sh so the only
# axis-of-variation is graph mode.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# --- ModelScope OFF ---------------------------------------------------------
unset VLLM_USE_MODELSCOPE LMDEPLOY_USE_MODELSCOPE MODELSCOPE_CACHE \
      MEGATRON_LM_PATH
export VLLM_USE_MODELSCOPE=False

# --- Config ------------------------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
PORT="${PORT:-8000}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "$MODEL")}"
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
LOG_DIR="${LOG_DIR:-playground/log/moe_a2a/cudagraph_$TS}"
mkdir -p "$LOG_DIR"
LOG_DIR="$(cd "$LOG_DIR" && pwd)"

# --- Profiler (explicitly OFF) ----------------------------------------------
unset VLLM_MOE_A2A_PROFILE
unset VLLM_MOE_A2A_PROFILE_PATH
export VLLM_MOE_A2A_PROFILE=0

# --- Python -----------------------------------------------------------------
PY="${PY:-/root/miniconda3/envs/vllm-nvlink/bin/python}"
if [[ ! -x "$PY" ]]; then
    echo "error: python not found at $PY" >&2
    exit 1
fi

echo "[start_server_cudagraph] model              = $MODEL"
echo "[start_server_cudagraph] served-model-name  = $SERVED_MODEL_NAME"
echo "[start_server_cudagraph] CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "[start_server_cudagraph] port               = $PORT"
echo "[start_server_cudagraph] log dir            = $LOG_DIR"
echo "[start_server_cudagraph] profiler           = DISABLED (VLLM_MOE_A2A_PROFILE=0)"
echo "[start_server_cudagraph] cuda graphs        = ENABLED (no --enforce-eager)"

# Only difference vs start_server_noprof.sh: --enforce-eager is removed.
exec setsid "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --port "$PORT" \
    --data-parallel-size 2 \
    --enable-expert-parallel \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --moe-backend triton \
    --disable-log-stats \
    --no-enable-log-requests \
    --no-enable-prefix-caching \
    --attention-backend FLASHINFER \
    --attention-config.use_trtllm_attention=False \
    --max-num-batched-tokens 2048 \
    --gpu-memory-utilization 0.85 \
    2>&1 | tee "$LOG_DIR/server.log"
