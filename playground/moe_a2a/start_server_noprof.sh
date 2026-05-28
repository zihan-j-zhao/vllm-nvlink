#!/usr/bin/env bash
# Launch Qwen3-30B-A3B as an OpenAI-compatible server with the *same*
# DP=EP=2 / triton MoE / FlashInfer attention configuration as
# `start_server.sh`, but with the MoE all-to-all profiler DISABLED.
#
# Purpose: apples-to-apples baseline for the AIPerf-driven time-series
# experiment (`plot_step_latency_timeseries.py`). Every CLI flag below
# matches `start_server.sh` exactly; the only difference is the absence
# of `VLLM_MOE_A2A_PROFILE=1`, so the per-call CUDA-event allocation,
# JSONL writes, and atexit drain in
# `vllm/distributed/moe_a2a_profiler.py` are bypassed.
#
# Env knobs (same as `start_server.sh`):
#   MODEL                 HF model id (default: Qwen/Qwen3-30B-A3B-Instruct-2507)
#   PORT                  Server port    (default: 8000)
#   CUDA_VISIBLE_DEVICES  Two GPU ids    (default: 0,1)
#   SERVED_MODEL_NAME     OpenAI model id  (default: same as MODEL basename)
#   LOG_DIR               Output dir for server log
#                         (default: playground/log/moe_a2a/noprof_<UTC stamp>)

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# --- ModelScope OFF ---------------------------------------------------------
# Same defensive unset as `start_server.sh`. See that file for rationale.
unset VLLM_USE_MODELSCOPE LMDEPLOY_USE_MODELSCOPE MODELSCOPE_CACHE \
      MEGATRON_LM_PATH
export VLLM_USE_MODELSCOPE=False

# --- Config ------------------------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
PORT="${PORT:-8000}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "$MODEL")}"
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
LOG_DIR="${LOG_DIR:-playground/log/moe_a2a/noprof_$TS}"
mkdir -p "$LOG_DIR"
# Resolve to absolute path so the workers (which may not share this
# script's CWD) write logs to the intended location.
LOG_DIR="$(cd "$LOG_DIR" && pwd)"

# --- Profiler (explicitly OFF) ----------------------------------------------
# Force-unset in case the caller's environment has it set from a previous
# `start_server.sh` invocation in the same shell.
unset VLLM_MOE_A2A_PROFILE
unset VLLM_MOE_A2A_PROFILE_PATH
export VLLM_MOE_A2A_PROFILE=0

# --- Python -----------------------------------------------------------------
PY="${PY:-/root/miniconda3/envs/vllm-nvlink/bin/python}"
if [[ ! -x "$PY" ]]; then
    echo "error: python not found at $PY" >&2
    echo "       set PY=/path/to/python or run build.sh first" >&2
    exit 1
fi

echo "[start_server_noprof] model              = $MODEL"
echo "[start_server_noprof] served-model-name  = $SERVED_MODEL_NAME"
echo "[start_server_noprof] CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "[start_server_noprof] log dir            = $LOG_DIR"
echo "[start_server_noprof] profiler           = DISABLED (VLLM_MOE_A2A_PROFILE=0)"

# `setsid` keeps Ctrl-C / SIGTERM semantics identical to start_server.sh
# so the workers can shut down cleanly. CLI flags are intentionally a
# byte-for-byte match with start_server.sh except for the absence of
# profiling envs above.
exec setsid "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --port "$PORT" \
    --data-parallel-size 2 \
    --enable-expert-parallel \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --enforce-eager \
    --moe-backend triton \
    --disable-log-stats \
    --no-enable-log-requests \
    --attention-backend FLASHINFER \
    --attention-config.use_trtllm_attention=False \
    --max-num-batched-tokens 2048 \
    --gpu-memory-utilization 0.85 \
    2>&1 | tee "$LOG_DIR/server.log"
