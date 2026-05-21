#!/usr/bin/env bash
# Launch Qwen3-30B-A3B as an OpenAI-compatible server with DP=EP=2 and
# the MoE all-to-all profiler enabled.
#
# Profiler hard-asserts the following preconditions (see
# vllm/distributed/moe_a2a_profiler.py); this script sets them explicitly:
#   * data_parallel_size == 2
#   * enable_expert_parallel
#   * enforce_eager (CUDA graphs hide the Python-level recorder)
#   * pcp_size == 1 (default)
#   * all2all backend in {naive, allgather_reducescatter}
#
# Per-rank JSONL traces are written under playground/log/moe_a2a/.
#
# Env knobs:
#   MODEL                 HF model id (default: Qwen/Qwen3-30B-A3B-Instruct-2507)
#   PORT                  Server port    (default: 8000)
#   CUDA_VISIBLE_DEVICES  Two GPU ids    (default: 0,1)
#   SERVED_MODEL_NAME     OpenAI model id  (default: same as MODEL basename)
#   LOG_DIR               Output dir for JSONL + server log
#                         (default: playground/log/moe_a2a/<UTC stamp>)

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# --- Config ------------------------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
PORT="${PORT:-8000}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "$MODEL")}"
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
LOG_DIR="${LOG_DIR:-playground/log/moe_a2a/$TS}"
mkdir -p "$LOG_DIR"
# Resolve to absolute path so the workers (which may not share this
# script's CWD) write traces to the intended location.
LOG_DIR="$(cd "$LOG_DIR" && pwd)"

# --- Profiler ---------------------------------------------------------------
export VLLM_MOE_A2A_PROFILE=1
export VLLM_MOE_A2A_PROFILE_PATH="$LOG_DIR/moe_a2a_rank{rank}.jsonl"

# Wipe any stale traces from a prior run sharing this path.
rm -f "$LOG_DIR"/moe_a2a_rank*.jsonl "$LOG_DIR"/moe_a2a_rank*.jsonl.summary

# --- Python -----------------------------------------------------------------
PY="${PY:-/root/miniconda3/envs/vllm-nvlink/bin/python}"
if [[ ! -x "$PY" ]]; then
    echo "error: python not found at $PY" >&2
    echo "       set PY=/path/to/python or run build.sh first" >&2
    exit 1
fi

echo "[start_server] model              = $MODEL"
echo "[start_server] served-model-name  = $SERVED_MODEL_NAME"
echo "[start_server] CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "[start_server] log dir            = $LOG_DIR"
echo "[start_server] profile path       = $VLLM_MOE_A2A_PROFILE_PATH"

# `setsid` puts the server + worker subprocesses in their own session so a
# Ctrl-C / SIGTERM to this script delivers cleanly to the workers, giving
# the profiler a chance to flush.
#
# `--moe-backend triton` forces the naive AG/RS dispatch/combine path
# through `CudaCommunicator.dispatch/combine` (the profiler's choke point).
# Without this, vLLM auto-selects `flashinfer_trtllm` on B200, which uses
# a modular kernel that bypasses the EP communicator and produces no
# traces.
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
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    2>&1 | tee "$LOG_DIR/server.log"
