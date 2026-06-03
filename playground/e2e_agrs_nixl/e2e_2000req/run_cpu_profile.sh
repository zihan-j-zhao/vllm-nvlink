#!/usr/bin/env bash
# Run a production-like CUDA-graph P/D NIXL experiment for high-level CPU/JSONL profiling.
#
# Method:
#   1. Start P/D service with CUDA graphs enabled and PD JSONL tracing enabled.
#   2. Warmup: 1000 requests @ 20 req/s. Wait for all requests to finish.
#   3. Main:   2000 requests @ 20 req/s. Wait for all requests to finish.
#
# The CUDA/Nsight/torch profiler paths are explicitly disabled here so this run
# is suitable for high-level behavior analysis without profiler perturbation.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

RUN_NAME="${RUN_NAME:-e2e_agrs_nixl_cpu_2000req_$(date -u +%Y-%m-%dT%H-%M-%SZ)}"
LOG_DIR="${LOG_DIR:-playground/log/e2e_agrs_nixl/$RUN_NAME}"
BASE_OUT_DIR="${BASE_OUT_DIR:-playground/out/aiperf_pd/e2e_agrs_nixl/$RUN_NAME}"
WARMUP_COUNT="${WARMUP_COUNT:-1000}"
MAIN_COUNT="${MAIN_COUNT:-2000}"
REQUEST_RATE="${REQUEST_RATE:-20}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-500}"

mkdir -p "$LOG_DIR" "$BASE_OUT_DIR"
printf '%s\n' "$RUN_NAME" | tee /tmp/e2e_2000req_run_name.txt

# Hard-disable CUDA/Nsight/vLLM profiler paths. These env vars are intentionally
# unset even if the shell still has them from earlier nsys experiments.
unset VLLM_NSYS_CAPTURE_DECODE_FORWARD_STEPS
unset VLLM_NSYS_CAPTURE_DECODE_ENGINE_STEPS
unset VLLM_NSYS_CAPTURE_TRIGGER_DP
unset VLLM_NSYS_ENGINE_NVTX
unset VLLM_NSYS_FORWARD_NVTX
unset VLLM_NSYS_KV_TRANSFER_NVTX
unset VLLM_NSYS_KV_NVTX
unset PROFILER_CONFIG
unset VLLM_TORCH_PROFILER_DIR

export ALL2ALL_BACKEND="${ALL2ALL_BACKEND:-allgather_reducescatter}"
export MOE_BACKEND="${MOE_BACKEND:-triton}"
export PD_TRACE=1
export VLLM_PD_TRACE_REQ_IDS=1
# CUDA graphs remain enabled by leaving ENFORCE_EAGER off.
export ENFORCE_EAGER=0
export VLLM_PD_TRACE_MODEL_PHASES="${VLLM_PD_TRACE_MODEL_PHASES:-0}"

SERVER_PID=""
cleanup() {
    if [[ -n "$SERVER_PID" ]]; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

mark_phase() {
    local name=$1
    local file="$BASE_OUT_DIR/phase_markers.jsonl"
    printf '{"phase":"%s","time_epoch_s":%s}\n' "$name" "$(date +%s.%N)" >> "$file"
}

echo "[cpu-profile] run name       = $RUN_NAME"
echo "[cpu-profile] log dir        = $LOG_DIR"
echo "[cpu-profile] output dir     = $BASE_OUT_DIR"
echo "[cpu-profile] warmup/main    = $WARMUP_COUNT / $MAIN_COUNT @ $REQUEST_RATE req/s"
echo "[cpu-profile] cuda graphs    = ENABLED (ENFORCE_EAGER=0)"
echo "[cpu-profile] nsys/profiler  = DISABLED (VLLM_NSYS_* and PROFILER_CONFIG unset)"

LOG_DIR="$LOG_DIR" bash playground/e2e_agrs_nixl/start_server.sh \
    >"$LOG_DIR/start_server.outer.log" 2>&1 &
SERVER_PID=$!

echo "[cpu-profile] server pid     = $SERVER_PID"
echo "[cpu-profile] waiting for proxy readiness..."
timeout 1800 bash -c 'until curl -sf http://localhost:8000/healthcheck >/dev/null; do sleep 2; done'
echo "[cpu-profile] proxy ready"

mark_phase warmup_start
OUT_DIR="$BASE_OUT_DIR/warmup" \
REQUEST_COUNT="$WARMUP_COUNT" \
REQUEST_RATE="$REQUEST_RATE" \
MAX_OUTPUT_TOKENS="$MAX_OUTPUT_TOKENS" \
bash playground/e2e_agrs_nixl/run_aiperf.sh >"$BASE_OUT_DIR/warmup.outer.log" 2>&1
mark_phase warmup_done

mark_phase main_start
OUT_DIR="$BASE_OUT_DIR/main" \
REQUEST_COUNT="$MAIN_COUNT" \
REQUEST_RATE="$REQUEST_RATE" \
MAX_OUTPUT_TOKENS="$MAX_OUTPUT_TOKENS" \
bash playground/e2e_agrs_nixl/run_aiperf.sh >"$BASE_OUT_DIR/main.outer.log" 2>&1
mark_phase main_done

echo "[cpu-profile] done; stopping server"
cleanup
SERVER_PID=""

echo "[cpu-profile] artifacts:"
echo "  trace:   $LOG_DIR/pd_trace"
echo "  markers: $BASE_OUT_DIR/phase_markers.jsonl"
echo "  warmup:  $BASE_OUT_DIR/warmup"
echo "  main:    $BASE_OUT_DIR/main"
