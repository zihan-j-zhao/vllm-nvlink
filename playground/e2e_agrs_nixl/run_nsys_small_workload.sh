#!/usr/bin/env bash
# Start the P/D service, capture only a small AIPerf workload with nsys
# cudaProfilerApi gating, then stop the service so nsys can finalize.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUN_NAME="${RUN_NAME:?RUN_NAME must be set}"
LOG_DIR="${LOG_DIR:-playground/log/e2e_agrs_nixl/$RUN_NAME}"
OUT_DIR="${OUT_DIR:-playground/out/aiperf_pd/e2e_agrs_nixl/$RUN_NAME}"
REQUEST_RATE="${REQUEST_RATE:-2}"
REQUEST_COUNT="${REQUEST_COUNT:-20}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-128}"

mkdir -p "$LOG_DIR" "$OUT_DIR"

SERVER_PID=""
cleanup() {
    if [[ -n "$SERVER_PID" ]]; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "[nsys-small] run name          = $RUN_NAME"
echo "[nsys-small] log dir           = $LOG_DIR"
echo "[nsys-small] artifact dir      = $OUT_DIR"
echo "[nsys-small] request count/rate= $REQUEST_COUNT @ $REQUEST_RATE req/s"
echo "[nsys-small] max output tokens = $MAX_OUTPUT_TOKENS"

ALL2ALL_BACKEND="${ALL2ALL_BACKEND:-allgather_reducescatter}" \
MOE_BACKEND="${MOE_BACKEND:-triton}" \
PD_TRACE="${PD_TRACE:-1}" \
VLLM_PD_TRACE_MODEL_PHASES="${VLLM_PD_TRACE_MODEL_PHASES:-1}" \
VLLM_NSYS_ENGINE_NVTX="${VLLM_NSYS_ENGINE_NVTX:-1}" \
VLLM_NSYS_FORWARD_NVTX="${VLLM_NSYS_FORWARD_NVTX:-1}" \
VLLM_NSYS_KV_TRANSFER_NVTX="${VLLM_NSYS_KV_TRANSFER_NVTX:-1}" \
LOG_DIR="$LOG_DIR" \
bash playground/e2e_agrs_nixl/start_server.sh >"$LOG_DIR/start_server.outer.log" 2>&1 &
SERVER_PID=$!

echo "[nsys-small] server pid        = $SERVER_PID"
echo "[nsys-small] waiting for proxy readiness..."
timeout 1800 bash -c 'until curl -sf http://localhost:8000/healthcheck >/dev/null; do sleep 2; done'
echo "[nsys-small] proxy ready; starting cudaProfilerApi capture"

REQUEST_RATE="$REQUEST_RATE" \
REQUEST_COUNT="$REQUEST_COUNT" \
MAX_OUTPUT_TOKENS="$MAX_OUTPUT_TOKENS" \
OUT_DIR="$OUT_DIR" \
python - <<'PY'
import os
import subprocess
import torch

env = os.environ.copy()
torch.cuda.profiler.start()
try:
    subprocess.run(
        ["bash", "playground/e2e_agrs_nixl/run_aiperf.sh"],
        check=True,
        env=env,
    )
finally:
    torch.cuda.profiler.stop()
PY

echo "[nsys-small] workload complete; stopping server"
cleanup
SERVER_PID=""
