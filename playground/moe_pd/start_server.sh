#!/usr/bin/env bash
# Launch Qwen3-30B-A3B as a P/D-disaggregated OpenAI-compatible service
# using vLLM's NixlConnector (NIXL/UCX KV transport), with optional
# data-parallel expert parallelism (DP-EP) on each side.
#
#   - 1 prefill vLLM server (KV producer) on $PREFILL_GPUS
#   - 1 decode  vLLM server (KV consumer) on $DECODE_GPUS
#   - 1 proxy server (playground/moe_pd/proxy.py, a fork of
#     tests/v1/kv_connector/nixl_integration/toy_proxy_server.py) that
#     mediates the prefill->decode handoff (sends `do_remote_decode:true`
#     to prefill, copies `kv_transfer_params` from prefill's response into
#     the decode request, then forwards the full body to decode).
#
# Default is 4 GPUs total (4-7): prefill on 4,5 and decode on 6,7, each with
# data-parallel-size=2 + --enable-expert-parallel and TP=PP=1. That matches
# the moe_a2a/start_server_cudagraph.sh DP-EP=2 baseline, just split across
# the producer and consumer roles. To go back to the original 1P1D-no-EP
# layout, set PREFILL_GPUS=6 DECODE_GPUS=7 (single GPU on each side
# implicitly disables --data-parallel-size / --enable-expert-parallel).
#
# NIXL side channel: each DP rank's scheduler binds
# VLLM_NIXL_SIDE_CHANNEL_PORT + data_parallel_index. We therefore reserve
# DP-many consecutive ports per side and require
# DECODE_SIDE_PORT >= PREFILL_SIDE_PORT + PREFILL_DP.
#
# All vLLM flags outside the disagg block stay aligned with
# moe_a2a/start_server_cudagraph.sh so behaviour outside the P/D split
# matches the cudagraph baseline.
#
# Env knobs:
#   MODEL              HF model id     (default: Qwen/Qwen3-30B-A3B-Instruct-2507)
#   PROXY_PORT         Proxy port      (default: 8000)
#   PREFILL_PORT       Prefill HTTP    (default: 8100)
#   DECODE_PORT        Decode HTTP    (default: 8200)
#   PREFILL_SIDE_PORT  NIXL side ch. base for prefill DP ranks (default: 5559)
#   DECODE_SIDE_PORT   NIXL side ch. base for decode  DP ranks (default: 5570)
#   PREFILL_GPUS       Comma-separated CUDA ids for prefill (default: 4,5)
#                      Legacy alias: PREFILL_GPU (single id)
#   DECODE_GPUS        Comma-separated CUDA ids for decode  (default: 6,7)
#                      Legacy alias: DECODE_GPU  (single id)
#   EXPERT_PARALLEL    "1"/"0" override; default "1" iff per-side DP > 1
#   SERVED_MODEL_NAME  OpenAI model id (default: basename of $MODEL)
#   GPU_MEM_UTIL       --gpu-memory-utilization per worker (default: 0.85)
#   LOG_DIR            Output dir      (default: playground/log/moe_pd/<UTC>)
#   PY                 Python binary   (default: vllm-nvlink conda env python)

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# --- ModelScope OFF ---------------------------------------------------------
# Same rationale as moe_a2a/start_server*.sh: scrub any inherited ModelScope
# env so vLLM uses Hugging Face.
unset VLLM_USE_MODELSCOPE LMDEPLOY_USE_MODELSCOPE MODELSCOPE_CACHE \
      MEGATRON_LM_PATH
export VLLM_USE_MODELSCOPE=False

# --- Config -----------------------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
PROXY_PORT="${PROXY_PORT:-8000}"
PREFILL_PORT="${PREFILL_PORT:-8100}"
DECODE_PORT="${DECODE_PORT:-8200}"
# NIXL side-channel base ports. The scheduler binds
# VLLM_NIXL_SIDE_CHANNEL_PORT + data_parallel_index, so we space the two
# bases by max(DP, 4) to leave headroom for higher-DP sweeps.
PREFILL_SIDE_PORT="${PREFILL_SIDE_PORT:-5559}"
DECODE_SIDE_PORT="${DECODE_SIDE_PORT:-5570}"
# Per-side GPU lists. Comma-separated; legacy single-GPU env vars
# (PREFILL_GPU / DECODE_GPU) are honored as fallbacks so the original
# 1P1D-no-EP invocation still works unchanged.
PREFILL_GPUS="${PREFILL_GPUS:-${PREFILL_GPU:-4,5}}"
DECODE_GPUS="${DECODE_GPUS:-${DECODE_GPU:-6,7}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "$MODEL")}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
LOG_DIR="${LOG_DIR:-playground/log/moe_pd/$TS}"
mkdir -p "$LOG_DIR"
LOG_DIR="$(cd "$LOG_DIR" && pwd)"

# Derive per-side DP size from the GPU list length. Single GPU per side ->
# DP=1 (no --data-parallel-size / no --enable-expert-parallel), matching
# the original 1P1D-no-EP layout. DP>1 -> add the DP-EP flags so behavior
# inside each role mirrors moe_a2a/start_server_cudagraph.sh.
_count_csv() { local IFS=','; read -r -a __arr <<< "$1"; echo "${#__arr[@]}"; }
PREFILL_DP="$(_count_csv "$PREFILL_GPUS")"
DECODE_DP="$(_count_csv "$DECODE_GPUS")"
if (( PREFILL_DP < 1 )) || (( DECODE_DP < 1 )); then
    echo "error: PREFILL_GPUS / DECODE_GPUS must list at least one GPU" >&2
    exit 1
fi
# Default EXPERT_PARALLEL on when either side has DP>1.
if [[ -z "${EXPERT_PARALLEL:-}" ]]; then
    if (( PREFILL_DP > 1 )) || (( DECODE_DP > 1 )); then
        EXPERT_PARALLEL=1
    else
        EXPERT_PARALLEL=0
    fi
fi
# Enforce non-overlapping side-channel port ranges. Each DP rank's
# scheduler binds base + dp_index, so [PREFILL_SIDE_PORT, +PREFILL_DP)
# must not collide with [DECODE_SIDE_PORT, +DECODE_DP).
if (( DECODE_SIDE_PORT < PREFILL_SIDE_PORT + PREFILL_DP )) && \
   (( PREFILL_SIDE_PORT < DECODE_SIDE_PORT + DECODE_DP )); then
    echo "error: NIXL side-channel ranges overlap:" >&2
    echo "       prefill uses [$PREFILL_SIDE_PORT, $((PREFILL_SIDE_PORT + PREFILL_DP - 1))]" >&2
    echo "       decode  uses [$DECODE_SIDE_PORT,  $((DECODE_SIDE_PORT  + DECODE_DP  - 1))]" >&2
    echo "       bump DECODE_SIDE_PORT past PREFILL_SIDE_PORT + PREFILL_DP" >&2
    exit 1
fi

# --- Profiler (explicitly OFF, matches start_server_cudagraph.sh) -----------
unset VLLM_MOE_A2A_PROFILE VLLM_MOE_A2A_PROFILE_PATH
export VLLM_MOE_A2A_PROFILE=0

# --- P/D-disagg JSONL tracer (env-gated, default ON when PD_TRACE=1) --------
# Writes per-process JSONL files under VLLM_PD_TRACE_DIR. Records:
#   * decode-side NIXL recv_start/recv_done (KV-xfer active intervals)
#   * per-engine-step token emissions (for system-wide TPOT/ITL timeseries)
# Plotter: playground/moe_pd/plot_itl_timeseries.py
# CUDA-graph safe: all events fire on CPU between forwards. See
# vllm/distributed/kv_transfer/kv_connector/v1/nixl/_trace.py.
# Default LOG_DIR-colocated trace dir; set VLLM_PD_TRACE_DIR explicitly to
# override, or PD_TRACE=0 to disable entirely.
PD_TRACE="${PD_TRACE:-1}"
if [[ "$PD_TRACE" == "1" ]]; then
    export VLLM_PD_TRACE_DIR="${VLLM_PD_TRACE_DIR:-$LOG_DIR/pd_trace}"
    mkdir -p "$VLLM_PD_TRACE_DIR"
    echo "[start_server] pd trace dir       = $VLLM_PD_TRACE_DIR"
else
    unset VLLM_PD_TRACE_DIR
    echo "[start_server] pd trace           = DISABLED (PD_TRACE=0)"
fi

# --- Python -----------------------------------------------------------------
PY="${PY:-/root/miniconda3/envs/vllm-nvlink/bin/python}"
if [[ ! -x "$PY" ]]; then
    echo "error: python not found at $PY" >&2
    exit 1
fi

# Local NIXL-aware proxy that ships the same handshake as
# tests/v1/kv_connector/nixl_integration/toy_proxy_server.py but forwards
# the upstream Content-Type verbatim. That's the one wart that makes
# --streaming usable: the toy proxy hard-codes
# StreamingResponse(media_type="application/json"), which trips AIPerf's
# SSE parser with `InvalidInferenceResultError: No responses with actual
# content`.
PROXY_SCRIPT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/proxy.py"
if [[ ! -f "$PROXY_SCRIPT" ]]; then
    echo "error: local proxy not found at $PROXY_SCRIPT" >&2
    exit 1
fi

# Hard preflight: NIXL python pkg, plus proxy deps.
if ! "$PY" -c "from nixl._api import nixl_agent" >/dev/null 2>&1; then
    echo "error: nixl python package missing. install with:" >&2
    echo "       $PY -m pip install nixl" >&2
    exit 1
fi
if ! "$PY" -c "import fastapi, uvicorn, httpx" >/dev/null 2>&1; then
    echo "error: proxy needs fastapi, uvicorn, httpx. install with:" >&2
    echo "       $PY -m pip install fastapi uvicorn httpx" >&2
    exit 1
fi

echo "[start_server] model              = $MODEL"
echo "[start_server] served-model-name  = $SERVED_MODEL_NAME"
echo "[start_server] prefill            = GPUs $PREFILL_GPUS (DP=$PREFILL_DP) :: http $PREFILL_PORT, NIXL side base $PREFILL_SIDE_PORT"
echo "[start_server] decode             = GPUs $DECODE_GPUS (DP=$DECODE_DP) :: http $DECODE_PORT, NIXL side base $DECODE_SIDE_PORT"
echo "[start_server] proxy              = http $PROXY_PORT"
echo "[start_server] log dir            = $LOG_DIR"
echo "[start_server] cuda graphs        = ENABLED (no --enforce-eager)"
if (( EXPERT_PARALLEL )); then
    echo "[start_server] expert parallel    = ENABLED (--enable-expert-parallel on sides with DP>1)"
else
    echo "[start_server] expert parallel    = DISABLED"
fi
echo "[start_server] kv connector       = NixlConnector"
echo "[start_server] gpu-memory-util    = $GPU_MEM_UTIL"

PIDS=()

cleanup() {
    echo "[start_server] cleaning up..."
    trap - INT TERM EXIT
    for pid in "${PIDS[@]}"; do
        # PIDs here are session leaders (setsid); negate to nuke the whole
        # process group so vLLM's worker subprocesses go down too.
        kill -TERM -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 2
    for pid in "${PIDS[@]}"; do
        kill -KILL -"$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

wait_for_http() {
    local url=$1
    local label=$2
    local timeout_s=${3:-1800}
    echo "[start_server] waiting for $label ($url) ..."
    timeout "$timeout_s" bash -c "
        until curl -sf '$url' > /dev/null; do
            sleep 2
        done"
    echo "[start_server] $label is up"
}

KV_CONFIG_PRODUCER='{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'
KV_CONFIG_CONSUMER='{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'

# vLLM serve args shared by prefill and decode. Same knobs as
# moe_a2a/start_server_cudagraph.sh, minus the DP/EP flags.
COMMON_ARGS=(
    --model "$MODEL"
    --served-model-name "$SERVED_MODEL_NAME"
    --host 0.0.0.0
    --tensor-parallel-size 1
    --pipeline-parallel-size 1
    --moe-backend triton
    --disable-log-stats
    --no-enable-log-requests
    --no-enable-prefix-caching
    --attention-backend FLASHINFER
    --attention-config.use_trtllm_attention=False
    --max-num-batched-tokens 2048
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --trust-remote-code
)

# Build per-side DP/EP flag lists. With DP=1 we leave both flags off so
# behavior is byte-identical to the original 1P1D-no-EP layout.
prefill_dp_args=()
if (( PREFILL_DP > 1 )); then
    prefill_dp_args+=( --data-parallel-size "$PREFILL_DP" )
    if (( EXPERT_PARALLEL )); then
        prefill_dp_args+=( --enable-expert-parallel )
    fi
fi
decode_dp_args=()
if (( DECODE_DP > 1 )); then
    decode_dp_args+=( --data-parallel-size "$DECODE_DP" )
    if (( EXPERT_PARALLEL )); then
        decode_dp_args+=( --enable-expert-parallel )
    fi
fi

# --- Prefill instance -------------------------------------------------------
echo "[start_server] launching prefill server on GPUs $PREFILL_GPUS ..."
CUDA_VISIBLE_DEVICES="$PREFILL_GPUS" \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT="$PREFILL_SIDE_PORT" \
setsid "$PY" -m vllm.entrypoints.openai.api_server \
    "${COMMON_ARGS[@]}" \
    "${prefill_dp_args[@]}" \
    --port "$PREFILL_PORT" \
    --kv-transfer-config "$KV_CONFIG_PRODUCER" \
    >"$LOG_DIR/prefill.log" 2>&1 &
PIDS+=("$!")

# --- Decode instance --------------------------------------------------------
echo "[start_server] launching decode server on GPUs $DECODE_GPUS ..."
CUDA_VISIBLE_DEVICES="$DECODE_GPUS" \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT="$DECODE_SIDE_PORT" \
setsid "$PY" -m vllm.entrypoints.openai.api_server \
    "${COMMON_ARGS[@]}" \
    "${decode_dp_args[@]}" \
    --port "$DECODE_PORT" \
    --kv-transfer-config "$KV_CONFIG_CONSUMER" \
    >"$LOG_DIR/decode.log" 2>&1 &
PIDS+=("$!")

wait_for_http "http://localhost:$PREFILL_PORT/v1/models" prefill
wait_for_http "http://localhost:$DECODE_PORT/v1/models"  decode

# --- Proxy ------------------------------------------------------------------
# Launched after both vLLM instances are reachable; the toy proxy opens
# httpx pools to them on startup and would fail otherwise.
echo "[start_server] launching NIXL toy proxy on port $PROXY_PORT ..."
setsid "$PY" "$PROXY_SCRIPT" \
    --host 0.0.0.0 \
    --port "$PROXY_PORT" \
    --prefiller-host localhost --prefiller-port "$PREFILL_PORT" \
    --decoder-host  localhost --decoder-port  "$DECODE_PORT" \
    >"$LOG_DIR/proxy.log" 2>&1 &
PIDS+=("$!")

wait_for_http "http://localhost:$PROXY_PORT/healthcheck" proxy 120

echo "[start_server] all servers ready."
echo "[start_server] OpenAI endpoint -> http://localhost:$PROXY_PORT/v1"
echo "[start_server] tailing logs (Ctrl-C to stop everything)..."
tail -n +1 -F "$LOG_DIR/prefill.log" "$LOG_DIR/decode.log" "$LOG_DIR/proxy.log"
