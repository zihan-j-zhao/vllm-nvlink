#!/usr/bin/env bash
# Launch Qwen3-235B-A22B-FP8 as a P/D-disaggregated OpenAI-compatible
# service using vLLM's NixlConnector (NIXL/UCX KV transport), with
# DP=EP=2 on each side. Sibling of start_server_qwen3_235b_dp4.sh; the
# only meaningful differences are the FP8 default model, the DP guard
# (enforces 2 GPUs per side), and the default GPU layout.
#
#   - 1 prefill vLLM server (KV producer) on $PREFILL_GPUS (default 0,1)
#   - 1 decode  vLLM server (KV consumer) on $DECODE_GPUS  (default 2,3)
#   - 1 proxy server (playground/moe_pd/proxy.py) mediating the P->D handoff
#
# Each side runs --data-parallel-size 2 + --enable-expert-parallel and
# TP=PP=1, so EP=DP=2 per side. Total = 4 GPUs.
#
# NIXL side channel: each DP rank's scheduler binds
# VLLM_NIXL_SIDE_CHANNEL_PORT + data_parallel_index, so DP-many
# consecutive ports per side are reserved; ranges must not overlap.
#
# Env knobs mirror start_server_qwen3_235b_dp4.sh:
#   MODEL              HF model id     (default: Qwen/Qwen3-235B-A22B-Instruct-2507-FP8)
#   PROXY_PORT         Proxy port      (default: 8000)
#   PREFILL_PORT       Prefill HTTP    (default: 8100)
#   DECODE_PORT        Decode HTTP    (default: 8200)
#   PREFILL_SIDE_PORT  NIXL side ch. base for prefill DP ranks (default: 5559)
#   DECODE_SIDE_PORT   NIXL side ch. base for decode  DP ranks (default: 5570)
#   PREFILL_GPUS       Comma-separated CUDA ids for prefill (default: 0,1)
#   DECODE_GPUS        Comma-separated CUDA ids for decode  (default: 2,3)
#   EXPERT_PARALLEL    Must be "1" for this DP=EP=2 setup (default: 1)
#   SERVED_MODEL_NAME  OpenAI model id (default: basename of $MODEL)
#   GPU_MEM_UTIL       --gpu-memory-utilization per worker (default: 0.85)
#   LOG_DIR            Output dir      (default: playground/log/moe_pd/qwen3_235b_fp8_dp2/<UTC>)
#   NIXL_FAKE_READ     "1"/"0" make decode skip NIXL READs and immediately
#                      mark receives complete (default: 0; outputs invalid)
#   VLLM_PROFILE_KIND  torch|cuda|none profiler backend for /start_profile
#                      (default: torch iff VLLM_TORCH_PROFILE=1 else none)
#   VLLM_TORCH_PROFILE legacy "1"/"0" torch profiler enable (default: 0;
#                      bandwidth sweeps usually do not want the torch
#                      profiler overhead, so this defaults OFF here)
#   TORCH_PROFILE_DIR  Trace root dir   (default: playground/log/torch_profiling/<UTC>)
#   TORCH_PROFILE_RECORD_SHAPES "1"/"0" record op input shapes (default: 0)
#   TORCH_PROFILE_WITH_MEMORY   "1"/"0" record memory stats    (default: 0)
#   PY                 Python binary   (default: first existing vllm-nvlink
#                      conda env under $HOME/miniconda3, $HOME/miniforge3,
#                      then /root/miniconda3)

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SCRIPT_TAG="start_server_235b_fp8_dp2"

# --- ModelScope OFF ---------------------------------------------------------
unset VLLM_USE_MODELSCOPE LMDEPLOY_USE_MODELSCOPE MODELSCOPE_CACHE \
      MEGATRON_LM_PATH
export VLLM_USE_MODELSCOPE=False

# --- Config -----------------------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen3-235B-A22B-Instruct-2507-FP8}"
PROXY_PORT="${PROXY_PORT:-8000}"
PREFILL_PORT="${PREFILL_PORT:-8100}"
DECODE_PORT="${DECODE_PORT:-8200}"
PREFILL_SIDE_PORT="${PREFILL_SIDE_PORT:-5559}"
DECODE_SIDE_PORT="${DECODE_SIDE_PORT:-5570}"
PREFILL_GPUS="${PREFILL_GPUS:-${PREFILL_GPU:-0,1}}"
DECODE_GPUS="${DECODE_GPUS:-${DECODE_GPU:-2,3}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "$MODEL")}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
NIXL_FAKE_READ="${NIXL_FAKE_READ:-0}"
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
LOG_DIR="${LOG_DIR:-playground/log/moe_pd/qwen3_235b_fp8_dp2/$TS}"
mkdir -p "$LOG_DIR"
LOG_DIR="$(cd "$LOG_DIR" && pwd)"
VLLM_TORCH_PROFILE="${VLLM_TORCH_PROFILE:-0}"
if [[ -z "${VLLM_PROFILE_KIND:-}" ]]; then
    if [[ "$VLLM_TORCH_PROFILE" == "1" ]]; then
        VLLM_PROFILE_KIND=torch
    else
        VLLM_PROFILE_KIND=none
    fi
fi
case "$VLLM_PROFILE_KIND" in
    torch|cuda|none) ;;
    *)
        echo "error: VLLM_PROFILE_KIND must be one of: torch, cuda, none" >&2
        exit 1
        ;;
esac
TORCH_PROFILE_DIR="${TORCH_PROFILE_DIR:-playground/log/torch_profiling/$TS}"
TORCH_PROFILE_RECORD_SHAPES="${TORCH_PROFILE_RECORD_SHAPES:-0}"
TORCH_PROFILE_WITH_MEMORY="${TORCH_PROFILE_WITH_MEMORY:-0}"
if [[ "$VLLM_PROFILE_KIND" == "torch" ]]; then
    mkdir -p "$TORCH_PROFILE_DIR/prefill" "$TORCH_PROFILE_DIR/decode"
    TORCH_PROFILE_DIR="$(cd "$TORCH_PROFILE_DIR" && pwd)"
fi

_count_csv() { local IFS=','; read -r -a __arr <<< "$1"; echo "${#__arr[@]}"; }
PREFILL_DP="$(_count_csv "$PREFILL_GPUS")"
DECODE_DP="$(_count_csv "$DECODE_GPUS")"
if (( PREFILL_DP != 2 )) || (( DECODE_DP != 2 )); then
    echo "error: this Qwen3-235B-FP8 script requires DP=2 on both sides" >&2
    echo "       PREFILL_GPUS=$PREFILL_GPUS gives DP=$PREFILL_DP; expected 2 GPUs" >&2
    echo "       DECODE_GPUS=$DECODE_GPUS gives DP=$DECODE_DP; expected 2 GPUs" >&2
    exit 1
fi
EXPERT_PARALLEL="${EXPERT_PARALLEL:-1}"
if [[ "$EXPERT_PARALLEL" != "1" ]]; then
    echo "error: EXPERT_PARALLEL must be 1 for the DP=EP=2 setup" >&2
    exit 1
fi
if (( DECODE_SIDE_PORT < PREFILL_SIDE_PORT + PREFILL_DP )) && \
   (( PREFILL_SIDE_PORT < DECODE_SIDE_PORT + DECODE_DP )); then
    echo "error: NIXL side-channel ranges overlap:" >&2
    echo "       prefill uses [$PREFILL_SIDE_PORT, $((PREFILL_SIDE_PORT + PREFILL_DP - 1))]" >&2
    echo "       decode  uses [$DECODE_SIDE_PORT,  $((DECODE_SIDE_PORT  + DECODE_DP  - 1))]" >&2
    echo "       bump DECODE_SIDE_PORT past PREFILL_SIDE_PORT + PREFILL_DP" >&2
    exit 1
fi

# --- Profiler (explicitly OFF for the moe_a2a kernel profiler) --------------
unset VLLM_MOE_A2A_PROFILE VLLM_MOE_A2A_PROFILE_PATH
export VLLM_MOE_A2A_PROFILE=0

# --- P/D-disagg JSONL tracer ------------------------------------------------
PD_TRACE="${PD_TRACE:-1}"
if [[ "$PD_TRACE" == "1" ]]; then
    export VLLM_PD_TRACE_DIR="${VLLM_PD_TRACE_DIR:-$LOG_DIR/pd_trace}"
    mkdir -p "$VLLM_PD_TRACE_DIR"
    echo "[$SCRIPT_TAG] pd trace dir       = $VLLM_PD_TRACE_DIR"
else
    unset VLLM_PD_TRACE_DIR
    echo "[$SCRIPT_TAG] pd trace           = DISABLED (PD_TRACE=0)"
fi

# --- Python -----------------------------------------------------------------
if [[ -z "${PY:-}" ]]; then
    for candidate in \
        "$HOME/miniconda3/envs/vllm-nvlink/bin/python" \
        "$HOME/miniforge3/envs/vllm-nvlink/bin/python" \
        "/root/miniconda3/envs/vllm-nvlink/bin/python"; do
        if [[ -x "$candidate" ]]; then
            PY="$candidate"
            break
        fi
    done
fi
PY="${PY:-/root/miniconda3/envs/vllm-nvlink/bin/python}"
if [[ ! -x "$PY" ]]; then
    echo "error: python not found at $PY" >&2
    echo "       set PY=/path/to/vllm-nvlink/bin/python" >&2
    exit 1
fi

PROXY_SCRIPT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/proxy.py"
if [[ ! -f "$PROXY_SCRIPT" ]]; then
    echo "error: local proxy not found at $PROXY_SCRIPT" >&2
    exit 1
fi

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

echo "[$SCRIPT_TAG] model              = $MODEL"
echo "[$SCRIPT_TAG] served-model-name  = $SERVED_MODEL_NAME"
echo "[$SCRIPT_TAG] prefill            = GPUs $PREFILL_GPUS (DP=$PREFILL_DP, EP=2) :: http $PREFILL_PORT, NIXL side base $PREFILL_SIDE_PORT"
echo "[$SCRIPT_TAG] decode             = GPUs $DECODE_GPUS (DP=$DECODE_DP, EP=2) :: http $DECODE_PORT, NIXL side base $DECODE_SIDE_PORT"
echo "[$SCRIPT_TAG] proxy              = http $PROXY_PORT"
echo "[$SCRIPT_TAG] log dir            = $LOG_DIR"
echo "[$SCRIPT_TAG] profiler kind      = $VLLM_PROFILE_KIND"
if [[ "$VLLM_PROFILE_KIND" == "torch" ]]; then
    echo "[$SCRIPT_TAG] torch profile dir  = $TORCH_PROFILE_DIR"
    echo "[$SCRIPT_TAG] profile shapes     = $TORCH_PROFILE_RECORD_SHAPES"
    echo "[$SCRIPT_TAG] profile memory     = $TORCH_PROFILE_WITH_MEMORY"
elif [[ "$VLLM_PROFILE_KIND" == "cuda" ]]; then
    echo "[$SCRIPT_TAG] cuda profiler      = ENABLED (/start_profile triggers cudaProfilerStart)"
else
    echo "[$SCRIPT_TAG] vLLM profiler      = DISABLED"
fi
echo "[$SCRIPT_TAG] cuda graphs        = ENABLED (no --enforce-eager)"
echo "[$SCRIPT_TAG] expert parallel    = ENABLED (--enable-expert-parallel, DP=EP=2)"
echo "[$SCRIPT_TAG] kv connector       = NixlConnector"
if [[ "$NIXL_FAKE_READ" == "1" ]]; then
    echo "[$SCRIPT_TAG] nixl fake read     = ENABLED (decode skips READ; outputs invalid)"
else
    echo "[$SCRIPT_TAG] nixl fake read     = DISABLED"
fi
echo "[$SCRIPT_TAG] gpu-memory-util    = $GPU_MEM_UTIL"

PIDS=()
SERVER_LOGS=()

cleanup() {
    echo "[$SCRIPT_TAG] cleaning up..."
    trap - INT TERM EXIT
    for pid in "${PIDS[@]}"; do
        kill -TERM -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 2
    for pid in "${PIDS[@]}"; do
        kill -KILL -"$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

has_startup_failure() {
    local log
    for log in "${SERVER_LOGS[@]}"; do
        [[ -f "$log" ]] || continue
        if grep -Eq \
            'EngineCore failed to start|WorkerProc hit an exception|RuntimeError: Worker failed|Traceback \(most recent call last\)|Buffer overflow when allocating memory|CUDA out of memory|OutOfMemoryError' \
            "$log"; then
            echo "[$SCRIPT_TAG] detected startup failure in $log" >&2
            tail -n 80 "$log" >&2 || true
            return 0
        fi
    done
    return 1
}

wait_for_http() {
    local url=$1
    local label=$2
    local timeout_s=${3:-1800}
    local start_s now_s
    echo "[$SCRIPT_TAG] waiting for $label ($url) ..."
    start_s=$(date +%s)
    while ! curl -sf "$url" > /dev/null; do
        if has_startup_failure; then
            echo "[$SCRIPT_TAG] $label failed while waiting for $url" >&2
            exit 1
        fi
        now_s=$(date +%s)
        if (( now_s - start_s >= timeout_s )); then
            echo "[$SCRIPT_TAG] timed out waiting for $label after ${timeout_s}s" >&2
            exit 1
        fi
        sleep 2
    done
    echo "[$SCRIPT_TAG] $label is up"
}

KV_CONFIG_PRODUCER='{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'
if [[ "$NIXL_FAKE_READ" == "1" ]]; then
    KV_CONFIG_CONSUMER='{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"fake_read":true}}'
else
    KV_CONFIG_CONSUMER='{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'
fi
SERVER_LOGS=("$LOG_DIR/prefill.log" "$LOG_DIR/decode.log")

# vLLM serve args shared by prefill and decode.
COMMON_ARGS=(
    --model "$MODEL"
    --served-model-name "$SERVED_MODEL_NAME"
    --host 0.0.0.0
    --tensor-parallel-size 1
    --pipeline-parallel-size 1
    --disable-log-stats
    --no-enable-log-requests
    --moe-backend triton
    --attention-backend FLASH_ATTN
    --max-model-len 65536
    --no-enable-prefix-caching
    --attention-config.use_trtllm_attention=False
    --max-num-batched-tokens 2048
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --trust-remote-code
)

prefill_dp_args=(
    --data-parallel-size "$PREFILL_DP"
    --enable-expert-parallel
)
decode_dp_args=(
    --data-parallel-size "$DECODE_DP"
    --enable-expert-parallel
)

prefill_profiler_args=()
decode_profiler_args=()
if [[ "$VLLM_PROFILE_KIND" == "torch" ]]; then
    prefill_profiler_args+=(
        --profiler-config.profiler=torch
        --profiler-config.torch_profiler_dir="$TORCH_PROFILE_DIR/prefill"
        --profiler-config.ignore_frontend=true
        --profiler-config.active_iterations=500
        --profiler-config.torch_profiler_with_stack=false
        --profiler-config.torch_profiler_record_shapes="$TORCH_PROFILE_RECORD_SHAPES"
        --profiler-config.torch_profiler_with_memory="$TORCH_PROFILE_WITH_MEMORY"
    )
    decode_profiler_args+=(
        --profiler-config.profiler=torch
        --profiler-config.torch_profiler_dir="$TORCH_PROFILE_DIR/decode"
        --profiler-config.ignore_frontend=true
        --profiler-config.active_iterations=500
        --profiler-config.torch_profiler_with_stack=false
        --profiler-config.torch_profiler_record_shapes="$TORCH_PROFILE_RECORD_SHAPES"
        --profiler-config.torch_profiler_with_memory="$TORCH_PROFILE_WITH_MEMORY"
    )
elif [[ "$VLLM_PROFILE_KIND" == "cuda" ]]; then
    prefill_profiler_args+=(
        --profiler-config.profiler=cuda
    )
    decode_profiler_args+=(
        --profiler-config.profiler=cuda
    )
fi

# --- Optional nsys wrap around each api_server ------------------------------
# NSYS_TRACE_VLLM=1 wraps each vLLM api_server in `nsys profile` so we can
# capture per-worker kernel timelines + NVTX ranges (alongside any GPU
# metrics that run_nsys_sweep.sh also collects).
#
# Capture is gated on cudaProfilerApi: nsys passes through with negligible
# overhead until vLLM's /start_profile triggers cudaProfilerStart(), then
# /stop_profile triggers cudaProfilerStop() and nsys writes the .nsys-rep.
# This pairs with --profiler-config.profiler=cuda (we force it on below).
prefill_nsys_prefix=()
decode_nsys_prefix=()
if [[ "${NSYS_TRACE_VLLM:-0}" == "1" ]]; then
    if [[ -z "${NSYS:-}" ]]; then
        for cand in \
            "$(command -v nsys 2>/dev/null || true)" \
            /usr/local/cuda/bin/nsys \
            /opt/nvidia/nsight-systems-*/bin/nsys \
            /data/donglinbai/miniconda3/envs/verl/nsight-compute-2025.1.1/host/target-linux-x64/nsys; do
            if [[ -n "$cand" && -x "$cand" ]]; then
                NSYS="$cand"
                break
            fi
        done
    fi
    if ! "${NSYS:-nsys}" --version >/dev/null 2>&1; then
        echo "error: NSYS_TRACE_VLLM=1 but nsys not found. set NSYS=/path/to/nsys" >&2
        exit 1
    fi
    echo "[$SCRIPT_TAG] nsys trace vLLM    = ENABLED ($NSYS)"
    echo "[$SCRIPT_TAG]   capture-range    = cudaProfilerApi (call /start_profile + /stop_profile)"
    echo "[$SCRIPT_TAG]   trace            = cuda,nvtx,osrt   sample=none"
    # Force cuda profiler endpoint on (overrides VLLM_PROFILE_KIND).
    prefill_profiler_args=(--profiler-config.profiler=cuda)
    decode_profiler_args=(--profiler-config.profiler=cuda)
    NSYS_VLLM_DIR="$LOG_DIR/nsys_vllm"
    mkdir -p "$NSYS_VLLM_DIR"
    # capture-range-end:
    #   stop          : single-cycle, file written on first /stop_profile.
    #                   Subsequent /start_profile calls are IGNORED — must restart
    #                   the server (model + cudagraph compile again) for another
    #                   capture.
    #   repeat        : multi-cycle, but the .nsys-rep is only finalised when the
    #                   wrapped server exits. Intermediate data sits in /tmp as
    #                   <random>.qdstrm and must be converted with QdstrmImporter.
    #   stop-shutdown : single-cycle, signals the server to exit after first
    #                   /stop_profile. Cleanest for one-shot captures.
    NSYS_VLLM_COMMON_OPTS=(
        profile
        --trace=cuda,nvtx,osrt
        --sample=none
        --cpuctxsw=none
        --cuda-graph-trace=node
        --capture-range=cudaProfilerApi
        --capture-range-end="${NSYS_CAPTURE_END:-repeat}"
        --force-overwrite=true
        --kill=none
    )
    prefill_nsys_prefix=(
        "$NSYS"
        "${NSYS_VLLM_COMMON_OPTS[@]}"
        --output="$NSYS_VLLM_DIR/prefill_%p"
    )
    decode_nsys_prefix=(
        "$NSYS"
        "${NSYS_VLLM_COMMON_OPTS[@]}"
        --output="$NSYS_VLLM_DIR/decode_%p"
    )
else
    echo "[$SCRIPT_TAG] nsys trace vLLM    = DISABLED (set NSYS_TRACE_VLLM=1)"
fi

# --- Prefill instance -------------------------------------------------------
echo "[$SCRIPT_TAG] launching prefill server on GPUs $PREFILL_GPUS ..."
CUDA_VISIBLE_DEVICES="$PREFILL_GPUS" \
UCX_NET_DEVICES="${UCX_NET_DEVICES:-all}" \
UCX_TLS="${UCX_TLS:-all}" \
VLLM_NIXL_SIDE_CHANNEL_PORT="$PREFILL_SIDE_PORT" \
setsid "${prefill_nsys_prefix[@]}" "$PY" -m vllm.entrypoints.openai.api_server \
    "${COMMON_ARGS[@]}" \
    "${prefill_dp_args[@]}" \
    "${prefill_profiler_args[@]}" \
    --port "$PREFILL_PORT" \
    --kv-transfer-config "$KV_CONFIG_PRODUCER" \
    >"$LOG_DIR/prefill.log" 2>&1 &
PIDS+=("$!")

# --- Decode instance --------------------------------------------------------
echo "[$SCRIPT_TAG] launching decode server on GPUs $DECODE_GPUS ..."
CUDA_VISIBLE_DEVICES="$DECODE_GPUS" \
UCX_NET_DEVICES="${UCX_NET_DEVICES:-all}" \
UCX_TLS="${UCX_TLS:-all}" \
VLLM_NIXL_SIDE_CHANNEL_PORT="$DECODE_SIDE_PORT" \
setsid "${decode_nsys_prefix[@]}" "$PY" -m vllm.entrypoints.openai.api_server \
    "${COMMON_ARGS[@]}" \
    "${decode_dp_args[@]}" \
    "${decode_profiler_args[@]}" \
    --port "$DECODE_PORT" \
    --kv-transfer-config "$KV_CONFIG_CONSUMER" \
    >"$LOG_DIR/decode.log" 2>&1 &
PIDS+=("$!")

wait_for_http "http://localhost:$PREFILL_PORT/v1/models" prefill
wait_for_http "http://localhost:$DECODE_PORT/v1/models"  decode

# --- Proxy ------------------------------------------------------------------
echo "[$SCRIPT_TAG] launching NIXL toy proxy on port $PROXY_PORT ..."
setsid "$PY" "$PROXY_SCRIPT" \
    --host 0.0.0.0 \
    --port "$PROXY_PORT" \
    --prefiller-host localhost --prefiller-port "$PREFILL_PORT" \
    --decoder-host  localhost --decoder-port  "$DECODE_PORT" \
    >"$LOG_DIR/proxy.log" 2>&1 &
PIDS+=("$!")

wait_for_http "http://localhost:$PROXY_PORT/healthcheck" proxy 120

# Record the topology so the sweep driver can attribute samples to
# "prefill" vs "decode" GPUs without re-reading this script.
TOPO_FILE="$LOG_DIR/topology.json"
cat >"$TOPO_FILE" <<EOF
{
  "model": "$MODEL",
  "served_model_name": "$SERVED_MODEL_NAME",
  "prefill_gpus": [$(echo "$PREFILL_GPUS" | sed 's/,/, /g')],
  "decode_gpus":  [$(echo "$DECODE_GPUS"  | sed 's/,/, /g')],
  "prefill_dp": $PREFILL_DP,
  "decode_dp":  $DECODE_DP,
  "expert_parallel": $EXPERT_PARALLEL,
  "proxy_port":   $PROXY_PORT,
  "prefill_port": $PREFILL_PORT,
  "decode_port":  $DECODE_PORT
}
EOF
echo "[$SCRIPT_TAG] topology written to $TOPO_FILE"

echo "[$SCRIPT_TAG] all servers ready."
echo "[$SCRIPT_TAG] OpenAI endpoint -> http://localhost:$PROXY_PORT/v1"
echo "[$SCRIPT_TAG] tailing logs (Ctrl-C to stop everything)..."
tail -n +1 -F "$LOG_DIR/prefill.log" "$LOG_DIR/decode.log" "$LOG_DIR/proxy.log"
