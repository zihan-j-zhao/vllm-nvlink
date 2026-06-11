#!/usr/bin/env bash
# Launch Qwen3-30B-A3B as a P/D-disaggregated OpenAI-compatible service using
# vLLM's LMCacheConnectorV1 (LMCache async PD backend over NIXL/UCX), as an
# alternative to start_server.sh's NixlConnector setup.
#
#   - 1 prefill vLLM server (KV producer / LMCache "sender")  on $PREFILL_GPUS
#   - 1 decode  vLLM server (KV consumer / LMCache "receiver") on $DECODE_GPUS
#   - 1 PD-aware proxy (playground/moe_pd/lmcache_proxy.py, vendored from
#     LMCache v0.4.6 examples/disagg_prefill/disagg_proxy_server.py) that
#     tokenizes the prompt, injects a `disagg_spec` (telling the prefiller
#     where the decoder's NIXL receiver listens), runs a 1-token prefill,
#     appends the first decoded token to the prompt, then streams the decode.
#     The prefiller notifies the proxy over ZMQ when KV transfer is done.
#
# Unlike NixlConnector (where the proxy shuttles `kv_transfer_params` blobs
# between prefill and decode), LMCache moves KV out-of-band over its own
# NIXL channel: the decoder (receiver) binds init/alloc ports, the prefiller
# (sender) connects to them and to the proxy's ZMQ PULL port.
#
# SCOPE: This launcher targets the verified LMCache 1P1D layout (single GPU
# per side, TP=DP=1). LMCache's async PD backend does not cleanly support
# per-engine data parallelism in this version (the vLLM adapter carries a
# "TODO dp > 1" and the lmcache_rpc_port / NIXL ports would collide across DP
# ranks), so --data-parallel-size / --enable-expert-parallel are intentionally
# NOT wired up here. Use start_server.sh (NixlConnector) for the DP-EP=2 sweep.
#
# Env knobs:
#   MODEL              HF model id     (default: Qwen/Qwen3-30B-A3B-Instruct-2507)
#   SERVED_MODEL_NAME  OpenAI model id (default: basename of $MODEL)
#   PROXY_PORT         Proxy HTTP port (default: 8000)
#   PREFILL_PORT       Prefill HTTP    (default: 8100)
#   DECODE_PORT        Decode HTTP     (default: 8200)
#   DECODE_INIT_PORT   Decoder NIXL init port  (default: 8300)
#   DECODE_ALLOC_PORT  Decoder NIXL alloc port (default: 8400)
#   PROXY_ZMQ_PORT     Proxy ZMQ PULL port for sender ProxyNotif (default: 8500)
#   PREFILL_GPUS       Single CUDA id for prefill (default: 6)
#   DECODE_GPUS        Single CUDA id for decode  (default: 7)
#   GPU_MEM_UTIL       --gpu-memory-utilization per worker (default: 0.85)
#   MAX_NUM_SEQS       --max-num-seqs per engine (default: 2048)
#   ENFORCE_EAGER      "1"/"0" pass --enforce-eager (default: 1; matches the
#                      LMCache reference example, which is the known-good path)
#   PD_BUFFER_SIZE     Decoder PD buffer bytes; also drives the proxy's
#                      in-flight slot semaphore (default: 2147483648 = 2GiB)
#   PREFILL_PD_BUFFER_SIZE  Prefiller PD buffer bytes (default: 2147483648 = 2GiB).
#                      Must hold pd_max_prefill_len tokens of KV: for
#                      Qwen3-30B (~98KiB/token) 16384 tokens needs ~1.5GiB, so
#                      the 2GiB default fits. Bump if you raise pd_max_prefill_len.
#   CHUNK_SIZE         LMCache chunk size in tokens (default: 256)
#   PYTHONHASHSEED     Hash seed shared by prefill+decode so KV chunk keys
#                      match (default: 123)
#   LOG_DIR            Output dir (default: playground/log/moe_pd_lmcache/<UTC>)
#   PY                 Python binary (default: first existing vllm-nvlink env)

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# --- ModelScope OFF ---------------------------------------------------------
unset VLLM_USE_MODELSCOPE LMDEPLOY_USE_MODELSCOPE MODELSCOPE_CACHE \
      MEGATRON_LM_PATH
export VLLM_USE_MODELSCOPE=False

# --- Config -----------------------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "$MODEL")}"
PROXY_PORT="${PROXY_PORT:-8000}"
PREFILL_PORT="${PREFILL_PORT:-8100}"
DECODE_PORT="${DECODE_PORT:-8200}"
DECODE_INIT_PORT="${DECODE_INIT_PORT:-8300}"
DECODE_ALLOC_PORT="${DECODE_ALLOC_PORT:-8400}"
PROXY_ZMQ_PORT="${PROXY_ZMQ_PORT:-8500}"
PREFILL_GPUS="${PREFILL_GPUS:-${PREFILL_GPU:-6}}"
DECODE_GPUS="${DECODE_GPUS:-${DECODE_GPU:-7}}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2048}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
PD_BUFFER_SIZE="${PD_BUFFER_SIZE:-2147483648}"
PREFILL_PD_BUFFER_SIZE="${PREFILL_PD_BUFFER_SIZE:-2147483648}"
CHUNK_SIZE="${CHUNK_SIZE:-256}"
PYTHONHASHSEED="${PYTHONHASHSEED:-123}"
export PYTHONHASHSEED
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
LOG_DIR="${LOG_DIR:-playground/log/moe_pd_lmcache/$TS}"
mkdir -p "$LOG_DIR"
LOG_DIR="$(cd "$LOG_DIR" && pwd)"

# LMCache PD moves KV out of band; prefix caching must stay off so every
# request actually round-trips through the prefill->decode path.
NO_PREFIX_CACHING=1

# --- Python -----------------------------------------------------------------
if [[ -z "${PY:-}" ]]; then
    for candidate in \
        "/opt/miniconda/envs/vllm-nvlink/bin/python" \
        "$HOME/miniconda3/envs/vllm-nvlink/bin/python" \
        "$HOME/miniforge3/envs/vllm-nvlink/bin/python" \
        "/root/miniconda3/envs/vllm-nvlink/bin/python"; do
        if [[ -x "$candidate" ]]; then
            PY="$candidate"
            break
        fi
    done
fi
PY="${PY:-/opt/miniconda/envs/vllm-nvlink/bin/python}"
if [[ ! -x "$PY" ]]; then
    echo "error: python not found at $PY" >&2
    echo "       set PY=/path/to/vllm-nvlink/bin/python" >&2
    exit 1
fi

PROXY_SCRIPT="$HERE/lmcache_proxy.py"
if [[ ! -f "$PROXY_SCRIPT" ]]; then
    echo "error: lmcache proxy not found at $PROXY_SCRIPT" >&2
    exit 1
fi

# --- Preflight: lmcache, nixl, proxy deps -----------------------------------
if ! "$PY" -c "import lmcache" >/dev/null 2>&1; then
    echo "error: lmcache python package missing. install with:" >&2
    echo "       $PY -m pip install lmcache" >&2
    exit 1
fi
if ! "$PY" -c "from nixl._api import nixl_agent" >/dev/null 2>&1; then
    echo "error: nixl python package missing. install with:" >&2
    echo "       $PY -m pip install nixl" >&2
    exit 1
fi
if ! "$PY" -c "import fastapi, uvicorn, httpx, zmq, msgspec, numpy, transformers" \
        >/dev/null 2>&1; then
    echo "error: proxy needs fastapi uvicorn httpx pyzmq msgspec numpy transformers" >&2
    exit 1
fi

# --- LMCache config files ---------------------------------------------------
PREFILL_CFG="$LOG_DIR/lmcache-prefiller-config.yaml"
DECODE_CFG="$LOG_DIR/lmcache-decoder-config.yaml"

cat >"$PREFILL_CFG" <<EOF
# Auto-generated by start_server_lmcache.sh — prefiller (LMCache PD sender).
local_cpu: False
enable_pd: True
transfer_channel: "nixl"
pd_role: "sender"
pd_proxy_host: "localhost"
pd_proxy_port: $PROXY_ZMQ_PORT
pd_buffer_size: $PREFILL_PD_BUFFER_SIZE
pd_buffer_device: "cuda"
nixl_backends: [UCX]
pd_backend_mode: "async"
pd_max_prefill_len: 16384
pd_allocation_timeout_sec: 10
pd_shutdown_timeout_sec: 5
pd_condition_poll_interval_sec: 0.05
chunk_size: $CHUNK_SIZE
EOF

cat >"$DECODE_CFG" <<EOF
# Auto-generated by start_server_lmcache.sh — decoder (LMCache PD receiver).
local_cpu: False
enable_pd: True
transfer_channel: "nixl"
pd_role: "receiver"
pd_peer_host: "localhost"
pd_peer_init_port: $DECODE_INIT_PORT
pd_peer_alloc_port: $DECODE_ALLOC_PORT
pd_buffer_size: $PD_BUFFER_SIZE
pd_buffer_device: "cuda"
nixl_backends: [UCX]
pd_backend_mode: "async"
pd_max_prefill_len: 16384
pd_allocation_timeout_sec: 10
pd_shutdown_timeout_sec: 5
pd_condition_poll_interval_sec: 0.05
chunk_size: $CHUNK_SIZE
EOF

echo "[start_server_lmcache] model              = $MODEL"
echo "[start_server_lmcache] served-model-name  = $SERVED_MODEL_NAME"
echo "[start_server_lmcache] prefill (sender)   = GPU $PREFILL_GPUS :: http $PREFILL_PORT"
echo "[start_server_lmcache] decode  (receiver) = GPU $DECODE_GPUS :: http $DECODE_PORT, init $DECODE_INIT_PORT, alloc $DECODE_ALLOC_PORT"
echo "[start_server_lmcache] proxy              = http $PROXY_PORT, zmq $PROXY_ZMQ_PORT"
echo "[start_server_lmcache] kv connector       = LMCacheConnectorV1 (async PD over NIXL/UCX)"
echo "[start_server_lmcache] enforce eager      = $ENFORCE_EAGER"
echo "[start_server_lmcache] pd buffer (dec/pre)= $PD_BUFFER_SIZE / $PREFILL_PD_BUFFER_SIZE bytes"
echo "[start_server_lmcache] chunk size         = $CHUNK_SIZE tokens"
echo "[start_server_lmcache] PYTHONHASHSEED     = $PYTHONHASHSEED"
echo "[start_server_lmcache] log dir            = $LOG_DIR"
echo "[start_server_lmcache] prefiller cfg      = $PREFILL_CFG"
echo "[start_server_lmcache] decoder cfg        = $DECODE_CFG"

PIDS=()

cleanup() {
    echo "[start_server_lmcache] cleaning up..."
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

wait_for_http() {
    local url=$1
    local label=$2
    local timeout_s=${3:-1800}
    echo "[start_server_lmcache] waiting for $label ($url) ..."
    timeout "$timeout_s" bash -c "
        until curl -sf '$url' > /dev/null; do
            sleep 2
        done"
    echo "[start_server_lmcache] $label is up"
}

KV_CONFIG_PRODUCER='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer","kv_connector_extra_config":{"discard_partial_chunks":false,"lmcache_rpc_port":"producer1"}}'
KV_CONFIG_CONSUMER='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config":{"discard_partial_chunks":false,"lmcache_rpc_port":"consumer1","skip_last_n_tokens":1}}'

# vLLM serve args shared by prefill and decode. Aligned with start_server.sh's
# COMMON_ARGS where it doesn't conflict with the LMCache PD requirements.
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
    --max-num-seqs "$MAX_NUM_SEQS"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --trust-remote-code
)
if [[ "$ENFORCE_EAGER" == "1" ]]; then
    COMMON_ARGS+=( --enforce-eager )
fi

# Shared env for both vLLM instances. UCX_TLS pins NVLink/intra-node transports
# (mirrors the LMCache example and the repo's NIXL notes).
LMCACHE_ENV=(
    UCX_TLS=cuda_ipc,cuda_copy,tcp
    LMCACHE_USE_EXPERIMENTAL=True
    VLLM_ENABLE_V1_MULTIPROCESSING=1
    VLLM_WORKER_MULTIPROC_METHOD=spawn
    PYTHONHASHSEED="$PYTHONHASHSEED"
)

# --- Proxy ------------------------------------------------------------------
# Launch first so the ZMQ PULL port is bound before the prefiller's sender
# tries to notify it (matches the LMCache reference launcher order).
echo "[start_server_lmcache] launching LMCache PD proxy on port $PROXY_PORT ..."
setsid "$PY" "$PROXY_SCRIPT" \
    --host 0.0.0.0 \
    --port "$PROXY_PORT" \
    --prefiller-host localhost --prefiller-port "$PREFILL_PORT" \
    --num-prefillers 1 \
    --decoder-host localhost --decoder-port "$DECODE_PORT" \
    --decoder-init-port "$DECODE_INIT_PORT" \
    --decoder-alloc-port "$DECODE_ALLOC_PORT" \
    --num-decoders 1 \
    --proxy-host localhost --proxy-port "$PROXY_ZMQ_PORT" \
    --model "$MODEL" \
    --pd-buffer-size "$PD_BUFFER_SIZE" \
    --chunk-size "$CHUNK_SIZE" \
    >"$LOG_DIR/proxy.log" 2>&1 &
PIDS+=("$!")

# --- Decode instance (receiver) ---------------------------------------------
echo "[start_server_lmcache] launching decode server on GPU $DECODE_GPUS ..."
CUDA_VISIBLE_DEVICES="$DECODE_GPUS" \
LMCACHE_CONFIG_FILE="$DECODE_CFG" \
env "${LMCACHE_ENV[@]}" \
setsid "$PY" -m vllm.entrypoints.openai.api_server \
    "${COMMON_ARGS[@]}" \
    --port "$DECODE_PORT" \
    --kv-transfer-config "$KV_CONFIG_CONSUMER" \
    >"$LOG_DIR/decode.log" 2>&1 &
PIDS+=("$!")

# --- Prefill instance (sender) ----------------------------------------------
echo "[start_server_lmcache] launching prefill server on GPU $PREFILL_GPUS ..."
CUDA_VISIBLE_DEVICES="$PREFILL_GPUS" \
LMCACHE_CONFIG_FILE="$PREFILL_CFG" \
env "${LMCACHE_ENV[@]}" \
setsid "$PY" -m vllm.entrypoints.openai.api_server \
    "${COMMON_ARGS[@]}" \
    --port "$PREFILL_PORT" \
    --kv-transfer-config "$KV_CONFIG_PRODUCER" \
    >"$LOG_DIR/prefill.log" 2>&1 &
PIDS+=("$!")

wait_for_http "http://localhost:$PREFILL_PORT/v1/models" prefill
wait_for_http "http://localhost:$DECODE_PORT/v1/models"  decode
wait_for_http "http://localhost:$PROXY_PORT/v1/models"   proxy 120

echo "[start_server_lmcache] all servers ready."
echo "[start_server_lmcache] OpenAI endpoint -> http://localhost:$PROXY_PORT/v1"
echo "[start_server_lmcache] tailing logs (Ctrl-C to stop everything)..."
tail -n +1 -F "$LOG_DIR/prefill.log" "$LOG_DIR/decode.log" "$LOG_DIR/proxy.log"
