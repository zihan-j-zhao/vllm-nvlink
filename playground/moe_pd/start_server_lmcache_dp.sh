#!/usr/bin/env bash
# Launch Qwen3-30B-A3B as a P/D-disaggregated, DP=EP=2 service using vLLM's
# LMCacheConnectorV1 (LMCache async PD backend over NIXL/UCX) on 4 GPUs.
#
# Topology (mirrors start_server.sh's NixlConnector DP-EP=2 layout, but with
# the LMCache connector):
#   - Prefill side: a DP=2 + expert-parallel group of 2 single-GPU vLLM
#     processes (ranks 0,1) on $PREFILL_GPUS, each a KV producer / LMCache
#     "sender".
#   - Decode side:  a DP=2 + expert-parallel group of 2 single-GPU vLLM
#     processes (ranks 0,1) on $DECODE_GPUS, each a KV consumer / LMCache
#     "receiver".
#   - 1 PD-aware proxy (playground/moe_pd/lmcache_proxy.py) that round-robins
#     across the 2 prefillers / 2 decoders (xPyD), injects the disagg_spec
#     (which decoder's NIXL receiver ports to push KV to), and waits for the
#     prefiller's ZMQ ProxyNotif.
#
# WHY external-LB DP (not in-process --data-parallel-size in one server):
#   LMCache 0.4.6 indexes its PD receiver ports by the worker's TP rank
#   (metadata.worker_id == parallel_config.rank), which is 0 for every
#   in-process DP rank when TP=1. Two in-process DP ranks would therefore both
#   bind pd_peer_init_port[0] and collide. Running each DP rank as its OWN
#   process (vLLM external load-balancer mode, enabled by --data-parallel-rank)
#   gives every rank its own env / LMCACHE_CONFIG_FILE / ports while still
#   sharing ONE expert-parallel group via --data-parallel-address +
#   --data-parallel-rpc-port. The external LB is our lmcache_proxy.py.
#
# Port map (defaults):
#   proxy HTTP             8000
#   proxy ZMQ (ProxyNotif) 8500
#   prefill rank r HTTP    8100 + r          (8100, 8101)
#   decode  rank r HTTP    8200 + r          (8200, 8201)
#   decode  rank r init    8300 + r          (8300, 8301)   NIXL PD init
#   decode  rank r alloc   8400 + r          (8400, 8401)   NIXL PD alloc
#   prefill DP rpc         9100              (coordinator for prefill EP group)
#   decode  DP rpc         9200              (coordinator for decode  EP group)
#
# Env knobs:
#   MODEL              HF model id     (default: Qwen/Qwen3-30B-A3B-Instruct-2507)
#   SERVED_MODEL_NAME  OpenAI model id (default: basename of $MODEL)
#   PREFILL_GPUS       Comma CUDA ids for prefill DP ranks (default: 4,5)
#   DECODE_GPUS        Comma CUDA ids for decode  DP ranks (default: 6,7)
#   PROXY_PORT/PROXY_ZMQ_PORT/PREFILL_PORT/DECODE_PORT/DECODE_INIT_PORT/
#   DECODE_ALLOC_PORT/PREFILL_DP_RPC/DECODE_DP_RPC  port bases (see above)
#   GPU_MEM_UTIL       --gpu-memory-utilization per worker (default: 0.85)
#   MAX_NUM_SEQS       --max-num-seqs per engine (default: 2048)
#   ENFORCE_EAGER      "1"/"0" pass --enforce-eager (default: 1)
#   PD_BUFFER_SIZE          Decoder PD buffer bytes (default 2147483648 = 2GiB)
#   PREFILL_PD_BUFFER_SIZE  Prefiller PD buffer bytes (default 2147483648 = 2GiB)
#   CHUNK_SIZE         LMCache chunk size in tokens (default: 256)
#   PYTHONHASHSEED     Shared hash seed (default: 123)
#   LOG_DIR            Output dir (default: playground/log/moe_pd_lmcache_dp/<UTC>)
#   PY                 Python binary (default: first existing vllm-nvlink env)

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_TAG="start_server_lmcache_dp"

unset VLLM_USE_MODELSCOPE LMDEPLOY_USE_MODELSCOPE MODELSCOPE_CACHE MEGATRON_LM_PATH
export VLLM_USE_MODELSCOPE=False

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "$MODEL")}"
PREFILL_GPUS="${PREFILL_GPUS:-4,5}"
DECODE_GPUS="${DECODE_GPUS:-6,7}"
PROXY_PORT="${PROXY_PORT:-8000}"
PROXY_ZMQ_PORT="${PROXY_ZMQ_PORT:-8500}"
PREFILL_PORT="${PREFILL_PORT:-8100}"
DECODE_PORT="${DECODE_PORT:-8200}"
DECODE_INIT_PORT="${DECODE_INIT_PORT:-8300}"
DECODE_ALLOC_PORT="${DECODE_ALLOC_PORT:-8400}"
PREFILL_DP_RPC="${PREFILL_DP_RPC:-9100}"
DECODE_DP_RPC="${DECODE_DP_RPC:-9200}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2048}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
PD_BUFFER_SIZE="${PD_BUFFER_SIZE:-2147483648}"
PREFILL_PD_BUFFER_SIZE="${PREFILL_PD_BUFFER_SIZE:-2147483648}"
CHUNK_SIZE="${CHUNK_SIZE:-256}"
PYTHONHASHSEED="${PYTHONHASHSEED:-123}"
export PYTHONHASHSEED
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
LOG_DIR="${LOG_DIR:-playground/log/moe_pd_lmcache_dp/$TS}"
mkdir -p "$LOG_DIR"
LOG_DIR="$(cd "$LOG_DIR" && pwd)"

IFS=',' read -r -a PREFILL_GPU_ARR <<< "$PREFILL_GPUS"
IFS=',' read -r -a DECODE_GPU_ARR <<< "$DECODE_GPUS"
PREFILL_DP="${#PREFILL_GPU_ARR[@]}"
DECODE_DP="${#DECODE_GPU_ARR[@]}"
if (( PREFILL_DP < 1 || DECODE_DP < 1 )); then
    echo "error: PREFILL_GPUS / DECODE_GPUS must list at least one GPU" >&2
    exit 1
fi

# --- Python -----------------------------------------------------------------
if [[ -z "${PY:-}" ]]; then
    for candidate in \
        "/opt/miniconda/envs/vllm-nvlink/bin/python" \
        "$HOME/miniconda3/envs/vllm-nvlink/bin/python" \
        "$HOME/miniforge3/envs/vllm-nvlink/bin/python" \
        "/root/miniconda3/envs/vllm-nvlink/bin/python"; do
        [[ -x "$candidate" ]] && { PY="$candidate"; break; }
    done
fi
PY="${PY:-/opt/miniconda/envs/vllm-nvlink/bin/python}"
if [[ ! -x "$PY" ]]; then
    echo "error: python not found at $PY (set PY=...)" >&2
    exit 1
fi

PROXY_SCRIPT="$HERE/lmcache_proxy.py"
[[ -f "$PROXY_SCRIPT" ]] || { echo "error: $PROXY_SCRIPT missing" >&2; exit 1; }

# --- Preflight --------------------------------------------------------------
if ! "$PY" -c "import lmcache" >/dev/null 2>&1; then
    echo "error: lmcache missing ($PY -m pip install lmcache)" >&2; exit 1
fi
if ! "$PY" -c "from nixl._api import nixl_agent" >/dev/null 2>&1; then
    echo "error: nixl missing ($PY -m pip install nixl)" >&2; exit 1
fi
if ! "$PY" -c "import fastapi, uvicorn, httpx, zmq, msgspec, numpy, transformers" \
        >/dev/null 2>&1; then
    echo "error: proxy needs fastapi uvicorn httpx pyzmq msgspec numpy transformers" >&2
    exit 1
fi

# --- LMCache config files ---------------------------------------------------
# Prefiller (sender) config is identical for all prefill ranks (sender only
# needs the proxy ZMQ endpoint; it connects out to each decoder's ports per
# request via the disagg_spec the proxy injects).
PREFILL_CFG="$LOG_DIR/lmcache-prefiller-config.yaml"
cat >"$PREFILL_CFG" <<EOF
# Auto-generated: LMCache PD sender (prefiller), shared by all prefill DP ranks.
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

# Decoder (receiver) config is per-rank: each binds distinct NIXL init/alloc
# ports (base + rank), matching the proxy's incremental decoder port scheme.
declare -a DECODE_CFG_ARR=()
for ((r=0; r<DECODE_DP; r++)); do
    cfg="$LOG_DIR/lmcache-decoder-rank${r}-config.yaml"
    cat >"$cfg" <<EOF
# Auto-generated: LMCache PD receiver (decoder) rank $r.
local_cpu: False
enable_pd: True
transfer_channel: "nixl"
pd_role: "receiver"
pd_peer_host: "localhost"
pd_peer_init_port: $((DECODE_INIT_PORT + r))
pd_peer_alloc_port: $((DECODE_ALLOC_PORT + r))
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
    DECODE_CFG_ARR+=("$cfg")
done

echo "[$SCRIPT_TAG] model              = $MODEL"
echo "[$SCRIPT_TAG] prefill (senders)  = GPUs $PREFILL_GPUS (DP=EP=$PREFILL_DP) http $PREFILL_PORT.. dp-rpc $PREFILL_DP_RPC"
echo "[$SCRIPT_TAG] decode  (receivers)= GPUs $DECODE_GPUS (DP=EP=$DECODE_DP) http $DECODE_PORT.. init $DECODE_INIT_PORT.. alloc $DECODE_ALLOC_PORT.. dp-rpc $DECODE_DP_RPC"
echo "[$SCRIPT_TAG] proxy              = http $PROXY_PORT, zmq $PROXY_ZMQ_PORT"
echo "[$SCRIPT_TAG] kv connector       = LMCacheConnectorV1 (async PD over NIXL/UCX)"
echo "[$SCRIPT_TAG] enforce eager      = $ENFORCE_EAGER"
echo "[$SCRIPT_TAG] log dir            = $LOG_DIR"

PIDS=()
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

wait_for_http() {
    local url=$1 label=$2 timeout_s=${3:-1800}
    echo "[$SCRIPT_TAG] waiting for $label ($url) ..."
    timeout "$timeout_s" bash -c "until curl -sf '$url' > /dev/null; do sleep 2; done"
    echo "[$SCRIPT_TAG] $label is up"
}

KV_PRODUCER_TMPL='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer","kv_connector_extra_config":{"discard_partial_chunks":false,"lmcache_rpc_port":"RPCID"}}'
KV_CONSUMER_TMPL='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config":{"discard_partial_chunks":false,"lmcache_rpc_port":"RPCID","skip_last_n_tokens":1}}'

COMMON_ARGS=(
    --model "$MODEL"
    --served-model-name "$SERVED_MODEL_NAME"
    --host 0.0.0.0
    --tensor-parallel-size 1
    --pipeline-parallel-size 1
    --enable-expert-parallel
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
[[ "$ENFORCE_EAGER" == "1" ]] && COMMON_ARGS+=( --enforce-eager )

BASE_ENV=(
    UCX_TLS=cuda_ipc,cuda_copy,tcp
    LMCACHE_USE_EXPERIMENTAL=True
    VLLM_ENABLE_V1_MULTIPROCESSING=1
    VLLM_WORKER_MULTIPROC_METHOD=spawn
    PYTHONHASHSEED="$PYTHONHASHSEED"
)

# --- Proxy (start first so its ZMQ PULL is bound for sender ProxyNotif) ------
echo "[$SCRIPT_TAG] launching LMCache PD proxy ..."
setsid "$PY" "$PROXY_SCRIPT" \
    --host 0.0.0.0 --port "$PROXY_PORT" \
    --prefiller-host localhost --prefiller-port "$PREFILL_PORT" --num-prefillers "$PREFILL_DP" \
    --decoder-host localhost --decoder-port "$DECODE_PORT" --num-decoders "$DECODE_DP" \
    --decoder-init-port "$DECODE_INIT_PORT" --decoder-alloc-port "$DECODE_ALLOC_PORT" \
    --proxy-host localhost --proxy-port "$PROXY_ZMQ_PORT" \
    --model "$MODEL" --pd-buffer-size "$PD_BUFFER_SIZE" --chunk-size "$CHUNK_SIZE" \
    >"$LOG_DIR/proxy.log" 2>&1 &
PIDS+=("$!")

# --- Launch a DP/EP group of single-GPU ranks (external-LB) -----------------
# launch_rank <role> <rank> <gpu> <http_port> <dp_size> <dp_rpc> <kv_config> <lmcache_cfg> <logfile>
launch_rank() {
    local role=$1 rank=$2 gpu=$3 http=$4 dp=$5 rpc=$6 kvcfg=$7 lmcfg=$8 logf=$9
    echo "[$SCRIPT_TAG] launching $role rank $rank on GPU $gpu (http $http) ..."
    CUDA_VISIBLE_DEVICES="$gpu" \
    LMCACHE_CONFIG_FILE="$lmcfg" \
    env "${BASE_ENV[@]}" \
    setsid "$PY" -m vllm.entrypoints.openai.api_server \
        "${COMMON_ARGS[@]}" \
        --port "$http" \
        --data-parallel-size "$dp" \
        --data-parallel-rank "$rank" \
        --data-parallel-address 127.0.0.1 \
        --data-parallel-rpc-port "$rpc" \
        --kv-transfer-config "$kvcfg" \
        >"$logf" 2>&1 &
    PIDS+=("$!")
}

# Decoders (receivers) first so their NIXL init/alloc ports are listening
# before any prefiller tries to push KV.
for ((r=0; r<DECODE_DP; r++)); do
    kv="${KV_CONSUMER_TMPL/RPCID/consumer$r}"
    launch_rank decode "$r" "${DECODE_GPU_ARR[$r]}" "$((DECODE_PORT + r))" \
        "$DECODE_DP" "$DECODE_DP_RPC" "$kv" "${DECODE_CFG_ARR[$r]}" \
        "$LOG_DIR/decode_rank${r}.log"
    sleep 5
done

# Prefillers (senders).
for ((r=0; r<PREFILL_DP; r++)); do
    kv="${KV_PRODUCER_TMPL/RPCID/producer$r}"
    launch_rank prefill "$r" "${PREFILL_GPU_ARR[$r]}" "$((PREFILL_PORT + r))" \
        "$PREFILL_DP" "$PREFILL_DP_RPC" "$kv" "$PREFILL_CFG" \
        "$LOG_DIR/prefill_rank${r}.log"
    sleep 5
done

# --- Readiness --------------------------------------------------------------
for ((r=0; r<PREFILL_DP; r++)); do
    wait_for_http "http://localhost:$((PREFILL_PORT + r))/v1/models" "prefill rank $r"
done
for ((r=0; r<DECODE_DP; r++)); do
    wait_for_http "http://localhost:$((DECODE_PORT + r))/v1/models" "decode rank $r"
done
wait_for_http "http://localhost:$PROXY_PORT/v1/models" proxy 120

# --- topology.json (for run_nsys_sweep.sh + postprocess_nsys.py) ------------
TOPO_FILE="$LOG_DIR/topology.json"
cat >"$TOPO_FILE" <<EOF
{
  "model": "$MODEL",
  "served_model_name": "$SERVED_MODEL_NAME",
  "prefill_gpus": [$(echo "$PREFILL_GPUS" | sed 's/,/, /g')],
  "decode_gpus":  [$(echo "$DECODE_GPUS"  | sed 's/,/, /g')],
  "prefill_dp": $PREFILL_DP,
  "decode_dp":  $DECODE_DP,
  "expert_parallel": 1,
  "kv_connector": "LMCacheConnectorV1",
  "proxy_port":   $PROXY_PORT,
  "prefill_port": $PREFILL_PORT,
  "decode_port":  $DECODE_PORT
}
EOF
echo "[$SCRIPT_TAG] topology written to $TOPO_FILE"

echo "[$SCRIPT_TAG] all servers ready."
echo "[$SCRIPT_TAG] OpenAI endpoint -> http://localhost:$PROXY_PORT/v1"
echo "[$SCRIPT_TAG] tailing logs (Ctrl-C to stop everything)..."
tail -n +1 -F "$LOG_DIR"/proxy.log "$LOG_DIR"/prefill_rank*.log "$LOG_DIR"/decode_rank*.log
