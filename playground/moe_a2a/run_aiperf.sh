#!/usr/bin/env bash
# Drive the running vLLM server with AIPerf:
#   * ShareGPT public dataset (downloaded automatically by AIPerf the first
#     time; cached under ~/.cache/huggingface for subsequent runs).
#   * 1000 requests total (--request-count 1000).
#   * Poisson arrivals at 20 req/s (--arrival-pattern poisson --request-rate 20).
#   * Shuffle sampling with a fixed seed (1000 unique-ish prompts).
#
# Env knobs:
#   URL                  Server base URL  (default: http://localhost:8000)
#   SERVED_MODEL_NAME    Model id served by the server
#                        (default: Qwen3-30B-A3B-Instruct-2507)
#   TOKENIZER            HF tokenizer id   (default: same as SERVED_MODEL_NAME
#                        if it looks like an HF id, else MODEL)
#   MODEL                Full HF id used as tokenizer fallback
#                        (default: Qwen/Qwen3-30B-A3B-Instruct-2507)
#   REQUEST_RATE         Average requests/sec (default: 20)
#   REQUEST_COUNT        Total requests to send (default: 1000)
#   RANDOM_SEED          Seed for shuffle sampling (default: 42)
#   OUT_DIR              Artifact dir       (default: playground/out/aiperf/<UTC>)

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

URL="${URL:-http://localhost:8000}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "$MODEL")}"
TOKENIZER="${TOKENIZER:-$MODEL}"
REQUEST_RATE="${REQUEST_RATE:-20}"
REQUEST_COUNT="${REQUEST_COUNT:-1000}"
RANDOM_SEED="${RANDOM_SEED:-42}"
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
OUT_DIR="${OUT_DIR:-playground/out/aiperf/$TS}"
mkdir -p "$OUT_DIR"

AIPERF="${AIPERF:-/root/miniconda3/envs/vllm-nvlink/bin/aiperf}"
if [[ ! -x "$AIPERF" ]]; then
    echo "error: aiperf not found at $AIPERF" >&2
    echo "       set AIPERF=/path/to/aiperf or run build.sh first" >&2
    exit 1
fi

echo "[run_aiperf] url                = $URL"
echo "[run_aiperf] served-model-name  = $SERVED_MODEL_NAME"
echo "[run_aiperf] tokenizer          = $TOKENIZER"
echo "[run_aiperf] request-rate       = $REQUEST_RATE req/s (Poisson)"
echo "[run_aiperf] request-count      = $REQUEST_COUNT"
echo "[run_aiperf] random-seed        = $RANDOM_SEED"
echo "[run_aiperf] artifact-dir       = $OUT_DIR"

# Wait for the server to be reachable. AIPerf has --wait-for-model-timeout
# built in so we just hand the burden to it.
exec "$AIPERF" profile \
    --model "$SERVED_MODEL_NAME" \
    --tokenizer "$TOKENIZER" \
    --url "$URL" \
    --endpoint-type chat \
    --streaming \
    --public-dataset sharegpt \
    --dataset-sampling-strategy shuffle \
    --random-seed "$RANDOM_SEED" \
    --request-count "$REQUEST_COUNT" \
    --request-rate "$REQUEST_RATE" \
    --arrival-pattern poisson \
    --wait-for-model-timeout 600 \
    --wait-for-model-mode models \
    --artifact-dir "$OUT_DIR" \
    --ui-type none
