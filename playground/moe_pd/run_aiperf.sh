#!/usr/bin/env bash
# Drive the running P/D-disaggregated vLLM service (proxy + prefill + decode
# launched by playground/moe_pd/start_server.sh) with AIPerf.
#
# Identical to playground/moe_a2a/run_aiperf.sh apart from:
#   * Default URL points at the NIXL proxy on port 8000.
#   * Default artifact dir lives under playground/out/aiperf_pd/<UTC>.
#
# Env knobs (same as moe_a2a/run_aiperf.sh):
#   URL                  Server base URL  (default: http://127.0.0.1:8000)
#   SERVED_MODEL_NAME    Model id served by the server
#                        (default: basename of $MODEL)
#   TOKENIZER            HF tokenizer id  (default: $MODEL)
#   MODEL                Full HF id used as tokenizer fallback
#                        (default: Qwen/Qwen3-30B-A3B-Instruct-2507)
#   REQUEST_RATE         Average requests/sec     (default: 20)
#   REQUEST_COUNT        Total requests to send   (default: 500)
#   RANDOM_SEED          Seed for shuffle sampling (default: 42)
#   MAX_OUTPUT_TOKENS    Per-request cap on generated tokens (default: 500).
#                        Passed via `--extra-inputs max_completion_tokens:N`;
#                        the AIPerf openai_chat formatter `payload.update(extras)`
#                        AFTER reading the per-row ShareGPT value, so this
#                        override wins.
#   OUT_DIR              Artifact dir             (default: playground/out/aiperf_pd/<UTC>)

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

URL="${URL:-http://127.0.0.1:8000}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "$MODEL")}"
TOKENIZER="${TOKENIZER:-$MODEL}"
REQUEST_RATE="${REQUEST_RATE:-20}"
REQUEST_COUNT="${REQUEST_COUNT:-500}"
RANDOM_SEED="${RANDOM_SEED:-42}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-500}"
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
OUT_DIR="${OUT_DIR:-playground/out/aiperf_pd/$TS}"
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
echo "[run_aiperf] max-output-tokens  = $MAX_OUTPUT_TOKENS"
echo "[run_aiperf] artifact-dir       = $OUT_DIR"

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
    --extra-inputs "max_completion_tokens:${MAX_OUTPUT_TOKENS}" \
    --wait-for-model-timeout 600 \
    --wait-for-model-mode models \
    --artifact-dir "$OUT_DIR" \
    --ui-type none

# Streaming + models-mode readiness both work because
# playground/moe_pd/proxy.py (1) forwards the upstream `text/event-stream`
# Content-Type verbatim so AIPerf's SSE parser sees real `data: {...}`
# events, and (2) implements GET /v1/models by forwarding to the first
# prefill instance.
