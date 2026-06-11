#!/usr/bin/env bash
# Drive a short workload while vLLM workers (wrapped in nsys with
# --capture-range=cudaProfilerApi) record a kernel-level trace.
#
# Requires the server to have been started with NSYS_TRACE_VLLM=1 set,
# so each api_server process tree is under `nsys profile`.
#
# Sequence:
#   1. Warmup load (a handful of throwaway requests so cudagraphs +
#      MoE JIT have already happened by the time we start the capture).
#   2. POST /start_profile on prefill AND decode  -> cudaProfilerStart()
#      -> nsys begins writing.
#   3. Run aiperf for the measurement window.
#   4. POST /stop_profile on both sides -> cudaProfilerStop() -> nsys
#      flushes the .nsys-rep files (one per process) under
#      ${LOG_DIR}/nsys_vllm/.
#
# Env knobs:
#   URL                proxy URL (default http://127.0.0.1:8000)
#   PREFILL_URL        direct prefill (default http://127.0.0.1:8100)
#   DECODE_URL         direct decode  (default http://127.0.0.1:8200)
#   MODEL              served model id (default Qwen3-235B-A22B-Instruct-2507-FP8)
#   TOKENIZER          HF tokenizer (default Qwen/$MODEL)
#   ISL / OSL          synthetic seq lengths (default 4096 / 128)
#   CONCURRENCY        aiperf --concurrency (default 8)
#   DURATION_S         measurement window in seconds (default 10).
#                      KEEP SHORT - kernel trace files balloon fast.
#   WARMUP_REQS        # of pre-capture warmup requests (default 4)
#   AIPERF             aiperf binary (default /opt/miniconda/envs/vllm-nvlink/bin/aiperf)
#   OUT_DIR            artifact dir (default playground/out/nsys_vllm_trace/<UTC>)

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SCRIPT_TAG="run_vllm_trace_capture"

URL="${URL:-http://127.0.0.1:8000}"
PREFILL_URL="${PREFILL_URL:-http://127.0.0.1:8100}"
DECODE_URL="${DECODE_URL:-http://127.0.0.1:8200}"
MODEL="${MODEL:-Qwen3-235B-A22B-Instruct-2507-FP8}"
TOKENIZER="${TOKENIZER:-Qwen/$MODEL}"
ISL="${ISL:-4096}"
OSL="${OSL:-128}"
CONCURRENCY="${CONCURRENCY:-8}"
DURATION_S="${DURATION_S:-10}"
WARMUP_REQS="${WARMUP_REQS:-4}"
AIPERF="${AIPERF:-/opt/miniconda/envs/vllm-nvlink/bin/aiperf}"
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
OUT_DIR="${OUT_DIR:-playground/out/nsys_vllm_trace/$TS}"

if [[ ! -x "$AIPERF" ]]; then
    echo "error: aiperf not found at $AIPERF" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

echo "[$SCRIPT_TAG] proxy URL          = $URL"
echo "[$SCRIPT_TAG] prefill URL        = $PREFILL_URL"
echo "[$SCRIPT_TAG] decode URL         = $DECODE_URL"
echo "[$SCRIPT_TAG] ISL=$ISL OSL=$OSL conc=$CONCURRENCY duration=${DURATION_S}s"
echo "[$SCRIPT_TAG] out dir            = $OUT_DIR"

# Sanity-check that the server is up and answering /v1/models.
for url in "$URL" "$PREFILL_URL" "$DECODE_URL"; do
    if ! curl -sf "$url/v1/models" >/dev/null 2>&1; then
        echo "error: $url not reachable; is the server running?" >&2
        exit 1
    fi
done

# --- 1. Warmup --------------------------------------------------------------
# Short pre-capture warmup so the first capture iteration is on a hot path
# (cudagraphs already replayed, MoE JIT compiled, KV transfer paths primed).
echo "[$SCRIPT_TAG] warming up ($WARMUP_REQS requests)..."
for _ in $(seq 1 "$WARMUP_REQS"); do
    curl -sf -X POST "$URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}" \
        > /dev/null || true
done

# --- 2. Start nsys capture on both sides ------------------------------------
# vLLM's /start_profile triggers torch.cuda.profiler.start() -> cudaProfilerStart
# which is what nsys --capture-range=cudaProfilerApi waits on.
echo "[$SCRIPT_TAG] POST /start_profile on prefill + decode ..."
curl -sf -X POST "$PREFILL_URL/start_profile" -o "$OUT_DIR/start_profile_prefill.json" || {
    echo "warning: /start_profile on prefill returned non-200" >&2
}
curl -sf -X POST "$DECODE_URL/start_profile"  -o "$OUT_DIR/start_profile_decode.json"  || {
    echo "warning: /start_profile on decode returned non-200" >&2
}

# Tiny pause so cudaProfilerStart() has actually landed on every worker
# before aiperf starts.
sleep 1

# --- 3. Drive load ----------------------------------------------------------
T_START_MS="$(date +%s%3N)"
echo "[$SCRIPT_TAG] running aiperf for ${DURATION_S}s ..."
REQUEST_COUNT="$(awk -v c="$CONCURRENCY" -v d="$DURATION_S" \
    'BEGIN { v = 10 * c * d; if (v < 50) v = 50; printf "%d", v }')"

set +e
"$AIPERF" profile \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --url "$URL" \
    --endpoint-type chat \
    --streaming \
    --random-seed 42 \
    --request-count "$REQUEST_COUNT" \
    --concurrency "$CONCURRENCY" \
    --warmup-duration 1 \
    --benchmark-duration "$DURATION_S" \
    --isl "$ISL" \
    --isl-stddev 0 \
    --osl "$OSL" \
    --osl-stddev 0 \
    --extra-inputs "max_completion_tokens:${OSL}" \
    --wait-for-model-timeout 60 \
    --wait-for-model-mode models \
    --artifact-dir "$OUT_DIR/aiperf" \
    --ui-type none \
    >"$OUT_DIR/aiperf.log" 2>&1
AIPERF_RC=$?
set -e
T_END_MS="$(date +%s%3N)"

# --- 4. Stop nsys capture ---------------------------------------------------
# cudaProfilerStop() makes nsys finalise and write the per-process .nsys-rep.
echo "[$SCRIPT_TAG] POST /stop_profile on prefill + decode ..."
curl -sf -X POST "$PREFILL_URL/stop_profile" -o "$OUT_DIR/stop_profile_prefill.json" || true
curl -sf -X POST "$DECODE_URL/stop_profile"  -o "$OUT_DIR/stop_profile_decode.json"  || true

# Give nsys a moment to flush. Files appear under the server's log dir,
# which is $LOG_DIR/nsys_vllm/ from start_server_*.sh.
echo "[$SCRIPT_TAG] waiting 10s for nsys to flush .nsys-rep files ..."
sleep 10

# Locate nsys files: glob the newest log dir under playground/log/moe_pd
# and copy its nsys_vllm/*.nsys-rep into our OUT_DIR for tidiness.
NSYS_SRC_DIR="$(ls -1td playground/log/moe_pd/*/*/nsys_vllm 2>/dev/null | head -n1 || true)"
if [[ -n "$NSYS_SRC_DIR" && -d "$NSYS_SRC_DIR" ]]; then
    echo "[$SCRIPT_TAG] nsys reports written to $NSYS_SRC_DIR"
    cp -v "$NSYS_SRC_DIR"/*.nsys-rep "$OUT_DIR/" 2>/dev/null || true
    cp -v "$NSYS_SRC_DIR"/*.qdstrm   "$OUT_DIR/" 2>/dev/null || true
else
    echo "warning: could not find $NSYS_SRC_DIR; check the server log dir" >&2
fi

# When the server was started with --capture-range-end=repeat (the default),
# nsys does NOT finalise the .nsys-rep until the wrapped process exits. The
# intermediate trace data is in /tmp as nsys-report-*.qdstrm. Convert any
# that look fresh (modified within the last 5 minutes) via QdstrmImporter.
QDSTRM_IMPORTER="$(ls -1 /data/donglinbai/miniconda3/envs/verl/nsight-compute-*/host/linux-desktop-glibc_*-x64/QdstrmImporter 2>/dev/null | head -n1 || true)"
if [[ -z "$QDSTRM_IMPORTER" ]]; then
    QDSTRM_IMPORTER="$(command -v QdstrmImporter 2>/dev/null || true)"
fi
if [[ -n "$QDSTRM_IMPORTER" && -x "$QDSTRM_IMPORTER" ]]; then
    echo "[$SCRIPT_TAG] scanning /tmp for fresh qdstrm files ..."
    found_qdstrm=0
    while IFS= read -r qd; do
        [[ -n "$qd" ]] || continue
        found_qdstrm=1
        base="$(basename "$qd" .qdstrm)"
        out="$OUT_DIR/${base}.nsys-rep"
        echo "[$SCRIPT_TAG]   convert $qd -> $out"
        "$QDSTRM_IMPORTER" -i "$qd" -o "$out" -f >/dev/null 2>&1 || true
    done < <(find /tmp -maxdepth 1 -name 'nsys-report-*.qdstrm' -mmin -5 2>/dev/null)
    if (( found_qdstrm == 0 )); then
        echo "[$SCRIPT_TAG] (no fresh qdstrm files found)"
    fi
fi

# Persist a small meta blob so this run can be located later.
cat >"$OUT_DIR/capture.json" <<EOF
{
  "isl": $ISL,
  "osl": $OSL,
  "concurrency": $CONCURRENCY,
  "duration_s": $DURATION_S,
  "warmup_reqs": $WARMUP_REQS,
  "t_start_ms": $T_START_MS,
  "t_end_ms":   $T_END_MS,
  "aiperf_rc":  $AIPERF_RC,
  "url":        "$URL",
  "model":      "$MODEL",
  "nsys_src":   "${NSYS_SRC_DIR:-<not found>}"
}
EOF

echo
ls -la "$OUT_DIR"
echo
echo "[$SCRIPT_TAG] done. open .nsys-rep in nsys-ui to view kernel timelines + NVTX."
echo "[$SCRIPT_TAG] tip: convert to sqlite with"
echo "    nsys export --type=sqlite --output=<f>.sqlite <f>.nsys-rep"
