#!/usr/bin/env bash
# nsys-based bandwidth-vs-workload sweep for a running P/D-disaggregated
# vLLM service (e.g. one launched by start_server_qwen3_235b_fp8_dp2.sh).
#
# Uses Nsight Systems GPU metrics profiling, which offers sample
# intervals down to ~100 μs and records per-GPU HBM, NVLink, SM, and
# tensor-core utilisation timeseries.
#
# For each (ISL, OSL) cell:
#   1. snapshot nvidia-smi nvlink counters BEFORE,
#   2. launch  nsys profile --gpu-metrics-device=all  wrapping aiperf,
#   3. export  nsys report to SQLite,
#   4. snapshot nvidia-smi nvlink counters AFTER,
#   5. write  cell.json  with timing + workload metadata.
#
# Post-process all cells with postprocess_nsys.py to get per-GPU
# timeseries figures (one figure per GPU, multiple resource subplots).
#
# Env knobs:
#   URL                Server base URL    (default: http://127.0.0.1:8000)
#   MODEL              HF model id / tokenizer fallback
#                      (default: Qwen/Qwen3-235B-A22B-Instruct-2507-FP8)
#   SERVED_MODEL_NAME  OpenAI model id    (default: basename of $MODEL)
#   TOKENIZER          HF tokenizer id    (default: $MODEL)
#   REQUEST_RATE       reqs/sec (Poisson) (default: 5). Ignored when
#                      CONCURRENCY_LIST is set.
#   CONCURRENCY_LIST   Comma-separated concurrency levels (closed-loop).
#                      Zipped with ISL/OSL; broadcast a singleton.
#   DURATION_S         steady-state seconds per cell (default: 60)
#   WARMUP_S           seconds to warm up (default: 10)
#   ISL_LIST           comma-separated input lengths   (default: 1024,4096,16384)
#   OSL_LIST           comma-separated output lengths  (default: 128)
#   ISL_STDDEV / OSL_STDDEV  (default: 0)
#   GPU_IDS            comma-separated CUDA ids to profile
#                      (default: 0,1,2,3)
#   TOPOLOGY_JSON      path to topology.json from the server script
#   NSYS               nsys binary (default: auto-detect; also searches
#                      /data/donglinbai/miniconda3 Nsight installs)
#   NSYS_GPU_FREQ      GPU metrics sample frequency in Hz
#                      (default: 10000 = 100 μs; max ~200000 on Blackwell)
#   NSYS_GPU_SET       --gpu-metrics-set (default: file:<repo>/playground/moe_pd/gb10x_lean.config
#                      which records only the 8 metrics postprocess_nsys.py plots.
#                      Set to empty string "" to let nsys auto-pick the full
#                      stock set (e.g. gb10x with 21 metrics); set to e.g.
#                      "gb10x" to force a specific built-in.)
#   NSYS_KEEP_REP      1 keep the .nsys-rep file, 0 delete it after exporting
#                      sqlite (default: 0 — sqlite is sufficient for the
#                      post-processor and is roughly the same size).
#   OUT_DIR            sweep root dir
#                      (default: playground/out/nsys_sweep/<UTC>)
#   RANDOM_SEED        aiperf --random-seed  (default: 42)
#   CONVERSATION_NUM   aiperf --conversation-num: number of *unique* synthetic
#                      conversations (prompts) to generate. Decoupled from
#                      --concurrency (which only sets in-flight count). Keeping
#                      this small bounds the dataset-prep time + capture size;
#                      sampling reuses entries to feed any concurrency.
#                      (default: 512)
#   DATASET_SAMPLING   aiperf --dataset-sampling-strategy: how entries are
#                      drawn during the run. shuffle/random reuse the pool with
#                      wrap-around so a small dataset sustains high concurrency.
#                      (default: shuffle)
#   REQUEST_COUNT_MAX  Hard cap on the computed --request-count upper bound, so
#                      aiperf never pre-materialises a huge request set. 0
#                      disables the cap. (default: 50000)
#   NSYS_DELAY         Seconds to delay nsys metric collection after launch
#                      (nsys --delay). With the bounded dataset, prep is ~1s and
#                      only ~5.5s of fixed client startup precedes serving, which
#                      the post-processor's active-window detection already
#                      excludes — so this defaults OFF. Set >0 only to shave that
#                      startup from the .sqlite. 0 disables. (default: 0)
#   AIPERF             aiperf binary
#                      (default: /root/miniconda3/envs/vllm-nvlink/bin/aiperf)

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SCRIPT_TAG="run_nsys_sweep"

URL="${URL:-http://127.0.0.1:8000}"
MODEL="${MODEL:-Qwen/Qwen3-235B-A22B-Instruct-2507-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "$MODEL")}"
TOKENIZER="${TOKENIZER:-$MODEL}"
REQUEST_RATE="${REQUEST_RATE:-5}"
CONCURRENCY_LIST="${CONCURRENCY_LIST:-}"
DURATION_S="${DURATION_S:-60}"
WARMUP_S="${WARMUP_S:-10}"
ISL_LIST="${ISL_LIST:-1024,4096,16384}"
OSL_LIST="${OSL_LIST:-128}"
ISL_STDDEV="${ISL_STDDEV:-0}"
OSL_STDDEV="${OSL_STDDEV:-0}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NSYS_GPU_FREQ="${NSYS_GPU_FREQ:-10000}"
# Default: lean custom metric set with only the 8 metrics we actually plot.
# Pass NSYS_GPU_SET="" to let nsys auto-pick the full stock set, or any
# other value (e.g. "gb10x") to force a different one.
_DEFAULT_NSYS_GPU_SET="file:$REPO_ROOT/playground/moe_pd/gb10x_lean.config"
NSYS_GPU_SET="${NSYS_GPU_SET-$_DEFAULT_NSYS_GPU_SET}"
NSYS_KEEP_REP="${NSYS_KEEP_REP:-0}"
RANDOM_SEED="${RANDOM_SEED:-42}"
# Dataset bounding: keep the synthetic conversation pool small (decoupled from
# concurrency) and reuse it via shuffle sampling, so dataset prep stays ~1s and
# captures stay small regardless of concurrency. See run_nsys_sweep.sh header.
CONVERSATION_NUM="${CONVERSATION_NUM:-512}"
DATASET_SAMPLING="${DATASET_SAMPLING:-shuffle}"
REQUEST_COUNT_MAX="${REQUEST_COUNT_MAX:-50000}"
# Optional nsys collection delay. With the bounded dataset, prep is ~1s and only
# ~5.5s of fixed client startup (tokenizer + HF HEAD + IPC init) precedes serving
# — the post-processor already excludes that idle, so this defaults OFF (0). Set
# >0 only to trim that startup from the .sqlite.
NSYS_DELAY="${NSYS_DELAY:-0}"
AIPERF="${AIPERF:-/root/miniconda3/envs/vllm-nvlink/bin/aiperf}"
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
OUT_DIR="${OUT_DIR:-playground/out/nsys_sweep/$TS}"

# --- Resolve TOPOLOGY_JSON --------------------------------------------------
if [[ -z "${TOPOLOGY_JSON:-}" ]]; then
    candidate="$(ls -1t playground/log/moe_pd/*/*/topology.json 2>/dev/null | head -n1 || true)"
    if [[ -n "$candidate" ]]; then
        TOPOLOGY_JSON="$candidate"
    fi
fi

# --- Locate nsys -------------------------------------------------------------
if [[ -z "${NSYS:-}" ]]; then
    for candidate in \
        "$(command -v nsys 2>/dev/null || true)" \
        /usr/local/cuda/bin/nsys \
        /opt/nvidia/nsight-systems-*/bin/nsys \
        /data/donglinbai/miniconda3/envs/verl/nsight-compute-2025.1.1/host/target-linux-x64/nsys \
        /data/donglinbai/miniconda3/pkgs/nsight-compute-2026.1.1.2-h934865d_0/nsight-compute-2026.1.1/host/target-linux-x64/nsys; do
        if [[ -n "$candidate" && -x "$candidate" ]]; then
            NSYS="$candidate"
            break
        fi
    done
fi
NSYS="${NSYS:-nsys}"
if ! "$NSYS" --version >/dev/null 2>&1; then
    echo "error: nsys not found at $NSYS" >&2
    echo "       install with one of:" >&2
    echo "         apt-get install -y nsight-systems-2024.7" >&2
    echo "         conda install -c nvidia nsight-systems" >&2
    echo "       or set NSYS=/path/to/nsys" >&2
    exit 1
fi
NSYS_VERSION="$("$NSYS" --version 2>&1 | head -1)"

if [[ ! -x "$AIPERF" ]]; then
    echo "error: aiperf not found at $AIPERF" >&2
    echo "       set AIPERF=/path/to/aiperf" >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "error: nvidia-smi not on PATH" >&2
    exit 1
fi

# Parse and zip ISL_LIST / OSL_LIST.
IFS=',' read -r -a ISL_ARR <<< "$ISL_LIST"
IFS=',' read -r -a OSL_ARR <<< "$OSL_LIST"
N_ISL="${#ISL_ARR[@]}"
N_OSL="${#OSL_ARR[@]}"
if (( N_OSL == 1 && N_ISL > 1 )); then
    OSL_ARR=(); for ((i=0; i<N_ISL; i++)); do OSL_ARR+=("${OSL_LIST}"); done
    N_OSL=$N_ISL
elif (( N_ISL == 1 && N_OSL > 1 )); then
    ISL_ARR=(); for ((i=0; i<N_OSL; i++)); do ISL_ARR+=("${ISL_LIST}"); done
    N_ISL=$N_OSL
fi
if (( N_ISL != N_OSL )); then
    echo "error: ISL_LIST and OSL_LIST must zip (lengths $N_ISL vs $N_OSL)" >&2
    exit 1
fi
N_CELLS=$N_ISL

MODE="rate"
CONC_ARR=()
if [[ -n "$CONCURRENCY_LIST" ]]; then
    MODE="concurrency"
    IFS=',' read -r -a CONC_ARR <<< "$CONCURRENCY_LIST"
    N_CONC="${#CONC_ARR[@]}"
    if (( N_CONC == 1 && N_CELLS > 1 )); then
        only="${CONC_ARR[0]}"
        CONC_ARR=(); for ((i=0; i<N_CELLS; i++)); do CONC_ARR+=("$only"); done
        N_CONC=$N_CELLS
    fi
    if (( N_CONC != N_CELLS )); then
        echo "error: CONCURRENCY_LIST length ($N_CONC) must match cell count ($N_CELLS)" >&2
        exit 1
    fi
fi

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

echo "[$SCRIPT_TAG] url                = $URL"
echo "[$SCRIPT_TAG] served-model-name  = $SERVED_MODEL_NAME"
echo "[$SCRIPT_TAG] tokenizer          = $TOKENIZER"
echo "[$SCRIPT_TAG] nsys               = $NSYS ($NSYS_VERSION)"
echo "[$SCRIPT_TAG] nsys gpu freq      = $NSYS_GPU_FREQ Hz"
echo "[$SCRIPT_TAG] nsys delay         = ${NSYS_DELAY}s"
echo "[$SCRIPT_TAG] dataset            = conv-num=$CONVERSATION_NUM sampling=$DATASET_SAMPLING req-count-max=$REQUEST_COUNT_MAX"
if [[ -n "$NSYS_GPU_SET" ]]; then
    echo "[$SCRIPT_TAG] nsys gpu set       = $NSYS_GPU_SET"
fi
if [[ "$MODE" == "concurrency" ]]; then
    echo "[$SCRIPT_TAG] mode               = CONCURRENCY (closed-loop)"
    echo "[$SCRIPT_TAG] CONCURRENCY_LIST   = ${CONC_ARR[*]}"
    echo "[$SCRIPT_TAG] warmup / measure   = $WARMUP_S s  / $DURATION_S s"
else
    echo "[$SCRIPT_TAG] mode               = RATE (Poisson)"
    echo "[$SCRIPT_TAG] request-rate       = $REQUEST_RATE req/s"
    echo "[$SCRIPT_TAG] duration / cell    = $DURATION_S s (warmup hint $WARMUP_S s)"
fi
echo "[$SCRIPT_TAG] ISL_LIST           = ${ISL_ARR[*]}"
echo "[$SCRIPT_TAG] OSL_LIST           = ${OSL_ARR[*]}"
echo "[$SCRIPT_TAG] cells              = $N_CELLS (zipped)"
echo "[$SCRIPT_TAG] GPU ids            = $GPU_IDS"
echo "[$SCRIPT_TAG] topology.json      = ${TOPOLOGY_JSON:-<unset>}"
echo "[$SCRIPT_TAG] out dir            = $OUT_DIR"

# Build nsys common args.
NSYS_COMMON_ARGS=(
    --gpu-metrics-devices="$GPU_IDS"
    --gpu-metrics-frequency="$NSYS_GPU_FREQ"
    --sample=none
    --trace=none
    --force-overwrite=true
)
if [[ -n "$NSYS_GPU_SET" ]]; then
    NSYS_COMMON_ARGS+=(--gpu-metrics-set="$NSYS_GPU_SET")
fi
if [[ "${NSYS_DELAY:-0}" != "0" ]]; then
    NSYS_COMMON_ARGS+=(--delay="$NSYS_DELAY")
fi

trap 'exit 130' INT TERM

for ((i=0; i<N_CELLS; i++)); do
    ISL="${ISL_ARR[$i]}"
    OSL="${OSL_ARR[$i]}"
    if [[ "$MODE" == "concurrency" ]]; then
        CONC="${CONC_ARR[$i]}"
        CELL_TAG="$(printf 'cell%02d_isl%s_osl%s_c%s' "$i" "$ISL" "$OSL" "$CONC")"
    else
        CONC=""
        CELL_TAG="$(printf 'cell%02d_isl%s_osl%s_r%s' "$i" "$ISL" "$OSL" "$REQUEST_RATE")"
    fi
    CELL_DIR="$OUT_DIR/$CELL_TAG"
    mkdir -p "$CELL_DIR"

    if [[ "$MODE" == "concurrency" ]]; then
        REQUEST_COUNT="$(awk -v c="$CONC" -v d="$DURATION_S" -v w="$WARMUP_S" -v m="$REQUEST_COUNT_MAX" \
            'BEGIN { v = 10 * c * (d + w); if (v < 100) v = 100; if (m > 0 && v > m) v = m; printf "%d", v }')"
    else
        REQUEST_COUNT="$(awk -v r="$REQUEST_RATE" -v d="$DURATION_S" -v w="$WARMUP_S" -v m="$REQUEST_COUNT_MAX" \
            'BEGIN { v = r * (d + w) + 1; if (m > 0 && v > m) v = m; printf "%d", v }')"
    fi

    echo
    if [[ "$MODE" == "concurrency" ]]; then
        echo "[$SCRIPT_TAG] === cell $((i+1))/$N_CELLS :: ISL=$ISL OSL=$OSL conc=$CONC ==="
    else
        echo "[$SCRIPT_TAG] === cell $((i+1))/$N_CELLS :: ISL=$ISL OSL=$OSL ==="
    fi
    echo "[$SCRIPT_TAG] cell dir         = $CELL_DIR"
    echo "[$SCRIPT_TAG] request-count    = $REQUEST_COUNT (upper bound)"

    # NVLink cumulative snapshot BEFORE.
    nvidia-smi nvlink -gt d -i "$GPU_IDS" \
        > "$CELL_DIR/nvlink_before.txt" 2>&1 || true

    NSYS_REPORT="$CELL_DIR/nsys_report"

    T_START_MS="$(date +%s%3N)"

    # Run aiperf under nsys with GPU metrics collection only (no API
    # tracing, no CPU sampling → minimal perturbation).
    set +e
    if [[ "$MODE" == "concurrency" ]]; then
        "$NSYS" profile \
            "${NSYS_COMMON_ARGS[@]}" \
            --output="$NSYS_REPORT" \
            -- \
            "$AIPERF" profile \
                --model "$SERVED_MODEL_NAME" \
                --tokenizer "$TOKENIZER" \
                --url "$URL" \
                --endpoint-type chat \
                --streaming \
                --random-seed "$RANDOM_SEED" \
                --conversation-num "$CONVERSATION_NUM" \
                --dataset-sampling-strategy "$DATASET_SAMPLING" \
                --request-count "$REQUEST_COUNT" \
                --concurrency "$CONC" \
                --warmup-duration "$WARMUP_S" \
                --benchmark-duration "$DURATION_S" \
                --isl "$ISL" \
                --isl-stddev "$ISL_STDDEV" \
                --osl "$OSL" \
                --osl-stddev "$OSL_STDDEV" \
                --extra-inputs "max_completion_tokens:${OSL}" \
                --wait-for-model-timeout 600 \
                --wait-for-model-mode models \
                --artifact-dir "$CELL_DIR/aiperf" \
                --ui-type none \
            >"$CELL_DIR/aiperf.log" 2>&1
    else
        "$NSYS" profile \
            "${NSYS_COMMON_ARGS[@]}" \
            --output="$NSYS_REPORT" \
            -- \
            "$AIPERF" profile \
                --model "$SERVED_MODEL_NAME" \
                --tokenizer "$TOKENIZER" \
                --url "$URL" \
                --endpoint-type chat \
                --streaming \
                --random-seed "$RANDOM_SEED" \
                --conversation-num "$CONVERSATION_NUM" \
                --dataset-sampling-strategy "$DATASET_SAMPLING" \
                --request-count "$REQUEST_COUNT" \
                --request-rate "$REQUEST_RATE" \
                --arrival-pattern poisson \
                --isl "$ISL" \
                --isl-stddev "$ISL_STDDEV" \
                --osl "$OSL" \
                --osl-stddev "$OSL_STDDEV" \
                --extra-inputs "max_completion_tokens:${OSL}" \
                --wait-for-model-timeout 600 \
                --wait-for-model-mode models \
                --artifact-dir "$CELL_DIR/aiperf" \
                --ui-type none \
            >"$CELL_DIR/aiperf.log" 2>&1
    fi
    AIPERF_RC=$?
    set -e

    T_LOAD_END_MS="$(date +%s%3N)"

    # NVLink cumulative snapshot AFTER.
    nvidia-smi nvlink -gt d -i "$GPU_IDS" \
        > "$CELL_DIR/nvlink_after.txt" 2>&1 || true

    # Export nsys report to SQLite for postprocess_nsys.py.
    NSYS_REP="${NSYS_REPORT}.nsys-rep"
    NSYS_SQLITE="${NSYS_REPORT}.sqlite"
    if [[ -f "$NSYS_REP" ]]; then
        echo "[$SCRIPT_TAG] exporting nsys report to sqlite ..."
        "$NSYS" export \
            --type=sqlite \
            --output="$NSYS_SQLITE" \
            "$NSYS_REP" \
            >"$CELL_DIR/nsys_export.log" 2>&1 || {
                echo "[$SCRIPT_TAG] WARNING: nsys export failed; see $CELL_DIR/nsys_export.log" >&2
            }
        if [[ "$NSYS_KEEP_REP" != "1" && -f "$NSYS_SQLITE" ]]; then
            echo "[$SCRIPT_TAG] deleting $NSYS_REP (NSYS_KEEP_REP=0)"
            rm -f "$NSYS_REP"
        fi
    else
        echo "[$SCRIPT_TAG] WARNING: nsys report not found at $NSYS_REP" >&2
    fi

    T_END_MS="$(date +%s%3N)"

    # Record per-cell metadata for the post-processor.
    python3 - <<PY
import json, os, shutil
meta = {
    "cell_tag": "$CELL_TAG",
    "cell_index": $i,
    "mode": "$MODE",
    "concurrency": (int("$CONC") if "$CONC" else None),
    "isl_mean": int("$ISL"),
    "osl_mean": int("$OSL"),
    "isl_stddev": float("$ISL_STDDEV"),
    "osl_stddev": float("$OSL_STDDEV"),
    "request_rate": float("$REQUEST_RATE"),
    "request_count": int("$REQUEST_COUNT"),
    "duration_s": float("$DURATION_S"),
    "warmup_s": float("$WARMUP_S"),
    "t_start_ms": int("$T_START_MS"),
    "t_load_end_ms": int("$T_LOAD_END_MS"),
    "t_end_ms": int("$T_END_MS"),
    "nsys_gpu_freq_hz": int("$NSYS_GPU_FREQ"),
    "gpu_ids": [int(x) for x in "$GPU_IDS".split(",")],
    "aiperf_rc": int("$AIPERF_RC"),
    "url": "$URL",
    "served_model_name": "$SERVED_MODEL_NAME",
    "profiler": "nsys",
    "nsys_version": "$NSYS_VERSION",
}
out = os.path.join("$CELL_DIR", "cell.json")
with open(out, "w") as f:
    json.dump(meta, f, indent=2)

topo_src = "${TOPOLOGY_JSON:-}"
if topo_src and os.path.isfile(topo_src):
    shutil.copy(topo_src, os.path.join("$CELL_DIR", "topology.json"))
PY

    if (( AIPERF_RC != 0 )); then
        echo "[$SCRIPT_TAG] WARNING: aiperf exit=$AIPERF_RC in $CELL_DIR" >&2
    fi
done

trap - INT TERM

echo
echo "[$SCRIPT_TAG] sweep complete. cells -> $OUT_DIR"
echo "[$SCRIPT_TAG] post-process with:"
echo "    python3 playground/moe_pd/postprocess_nsys.py $OUT_DIR"
