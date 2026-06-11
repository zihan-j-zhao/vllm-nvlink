#!/usr/bin/env bash
# 2D (sequence-length x batch-size) resource-utilisation grid sweep for the
# Qwen3-30B-A3B P/D-disaggregated service (DP=EP=2, 4 GPUs) launched by
# start_server.sh.
#
# This is a thin driver on top of run_nsys_sweep.sh. run_nsys_sweep.sh only
# *zips* ISL_LIST with CONCURRENCY_LIST 1:1; here we expand a full cross
# product of ISL_GRID x BATCH_GRID into the flattened ISL_LIST /
# CONCURRENCY_LIST that run_nsys_sweep.sh consumes, applying a per-ISL batch
# cap derived from decode-side KV-cache capacity so we don't waste cells on
# oversubscribed (queued) batch sizes.
#
# Env knobs:
#   URL                Proxy base URL          (default: http://127.0.0.1:8000)
#   MODEL              HF model id             (default: Qwen/Qwen3-30B-A3B-Instruct-2507)
#   SERVED_MODEL_NAME  OpenAI model id         (default: basename of $MODEL)
#   TOKENIZER          HF tokenizer id         (default: $MODEL)
#   ISL_GRID           comma input lengths     (default: 128,256,512,1024,2048)
#   BATCH_GRID         comma batch sizes       (default: 1,8,16,32,64,128,256,512,1024,2048)
#   OSL                output length per req   (default: 128)
#   KV_CACHE_TOKENS    total decode KV tokens across DP ranks. When > 0 the
#                      per-ISL batch cap = floor(KV/(ISL+OSL)); batch sizes
#                      above the cap are dropped and replaced by a single
#                      cell at the cap. 0 disables capping. (default: 0)
#   GPU_IDS            comma CUDA ids to profile  (default: 4,5,6,7)
#   TOPOLOGY_JSON      topology.json from start_server.sh (default: auto)
#   DURATION_S         steady-state seconds per cell (default: 30)
#   WARMUP_S           warmup seconds per cell       (default: 8)
#   OUT_DIR            sweep root dir          (default: playground/out/nsys_sweep_30b/<UTC>)
#   DRY_RUN            1 = print the expanded grid and exit (default: 0)
#
# All other knobs (NSYS, AIPERF, NSYS_GPU_FREQ, NSYS_GPU_SET, RANDOM_SEED,
# NSYS_KEEP_REP, ...) are passed straight through to run_nsys_sweep.sh.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

SCRIPT_TAG="run_30b_grid_sweep"

URL="${URL:-http://127.0.0.1:8000}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "$MODEL")}"
TOKENIZER="${TOKENIZER:-$MODEL}"
ISL_GRID="${ISL_GRID:-128,256,512,1024,2048}"
BATCH_GRID="${BATCH_GRID:-1,8,16,32,64,128,256,512,1024,2048}"
OSL="${OSL:-128}"
KV_CACHE_TOKENS="${KV_CACHE_TOKENS:-0}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
DURATION_S="${DURATION_S:-30}"
WARMUP_S="${WARMUP_S:-8}"
DRY_RUN="${DRY_RUN:-0}"
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
OUT_DIR="${OUT_DIR:-playground/out/nsys_sweep_30b/$TS}"

IFS=',' read -r -a ISL_ARR <<< "$ISL_GRID"
IFS=',' read -r -a BATCH_ARR <<< "$BATCH_GRID"

# Expand the cross product with per-ISL capping.
ISL_FLAT=()
CONC_FLAT=()
for isl in "${ISL_ARR[@]}"; do
    cap=-1
    if (( KV_CACHE_TOKENS > 0 )); then
        cap=$(( KV_CACHE_TOKENS / (isl + OSL) ))
        if (( cap < 1 )); then cap=1; fi
    fi
    over=0
    declare -a sel=()
    for b in "${BATCH_ARR[@]}"; do
        if (( cap < 0 || b <= cap )); then
            sel+=("$b")
        else
            over=1
        fi
    done
    # If some requested batches exceeded the cap, add a single cell at the cap
    # (the max supported batch) unless it is already in the list.
    if (( over == 1 && cap >= 1 )); then
        present=0
        for b in "${sel[@]}"; do [[ "$b" == "$cap" ]] && present=1; done
        if (( present == 0 )); then sel+=("$cap"); fi
    fi
    for b in "${sel[@]}"; do
        ISL_FLAT+=("$isl")
        CONC_FLAT+=("$b")
    done
done

# Join helpers.
join_csv() { local IFS=','; echo "$*"; }
ISL_LIST="$(join_csv "${ISL_FLAT[@]}")"
CONCURRENCY_LIST="$(join_csv "${CONC_FLAT[@]}")"

echo "[$SCRIPT_TAG] model              = $MODEL"
echo "[$SCRIPT_TAG] served-model-name  = $SERVED_MODEL_NAME"
echo "[$SCRIPT_TAG] ISL grid           = $ISL_GRID"
echo "[$SCRIPT_TAG] batch grid         = $BATCH_GRID"
echo "[$SCRIPT_TAG] OSL                = $OSL"
echo "[$SCRIPT_TAG] KV cache tokens    = $KV_CACHE_TOKENS (0 = no cap)"
echo "[$SCRIPT_TAG] expanded cells     = ${#ISL_FLAT[@]}"
echo "[$SCRIPT_TAG] GPU ids            = $GPU_IDS"
echo "[$SCRIPT_TAG] out dir            = $OUT_DIR"
echo "[$SCRIPT_TAG] ----- expanded (ISL, batch) grid -----"
prev_isl=""
for ((i=0; i<${#ISL_FLAT[@]}; i++)); do
    if [[ "${ISL_FLAT[$i]}" != "$prev_isl" ]]; then
        [[ -n "$prev_isl" ]] && echo
        printf "  ISL=%-5s batches:" "${ISL_FLAT[$i]}"
        prev_isl="${ISL_FLAT[$i]}"
    fi
    printf " %s" "${CONC_FLAT[$i]}"
done
echo
echo "[$SCRIPT_TAG] --------------------------------------"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[$SCRIPT_TAG] DRY_RUN=1, not launching sweep."
    exit 0
fi

export URL MODEL SERVED_MODEL_NAME TOKENIZER GPU_IDS OUT_DIR DURATION_S WARMUP_S
export ISL_LIST OSL_LIST="$OSL" CONCURRENCY_LIST
if [[ -n "${TOPOLOGY_JSON:-}" ]]; then export TOPOLOGY_JSON; fi

exec "$HERE/run_nsys_sweep.sh"
