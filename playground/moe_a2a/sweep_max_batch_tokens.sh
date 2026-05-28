#!/usr/bin/env bash
# Sweep --max-num-batched-tokens across {4096, 8192, 16384} on GPUs 6,7.
# For each value: start a profiled vLLM server, drive it with aiperf, then
# fold the trace into a per-step CSV and render transfer-time plots, all
# named with a `maxbatch{M}` suffix so the runs are distinguishable.
#
# Requires the `vllm-nvlink` conda env at /root/miniconda3/envs/vllm-nvlink.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PY=/root/miniconda3/envs/vllm-nvlink/bin/python
AIPERF=/root/miniconda3/envs/vllm-nvlink/bin/aiperf
export CUDA_VISIBLE_DEVICES=6,7

# Persist the same workload settings across runs so only --max-num-batched-tokens varies.
export REQUEST_RATE=20
export REQUEST_COUNT=500
export RANDOM_SEED=42
export MAX_OUTPUT_TOKENS=500

# Drop CUDA caches between runs by waiting for the PIDs to fully exit.
wait_for_no_vllm() {
    local deadline=$((SECONDS + 60))
    while pgrep -f 'vllm\.entrypoints\.openai\.api_server' >/dev/null 2>&1; do
        if (( SECONDS > deadline )); then
            echo "[sweep] warning: vLLM still running after 60s" >&2
            return 0
        fi
        sleep 2
    done
}

wait_for_ready() {
    local deadline=$((SECONDS + 600))
    while ! curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; do
        if (( SECONDS > deadline )); then
            echo "[sweep] error: server didn't come up in 600s" >&2
            return 1
        fi
        sleep 5
    done
}

run_one() {
    local M=$1
    local TS RUN LOG_DIR AIPERF_DIR CSV PLOT_DIR
    TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
    RUN="maxbatch${M}_${TS}"
    LOG_DIR="$REPO_ROOT/playground/log/moe_a2a/${RUN}"
    AIPERF_DIR="$REPO_ROOT/playground/out/aiperf/${RUN}"
    CSV="$REPO_ROOT/playground/out/moe_a2a_sharegpt_${RUN}.csv"
    PLOT_DIR="$REPO_ROOT/playground/out/figures/${RUN}"
    mkdir -p "$LOG_DIR" "$PLOT_DIR"

    echo
    echo "================================================================"
    echo "[sweep] starting run: M=${M}   ${RUN}"
    echo "================================================================"

    # Launch the server in its own process group so a SIGTERM here doesn't
    # leak workers.
    MAX_BATCH_TOKENS=$M LOG_DIR="$LOG_DIR" \
        setsid bash playground/moe_a2a/start_server.sh \
        >"$LOG_DIR/server.outer.log" 2>&1 &
    local SERVER_PGID=$!
    echo "[sweep] server pgid=$SERVER_PGID  log=$LOG_DIR/server.log"

    trap 'kill -TERM -$SERVER_PGID 2>/dev/null || true' EXIT INT TERM

    if ! wait_for_ready; then
        echo "[sweep] tail of server.log:" >&2
        tail -30 "$LOG_DIR/server.log" >&2 || true
        kill -TERM -$SERVER_PGID 2>/dev/null || true
        wait_for_no_vllm
        trap - EXIT INT TERM
        return 1
    fi

    echo "[sweep] server ready, launching aiperf"
    mkdir -p "$AIPERF_DIR"
    OUT_DIR="$AIPERF_DIR" bash playground/moe_a2a/run_aiperf.sh \
        >"$AIPERF_DIR/aiperf.outer.log" 2>&1 || {
            echo "[sweep] aiperf failed; see $AIPERF_DIR/aiperf.outer.log" >&2
        }

    echo "[sweep] aiperf done; shutting down server"
    # SIGTERM the actual API server PID so its atexit/SIGTERM handler flushes
    # the profiler JSONLs cleanly. The whole pgid takedown follows.
    local api_pid
    api_pid=$(pgrep -f 'vllm\.entrypoints\.openai\.api_server' | head -1 || true)
    if [[ -n "$api_pid" ]]; then
        kill -TERM "$api_pid" 2>/dev/null || true
    fi
    # Give workers up to 30s to flush before pgkill.
    local i
    for i in $(seq 1 30); do
        pgrep -f 'vllm\.entrypoints\.openai\.api_server' >/dev/null 2>&1 || break
        sleep 1
    done
    kill -TERM -$SERVER_PGID 2>/dev/null || true
    wait_for_no_vllm
    trap - EXIT INT TERM

    echo "[sweep] extracting per-step CSV → $CSV"
    "$PY" playground/moe_a2a/extract_per_step.py \
        --jsonl-glob "$LOG_DIR/moe_a2a_rank*.jsonl" \
        --num-layers 48 \
        --output "$CSV"

    echo "[sweep] plotting → $PLOT_DIR"
    "$PY" playground/moe_a2a/plot_transfer_time.py \
        --csv "$CSV" --output-dir "$PLOT_DIR" \
        --time-pctl-cap 99.9

    echo "[sweep] M=${M} complete"
}

mkdir -p playground/log/moe_a2a playground/out/aiperf playground/out/figures
for M in 4096 8192 16384; do
    run_one "$M"
done
echo
echo "[sweep] all runs done"
