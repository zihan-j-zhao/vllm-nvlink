#!/usr/bin/env bash
# Reprocess a traced e2e_agrs_nixl run into fixed analysis windows.
#
# Usage:
#   TRACE_DIR=playground/log/e2e_agrs_nixl/<RUN>/pd_trace \
#     bash playground/e2e_agrs_nixl/run_window_analysis.sh
#
# Optional env:
#   OUT_ROOT      default: playground/out/overlap/e2e_agrs_nixl
#   PY            default: python on PATH
#   ANALYZER      default: playground/e2e_agrs_nixl/analyze_overlap.py
#   WINDOW_SEC    default: 60

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-$(command -v python || true)}"
if [[ ! -x "$PY" ]]; then
    echo "error: python not found; activate the vllm-nvlink env or set PY=/path/to/python" >&2
    exit 1
fi

if [[ -z "${TRACE_DIR:-}" ]]; then
    latest="$(ls -td playground/log/e2e_agrs_nixl/*/pd_trace 2>/dev/null | head -1 || true)"
    if [[ -z "$latest" ]]; then
        echo "error: set TRACE_DIR=playground/log/e2e_agrs_nixl/<RUN>/pd_trace" >&2
        exit 1
    fi
    TRACE_DIR="$latest"
fi

ANALYZER="${ANALYZER:-playground/e2e_agrs_nixl/analyze_overlap.py}"
OUT_ROOT="${OUT_ROOT:-playground/out/overlap/e2e_agrs_nixl}"
WINDOW_SEC="${WINDOW_SEC:-60}"
mkdir -p "$OUT_ROOT"

run_window() {
    local name="$1"
    local start_sec="$2"
    local out_dir="$OUT_ROOT/$name"
    echo "[window] $name: +${start_sec}s for ${WINDOW_SEC}s -> $out_dir"
    "$PY" "$ANALYZER" \
        --trace-dir "$TRACE_DIR" \
        --out-dir "$out_dir" \
        --window-start-sec "$start_sec" \
        --window-sec "$WINDOW_SEC" \
        --recv-interval-source nixl \
        --nixl-start start_time \
        --no-itl
}

run_window telemetry_5k_starttime_union_early_120 120
run_window telemetry_5k_starttime_union_mid_180 180
run_window telemetry_5k_starttime_union_mid_240 240
run_window telemetry_5k_starttime_union_mid_300 300
run_window telemetry_5k_starttime_union_late_360 360

echo "[window] done. Next:"
echo "  $PY playground/e2e_agrs_nixl/plot_transfer_volume_slowdown.py --window-root $OUT_ROOT"
