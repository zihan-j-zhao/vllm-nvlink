#!/usr/bin/env bash
# Post-process + aggregate the two concurrent 30B grid sweeps (A: ISL
# 128/256/512 on GPUs 4-7; B: ISL 1024/2048 on GPUs 0-3) into a single
# combined summary CSV and ISL x batch heatmaps for HBM / NVLink / SM.
#
# nsys reports the profiled GPUs as RELATIVE indices 0-3 for BOTH servers,
# with prefill = first two (0,1) and decode = last two (2,3).
set -euo pipefail
source /tmp/full_env.sh
ROOT="$FULL_ROOT"
PY="conda run -n vllm-nvlink python"
PP=playground/moe_pd/postprocess_nsys.py
AG=playground/moe_pd/aggregate_grid.py

echo "[analyze] post-processing sweep A ($ROOT/A) ..."
$PY $PP "$ROOT/A" --prefill-gpus 0,1 --decode-gpus 2,3 --bin-ms 50
echo "[analyze] post-processing sweep B ($ROOT/B) ..."
$PY $PP "$ROOT/B" --prefill-gpus 0,1 --decode-gpus 2,3 --bin-ms 50

# Merge the two per-sweep summary CSVs (header once).
COMB="$ROOT/summary_nsys.csv"
head -1 "$ROOT/A/summary_nsys.csv" > "$COMB"
tail -n +2 "$ROOT/A/summary_nsys.csv" >> "$COMB"
tail -n +2 "$ROOT/B/summary_nsys.csv" >> "$COMB"
echo "[analyze] combined summary -> $COMB ($(($(wc -l < "$COMB")-1)) rows)"

echo "[analyze] aggregating into ISL x batch grids + heatmaps ..."
$PY $AG "$COMB"
echo "[analyze] DONE -> $ROOT/grid_analysis"
