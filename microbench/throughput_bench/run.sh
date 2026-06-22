#!/usr/bin/env bash
# Launch the decode throughput/stability benchmark with torchrun.
#
# Examples:
#   bash run.sh --batch-size 256 --seq-len 2048 --iters 50
#   NPROC=4 bash run.sh --tp 4 --dp 1 --batch-size 1024 --seq-len 2048
set -euo pipefail

cd "$(dirname "$0")/.."
NPROC="${NPROC:-4}"

exec torchrun --standalone --nproc_per_node="${NPROC}" -m throughput_bench.main "$@"

