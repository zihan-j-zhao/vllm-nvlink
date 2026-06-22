#!/usr/bin/env bash
# Launch the realistic CUDA-graph decode microbenchmark with torchrun.
#
# Usage:
#   bash run.sh                                            # 2 GPUs, defaults (DP=2)
#   bash run.sh --batch-size 16 --iters 100
#   NPROC=4 bash run.sh                                    # 4-way DP
#   NPROC=4 bash run.sh --tp 4                             # 4-way TP, DP=1 auto-derived
#   NPROC=4 bash run.sh --tp 2 --dp 2                      # mixed TP+DP
#   NPROC=4 bash run.sh --tp 4 --no-enable-expert-parallel # TP-only, no EP
#
# All flags after the first ones are forwarded to main.py. main.py asserts
# tp * pp * dp == WORLD_SIZE, so just set NPROC to match.
set -euo pipefail

# Run from the parent dir so `-m sweep.main` resolves the package.
cd "$(dirname "$0")/.."
NPROC="${NPROC:-4}"

exec torchrun --standalone --nproc_per_node="${NPROC}" -m sweep.main "$@"
