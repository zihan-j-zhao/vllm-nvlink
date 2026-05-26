#!/usr/bin/env bash
# Local editable install of vLLM using a pinned precompiled wheel.
#
# What this script does:
#   1. Creates (or reuses) a conda env named `vllm-nvlink` with Python 3.12.
#   2. Installs `uv` inside it (per AGENTS.md, all Python installs go through uv).
#   3. Performs an editable install of this repo, pulling the precompiled wheel
#      that matches the pinned upstream commit SHA below. This already brings
#      in `flashinfer-python` / `flashinfer-cubin` at the version pinned in
#      requirements/cuda.txt, so the FlashInfer attention backend is usable
#      out of the box (select it at runtime via VLLM_ATTENTION_BACKEND=FLASHINFER).
#   4. Installs extras not bundled with vLLM:
#        - figure-drawing packages: matplotlib, seaborn, plotly
#        - NVIDIA AIPerf evaluation tool: aiperf
#
# Re-running the script is safe: existing env / installed packages are reused.

set -euo pipefail

# ---- Config -----------------------------------------------------------------
ENV_NAME="${VLLM_CONDA_ENV:-vllm-nvlink}"
PYTHON_VERSION="${VLLM_PYTHON_VERSION:-3.12}"
TORCH_BACKEND="${VLLM_TORCH_BACKEND:-auto}"
# Pinned upstream commit whose precompiled wheel we want to reuse.
PRECOMPILED_COMMIT="88d34c6409e9fb3c7b8ca0c04756f061d2099eb1"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ---- Locate conda -----------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    echo "error: 'conda' not found on PATH. Install Miniconda/Anaconda first." >&2
    exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ---- Create / reuse conda env ----------------------------------------------
if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "[build.sh] Reusing existing conda env: $ENV_NAME"
else
    echo "[build.sh] Creating conda env: $ENV_NAME (python=$PYTHON_VERSION)"
    conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi

conda activate "$ENV_NAME"

# Sanity check: make sure we're using the env's interpreter.
echo "[build.sh] Using python: $(command -v python)"
python --version

# ---- Bootstrap uv inside the env -------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "[build.sh] Installing uv into env $ENV_NAME"
    python -m pip install --upgrade pip
    python -m pip install uv
fi
echo "[build.sh] Using uv: $(command -v uv)"
uv --version

# Tell uv to target the active conda env rather than creating a .venv.
export VIRTUAL_ENV="$CONDA_PREFIX"

uv_install_vllm() {
    if [[ -n "$TORCH_BACKEND" && "$TORCH_BACKEND" != "none" ]]; then
        uv pip install -e . --torch-backend="$TORCH_BACKEND"
    else
        uv pip install -e .
    fi
}

# ---- Editable install with precompiled wheel -------------------------------
export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_COMMIT="$PRECOMPILED_COMMIT"

echo "[build.sh] Installing vLLM (editable) with precompiled wheel @ $PRECOMPILED_COMMIT"
if ! uv_install_vllm; then
    if [[ -z "${VLLM_TORCH_BACKEND+x}" && "$TORCH_BACKEND" == "auto" ]]; then
        echo "[build.sh] uv torch backend auto failed; retrying with PyPI defaults."
        echo "[build.sh] Set VLLM_TORCH_BACKEND to keep a specific uv torch backend."
        TORCH_BACKEND=none
        uv_install_vllm
    else
        exit 1
    fi
fi

# ---- Extras ----------------------------------------------------------------
# Figure drawing libs (not part of vLLM's runtime deps; useful for benchmarks
# and notebooks).
echo "[build.sh] Installing figure-drawing extras (matplotlib, seaborn, plotly)"
uv pip install matplotlib seaborn plotly

# NVIDIA AIPerf — LLM serving evaluation tool.
# https://github.com/ai-dynamo/aiperf
echo "[build.sh] Installing NVIDIA aiperf"
uv pip install aiperf

echo
echo "[build.sh] Done. Activate the env with:"
echo "    conda activate $ENV_NAME"
echo "To use the FlashInfer attention backend at runtime:"
echo "    export VLLM_ATTENTION_BACKEND=FLASHINFER"
