# vLLM FusedMoE two-GPU runner

First milestone: run vLLM's Qwen3 MoE block on two GPUs with expert parallelism,
using the real `Qwen3MoeSparseMoeBlock -> FusedMoE -> FusedMoEKernel` path.

```bash
CUDA_VISIBLE_DEVICES=0,1 bash run_fused_moe_2gpu.sh \
  --tokens 55 --layers 1 --warmup 3 --iters 10 --sync-per-layer
```

Raw CUDA graph mode is wired separately from explicit per-layer barriers, but
two-rank AGRS collective replay currently hangs in this standalone harness. It is
therefore opt-in for debugging only:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash run_fused_moe_2gpu.sh \
  --tokens 55 --layers 1 --warmup 3 --iters 10 \
  --cuda-graph --allow-graph-collectives
```

The benchmark writes a CSV under `results/` and prints per-rank p50/p90/p99.

## Add NIXL KV READ traffic

Pass `--kv-transfer-mb` to launch two extra torchrun ranks as NIXL KV producers.
By default the launcher uses visible GPUs `0,1,3,4`: ranks 0-1 run vLLM
FusedMoE, while ranks 2-3 hold source KV buffers. Each decode rank issues
real NIXL READs from the producer ranks before each MoE forward.

```bash
CUDA_VISIBLE_DEVICES=0,1,3,4 bash run_fused_moe_2gpu.sh \
  --tokens 2048 --layers 1 --warmup 5 --iters 20 --sync-per-layer \
  --kv-transfer-mb 64.5 --kv-concurrent-reads 1 \
  --out results/fused_moe_2gpu_nixl_64mb.csv
```

Useful e2e-derived transfer sizes are around 12.5, 64.5, and 88.1 MiB per
READ. Increase `--kv-concurrent-reads` to model multiple simultaneous KV reads
per decode rank.

To make overlap easier to inspect in Nsight Systems, run both phases in one
capture and use `--kv-waves` to queue a longer train of e2e-sized NIXL READs
before each MoE forward:

```bash
CUDA_VISIBLE_DEVICES=0,1,3,4 bash run_fused_moe_2gpu.sh \
  --tokens 2048 --layers 1 --warmup 5 --iters 20 --sync-per-layer \
  --kv-transfer-mb 64.5 --kv-concurrent-reads 1 --kv-waves 8 \
  --phases both \
  --out results/fused_moe_2gpu_two_phase_overlap.csv
```

The CSV contains one `moe` row and one `moe+nixl` row per decode rank. In nsys,
look for the NVTX ranges `phase_moe`, `phase_moe+nixl`, `moe_forward_*`,
`nixl_wave_*`, and `nixl_wait_*`. Keep `--kv-waves 1` for the e2e-like case;
increase it only when you intentionally want a visible contention window.

## DeepEP V1 setup

This repository can run vLLM's DeepEP V1-style MoE all-to-all backend through
`--all2all-backend deepep_low_latency`. The setup below keeps DeepEP isolated
from the working `vllm-nvlink` conda environment.

The validated source checkout is DeepEP branch `hybrid-ep`:

```bash
cd /home/wxzheng/uccl
git clone --branch hybrid-ep https://github.com/deepseek-ai/DeepEP.git DeepEP-v1
cd DeepEP-v1
git submodule update --init --recursive
git rev-parse --short HEAD
```

The validated commit was `e0a5b1d`.

### Create an isolated conda environment

Clone the existing vLLM environment so the vLLM, torch, NIXL, and AIPerf stack
match the normal P/D experiments:

```bash
source /home/wxzheng/miniconda3/etc/profile.d/conda.sh
conda create --name vllm-deepep-v1 --clone vllm-nvlink
conda activate vllm-deepep-v1
```

Sanity check the CUDA/PyTorch pair:

```bash
python - <<'PY'
import sys
import torch

print(sys.executable)
print(torch.__version__, torch.version.cuda)
PY
```

Expected: Python from `vllm-deepep-v1`, PyTorch `2.11.0+cu130`, and CUDA
`13.0`.

### Install the CUDA 13.0 wheel toolchain

The host `/usr/local/cuda` may point at CUDA 12.9, while this environment's
PyTorch was built for CUDA 13.0. Build DeepEP with the CUDA wheel toolchain
inside the conda environment instead of the system toolkit.

```bash
python -m pip install --force-reinstall \
  'nvidia-cuda-nvcc==13.0.88' \
  'nvidia-cuda-runtime==13.0.96' \
  'nvidia-cuda-cccl==13.0.85' \
  'nvidia-cuda-crt==13.0.88' \
  'nvidia-nvvm==13.0.88'
```

Set the build environment:

```bash
export CUDA_HOME="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib:${LIBRARY_PATH:-}"
```

Two compatibility details are needed for this machine:

```bash
# CUDA CCCL headers provide cuda/std/*; the CUDA 12.9 target include provides
# cuda_profiler_api.h, which the DeepEP hybrid extension includes.
export CPATH="$CUDA_HOME/include/cccl:/usr/local/cuda-12.9/targets/x86_64-linux/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/include/cccl:/usr/local/cuda-12.9/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}"

# The CUDA 13 wheel ships libnvtx3interop.so.1 but the DeepEP link step asks
# for -lnvtx3interop.
ln -sf libnvtx3interop.so.1 "$CUDA_HOME/lib/libnvtx3interop.so"
```

### Build and install DeepEP V1

Build for Blackwell/B200 (`sm_100`) and disable the aggressive PTX path that is
not portable across all CUDA versions:

```bash
cd /home/wxzheng/uccl/DeepEP-v1
export TORCH_CUDA_ARCH_LIST=10.0
export DISABLE_AGGRESSIVE_PTX_INSTRS=1
export MAX_JOBS=4
python -m pip install --no-build-isolation --no-deps --force-reinstall -v .
```

Validate the Python package and both native extensions:

```bash
python - <<'PY'
import deep_ep
import deep_ep_cpp
import hybrid_ep_cpp

print(deep_ep.__file__)
print(deep_ep.Buffer)
print(hasattr(deep_ep.Buffer, 'get_low_latency_rdma_size_hint'))
PY
```

Expected: `deep_ep`, `deep_ep_cpp`, and `hybrid_ep_cpp` all import, and the last
line prints `True`.

### Validate vLLM integration

From the vLLM checkout:

```bash
cd /home/wxzheng/uccl/vllm-nvlink
PYTHONPATH=/home/wxzheng/uccl/vllm-nvlink python - <<'PY'
from vllm.utils.import_utils import has_deep_ep

print(has_deep_ep())
PY
```

Expected: `True`.

### Smoke test the P/D service

Run the P/D NIXL service with DeepEP low-latency all-to-all:

```bash
cd /home/wxzheng/uccl/vllm-nvlink
source /home/wxzheng/miniconda3/etc/profile.d/conda.sh
conda activate vllm-deepep-v1

export PY="$CONDA_PREFIX/bin/python"
export CUDA_HOME="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib:${LIBRARY_PATH:-}"
export CPATH="$CUDA_HOME/include/cccl:/usr/local/cuda-12.9/targets/x86_64-linux/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/include/cccl:/usr/local/cuda-12.9/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}"

ALL2ALL_BACKEND=deepep_low_latency \
MOE_BACKEND=auto \
GPU_MEM_UTIL=0.75 \
PD_TRACE=1 \
VLLM_PD_TRACE_REQ_IDS=1 \
ENFORCE_EAGER=0 \
NVSHMEM_QP_DEPTH=4096 \
LOG_DIR=playground/log/e2e_agrs_nixl/deepep_v1_smoke_$(date -u +%Y-%m-%dT%H-%M-%SZ) \
bash playground/e2e_agrs_nixl/start_server.sh
```

In the logs, check for these two lines on both prefill/decode sides:

```text
Using DeepEPLLAll2AllManager all2all manager.
Using DeepEPLLPrepareAndFinalize
```

Then send a tiny request through the proxy:

```bash
curl -sS -m 120 http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-30B-A3B-Instruct-2507","messages":[{"role":"user","content":"Say ok."}],"max_tokens":4,"temperature":0}' \
  | python -m json.tool
```

### Run the profiling example

The validated profiling run used 1000 warmup requests, then 2000 main requests
at 20 requests/s with `max_completion_tokens=1000`:

```bash
cd /home/wxzheng/uccl/vllm-nvlink
source /home/wxzheng/miniconda3/etc/profile.d/conda.sh
conda activate vllm-deepep-v1

export PY="$CONDA_PREFIX/bin/python"
export CUDA_HOME="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib:${LIBRARY_PATH:-}"
export CPATH="$CUDA_HOME/include/cccl:/usr/local/cuda-12.9/targets/x86_64-linux/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/include/cccl:/usr/local/cuda-12.9/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}"

RUN_NAME=e2e_agrs_nixl_deepepv1_r20_o1000_$(date -u +%Y-%m-%dT%H-%M-%SZ) \
ALL2ALL_BACKEND=deepep_low_latency \
MOE_BACKEND=auto \
NVSHMEM_QP_DEPTH=4096 \
REQUEST_RATE=20 \
MAX_OUTPUT_TOKENS=1000 \
GPU_MEM_UTIL=0.75 \
bash playground/e2e_agrs_nixl/e2e_2000req/run_cpu_profile.sh
```

Analyze the run:

```bash
python playground/e2e_agrs_nixl/e2e_2000req/analyze_cpu_profile.py \
  --run-name "$RUN_NAME"
```

The successful validation run was
`e2e_agrs_nixl_deepepv1_r20_o1000_2026-06-02T09-12-13Z`.

### Known observations

On this host, startup prints NVSHMEM `IBGDA` transport initialization warnings,
but the service continues and runs successfully. In the validated workload,
DeepEP V1 low-latency was functional but slower than the AGRS baseline for
client-visible latency and decode engine-step time, so treat it as an
integration point to investigate rather than a proven performance win.
