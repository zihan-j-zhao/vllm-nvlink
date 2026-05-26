# vllm-nvlink

A snapshot of [vLLM](https://github.com/vllm-project/vllm) **v0.20.0** (upstream
commit `88d34c6409e9fb3c7b8ca0c04756f061d2099eb1`), set up for fast local
editable development against the matching precompiled wheel.

The original upstream README is preserved as [README.old.md](README.old.md).

---

## 1. Install locally with `build.sh`

The `build.sh` script at the repo root automates the whole setup:

1. Creates (or reuses) a conda env named **`vllm-nvlink`** with Python 3.12.
2. Installs `uv` inside it (per `AGENTS.md`, all Python installs go through
   `uv`).
3. Performs an editable install of this repo, pulling the precompiled wheel
   matching upstream commit `88d34c6409e9fb3c7b8ca0c04756f061d2099eb1`. This
   also brings in `flashinfer-python` / `flashinfer-cubin` at the version
   pinned in [requirements/cuda.txt](requirements/cuda.txt), so the FlashInfer
   attention backend is ready to use.
4. Installs extras not bundled with vLLM:
   - Figure-drawing: `matplotlib`, `seaborn`, `plotly`
   - NVIDIA [`aiperf`](https://github.com/ai-dynamo/aiperf) evaluation tool

### Prerequisites

- Linux with NVIDIA GPU(s) and a working CUDA driver
- `conda` (Miniconda / Anaconda / Miniforge) on `PATH`

### Run it

```bash
./build.sh
```

The script is **idempotent** — re-running it reuses the existing env and only
reinstalls packages that changed.

Optional environment overrides:

| Variable             | Default       | Purpose                          |
| -------------------- | ------------- | -------------------------------- |
| `VLLM_CONDA_ENV`     | `vllm-nvlink` | Conda env name                   |
| `VLLM_PYTHON_VERSION`| `3.12`        | Python version for the conda env |
| `VLLM_TORCH_BACKEND` | `auto`        | `uv` PyTorch backend; use `none` for PyPI defaults |

Activate the env afterwards:

```bash
conda activate vllm-nvlink
```

### Verify the install

```bash
python -c "import vllm; print(vllm.__version__)"
```

---

## 2. Start an OpenAI-compatible server

vLLM ships an OpenAI-compatible HTTP server. Activate the env first, then
launch it with the model of your choice:

```bash
conda activate vllm-nvlink

# Optional: use the FlashInfer attention backend
export VLLM_ATTENTION_BACKEND=FLASHINFER

vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000
```

Common flags:

| Flag                          | Purpose                                            |
| ----------------------------- | -------------------------------------------------- |
| `--host` / `--port`           | Bind address (default `0.0.0.0:8000`)              |
| `--tensor-parallel-size N`    | Shard the model across `N` GPUs                    |
| `--gpu-memory-utilization F`  | Fraction of GPU memory to use (default `0.9`)      |
| `--max-model-len N`           | Cap on context length                              |
| `--api-key <key>`             | Require this key in the `Authorization` header     |
| `--served-model-name <name>`  | Name to expose in the OpenAI API responses         |

Run `vllm serve --help` to see all options.

### Smoke-test with `curl`

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [{"role": "user", "content": "Hello!"}]
    }'
```

### Use it from the OpenAI Python client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)
```

### Benchmark with NVIDIA AIPerf

With the server running:

```bash
aiperf profile \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --endpoint-type chat \
    --url http://localhost:8000
```

See the [`aiperf` docs](https://github.com/ai-dynamo/aiperf) for full options.

---

## See also

- [README.old.md](README.old.md) — original upstream vLLM README
- [AGENTS.md](AGENTS.md) — contribution / tooling rules
- [build.sh](build.sh) — the install script described above
