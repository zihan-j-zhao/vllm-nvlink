# Playground

Scratch space for experiments related to vLLM on NVLink-connected hardware.
Each experiment lives in its own subdirectory (currently `moe_a2a/`) and
follows a shared layout for inputs, logs, and outputs.

## Directory layout

```
playground/
├── data/                 # experiment-ready datasets
├── log/<experiment>/     # raw logs from runs (one folder per run)
├── out/<experiment>/     # collected results (CSV, JSON, plots, …)
└── <experiment>/         # the experiment's scripts
```

Conventions:

- `data/`, `log/`, `out/` are git-ignored for bulky contents — commit
  only small reproducible artifacts; large files should be
  regeneratable from scripts in `<experiment>/`.
- Hugging Face datasets stay under `~/.cache/huggingface/`. Use
  `data/<experiment>/` only for *derived* files (subsets, reformatted
  versions, synthetic data).
- Per-run subfolders should be timestamped (`<YYYY-MM-DDTHH-MM-SSZ>`)
  so multiple runs of the same experiment don't overwrite each other.

---

## Experiments

### `moe_a2a/` — MoE all-to-all transfer profiling

End-to-end pipeline that measures per-rank, per-layer, per-step MoE
all-to-all transfer (bytes and tokens, in and out) on the AG/RS naive
EP backend, driven by an online ShareGPT workload.

**Scope this experiment is hard-asserted against (see
[`vllm/distributed/moe_a2a_profiler.py`](../vllm/distributed/moe_a2a_profiler.py)):**

- DP=2, EP=2 (single node), `--enforce-eager`, `pcp_size=1`.
- `all2all_backend` in `{naive, allgather_reducescatter}` — other
  backends (DeepEP, Mori, NIXL, FlashInfer) bypass the EP-communicator
  choke point and would produce empty traces.
- The launcher pins `--moe-backend triton`; the auto-selected
  `flashinfer_trtllm` backend on B200 also bypasses the choke point.

If any precondition fails the profiler disables itself and prints
`[moe_a2a_profiler] disabled because preconditions failed: ...` to the
server log.

**Files:**

- [`start_server.sh`](moe_a2a/start_server.sh) — launches the vLLM
  OpenAI server with the profiler enabled.
- [`run_aiperf.sh`](moe_a2a/run_aiperf.sh) — drives 1000 ShareGPT
  requests at 20 req/s (Poisson) against the server.
- [`extract_per_step.py`](moe_a2a/extract_per_step.py) — folds raw
  per-rank JSONL traces into a long-format per-`(rank, step, layer)`
  CSV; drops warmup/idle rows by default.
- [`plot_per_rank.py`](moe_a2a/plot_per_rank.py) — produces one PNG
  per rank with a 2×2 grid of `{dispatch,combine} × {in,out}` byte
  scatter plots vs scheduled tokens.

#### How to run

Pick a UTC timestamp once and reuse it across the steps so files
don't collide with earlier runs:

```bash
RUN_TS=$(date -u +%Y-%m-%dT%H-%M-%SZ)
```

**1. Launch the server** (terminal A, keep it running):

```bash
mkdir -p playground/log/moe_a2a/sharegpt_${RUN_TS}
CUDA_VISIBLE_DEVICES=6,7 \
    LOG_DIR=playground/log/moe_a2a/sharegpt_${RUN_TS} \
    ./playground/moe_a2a/start_server.sh \
    > playground/log/moe_a2a/sharegpt_${RUN_TS}/server.log 2>&1 &
```

Override env vars as needed: `MODEL`, `PORT` (default 8000),
`SERVED_MODEL_NAME`, `LOG_DIR`, `PY` (path to a Python interpreter
with vLLM installed; defaults to the `vllm-nvlink` conda env). Wait
for `/v1/models` to return 200 before continuing — model load takes
~1–2 min once FlashInfer cubins are cached.

**2. Drive load with aiperf** (terminal B):

```bash
./playground/moe_a2a/run_aiperf.sh \
    > playground/out/aiperf_${RUN_TS}.log 2>&1
```

Override env vars: `URL` (default `http://localhost:8000`),
`REQUEST_RATE` (20), `REQUEST_COUNT` (1000), `RANDOM_SEED` (42),
`OUT_DIR` (default `playground/out/aiperf/<UTC>`). The script
uses `aiperf`'s built-in `--public-dataset sharegpt`, so ShareGPT
is downloaded automatically on first run (≈580 MB, cached under
`~/.cache/huggingface/`).

Expect ~5 min wall time for the benchmark (≈313 s) plus tail latency.

**3. Stop the server with SIGTERM** so the profiler flushes:

```bash
pkill -TERM -f vllm.entrypoints.openai.api_server
```

**4. Extract the per-step CSV** from the raw JSONL traces:

```bash
/root/miniconda3/envs/vllm-nvlink/bin/python \
    playground/moe_a2a/extract_per_step.py \
    --jsonl-glob "playground/log/moe_a2a/sharegpt_${RUN_TS}/moe_a2a_rank*.jsonl" \
    --num-layers 48 \
    --output playground/out/moe_a2a_sharegpt_${RUN_TS}.csv
```

`--num-layers 48` matches Qwen3-30B-A3B; change for other MoE models.
The extractor drops warmup/idle rows by default — pass
`--keep-warmup` to retain them, or `--no-assert` to skip per-step
invariant checks.

**5. Plot per-rank figures:**

```bash
/root/miniconda3/envs/vllm-nvlink/bin/python \
    playground/moe_a2a/plot_per_rank.py \
    --csv playground/out/moe_a2a_sharegpt_${RUN_TS}.csv \
    --output-dir playground/out/figures
```

Writes `playground/out/figures/moe_a2a_rank{0,1}.png`. Each PNG is a
2×2 grid of `Dispatch (outbound)`, `Dispatch (inbound)`,
`Combine (outbound)`, `Combine (inbound)` scattered against the
per-step token count (x-axis capped at 2048; the long tail of
chunked-prefill outliers above that compresses the bulk).

#### Output schema

The per-step CSV has one row per `(rank, step_id, layer_idx)`:

| Column | Meaning |
|---|---|
| `rank` | EP rank (0 or 1). |
| `step_id` | Per-worker monotonic counter; **not** globally aligned across ranks (see caveat below). |
| `layer_idx` | MoE layer index (0..47 for Qwen3-30B-A3B). |
| `scheduled_tokens` | `SchedulerOutput.total_num_scheduled_tokens` on this rank for this step. |
| `world_size` | EP world size (= 2). |
| `dispatch_in_tokens` / `dispatch_in_bytes` | Local tensors fed into the AG dispatch. |
| `dispatch_out_tokens` / `dispatch_out_bytes` | Tensors received from the AG (own + peers). |
| `combine_in_tokens` / `combine_in_bytes` | Full pre-RS tensor produced by experts. |
| `combine_out_tokens` / `combine_out_bytes` | Local shard received from RS. |

#### Caveats

- **`step_id` is per-rank, not global.** Each worker increments its
  own counter inside `execute_model`; DP coordination causes the two
  workers to advance at different rates. Joining rows from rank 0
  and rank 1 on `step_id` is *not* safe — they do not represent the
  same forward pass.
- **Per-layer values are identical within a step under the AG/RS
  backend** because the AG/RS plumbing ships the whole batch to
  every rank regardless of routing. Routing-dependent per-layer
  variation only shows up under true all-to-all backends (DeepEP /
  Mori / NIXL), which the current profiler does not hook.
- **CUDA graphs hide the recorder.** `--enforce-eager` is mandatory;
  the launcher sets it.
- **Pre-`execute_model` warmup and DP-idle bubbles** are tagged
  `step_id=-1` / `scheduled_tokens<=0` and are dropped by the
  extractor unless `--keep-warmup` is passed.
