# E2E AGRS NIXL Experiment

This folder contains the runnable scripts for the end-to-end P/D-disaggregated MoE experiment with NIXL KV transfer telemetry. The current story is: higher per-engine-step KV transfer pressure, especially estimated transfer volume per unit time, correlates with higher decode engine-step p90/p99 tail latency.

## What This Experiment Runs

The service is a three-process P/D setup:

- Prefill vLLM server: NIXL KV producer, default GPUs `4,5`
- Decode vLLM server: NIXL KV consumer, default GPUs `6,7`
- Local proxy: forwards OpenAI-compatible requests through prefill, copies `kv_transfer_params`, then streams decode output back to the client

Default model:

```bash
Qwen/Qwen3-30B-A3B-Instruct-2507
```

Default runtime shape:

```text
per-side DP = 2
TP = 1
PP = 1
expert parallel = enabled when per-side DP > 1
CUDA graphs = enabled
prefix caching = disabled
NIXL connector = enabled
```

## Files

- `start_server.sh`: starts prefill, decode, and proxy; enables JSONL trace capture by default.
- `run_aiperf.sh`: drives the proxy with ShareGPT via AIPerf.
- `proxy.py`: NIXL-aware P/D proxy with streaming Content-Type forwarding for AIPerf.
- `analyze_overlap.py`: converts trace JSONL into engine-step, phase, and NIXL transfer overlap CSVs.
- `run_window_analysis.sh`: reprocesses one traced run into fixed windows.
- `plot_phase_timeline.py`: draws a short wall-clock P/D phase timeline with prefill/decode engine, forward, KV-cache, and NIXL-transfer lanes.
- `plot_transfer_volume_slowdown.py`: regenerates pressure, estimated-transfer-volume, normalized-transfer-volume, and slowdown figures.
- `plot_itl_timeseries.py`: older wall-clock ITL timeline plotter.

The trace instrumentation itself lives in vLLM code, mainly:

- `vllm/distributed/kv_transfer/kv_connector/v1/nixl/_trace.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py`
- `vllm/v1/engine/core.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/worker/kv_connector_model_runner_mixin.py`

## Environment Setup

Activate the environment used to build this vLLM tree:

```bash
cd /home/wxzheng/uccl/vllm-nvlink
source /home/wxzheng/miniconda3/etc/profile.d/conda.sh
conda activate vllm-nvlink
```

Check required commands and Python packages:

```bash
which python
which aiperf
python - <<'PY'
import fastapi, httpx, uvicorn
from nixl._api import nixl_agent
print('ok')
PY
```

If the NIXL import fails, install/fix the NIXL Python package in this environment before running the service.

## Run The Service

Terminal A:

```bash
cd /home/wxzheng/uccl/vllm-nvlink
source /home/wxzheng/miniconda3/etc/profile.d/conda.sh
conda activate vllm-nvlink

PD_TRACE=1 \
LOG_DIR=playground/log/e2e_agrs_nixl/telemetry_5k_starttime_$(date -u +%Y-%m-%dT%H-%M-%SZ) \
bash playground/e2e_agrs_nixl/start_server.sh
```

The important default output is:

```text
$LOG_DIR/pd_trace/role=*.jsonl
```

Those JSONL files contain aligned CPU timestamps for engine steps, model-runner phases, NIXL post points, and NIXL transfer telemetry.

The current trace does not contain attention or MoE kernel timestamps. Model work is visible as the `forward` launch-scope interval, while actual CUDA kernel order inside attention/MoE remains hidden unless captured with Nsight/CUPTI or explicit CUDA-event instrumentation.

Useful overrides:

```bash
PREFILL_GPUS=4,5
DECODE_GPUS=6,7
PREFILL_PORT=8100
DECODE_PORT=8200
PROXY_PORT=8000
GPU_MEM_UTIL=0.85
EXPERT_PARALLEL=1
ALL2ALL_BACKEND=allgather_reducescatter
PD_TRACE=1
VLLM_PD_TRACE_MODEL_PHASES=1
ENFORCE_EAGER=1
```

`VLLM_PD_TRACE_MODEL_PHASES=1` adds Qwen3-MoE per-layer CPU timestamp events for attention, MLP, and sparse-MoE envelopes. These are ordering/overlap events, not synchronized CUDA kernel timings. They are most useful for short educational timeline runs; with CUDA graphs, Python events inside the captured model can describe capture-time behavior rather than every replay.

For the clearest per-layer educational timeline, pair it with `ENFORCE_EAGER=1` for a short run. Leave eager mode off for production-like CUDA-graph performance runs.

To try NIXL EP for MoE dispatch/combine instead of the default AGRS path:

```bash
ALL2ALL_BACKEND=nixl_ep \
PD_TRACE=1 \
LOG_DIR=playground/log/e2e_agrs_nixl/nixl_ep_$(date -u +%Y-%m-%dT%H-%M-%SZ) \
bash playground/e2e_agrs_nixl/start_server.sh
```

This requires the optional `nixl_ep` Python package. In this environment it is installed, and the target model's hidden size (`2048`) is in the supported NIXL EP hidden-size set.

## Drive AIPerf Load

Terminal B, after `/healthcheck` on the proxy is ready:

```bash
cd /home/wxzheng/uccl/vllm-nvlink
source /home/wxzheng/miniconda3/etc/profile.d/conda.sh
conda activate vllm-nvlink

REQUEST_RATE=20 \
REQUEST_COUNT=5000 \
MAX_OUTPUT_TOKENS=500 \
OUT_DIR=playground/out/aiperf_pd/e2e_agrs_nixl/telemetry_5k_starttime_main_$(date -u +%Y-%m-%dT%H-%M-%SZ) \
bash playground/e2e_agrs_nixl/run_aiperf.sh
```

The previous final run used roughly:

```text
request rate: 20 rps
request count: 5000
max output tokens: 500
proxy URL: http://127.0.0.1:8000
```

## Analyze A Trace Run

Set `TRACE_DIR` to the trace directory produced by `start_server.sh`:

```bash
TRACE_DIR=playground/log/e2e_agrs_nixl/<RUN>/pd_trace \
OUT_ROOT=playground/out/overlap/e2e_agrs_nixl \
bash playground/e2e_agrs_nixl/run_window_analysis.sh
```

This creates five window folders:

```text
telemetry_5k_starttime_union_early_120
telemetry_5k_starttime_union_mid_180
telemetry_5k_starttime_union_mid_240
telemetry_5k_starttime_union_mid_300
telemetry_5k_starttime_union_late_360
```

Each window contains:

```text
overlap_engine_step_rows.csv
overlap_phase_rows.csv
overlap_xfer_rows.csv
overlap_summary.md
overlap_timeline.png
```

For one-off analysis, call the analyzer directly:

```bash
python playground/e2e_agrs_nixl/analyze_overlap.py \
  --trace-dir "$TRACE_DIR" \
  --out-dir playground/out/overlap/e2e_agrs_nixl/example_window \
  --window-start-sec 240 \
  --window-sec 60 \
  --recv-interval-source nixl \
  --nixl-start start_time \
  --no-itl
```

`--nixl-start start_time` uses NIXL telemetry `startTime` plus the per-process boot offset, which was verified to share the trace clock domain.

## Plot A Short P/D Timeline

For a visual microscope over a few seconds of one run:

```bash
python playground/e2e_agrs_nixl/plot_phase_timeline.py \
  --trace-dir playground/log/e2e_agrs_nixl/<RUN>/pd_trace \
  --window-sec 2.0 \
  --nixl-start start_time
```

By default the script chooses the densest NIXL-transfer window. To inspect a fixed offset from the first traced event:

```bash
python playground/e2e_agrs_nixl/plot_phase_timeline.py \
  --trace-dir playground/log/e2e_agrs_nixl/<RUN>/pd_trace \
  --window-start-sec 240 \
  --window-sec 1.0
```

Outputs are written under:

```text
playground/out/timeline/e2e_agrs_nixl/<RUN>/
```

Main outputs:

```text
pd_phase_timeline.png
pd_phase_timeline.pdf
pd_phase_timeline_summary.md
```

Interpretation notes:

- `NIXL KV transfer` bars use NIXL `xferDuration` and `startTime`, aligned through the process boot offset.
- `KV recv CPU scope` bars show the host-side `recv_start` to `recv_done` interval.
- `forward` bars show the CPU-side model-forward envelope; they are not attention/MoE kernel durations.
- With `VLLM_PD_TRACE_MODEL_PHASES=1`, the plotter also shows `attention`, `mlp`, `moe`, `moe_router`, and `moe_experts` CPU phase bars for Qwen3-MoE layers.

## Generate Slowdown Figures

After window analysis:

```bash
python playground/e2e_agrs_nixl/plot_transfer_volume_slowdown.py \
  --window-root playground/out/overlap/e2e_agrs_nixl
```

Main outputs go to:

```text
playground/out/overlap/e2e_agrs_nixl/telemetry_5k_starttime_union_report
```

Most useful final figures:

```text
engine_step_tail_by_estimated_transfer_mb_steady.png
engine_step_tail_slowdown_by_estimated_transfer_mb_steady.png
engine_step_tail_survival_by_estimated_transfer_mb_steady.png
engine_step_tail_by_normalized_transfer_volume_steady.png
engine_step_tail_slowdown_by_normalized_transfer_volume_steady.png
```

Most useful CSVs/reports:

```text
engine_step_pressure_rows.csv
engine_step_estimated_transfer_mb_bucket_summary.csv
engine_step_normalized_transfer_volume_bucket_summary.csv
e2e_agrs_nixl_slowdown_report.md
```

## Metrics

For each engine step, `plot_transfer_volume_slowdown.py` clips each overlapping NIXL transfer interval to the engine-step interval.

Estimated data transferred during the step:

```text
estimated_MB = sum(bytes_transferred * clipped_overlap_time / transfer_duration) / 1e6
```

Normalized transfer volume:

```text
normalized_MB_per_ms = estimated_MB / engine_step_duration_ms
```

`MB/ms` is numerically equivalent to `GB/s`, so the normalized metric behaves like average in-step transfer throughput over the full engine-step interval.

Other overlap metrics from `analyze_overlap.py`:

```text
overlap_ms: summed transfer exposure; concurrent regions are counted once per transfer
union_overlap_ms: merged wall-clock transfer-active time; concurrent regions are counted once
overlap_intensity: overlap_ms / union_overlap_ms
max_active_recv: maximum simultaneous NIXL transfer intervals inside the measured interval
```

## Current Interpretation

The most defensible high-level result is:

```text
Median decode engine-step duration is stable, but p90/p99 tail latency increases when estimated in-step NIXL transfer volume and normalized transfer volume are high.
```

The raw `union_overlap_ms` plot is useful as a coverage diagnostic, but it is not the main slowdown plot because `union_overlap_ms` is bounded by engine-step duration.

The estimated transfer-volume plots are better for the slowdown story because they include both overlap time and bytes transferred. The normalized plots help address the concern that longer engine steps naturally have more time to overlap transfer activity.

## Exploring AGRS vs NIXL EP

The current baseline uses vLLM's default MoE all-to-all backend, `allgather_reducescatter`, which implements dispatch as all-gather and combine as reduce-scatter. That is the AGRS path.

There is already a separate `nixl_ep` backend in this tree:

```text
--all2all-backend=nixl_ep
```

This is not literally AGRS implemented on top of the generic NIXL KV connector. It is a specialized NIXL EP MoE backend that plugs into the fused-MoE prepare/finalize path and uses `nixl_ep.Buffer.dispatch()` / `nixl_ep.Buffer.combine()`.

For a fair comparison, run two otherwise identical 5k experiments:

```bash
# Baseline AGRS
ALL2ALL_BACKEND=allgather_reducescatter \
PD_TRACE=1 \
LOG_DIR=playground/log/e2e_agrs_nixl/agrs_$(date -u +%Y-%m-%dT%H-%M-%SZ) \
bash playground/e2e_agrs_nixl/start_server.sh

# Candidate NIXL EP
ALL2ALL_BACKEND=nixl_ep \
PD_TRACE=1 \
LOG_DIR=playground/log/e2e_agrs_nixl/nixl_ep_$(date -u +%Y-%m-%dT%H-%M-%SZ) \
bash playground/e2e_agrs_nixl/start_server.sh
```

Use the same AIPerf command for both runs:

```bash
REQUEST_RATE=20 \
REQUEST_COUNT=5000 \
MAX_OUTPUT_TOKENS=500 \
bash playground/e2e_agrs_nixl/run_aiperf.sh
```

Expected things to verify in logs:

```text
Using NixlEPPrepareAndFinalize
all2all backend = nixl_ep
no hidden-size / quantization assertion failures
```

Main risk: `nixl_ep` is a specialized EP backend with its own constraints. It currently supports a fixed hidden-size set and has some quantization restrictions. It should be plausible for Qwen3-30B-A3B (`hidden_size=2048`), but it still needs an actual server boot and short AIPerf smoke test before a full 5k run.

## Artifact Retention

For the compact final evidence set, keep:

```text
playground/out/overlap/e2e_agrs_nixl/telemetry_5k_starttime_union_report
playground/out/overlap/e2e_agrs_nixl/telemetry_5k_starttime_union_mid_240
playground/out/overlap/e2e_agrs_nixl/telemetry_5k_starttime_union_mid_300
playground/out/aiperf_pd/e2e_agrs_nixl/<final 5k run>
```

Optionally keep all five `telemetry_5k_starttime_union_*` windows if you want to regenerate five-window union-overlap plots.

Older `telemetry_example_*`, non-union `telemetry_5k_starttime_*`, `steady_5k_*`, and `overlap_manual_*` folders were exploratory/debug outputs and can be archived separately.
