# `cudagraph_moe/` — CUDA-graph decode microbenchmark

Microbenchmark for the **decode CUDA-graph replay** step that vLLM dispatches
once warmup + capture are done, for the Qwen3-30B-A3B (MoE) + AgRS setup.

## What this measures

At steady-state decoding, every forward in vLLM is a `cudaGraphLaunch` of a
graph that was captured during `compile_or_warm_up_model() → capture_model()`.
The end-to-end benchmark (`vllm bench latency`) buries that step inside
prefill, sampling, scheduling, etc.

This script:

1. Constructs `LLM(..., distributed_executor_backend="external_launcher")` so
   vLLM does all the heavy lifting itself: distributed init, model load, KV
   cache allocation, kernel warmup, and CUDA-graph capture across all decode
   batch sizes.
2. Reaches into
   `llm.llm_engine.model_executor.driver_worker.worker.model_runner`
   and calls `_dummy_run(num_tokens=B, uniform_decode=True)` directly. With
   the dispatcher unbiased, that path hits the captured `FULL` (or
   `PIECEWISE`) CUDA graph for batch size `B`.
3. Times each call with paired `torch.cuda.Event(enable_timing=True)`,
   warms up first, and reports `p50/p90/p99/min/max` per rank.

The result is a tight, on-device measurement of one decode forward — exactly
the thing vLLM replays at steady state, including MoE EP all-to-all.

## Layout

| File | What it is |
| --- | --- |
| `decode_cudagraph_bench.py` | The benchmark driver. Imports vLLM, drives `_dummy_run`. |
| `run.sh`                    | `torchrun --standalone --nproc_per_node=...` launcher. |
| `results/`                  | Default location for the JSON summary. |

No files in `vllm/` are modified.

## Run it

```bash
# Defaults: 2 GPUs (0,1), DP=2, EP, FLASHINFER, triton MoE, AgRS, 5 timed iters.
bash run.sh

# Pick batch sizes and how many decode steps to time per batch size.
bash run.sh --batch-sizes 1,4,16,64 --iters 50 --warmup-iters 10

# Different GPU pair.
CUDA_VISIBLE_DEVICES=4,5 bash run.sh

# Eager baseline (skip CUDA-graph capture & replay).
bash run.sh --enforce-eager --iters 20

# Custom model location.
MODEL=/data/models/Qwen3-30B-A3B-Instruct-2507 bash run.sh
```

After capture finishes (5–20 s typically), each rank prints something like:

```
[rank 0] >>> bench decode batch_size=16
[rank 0]   p50=0.823 ms p90=0.841 ms p99=0.867 ms min=0.811 max=0.871 ...
```

and rank 0 writes `results/cudagraph_decode.json` with per-rank rows for each
batch size (including the full per-iteration latency list).

## Notes / caveats

- `--batch-sizes` are values of `num_tokens` per rank, treated as uniform
  decode. Batch size must be ≤ `--max-num-seqs` (default 256).
- The selected batch sizes should already be in vLLM's CUDA-graph capture
  set (the standard powers-of-two/steps that vLLM captures by default).
  If a value isn't captured, the dispatcher falls back to the next captured
  size + padding, which is still a graph replay (just at the padded size).
- Per-iteration latency is GPU time only (`cuda.Event.elapsed_time`), but the
  trailing `synchronize()` waits for the graph to finish — there's no host
  overhead in the per-iter timing.
- All ranks must call `_dummy_run` in lockstep for DP coordination (the
  benchmark does this by structure).
- The script doesn't touch vLLM source, only `import`s public-ish entry points.
