# Decode throughput and contention microbenchmark

This microbenchmark measures vLLM CUDA-graph decode forward latency under
controlled decode shapes and optional LMCache-shaped GPU-to-GPU KV transfer
traffic. It is intended for research/debugging, not end-to-end serving
benchmarking.

## Implementation overview

The benchmark reuses the realistic CUDA-graph decode harness from
`microbench/cudagraph_moe/sweep`:

- vLLM initializes the model, KV cache, distributed groups, and CUDA graph
  capture through `LLM(..., distributed_executor_backend="external_launcher")`.
- `RealisticDecodeDriver` drives the captured decode forward path.
- A synthetic workload seeds vLLM's `InputBatch` with request rows, block
  tables, slot mappings, positions, and sequence lengths.
- By default, request ages are restored after every step, so every timed
  forward uses the same fixed `seq_len`. This makes the forward loop highly
  stable and easier to compare.
- Timed iterations are wrapped by CUDA profiler start/stop and NVTX ranges, so
  Nsight Systems can capture only the measured forwards.

The main entry point is:

```bash
cd /home/wxzheng/uccl/vllm-nvlink/microbench/throughput_bench
NPROC=<num_ranks> bash run.sh [args...]
```

`run.sh` changes directory to `microbench/` and runs:

```bash
torchrun --standalone --nproc_per_node="${NPROC}" -m throughput_bench.main "$@"
```

## Basic decode benchmark

### DP=EP decode

`batch-size` is per DP rank. With `--tp 1 --dp 2`, each of two decoder GPUs
decodes `B` requests per forward.

```bash
NPROC=2 bash run.sh \
  --tp 1 --dp 2 \
  --batch-size 600 \
  --seq-len 2048 \
  --max-num-seqs 600 \
  --max-decode-steps 2 \
  --gpu-memory-utilization 0.95 \
  --iters 50 \
  --warmup-iters 10 \
  --skip-final-barrier \
  --hard-exit-after-write \
  --output-json results/dp_ep2_baseline/b600_s2048.json
```

### TP decode

For TP=4 on 4 GPUs:

```bash
NPROC=4 bash run.sh \
  --model Qwen/Qwen3-235B-A22B-Instruct-2507 \
  --tp 4 --dp 1 \
  --batch-size 100 \
  --seq-len 1024 \
  --max-num-seqs 100 \
  --max-decode-steps 2 \
  --gpu-memory-utilization 0.95 \
  --iters 20 \
  --warmup-iters 5 \
  --skip-final-barrier \
  --hard-exit-after-write \
  --output-json results/tp_ep4_traffic_explore/json/qwen235b_tp4_b100_s1024.json
```

### Batch and sequence length sweeps

Example: sweep batch at fixed context length:

```bash
for B in 100 200 300 400 500 600 700; do
  NPROC=2 bash run.sh \
    --tp 1 --dp 2 \
    --batch-size "$B" \
    --seq-len 2048 \
    --max-num-seqs "$B" \
    --max-decode-steps 2 \
    --gpu-memory-utilization 0.95 \
    --iters 50 \
    --warmup-iters 10 \
    --skip-final-barrier \
    --hard-exit-after-write \
    --output-json "results/dp_ep2_baseline/b${B}_s2048.json"
done
```

Example: sweep sequence length at fixed batch:

```bash
for S in 1024 2048 4096 8192; do
  NPROC=2 bash run.sh \
    --tp 1 --dp 2 \
    --batch-size 300 \
    --seq-len "$S" \
    --max-num-seqs 300 \
    --max-decode-steps 2 \
    --gpu-memory-utilization 0.95 \
    --iters 50 \
    --warmup-iters 10 \
    --skip-final-barrier \
    --hard-exit-after-write \
    --output-json "results/dp_ep2_baseline/b300_s${S}.json"
done
```

## Important knobs

| Flag | Meaning |
|---|---|
| `--batch-size` | Number of active decode requests per DP rank. Each forward emits one token per request. |
| `--seq-len` | Fixed context length for each request. Controls FMHA/KV-read pressure. |
| `--max-num-seqs` | vLLM request capacity. For tight memory usage, set equal to `--batch-size`. |
| `--max-decode-steps` | Future KV blocks preallocated per request. For fixed-seq benchmarks, `2` is enough and saves memory. |
| `--tp`, `--dp`, `--pp` | vLLM parallelism. `NPROC` must equal `tp*dp*pp` unless LMCache sender ranks are added. |
| `--block-layout` | `contiguous` or `interleaved` synthetic KV block placement. |
| `--quantization` | Optional vLLM quantization method. `experts_int8` quantizes MoE experts while leaving Linear layers unquantized. |
| `--skip-final-barrier` | Avoid final distributed barrier after JSON/CSV are written. |
| `--hard-exit-after-write` | Use `os._exit(0)` after outputs are written to avoid vLLM/CUDA teardown hangs. |

## Throughput and plotting

Each JSON contains:

- per-rank `per_iter_ms`
- aggregate p50/p90/p99/min/max
- coefficient of variation
- logical decode throughput

For DP runs, aggregate logical throughput is:

```text
batch_size * dp / p50_seconds
```

Plot a directory of per-batch JSON files:

```bash
/home/wxzheng/miniconda3/envs/vllm-nvlink/bin/python \
  microbench/throughput_bench/plot_sweep.py \
  microbench/results/dp_ep2_baseline \
  --drop-first 1
```

Compare multiple result directories:

```bash
/home/wxzheng/miniconda3/envs/vllm-nvlink/bin/python \
  microbench/throughput_bench/plot_compare.py \
  --series baseline=microbench/results/dp_ep2_hbm/baseline \
  --series lmcache=microbench/results/dp_ep2_hbm/lmcache \
  --out-dir microbench/results/dp_ep2_hbm/figs \
  --drop-first 1
```

## LMCache-shaped KV transfer traffic

The benchmark can add synthetic LMCache-style D2D traffic while decode runs.

Implementation details:

- Extra torchrun ranks can be assigned as synthetic prefill sender ranks.
- Sender ranks run `lmcache_kv_runner.cu`.
- The copy primitive is `cuMemcpyDtoDAsync`.
- Transfer bursts are step-synchronized: a burst is triggered immediately before
  each timed decode forward.
- Traffic stats are written into the output JSON under `bg_traffic`.

### Rank layout

Without LMCache traffic:

```text
NPROC = tp * dp * pp
```

With LMCache sender ranks:

```text
NPROC = decoder_world_size + lmcache_prefill_ranks
decoder_world_size = tp * dp * pp
```

Example for `TP=1, DP=2`:

```text
NPROC=4
decoder ranks: GPU0, GPU1
prefill sender ranks: GPU2, GPU3
```

Example for `TP=4, DP=1`:

```text
NPROC=8
decoder ranks: GPU0..GPU3
prefill sender ranks: GPU4..GPU7
```

### Direction

```bash
--lmcache-direction ingress  # prefill -> decode, KV push
--lmcache-direction egress   # decode -> prefill, reverse/multi-turn stress
--lmcache-direction both     # both directions from each prefill sender rank
```

For `TP=1, DP=2, NPROC=4`:

```text
ingress: GPU2 -> GPU0, GPU3 -> GPU1
egress:  GPU0 -> GPU2, GPU1 -> GPU3
both:    both of the above
```

### Traffic size knobs

Prefer direct byte/count control for experiments:

```bash
--lmcache-chunk-bytes <BYTES>
--lmcache-chunks-per-burst <N>
```

Total transfer per sender per decode step:

```text
chunk_bytes * chunks_per_burst
```

Token-based controls are also available:

```bash
--lmcache-chunk-tokens 256
--lmcache-prefill-tokens-per-burst <TOKENS>
```

If `--lmcache-prefill-tokens-per-burst` is set, chunks per burst is:

```text
ceil(prefill_tokens_per_burst / chunk_tokens)
```

Do not combine `--lmcache-chunk-bytes` with
`--lmcache-prefill-tokens-per-burst`; use `--lmcache-chunks-per-burst` instead.

## LMCache traffic examples

### DP=EP=2 ingress traffic

```bash
NPROC=4 bash run.sh \
  --tp 1 --dp 2 \
  --batch-size 600 \
  --seq-len 2048 \
  --max-num-seqs 600 \
  --max-decode-steps 2 \
  --gpu-memory-utilization 0.95 \
  --iters 50 \
  --warmup-iters 10 \
  --lmcache-kv-traffic \
  --lmcache-prefill-ranks 2 \
  --lmcache-direction ingress \
  --lmcache-chunk-bytes 50331648 \
  --lmcache-chunks-per-burst 128 \
  --skip-final-barrier \
  --hard-exit-after-write \
  --output-json results/dp_ep2_lmcache/b600_s2048_ingress.json
```

### Bidirectional traffic

```bash
NPROC=4 bash run.sh \
  --tp 1 --dp 2 \
  --batch-size 600 \
  --seq-len 2048 \
  --max-num-seqs 600 \
  --max-decode-steps 2 \
  --gpu-memory-utilization 0.95 \
  --iters 50 \
  --warmup-iters 10 \
  --lmcache-kv-traffic \
  --lmcache-prefill-ranks 2 \
  --lmcache-direction both \
  --lmcache-chunk-bytes 50331648 \
  --lmcache-chunks-per-burst 128 \
  --skip-final-barrier \
  --hard-exit-after-write \
  --output-json results/dp_ep2_lmcache/b600_s2048_both.json
```

## Hotspot and incast experiments

Use `--lmcache-hotspot-rank` to send all prefill traffic to one decoder GPU.

For `TP=4, DP=1`, decoder ranks are GPU0..GPU3. With four prefill senders and
hotspot rank 0:

```text
GPU4 -> GPU0
GPU5 -> GPU0
GPU6 -> GPU0
GPU7 -> GPU0
```

Example:

```bash
NPROC=8 bash run.sh \
  --model Qwen/Qwen3-235B-A22B-Instruct-2507 \
  --tp 4 --dp 1 \
  --batch-size 100 \
  --seq-len 1024 \
  --max-num-seqs 100 \
  --max-decode-steps 2 \
  --gpu-memory-utilization 0.95 \
  --iters 20 \
  --warmup-iters 5 \
  --lmcache-kv-traffic \
  --lmcache-prefill-ranks 4 \
  --lmcache-direction ingress \
  --lmcache-hotspot-rank 0 \
  --lmcache-chunk-bytes 50331648 \
  --lmcache-chunks-per-burst 128 \
  --skip-final-barrier \
  --hard-exit-after-write \
  --output-json results/tp_ep4_incast/qwen235b_b100_s1024_senders4.json
```

### Fixed total volume incast

To keep total KV bytes per decode step constant while changing sender count,
divide chunks per sender by number of senders.

```bash
# 1 sender, total 128 chunks
--lmcache-prefill-ranks 1 --lmcache-chunks-per-burst 128

# 2 senders, total 128 chunks
--lmcache-prefill-ranks 2 --lmcache-chunks-per-burst 64

# 4 senders, total 128 chunks
--lmcache-prefill-ranks 4 --lmcache-chunks-per-burst 32
```

This isolates incast/fan-in effects from total transfer volume.

## Nsight Systems profiling

The decode loop already uses `cudaProfilerStart/Stop`, so use
`--capture-range=cudaProfilerApi`.

### Baseline Nsight

```bash
sudo -E env \
  PATH="/home/wxzheng/miniconda3/envs/vllm-nvlink/bin:$PATH" \
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
  nsys profile \
    --gpu-metrics-devices=0,1 \
    --gpu-metrics-frequency=10000 \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --force-overwrite=true \
    -o ../results/nsys_example/b600_s2048_baseline \
  env NPROC=2 bash run.sh \
    --tp 1 --dp 2 \
    --batch-size 600 \
    --seq-len 2048 \
    --max-num-seqs 600 \
    --max-decode-steps 2 \
    --gpu-memory-utilization 0.95 \
    --iters 20 \
    --warmup-iters 5 \
    --skip-final-barrier \
    --hard-exit-after-write \
    --output-json results/nsys_example/b600_s2048_baseline.json
```

### Nsight with LMCache sender ranks

Include all decoder and prefill GPUs in `--gpu-metrics-devices`.

```bash
sudo -E env \
  PATH="/home/wxzheng/miniconda3/envs/vllm-nvlink/bin:$PATH" \
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
  nsys profile \
    --gpu-metrics-devices=0,1,2,3 \
    --gpu-metrics-frequency=10000 \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --force-overwrite=true \
    -o ../results/nsys_example/b600_s2048_ingress \
  env NPROC=4 bash run.sh \
    --tp 1 --dp 2 \
    --batch-size 600 \
    --seq-len 2048 \
    --max-num-seqs 600 \
    --max-decode-steps 2 \
    --gpu-memory-utilization 0.95 \
    --iters 20 \
    --warmup-iters 5 \
    --lmcache-kv-traffic \
    --lmcache-prefill-ranks 2 \
    --lmcache-direction ingress \
    --lmcache-chunk-bytes 50331648 \
    --lmcache-chunks-per-burst 128 \
    --skip-final-barrier \
    --hard-exit-after-write \
    --output-json results/nsys_example/b600_s2048_ingress.json
```

## Extracting and analyzing Nsight traces

Export SQLite and extract per-iter/kernel metrics:

```bash
cd /home/wxzheng/uccl/vllm-nvlink/microbench/cudagraph_moe

nsys export -t sqlite \
  --force-overwrite=true \
  -o ../results/nsys_example/b600_s2048_ingress.sqlite \
  ../results/nsys_example/b600_s2048_ingress.nsys-rep

/home/wxzheng/miniconda3/envs/vllm-nvlink/bin/python -m sweep.extract_per_iter \
  ../results/nsys_example/b600_s2048_ingress.sqlite \
  --iter-out ../results/nsys_example/b600_s2048_ingress.iter.csv \
  --kernel-out ../results/nsys_example/b600_s2048_ingress.kernels.csv \
  --kernel-hw-out ../results/nsys_example/b600_s2048_ingress.kernel_hw.csv
```

Useful plotters:

```bash
# FMHA/fused_moe distributions
/home/wxzheng/miniconda3/envs/vllm-nvlink/bin/python \
  microbench/throughput_bench/plot_kernel_distributions.py \
  microbench/results/dp_ep2_hbm_nsys

# Communication kernel distributions
/home/wxzheng/miniconda3/envs/vllm-nvlink/bin/python \
  microbench/throughput_bench/plot_comm_kernel_distributions.py \
  microbench/results/dp_ep2_hbm_nsys \
  --seq-len 2048

# Kernel duration split by overlap with P2P D2D copies
/home/wxzheng/miniconda3/envs/vllm-nvlink/bin/python \
  microbench/throughput_bench/plot_kernel_overlap.py \
  --baseline-sqlite microbench/results/tp_ep4_traffic_explore/nsys/qwen235b_tp4_b100_s1024_baseline.sqlite \
  --traffic-sqlite microbench/results/tp_ep4_traffic_explore/nsys/qwen235b_tp4_b100_s1024_ingress.sqlite \
  --tag qwen235b_tp4_b100_s1024_ingress \
  --out-dir microbench/results/tp_ep4_traffic_explore/figs/overlap
```

## Notes and caveats

- `--hard-exit-after-write` can produce Python resource-tracker warnings. This
  is expected; it avoids shutdown hangs after JSON/CSV are already written.
- For fixed-seq decode, `--max-decode-steps 2` is safe because the driver
  restores request age after every step unless `--advance-age` is set.
- Weight quantization usually reduces expert/GEMM time but does not
  proportionally reduce activation communication volume.
- With `TP > 1`, LMCache traffic requires `--lmcache-chunk-bytes` because
  automatic KV-bytes-per-token inference is TP=1-only.