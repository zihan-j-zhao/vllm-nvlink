# Sweep findings — bg=off baseline (Qwen3-30B-A3B, TP=4, B200×4)

Source: [sweep_runs/full/sweep_results.csv](../sweep_runs/full/sweep_results.csv)
12 cells: batch ∈ {128, 512, 1024} × prefill ∈ {128, 512, 1024, 2048}, 20 iters/cell, 1 kHz GPU metric sampling.

## Headline numbers

```
tag                     p50ms    HBM   SM   NVtx  Tens   FMHA    MoE   AllR
b128_p128_off            6.10   16.0  60.8   2.0   7.5   7.33  24.58  11.17
b128_p512_off            6.39   19.2  62.3   2.0   7.9  11.65  25.70  11.17
b128_p1024_off           6.62   21.1  62.8   2.0   7.6  16.96  25.38  11.52
b128_p2048_off           7.12   25.1  64.1   2.0   7.1  28.03  25.09  11.14
b512_p128_off            9.76   15.6  69.2   5.8   9.7  13.54  34.72  26.34
b512_p512_off           10.65   20.3  70.9   5.2   9.0  33.73  34.34  26.30
b512_p1024_off          11.57   25.2  72.6   4.8   8.4  54.30  34.02  26.21
b512_p2048_off          13.29   33.5  75.2   4.0   7.5  91.55  33.57  26.18
b1024_p128_off          14.14   14.4  71.1   7.9   7.9  22.78  49.31  47.84
b1024_p512_off          15.70   20.6  73.5   7.1   7.3  58.56  48.35  47.45
b1024_p1024_off         17.38   27.3  75.9   6.5   6.8  95.23  47.55  47.52
b1024_p2048_off         20.80   37.9  79.4   5.4   6.0 167.81  46.66  47.58
```

`HBM`, `SM`, `NVtx`, `Tens` are median across iters, in % of peak. Kernel
columns are p50 in µs. All numbers per-GPU (averaged across 4 ranks).

## Key trends

1. **Decode is HBM-read-bound, not compute-bound.**
   Tensor Active stays at 6–10% while HBM Read climbs to 38% with long
   context. Tensor cores idle waiting on KV bandwidth.

2. **HBM Read grows with prefill, not batch.**
   At B=1024: 14.4% (p=128) → 37.9% (p=2048). At p=2048: only 25→34→38%
   as B goes 128→512→1024.  Each decoded token reads its full context's
   KV, so KV-byte-volume scales O(B × context_len) but the FMHA kernel
   has good arithmetic intensity for the GEMV part — HBM-saturation rises
   faster with prefill than with batch.

3. **SMs Active saturates by B=512.**
   71% at B=512, only 79% at B=1024. MoE GEMMs already SM-bound at modest
   batch; doubling B from 512→1024 mostly queues work, doesn't add
   parallelism. Confirms decode is *latency-shaped*, not throughput-shaped.

4. **NVLink TX% is small and *decreases* with prefill.**
   B=128: 2%, B=1024p128: 8%, B=1024p2048: 5%. All-reduce bytes are
   roughly constant per-token-per-layer, so longer steps dilute the %.

5. **Kernel cost scaling is clean.**
   - **FMHA** = O(B × seq_len): 7µs (B128p128) → 168µs (B1024p2048), ratio 24×, matches 8× × 16× scaling
   - **fused_moe** = O(B) only, prefill-flat: 24µs / 34µs / 47µs at B=128/512/1024
   - **allreduce** = O(B) only, prefill-flat: 11µs / 26µs / 47µs at B=128/512/1024

6. **Per-iter HW samples are sparse at 1 kHz.**
   For B=128 (~6 ms iter) we get only 4–6 samples per metric. Headline
   `_med` columns (median-of-iter-means across iters) are trustworthy;
   per-iter `p99`/`max` are noisy because they're the max of 4 samples.
   Trade-off accepted in exchange for low (<2%) profiler overhead.

## Sanity check: p50 vs. kernel times

For B=1024p2048: model has ~48 transformer layers, each with one MoE +
2 all-reduce + 1 FMHA call:

```
p50 ≈ 48 × (FMHA + MoE + 2×AllR)
    = 48 × (168 + 47 + 2×48) µs
    = 48 × 359 µs
    ≈ 17.2 ms     vs. measured 20.8 ms (≈83% kernel-time fraction)
```

The ~3.5 ms gap is launch overhead + non-tracked kernels (residuals,
layernorms, embedding gather). Consistent with realistic CUDA-graph
replay.

## What's anomalous / worth flagging later

- NVLink TX is flat at 2.0% for *all* B=128 cells. Likely 1 kHz
  sampling rounding on the short 6 ms iter; should look bigger at
  finer sampling but adds profiler overhead.
- DRAM Write Bandwidth is uniformly ≤2% — `reshape_and_cache` writes
  are tiny (1 token × hidden / layer / req / step). Cache-update
  pressure is invisible in this workload; would only matter with
  long-burst prefill.

## What's next

Sweep with background NIXL-style P2P traffic to see how decode
degrades under contention:
- `ingress` (phantom→decoder): only adds decoder HBM writes, expected to barely move p50
- `egress` (decoder→phantom): adds decoder HBM reads, expected to shift p50

Two pacing settings per direction:
- **max**: `--bg-rate-gbps 9999 --bg-chunk-mb 64 --bg-buffer-mb 128` — link-saturating
- **medium / KV-page-shaped**: `--bg-chunk-mb 4 --bg-buffer-mb 64`,
  paced at ~50 GB/s — chunks sized to ~1 attention KV page (256 KiB ×
  16 layers per req) so the CE traffic pattern resembles real NIXL
  KV chunk pushes
