# Sweep findings — bg traffic impact (Qwen3-30B-A3B, TP=4, B200×4)

Source: [sweep_runs/bg/sweep_results.csv](../sweep_runs/bg/sweep_results.csv)
60 cells: 12 (B, prefill) × 5 (bg=off, ingress_med, ingress_max, egress_med, egress_max).

## TL;DR

| bg profile     | NVLink dir | Achieved Gbps/rank | p50 slowdown vs off |
|----------------|------------|-------------------:|--------------------:|
| `off`          | -          | 0                  | baseline            |
| `ingress_med`  | RX         | 1–4                | **+0.4 to +1.1%**   |
| `ingress_max`  | RX         | 12–30              | **+8 to +12%**      |
| `egress_med`   | TX         | 1–5                | **+1.5 to +1.9%**   |
| `egress_max`   | TX         | 13–30              | **+10 to +20%**     |

Egress hurts decode **more** than ingress at the same achieved bandwidth,
and that effect **grows with batch size**: at B=1024 egress_max takes
+19.7% p50 vs ingress_max's +9.1%. **At B=128 the two are roughly tied
(~10–11% each).**

## Slowdown heatmap (p50 vs off baseline)

See [figs_bg/bg_p50_heatmap.png](../sweep_runs/bg/figs_bg/bg_p50_heatmap.png).

```
ingress_med  ingress_max  egress_med  egress_max
[+0.4..+1%]  [+8..+12%]   [+1..+2%]   [+10..+20%]
```

Key gradient: **egress_max is the only profile where slowdown grows
with batch size** (10% → 20% as B goes 128 → 1024). The other three
profiles have slowdown that's roughly batch-invariant.

## What each profile loads up

(from [figs_bg/bg_hw_vs_batch_p2048.png](../sweep_runs/bg/figs_bg/bg_hw_vs_batch_p2048.png))

| Profile        | NVLink TX | NVLink RX | HBM Read | HBM Write | SMs |
|----------------|----------:|----------:|---------:|----------:|----:|
| off (B=1024)   | 5%        | 5%        | 38%      | 5%        | 79% |
| ingress_med    | unchanged | +8 pp     | unchanged| unchanged | unchanged |
| ingress_max    | unchanged | **+54 pp**| **-4 pp**| **+7 pp** | unchanged |
| egress_med     | +10 pp    | unchanged | unchanged| unchanged | unchanged |
| egress_max     | **+50 pp**| unchanged | **+1 pp**| -1 pp     | **-3 pp** |

The most telling rows:
- **ingress_max reduces decoder HBM Read by ~4 pp**, despite the
  decoder GPU only doing CE *writes* in that direction. The cause is
  HBM-controller arbitration: incoming CE writes from the peer steal
  some of the controller's read-port time, slowing FMHA's KV reads.
- **egress_max increases decoder HBM Read by ~1 pp and reduces SMs
  Active by ~3 pp.** The CE engine on the decoder issues reads from
  the same HBM ports FMHA uses; some SM time is wasted waiting on the
  contented reads.

## Why egress hurts more than ingress (especially at high B)

Decode is HBM-read-bound at B=1024, p=2048 (38% HBM Read in baseline).
Adding more readers (egress) is strictly worse than adding more writers
(ingress) because:
- B200's HBM3e has plenty of *write* port headroom for the ~5% decoder
  baseline; an extra +7pp from ingress_max stays below saturation.
- The *read* port is at 38% baseline and any extra reads land on the
  same controller queues as FMHA. The egress CE engine reads ~20 GB/s
  of bytes from local HBM, directly competing.

At low B (=128) baseline HBM Read is only 16–25%, so neither direction
hits the wall. Both profiles cost about the same ~10–11% p50.

## Achieved-Gbps reality check

The `max` profiles only push ~13–30 Gbps/rank — far below the
~900 Gbps NVLink can sustain on B200. Two reasons:
1. **CPU thread overhead.** The bg thread does ~150 ops/sec (64 MiB
   each at 30 Gbps) but each op crosses the GIL boundary and contends
   with vLLM's hot launch thread. The 1 kHz nsys sampling adds another
   constant tax.
2. **Buffer depth.** `--bg-buffer-mb 128` at 64 MiB chunks means only 2
   chunks in flight; the CE engine drains them quickly and waits.

Even at this modest rate, contention shows up. Pushing closer to link
saturation would require: bigger buffers (1 GiB), more chunks in flight
(8+ deep), and probably moving the issuer out of the Python thread.

## What this means for NIXL P-D disaggregation design

- **One-directional ingress (prefill → decode KV push)** has small but
  not zero cost on decode (~10% at high B for max-rate). Mostly safe.
- **Bidirectional or egress-dominant** (decode pushing context KV back
  for prefix-cache reuse, etc.) is **2× worse**. Should be paced or
  avoided during peak decode batches.
- The damage scales with the workload's intrinsic HBM-Read pressure
  (here: batch + prefill). Decode benchmarks with short context will
  *underestimate* the cost.

## Confounders / caveats

- Bench is 20 iters at 1 kHz → ~6 ms/iter × 20 = ~120 ms of GPU-metric
  window. Sample counts per metric per iter are 3–25 (varies with iter
  length), so p99 within an iter is shaky. Headline `_med` columns are
  cross-iter medians of means and are reliable.
- `b512_p2048_ingress_max` and the four `b1024_p2048` bg cells took
  several minutes each — the bg side seems to slow LLM init when the
  KV pool is largest. Total sweep was ~70 min (12 off + 36 new + 12
  re-summarize).
- `bg_achieved_gbps.png` mislabels the x-axis (all bars show as
  `egress_max`; this is a plot bug, the y-values are correct because
  the loop reads from the right column).

## Files

- `sweep_runs/bg/sweep_results.csv` — 60 rows, ~80 columns
- `sweep_runs/bg/figs_bg/bg_p50_heatmap.png` — headline figure
- `sweep_runs/bg/figs_bg/bg_hw_vs_batch_p2048.png` — HW pressure curves
- `sweep_runs/bg/figs_bg/bg_hw_vs_prefill_b1024.png` — same, vs prefill
- `sweep_runs/bg/nsys/*.nsys-rep` — 48 raw profiler reports (~10 GB)
- `sweep_runs/bg/iter_csv/*.iter.csv` — per-iter HW distribution per cell
- `sweep_runs/bg/kernel_csv/*.kernels.csv` — FMHA / fused_moe / allreduce
  duration stats per cell
