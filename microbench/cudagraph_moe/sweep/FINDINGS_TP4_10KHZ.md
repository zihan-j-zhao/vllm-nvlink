# Sweep findings — TP=4, 10 kHz sampling (Qwen3-30B-A3B, B200×4)

Source: [sweep_runs/tp4_10khz/sweep_results.csv](../sweep_runs/tp4_10khz/sweep_results.csv)
60 cells: 12 (B, prefill) × 5 (bg=off, ingress_med, ingress_max, egress_med, egress_max),
20 iters per cell, **10 kHz GPU-metric sampling** (vs 1 kHz in `sweep_runs/bg/`).

Re-run of the bg-impact sweep at 10× higher sampling rate to extract
**per-kernel HW pressure** for the three hot kernels (fmha, fused_moe,
allreduce). Total wall time ≈ 90 min (vs ≈ 70 min at 1 kHz).

## TL;DR — bg-traffic impact (identical to 1 kHz within 1 pp)

| bg profile     | NVLink dir | Achieved Gbps/rank | p50 slowdown vs off |
|----------------|------------|-------------------:|---------------------|
| `off`          | -          | 0                  | baseline            |
| `ingress_med`  | RX         | 1–4                | **+0.3% to +1.2%**  |
| `ingress_max`  | RX         | 13–30              | **+8% to +12%**     |
| `egress_med`   | TX         | 1–5                | **+1.7% to +2.1%**  |
| `egress_max`   | TX         | 13–30              | **+10% to +20%**    |

The slowdown picture is unchanged from the 1 kHz sweep — see
[figs_bg/bg_p50_heatmap.png](../sweep_runs/tp4_10khz/figs_bg/bg_p50_heatmap.png).
**egress_max remains the only profile whose slowdown grows with batch
size** (+10% at B=128 → +20% at B=1024).

## Headline new finding — per-kernel HW pressure

10 kHz sampling (100 µs window) gives us enough signal to separate the
three hot kernels. For the most-stressed cell **B=1024, p=2048, off**
([figs_kernel/per_kernel_hw_b1024_p2048.png](../sweep_runs/tp4_10khz/figs_kernel/per_kernel_hw_b1024_p2048.png)):

| Kernel        | HBM Read | HBM Write | SM Active | Tensor Active | Interpretation                |
|---------------|---------:|----------:|----------:|--------------:|-------------------------------|
| **fmha**      | **71%**  | 1%        | 85%       | 3%            | HBM-read-bound (KV cache scan), tensor cores idle |
| **fused_moe** | 16%      | 7%        | **90%**   | 7%            | SM-compute-bound, weights cache-resident, modest HBM |
| **allreduce** | 23%      | 7%        | 77%       | 4%            | L2/scratch-bound, modest HBM, all-SM-resident |

This validates the bg-traffic story mechanically:
- **FMHA sits at 71% HBM Read** — adding 30 GB/s of egress reads from
  the same HBM controller (egress_max) pushes total demand past 90% →
  FMHA wall-time stretches → p50 increases.
- **Ingress traffic competes on HBM Write port** which FMHA barely uses
  (1%). So ingress_max can add 9 pp of write traffic with almost no
  effect on FMHA.

## Per-kernel HW vs (B, prefill) — off baseline

See [figs_kernel/per_kernel_hw_off.png](../sweep_runs/tp4_10khz/figs_kernel/per_kernel_hw_off.png).

### fmha (HBM Read across batch and prefill)

| Cell                   | fmha HBM Read % |
|------------------------|----------------:|
| B=128,  p=128          | 19              |
| B=128,  p=2048         | 28              |
| B=1024, p=128          | **8**           |
| B=1024, p=512          | 19              |
| B=1024, p=1024         | 39              |
| B=1024, p=2048         | **71**          |

The **batch × prefill interaction** that the iter-level data hinted at is
now sharp:
- At B=128, FMHA HBM Read stays in 19–28% — attention is so cheap per
  request that it barely uses the controller even at p=2048.
- At B=1024, FMHA HBM Read climbs **9× from p=128 to p=2048**, because
  each request now reads ceil(2048/128)×16× more KV per step. The
  controller hits 71% — within striking distance of saturation, which
  is exactly why egress_max hurts this cell more than any other.

### fused_moe / allreduce — batch-dominated, prefill-flat

- **fused_moe SMs Active = 56% at B=128 → 90% at B=1024+** regardless of
  prefill. MoE saturates the SMs once we have enough tokens per batch
  to keep all experts busy.
- **allreduce SMs Active = 58% at B=128 → 77% at B=1024**. Similar
  story; the ring/oneshot kernels are SM-resident and need batch volume
  to fill the SMs.
- Both kernels have HBM Read in the 13–25% range and HBM Write 1–7%,
  ~prefill-independent. They don't touch the KV cache.

## Per-kernel impact of bg traffic (B=1024, p=2048)

See [figs_kernel/per_kernel_hw_b1024_p2048.png](../sweep_runs/tp4_10khz/figs_kernel/per_kernel_hw_b1024_p2048.png).

Key observations:
- **FMHA HBM Read is barely shifted by ingress** (71 → 68 with
  ingress_max) but **rises with egress_max** (71 → 74). This is the
  smoking gun for HBM-controller contention with egress reads.
- **HBM Write rises sharply with ingress** for all three kernels (fmha
  1 → 8%, fused_moe 7 → 14%, allreduce 7 → 15%) because the bg traffic
  is writing into decoder HBM and the controller queues now include
  CE-driven writes.
- **NVLink TX = 53–56%** during all three kernels under egress_max —
  bg is constantly streaming during MoE/allreduce/FMHA execution, not
  just in bursts.
- **NVLink RX = 58–60%** during all three kernels under ingress_max —
  same constant streaming on the opposite direction.
- **SMs Active is the most invariant** — bg traffic doesn't use SMs
  at all (CE engines), so this metric stays at off-baseline ±2 pp
  across all profiles.

## Sanity check: 10 kHz vs 1 kHz

Same headline numbers on the iter-level aggregates. Spot-check on
B=1024p2048 off:

| Metric         | 1 kHz | 10 kHz |
|----------------|------:|-------:|
| p50_ms         | 20.80 | 20.80  |
| HBM Read med   | 37.9% | 37.8%  |
| SMs Active med | 79.4% | 79.5%  |
| NVLink TX med  | 5.4%  | 5.2%   |

The p50 difference is < 0.5%. **All slowdown deltas vs off are within
1 pp of the 1 kHz numbers**, confirming the higher sampling rate adds
~10% wall-clock but doesn't change conclusions on the bg-impact axis.

What 10 kHz **does** unlock:
- Per-kernel HBM/SM/NVLink figures with enough samples per kernel-call
  for FMHA (1.7 samples / call at 168 µs) to give credible distributions.
- For fused_moe (47 µs) and allreduce (11–48 µs) per-kernel samples are
  still partial-cycle (0.1–0.5 per call), so values reflect the
  surrounding ~100 µs window. They're directional, not exact.

## Confounders / caveats

- **Per-kernel HW for short kernels is biased toward neighbours.** A
  100 µs sample window that touches a 47 µs fused_moe call also captures
  ~50 µs of whatever ran before/after. On the decode hot path these are
  *other* fused_moe / allreduce calls, so the bias is correlated but
  not zero.
- **NVLink % for fmha under bg traffic** (TX 53%, RX 58%) is mostly the
  side-stream bg traffic, not FMHA-attributable. NVLink isn't a hot
  resource for FMHA itself — the % is "what was happening on NVLink
  while FMHA ran."
- **10 kHz did not produce missing-data bands** in the GUI, unlike the
  earlier 100 kHz attempts. The PM-unit ceiling sits between these.

## Files

- [sweep_results.csv](../sweep_runs/tp4_10khz/sweep_results.csv) — 60 rows, ~80 cols
- [figs_bg/bg_p50_heatmap.png](../sweep_runs/tp4_10khz/figs_bg/bg_p50_heatmap.png) — p50 slowdown
- [figs_bg/bg_hw_vs_batch_p2048.png](../sweep_runs/tp4_10khz/figs_bg/bg_hw_vs_batch_p2048.png) — HW vs batch
- [figs_bg/bg_hw_vs_prefill_b1024.png](../sweep_runs/tp4_10khz/figs_bg/bg_hw_vs_prefill_b1024.png) — HW vs prefill
- [figs_bg/bg_achieved_gbps.png](../sweep_runs/tp4_10khz/figs_bg/bg_achieved_gbps.png) — bg traffic delivered Gbps
- [figs_kernel/per_kernel_hw_off.png](../sweep_runs/tp4_10khz/figs_kernel/per_kernel_hw_off.png) — per-kernel HW vs prefill
- [figs_kernel/per_kernel_hw_b1024_p2048.png](../sweep_runs/tp4_10khz/figs_kernel/per_kernel_hw_b1024_p2048.png) — per-kernel HW under bg, hot cell
- [figs_kernel/per_kernel_hw_b128_p128.png](../sweep_runs/tp4_10khz/figs_kernel/per_kernel_hw_b128_p128.png) — per-kernel HW under bg, cold cell
- `sweep_runs/tp4_10khz/nsys/*.nsys-rep` — 60 raw profiler reports
- `sweep_runs/tp4_10khz/nsys/*.kernel_hw.csv` — per-kernel HW samples (new at 10 kHz)
- `sweep_runs/tp4_10khz/iter_csv/*.iter.csv` — per-iter HW distributions
- `sweep_runs/tp4_10khz/kernel_csv/*.kernels.csv` — kernel duration stats
