# Figure-generation techniques

## End-to-end benchmark figures

- Treat each benchmark run as a structured JSON record: config, per-rank per-iteration latency, and optional background-traffic metadata.
- Aggregate ranks by taking the max latency per iteration, because distributed forward latency is gated by the slowest rank.
- Drop warmup or first measured iterations when comparing steady-state behavior.
- Report robust latency percentiles such as p50/p90/p99, plus coefficient of variation for stability.
- Convert latency to logical decode throughput with:

```text
throughput = batch_size * dp * 1000 / latency_ms
```

- Plot one metric against one controlled variable, usually batch size or sequence length. Keep labels explicit about whether batch is per-rank or global.

## Nsight collection and extraction

- Mark the measured region with NVTX ranges, then only analyze kernels and copies whose timestamps fall inside that benchmark window.
- Export `.nsys-rep` traces to SQLite and query CUPTI tables directly.
- Use `CUPTI_ACTIVITY_KIND_KERNEL` for GPU kernels and join `StringIds` to filter by demangled kernel-name substrings.
- Use `CUPTI_ACTIVITY_KIND_MEMCPY` for device-to-device/P2P copies, filtering by copy kind when studying synthetic KV traffic.

## Kernel breakdown figures

- Group kernels by stable name patterns rather than exact full names, e.g. attention kernels, MoE kernels, NCCL/allreduce kernels, all-gather, reduce-scatter.
- Extract per-invocation duration distributions in microseconds.
- Summarize each group with count, total time, mean, p50, p90, p99, min, and max.
- Use histograms to show distribution shifts and boxplots to compare many conditions compactly.
- Hide or clip extreme outliers in visualizations when they dominate the x-axis, but keep them in CSV summaries.

## Overlap analysis

- Extract P2P copy intervals from the same Nsight SQLite trace.
- For every kernel invocation, compute timestamp intersection with all P2P intervals.
- Classify kernels into:
  - `baseline`: no background traffic trace
  - `no_overlap`: traffic trace, but this kernel did not overlap a P2P copy
  - `overlap`: traffic trace, and this kernel overlapped at least one P2P copy
- Compare `baseline` vs `no_overlap` vs `overlap`. This separates general run-to-run effects from true contention during copy overlap.
- Report both duration statistics and mean overlap fraction, because a kernel that barely overlaps traffic should not be interpreted the same way as a fully overlapped kernel.

## Presentation rules

- Keep raw summaries in CSV next to each generated figure.
- Prefer matched-pair comparisons: same model, batch size, sequence length, topology, and profiling settings.
- Compare profiled runs only against similarly profiled runs, since Nsight overhead can shift absolute timings.
- Use consistent colors for traffic modes and overlap classes across figures.
- Make titles and axes describe the measured unit and traffic condition directly.