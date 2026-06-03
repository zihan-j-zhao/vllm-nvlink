# Union Overlap Model Report

This reprocesses the 5k startTime-aligned NIXL trace and adds `union_overlap_ms`, which merges overlapping transfer intervals before measuring overlap. The old `overlap_ms` is kept as summed transfer exposure.

Definitions:
- `union_overlap_ms`: wall-clock time inside the engine/phase interval where at least one NIXL transfer is active; common regions are counted once.
- `overlap_ms`: summed intersections across all transfers; common regions are counted once per active transfer.
- `overlap_intensity = overlap_ms / union_overlap_ms`: average active transfer count during transfer-active time.

## Why this matters
The previous x-axis mixed duration and concurrency. With `union_overlap_ms`, red `>=2 active` points are no longer artificially shifted right just because two transfers share the same wall-clock interval. Use union overlap for elapsed-time comparisons and intensity for concurrency pressure.

## Plots
- engine_step_quantiles_by_union_and_intensity.png
- engine_step_sum_vs_union_overlap_zoom.png
- engine_step_union_overlap_by_intensity_zoom.png
- kv_start_load_sum_vs_union_overlap_zoom.png

## Summary by active group
| window | active | n | dur p50 | dur p90 | dur p99 | sum ov p50 | union ov p50 | intensity p50 | intensity p90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| early +120s | 0 | 3513 | 12.358 | 12.835 | 13.814 | 0.000 | 0.000 | nan | nan |
| early +120s | 1 | 1280 | 12.370 | 13.135 | 22.662 | 5.567 | 5.567 | 1.000 | 1.000 |
| early +120s | >=2 | 463 | 12.441 | 22.458 | 324.378 | 15.340 | 12.043 | 1.542 | 2.307 |
| mid +180s | 0 | 6554 | 12.299 | 12.503 | 12.697 | 0.000 | 0.000 | nan | nan |
| mid +180s | 1 | 2374 | 12.275 | 12.506 | 20.450 | 4.307 | 4.307 | 1.000 | 1.000 |
| mid +180s | >=2 | 710 | 12.290 | 15.610 | 26.707 | 15.191 | 12.112 | 1.441 | 2.000 |
| mid +240s | 0 | 6050 | 12.349 | 12.578 | 12.909 | 0.000 | 0.000 | nan | nan |
| mid +240s | 1 | 2576 | 12.363 | 12.722 | 17.954 | 9.416 | 9.416 | 1.000 | 1.000 |
| mid +240s | >=2 | 970 | 12.412 | 14.444 | 24.640 | 17.283 | 12.232 | 1.745 | 2.337 |
| mid +300s | 0 | 5686 | 12.372 | 12.656 | 12.993 | 0.000 | 0.000 | nan | nan |
| mid +300s | 1 | 2730 | 12.373 | 12.791 | 16.675 | 9.429 | 9.429 | 1.000 | 1.000 |
| mid +300s | >=2 | 1142 | 12.409 | 14.766 | 25.284 | 18.270 | 12.231 | 1.799 | 2.196 |
| late +360s | 0 | 4382 | 12.314 | 12.577 | 13.000 | 0.000 | 0.000 | nan | nan |
| late +360s | 1 | 1538 | 12.355 | 12.700 | 20.206 | 9.391 | 9.391 | 1.000 | 1.000 |
| late +360s | >=2 | 664 | 12.420 | 18.275 | 26.885 | 17.786 | 12.302 | 1.616 | 2.350 |
