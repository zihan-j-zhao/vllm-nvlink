# Estimated Transfer Volume Slowdown Report

This uses the proposed formula for each engine step:

`estimated_MB = sum(bytes_transferred * clipped_overlap_time / transfer_duration) / 1e6`

where `clipped_overlap_time` is the intersection between the NIXL transfer interval and the engine-step interval. The report uses the steady middle windows only: `mid +240s` and `mid +300s`.

## Outputs

- engine_step_tail_by_estimated_transfer_mb_steady.png
- engine_step_tail_slowdown_by_estimated_transfer_mb_steady.png
- engine_step_tail_survival_by_estimated_transfer_mb_steady.png
- engine_step_estimated_transfer_mb_bucket_summary.csv

## Bucket Summary

| estimated MB bucket | n | p50 ms | p90 ms | p95 ms | p99 ms | p90 slowdown | p99 slowdown | >=2 active % | p50 pressure GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 11736 | 12.359 | 12.612 | 12.764 | 12.961 | 1.000 | 1.000 | 0.0 | 0.000 |
| (0, 4] | 2057 | 12.356 | 12.617 | 12.772 | 12.966 | 1.000 | 1.000 | 6.3 | 0.376 |
| (4, 8] | 1573 | 12.373 | 12.649 | 12.806 | 13.394 | 1.003 | 1.033 | 19.3 | 2.013 |
| (8, 16] | 743 | 12.355 | 12.875 | 14.158 | 18.716 | 1.021 | 1.444 | 56.9 | 2.193 |
| (16, 32] | 2167 | 12.366 | 12.680 | 12.923 | 14.314 | 1.005 | 1.104 | 27.7 | 2.330 |
| (32, 64] | 819 | 12.909 | 18.658 | 21.501 | 24.618 | 1.479 | 1.899 | 74.0 | 3.545 |
| >64 | 59 | 24.696 | 31.540 | 37.795 | 38.476 | 2.501 | 2.969 | 83.1 | 4.489 |
