# E2E AGRS NIXL Slowdown Report

This report uses per-engine-step NIXL transfer telemetry. Estimated transfer volume is prorated by the clipped overlap between each NIXL transfer interval and each engine-step interval.

`estimated_MB = sum(bytes_transferred * clipped_overlap_time / transfer_duration) / 1e6`

`normalized_MB_per_ms = estimated_MB / engine_step_duration_ms`

The primary steady-state figures use only `mid +240s` and `mid +300s`.

## Estimated Transfer Volume Buckets

| bucket | n | p50 ms | p90 ms | p95 ms | p99 ms | >=2 active % | estimated MB p50 | normalized MB/ms p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 11736 | 12.359 | 12.612 | 12.764 | 12.961 | 0.0 | 0.000 | 0.000 |
| (0, 4] | 2057 | 12.356 | 12.617 | 12.772 | 12.966 | 6.3 | 1.610 | 0.132 |
| (4, 8] | 1573 | 12.373 | 12.649 | 12.806 | 13.394 | 19.3 | 5.866 | 0.473 |
| (8, 16] | 743 | 12.355 | 12.875 | 14.158 | 18.716 | 56.9 | 11.010 | 0.875 |
| (16, 32] | 2167 | 12.366 | 12.680 | 12.923 | 14.314 | 27.7 | 23.513 | 1.895 |
| (32, 64] | 819 | 12.909 | 18.658 | 21.501 | 24.618 | 74.0 | 41.603 | 2.887 |
| >64 | 59 | 24.696 | 31.540 | 37.795 | 38.476 | 83.1 | 74.372 | 3.268 |

## Normalized Transfer Volume Buckets

| bucket | n | p50 ms | p90 ms | p95 ms | p99 ms | >=2 active % | estimated MB p50 | normalized MB/ms p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 11736 | 12.359 | 12.612 | 12.764 | 12.961 | 0.0 | 0.000 | 0.000 |
| (0, 0.5] | 2991 | 12.364 | 12.638 | 12.798 | 13.079 | 9.3 | 2.546 | 0.208 |
| (0.5, 1] | 1088 | 12.367 | 12.773 | 13.047 | 15.332 | 42.7 | 7.569 | 0.610 |
| (1, 2] | 1573 | 12.352 | 12.630 | 12.847 | 18.736 | 25.7 | 20.048 | 1.616 |
| (2, 3] | 1408 | 12.461 | 16.175 | 20.700 | 27.032 | 43.5 | 29.429 | 2.316 |
| (3, 4] | 283 | 12.467 | 17.111 | 21.364 | 37.800 | 97.9 | 44.483 | 3.393 |
| >4 | 75 | 12.421 | 13.966 | 16.547 | 21.673 | 100.0 | 54.109 | 4.377 |
