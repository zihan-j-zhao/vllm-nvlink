# Normalized Transfer Volume Slowdown Report

This normalizes the estimated in-step transfer volume by the engine-step duration to reduce the circular effect where longer steps naturally have more time to overlap transfers.

`normalized_MB_per_ms = estimated_MB / engine_step_duration_ms`

Since `MB/ms` is numerically equivalent to `GB/s`, this behaves like average in-step transfer throughput over the whole engine-step interval. The report uses `mid +240s` and `mid +300s`.

## Outputs

- engine_step_tail_by_normalized_transfer_volume_steady.png
- engine_step_tail_slowdown_by_normalized_transfer_volume_steady.png
- engine_step_normalized_transfer_volume_bucket_summary.csv

## Bucket Summary

| normalized MB/ms bucket | n | p50 ms | p90 ms | p95 ms | p99 ms | p90 slowdown | p99 slowdown | >=2 active % | estimated MB p50 | union overlap p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 11736 | 12.359 | 12.612 | 12.764 | 12.961 | 1.000 | 1.000 | 0.0 | 0.000 | 0.000 |
| (0, 0.5] | 2991 | 12.364 | 12.638 | 12.798 | 13.079 | 1.002 | 1.009 | 9.3 | 2.546 | 2.623 |
| (0.5, 1] | 1088 | 12.367 | 12.773 | 13.047 | 15.332 | 1.013 | 1.183 | 42.7 | 7.569 | 3.226 |
| (1, 2] | 1573 | 12.352 | 12.630 | 12.847 | 18.736 | 1.001 | 1.446 | 25.7 | 20.048 | 9.855 |
| (2, 3] | 1408 | 12.461 | 16.175 | 20.700 | 27.032 | 1.282 | 2.086 | 43.5 | 29.429 | 12.294 |
| (3, 4] | 283 | 12.467 | 17.111 | 21.364 | 37.800 | 1.357 | 2.917 | 97.9 | 44.483 | 12.376 |
| >4 | 75 | 12.421 | 13.966 | 16.547 | 21.673 | 1.107 | 1.672 | 100.0 | 54.109 | 12.310 |
