# P/D Overlap Summary
trace_dir: `playground/log/moe_pd/telemetry_5k_starttime_2026-06-01T02-23-22Z/pd_trace`
window_s: 59.994
decode_trace_files: 4
overlap_interval_source: nixl
kv_recv_intervals: 1133
software_recv_intervals: 1133
nixl_xfer_intervals: 1133
decode_engine_step_intervals: 9638
decode_forward_intervals: 9640
decode_phase_intervals: 86758
itl_samples: 0

## Decode Engine Step Duration By KV-Receive Overlap
| group | n | mean_ms | p50_ms | p90_ms | p95_ms | p99_ms | max_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_recv_overlap | 6554 | 12.291 | 12.299 | 12.503 | 12.557 | 12.697 | 13.673 |
| recv_overlap | 3084 | 12.698 | 12.277 | 12.552 | 14.925 | 23.862 | 34.785 |
| max_active_recv_1 | 2374 | 12.492 | 12.275 | 12.506 | 12.618 | 20.450 | 29.371 |
| max_active_recv_ge2 | 710 | 13.387 | 12.290 | 15.610 | 21.147 | 26.707 | 34.785 |

## Decode Forward Launch-Scope Duration By KV-Receive Overlap
These CPU intervals surround `_model_forward()` without CUDA sync; they are useful as launch-scope context, not full GPU execution time.

| group | n | mean_ms | p50_ms | p90_ms | p95_ms | p99_ms | max_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_recv_overlap | 7004 | 0.304 | 0.305 | 0.343 | 0.349 | 0.361 | 0.426 |
| recv_overlap | 2636 | 0.309 | 0.317 | 0.348 | 0.357 | 0.380 | 0.475 |
| max_active_recv_1 | 2154 | 0.309 | 0.317 | 0.347 | 0.357 | 0.380 | 0.475 |
| max_active_recv_ge2 | 482 | 0.311 | 0.316 | 0.350 | 0.364 | 0.386 | 0.436 |

## Decode Subphase Duration By KV-Receive Overlap
Rows are CPU wall-clock intervals. `sample` and `bookkeeping` were added for the steady-state follow-up run; older traces may have zero rows for those phases.

| phase | group | n | mean_ms | p50_ms | p90_ms | p95_ms | p99_ms | max_ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| engine_step_fn | no_recv_overlap | 6554 | 12.284 | 12.292 | 12.495 | 12.551 | 12.689 | 13.667 |
| engine_step_fn | recv_overlap | 3084 | 12.691 | 12.270 | 12.545 | 14.919 | 23.856 | 34.780 |
| engine_step_fn | max_active_recv_1 | 2374 | 12.485 | 12.268 | 12.500 | 12.610 | 20.444 | 29.364 |
| engine_step_fn | max_active_recv_ge2 | 710 | 13.380 | 12.282 | 15.604 | 21.141 | 26.701 | 34.780 |
| engine_post_step | no_recv_overlap | 8080 | 0.007 | 0.007 | 0.008 | 0.009 | 0.012 | 0.023 |
| engine_post_step | recv_overlap | 1560 | 0.007 | 0.007 | 0.008 | 0.009 | 0.012 | 0.025 |
| engine_post_step | max_active_recv_1 | 1402 | 0.007 | 0.007 | 0.008 | 0.009 | 0.012 | 0.025 |
| engine_post_step | max_active_recv_ge2 | 158 | 0.007 | 0.007 | 0.009 | 0.010 | 0.011 | 0.012 |
| kv_start_load | no_recv_overlap | 7271 | 0.011 | 0.010 | 0.011 | 0.011 | 0.019 | 0.035 |
| kv_start_load | recv_overlap | 2369 | 2.560 | 0.011 | 7.074 | 10.122 | 19.004 | 32.211 |
| kv_start_load | max_active_recv_1 | 1848 | 2.117 | 0.011 | 6.548 | 9.853 | 18.270 | 24.165 |
| kv_start_load | max_active_recv_ge2 | 521 | 4.128 | 3.388 | 8.357 | 10.891 | 23.325 | 32.211 |
| kv_finalize | no_recv_overlap | 7081 | 0.030 | 0.026 | 0.033 | 0.065 | 0.083 | 0.151 |
| kv_finalize | recv_overlap | 2559 | 0.068 | 0.057 | 0.126 | 0.144 | 0.239 | 0.564 |
| kv_finalize | max_active_recv_1 | 2087 | 0.060 | 0.043 | 0.113 | 0.130 | 0.157 | 0.436 |
| kv_finalize | max_active_recv_ge2 | 472 | 0.103 | 0.099 | 0.165 | 0.207 | 0.325 | 0.564 |
| kv_wait_for_save | no_recv_overlap | 7108 | 0.001 | 0.001 | 0.001 | 0.001 | 0.002 | 0.021 |
| kv_wait_for_save | recv_overlap | 2532 | 0.001 | 0.001 | 0.001 | 0.001 | 0.002 | 0.021 |
| kv_wait_for_save | max_active_recv_1 | 2063 | 0.001 | 0.001 | 0.001 | 0.001 | 0.002 | 0.021 |
| kv_wait_for_save | max_active_recv_ge2 | 469 | 0.001 | 0.001 | 0.001 | 0.001 | 0.002 | 0.002 |
| kv_get_finished | no_recv_overlap | 7093 | 0.022 | 0.019 | 0.024 | 0.047 | 0.068 | 0.132 |
| kv_get_finished | recv_overlap | 2547 | 0.058 | 0.047 | 0.110 | 0.124 | 0.221 | 0.546 |
| kv_get_finished | max_active_recv_1 | 2077 | 0.050 | 0.036 | 0.098 | 0.113 | 0.139 | 0.416 |
| kv_get_finished | max_active_recv_ge2 | 470 | 0.090 | 0.084 | 0.146 | 0.188 | 0.306 | 0.546 |
| sample_tokens | no_recv_overlap | 7677 | 0.549 | 0.548 | 0.580 | 0.593 | 0.634 | 0.774 |
| sample_tokens | recv_overlap | 1963 | 0.571 | 0.556 | 0.629 | 0.719 | 0.786 | 1.144 |
| sample_tokens | max_active_recv_1 | 1699 | 0.571 | 0.556 | 0.624 | 0.719 | 0.791 | 1.144 |
| sample_tokens | max_active_recv_ge2 | 264 | 0.572 | 0.558 | 0.650 | 0.718 | 0.765 | 0.786 |
| sample | no_recv_overlap | 7678 | 0.267 | 0.264 | 0.287 | 0.294 | 0.349 | 0.457 |
| sample | recv_overlap | 1962 | 0.286 | 0.271 | 0.329 | 0.416 | 0.461 | 0.838 |
| sample | max_active_recv_1 | 1698 | 0.285 | 0.270 | 0.319 | 0.415 | 0.462 | 0.838 |
| sample | max_active_recv_ge2 | 264 | 0.290 | 0.276 | 0.342 | 0.421 | 0.459 | 0.464 |
| bookkeeping | no_recv_overlap | 7714 | 0.122 | 0.121 | 0.137 | 0.145 | 0.157 | 0.215 |
| bookkeeping | recv_overlap | 1926 | 0.122 | 0.122 | 0.136 | 0.144 | 0.153 | 0.175 |
| bookkeeping | max_active_recv_1 | 1666 | 0.123 | 0.122 | 0.137 | 0.145 | 0.154 | 0.175 |
| bookkeeping | max_active_recv_ge2 | 260 | 0.120 | 0.119 | 0.136 | 0.139 | 0.149 | 0.153 |

## ITL By KV-Receive Overlap
| group | n | mean_ms | p50_ms | p90_ms | p95_ms | p99_ms | max_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_recv_overlap | nan | nan | nan | nan | nan | nan | nan |
| recv_overlap | nan | nan | nan | nan | nan | nan | nan |
| max_active_recv_1 | nan | nan | nan | nan | nan | nan | nan |
| max_active_recv_ge2 | nan | nan | nan | nan | nan | nan | nan |

## KV Receive / Transfer Duration
n=1133, mean=11.968 ms, p50=11.142 ms, p90=26.962 ms, p99=38.859 ms, max=46.331 ms

## NIXL Transfer Telemetry
xfer_duration_ms: n=1133, mean=11.968, p50=11.142, p90=26.962, p99=38.859, max=46.331
throughput_GBps: mean=1.948, p50=1.933, p90=2.741, p99=3.349
poll_slack_ms: mean=0.223, p50=0.052, p90=0.412, p99=0.485
