# P/D Overlap Summary
trace_dir: `playground/log/moe_pd/telemetry_5k_starttime_2026-06-01T02-23-22Z/pd_trace`
window_s: 59.985
decode_trace_files: 4
overlap_interval_source: nixl
kv_recv_intervals: 1193
software_recv_intervals: 1193
nixl_xfer_intervals: 1193
decode_engine_step_intervals: 9596
decode_forward_intervals: 9598
decode_phase_intervals: 86380
itl_samples: 0

## Decode Engine Step Duration By KV-Receive Overlap
| group | n | mean_ms | p50_ms | p90_ms | p95_ms | p99_ms | max_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_recv_overlap | 6050 | 12.338 | 12.349 | 12.578 | 12.687 | 12.909 | 13.867 |
| recv_overlap | 3546 | 12.701 | 12.374 | 12.894 | 14.473 | 21.584 | 38.537 |
| max_active_recv_1 | 2576 | 12.540 | 12.363 | 12.722 | 13.258 | 17.954 | 30.763 |
| max_active_recv_ge2 | 970 | 13.130 | 12.412 | 14.444 | 18.422 | 24.640 | 38.537 |

## Decode Forward Launch-Scope Duration By KV-Receive Overlap
These CPU intervals surround `_model_forward()` without CUDA sync; they are useful as launch-scope context, not full GPU execution time.

| group | n | mean_ms | p50_ms | p90_ms | p95_ms | p99_ms | max_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_recv_overlap | 6520 | 0.223 | 0.199 | 0.304 | 0.324 | 0.351 | 0.750 |
| recv_overlap | 3078 | 0.246 | 0.223 | 0.329 | 0.351 | 0.388 | 0.927 |
| max_active_recv_1 | 2298 | 0.244 | 0.219 | 0.328 | 0.352 | 0.391 | 0.927 |
| max_active_recv_ge2 | 780 | 0.250 | 0.233 | 0.331 | 0.350 | 0.378 | 0.426 |

## Decode Subphase Duration By KV-Receive Overlap
Rows are CPU wall-clock intervals. `sample` and `bookkeeping` were added for the steady-state follow-up run; older traces may have zero rows for those phases.

| phase | group | n | mean_ms | p50_ms | p90_ms | p95_ms | p99_ms | max_ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| engine_step_fn | no_recv_overlap | 6050 | 12.331 | 12.341 | 12.571 | 12.678 | 12.902 | 13.860 |
| engine_step_fn | recv_overlap | 3546 | 12.694 | 12.367 | 12.885 | 14.465 | 21.576 | 38.529 |
| engine_step_fn | max_active_recv_1 | 2576 | 12.532 | 12.356 | 12.714 | 13.253 | 17.948 | 30.756 |
| engine_step_fn | max_active_recv_ge2 | 970 | 13.123 | 12.405 | 14.437 | 18.415 | 24.634 | 38.529 |
| engine_post_step | no_recv_overlap | 7472 | 0.007 | 0.007 | 0.008 | 0.009 | 0.011 | 0.023 |
| engine_post_step | recv_overlap | 2126 | 0.007 | 0.007 | 0.009 | 0.009 | 0.011 | 0.016 |
| engine_post_step | max_active_recv_1 | 1760 | 0.007 | 0.007 | 0.008 | 0.009 | 0.012 | 0.016 |
| engine_post_step | max_active_recv_ge2 | 366 | 0.007 | 0.007 | 0.009 | 0.009 | 0.010 | 0.016 |
| kv_start_load | no_recv_overlap | 6679 | 0.010 | 0.010 | 0.011 | 0.011 | 0.020 | 0.035 |
| kv_start_load | recv_overlap | 2919 | 2.303 | 0.011 | 8.533 | 10.562 | 16.658 | 35.854 |
| kv_start_load | max_active_recv_1 | 2147 | 1.885 | 0.010 | 7.999 | 10.427 | 15.903 | 28.088 |
| kv_start_load | max_active_recv_ge2 | 772 | 3.465 | 2.115 | 9.397 | 11.595 | 18.022 | 35.854 |
| kv_finalize | no_recv_overlap | 6637 | 0.030 | 0.027 | 0.042 | 0.046 | 0.076 | 0.147 |
| kv_finalize | recv_overlap | 2961 | 0.083 | 0.082 | 0.144 | 0.166 | 0.265 | 0.610 |
| kv_finalize | max_active_recv_1 | 2234 | 0.072 | 0.067 | 0.131 | 0.146 | 0.224 | 0.364 |
| kv_finalize | max_active_recv_ge2 | 727 | 0.116 | 0.106 | 0.184 | 0.209 | 0.367 | 0.610 |
| kv_wait_for_save | no_recv_overlap | 6679 | 0.001 | 0.001 | 0.002 | 0.002 | 0.002 | 0.021 |
| kv_wait_for_save | recv_overlap | 2919 | 0.001 | 0.001 | 0.002 | 0.002 | 0.002 | 0.019 |
| kv_wait_for_save | max_active_recv_1 | 2200 | 0.001 | 0.001 | 0.002 | 0.002 | 0.002 | 0.019 |
| kv_wait_for_save | max_active_recv_ge2 | 719 | 0.001 | 0.001 | 0.002 | 0.002 | 0.002 | 0.009 |
| kv_get_finished | no_recv_overlap | 6650 | 0.022 | 0.020 | 0.028 | 0.039 | 0.060 | 0.136 |
| kv_get_finished | recv_overlap | 2948 | 0.072 | 0.072 | 0.125 | 0.148 | 0.253 | 0.586 |
| kv_get_finished | max_active_recv_1 | 2223 | 0.062 | 0.058 | 0.113 | 0.127 | 0.214 | 0.354 |
| kv_get_finished | max_active_recv_ge2 | 725 | 0.102 | 0.093 | 0.162 | 0.189 | 0.354 | 0.586 |
| sample_tokens | no_recv_overlap | 7294 | 0.542 | 0.541 | 0.577 | 0.589 | 0.618 | 0.954 |
| sample_tokens | recv_overlap | 2304 | 0.585 | 0.559 | 0.712 | 0.744 | 0.815 | 1.037 |
| sample_tokens | max_active_recv_1 | 1869 | 0.582 | 0.557 | 0.706 | 0.741 | 0.815 | 1.037 |
| sample_tokens | max_active_recv_ge2 | 435 | 0.600 | 0.569 | 0.728 | 0.750 | 0.853 | 0.981 |
| sample | no_recv_overlap | 7295 | 0.262 | 0.259 | 0.284 | 0.293 | 0.313 | 0.467 |
| sample | recv_overlap | 2303 | 0.293 | 0.275 | 0.375 | 0.414 | 0.473 | 0.725 |
| sample | max_active_recv_1 | 1869 | 0.291 | 0.273 | 0.364 | 0.413 | 0.472 | 0.725 |
| sample | max_active_recv_ge2 | 434 | 0.304 | 0.287 | 0.392 | 0.419 | 0.475 | 0.499 |
| bookkeeping | no_recv_overlap | 7308 | 0.122 | 0.121 | 0.138 | 0.144 | 0.157 | 0.281 |
| bookkeeping | recv_overlap | 2290 | 0.129 | 0.123 | 0.149 | 0.185 | 0.226 | 0.294 |
| bookkeeping | max_active_recv_1 | 1861 | 0.128 | 0.123 | 0.146 | 0.175 | 0.219 | 0.277 |
| bookkeeping | max_active_recv_ge2 | 429 | 0.132 | 0.123 | 0.169 | 0.209 | 0.238 | 0.294 |

## ITL By KV-Receive Overlap
| group | n | mean_ms | p50_ms | p90_ms | p95_ms | p99_ms | max_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_recv_overlap | nan | nan | nan | nan | nan | nan | nan |
| recv_overlap | nan | nan | nan | nan | nan | nan | nan |
| max_active_recv_1 | nan | nan | nan | nan | nan | nan | nan |
| max_active_recv_ge2 | nan | nan | nan | nan | nan | nan | nan |

## KV Receive / Transfer Duration
n=1193, mean=14.948 ms, p50=12.562 ms, p90=27.097 ms, p99=37.919 ms, max=46.582 ms

## NIXL Transfer Telemetry
xfer_duration_ms: n=1193, mean=14.948, p50=12.562, p90=27.097, p99=37.919, max=46.582
throughput_GBps: mean=1.501, p50=1.663, p90=2.656, p99=3.113
poll_slack_ms: mean=0.092, p50=0.041, p90=0.276, p99=0.421
