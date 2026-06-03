# P/D Overlap Summary
trace_dir: `playground/log/moe_pd/telemetry_5k_starttime_2026-06-01T02-23-22Z/pd_trace`
window_s: 40.867
decode_trace_files: 4
overlap_interval_source: nixl
kv_recv_intervals: 702
software_recv_intervals: 702
nixl_xfer_intervals: 702
decode_engine_step_intervals: 6584
decode_forward_intervals: 6565
decode_phase_intervals: 59137
itl_samples: 0

## Decode Engine Step Duration By KV-Receive Overlap
| group | n | mean_ms | p50_ms | p90_ms | p95_ms | p99_ms | max_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_recv_overlap | 4382 | 12.077 | 12.314 | 12.577 | 12.748 | 13.000 | 112.851 |
| recv_overlap | 2202 | 12.955 | 12.371 | 13.017 | 16.496 | 25.610 | 41.587 |
| max_active_recv_1 | 1538 | 12.585 | 12.355 | 12.700 | 13.000 | 20.206 | 27.283 |
| max_active_recv_ge2 | 664 | 13.812 | 12.420 | 18.275 | 22.615 | 26.885 | 41.587 |

## Decode Forward Launch-Scope Duration By KV-Receive Overlap
These CPU intervals surround `_model_forward()` without CUDA sync; they are useful as launch-scope context, not full GPU execution time.

| group | n | mean_ms | p50_ms | p90_ms | p95_ms | p99_ms | max_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_recv_overlap | 4561 | 0.224 | 0.203 | 0.328 | 0.343 | 0.360 | 0.769 |
| recv_overlap | 2004 | 0.241 | 0.221 | 0.333 | 0.351 | 0.377 | 0.419 |
| max_active_recv_1 | 1468 | 0.240 | 0.220 | 0.332 | 0.349 | 0.377 | 0.419 |
| max_active_recv_ge2 | 536 | 0.244 | 0.222 | 0.336 | 0.360 | 0.377 | 0.387 |

## Decode Subphase Duration By KV-Receive Overlap
Rows are CPU wall-clock intervals. `sample` and `bookkeeping` were added for the steady-state follow-up run; older traces may have zero rows for those phases.

| phase | group | n | mean_ms | p50_ms | p90_ms | p95_ms | p99_ms | max_ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| engine_step_fn | no_recv_overlap | 4382 | 12.071 | 12.308 | 12.570 | 12.739 | 12.991 | 112.841 |
| engine_step_fn | recv_overlap | 2202 | 12.948 | 12.365 | 13.011 | 16.488 | 25.604 | 41.582 |
| engine_step_fn | max_active_recv_1 | 1538 | 12.578 | 12.348 | 12.693 | 12.993 | 20.199 | 27.278 |
| engine_step_fn | max_active_recv_ge2 | 664 | 13.805 | 12.413 | 18.267 | 22.607 | 26.878 | 41.582 |
| engine_post_step | no_recv_overlap | 5178 | 0.007 | 0.007 | 0.008 | 0.009 | 0.011 | 0.021 |
| engine_post_step | recv_overlap | 1408 | 0.007 | 0.007 | 0.008 | 0.009 | 0.011 | 0.016 |
| engine_post_step | max_active_recv_1 | 1148 | 0.007 | 0.007 | 0.008 | 0.009 | 0.011 | 0.016 |
| engine_post_step | max_active_recv_ge2 | 260 | 0.007 | 0.007 | 0.008 | 0.009 | 0.011 | 0.014 |
| kv_start_load | no_recv_overlap | 4723 | 0.010 | 0.010 | 0.011 | 0.011 | 0.024 | 0.032 |
| kv_start_load | recv_overlap | 1846 | 2.176 | 0.010 | 7.144 | 11.126 | 20.503 | 38.933 |
| kv_start_load | max_active_recv_1 | 1337 | 1.727 | 0.010 | 6.028 | 10.220 | 19.762 | 24.724 |
| kv_start_load | max_active_recv_ge2 | 509 | 3.354 | 1.856 | 9.384 | 12.993 | 23.906 | 38.933 |
| kv_finalize | no_recv_overlap | 4636 | 0.027 | 0.026 | 0.029 | 0.033 | 0.061 | 0.133 |
| kv_finalize | recv_overlap | 1933 | 0.078 | 0.077 | 0.134 | 0.158 | 0.267 | 0.419 |
| kv_finalize | max_active_recv_1 | 1401 | 0.068 | 0.062 | 0.119 | 0.139 | 0.255 | 0.419 |
| kv_finalize | max_active_recv_ge2 | 532 | 0.103 | 0.096 | 0.162 | 0.185 | 0.286 | 0.378 |
| kv_wait_for_save | no_recv_overlap | 4646 | 0.001 | 0.001 | 0.001 | 0.001 | 0.002 | 0.024 |
| kv_wait_for_save | recv_overlap | 1919 | 0.001 | 0.001 | 0.001 | 0.002 | 0.002 | 0.022 |
| kv_wait_for_save | max_active_recv_1 | 1401 | 0.001 | 0.001 | 0.001 | 0.001 | 0.002 | 0.022 |
| kv_wait_for_save | max_active_recv_ge2 | 518 | 0.001 | 0.001 | 0.001 | 0.002 | 0.002 | 0.010 |
| kv_get_finished | no_recv_overlap | 4642 | 0.020 | 0.019 | 0.021 | 0.024 | 0.049 | 0.114 |
| kv_get_finished | recv_overlap | 1927 | 0.067 | 0.067 | 0.116 | 0.139 | 0.256 | 0.410 |
| kv_get_finished | max_active_recv_1 | 1399 | 0.058 | 0.055 | 0.102 | 0.120 | 0.246 | 0.410 |
| kv_get_finished | max_active_recv_ge2 | 528 | 0.089 | 0.083 | 0.142 | 0.164 | 0.275 | 0.360 |
| sample_tokens | no_recv_overlap | 5117 | 0.546 | 0.528 | 0.576 | 0.597 | 0.679 | 117.019 |
| sample_tokens | recv_overlap | 1448 | 0.573 | 0.554 | 0.687 | 0.726 | 0.776 | 1.111 |
| sample_tokens | max_active_recv_1 | 1149 | 0.567 | 0.552 | 0.667 | 0.704 | 0.767 | 1.111 |
| sample_tokens | max_active_recv_ge2 | 299 | 0.596 | 0.565 | 0.728 | 0.746 | 0.818 | 0.975 |
| sample | no_recv_overlap | 5131 | 0.282 | 0.253 | 0.285 | 0.303 | 0.474 | 116.764 |
| sample | recv_overlap | 1434 | 0.286 | 0.270 | 0.361 | 0.391 | 0.449 | 0.762 |
| sample | max_active_recv_1 | 1136 | 0.282 | 0.267 | 0.350 | 0.385 | 0.441 | 0.762 |
| sample | max_active_recv_ge2 | 298 | 0.301 | 0.281 | 0.385 | 0.412 | 0.495 | 0.556 |
| bookkeeping | no_recv_overlap | 5135 | 0.113 | 0.117 | 0.139 | 0.152 | 0.176 | 0.217 |
| bookkeeping | recv_overlap | 1430 | 0.130 | 0.123 | 0.164 | 0.183 | 0.212 | 0.241 |
| bookkeeping | max_active_recv_1 | 1135 | 0.128 | 0.123 | 0.157 | 0.177 | 0.208 | 0.241 |
| bookkeeping | max_active_recv_ge2 | 295 | 0.136 | 0.124 | 0.190 | 0.203 | 0.215 | 0.230 |

## ITL By KV-Receive Overlap
| group | n | mean_ms | p50_ms | p90_ms | p95_ms | p99_ms | max_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_recv_overlap | nan | nan | nan | nan | nan | nan | nan |
| recv_overlap | nan | nan | nan | nan | nan | nan | nan |
| max_active_recv_1 | nan | nan | nan | nan | nan | nan | nan |
| max_active_recv_ge2 | nan | nan | nan | nan | nan | nan | nan |

## KV Receive / Transfer Duration
n=702, mean=17.144 ms, p50=12.786 ms, p90=29.185 ms, p99=42.368 ms, max=52.299 ms

## NIXL Transfer Telemetry
xfer_duration_ms: n=702, mean=17.144, p50=12.786, p90=29.185, p99=42.368, max=52.299
throughput_GBps: mean=1.285, p50=1.306, p90=2.620, p99=3.078
poll_slack_ms: mean=0.044, p50=0.027, p90=0.063, p99=0.340
