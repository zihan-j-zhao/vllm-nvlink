# P/D Overlap Summary
trace_dir: `playground/log/moe_pd/telemetry_5k_starttime_2026-06-01T02-23-22Z/pd_trace`
window_s: 37.417
decode_trace_files: 4
overlap_interval_source: nixl
kv_recv_intervals: 745
software_recv_intervals: 745
nixl_xfer_intervals: 745
decode_engine_step_intervals: 5256
decode_forward_intervals: 5188
decode_phase_intervals: 46942
itl_samples: 0

## Decode Engine Step Duration By KV-Receive Overlap
| group | n | mean_ms | p50_ms | p90_ms | p95_ms | p99_ms | max_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_recv_overlap | 3513 | 12.798 | 12.358 | 12.835 | 13.200 | 13.814 | 818.393 |
| recv_overlap | 1743 | 15.532 | 12.384 | 13.449 | 21.079 | 36.812 | 1036.932 |
| max_active_recv_1 | 1280 | 12.815 | 12.370 | 13.135 | 16.040 | 22.662 | 38.264 |
| max_active_recv_ge2 | 463 | 23.042 | 12.441 | 22.458 | 27.852 | 324.378 | 1036.932 |

## Decode Forward Launch-Scope Duration By KV-Receive Overlap
These CPU intervals surround `_model_forward()` without CUDA sync; they are useful as launch-scope context, not full GPU execution time.

| group | n | mean_ms | p50_ms | p90_ms | p95_ms | p99_ms | max_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_recv_overlap | 3744 | 0.260 | 0.271 | 0.305 | 0.322 | 0.347 | 8.255 |
| recv_overlap | 1444 | 0.286 | 0.275 | 0.331 | 0.349 | 0.404 | 5.292 |
| max_active_recv_1 | 1121 | 0.265 | 0.276 | 0.328 | 0.345 | 0.379 | 3.468 |
| max_active_recv_ge2 | 323 | 0.362 | 0.274 | 0.336 | 0.368 | 4.043 | 5.292 |

## Decode Subphase Duration By KV-Receive Overlap
Rows are CPU wall-clock intervals. `sample` and `bookkeeping` were added for the steady-state follow-up run; older traces may have zero rows for those phases.

| phase | group | n | mean_ms | p50_ms | p90_ms | p95_ms | p99_ms | max_ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| engine_step_fn | no_recv_overlap | 3513 | 12.791 | 12.351 | 12.828 | 13.193 | 13.800 | 818.384 |
| engine_step_fn | recv_overlap | 1743 | 15.524 | 12.376 | 13.440 | 21.072 | 36.799 | 1036.922 |
| engine_step_fn | max_active_recv_1 | 1280 | 12.808 | 12.363 | 13.128 | 16.035 | 22.656 | 38.248 |
| engine_step_fn | max_active_recv_ge2 | 463 | 23.035 | 12.432 | 22.452 | 27.844 | 324.370 | 1036.922 |
| engine_post_step | no_recv_overlap | 4343 | 0.007 | 0.007 | 0.009 | 0.010 | 0.013 | 0.030 |
| engine_post_step | recv_overlap | 913 | 0.007 | 0.007 | 0.009 | 0.010 | 0.014 | 0.026 |
| engine_post_step | max_active_recv_1 | 774 | 0.007 | 0.007 | 0.009 | 0.010 | 0.014 | 0.026 |
| engine_post_step | max_active_recv_ge2 | 139 | 0.008 | 0.007 | 0.010 | 0.011 | 0.014 | 0.015 |
| kv_start_load | no_recv_overlap | 3862 | 0.011 | 0.010 | 0.011 | 0.011 | 0.027 | 0.608 |
| kv_start_load | recv_overlap | 1364 | 3.225 | 0.011 | 6.390 | 14.420 | 25.862 | 170.040 |
| kv_start_load | max_active_recv_1 | 1008 | 2.250 | 0.011 | 5.891 | 10.592 | 20.252 | 35.096 |
| kv_start_load | max_active_recv_ge2 | 356 | 5.985 | 2.546 | 10.573 | 20.225 | 94.775 | 170.040 |
| kv_finalize | no_recv_overlap | 3824 | 0.031 | 0.026 | 0.042 | 0.070 | 0.098 | 0.151 |
| kv_finalize | recv_overlap | 1402 | 0.102 | 0.073 | 0.148 | 0.189 | 0.368 | 20.806 |
| kv_finalize | max_active_recv_1 | 1081 | 0.070 | 0.059 | 0.126 | 0.148 | 0.261 | 0.334 |
| kv_finalize | max_active_recv_ge2 | 321 | 0.210 | 0.102 | 0.223 | 0.354 | 0.979 | 20.806 |
| kv_wait_for_save | no_recv_overlap | 3824 | 0.001 | 0.001 | 0.001 | 0.001 | 0.002 | 0.022 |
| kv_wait_for_save | recv_overlap | 1364 | 0.001 | 0.001 | 0.001 | 0.001 | 0.002 | 0.022 |
| kv_wait_for_save | max_active_recv_1 | 1053 | 0.001 | 0.001 | 0.001 | 0.001 | 0.002 | 0.022 |
| kv_wait_for_save | max_active_recv_ge2 | 311 | 0.001 | 0.001 | 0.001 | 0.001 | 0.002 | 0.010 |
| kv_get_finished | no_recv_overlap | 3830 | 0.023 | 0.019 | 0.026 | 0.054 | 0.089 | 0.131 |
| kv_get_finished | recv_overlap | 1396 | 0.091 | 0.065 | 0.128 | 0.166 | 0.349 | 20.772 |
| kv_get_finished | max_active_recv_1 | 1075 | 0.060 | 0.050 | 0.110 | 0.129 | 0.245 | 0.322 |
| kv_get_finished | max_active_recv_ge2 | 321 | 0.196 | 0.089 | 0.206 | 0.334 | 0.959 | 20.772 |
| sample_tokens | no_recv_overlap | 4085 | 0.732 | 0.550 | 0.592 | 0.609 | 0.757 | 251.566 |
| sample_tokens | recv_overlap | 1103 | 0.579 | 0.561 | 0.644 | 0.727 | 0.843 | 1.327 |
| sample_tokens | max_active_recv_1 | 920 | 0.574 | 0.560 | 0.633 | 0.719 | 0.801 | 1.300 |
| sample_tokens | max_active_recv_ge2 | 183 | 0.601 | 0.575 | 0.703 | 0.781 | 1.080 | 1.327 |
| sample | no_recv_overlap | 4085 | 0.435 | 0.263 | 0.291 | 0.302 | 0.520 | 230.846 |
| sample | recv_overlap | 1103 | 0.287 | 0.271 | 0.332 | 0.408 | 0.540 | 0.976 |
| sample | max_active_recv_1 | 921 | 0.283 | 0.270 | 0.314 | 0.394 | 0.484 | 0.974 |
| sample | max_active_recv_ge2 | 182 | 0.306 | 0.276 | 0.405 | 0.454 | 0.752 | 0.976 |
| bookkeeping | no_recv_overlap | 4091 | 0.126 | 0.124 | 0.148 | 0.156 | 0.174 | 0.213 |
| bookkeeping | recv_overlap | 1097 | 0.128 | 0.126 | 0.152 | 0.161 | 0.191 | 0.267 |
| bookkeeping | max_active_recv_1 | 916 | 0.128 | 0.125 | 0.150 | 0.160 | 0.189 | 0.267 |
| bookkeeping | max_active_recv_ge2 | 181 | 0.131 | 0.127 | 0.159 | 0.166 | 0.199 | 0.248 |

## ITL By KV-Receive Overlap
| group | n | mean_ms | p50_ms | p90_ms | p95_ms | p99_ms | max_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_recv_overlap | nan | nan | nan | nan | nan | nan | nan |
| recv_overlap | nan | nan | nan | nan | nan | nan | nan |
| max_active_recv_1 | nan | nan | nan | nan | nan | nan | nan |
| max_active_recv_ge2 | nan | nan | nan | nan | nan | nan | nan |

## KV Receive / Transfer Duration
n=745, mean=18.533 ms, p50=12.397 ms, p90=35.373 ms, p99=136.237 ms, max=544.280 ms

## NIXL Transfer Telemetry
xfer_duration_ms: n=745, mean=18.533, p50=12.397, p90=35.373, p99=136.237, max=544.280
throughput_GBps: mean=1.729, p50=1.856, p90=2.561, p99=3.731
poll_slack_ms: mean=1.057, p50=0.043, p90=0.393, p99=39.275
