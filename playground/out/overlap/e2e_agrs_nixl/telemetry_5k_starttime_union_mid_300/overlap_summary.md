# P/D Overlap Summary
trace_dir: `playground/log/moe_pd/telemetry_5k_starttime_2026-06-01T02-23-22Z/pd_trace`
window_s: 59.998
decode_trace_files: 4
overlap_interval_source: nixl
kv_recv_intervals: 1226
software_recv_intervals: 1226
nixl_xfer_intervals: 1226
decode_engine_step_intervals: 9558
decode_forward_intervals: 9560
decode_phase_intervals: 86036
itl_samples: 0

## Decode Engine Step Duration By KV-Receive Overlap
| group | n | mean_ms | p50_ms | p90_ms | p95_ms | p99_ms | max_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_recv_overlap | 5686 | 12.388 | 12.372 | 12.656 | 12.818 | 12.993 | 13.173 |
| recv_overlap | 3872 | 12.730 | 12.381 | 12.910 | 14.433 | 22.171 | 37.842 |
| max_active_recv_1 | 2730 | 12.545 | 12.373 | 12.791 | 12.995 | 16.675 | 29.428 |
| max_active_recv_ge2 | 1142 | 13.172 | 12.409 | 14.766 | 18.424 | 25.284 | 37.842 |

## Decode Forward Launch-Scope Duration By KV-Receive Overlap
These CPU intervals surround `_model_forward()` without CUDA sync; they are useful as launch-scope context, not full GPU execution time.

| group | n | mean_ms | p50_ms | p90_ms | p95_ms | p99_ms | max_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_recv_overlap | 6054 | 0.235 | 0.206 | 0.336 | 0.346 | 0.359 | 0.473 |
| recv_overlap | 3506 | 0.251 | 0.228 | 0.338 | 0.349 | 0.374 | 0.437 |
| max_active_recv_1 | 2527 | 0.250 | 0.225 | 0.339 | 0.349 | 0.370 | 0.437 |
| max_active_recv_ge2 | 979 | 0.256 | 0.237 | 0.337 | 0.349 | 0.379 | 0.437 |

## Decode Subphase Duration By KV-Receive Overlap
Rows are CPU wall-clock intervals. `sample` and `bookkeeping` were added for the steady-state follow-up run; older traces may have zero rows for those phases.

| phase | group | n | mean_ms | p50_ms | p90_ms | p95_ms | p99_ms | max_ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| engine_step_fn | no_recv_overlap | 5686 | 12.381 | 12.365 | 12.650 | 12.810 | 12.984 | 13.164 |
| engine_step_fn | recv_overlap | 3872 | 12.723 | 12.374 | 12.903 | 14.426 | 22.165 | 37.835 |
| engine_step_fn | max_active_recv_1 | 2730 | 12.538 | 12.366 | 12.785 | 12.988 | 16.669 | 29.421 |
| engine_step_fn | max_active_recv_ge2 | 1142 | 13.165 | 12.402 | 14.759 | 18.418 | 25.279 | 37.835 |
| engine_post_step | no_recv_overlap | 7136 | 0.007 | 0.007 | 0.008 | 0.010 | 0.012 | 0.020 |
| engine_post_step | recv_overlap | 2424 | 0.007 | 0.007 | 0.008 | 0.010 | 0.014 | 0.018 |
| engine_post_step | max_active_recv_1 | 1960 | 0.007 | 0.007 | 0.008 | 0.009 | 0.014 | 0.018 |
| engine_post_step | max_active_recv_ge2 | 464 | 0.007 | 0.007 | 0.008 | 0.010 | 0.014 | 0.017 |
| kv_start_load | no_recv_overlap | 6312 | 0.011 | 0.010 | 0.011 | 0.011 | 0.017 | 0.034 |
| kv_start_load | recv_overlap | 3248 | 2.153 | 0.011 | 8.067 | 10.225 | 17.337 | 35.336 |
| kv_start_load | max_active_recv_1 | 2337 | 1.668 | 0.010 | 7.076 | 9.542 | 14.102 | 24.913 |
| kv_start_load | max_active_recv_ge2 | 911 | 3.395 | 1.831 | 9.411 | 12.127 | 20.299 | 35.336 |
| kv_finalize | no_recv_overlap | 6203 | 0.029 | 0.027 | 0.031 | 0.036 | 0.068 | 0.258 |
| kv_finalize | recv_overlap | 3357 | 0.086 | 0.087 | 0.146 | 0.181 | 0.293 | 0.542 |
| kv_finalize | max_active_recv_1 | 2441 | 0.075 | 0.074 | 0.134 | 0.152 | 0.260 | 0.412 |
| kv_finalize | max_active_recv_ge2 | 916 | 0.115 | 0.103 | 0.181 | 0.229 | 0.330 | 0.542 |
| kv_wait_for_save | no_recv_overlap | 6243 | 0.001 | 0.001 | 0.002 | 0.002 | 0.002 | 0.022 |
| kv_wait_for_save | recv_overlap | 3317 | 0.001 | 0.001 | 0.002 | 0.002 | 0.002 | 0.020 |
| kv_wait_for_save | max_active_recv_1 | 2419 | 0.001 | 0.001 | 0.002 | 0.002 | 0.002 | 0.020 |
| kv_wait_for_save | max_active_recv_ge2 | 898 | 0.001 | 0.001 | 0.002 | 0.002 | 0.002 | 0.007 |
| kv_get_finished | no_recv_overlap | 6212 | 0.021 | 0.020 | 0.022 | 0.025 | 0.049 | 0.248 |
| kv_get_finished | recv_overlap | 3348 | 0.074 | 0.075 | 0.127 | 0.157 | 0.277 | 0.524 |
| kv_get_finished | max_active_recv_1 | 2438 | 0.065 | 0.065 | 0.117 | 0.133 | 0.247 | 0.393 |
| kv_get_finished | max_active_recv_ge2 | 910 | 0.100 | 0.088 | 0.157 | 0.215 | 0.312 | 0.524 |
| sample_tokens | no_recv_overlap | 7042 | 0.545 | 0.545 | 0.582 | 0.597 | 0.630 | 0.855 |
| sample_tokens | recv_overlap | 2516 | 0.589 | 0.566 | 0.706 | 0.737 | 0.797 | 1.179 |
| sample_tokens | max_active_recv_1 | 1971 | 0.585 | 0.563 | 0.700 | 0.732 | 0.793 | 1.116 |
| sample_tokens | max_active_recv_ge2 | 545 | 0.604 | 0.579 | 0.723 | 0.747 | 0.827 | 1.179 |
| sample | no_recv_overlap | 7050 | 0.264 | 0.262 | 0.289 | 0.300 | 0.324 | 0.558 |
| sample | recv_overlap | 2510 | 0.295 | 0.277 | 0.370 | 0.412 | 0.461 | 0.806 |
| sample | max_active_recv_1 | 1965 | 0.292 | 0.275 | 0.366 | 0.408 | 0.457 | 0.799 |
| sample | max_active_recv_ge2 | 545 | 0.306 | 0.288 | 0.392 | 0.414 | 0.481 | 0.806 |
| bookkeeping | no_recv_overlap | 7074 | 0.124 | 0.122 | 0.143 | 0.153 | 0.178 | 0.329 |
| bookkeeping | recv_overlap | 2486 | 0.130 | 0.125 | 0.158 | 0.182 | 0.220 | 0.350 |
| bookkeeping | max_active_recv_1 | 1950 | 0.130 | 0.124 | 0.155 | 0.175 | 0.217 | 0.350 |
| bookkeeping | max_active_recv_ge2 | 536 | 0.134 | 0.126 | 0.170 | 0.195 | 0.230 | 0.244 |

## ITL By KV-Receive Overlap
| group | n | mean_ms | p50_ms | p90_ms | p95_ms | p99_ms | max_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_recv_overlap | nan | nan | nan | nan | nan | nan | nan |
| recv_overlap | nan | nan | nan | nan | nan | nan | nan |
| max_active_recv_1 | nan | nan | nan | nan | nan | nan | nan |
| max_active_recv_ge2 | nan | nan | nan | nan | nan | nan | nan |

## KV Receive / Transfer Duration
n=1226, mean=16.487 ms, p50=12.712 ms, p90=27.218 ms, p99=37.700 ms, max=46.064 ms

## NIXL Transfer Telemetry
xfer_duration_ms: n=1226, mean=16.487, p50=12.712, p90=27.218, p99=37.700, max=46.064
throughput_GBps: mean=1.312, p50=1.435, p90=2.660, p99=3.021
poll_slack_ms: mean=0.060, p50=0.035, p90=0.075, p99=0.429
