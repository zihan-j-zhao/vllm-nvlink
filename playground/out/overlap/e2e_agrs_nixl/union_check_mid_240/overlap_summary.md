# P/D Overlap Summary
trace_dir: `playground/log/moe_pd/telemetry_5k_starttime_2026-06-01T02-23-22Z/pd_trace`
window_s: 9.991
decode_trace_files: 4
overlap_interval_source: nixl
kv_recv_intervals: 184
software_recv_intervals: 184
nixl_xfer_intervals: 184
decode_engine_step_intervals: 1630
decode_forward_intervals: 1632
decode_phase_intervals: 14686
itl_samples: 0

## Decode Engine Step Duration By KV-Receive Overlap
| group | n | mean_ms | p50_ms | p90_ms | p95_ms | p99_ms | max_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_recv_overlap | 1136 | 12.133 | 12.144 | 12.314 | 12.364 | 12.460 | 13.197 |
| recv_overlap | 494 | 12.442 | 12.138 | 12.351 | 13.458 | 21.666 | 24.474 |
| max_active_recv_1 | 384 | 12.395 | 12.137 | 12.342 | 12.869 | 21.516 | 23.518 |
| max_active_recv_ge2 | 110 | 12.605 | 12.141 | 12.896 | 14.385 | 23.914 | 24.474 |

## Decode Forward Launch-Scope Duration By KV-Receive Overlap
These CPU intervals surround `_model_forward()` without CUDA sync; they are useful as launch-scope context, not full GPU execution time.

| group | n | mean_ms | p50_ms | p90_ms | p95_ms | p99_ms | max_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_recv_overlap | 1234 | 0.238 | 0.208 | 0.307 | 0.316 | 0.325 | 0.373 |
| recv_overlap | 398 | 0.270 | 0.291 | 0.351 | 0.373 | 0.392 | 0.402 |
| max_active_recv_1 | 334 | 0.268 | 0.288 | 0.352 | 0.374 | 0.393 | 0.402 |
| max_active_recv_ge2 | 64 | 0.280 | 0.298 | 0.348 | 0.360 | 0.379 | 0.390 |

## Decode Subphase Duration By KV-Receive Overlap
Rows are CPU wall-clock intervals. `sample` and `bookkeeping` were added for the steady-state follow-up run; older traces may have zero rows for those phases.

| phase | group | n | mean_ms | p50_ms | p90_ms | p95_ms | p99_ms | max_ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| engine_step_fn | no_recv_overlap | 1136 | 12.126 | 12.136 | 12.308 | 12.355 | 12.452 | 13.188 |
| engine_step_fn | recv_overlap | 494 | 12.434 | 12.131 | 12.344 | 13.451 | 21.660 | 24.468 |
| engine_step_fn | max_active_recv_1 | 384 | 12.387 | 12.130 | 12.335 | 12.862 | 21.508 | 23.511 |
| engine_step_fn | max_active_recv_ge2 | 110 | 12.598 | 12.133 | 12.889 | 14.378 | 23.907 | 24.468 |
| engine_post_step | no_recv_overlap | 1388 | 0.007 | 0.007 | 0.008 | 0.009 | 0.012 | 0.023 |
| engine_post_step | recv_overlap | 244 | 0.007 | 0.007 | 0.009 | 0.010 | 0.011 | 0.014 |
| engine_post_step | max_active_recv_1 | 224 | 0.007 | 0.007 | 0.009 | 0.010 | 0.011 | 0.014 |
| engine_post_step | max_active_recv_ge2 | 20 | 0.007 | 0.007 | 0.010 | 0.010 | 0.010 | 0.010 |
| kv_start_load | no_recv_overlap | 1250 | 0.011 | 0.010 | 0.011 | 0.012 | 0.016 | 0.030 |
| kv_start_load | recv_overlap | 382 | 2.338 | 0.012 | 7.070 | 9.714 | 16.796 | 22.232 |
| kv_start_load | max_active_recv_1 | 303 | 2.026 | 0.011 | 6.489 | 9.688 | 18.749 | 22.232 |
| kv_start_load | max_active_recv_ge2 | 79 | 3.538 | 2.847 | 7.954 | 9.509 | 11.779 | 13.591 |
| kv_finalize | no_recv_overlap | 1253 | 0.033 | 0.027 | 0.046 | 0.065 | 0.081 | 0.099 |
| kv_finalize | recv_overlap | 379 | 0.073 | 0.067 | 0.132 | 0.143 | 0.238 | 0.366 |
| kv_finalize | max_active_recv_1 | 314 | 0.066 | 0.059 | 0.125 | 0.137 | 0.152 | 0.290 |
| kv_finalize | max_active_recv_ge2 | 65 | 0.108 | 0.099 | 0.174 | 0.187 | 0.366 | 0.366 |
| kv_wait_for_save | no_recv_overlap | 1258 | 0.001 | 0.001 | 0.001 | 0.001 | 0.002 | 0.015 |
| kv_wait_for_save | recv_overlap | 374 | 0.001 | 0.001 | 0.001 | 0.002 | 0.002 | 0.016 |
| kv_wait_for_save | max_active_recv_1 | 310 | 0.001 | 0.001 | 0.001 | 0.002 | 0.002 | 0.016 |
| kv_wait_for_save | max_active_recv_ge2 | 64 | 0.001 | 0.001 | 0.001 | 0.002 | 0.002 | 0.002 |
| kv_get_finished | no_recv_overlap | 1253 | 0.025 | 0.020 | 0.039 | 0.048 | 0.067 | 0.093 |
| kv_get_finished | recv_overlap | 379 | 0.062 | 0.054 | 0.114 | 0.125 | 0.219 | 0.345 |
| kv_get_finished | max_active_recv_1 | 314 | 0.056 | 0.050 | 0.107 | 0.118 | 0.133 | 0.271 |
| kv_get_finished | max_active_recv_ge2 | 65 | 0.093 | 0.084 | 0.153 | 0.167 | 0.343 | 0.345 |
| sample_tokens | no_recv_overlap | 1319 | 0.550 | 0.548 | 0.589 | 0.597 | 0.623 | 0.757 |
| sample_tokens | recv_overlap | 313 | 0.575 | 0.562 | 0.638 | 0.697 | 0.800 | 0.897 |
| sample_tokens | max_active_recv_1 | 281 | 0.576 | 0.562 | 0.638 | 0.708 | 0.802 | 0.897 |
| sample_tokens | max_active_recv_ge2 | 32 | 0.570 | 0.552 | 0.657 | 0.677 | 0.730 | 0.752 |
| sample | no_recv_overlap | 1319 | 0.269 | 0.266 | 0.296 | 0.304 | 0.329 | 0.467 |
| sample | recv_overlap | 313 | 0.289 | 0.278 | 0.330 | 0.398 | 0.478 | 0.602 |
| sample | max_active_recv_1 | 281 | 0.289 | 0.278 | 0.324 | 0.401 | 0.483 | 0.602 |
| sample | max_active_recv_ge2 | 32 | 0.287 | 0.268 | 0.358 | 0.376 | 0.425 | 0.447 |
| bookkeeping | no_recv_overlap | 1324 | 0.121 | 0.119 | 0.135 | 0.140 | 0.147 | 0.187 |
| bookkeeping | recv_overlap | 308 | 0.122 | 0.119 | 0.136 | 0.139 | 0.152 | 0.180 |
| bookkeeping | max_active_recv_1 | 277 | 0.122 | 0.119 | 0.136 | 0.139 | 0.152 | 0.180 |
| bookkeeping | max_active_recv_ge2 | 31 | 0.120 | 0.117 | 0.133 | 0.137 | 0.144 | 0.146 |

## ITL By KV-Receive Overlap
| group | n | mean_ms | p50_ms | p90_ms | p95_ms | p99_ms | max_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_recv_overlap | nan | nan | nan | nan | nan | nan | nan |
| recv_overlap | nan | nan | nan | nan | nan | nan | nan |
| max_active_recv_1 | nan | nan | nan | nan | nan | nan | nan |
| max_active_recv_ge2 | nan | nan | nan | nan | nan | nan | nan |

## KV Receive / Transfer Duration
n=184, mean=10.806 ms, p50=7.154 ms, p90=25.223 ms, p99=37.975 ms, max=43.544 ms

## NIXL Transfer Telemetry
xfer_duration_ms: n=184, mean=10.806, p50=7.154, p90=25.223, p99=37.975, max=43.544
throughput_GBps: mean=2.018, p50=2.139, p90=2.632, p99=3.685
poll_slack_ms: mean=0.169, p50=0.049, p90=0.355, p99=0.746
