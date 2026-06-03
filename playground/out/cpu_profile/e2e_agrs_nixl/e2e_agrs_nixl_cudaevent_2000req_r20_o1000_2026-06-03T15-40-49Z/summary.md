# CPU Profile Summary

run_name: `e2e_agrs_nixl_cudaevent_2000req_r20_o1000_2026-06-03T15-40-49Z`

trace_dir: `playground/log/e2e_agrs_nixl/e2e_agrs_nixl_cudaevent_2000req_r20_o1000_2026-06-03T15-40-49Z/pd_trace`

main_window_epoch: `1780501492.976046` to `1780501708.583020` (`215.607s`)

| metric | role_dp | count | avg | p50 | p90 | p95 | p99 | min | max | unit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| engine_step_ms | decode_dp0 | 8512 | 13.624 | 13.832 | 14.142 | 14.252 | 22.029 | 0.005 | 36.065 | ms |
| engine_step_ms | decode_dp1 | 8514 | 13.615 | 13.829 | 14.142 | 14.246 | 21.998 | 0.004 | 36.031 | ms |
| engine_step_ms | prefill_dp0 | 10901 | 2.867 | 0.005 | 0.306 | 19.789 | 75.718 | 0.003 | 90.646 | ms |
| engine_step_ms | prefill_dp1 | 10895 | 2.360 | 0.005 | 0.247 | 19.050 | 72.886 | 0.003 | 91.075 | ms |
| forward_ms | decode_dp0 | 8501 | 0.344 | 0.335 | 0.394 | 0.421 | 0.505 | 0.240 | 1.038 | ms |
| forward_ms | decode_dp1 | 8494 | 0.327 | 0.327 | 0.378 | 0.404 | 0.465 | 0.226 | 0.813 | ms |
| forward_ms | prefill_dp0 | 1012 | 22.573 | 10.506 | 70.038 | 71.141 | 73.574 | 7.197 | 83.325 | ms |
| forward_ms | prefill_dp1 | 857 | 20.783 | 10.367 | 67.331 | 68.391 | 70.110 | 7.122 | 83.182 | ms |
| batch_n_reqs | decode_dp0 | 8501 | 87.904 | 98.000 | 109.000 | 111.000 | 115.000 | 1.000 | 117.000 | requests |
| batch_n_reqs | decode_dp1 | 8494 | 87.259 | 98.000 | 109.000 | 110.000 | 113.000 | 1.000 | 119.000 | requests |
| batch_n_reqs | prefill_dp0 | 1012 | 1.069 | 1.000 | 1.000 | 2.000 | 2.000 | 1.000 | 3.000 | requests |
| batch_n_reqs | prefill_dp1 | 857 | 1.071 | 1.000 | 1.000 | 2.000 | 2.000 | 1.000 | 3.000 | requests |
| batch_n_sched_tokens | decode_dp0 | 8501 | 87.904 | 98.000 | 109.000 | 111.000 | 115.000 | 1.000 | 117.000 | tokens |
| batch_n_sched_tokens | decode_dp1 | 8494 | 87.259 | 98.000 | 109.000 | 110.000 | 113.000 | 1.000 | 119.000 | tokens |
| batch_n_sched_tokens | prefill_dp0 | 1012 | 261.768 | 161.000 | 687.000 | 784.000 | 981.000 | 12.000 | 1365.000 | tokens |
| batch_n_sched_tokens | prefill_dp1 | 857 | 244.060 | 152.000 | 650.000 | 779.000 | 980.000 | 12.000 | 1121.000 | tokens |
| per_request_scheduled_tokens | decode_dp0 | 747270 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | tokens |
| per_request_scheduled_tokens | decode_dp1 | 741182 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | tokens |
| per_request_scheduled_tokens | prefill_dp0 | 1082 | 244.833 | 133.000 | 655.000 | 776.000 | 892.000 | 12.000 | 1030.000 | tokens |
| per_request_scheduled_tokens | prefill_dp1 | 918 | 227.842 | 125.000 | 617.000 | 774.000 | 870.000 | 12.000 | 1018.000 | tokens |
| per_request_step_counts | decode_dp0 | 1004 | 744.293 | 934.000 | 1000.000 | 1000.000 | 1000.000 | 5.000 | 1000.000 | steps |
| per_request_step_counts | decode_dp1 | 996 | 744.159 | 943.000 | 1000.000 | 1000.000 | 1000.000 | 3.000 | 1000.000 | steps |
| per_request_step_counts | prefill_dp0 | 1082 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | steps |
| per_request_step_counts | prefill_dp1 | 918 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | steps |
| kv_transfer_bytes | decode_dp0 | 1004 | 23936044.876 | 14155776.000 | 62914560.000 | 77070336.000 | 88080384.000 | 1572864.000 | 102236160.000 | bytes |
| kv_transfer_bytes | decode_dp1 | 996 | 24136198.169 | 14155776.000 | 66060288.000 | 77070336.000 | 88080384.000 | 1572864.000 | 100663296.000 | bytes |
| kv_transfer_ms | decode_dp0 | 1004 | 18.337 | 14.362 | 30.969 | 34.532 | 41.266 | 0.361 | 48.222 | ms |
| kv_transfer_ms | decode_dp1 | 996 | 18.416 | 14.258 | 31.715 | 35.425 | 42.200 | 0.497 | 55.283 | ms |
