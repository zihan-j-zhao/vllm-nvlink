# CPU Profile Summary

run_name: `e2e_agrs_nixl_cpu_2000req_r20_o1000_2026-06-02T06-55-58Z`

trace_dir: `playground/log/e2e_agrs_nixl/e2e_agrs_nixl_cpu_2000req_r20_o1000_2026-06-02T06-55-58Z/pd_trace`

main_window_epoch: `1780383598.206711` to `1780383812.975301` (`214.769s`)

| metric | role_dp | count | avg | p50 | p90 | p95 | p99 | min | max | unit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| engine_step_ms | decode_dp0 | 8480 | 13.672 | 14.059 | 14.414 | 14.533 | 19.969 | 0.004 | 43.027 | ms |
| engine_step_ms | decode_dp1 | 8480 | 13.691 | 14.060 | 14.409 | 14.525 | 19.884 | 0.004 | 43.162 | ms |
| engine_step_ms | prefill_dp0 | 11019 | 2.784 | 0.005 | 0.327 | 19.697 | 70.944 | 0.004 | 87.611 | ms |
| engine_step_ms | prefill_dp1 | 11016 | 2.305 | 0.004 | 0.217 | 19.109 | 68.998 | 0.003 | 88.715 | ms |
| forward_ms | decode_dp0 | 8421 | 0.301 | 0.317 | 0.341 | 0.350 | 0.386 | 0.137 | 0.890 | ms |
| forward_ms | decode_dp1 | 8439 | 0.237 | 0.226 | 0.290 | 0.309 | 0.382 | 0.162 | 2.174 | ms |
| forward_ms | prefill_dp0 | 1015 | 21.604 | 10.519 | 64.913 | 66.003 | 68.641 | 6.954 | 74.210 | ms |
| forward_ms | prefill_dp1 | 872 | 20.036 | 10.369 | 62.792 | 64.382 | 68.065 | 6.738 | 81.259 | ms |
| batch_n_reqs | decode_dp0 | 8421 | 88.960 | 100.000 | 110.000 | 112.000 | 116.000 | 1.000 | 120.000 | requests |
| batch_n_reqs | decode_dp1 | 8439 | 87.685 | 99.000 | 109.000 | 112.000 | 115.000 | 1.000 | 117.000 | requests |
| batch_n_reqs | prefill_dp0 | 1015 | 1.060 | 1.000 | 1.000 | 2.000 | 2.000 | 1.000 | 3.000 | requests |
| batch_n_reqs | prefill_dp1 | 872 | 1.060 | 1.000 | 1.000 | 2.000 | 2.000 | 1.000 | 3.000 | requests |
| batch_n_sched_tokens | decode_dp0 | 8421 | 88.960 | 100.000 | 110.000 | 112.000 | 116.000 | 1.000 | 120.000 | tokens |
| batch_n_sched_tokens | decode_dp1 | 8439 | 87.685 | 99.000 | 109.000 | 112.000 | 115.000 | 1.000 | 117.000 | tokens |
| batch_n_sched_tokens | prefill_dp0 | 1015 | 263.503 | 173.000 | 676.000 | 782.000 | 948.000 | 12.000 | 1788.000 | tokens |
| batch_n_sched_tokens | prefill_dp1 | 872 | 236.940 | 131.000 | 627.000 | 777.000 | 926.000 | 12.000 | 1340.000 | tokens |
| per_request_scheduled_tokens | decode_dp0 | 749135 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | tokens |
| per_request_scheduled_tokens | decode_dp1 | 739977 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | tokens |
| per_request_scheduled_tokens | prefill_dp0 | 1076 | 248.565 | 142.000 | 655.000 | 776.000 | 854.000 | 12.000 | 1011.000 | tokens |
| per_request_scheduled_tokens | prefill_dp1 | 924 | 223.606 | 113.000 | 608.000 | 775.000 | 906.000 | 12.000 | 1030.000 | tokens |
| per_request_step_counts | decode_dp0 | 1010 | 741.718 | 952.000 | 1000.000 | 1000.000 | 1000.000 | 3.000 | 1000.000 | steps |
| per_request_step_counts | decode_dp1 | 990 | 747.452 | 955.000 | 1000.000 | 1000.000 | 1000.000 | 3.000 | 1000.000 | steps |
| per_request_step_counts | prefill_dp0 | 1076 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | steps |
| per_request_step_counts | prefill_dp1 | 924 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | steps |
| kv_transfer_bytes | decode_dp0 | 1010 | 22990288.349 | 12582912.000 | 61341696.000 | 77070336.000 | 88080384.000 | 1572864.000 | 102236160.000 | bytes |
| kv_transfer_bytes | decode_dp1 | 990 | 25102273.939 | 14155776.000 | 66060288.000 | 77070336.000 | 86507520.000 | 1572864.000 | 100663296.000 | bytes |
| kv_transfer_ms | decode_dp0 | 1010 | 12.036 | 9.688 | 28.539 | 32.027 | 37.351 | 0.434 | 45.651 | ms |
| kv_transfer_ms | decode_dp1 | 990 | 18.827 | 14.484 | 31.737 | 35.011 | 43.146 | 0.431 | 49.655 | ms |
