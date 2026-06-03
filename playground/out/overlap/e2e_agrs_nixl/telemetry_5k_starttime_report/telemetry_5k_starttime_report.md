# 5k StartTime-Aligned NIXL Telemetry Report

Run: `REQUEST_RATE=20 REQUEST_COUNT=5000 MAX_OUTPUT_TOKENS=500`; AIPerf completed 5,000 requests with 4,999 valid and 1 error. Analysis uses `--recv-interval-source nixl --nixl-start start_time --no-itl`.

## Window Summary
| window | xfers | xfer p50 | xfer p90 | xfer p99 | GB/s p50 | no p99 | active>=2 n | active>=2 p90 | active>=2 p99 | delta p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| early +120s | 745 | 12.397 | 35.373 | 136.237 | 1.856 | 13.814 | 463 | 22.458 | 324.378 | 310.565 |
| mid +180s | 1133 | 11.142 | 26.962 | 38.859 | 1.933 | 12.697 | 710 | 15.610 | 26.707 | 14.010 |
| mid +240s | 1193 | 12.562 | 27.097 | 37.919 | 1.663 | 12.909 | 970 | 14.444 | 24.640 | 11.731 |
| mid +300s | 1226 | 12.712 | 27.218 | 37.700 | 1.435 | 12.993 | 1142 | 14.766 | 25.284 | 12.292 |
| late +360s | 702 | 12.786 | 29.185 | 42.368 | 1.306 | 13.000 | 664 | 18.275 | 26.885 | 13.885 |

## Key Reading
- NIXL `startTime` alignment removes the artificial software polling lifetime and keeps poll slack near sub-ms in steady windows.
- The main steady windows show active>=2 p99 around 24-27 ms versus no-overlap p99 around 12.7-13.0 ms.
- The early window contains large startup/drain outliers; read it separately from steady windows.
- `kv_start_load` remains the clearest attribution path: p99 rises from ~0.02 ms no-overlap to ~18-24 ms when active>=2 in steady windows.

## Plots
- engine_step_p99_by_window.png
- engine_step_scatter_by_active_panel_zoom.png
- engine_step_scatter_by_nixl_overlap_full.png
- engine_step_scatter_by_nixl_overlap_zoom.png
- kv_start_load_scatter_by_active_full.png
- kv_start_load_scatter_by_active_zoom.png
- nixl_xfer_duration_throughput_scatter_full.png
- nixl_xfer_duration_throughput_scatter_zoom.png
