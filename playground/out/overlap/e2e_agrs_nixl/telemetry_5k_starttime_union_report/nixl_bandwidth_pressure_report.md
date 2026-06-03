# NIXL Bandwidth Pressure Report

Pressure is computed per engine step by clipping all overlapping NIXL telemetry transfer intervals to the engine-step window. Each transfer is assigned a constant byte rate: `bytes_transferred / xfer_duration`.

- `max_xfer_pressure_GBps`: maximum aggregate active transfer rate at any instant inside the engine step.
- `avg_step_xfer_pressure_GBps`: time-weighted average aggregate rate over the whole engine step.
- `avg_active_xfer_pressure_GBps`: time-weighted average aggregate rate only over transfer-active time.

## Outputs

- engine_step_pressure_rows.csv
- engine_step_vs_max_xfer_pressure.png
- engine_step_vs_max_xfer_pressure.pdf
- engine_step_vs_max_xfer_pressure_steady_mid.png
- engine_step_vs_max_xfer_pressure_steady_mid.pdf

## Steady Mid Windows By Pressure Bucket

| max pressure bucket GB/s | n | dur p50 | dur p90 | dur p99 | active>=2 % | union ov p50 | intensity p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| [0.0, 0.5) | 1452 | 12.375 | 12.660 | 14.497 | 3.4 | 2.930 | 1.000 |
| [0.5, 1.0) | 669 | 12.378 | 12.746 | 15.169 | 22.9 | 9.734 | 1.000 |
| [1.0, 1.5) | 314 | 12.335 | 12.701 | 13.289 | 19.7 | 4.404 | 1.000 |
| [1.5, 2.0) | 1164 | 12.349 | 12.599 | 13.003 | 10.7 | 9.577 | 1.000 |
| [2.0, 3.0) | 2689 | 12.388 | 13.240 | 22.001 | 24.4 | 9.679 | 1.000 |
| [3.0, 10.0) | 1130 | 12.410 | 15.492 | 27.117 | 94.5 | 12.205 | 1.775 |
