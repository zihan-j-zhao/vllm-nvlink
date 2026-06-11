#!/usr/bin/env python3
"""Aggregate a 2D (ISL x batch) nsys sweep into resource-utilisation tables
and heatmaps.

Consumes ``summary_nsys.csv`` produced by ``postprocess_nsys.py`` (one row
per cell per GPU, with mean throughput-% per metric) and produces, per role
(prefill / decode):

  * a tidy CSV ``grid_<role>.csv`` with ISL, batch, and aggregated metrics,
  * heatmaps (ISL rows x batch cols) for HBM BW, NVLink BW and SM activity.

Metrics are nsys "throughput %" of peak (0-100), averaged across the GPUs of
a role and across the steady-state window.  HBM is also reported in GB/s
using the per-GPU peak from ``TARGET_INFO_GPU``.

Usage:
    python playground/moe_pd/aggregate_grid.py <sweep_dir>
    python playground/moe_pd/aggregate_grid.py <sweep_dir>/summary_nsys.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    print(f"error: matplotlib required ({exc})", file=sys.stderr)
    raise


def _to_float(s: str) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def _sum_cols(row: dict, substrs: list[str]) -> float:
    """Sum metric columns whose name contains ALL of the given substrings."""
    total = 0.0
    found = False
    for k, v in row.items():
        if all(s in k for s in substrs):
            val = _to_float(v)
            if not np.isnan(val):
                total += val
                found = True
    return total if found else float("nan")


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def aggregate(rows: list[dict]) -> dict[str, dict[tuple[int, int], dict]]:
    """Return {role: {(isl, batch): {metric: value}}} averaged over GPUs."""
    # bucket[role][(isl,batch)] = list of per-gpu metric dicts
    buckets: dict[str, dict[tuple[int, int], list[dict]]] = {}
    for r in rows:
        role = r.get("role", "unknown")
        if role not in ("prefill", "decode"):
            continue
        isl = int(_to_float(r.get("isl", "nan")))
        batch = r.get("concurrency") or ""
        if not batch:
            continue
        batch = int(_to_float(batch))
        peak_hbm = _to_float(r.get("peak_hbm_gbps", "nan"))
        hbm_r = _sum_cols(r, ["DRAM Read"])
        hbm_w = _sum_cols(r, ["DRAM Write"])
        hbm_tot = np.nansum([hbm_r, hbm_w])
        nvl_tx = _sum_cols(r, ["NVLink TX"])
        nvl_rx = _sum_cols(r, ["NVLink RX"])
        nvl_tot = np.nansum([nvl_tx, nvl_rx])
        sm = _sum_cols(r, ["SMs Active"])
        tensor = _sum_cols(r, ["Tensor Active"])
        metrics = {
            "hbm_read_pct": hbm_r,
            "hbm_write_pct": hbm_w,
            "hbm_total_pct": hbm_tot,
            "hbm_total_gbps": hbm_tot / 100.0 * peak_hbm if not np.isnan(peak_hbm) else float("nan"),
            "nvlink_tx_pct": nvl_tx,
            "nvlink_rx_pct": nvl_rx,
            "nvlink_total_pct": nvl_tot,
            "sm_active_pct": sm,
            "tensor_active_pct": tensor,
        }
        buckets.setdefault(role, {}).setdefault((isl, batch), []).append(metrics)

    out: dict[str, dict[tuple[int, int], dict]] = {}
    for role, cells in buckets.items():
        out[role] = {}
        for key, lst in cells.items():
            agg = {}
            for m in lst[0]:
                vals = [d[m] for d in lst]
                agg[m] = float(np.nanmean(vals)) if vals else float("nan")
            out[role][key] = agg
    return out


def write_tidy_csv(role: str, cells: dict, out_dir: Path) -> None:
    metric_keys = [
        "hbm_read_pct", "hbm_write_pct", "hbm_total_pct", "hbm_total_gbps",
        "nvlink_tx_pct", "nvlink_rx_pct", "nvlink_total_pct",
        "sm_active_pct", "tensor_active_pct",
    ]
    out_path = out_dir / f"grid_{role}.csv"
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["isl", "batch", *metric_keys])
        for (isl, batch) in sorted(cells):
            m = cells[(isl, batch)]
            w.writerow([isl, batch, *[f"{m[k]:.3f}" for k in metric_keys]])
    print(f"[aggregate] {out_path}")


def heatmap(role: str, cells: dict, metric: str, title: str, unit: str,
            out_dir: Path) -> None:
    isls = sorted({k[0] for k in cells})
    batches = sorted({k[1] for k in cells})
    grid = np.full((len(isls), len(batches)), np.nan)
    for i, isl in enumerate(isls):
        for j, b in enumerate(batches):
            if (isl, b) in cells:
                grid[i, j] = cells[(isl, b)][metric]

    fig, ax = plt.subplots(figsize=(1.1 * len(batches) + 2.5, 0.8 * len(isls) + 2))
    im = ax.imshow(grid, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(range(len(batches)))
    ax.set_xticklabels(batches)
    ax.set_yticks(range(len(isls)))
    ax.set_yticklabels(isls)
    ax.set_xlabel("batch size (concurrency)")
    ax.set_ylabel("input seq len (ISL)")
    ax.set_title(f"{role}  —  {title}")
    for i in range(len(isls)):
        for j in range(len(batches)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center",
                        color="white" if grid[i, j] < np.nanmax(grid) * 0.6 else "black",
                        fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(unit)
    fig.tight_layout()
    out_path = out_dir / f"heatmap_{role}_{metric}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[aggregate] {out_path}")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    arg = Path(sys.argv[1])
    csv_path = arg if arg.suffix == ".csv" else arg / "summary_nsys.csv"
    if not csv_path.is_file():
        print(f"error: summary CSV not found: {csv_path}", file=sys.stderr)
        return 1
    sweep_dir = csv_path.parent
    out_dir = sweep_dir / "grid_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(csv_path)
    agg = aggregate(rows)
    if not agg:
        print("error: no prefill/decode rows in summary CSV", file=sys.stderr)
        return 1

    panels = [
        ("hbm_total_pct", "HBM Bandwidth (R+W)", "% of peak"),
        ("hbm_total_gbps", "HBM Bandwidth (R+W)", "GB/s"),
        ("nvlink_total_pct", "NVLink Bandwidth (TX+RX user)", "% of peak"),
        ("sm_active_pct", "SM Active", "% of peak"),
        ("tensor_active_pct", "Tensor Active", "% of peak"),
    ]
    for role, cells in agg.items():
        write_tidy_csv(role, cells, out_dir)
        for metric, title, unit in panels:
            heatmap(role, cells, metric, title, unit, out_dir)

    print(f"[aggregate] done -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
