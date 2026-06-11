#!/usr/bin/env python3
"""Fast per-GPU resource-utilisation timeseries plots for one sweep cell.

Single binned SQL scan per cell (vs ~32 full-table scans in
postprocess_nsys.py, which is pathologically slow on multi-100MB sqlite).
For each profiled GPU, emits one figure with stacked subplots:
  1. Compute: SMs Active %, Tensor Active %
  2. HBM bandwidth: DRAM Read %, DRAM Write %  (right axis: GB/s)
  3. NVLink user data: TX (req+resp) %, RX (req+resp) %

Usage:
    python fast_timeseries.py <cell_dir> [--bin-ms 100] [--out <dir>]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PREFILL = {0, 1}
DECODE = {2, 3}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cell_dir", type=Path)
    ap.add_argument("--bin-ms", type=float, default=100.0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--start-s", type=float, default=None,
                    help="Crop the plot to start at this many seconds from "
                         "capture start. Overrides auto-crop.")
    ap.add_argument("--end-s", type=float, default=None,
                    help="Crop the plot to end at this many seconds.")
    ap.add_argument("--no-auto-crop", action="store_true",
                    help="Disable automatic trimming of leading idle (start "
                         "5s before first GPU activity).")
    args = ap.parse_args()

    cd = args.cell_dir
    meta = json.loads((cd / "cell.json").read_text())
    tag = meta.get("cell_tag", cd.name)
    sq = cd / "nsys_report.sqlite"
    out_dir = args.out or (cd / "ts_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(str(sq))
    cur = con.cursor()
    names = {}
    cur.execute("SELECT DISTINCT metricId, metricName FROM TARGET_INFO_GPU_METRICS")
    for mid, mn in cur.fetchall():
        names[int(mid)] = str(mn)
    cur.execute("SELECT MIN(typeId), MIN(timestamp) FROM GPU_METRICS")
    base, t0 = cur.fetchone()
    base = int(base); t0 = int(t0)
    peak_hbm_gbps = 0.0
    try:
        cur.execute("SELECT memoryBandwidth FROM TARGET_INFO_GPU LIMIT 1")
        r = cur.fetchone()
        if r and r[0]:
            peak_hbm_gbps = float(r[0]) / 1e9
    except sqlite3.OperationalError:
        pass

    bin_ns = int(args.bin_ms * 1e6)
    cur.execute(
        "SELECT CAST((timestamp-?)/? AS INT) b, typeId, metricId, AVG(value) "
        "FROM GPU_METRICS GROUP BY b, typeId, metricId ORDER BY b",
        (t0, bin_ns),
    )
    # data[gpu][metricName] = {bin: value}
    data: dict[int, dict[str, dict[int, float]]] = {}
    maxbin = 0
    for b, tid, mid, v in cur.fetchall():
        gid = int(tid) - base
        data.setdefault(gid, {}).setdefault(names.get(int(mid), str(mid)), {})[int(b)] = float(v)
        maxbin = max(maxbin, int(b))
    con.close()

    x = np.arange(maxbin + 1) * args.bin_ms / 1000.0

    def series(gid, substrs):
        """Sum of bins for metrics whose name contains all substrs."""
        acc = np.full(maxbin + 1, np.nan)
        found = False
        for mn, bins in data.get(gid, {}).items():
            if all(s in mn for s in substrs):
                arr = np.zeros(maxbin + 1)
                for bi, val in bins.items():
                    arr[bi] = val
                acc = arr if not found else acc + arr
                found = True
        return acc if found else None

    # --- Determine crop window (trim leading idle) ----------------------
    # System activity = max SMs Active across all GPUs per bin.
    sys_sm = np.zeros(maxbin + 1)
    for gid in data:
        s = series(gid, ["SMs Active"])
        if s is not None:
            sys_sm = np.maximum(sys_sm, np.nan_to_num(s))
    if args.start_s is not None:
        start_s = args.start_s
    elif args.no_auto_crop:
        start_s = 0.0
    else:
        active = np.where(sys_sm > 2.0)[0]
        start_s = max(0.0, active[0] * args.bin_ms / 1000.0 - 5.0) if len(active) else 0.0
    end_s = args.end_s if args.end_s is not None else x[-1] if len(x) else 0.0
    mask = (x >= start_s) & (x <= end_s)
    xm = x[mask]

    def crop(arr):
        return arr[mask] if arr is not None else None

    for gid in sorted(data):
        role = "prefill" if gid in PREFILL else "decode" if gid in DECODE else "gpu"
        fig, axs = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
        # 1. compute
        sm = crop(series(gid, ["SMs Active"])); te = crop(series(gid, ["Tensor Active"]))
        if sm is not None: axs[0].plot(xm, sm, label="SMs Active %", color="#9467bd", lw=0.9)
        if te is not None: axs[0].plot(xm, te, label="Tensor Active %", color="#e377c2", lw=0.9)
        axs[0].set_ylabel("Compute %"); axs[0].set_ylim(0, 100); axs[0].legend(loc="upper right", fontsize=8)
        axs[0].set_title(f"GPU{gid} ({role})  —  {tag}  [t={start_s:.0f}–{end_s:.0f}s]", fontsize=11)
        # 2. HBM
        hr = crop(series(gid, ["DRAM Read"])); hw = crop(series(gid, ["DRAM Write"]))
        if hr is not None: axs[1].plot(xm, hr, label="HBM Read %", color="#1f77b4", lw=0.9)
        if hw is not None: axs[1].plot(xm, hw, label="HBM Write %", color="#ff7f0e", lw=0.9)
        axs[1].set_ylabel("HBM %"); axs[1].set_ylim(0, 100); axs[1].legend(loc="upper right", fontsize=8)
        if peak_hbm_gbps:
            ax1b = axs[1].twinx(); ax1b.set_ylim(0, peak_hbm_gbps); ax1b.set_ylabel("GB/s")
        # 3. NVLink
        tx = crop(series(gid, ["NVLink TX"])); rx = crop(series(gid, ["NVLink RX"]))
        if tx is not None: axs[2].plot(xm, tx, label="NVLink TX %", color="#2ca02c", lw=0.9)
        if rx is not None: axs[2].plot(xm, rx, label="NVLink RX %", color="#d62728", lw=0.9)
        axs[2].set_ylabel("NVLink %"); axs[2].legend(loc="upper right", fontsize=8)
        axs[2].set_xlabel("time (s)")
        for a in axs: a.grid(alpha=0.3)
        fig.tight_layout()
        p = out_dir / f"ts_gpu{gid}_{role}_{tag}.png"
        fig.savefig(str(p), dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"[ts] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
