#!/usr/bin/env python3
"""Plot per-kernel HW utilization vs (batch, prefill, bg).

Reads <sweep_dir>/nsys/*.kernel_hw.csv files and produces:

  - <sweep_dir>/figs_kernel/per_kernel_hw_off.png
      For the off-baseline only: 4-panel grid, x=prefill, line per batch,
      one row of subplots per (HBM Read, HBM Write, SMs Active),
      one column per kernel.

  - <sweep_dir>/figs_kernel/per_kernel_hw_b1024_p2048.png
      Most-stressed cell only: bar chart of 5 bg profiles x 3 kernels
      x {HBM Read, HBM Write, NVLink TX/RX, SMs Active}.

Only worth running on a 10kHz+ sweep where per-kernel HW has signal.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("matplotlib/numpy missing"); sys.exit(1)


KERNELS = ["fmha", "fused_moe", "allreduce"]
METRIC_LABEL = {
    "DRAM Read Bandwidth [Throughput %]": "HBM Read %",
    "DRAM Write Bandwidth [Throughput %]": "HBM Write %",
    "SMs Active [Throughput %]": "SMs Active %",
    "Tensor Active [Throughput %]": "Tensor Active %",
    "NVLink TX Requests User Data [Throughput %]": "NVLink TX %",
    "NVLink RX Requests User Data [Throughput %]": "NVLink RX %",
}
BG_ORDER = ["off", "ingress_med", "ingress_max", "egress_med", "egress_max"]
BG_COLORS = {
    "off":         "#444444",
    "ingress_med": "#1f77b4",
    "ingress_max": "#08306b",
    "egress_med":  "#ff7f0e",
    "egress_max":  "#7f2704",
}

TAG_RE = re.compile(r"b(\d+)_p(\d+)_([a-z_]+)")


def load_kernel_hw(nsys_dir: Path) -> list[dict]:
    """Return rows: {tag, batch, prefill, bg, kernel, gpu, metric, p50, mean, ...}."""
    out: list[dict] = []
    for f in sorted(glob.glob(str(nsys_dir / "*.kernel_hw.csv"))):
        tag = os.path.basename(f).replace(".kernel_hw.csv", "")
        m = TAG_RE.match(tag)
        if not m:
            continue
        b, p, bg = int(m.group(1)), int(m.group(2)), m.group(3)
        with open(f) as fh:
            for r in csv.DictReader(fh):
                r["tag"] = tag
                r["batch"] = b
                r["prefill"] = p
                r["bg"] = bg
                for k in ("p50", "mean", "min", "max"):
                    r[k] = float(r[k])
                r["gpu"] = int(r["gpu"])
                out.append(r)
    return out


# ---------------------------------------------------------------------------
# Per-kernel HW vs prefill, off baseline only
# ---------------------------------------------------------------------------
def plot_per_kernel_off(rows, out_path: Path):
    off = [r for r in rows if r["bg"] == "off"]
    if not off:
        print("[skip] no off rows"); return

    metrics = [
        "DRAM Read Bandwidth [Throughput %]",
        "DRAM Write Bandwidth [Throughput %]",
        "SMs Active [Throughput %]",
    ]
    fig, axes = plt.subplots(
        len(metrics), len(KERNELS), figsize=(13, 9), sharex=True
    )
    batches = sorted({r["batch"] for r in off})
    prefills = sorted({r["prefill"] for r in off})
    cmap = plt.get_cmap("viridis")
    batch_colors = {b: cmap(i / max(1, len(batches) - 1))
                    for i, b in enumerate(batches)}

    for col, kernel in enumerate(KERNELS):
        for row, metric in enumerate(metrics):
            ax = axes[row, col]
            # For each batch line, aggregate across 4 GPUs (median of p50).
            sub = [r for r in off
                   if r["kernel_label"] == kernel and r["metric"] == metric]
            for b in batches:
                pts = defaultdict(list)
                for r in sub:
                    if r["batch"] == b:
                        pts[r["prefill"]].append(r["p50"])
                if not pts:
                    continue
                xs = sorted(pts)
                ys = [sorted(pts[x])[len(pts[x]) // 2] for x in xs]
                ax.plot(xs, ys, marker="o", label=f"B={b}",
                        color=batch_colors[b])
            ax.set_xscale("log", base=2)
            if row == 0:
                ax.set_title(kernel)
            if col == 0:
                ax.set_ylabel(METRIC_LABEL[metric])
            if row == len(metrics) - 1:
                ax.set_xlabel("prefill")
            ax.grid(True, alpha=0.3)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle("Per-kernel HW pressure (off baseline, TP=4, 10 kHz)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out_path}")


# ---------------------------------------------------------------------------
# Per-kernel HW for one cell, all bg profiles
# ---------------------------------------------------------------------------
def plot_per_kernel_one_cell(rows, batch: int, prefill: int, out_path: Path):
    sub = [r for r in rows
           if r["batch"] == batch and r["prefill"] == prefill]
    if not sub:
        return

    metrics = [
        "DRAM Read Bandwidth [Throughput %]",
        "DRAM Write Bandwidth [Throughput %]",
        "SMs Active [Throughput %]",
        "NVLink TX Requests User Data [Throughput %]",
        "NVLink RX Requests User Data [Throughput %]",
    ]
    bgs = [b for b in BG_ORDER
           if b in {r["bg"] for r in sub}]
    n_groups = len(KERNELS)
    n_bgs = len(bgs)
    width = 0.8 / n_bgs

    fig, axes = plt.subplots(len(metrics), 1, figsize=(11, 2.0 * len(metrics)),
                             sharex=True)
    for ax, metric in zip(axes, metrics):
        for i, bg in enumerate(bgs):
            vals = []
            for kernel in KERNELS:
                kr = [r["p50"] for r in sub
                      if r["kernel_label"] == kernel
                      and r["metric"] == metric and r["bg"] == bg]
                # median across 4 GPUs
                vals.append(sorted(kr)[len(kr) // 2] if kr else 0)
            offsets = np.arange(n_groups) - 0.4 + (i + 0.5) * width
            ax.bar(offsets, vals, width=width, label=bg,
                   color=BG_COLORS.get(bg))
        ax.set_xticks(range(n_groups), KERNELS)
        ax.set_ylabel(METRIC_LABEL[metric])
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8, ncol=n_bgs)
    fig.suptitle(f"Per-kernel HW pressure under bg traffic — B={batch}, p={prefill}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out_path}")


# ---------------------------------------------------------------------------
# Pie chart: per-iter time breakdown (fmha / fused_moe / allreduce / others)
# ---------------------------------------------------------------------------
def _read_p50_ms(sweep_csv: Path, tag: str) -> float | None:
    if not sweep_csv.exists():
        return None
    with open(sweep_csv) as fh:
        for r in csv.DictReader(fh):
            if r["tag"] == tag:
                return float(r["p50_ms"])
    return None


def _read_kernel_totals_us(kernels_csv: Path) -> dict[str, tuple[float, int]]:
    """Return {kernel_label: (total_us, count)} for the three kernels we care about."""
    out: dict[str, tuple[float, int]] = {}
    with open(kernels_csv) as fh:
        for r in csv.DictReader(fh):
            label = r["kernel_label"]
            if label in KERNELS:
                out[label] = (float(r["total_us"]), int(r["count"]))
    return out


def plot_forward_pie(sweep_dir: Path, batch: int, prefill: int, bg: str,
                     n_iters: int, n_gpus: int, out_path: Path) -> None:
    tag = f"b{batch}_p{prefill}_{bg}"
    sweep_csv = sweep_dir / "sweep_results.csv"
    kernels_csv = sweep_dir / "kernel_csv" / f"{tag}.kernels.csv"
    if not kernels_csv.exists():
        print(f"[skip] missing {kernels_csv}"); return
    p50_ms = _read_p50_ms(sweep_csv, tag)
    if p50_ms is None:
        print(f"[skip] no p50_ms for {tag} in {sweep_csv}"); return

    totals_us = _read_kernel_totals_us(kernels_csv)
    # total_us is aggregated across all decoder GPUs and all iters in the
    # nsys capture; divide to get per-GPU per-iter time in ms.
    denom = n_iters * n_gpus
    per_iter_ms = {k: totals_us.get(k, (0.0, 0))[0] / denom / 1000.0 for k in KERNELS}
    accounted_ms = sum(per_iter_ms.values())
    others_ms = max(0.0, p50_ms - accounted_ms)

    labels = ["fmha", "fused_moe", "allreduce", "others"]
    values_ms = [per_iter_ms["fmha"], per_iter_ms["fused_moe"],
                 per_iter_ms["allreduce"], others_ms]
    total_ms = sum(values_ms)
    pcts = [100 * v / total_ms for v in values_ms]
    colors = ["#d62728", "#2ca02c", "#1f77b4", "#888888"]

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    wedges, _ = ax.pie(
        values_ms, labels=None, startangle=90, counterclock=False,
        colors=colors,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=1.5),
    )
    legend_labels = [
        f"{lbl}: {ms:.2f} ms ({pct:.1f}%)"
        for lbl, ms, pct in zip(labels, values_ms, pcts)
    ]
    ax.legend(wedges, legend_labels, loc="center left",
              bbox_to_anchor=(1.02, 0.5), fontsize=10, frameon=False)
    ax.text(0, 0, f"{p50_ms:.2f} ms\nper forward",
            ha="center", va="center", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Forward-pass time breakdown — B={batch}, p={prefill}, bg={bg}\n"
        f"({n_gpus} decoders, {n_iters} iters; per-GPU per-iter time)",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out_path}  "
          f"(fmha={per_iter_ms['fmha']:.2f} fused_moe={per_iter_ms['fused_moe']:.2f} "
          f"allreduce={per_iter_ms['allreduce']:.2f} others={others_ms:.2f} ms)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--pie-batch", type=int, default=1024,
                    help="batch size for the pie chart cell (default 1024)")
    ap.add_argument("--pie-prefill", type=int, default=2048,
                    help="prefill length for the pie chart cell (default 2048)")
    ap.add_argument("--pie-bg", default="off",
                    help="bg profile for the pie chart (default off)")
    ap.add_argument("--n-iters", type=int, default=20,
                    help="iters per cell in the sweep (default 20)")
    ap.add_argument("--n-gpus", type=int, default=4,
                    help="number of decoder GPUs in the capture (default 4)")
    args = ap.parse_args()
    nsys_dir = args.sweep_dir / "nsys"
    if not nsys_dir.exists():
        print(f"no nsys/ in {args.sweep_dir}"); return 1
    out_dir = args.out_dir or args.sweep_dir / "figs_kernel"
    rows = load_kernel_hw(nsys_dir)
    print(f"[loaded] {len(rows)} per-kernel HW rows")
    plot_per_kernel_off(rows, out_dir / "per_kernel_hw_off.png")
    # The most-stressed cell shows contention impact most clearly.
    plot_per_kernel_one_cell(rows, 1024, 2048, out_dir / "per_kernel_hw_b1024_p2048.png")
    plot_per_kernel_one_cell(rows, 128, 128, out_dir / "per_kernel_hw_b128_p128.png")
    pie_name = (f"forward_pie_b{args.pie_batch}_p{args.pie_prefill}"
                f"_{args.pie_bg}.png")
    plot_forward_pie(args.sweep_dir, args.pie_batch, args.pie_prefill,
                     args.pie_bg, args.n_iters, args.n_gpus,
                     out_dir / pie_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
