"""Pie chart for DP=EP=4 case where TP allreduce is replaced by EP all2all.

The standard plot_per_kernel.plot_forward_pie only knows about
(fmha, fused_moe, allreduce). Under DP=EP, the collective is split into
AllGather + ReduceScatter (the AgRsAll2AllManager backend in vLLM). Run
nsys cuda_gpu_kern_sum to grab those totals directly and plot 5 slices.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


KERNEL_PATTERNS = {
    "fmha":          "fmhaSm100fKernel",
    "fused_moe":     "fused_moe_kernel",
    "AllGather":     "ncclDevKernel_AllGather_RING_LL",
    "ReduceScatter": "ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL",
}


def kernel_totals_ns(nsys_rep: Path) -> dict[str, float]:
    out = subprocess.check_output(
        ["nsys", "stats", "--report", "cuda_gpu_kern_sum",
         "--format", "csv", "--force-export=true", str(nsys_rep)],
        text=True, stderr=subprocess.DEVNULL,
    )
    totals: dict[str, float] = {k: 0.0 for k in KERNEL_PATTERNS}
    for line in out.splitlines():
        for label, needle in KERNEL_PATTERNS.items():
            if needle in line:
                # CSV: Time%,Total(ns),Instances,Avg,Median,Min,Max,StdDev,Name
                parts = line.split(",")
                totals[label] += float(parts[1])  # nanoseconds
                break
    return totals


def p50_ms(sweep_csv: Path, tag: str) -> float:
    with open(sweep_csv) as fh:
        for r in csv.DictReader(fh):
            if r["tag"] == tag:
                return float(r["p50_ms"])
    raise SystemExit(f"no row {tag} in {sweep_csv}")


def tags_from_csv(sweep_csv: Path) -> list[str]:
    with open(sweep_csv) as fh:
        return [r["tag"] for r in csv.DictReader(fh)]


def plot_pie(sweep_dir: Path, tag: str, n_iters: int, n_gpus: int,
             out_path: Path) -> None:
    nsys_rep = sweep_dir / "nsys" / f"{tag}.nsys-rep"
    sweep_csv = sweep_dir / "sweep_results.csv"
    totals_ns = kernel_totals_ns(nsys_rep)
    denom = n_iters * n_gpus
    # ns / (iters*gpus) / 1e6 -> ms per GPU per iter
    ms = {k: v / denom / 1.0e6 for k, v in totals_ns.items()}
    p50 = p50_ms(sweep_csv, tag)
    accounted = sum(ms.values())
    others = max(0.0, p50 - accounted)

    labels = ["fmha", "fused_moe", "AllGather", "ReduceScatter", "others"]
    values = [ms["fmha"], ms["fused_moe"], ms["AllGather"],
              ms["ReduceScatter"], others]
    pcts = [100 * v / p50 for v in values]
    colors = ["#d62728", "#2ca02c", "#1f77b4", "#9467bd", "#888888"]

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    wedges, _ = ax.pie(
        values, labels=None, startangle=90, counterclock=False,
        colors=colors,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=1.5),
    )
    legend = [f"{lbl}: {v:.2f} ms ({p:.1f}%)"
              for lbl, v, p in zip(labels, values, pcts)]
    ax.legend(wedges, legend, loc="center left",
              bbox_to_anchor=(1.02, 0.5), fontsize=10, frameon=False)
    ax.text(0, 0, f"{p50:.2f} ms\nper forward",
            ha="center", va="center", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Forward-pass breakdown (DP=EP=4) — {tag}\n"
        f"({n_gpus} GPUs, {n_iters} iters; per-GPU per-iter time)",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out_path}  "
          + " ".join(f"{l}={v:.2f}" for l, v in zip(labels, values)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir", type=Path)
    ap.add_argument("--n-iters", type=int, default=20)
    ap.add_argument("--n-gpus", type=int, default=4)
    ap.add_argument(
        "--tags", type=str, default=None,
        help="Comma-separated cell tags to plot. Default: all rows in sweep_results.csv.",
    )
    args = ap.parse_args()
    out_dir = args.sweep_dir / "figs_kernel"
    sweep_csv = args.sweep_dir / "sweep_results.csv"
    tags = (
        [t.strip() for t in args.tags.split(",") if t.strip()]
        if args.tags else tags_from_csv(sweep_csv)
    )
    for tag in tags:
        plot_pie(args.sweep_dir, tag, args.n_iters, args.n_gpus,
                 out_dir / f"forward_pie_{tag}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
