#!/usr/bin/env python3
"""Line plot for DP=EP kernel/e2e latency across background-load cases."""

from __future__ import annotations

import argparse
import csv
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_pie_dpep import kernel_totals_ns, p50_ms


KERNEL_PATTERNS = {
    "attention": "fmhaSm",
    "moe": "fused_moe_kernel",
    "ag": "ncclDevKernel_AllGather_RING_LL",
    "rs": "ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL",
}


@dataclass(frozen=True)
class Case:
    label: str
    sweep_dir: Path
    tag: str


def parse_case(raw: str) -> Case:
    parts = raw.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "case must be LABEL:SWEEP_DIR:TAG"
        )
    label, sweep_dir, tag = parts
    if not label or not sweep_dir or not tag:
        raise argparse.ArgumentTypeError(
            "case must be LABEL:SWEEP_DIR:TAG with no empty fields"
        )
    return Case(label=label, sweep_dir=Path(sweep_dir), tag=tag)


def _pct(values: list[float], p: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, max(0, round((len(values) - 1) * p / 100.0)))]


def e2e_latency_ms(sweep_csv: Path, tag: str, stat: str) -> float:
    col = "p50_ms" if stat == "mean" else f"{stat}_ms"
    with sweep_csv.open() as fh:
        for row in csv.DictReader(fh):
            if row["tag"] == tag:
                return float(row[col])
    raise RuntimeError(f"no row {tag} in {sweep_csv}")


def per_iter_kernel_slice_stats(
    sqlite_path: Path, *, stat: str, n_gpus: int,
) -> dict[str, float]:
    with sqlite3.connect(sqlite_path) as conn:
        ranges = list(conn.execute(
            "SELECT text, start, end FROM NVTX_EVENTS "
            "WHERE text LIKE 'iter_%' AND end IS NOT NULL ORDER BY start"
        ))
        if not ranges:
            raise RuntimeError(f"no iter_ NVTX ranges in {sqlite_path}")

        by_label: dict[str, list[float]] = {label: [] for label in KERNEL_PATTERNS}
        for _, start_ns, end_ns in ranges:
            for label, substr in KERNEL_PATTERNS.items():
                cur = conn.execute(
                    """
                    SELECT SUM(k.end - k.start) / 1.0e6
                    FROM CUPTI_ACTIVITY_KIND_KERNEL k
                    JOIN StringIds s ON k.demangledName = s.id
                    WHERE k.start >= ? AND k.end <= ? AND s.value LIKE ?
                    """,
                    [start_ns, end_ns, f"%{substr}%"],
                )
                value_ms = cur.fetchone()[0]
                # Each rank's iter NVTX range overlaps all ranks' GPU kernels
                # in the exported report, so normalize the accumulated slice
                # back to per-GPU per-forward time.
                by_label[label].append(float(value_ms or 0.0) / n_gpus)

    pct = {"p50": 50.0, "p90": 90.0, "p99": 99.0}[stat]
    return {label: _pct(values, pct) for label, values in by_label.items()}


def case_values(
    case: Case, *, n_iters: int, n_gpus: int, stat: str,
) -> dict[str, float]:
    nsys_rep = case.sweep_dir / "nsys" / f"{case.tag}.nsys-rep"
    sqlite_path = case.sweep_dir / "nsys" / f"{case.tag}.sqlite"
    sweep_csv = case.sweep_dir / "sweep_results.csv"
    if not nsys_rep.exists():
        raise FileNotFoundError(nsys_rep)
    if not sqlite_path.exists():
        raise FileNotFoundError(sqlite_path)
    if not sweep_csv.exists():
        raise FileNotFoundError(sweep_csv)

    if stat == "mean":
        totals_ns = kernel_totals_ns(nsys_rep)
        denom = n_iters * n_gpus * 1.0e6
        kernel_values = {
            "attention": totals_ns["fmha"] / denom,
            "moe": totals_ns["fused_moe"] / denom,
            "ag": totals_ns["AllGather"] / denom,
            "rs": totals_ns["ReduceScatter"] / denom,
        }
    else:
        kernel_values = per_iter_kernel_slice_stats(
            sqlite_path, stat=stat, n_gpus=n_gpus)

    return {
        "e2e": e2e_latency_ms(sweep_csv, case.tag, stat),
        **kernel_values,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--case", dest="cases", type=parse_case, action="append", required=True,
        help="Case spec LABEL:SWEEP_DIR:TAG. Repeat in x-axis order.",
    )
    ap.add_argument("--n-iters", type=int, default=20)
    ap.add_argument("--n-gpus", type=int, default=4)
    ap.add_argument(
        "--stat", choices=["mean", "p50", "p90", "p99"], default="mean",
        help="Kernel statistic to plot. mean matches the pie-chart average; "
             "p50/p90/p99 use per-iter accumulated kernel slices.",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cases: list[Case] = args.cases
    labels = [case.label for case in cases]
    values = [case_values(case, n_iters=args.n_iters, n_gpus=args.n_gpus,
                          stat=args.stat)
              for case in cases]

    series = [
        ("e2e latency", "e2e", "#111111"),
        ("attention", "attention", "#d62728"),
        ("moe", "moe", "#2ca02c"),
        ("AllGather", "ag", "#1f77b4"),
        ("ReduceScatter", "rs", "#9467bd"),
    ]

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    xs = list(range(len(labels)))
    for name, key, color in series:
        ys = [v[key] for v in values]
        ax.plot(xs, ys, marker="o", linewidth=2.0, label=name, color=color)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.2f}", xy=(x, y), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=8)

    ax.set_xticks(xs, labels)
    ax.set_ylabel("per-GPU per-forward time [ms]")
    ax.set_xlabel("background ingress CE load")
    stat_title = "mean" if args.stat == "mean" else args.stat.upper()
    ax.set_title(
        f"DP=EP=4, B=256, prefill=2048: {stat_title} latency and kernel slices"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[wrote] {args.out}")
    for label, row in zip(labels, values):
        print(
            f"{label}: e2e={row['e2e']:.2f} attention={row['attention']:.2f} "
            f"moe={row['moe']:.2f} ag={row['ag']:.2f} rs={row['rs']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
