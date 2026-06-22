#!/usr/bin/env python3
"""Plot communication kernel duration distributions from Nsight SQLite files."""

from __future__ import annotations

import argparse
import csv
import sqlite3
import statistics
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib missing", file=sys.stderr)
    sys.exit(1)


COMM_PATTERNS = {
    "AllGather": "ncclDevKernel_AllGather",
    "ReduceScatter": "ncclDevKernel_ReduceScatter",
}


def percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round((len(values) - 1) * pct / 100.0)))
    return values[idx]


def bench_window(conn: sqlite3.Connection) -> tuple[int, int]:
    row = conn.execute(
        "SELECT MIN(start), MAX(end) FROM NVTX_EVENTS "
        "WHERE text LIKE 'realistic_bench%' AND end IS NOT NULL"
    ).fetchone()
    if row is None or row[0] is None:
        raise RuntimeError("no realistic_bench NVTX window found")
    return int(row[0]), int(row[1])


def parse_tag(path: Path) -> tuple[int, str]:
    # b16000_s16_ingress.sqlite -> (16000, ingress)
    parts = path.stem.split("_")
    return int(parts[0].removeprefix("b")), parts[-1]


def durations_us(sqlite_path: Path, pattern: str) -> list[float]:
    with sqlite3.connect(sqlite_path) as conn:
        start, end = bench_window(conn)
        cur = conn.execute(
            """
            SELECT (k.end - k.start) AS dur_ns
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.demangledName = s.id
            WHERE k.start >= ? AND k.end <= ? AND s.value LIKE ?
            """,
            [start, end, f"%{pattern}%"],
        )
        return [int(row[0]) / 1000.0 for row in cur]


def load(root: Path, seq_len: int) -> dict[tuple[int, str, str], list[float]]:
    data: dict[tuple[int, str, str], list[float]] = {}
    for sqlite_path in sorted((root / "nsys").glob(f"b*_s{seq_len}_*.sqlite")):
        batch, mode = parse_tag(sqlite_path)
        for label, pattern in COMM_PATTERNS.items():
            vals = durations_us(sqlite_path, pattern)
            if vals:
                data[(batch, mode, label)] = vals
    return data


def write_summary(data: dict[tuple[int, str, str], list[float]], out: Path) -> None:
    fields = [
        "batch_size", "mode", "kernel", "count", "total_us", "mean_us",
        "p50_us", "p90_us", "p99_us", "min_us", "max_us",
    ]
    rows = []
    for (batch, mode, kernel), vals in sorted(data.items()):
        rows.append({
            "batch_size": batch,
            "mode": mode,
            "kernel": kernel,
            "count": len(vals),
            "total_us": sum(vals),
            "mean_us": statistics.fmean(vals),
            "p50_us": statistics.median(vals),
            "p90_us": percentile(vals, 90),
            "p99_us": percentile(vals, 99),
            "min_us": min(vals),
            "max_us": max(vals),
        })
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_histograms(
    data: dict[tuple[int, str, str], list[float]],
    out_dir: Path,
) -> None:
    batches = sorted({key[0] for key in data})
    modes = sorted({key[1] for key in data})
    colors = {"baseline": "#444444", "ingress": "#1f77b4", "egress": "#d62728", "both": "#9467bd"}
    for kernel in COMM_PATTERNS:
        fig, axes = plt.subplots(
            len(batches), 1, figsize=(9, 2.4 * len(batches)), sharex=False
        )
        if len(batches) == 1:
            axes = [axes]
        for ax, batch in zip(axes, batches):
            for mode in modes:
                vals = data.get((batch, mode, kernel), [])
                if vals:
                    ax.hist(
                        vals, bins=45, density=True, alpha=0.38,
                        label=mode, color=colors.get(mode),
                    )
            ax.set_ylabel(f"B={batch}\ndensity")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
        axes[-1].set_xlabel(f"{kernel} duration [us]")
        fig.suptitle(f"{kernel} communication kernel duration distribution")
        fig.tight_layout()
        path = out_dir / f"{kernel.lower()}_comm_duration_histograms.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[wrote] {path}")


def plot_box(data: dict[tuple[int, str, str], list[float]], out_dir: Path) -> None:
    batches = sorted({key[0] for key in data})
    modes = sorted({key[1] for key in data})
    fig, axes = plt.subplots(1, len(COMM_PATTERNS), figsize=(13, 5))
    if len(COMM_PATTERNS) == 1:
        axes = [axes]
    for ax, kernel in zip(axes, COMM_PATTERNS):
        vals = []
        labels = []
        for batch in batches:
            for mode in modes:
                d = data.get((batch, mode, kernel), [])
                if d:
                    vals.append(d)
                    labels.append(f"B{batch}\n{mode}")
        ax.boxplot(vals, tick_labels=labels, showfliers=False)
        ax.set_title(kernel)
        ax.set_ylabel("duration [us]")
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelrotation=45)
    fig.suptitle("Communication kernel duration distributions (outliers hidden)")
    fig.tight_layout()
    path = out_dir / "comm_kernel_duration_boxplots.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    data = load(args.root, args.seq_len)
    if not data:
        print(f"no communication kernels found under {args.root}/nsys", file=sys.stderr)
        return 1
    out_dir = args.out_dir or args.root / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / "comm_kernel_distribution_summary.csv"
    write_summary(data, summary)
    print(f"[wrote] {summary}")
    plot_histograms(data, out_dir)
    plot_box(data, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

