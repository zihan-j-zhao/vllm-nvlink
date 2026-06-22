#!/usr/bin/env python3
"""Classify kernel durations by overlap with peer-to-peer D2D copies."""

from __future__ import annotations

import argparse
import bisect
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


KERNEL_PATTERNS = {
    "fmha": "fmhaSm",
    "fused_moe": "fused_moe_kernel",
    "allreduce_comm": "allreduce_fusion_kernel",
    "AllGather": "ncclDevKernel_AllGather",
    "ReduceScatter": "ncclDevKernel_ReduceScatter",
}

P2P_COPY_KIND = 10


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


def p2p_intervals(conn: sqlite3.Connection, start: int, end: int) -> list[tuple[int, int]]:
    cur = conn.execute(
        """
        SELECT start, end
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        WHERE copyKind = ? AND start >= ? AND end <= ?
        ORDER BY start
        """,
        [P2P_COPY_KIND, start, end],
    )
    return [(int(a), int(b)) for a, b in cur]


def overlap_ns(a: int, b: int, intervals: list[tuple[int, int]], starts: list[int]) -> int:
    if not intervals:
        return 0
    idx = max(0, bisect.bisect_right(starts, a) - 2)
    total = 0
    while idx < len(intervals):
        s, e = intervals[idx]
        if s >= b:
            break
        if e > a:
            total += max(0, min(b, e) - max(a, s))
        idx += 1
    return total


def kernel_rows(
    conn: sqlite3.Connection,
    start: int,
    end: int,
    kernel: str,
    intervals: list[tuple[int, int]],
) -> list[dict[str, float | str]]:
    pattern = KERNEL_PATTERNS[kernel]
    starts = [s for s, _ in intervals]
    cur = conn.execute(
        """
        SELECT k.start, k.end
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.demangledName = s.id
        WHERE k.start >= ? AND k.end <= ? AND s.value LIKE ?
        ORDER BY k.start
        """,
        [start, end, f"%{pattern}%"],
    )
    rows = []
    for a, b in cur:
        a = int(a)
        b = int(b)
        dur = b - a
        ov = overlap_ns(a, b, intervals, starts)
        rows.append({
            "kernel": kernel,
            "duration_us": dur / 1000.0,
            "overlap_us": ov / 1000.0,
            "overlap_frac": (ov / dur) if dur else 0.0,
            "overlap": "overlap" if ov > 0 else "no_overlap",
        })
    return rows


def load_baseline(path: Path) -> list[dict[str, float | str]]:
    with sqlite3.connect(path) as conn:
        start, end = bench_window(conn)
        rows = []
        for kernel in KERNEL_PATTERNS:
            for row in kernel_rows(conn, start, end, kernel, []):
                row["group"] = "baseline"
                rows.append(row)
        return rows


def load_traffic(path: Path) -> list[dict[str, float | str]]:
    with sqlite3.connect(path) as conn:
        start, end = bench_window(conn)
        copies = p2p_intervals(conn, start, end)
        rows = []
        for kernel in KERNEL_PATTERNS:
            for row in kernel_rows(conn, start, end, kernel, copies):
                row["group"] = str(row["overlap"])
                rows.append(row)
        return rows


def write_summary(rows: list[dict[str, float | str]], out: Path) -> None:
    groups: dict[tuple[str, str], list[float]] = {}
    overlap_fracs: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        key = (str(row["kernel"]), str(row["group"]))
        groups.setdefault(key, []).append(float(row["duration_us"]))
        overlap_fracs.setdefault(key, []).append(float(row["overlap_frac"]))

    fields = [
        "kernel", "group", "count", "mean_us", "p50_us", "p90_us",
        "p99_us", "min_us", "max_us", "mean_overlap_frac",
    ]
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for key in sorted(groups):
            vals = groups[key]
            fracs = overlap_fracs[key]
            writer.writerow({
                "kernel": key[0],
                "group": key[1],
                "count": len(vals),
                "mean_us": statistics.fmean(vals),
                "p50_us": statistics.median(vals),
                "p90_us": percentile(vals, 90),
                "p99_us": percentile(vals, 99),
                "min_us": min(vals),
                "max_us": max(vals),
                "mean_overlap_frac": statistics.fmean(fracs),
            })


def plot(rows: list[dict[str, float | str]], out_dir: Path, tag: str) -> None:
    colors = {
        "baseline": "#444444",
        "no_overlap": "#2ca02c",
        "overlap": "#d62728",
    }
    for kernel in KERNEL_PATTERNS:
        fig, ax = plt.subplots(figsize=(8, 4.6))
        vals_by_group = {
            group: [
                float(row["duration_us"])
                for row in rows
                if row["kernel"] == kernel and row["group"] == group
            ]
            for group in ("baseline", "no_overlap", "overlap")
        }
        all_vals = [v for vals in vals_by_group.values() for v in vals]
        if not all_vals:
            continue
        xmax = percentile(all_vals, 99.5)
        for group, vals in vals_by_group.items():
            vals = [v for v in vals if v <= xmax]
            if vals:
                ax.hist(
                    vals, bins=50, density=True, alpha=0.42,
                    label=group, color=colors[group],
                )
        ax.set_xlabel(f"{kernel} duration [us]")
        ax.set_ylabel("density")
        ax.set_title(f"{tag}: {kernel} duration by P2P-overlap class")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        path = out_dir / f"{tag}_{kernel}_overlap_histogram.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[wrote] {path}")

    kernels_with_data = [
        kernel for kernel in KERNEL_PATTERNS
        if any(row["kernel"] == kernel for row in rows)
    ]
    fig, axes = plt.subplots(
        1, len(kernels_with_data), figsize=(4 * len(kernels_with_data), 4.8)
    )
    if len(kernels_with_data) == 1:
        axes = [axes]
    for ax, kernel in zip(axes, kernels_with_data):
        vals = []
        labels = []
        for group in ("baseline", "no_overlap", "overlap"):
            group_vals = [
                float(row["duration_us"])
                for row in rows
                if row["kernel"] == kernel and row["group"] == group
            ]
            if group_vals:
                vals.append(group_vals)
                labels.append(group)
        if not vals:
            ax.set_visible(False)
            continue
        ax.boxplot(vals, tick_labels=labels, showfliers=False)
        ax.set_title(kernel)
        ax.set_ylabel("duration [us]")
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelrotation=35)
    fig.suptitle(f"{tag}: kernel duration by P2P-overlap class")
    fig.tight_layout()
    path = out_dir / f"{tag}_overlap_boxplots.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-sqlite", type=Path, required=True)
    ap.add_argument("--traffic-sqlite", type=Path, required=True)
    ap.add_argument("--tag", default="overlap")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    rows = load_baseline(args.baseline_sqlite) + load_traffic(args.traffic_sqlite)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = args.out_dir / f"{args.tag}_overlap_summary.csv"
    write_summary(rows, summary)
    print(f"[wrote] {summary}")
    plot(rows, args.out_dir, args.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
