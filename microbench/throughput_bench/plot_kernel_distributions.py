#!/usr/bin/env python3
"""Plot per-kernel duration distributions from extracted nsys SQLite files."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
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
}


def bench_window(conn: sqlite3.Connection) -> tuple[int, int]:
    row = conn.execute(
        "SELECT MIN(start), MAX(end) FROM NVTX_EVENTS "
        "WHERE text LIKE 'realistic_bench%' AND end IS NOT NULL"
    ).fetchone()
    if row is None or row[0] is None:
        raise RuntimeError("no realistic_bench NVTX window found")
    return int(row[0]), int(row[1])


def durations_us(sqlite_path: Path, kernel: str) -> list[float]:
    substr = KERNEL_PATTERNS[kernel]
    with sqlite3.connect(sqlite_path) as conn:
        start, end = bench_window(conn)
        cur = conn.execute(
            """
            SELECT (k.end - k.start) AS dur_ns
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.demangledName = s.id
            WHERE k.start >= ? AND k.end <= ? AND s.value LIKE ?
            """,
            [start, end, f"%{substr}%"],
        )
        return [int(row[0]) / 1000.0 for row in cur]


def parse_batch(path: Path) -> int:
    # b600_s2048_ingress.sqlite -> 600
    return int(path.name.split("_", 1)[0].removeprefix("b"))


def load(root: Path) -> dict[tuple[int, str, str], list[float]]:
    out: dict[tuple[int, str, str], list[float]] = {}
    for sqlite_path in sorted((root / "nsys").glob("b*_s2048_*.sqlite")):
        parts = sqlite_path.stem.split("_")
        if len(parts) < 3:
            continue
        batch = parse_batch(sqlite_path)
        mode = parts[-1]
        if mode not in ("baseline", "ingress"):
            continue
        for kernel in KERNEL_PATTERNS:
            out[(batch, mode, kernel)] = durations_us(sqlite_path, kernel)
    return out


def plot_histograms(
    data: dict[tuple[int, str, str], list[float]],
    batches: list[int],
    out_dir: Path,
) -> None:
    for kernel in KERNEL_PATTERNS:
        fig, axes = plt.subplots(
            len(batches), 1, figsize=(8, 2.2 * len(batches)), sharex=False
        )
        if len(batches) == 1:
            axes = [axes]
        for ax, batch in zip(axes, batches):
            base = data.get((batch, "baseline", kernel), [])
            ing = data.get((batch, "ingress", kernel), [])
            if base:
                ax.hist(base, bins=40, alpha=0.55, density=True, label="baseline")
            if ing:
                ax.hist(ing, bins=40, alpha=0.55, density=True, label="ingress")
            ax.set_ylabel(f"B={batch}\ndensity")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=8)
        axes[-1].set_xlabel(f"{kernel} kernel duration [us]")
        fig.suptitle(f"{kernel} duration distribution: baseline vs ingress")
        fig.tight_layout()
        path = out_dir / f"{kernel}_duration_histograms.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[wrote] {path}")


def plot_box(
    data: dict[tuple[int, str, str], list[float]],
    batches: list[int],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, len(KERNEL_PATTERNS), figsize=(13, 5))
    if len(KERNEL_PATTERNS) == 1:
        axes = [axes]
    for ax, kernel in zip(axes, KERNEL_PATTERNS):
        vals = []
        labels = []
        for batch in batches:
            for mode in ("baseline", "ingress"):
                d = data.get((batch, mode, kernel), [])
                if d:
                    vals.append(d)
                    labels.append(f"B{batch}\n{mode}")
        ax.boxplot(vals, tick_labels=labels, showfliers=False)
        ax.set_title(kernel)
        ax.set_ylabel("duration [us]")
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelrotation=45)
    fig.suptitle("Kernel duration distributions (outliers hidden)")
    fig.tight_layout()
    path = out_dir / "kernel_duration_boxplots.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    data = load(args.root)
    if not data:
        print(f"no kernel durations found under {args.root}/nsys", file=sys.stderr)
        return 1
    batches = sorted({k[0] for k in data})
    out_dir = args.out_dir or args.root / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_histograms(data, batches, out_dir)
    plot_box(data, batches, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
