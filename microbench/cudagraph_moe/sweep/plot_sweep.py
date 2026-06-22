#!/usr/bin/env python3
"""Plot HBM/SM/NVLink/Tensor pressure vs. batch size & prefill length.

Reads sweep_results.csv produced by sweep.py and emits two figures:

  - <sweep_dir>/figs/by_batch.png    one panel per metric, x=batch, line=prefill
  - <sweep_dir>/figs/by_prefill.png  one panel per metric, x=prefill, line=batch

Each panel is a 2x2 grid:
  (DRAM Read | SMs Active)
  (NVLink TX | p50 latency)

A small bonus figure prints kernel-duration p50s alongside.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not installed in current env; pip install matplotlib")
    sys.exit(1)


METRICS = [
    ("DRAM Read Bandwidth [Throughput %]_med", "HBM Read [%]"),
    ("SMs Active [Throughput %]_med",          "SMs Active [%]"),
    ("NVLink TX Requests User Data [Throughput %]_med", "NVLink TX [%]"),
    ("p50_ms",                                  "p50 latency [ms]"),
]
KERNEL_COLS = [
    ("fmha_p50_us",      "FMHA p50 [us]"),
    ("fused_moe_p50_us", "fused_moe p50 [us]"),
    ("allreduce_p50_us", "allreduce p50 [us]"),
]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def panel(ax, rows, x_key: str, line_key: str, y_key: str, ylabel: str):
    """Group rows by `line_key`, plot one line per group, x = `x_key`."""
    groups: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        if r.get(y_key) in (None, ""):
            continue
        x = float(r[x_key])
        y = float(r[y_key])
        groups.setdefault(r[line_key], []).append((x, y))
    for label in sorted(groups, key=lambda s: int(s)):
        pts = sorted(groups[label])
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", label=f"{line_key}={label}")
    ax.set_xscale("log", base=2)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def make_figure(rows, x_key: str, line_key: str, out_path: Path, title: str):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for ax, (col, ylabel) in zip(axes.flat, METRICS):
        panel(ax, rows, x_key, line_key, col, ylabel)
    axes[0, 0].legend(loc="best", fontsize=8)
    axes[-1, 0].set_xlabel(x_key)
    axes[-1, 1].set_xlabel(x_key)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out_path}")


def make_kernel_figure(rows, out_path: Path, title: str):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=False)
    for ax, (col, ylabel) in zip(axes, KERNEL_COLS):
        # x = batch_size, line = prefill_len
        panel(ax, rows, "batch_size", "prefill_len", col, ylabel)
    axes[0].legend(loc="best", fontsize=8)
    for ax in axes:
        ax.set_xlabel("batch size")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path, help="sweep_results.csv from sweep.py")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Default: <csv-dir>/figs")
    args = ap.parse_args()

    rows = load_rows(args.csv)
    if not rows:
        print(f"[err] no rows in {args.csv}", file=sys.stderr)
        return 1
    print(f"[loaded] {len(rows)} rows from {args.csv}")

    out_dir = args.out_dir or args.csv.parent / "figs"

    # Figure A: HW vs batch size, one line per prefill.
    make_figure(
        rows,
        x_key="batch_size",
        line_key="prefill_len",
        out_path=out_dir / "by_batch.png",
        title="HW pressure vs. batch size (TP=4, bg=off)",
    )

    # Figure B: HW vs prefill length, one line per batch.
    make_figure(
        rows,
        x_key="prefill_len",
        line_key="batch_size",
        out_path=out_dir / "by_prefill.png",
        title="HW pressure vs. prefill length (TP=4, bg=off)",
    )

    # Figure C: kernel durations, batch on x, prefill as line.
    if any(r.get(KERNEL_COLS[0][0]) for r in rows):
        make_kernel_figure(
            rows,
            out_path=out_dir / "kernels.png",
            title="Hot-kernel p50 duration vs. batch size (TP=4)",
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
