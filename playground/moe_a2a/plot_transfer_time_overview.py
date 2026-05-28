#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Overview plot: per-call MoE all-to-all transfer time, four configs side
by side, with a shared quantile table beneath.

Layout::

    +-----------------+-----------------+
    |   Run 1 (top-L) |   Run 2 (top-R) |
    +-----------------+-----------------+
    |   Run 3 (bot-L) |   Run 4 (bot-R) |
    +-----------------+-----------------+
    |       quantile table (full width)            |
    +----------------------------------------------+

All four scatter subplots share x- and y-limits so configurations compare
directly. Each subplot uses the same per-rank-merged data and the same
``max(send, recv)`` per-direction wire size as the single-rank script.
Tail outliers above ``--time-pctl-cap`` (default 99.9) are dropped from
the scatter but still counted in the bottom quantile table, so the tail
is visible.

Usage::

    python playground/moe_a2a/plot_transfer_time_overview.py \\
        --run 2k=playground/out/moe_a2a_sharegpt_2026-05-27T15-16-56Z.csv \\
        --run 4k=playground/out/moe_a2a_sharegpt_maxbatch4096_<ts>.csv \\
        --run 8k=playground/out/moe_a2a_sharegpt_maxbatch8192_<ts>.csv \\
        --run 16k=playground/out/moe_a2a_sharegpt_maxbatch16384_<ts>.csv \\
        --output playground/out/figures/transfer_time_overview.png
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

MIB = 1024 * 1024


def _load(csv_path: Path) -> dict[str, list[tuple[float, float]]]:
    """Return ``{"dispatch": [(max_dir_bytes, ms), ...], "combine": [...]}``.

    Same per-direction wire-size definition as plot_transfer_time.py:
    dispatch send = in_bytes, recv = out_bytes - in_bytes; combine send =
    in_bytes - out_bytes, recv = out_bytes (both for W=2 AG/RS).
    Records without timing are dropped silently.
    """
    out: dict[str, list[tuple[float, float]]] = {"dispatch": [], "combine": []}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            d_t = row.get("dispatch_time_ms", "").strip()
            c_t = row.get("combine_time_ms", "").strip()
            if d_t:
                in_b = int(row["dispatch_in_bytes"])
                out_b = int(row["dispatch_out_bytes"])
                out["dispatch"].append((max(in_b, out_b - in_b), float(d_t)))
            if c_t:
                in_b = int(row["combine_in_bytes"])
                out_b = int(row["combine_out_bytes"])
                out["combine"].append((max(in_b - out_b, out_b), float(c_t)))
    return out


def _quantile(sorted_vals: list[float], q: float) -> float | None:
    if not sorted_vals:
        return None
    idx = max(0, min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def _global_limits(
    runs: list[tuple[str, dict[str, list[tuple[float, float]]]]],
    time_pctl_cap: float | None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute shared (xlim, ylim) across all runs/series.

    x is in MiB on log scale; floor to the nearest power of 10 below the
    min nonzero size for a tidy log axis. y is in ms; if time_pctl_cap is
    set, use the max of the per-series clipped maxes (so all subplots
    show the same y range and outliers stay off-axis); otherwise use the
    global max.
    """
    sizes_mib: list[float] = []
    ys_per_series_clipped: list[float] = []
    for _, series in runs:
        for kind in ("dispatch", "combine"):
            pts = series.get(kind, [])
            if not pts:
                continue
            sizes_mib.extend(b / MIB for b, _ in pts if b > 0)
            ts = sorted(t for _, t in pts)
            if time_pctl_cap is not None and 0 < time_pctl_cap < 100:
                idx = max(
                    0,
                    min(len(ts) - 1, int(round(time_pctl_cap / 100.0 * (len(ts) - 1)))),
                )
                ys_per_series_clipped.append(ts[idx])
            else:
                ys_per_series_clipped.append(ts[-1])
    if not sizes_mib:
        raise RuntimeError("no positive-size points found across any run")
    xmin = 10 ** math.floor(math.log10(min(sizes_mib)))
    xmax = 10 ** math.ceil(math.log10(max(sizes_mib)))
    ymax = max(ys_per_series_clipped) * 1.05
    return (xmin, xmax), (0.0, ymax)


def _draw_subplot(
    ax,
    title: str,
    series: dict[str, list[tuple[float, float]]],
    time_pctl_cap: float | None,
    marker_size: float,
    alpha: float,
    show_xlabel: bool,
    show_ylabel: bool,
) -> dict[str, tuple[int, float]]:
    """Render one scatter; return per-series (dropped_count, threshold)."""
    dropped: dict[str, tuple[int, float]] = {}
    for label, color in [("dispatch", "tab:blue"), ("combine", "tab:orange")]:
        pts = series.get(label, [])
        if not pts:
            continue
        if time_pctl_cap is not None and 0 < time_pctl_cap < 100:
            ts = sorted(t for _, t in pts)
            idx = max(
                0,
                min(len(ts) - 1, int(round(time_pctl_cap / 100.0 * (len(ts) - 1)))),
            )
            thresh = ts[idx]
            kept = [(b, t) for b, t in pts if t <= thresh]
            dropped[label] = (len(pts) - len(kept), thresh)
            pts = kept
        xs = [b / MIB for b, _ in pts]
        ys = [t for _, t in pts]
        ax.scatter(
            xs, ys,
            s=marker_size, alpha=alpha, color=color,
            label=f"{label} (n={len(pts):,})", edgecolors="none",
        )
    ax.set_xscale("log")
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=11)
    if show_xlabel:
        ax.set_xlabel("Max(Send, Recv) per Call (MiB)", fontsize=13)
    if show_ylabel:
        ax.set_ylabel("GPU Transfer Time (ms)", fontsize=13)
    leg = ax.legend(loc="upper right", fontsize=9, framealpha=0.92)
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    return dropped


def _build_table(
    ax,
    runs: list[tuple[str, dict[str, list[tuple[float, float]]]]],
    quantiles: list[float],
) -> None:
    """Render the bottom quantile table.

    Rows: one per (config, kind). Columns: config, kind, p50..max.
    Quantiles are computed on the FULL (pre-clip) data so the tail beyond
    the scatter cap is still surfaced numerically.
    """
    ax.axis("off")
    q_headers = [f"p{q*100:g}" for q in quantiles] + ["max"]
    col_labels = ["Max-Batch Tokens", "Stage"] + q_headers
    rows: list[list[str]] = []
    cell_colors: list[list[str]] = []
    # Group by stage (all dispatch first, then all combine) so the reader
    # can scan vertically to compare a single stage across configs.
    for kind, color in [("dispatch", "#dde6f4"), ("combine", "#fae3cf")]:
        for cfg_label, series in runs:
            ts = sorted(t for _, t in series.get(kind, []))
            n = len(ts)
            row = [cfg_label, f"{kind} (n={n:,})"]
            for q in quantiles:
                v = _quantile(ts, q)
                row.append("-" if v is None else f"{v:.3f}")
            row.append("-" if not ts else f"{ts[-1]:.3f}")
            rows.append(row)
            cell_colors.append([color] * len(col_labels))

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellColours=cell_colors,
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    # Bold header row.
    n_cols = len(col_labels)
    for j in range(n_cols):
        h = table[(0, j)]
        h.set_text_props(weight="bold", color="white")
        h.set_facecolor("#33476e")
    # Slim down the kind/config columns; spread the metric columns.
    table.auto_set_column_width(col=list(range(n_cols)))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--run", action="append", required=True,
        help="Repeatable: NAME=CSV_PATH. Order determines subplot order "
             "(top-left, top-right, bottom-left, bottom-right). Exactly 4 "
             "runs are expected for the 2x2 layout.",
    )
    ap.add_argument(
        "--output", type=Path, required=True, help="Output PNG path.",
    )
    ap.add_argument(
        "--time-pctl-cap", type=float, default=99.9,
        help="Drop transfer-time outliers above this per-series percentile "
             "from the scatter (default 99.9). The bottom table is computed "
             "on the FULL data so the tail remains visible.",
    )
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--marker-size", type=float, default=5.0)
    ap.add_argument(
        "--quantiles", nargs="+", type=float,
        default=[0.50, 0.90, 0.99, 0.999],
        help="Quantiles (as fractions in [0, 1]) shown in the table "
             "(default: 0.50 0.90 0.99 0.999). 'max' is always appended.",
    )
    args = ap.parse_args()

    runs: list[tuple[str, dict[str, list[tuple[float, float]]]]] = []
    for spec in args.run:
        if "=" not in spec:
            print(f"error: --run expects NAME=CSV_PATH, got {spec!r}", file=sys.stderr)
            return 2
        name, _, path = spec.partition("=")
        runs.append((name, _load(Path(path))))
    if len(runs) != 4:
        print(
            f"error: expected exactly 4 --run entries for the 2x2 layout, "
            f"got {len(runs)}",
            file=sys.stderr,
        )
        return 2

    xlim, ylim = _global_limits(runs, args.time_pctl_cap)

    fig = plt.figure(figsize=(13, 11), constrained_layout=True)
    gs = GridSpec(
        nrows=3, ncols=2,
        height_ratios=[1.0, 1.0, 0.85],
        figure=fig,
    )
    # Four scatter subplots in a 2x2 grid (top two rows).
    sp_axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    # Quantile table spans the full bottom row.
    tbl_ax = fig.add_subplot(gs[2, :])

    for i, (ax, (name, series)) in enumerate(zip(sp_axes, runs)):
        # Show xlabel on bottom row only; ylabel on left column only.
        show_xlabel = i >= 2
        show_ylabel = i % 2 == 0
        _draw_subplot(
            ax,
            title=f"Max-Batch Tokens = {name}",
            series=series,
            time_pctl_cap=args.time_pctl_cap,
            marker_size=args.marker_size,
            alpha=args.alpha,
            show_xlabel=show_xlabel,
            show_ylabel=show_ylabel,
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    _build_table(tbl_ax, runs, args.quantiles)

    fig.suptitle(
        "MoE All-to-All Transfer Time vs Max(Send, Recv) "
        "— DP=EP=2, AG/RS, Both Ranks Pooled",
        fontsize=16,
        fontweight="bold",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
