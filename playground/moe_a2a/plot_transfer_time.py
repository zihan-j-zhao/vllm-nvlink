#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot per-rank MoE all-to-all transfer time vs per-rank send size.

Reads the per-step CSV from ``extract_per_step.py`` and writes one PNG per
rank, each showing two scatters:

* ``dispatch`` (NCCL ``all_gatherv``): x = ``dispatch_in_bytes`` per call,
  i.e. the bytes this rank pushes onto the gather. y = ``dispatch_time_ms``.
* ``combine`` (NCCL ``reduce_scatterv``): x = ``combine_in_bytes`` per call,
  i.e. the bytes this rank pushes onto the reduce. y = ``combine_time_ms``.

We plot the **per-rank send size** (``*_in_bytes``) on the x-axis rather
than ``in + out`` because that's the actual NCCL message size that drives
runtime: at DP=EP=2 with the naive AG/RS backend, two calls with the same
``in + out`` but different ``(in, out)`` split execute different NCCL
traffic and can take different times.

Calls with empty ``dispatch_time_ms`` / ``combine_time_ms`` (rows with no
timing recorded — e.g. from an older trace whose timing sidecar was
truncated, or any record the profiler emitted before timing was inlined)
are dropped.

Usage::

    python playground/moe_a2a/plot_transfer_time.py \\
        --csv playground/out/moe_a2a_sharegpt_<ts>.csv \\
        --output-dir playground/out/figures
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(
    csv_path: Path,
    layer_filter: set[int] | None = None,
) -> dict[int, dict[str, list[tuple[float, float]]]]:
    """Return ``{rank: {"dispatch": [(max_dir_bytes, ms), ...], "combine": [...]}}``.

    x is the **per-direction max** wire-size on full-duplex NVLink, derived
    from the CSV in_bytes/out_bytes columns. For ws=2 (the AG/RS naive
    backend at DP=EP=2) the per-direction send and recv on the wire are::

        dispatch (all_gatherv):  send = in_bytes,            recv = out_bytes - in_bytes
        combine  (reduce_scatterv): send = in_bytes - out_bytes, recv = out_bytes

    The kernel runs once with both directions in flight concurrently, so
    its wall-clock duration is bounded by the larger direction. Plotting
    ``max(send, recv)`` captures the bandwidth-limiting half regardless of
    whether the two DP workers had symmetric or asymmetric loads (decoder
    + prefill mix).

    Rows without the corresponding timing column are skipped.
    """
    out: dict[int, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: {"dispatch": [], "combine": []}
    )
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rank = int(row["rank"])
            if layer_filter is not None and int(row["layer_idx"]) not in layer_filter:
                continue
            d_t = row.get("dispatch_time_ms", "").strip()
            c_t = row.get("combine_time_ms", "").strip()
            if d_t:
                in_b = int(row["dispatch_in_bytes"])
                out_b = int(row["dispatch_out_bytes"])
                # send = in_b, recv = out_b - in_b (ws=2 all_gatherv).
                x = max(in_b, out_b - in_b)
                out[rank]["dispatch"].append((x, float(d_t)))
            if c_t:
                in_b = int(row["combine_in_bytes"])
                out_b = int(row["combine_out_bytes"])
                # send = in_b - out_b, recv = out_b (ws=2 reduce_scatterv).
                x = max(in_b - out_b, out_b)
                out[rank]["combine"].append((x, float(c_t)))
    return dict(out)


def _plot_rank(
    rank: int,
    series: dict[str, list[tuple[float, float]]],
    out_path: Path,
    *,
    alpha: float,
    marker_size: float,
    logx: bool,
    logy: bool,
    ymax: float | None,
    time_pctl_cap: float | None = None,
    show_quantile_table: bool = True,
    layer_filter: set[int] | None = None,
) -> None:
    MIB = 1024 * 1024
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    ax.tick_params(axis="both", labelsize=12)

    n_d = len(series["dispatch"])
    n_c = len(series["combine"])
    dropped: dict[str, tuple[int, float]] = {}
    for label, color in [("dispatch", "tab:blue"), ("combine", "tab:orange")]:
        pts = series[label]
        if not pts:
            continue
        # Optional per-series percentile clip on transfer time. Drops the
        # top (100 - p)% slowest calls so the bulk distribution is readable
        # without a few-ms tail compressing the y-axis.
        if time_pctl_cap is not None and 0 < time_pctl_cap < 100:
            ts = sorted(t for _, t in pts)
            idx = max(0, min(len(ts) - 1, int(round(time_pctl_cap / 100.0 * (len(ts) - 1)))))
            thresh = ts[idx]
            kept = [(b, t) for b, t in pts if t <= thresh]
            dropped[label] = (len(pts) - len(kept), thresh)
            pts = kept
        xs = [b / MIB for b, _ in pts]
        ys = [t for _, t in pts]
        ax.scatter(
            xs, ys,
            s=marker_size, alpha=alpha, color=color,
            label=f"{label} (n={len(pts)})",
            edgecolors="none",
        )

    ax.set_xlabel("Max(Send, Recv) per Call (MiB)", fontsize=14)
    ax.set_ylabel("GPU Transfer Time (ms)", fontsize=14)
    ax.set_title(
        f"MoE All-to-All Transfer Time vs Max(Send, Recv) "
        f"— Rank {rank}, DP=EP=2, AG/RS"
        + (f", layers={sorted(layer_filter)}" if layer_filter else ""),
        fontsize=14,
    )
    ax.grid(True, alpha=0.3)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    if ymax is not None:
        ax.set_ylim(top=ymax)
    leg = ax.legend(loc="upper right", fontsize=11, framealpha=0.92)
    for h in leg.legend_handles:
        h.set_alpha(1.0)

    if show_quantile_table:
        # Quantiles computed on the FULL series (pre-clip) so the table
        # still reports the tail even when --time-pctl-cap hides it from
        # the scatter. Anchored top-left in axes coords; the figure
        # consistently has whitespace there because most points cluster
        # at low y for nearly all x.
        qs = [0.50, 0.75, 0.90, 0.95, 0.99, 0.999]
        d_ts = sorted(t for _, t in series.get("dispatch", []))
        c_ts = sorted(t for _, t in series.get("combine", []))

        def _q(ts: list[float], q: float) -> str:
            if not ts:
                return "     -"
            idx = max(0, min(len(ts) - 1, int(round(q * (len(ts) - 1)))))
            return f"{ts[idx]:6.3f}"

        lines = [
            "Transfer Time (ms)",
            f"{'quantile':>9}  {'dispatch':>8}  {'combine':>8}",
            "-" * 31,
        ]
        for q in qs:
            tag = f"p{q*100:g}"
            lines.append(f"{tag:>9}  {_q(d_ts, q):>8}  {_q(c_ts, q):>8}")
        if d_ts or c_ts:
            lines.append(
                f"{'max':>9}  "
                f"{(f'{d_ts[-1]:6.3f}' if d_ts else '     -'):>8}  "
                f"{(f'{c_ts[-1]:6.3f}' if c_ts else '     -'):>8}"
            )
        ax.text(
            0.988, 0.78, "\n".join(lines),
            transform=ax.transAxes,
            family="monospace", fontsize=11,
            va="top", ha="right",
            bbox=dict(
                boxstyle="round,pad=0.45",
                facecolor="white", edgecolor="0.6", alpha=0.92,
            ),
        )

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot_transfer_time] wrote {out_path} (dispatch={n_d}, combine={n_c})")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Per-step CSV from extract_per_step.py.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Directory to write per-rank PNGs into. Files are named "
            "transfer_time_rank{rank}.png."
        ),
    )
    ap.add_argument(
        "--alpha", type=float, default=0.25,
        help="Per-point alpha (default 0.25).",
    )
    ap.add_argument(
        "--marker-size", type=float, default=6.0,
        help="Marker area in points^2 (default 6).",
    )
    ap.add_argument("--logx", action="store_true")
    ap.add_argument("--logy", action="store_true")
    ap.add_argument(
        "--ymax", type=float, default=None,
        help="Optional fixed y-axis max in ms (otherwise auto).",
    )
    ap.add_argument(
        "--time-pctl-cap", type=float, default=None,
        help=(
            "Drop transfer-time outliers above this per-series percentile "
            "(e.g. 99.9 keeps only the bottom 99.9%% of each series). "
            "Drop count + threshold are noted in the legend."
        ),
    )
    ap.add_argument(
        "--no-quantile-table", dest="quantile_table", action="store_false",
        help=(
            "Suppress the per-series quantile table overlay in the upper-"
            "left of each plot (table is shown by default)."
        ),
    )
    ap.set_defaults(quantile_table=True)
    ap.add_argument(
        "--layer-idx", nargs="+", type=int, default=None,
        help=(
            "Restrict to calls at the given layer indices. "
            "E.g. --layer-idx 0 isolates the first MoE layer of each step, "
            "where DP-rank drift is minimal (~9% slow band)."
        ),
    )
    args = ap.parse_args()

    layer_filter = set(args.layer_idx) if args.layer_idx is not None else None
    by_rank = _load(args.csv, layer_filter=layer_filter)
    if not by_rank:
        print(f"error: {args.csv} has no rows with timing data", file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for rank in sorted(by_rank):
        out = args.output_dir / f"transfer_time_rank{rank}.png"
        _plot_rank(
            rank, by_rank[rank], out,
            alpha=args.alpha, marker_size=args.marker_size,
            logx=args.logx, logy=args.logy, ymax=args.ymax,
            time_pctl_cap=args.time_pctl_cap,
            show_quantile_table=args.quantile_table,
            layer_filter=layer_filter,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
