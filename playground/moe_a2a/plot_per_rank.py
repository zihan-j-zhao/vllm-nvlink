#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot per-rank, per-layer-step MoE all-to-all byte volumes.

Reads the cleaned per-step CSV produced by ``extract_per_step.py`` and writes
one PNG per rank. Each PNG is a 2x2 grid:

    +-------------------------+-------------------------+
    | dispatch_in_bytes       | dispatch_out_bytes      |
    | vs scheduled_tokens     | vs scheduled_tokens     |
    +-------------------------+-------------------------+
    | combine_in_bytes        | combine_out_bytes       |
    | vs scheduled_tokens     | vs scheduled_tokens     |
    +-------------------------+-------------------------+

Each row of the CSV is one (rank, step_id, layer_idx) sample, so every step
contributes ``num_layers`` points per subplot. Points are colored by
``layer_idx`` so the per-layer dimension is visible even when AG/RS sizes
happen to be identical across layers within a step (which is the typical
case for standard MoE).

Usage::

    python playground/moe_a2a/plot_per_rank.py \\
        --csv playground/out/moe_a2a_sharegpt_<ts>.csv \\
        --output-dir playground/out/figures
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


_COLS = (
    "layer_idx",
    "scheduled_tokens",
    "dispatch_in_bytes",
    "dispatch_out_bytes",
    "combine_in_bytes",
    "combine_out_bytes",
)


def _load(csv_path: Path) -> dict[int, dict[str, list[int]]]:
    """Return ``{rank: {column: [values...]}}`` for the columns we plot."""
    by_rank: dict[int, dict[str, list[int]]] = defaultdict(
        lambda: {c: [] for c in _COLS}
    )
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rank = int(row["rank"])
            for c in _COLS:
                by_rank[rank][c].append(int(row[c]))
    return dict(by_rank)


def _plot_rank(
    rank: int,
    data: dict[str, list[int]],
    out_path: Path,
) -> None:
    MIB = 1024 * 1024
    XMAX = 2048

    fig, axes = plt.subplots(
        2, 2,
        figsize=(13, 10),
        constrained_layout=True,
        sharex="col",
    )
    fig.suptitle(
        f"MoE Transfer at DP=2, EP=2 (Rank {rank})",
        fontsize=20,
    )

    # Semantics for the rank-local "outbound" / "inbound" labels (wire
    # traffic per direction, W=2 AG/RS). The CSV's dispatch_out_bytes and
    # combine_in_bytes columns include this rank's OWN share (the full
    # gathered / pre-scatter tensor sizes), so the inbound side of dispatch
    # and the outbound side of combine must be derived by subtracting the
    # rank-local contribution; otherwise those two panels overstate the
    # wire transfer by ~2x (showing up to ~16 MiB instead of the true ~8
    # MiB at max-batch).
    #
    #   dispatch outbound = dispatch_in_bytes                       (own AG send)
    #   dispatch inbound  = dispatch_out_bytes - dispatch_in_bytes  (peer's AG share)
    #   combine  outbound = combine_in_bytes  - combine_out_bytes   (peer's RS share)
    #   combine  inbound  = combine_out_bytes                       (own RS keep)
    di  = data["dispatch_in_bytes"]
    do  = data["dispatch_out_bytes"]
    ci  = data["combine_in_bytes"]
    co  = data["combine_out_bytes"]
    n = len(di)
    dispatch_out_wire = di
    dispatch_in_wire  = [do[i] - di[i] for i in range(n)]
    combine_out_wire  = [ci[i] - co[i] for i in range(n)]
    combine_in_wire   = co
    panels = [
        (axes[0, 0], dispatch_out_wire, "Dispatch (outbound)"),
        (axes[0, 1], dispatch_in_wire,  "Dispatch (inbound)"),
        (axes[1, 0], combine_out_wire,  "Combine (outbound)"),
        (axes[1, 1], combine_in_wire,   "Combine (inbound)"),
    ]
    x_all = data["scheduled_tokens"]
    c_all = data["layer_idx"]
    # Mask once: keep only points within the x cap so y-autoscale fits them.
    mask = [v <= XMAX for v in x_all]
    kept_idx = [i for i, m in enumerate(mask) if m]
    # Shuffle the draw order so layers cover each other uniformly. Per-layer
    # AG/RS volumes are identical within a step (the size depends on the
    # batch, not the layer), so all 48 layer points land at the same (x, y);
    # without shuffling, the highest layer_idx is always drawn last and ends
    # up the only visible color everywhere.
    random.Random(0).shuffle(kept_idx)
    x = [x_all[i] for i in kept_idx]
    c = [c_all[i] for i in kept_idx]

    sc = None
    for ax, series, title in panels:
        y = [series[i] / MIB for i in kept_idx]
        sc = ax.scatter(
            x, y, c=c, cmap="turbo",
            s=20, alpha=0.45, edgecolors="none",
        )
        ax.set_ylabel("Transfer Size (MiB)", fontsize=15)
        ax.set_title(title, fontsize=17)
        ax.grid(True, which="both", alpha=0.2, linestyle="--")
        ax.tick_params(axis="both", labelsize=13)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        # Fit y to the visible (within-cap) data with a small headroom.
        if y:
            ymin, ymax = min(y), max(y)
            pad = max((ymax - ymin) * 0.05, ymax * 0.01)
            ax.set_ylim(bottom=max(0.0, ymin - pad), top=ymax + pad)

    # Cap x-axis at 2048 tokens; the long tail of outliers beyond that
    # (chunked-prefill steps with very large local chunks) compresses the
    # bulk of the data when shown.
    for ax in axes.flat:
        ax.set_xlim(left=0, right=XMAX)

    # Only put the x-label on the bottom row (top row shares x with bottom).
    for ax in axes[1, :]:
        ax.set_xlabel("Token Count", fontsize=15)

    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes, shrink=0.75, pad=0.02, location="right")
        cbar.set_label("layer_idx", fontsize=15)
        cbar.ax.tick_params(labelsize=13)

    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[plot_per_rank] wrote {out_path}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, type=Path,
                    help="Per-step CSV from extract_per_step.py")
    ap.add_argument("--output-dir", required=True, type=Path,
                    help="Directory to write PNG files into.")
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    by_rank = _load(args.csv)
    for rank in sorted(by_rank):
        out_path = args.output_dir / f"moe_a2a_rank{rank}.png"
        _plot_rank(rank, by_rank[rank], out_path)


if __name__ == "__main__":
    main()
