#!/usr/bin/env python3
"""Compare-vs-baseline plots: bg traffic impact on decode HW pressure.

Reads a sweep_results.csv that has multiple `bg` values per (batch,
prefill) cell and produces:

  - <out>/bg_p50_heatmap.png    per-bg slowdown vs off, heatmap over (B, P)
  - <out>/bg_hw_vs_batch_p2048.png   one panel per metric, line per bg
  - <out>/bg_hw_vs_prefill_b1024.png same, different axis
  - <out>/bg_achieved_gbps.png       sanity check: did bg actually saturate?
"""

from __future__ import annotations

import argparse
import csv
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


METRICS = [
    ("DRAM Read Bandwidth [Throughput %]_med", "HBM Read [%]"),
    ("DRAM Write Bandwidth [Throughput %]_med", "HBM Write [%]"),
    ("SMs Active [Throughput %]_med",          "SMs Active [%]"),
    ("NVLink TX Requests User Data [Throughput %]_med", "NVLink TX [%]"),
    ("NVLink RX Requests User Data [Throughput %]_med", "NVLink RX [%]"),
    ("p50_ms",                                  "p50 latency [ms]"),
]
BG_ORDER = ["off", "ingress_med", "ingress_max", "egress_med", "egress_max"]
BG_COLORS = {
    "off":         "#444444",
    "ingress_med": "#1f77b4",
    "ingress_max": "#08306b",
    "egress_med":  "#ff7f0e",
    "egress_max":  "#7f2704",
}


def bg_sort_key(bg: str) -> tuple[int, float | str]:
    if bg in BG_ORDER:
        return (BG_ORDER.index(bg), 0)
    if bg.startswith("ingress_pct"):
        pct = bg.removeprefix("ingress_pct").replace("p", ".")
        try:
            return (2, float(pct))
        except ValueError:
            pass
    return (99, bg)


def bg_color(bg: str) -> str | None:
    if bg in BG_COLORS:
        return BG_COLORS[bg]
    if bg.startswith("ingress_pct"):
        return "#1f77b4"
    return None


def ordered_bgs(rows: list[dict], *, include_off: bool) -> list[str]:
    bgs = {r["bg"] for r in rows if include_off or r["bg"] != "off"}
    return sorted(bgs, key=bg_sort_key)


def load(path: Path) -> list[dict]:
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    # Coerce types.
    for r in rows:
        r["batch_size"] = int(r["batch_size"])
        r["prefill_len"] = int(r["prefill_len"])
        r["p50_ms"] = float(r["p50_ms"])
    return rows


# ---------------------------------------------------------------------------
# Headline: slowdown heatmap per bg
# ---------------------------------------------------------------------------
def slowdown_heatmap(rows, out: Path):
    """4-panel heatmap: cell value = p50(bg) / p50(off) - 1, in %."""
    bgs_present = ordered_bgs(rows, include_off=False)
    if not bgs_present:
        print("[skip] no bg cells to compare"); return

    batches = sorted({r["batch_size"] for r in rows})
    prefills = sorted({r["prefill_len"] for r in rows})

    # Index off baseline.
    off = {(r["batch_size"], r["prefill_len"]): r["p50_ms"]
           for r in rows if r["bg"] == "off"}

    n = len(bgs_present)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n + 1, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, bg in zip(axes, bgs_present):
        mat = np.full((len(prefills), len(batches)), np.nan)
        for r in rows:
            if r["bg"] != bg:
                continue
            base = off.get((r["batch_size"], r["prefill_len"]))
            if base is None or base == 0:
                continue
            i = prefills.index(r["prefill_len"])
            j = batches.index(r["batch_size"])
            mat[i, j] = 100 * (r["p50_ms"] / base - 1.0)
        im = ax.imshow(mat, cmap="RdYlGn_r", vmin=-5, vmax=30, aspect="auto")
        ax.set_xticks(range(len(batches)), batches)
        ax.set_yticks(range(len(prefills)), prefills)
        ax.set_xlabel("batch size")
        ax.set_title(bg)
        for i in range(len(prefills)):
            for j in range(len(batches)):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:+.1f}%", ha="center", va="center",
                        fontsize=8, color="black")
    axes[0].set_ylabel("prefill len")
    fig.colorbar(im, ax=axes, label="p50 slowdown vs off [%]",
                 fraction=0.025, pad=0.02)
    fig.suptitle("Decode-step p50 slowdown under bg traffic")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out}")


# ---------------------------------------------------------------------------
# HW pressure: line plot, one line per bg
# ---------------------------------------------------------------------------
def hw_lines(rows, x_key: str, fixed_key: str, fixed_val: int,
             out: Path, title: str):
    sub = [r for r in rows if r.get(fixed_key) == fixed_val
           or (isinstance(r.get(fixed_key), int) and r[fixed_key] == fixed_val)]
    sub = [r for r in sub if int(r[fixed_key]) == fixed_val]
    if not sub:
        print(f"[skip] no rows with {fixed_key}={fixed_val}")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for ax, (col, ylabel) in zip(axes.flat, METRICS):
        groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for r in sub:
            v = r.get(col)
            if v in (None, ""):
                continue
            try:
                groups[r["bg"]].append((float(r[x_key]), float(v)))
            except Exception:
                pass
        for bg in ordered_bgs(sub, include_off=True):
            if bg not in groups:
                continue
            pts = sorted(groups[bg])
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", label=bg, color=bg_color(bg))
        ax.set_xscale("log", base=2)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(loc="best", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel(x_key)
    fig.suptitle(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out}")


# ---------------------------------------------------------------------------
# Achieved bg Gbps (sanity)
# ---------------------------------------------------------------------------
def achieved_gbps(rows, out: Path):
    """Bar chart per cell: how much bandwidth bg actually pushed?"""
    sub = [r for r in rows if r["bg"] != "off"]
    if not sub:
        return
    # Pull bg_<dir>_gbps_mean across directions.
    by_bg = defaultdict(list)
    for r in sub:
        per = []
        for k, v in r.items():
            if k.startswith("bg_") and k.endswith("_gbps_mean") and v not in (None, ""):
                per.append(float(v))
        if per:
            by_bg[r["bg"]].append((r["tag"], sum(per)))
    if not by_bg:
        return
    bgs = sorted(by_bg, key=bg_sort_key)
    fig, axes = plt.subplots(len(bgs), 1, figsize=(11, 2.0 * len(bgs)),
                             sharex=True)
    if len(bgs) == 1:
        axes = [axes]
    # All subplots share the x axis. Use the (B, prefill) part of the tag,
    # stripped of the bg suffix, so the labels are correct in every panel
    # (otherwise the last-set tick labels overwrite earlier ones).
    def strip_bg(tag: str, bg: str) -> str:
        return tag.removesuffix("_" + bg)
    for ax, bg in zip(axes, bgs):
        pts = sorted(by_bg[bg], key=lambda x: x[0])
        tags = [strip_bg(t, bg) for t, _ in pts]
        vals = [v for _, v in pts]
        ax.bar(range(len(vals)), vals, color=bg_color(bg))
        ax.set_xticks(range(len(tags)), tags, rotation=60, ha="right",
                      fontsize=7)
        ax.set_ylabel("Gbps")
        ax.set_title(f"{bg}  (achieved bg bandwidth, sum over directions, per rank)")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Sanity check: did bg traffic actually saturate the link?")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    rows = load(args.csv)
    out_dir = args.out_dir or args.csv.parent / "figs_bg"
    slowdown_heatmap(rows, out_dir / "bg_p50_heatmap.png")
    hw_lines(rows, x_key="batch_size", fixed_key="prefill_len",
             fixed_val=2048,
             out=out_dir / "bg_hw_vs_batch_p2048.png",
             title="HW pressure vs batch (prefill=2048, TP=4)")
    hw_lines(rows, x_key="prefill_len", fixed_key="batch_size",
             fixed_val=1024,
             out=out_dir / "bg_hw_vs_prefill_b1024.png",
             title="HW pressure vs prefill (batch=1024, TP=4)")
    achieved_gbps(rows, out_dir / "bg_achieved_gbps.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
