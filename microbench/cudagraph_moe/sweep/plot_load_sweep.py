#!/usr/bin/env python3
"""Focused plot for one background-load sweep case."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def bg_load_gbps(row: dict[str, str]) -> float:
    if row["bg"] == "off":
        return 0.0
    val = row.get("bg_ingress_gbps_mean", "")
    return float(val) if val else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--max-gbps", type=float, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = load_rows(args.csv)
    if not rows:
        raise SystemExit(f"no rows in {args.csv}")

    rows = sorted(rows, key=bg_load_gbps)
    base = next((float(r["p50_ms"]) for r in rows if r["bg"] == "off"), None)
    if base is None:
        raise SystemExit("missing off baseline row")

    xs = [bg_load_gbps(r) for r in rows]
    ys = [float(r["p50_ms"]) for r in rows]
    labels = [r["bg"] for r in rows]
    slowdowns = [100.0 * (y / base - 1.0) for y in ys]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(xs, ys, marker="o", linewidth=2.0, color="#1f77b4")
    for x, y, label, slowdown in zip(xs, ys, labels, slowdowns):
        suffix = "baseline" if label == "off" else f"{slowdown:+.1f}%"
        ax.annotate(
            f"{label}\n{y:.2f} ms\n{suffix}",
            xy=(x, y), xytext=(0, 10), textcoords="offset points",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_xlabel("achieved ingress CE traffic per decoder [GB/s]")
    ax.set_ylabel("forward p50 [ms]")
    ax.grid(True, alpha=0.3)
    title = "DP=EP=4, B=256, prefill=2048: ingress CE load sweep"
    if args.max_gbps:
        top = ax.secondary_xaxis(
            "top",
            functions=(lambda x: 100.0 * x / args.max_gbps,
                       lambda p: p * args.max_gbps / 100.0),
        )
        top.set_xlabel("% of calibrated ingress_max")
    ax.set_title(title)
    fig.tight_layout()

    out = args.out or args.csv.parent / "figs_bg" / "ingress_load_latency.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[wrote] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
