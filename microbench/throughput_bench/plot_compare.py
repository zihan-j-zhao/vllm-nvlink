#!/usr/bin/env python3
"""Compare multiple decode throughput result directories."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib missing", file=sys.stderr)
    sys.exit(1)


def percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round((len(values) - 1) * pct / 100.0)))
    return values[idx]


def load_row(path: Path, label: str, drop_first: int) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    cfg = payload["config"]
    ranks = payload["ranks"]
    n_iters = min(len(r["per_iter_ms"]) for r in ranks)
    vals = [
        max(float(r["per_iter_ms"][i]) for r in ranks)
        for i in range(n_iters)
    ][drop_first:]
    if not vals:
        raise ValueError(f"{path} has no iterations after drop_first={drop_first}")
    batch = int(cfg["batch_size"])
    dp = int(cfg["dp"])
    throughput = [batch * dp * 1000.0 / v for v in vals]
    mean_ms = statistics.fmean(vals)
    stdev_ms = statistics.stdev(vals) if len(vals) > 1 else 0.0
    bg = []
    for rank in ranks:
        bg.extend(rank.get("bg_traffic", []) or [])
    return {
        "label": label,
        "path": str(path),
        "batch_size": batch,
        "seq_len": int(cfg["seq_len"]),
        "tp": int(cfg["tp"]),
        "dp": dp,
        "iters": len(vals),
        "p50_ms": statistics.median(vals),
        "p90_ms": percentile(vals, 90),
        "p99_ms": percentile(vals, 99),
        "mean_ms": mean_ms,
        "cv_pct": 100.0 * stdev_ms / mean_ms if mean_ms else 0.0,
        "throughput_p50_tok_s": statistics.median(throughput),
        "throughput_mean_tok_s": statistics.fmean(throughput),
        "bg_achieved_gbps_total": sum(
            float(x.get("achieved_gbps_total", 0.0)) for x in bg
        ),
        "bg_chunk_bytes": next(
            (x.get("chunk_bytes") for x in bg if "chunk_bytes" in x), ""
        ),
        "bg_chunks_per_burst": next(
            (x.get("chunks_per_burst") for x in bg if "chunks_per_burst" in x), ""
        ),
    }


def load_dir(path: Path, label: str, drop_first: int) -> list[dict[str, Any]]:
    return [
        load_row(p, label, drop_first)
        for p in sorted(path.glob("*.json"))
    ]


def write_summary(rows: list[dict[str, Any]], out: Path) -> None:
    fields = [
        "label", "batch_size", "seq_len", "tp", "dp", "iters",
        "p50_ms", "p90_ms", "p99_ms", "mean_ms", "cv_pct",
        "throughput_p50_tok_s", "throughput_mean_tok_s",
        "bg_achieved_gbps_total", "bg_chunk_bytes", "bg_chunks_per_burst",
        "path",
    ]
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})


def plot_metric(
    rows: list[dict[str, Any]],
    metric: str,
    ylabel: str,
    title: str,
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label in sorted({r["label"] for r in rows}):
        pts = sorted(
            (r["batch_size"], float(r[metric]))
            for r in rows
            if r["label"] == label
        )
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_xlabel("decode batch size per DP rank")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--series",
        action="append",
        required=True,
        help="LABEL=DIR. Can be repeated.",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--drop-first", type=int, default=1)
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    for spec in args.series:
        label, sep, raw_path = spec.partition("=")
        if not sep:
            raise SystemExit(f"--series must be LABEL=DIR, got {spec!r}")
        rows.extend(load_dir(Path(raw_path), label, args.drop_first))
    rows.sort(key=lambda r: (r["label"], r["batch_size"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_summary(rows, args.out_dir / "compare_summary.csv")
    plot_metric(
        rows,
        "p50_ms",
        "p50 forward latency [ms]",
        "Decode latency under LMCache-shaped KV traffic",
        args.out_dir / "latency_vs_batch_compare.png",
    )
    plot_metric(
        rows,
        "throughput_p50_tok_s",
        "p50 logical decode throughput [tokens/s]",
        "Decode throughput under LMCache-shaped KV traffic",
        args.out_dir / "throughput_vs_batch_compare.png",
    )
    plot_metric(
        rows,
        "cv_pct",
        "latency CV [%]",
        "Decode stability under LMCache-shaped KV traffic",
        args.out_dir / "cv_vs_batch_compare.png",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

