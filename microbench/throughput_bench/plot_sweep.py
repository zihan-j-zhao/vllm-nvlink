#!/usr/bin/env python3
"""Plot decode throughput sweep results.

Reads one or more ``decode_stability.json``-style files from a directory and
plots the stage-2 no-background baseline:

  - throughput_vs_batch.png
  - latency_vs_batch.png
  - cv_vs_batch.png
  - sweep_summary.csv
"""

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


def iter_wall_latencies(payload: dict[str, Any], drop_first: int) -> list[float]:
    ranks = payload["ranks"]
    n_iters = min(len(r["per_iter_ms"]) for r in ranks)
    vals = [
        max(float(r["per_iter_ms"][i]) for r in ranks)
        for i in range(n_iters)
    ]
    return vals[drop_first:]


def load_row(path: Path, drop_first: int) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    cfg = payload["config"]
    vals = iter_wall_latencies(payload, drop_first)
    if not vals:
        raise ValueError(f"{path} has no iterations after --drop-first={drop_first}")

    batch = int(cfg["batch_size"])
    dp = int(cfg["dp"])
    seq_len = int(cfg["seq_len"])
    tokens_per_step = batch * dp
    throughput = [tokens_per_step * 1000.0 / v for v in vals]
    mean_ms = statistics.fmean(vals)
    stdev_ms = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return {
        "path": str(path),
        "batch_size": batch,
        "seq_len": seq_len,
        "tp": int(cfg["tp"]),
        "dp": dp,
        "world_size": int(cfg["world_size"]),
        "iters": len(vals),
        "drop_first": drop_first,
        "p50_ms": statistics.median(vals),
        "p90_ms": percentile(vals, 90),
        "p99_ms": percentile(vals, 99),
        "mean_ms": mean_ms,
        "cv_pct": 100.0 * stdev_ms / mean_ms if mean_ms else 0.0,
        "throughput_p50_tok_s": statistics.median(throughput),
        "throughput_mean_tok_s": statistics.fmean(throughput),
        "logical_tokens_per_step": tokens_per_step,
    }


def find_jsons(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        p for p in path.glob("*.json")
        if p.name != "sweep_summary.json"
    )


def write_summary(rows: list[dict[str, Any]], out: Path) -> None:
    fields = [
        "batch_size", "seq_len", "tp", "dp", "world_size", "iters",
        "drop_first", "p50_ms", "p90_ms", "p99_ms", "mean_ms", "cv_pct",
        "throughput_p50_tok_s", "throughput_mean_tok_s",
        "logical_tokens_per_step", "path",
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
    seq_lens = sorted({r["seq_len"] for r in rows})
    for seq_len in seq_lens:
        pts = sorted(
            (r["batch_size"], float(r[metric]))
            for r in rows
            if r["seq_len"] == seq_len
        )
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", label=f"seq_len={seq_len}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({r["batch_size"] for r in rows}))
    ax.set_xticklabels([str(x) for x in sorted({r["batch_size"] for r in rows})])
    ax.set_xlabel("decode batch size per DP rank")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if len(seq_lens) > 1:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "path", type=Path,
        help="Directory of per-batch JSON files, or one JSON file.",
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--drop-first", type=int, default=1,
        help="Drop first timed iteration when computing plotted metrics. "
             "Use 0 to exactly match JSON aggregate.",
    )
    args = ap.parse_args()

    jsons = find_jsons(args.path)
    if not jsons:
        print(f"no JSON files found in {args.path}", file=sys.stderr)
        return 1
    rows = sorted(
        (load_row(path, args.drop_first) for path in jsons),
        key=lambda r: (r["seq_len"], r["batch_size"]),
    )

    out_dir = args.out_dir or (args.path if args.path.is_dir() else args.path.parent) / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "sweep_summary.csv"
    write_summary(rows, summary_csv)
    print(f"[wrote] {summary_csv}")

    title_suffix = (
        f"TP={rows[0]['tp']}, DP=EP={rows[0]['dp']}, "
        f"drop_first={args.drop_first}"
    )
    plot_metric(
        rows,
        "throughput_p50_tok_s",
        "p50 logical decode throughput [tokens/s]",
        f"Decode throughput vs batch ({title_suffix})",
        out_dir / "throughput_vs_batch.png",
    )
    plot_metric(
        rows,
        "p50_ms",
        "p50 forward latency [ms]",
        f"Decode latency vs batch ({title_suffix})",
        out_dir / "latency_vs_batch.png",
    )
    plot_metric(
        rows,
        "cv_pct",
        "latency CV [%]",
        f"Decode stability vs batch ({title_suffix})",
        out_dir / "cv_vs_batch.png",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

