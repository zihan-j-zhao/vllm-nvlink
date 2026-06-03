#!/usr/bin/env python3
"""Plot tail slowdown vs normalized NIXL transfer volume for one overlap run."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"matplotlib is required: {exc}")


def _float(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        return float(row[key])
    except Exception:
        return default


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _pressure_for_step(step: dict[str, str], xfers: list[dict[str, str]]) -> dict[str, float]:
    start = _float(step, "start")
    end = _float(step, "end")
    duration_s = max(end - start, 1e-12)
    events: list[tuple[float, float]] = []
    byte_rate_seconds = 0.0
    bytes_during_step = 0.0
    for xfer in xfers:
        xfer_start = _float(xfer, "start")
        xfer_end = _float(xfer, "end")
        if xfer_end <= start or xfer_start >= end:
            continue
        lo = max(start, xfer_start)
        hi = min(end, xfer_end)
        if hi <= lo:
            continue
        xfer_duration = max(xfer_end - xfer_start, 1e-12)
        bytes_total = _float(xfer, "bytes_transferred")
        if not math.isfinite(bytes_total):
            continue
        rate_gbps = bytes_total / xfer_duration / 1e9
        events.append((lo, rate_gbps))
        events.append((hi, -rate_gbps))
        byte_rate_seconds += rate_gbps * (hi - lo)
        bytes_during_step += bytes_total * ((hi - lo) / xfer_duration)
    active_bw = 0.0
    max_bw = 0.0
    for _, delta in sorted(events, key=lambda item: (item[0], -item[1])):
        active_bw += delta
        max_bw = max(max_bw, active_bw)
    union_ms = _float(step, "union_overlap_ms", 0.0)
    return {
        "max_xfer_pressure_GBps": max_bw,
        "avg_step_xfer_pressure_GBps": byte_rate_seconds / duration_s,
        "avg_active_xfer_pressure_GBps": (
            byte_rate_seconds / max(union_ms / 1000.0, 1e-12) if union_ms > 0 else 0.0
        ),
        "bytes_during_step_MB": bytes_during_step / 1e6,
        "normalized_transfer_MB_per_ms": (bytes_during_step / 1e6) / max(_float(step, "duration_ms"), 1e-9),
    }


def _quantile(values: list[float], p: float) -> float:
    values = [v for v in values if math.isfinite(v)]
    if not values:
        return float("nan")
    return float(np.percentile(values, p))


def _summarize(rows: list[dict[str, Any]], metric: str, buckets: list[tuple[str, float, float]]) -> list[dict[str, Any]]:
    out = []
    for label, lo, hi in buckets:
        def in_bucket(row: dict[str, Any]) -> bool:
            value = float(row[metric])
            if label == "0":
                return value <= 1e-12
            if math.isinf(hi):
                return value > lo
            return value > lo and value <= hi

        group = [row for row in rows if in_bucket(row)]
        durations = [float(row["duration_ms"]) for row in group]
        out.append({
            "bucket": label,
            "n": len(group),
            "p50": _quantile(durations, 50),
            "p90": _quantile(durations, 90),
            "p95": _quantile(durations, 95),
            "p99": _quantile(durations, 99),
            "multi_pct": 100.0 * sum(int(float(row["max_active_recv"])) >= 2 for row in group) / len(group) if group else float("nan"),
            "estimated_mb_p50": _quantile([float(row["bytes_during_step_MB"]) for row in group], 50),
            "normalized_mb_per_ms_p50": _quantile([float(row["normalized_transfer_MB_per_ms"]) for row in group], 50),
            "union_overlap_ms_p50": _quantile([float(row["union_overlap_ms"]) for row in group], 50),
        })
    return out


def _plot_slowdown(out: Path, summary: list[dict[str, Any]], title: str, xlabel: str) -> None:
    baseline = summary[0]
    labels = [str(row["bucket"]) for row in summary]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=170)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfbfd")
    width = 0.25
    for offset, key, color, label in [
        (-width, "p90", "#60a5fa", "p90 / zero-transfer p90"),
        (0.0, "p95", "#a78bfa", "p95 / zero-transfer p95"),
        (width, "p99", "#f87171", "p99 / zero-transfer p99"),
    ]:
        values = []
        for row in summary:
            denom = float(baseline[key])
            values.append(float(row[key]) / denom if denom and math.isfinite(denom) else float("nan"))
        ax.bar(x + offset, values, width=width, color=color, label=label)
    ax.axhline(1.0, color="#111827", lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("slowdown relative to zero-transfer bucket")
    ax.set_title(title, fontsize=13.0, pad=10)
    ax.grid(True, axis="y", color="#d1d5db", alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=8.5)
    total_n = sum(int(row["n"]) for row in summary)
    for xi, row in zip(x, summary):
        if int(row["n"]) > 0:
            pct = 100.0 * int(row["n"]) / total_n if total_n else float("nan")
            ax.text(
                xi,
                0.04,
                f"n={int(row['n']):,}\n{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#4b5563",
            )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")


def _plot_quantiles(out: Path, summary: list[dict[str, Any]], title: str, xlabel: str) -> None:
    labels = [str(row["bucket"]) for row in summary]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=170)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfbfd")
    for name, color in [("p50", "#111827"), ("p90", "#2563eb"), ("p95", "#7c3aed"), ("p99", "#dc2626")]:
        ax.plot(x, [float(row[name]) for row in summary], marker="o", lw=2.0, color=color, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("engine_step duration_ms")
    ax.set_title(title, fontsize=13.0, pad=10)
    ax.grid(True, axis="y", color="#d1d5db", alpha=0.55)
    ax.grid(True, axis="x", color="#e5e7eb", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")


def _report(path: Path, summary: list[dict[str, Any]]) -> None:
    lines = [
        "# Single-Run Transfer Volume Slowdown Report\n\n",
        "| bucket | n | population % | p50 ms | p90 ms | p95 ms | p99 ms | >=2 active % | estimated MB p50 | normalized MB/ms p50 |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    total_n = sum(int(row["n"]) for row in summary)
    for row in summary:
        pct = 100.0 * int(row["n"]) / total_n if total_n else float("nan")
        lines.append(
            f"| {row['bucket']} | {int(row['n'])} | {pct:.2f} | {float(row['p50']):.3f} | "
            f"{float(row['p90']):.3f} | {float(row['p95']):.3f} | {float(row['p99']):.3f} | "
            f"{float(row['multi_pct']):.1f} | {float(row['estimated_mb_p50']):.3f} | "
            f"{float(row['normalized_mb_per_ms_p50']):.3f} |\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlap-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    steps = _read_csv(args.overlap_dir / "overlap_engine_step_rows.csv")
    xfers = _read_csv(args.overlap_dir / "overlap_xfer_rows.csv")
    rows = []
    for step in steps:
        row: dict[str, Any] = dict(step)
        row.update(_pressure_for_step(step, xfers))
        rows.append(row)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "engine_step_pressure_rows.csv", rows)
    buckets = [
        ("0", 0.0, 0.0),
        ("(0, 0.5]", 0.0, 0.5),
        ("(0.5, 1]", 0.5, 1.0),
        ("(1, 2]", 1.0, 2.0),
        ("(2, 3]", 2.0, 3.0),
        ("(3, 4]", 3.0, 4.0),
        (">4", 4.0, float("inf")),
    ]
    summary = _summarize(rows, "normalized_transfer_MB_per_ms", buckets)
    _write_csv(args.out_dir / "engine_step_normalized_transfer_volume_bucket_summary.csv", summary)
    _plot_slowdown(
        args.out_dir / "engine_step_tail_slowdown_by_normalized_transfer_volume.png",
        summary,
        "Main Run: Tail Slowdown vs Normalized In-Step Transfer Volume",
        "estimated NIXL data per engine-step ms (MB/ms, equivalent to GB/s)",
    )
    _plot_quantiles(
        args.out_dir / "engine_step_tail_by_normalized_transfer_volume.png",
        summary,
        "Main Run: Engine-Step Tail vs Normalized In-Step Transfer Volume",
        "estimated NIXL data per engine-step ms (MB/ms, equivalent to GB/s)",
    )
    _report(args.out_dir / "single_run_transfer_volume_slowdown_report.md", summary)
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
