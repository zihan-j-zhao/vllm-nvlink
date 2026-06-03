#!/usr/bin/env python3
"""Build slowdown figures from e2e_agrs_nixl window-level overlap CSVs.

This script implements the final pressure metrics used for the report:

* estimated in-step transfer volume:
  sum(bytes_transferred * clipped_overlap_time / transfer_duration)
* normalized in-step transfer volume:
  estimated_MB / engine_step_duration_ms

Inputs are the CSVs produced by analyze_overlap.py with
--recv-interval-source nixl --nixl-start start_time.
"""

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

WINDOWS = [
    ("telemetry_5k_starttime_union_early_120", "early +120s"),
    ("telemetry_5k_starttime_union_mid_180", "mid +180s"),
    ("telemetry_5k_starttime_union_mid_240", "mid +240s"),
    ("telemetry_5k_starttime_union_mid_300", "mid +300s"),
    ("telemetry_5k_starttime_union_late_360", "late +360s"),
]
STEADY_WINDOWS = {"mid +240s", "mid +300s"}


def _float(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        return float(row[key])
    except Exception:
        return default


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


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
        rate_GBps = bytes_total / xfer_duration / 1e9
        events.append((lo, rate_GBps))
        events.append((hi, -rate_GBps))
        byte_rate_seconds += rate_GBps * (hi - lo)
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
    }


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


def _quantile(values: list[float], p: float) -> float:
    values = [v for v in values if math.isfinite(v)]
    if not values:
        return float("nan")
    return float(np.percentile(values, p))


def _summarize_buckets(
    rows: list[dict[str, Any]],
    metric: str,
    buckets: list[tuple[str, float, float]],
) -> list[dict[str, float | int | str]]:
    out: list[dict[str, float | int | str]] = []
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
            "multi_pct": (
                100.0 * sum(int(float(row["max_active_recv"])) >= 2 for row in group) / len(group)
                if group else float("nan")
            ),
            "estimated_mb_p50": _quantile([float(row["bytes_during_step_MB"]) for row in group], 50),
            "normalized_mb_per_ms_p50": _quantile(
                [float(row["normalized_transfer_MB_per_ms"]) for row in group], 50
            ),
            "union_overlap_ms_p50": _quantile([float(row["union_overlap_ms"]) for row in group], 50),
        })
    return out


def _plot_quantiles(
    out: Path,
    summary: list[dict[str, float | int | str]],
    title: str,
    xlabel: str,
    ylabel: str = "engine_step duration_ms",
) -> None:
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
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13.0, pad=10)
    ax.grid(True, axis="y", color="#d1d5db", alpha=0.55)
    ax.grid(True, axis="x", color="#e5e7eb", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92)
    for xi, row in zip(x, summary):
        ax.text(xi, 11.05, f"n={int(row['n']):,}", ha="center", va="bottom", fontsize=8, color="#4b5563")
    finite_p99 = [float(row["p99"]) for row in summary if math.isfinite(float(row["p99"]))]
    ax.set_ylim(10.8, max(30.0, max(finite_p99) * 1.08 if finite_p99 else 30.0))
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")


def _plot_slowdown(
    out: Path,
    summary: list[dict[str, float | int | str]],
    title: str,
    xlabel: str,
) -> None:
    baseline = summary[0]
    labels = [str(row["bucket"]) for row in summary]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=170)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfbfd")
    width = 0.25
    ax.bar(
        x - width,
        [float(row["p90"]) / float(baseline["p90"]) for row in summary],
        width=width,
        color="#60a5fa",
        label="p90 / zero-transfer p90",
    )
    ax.bar(
        x,
        [float(row["p95"]) / float(baseline["p95"]) for row in summary],
        width=width,
        color="#a78bfa",
        label="p95 / zero-transfer p95",
    )
    ax.bar(
        x + width,
        [float(row["p99"]) / float(baseline["p99"]) for row in summary],
        width=width,
        color="#f87171",
        label="p99 / zero-transfer p99",
    )
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
    for xi, row in zip(x, summary):
        ax.text(xi, 0.04, f"{float(row['multi_pct']):.0f}%\n>=2", ha="center", va="bottom", fontsize=7.5, color="#4b5563")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")


def _plot_survival(out: Path, rows: list[dict[str, Any]]) -> None:
    groups = [
        ("0 MB", [row for row in rows if float(row["bytes_during_step_MB"]) <= 1e-12], "#6b7280"),
        ("0-16 MB", [row for row in rows if 0 < float(row["bytes_during_step_MB"]) <= 16], "#2563eb"),
        ("16-32 MB", [row for row in rows if 16 < float(row["bytes_during_step_MB"]) <= 32], "#7c3aed"),
        (">32 MB", [row for row in rows if float(row["bytes_during_step_MB"]) > 32], "#dc2626"),
    ]
    fig, ax = plt.subplots(figsize=(8.7, 5.1), dpi=170)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfbfd")
    for label, group, color in groups:
        values = np.sort(np.array([float(row["duration_ms"]) for row in group], dtype=float))
        if len(values) == 0:
            continue
        survival = 1.0 - np.arange(1, len(values) + 1) / len(values)
        ax.step(values, survival, where="post", color=color, lw=2.0, label=f"{label} (n={len(values):,})")
    ax.set_yscale("log")
    ax.set_xlim(11.5, 32)
    ax.set_ylim(8e-4, 1.0)
    ax.set_xlabel("engine_step duration_ms")
    ax.set_ylabel("P(duration >= x)")
    ax.set_title("Steady Windows: Engine-Step Tail Survival by Estimated Transfer Volume", fontsize=13.0, pad=10)
    ax.grid(True, which="both", color="#d1d5db", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")


def _report(path: Path, raw: list[dict[str, float | int | str]], norm: list[dict[str, float | int | str]]) -> None:
    def table(title: str, rows: list[dict[str, float | int | str]]) -> list[str]:
        lines = [f"\n## {title}\n\n"]
        lines.append("| bucket | n | p50 ms | p90 ms | p95 ms | p99 ms | >=2 active % | estimated MB p50 | normalized MB/ms p50 |\n")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            lines.append(
                f"| {row['bucket']} | {int(row['n'])} | {float(row['p50']):.3f} | {float(row['p90']):.3f} | "
                f"{float(row['p95']):.3f} | {float(row['p99']):.3f} | {float(row['multi_pct']):.1f} | "
                f"{float(row['estimated_mb_p50']):.3f} | {float(row['normalized_mb_per_ms_p50']):.3f} |\n"
            )
        return lines

    lines = [
        "# E2E AGRS NIXL Slowdown Report\n\n",
        "This report uses per-engine-step NIXL transfer telemetry. Estimated transfer volume is prorated by the clipped overlap between each NIXL transfer interval and each engine-step interval.\n\n",
        "`estimated_MB = sum(bytes_transferred * clipped_overlap_time / transfer_duration) / 1e6`\n\n",
        "`normalized_MB_per_ms = estimated_MB / engine_step_duration_ms`\n\n",
        "The primary steady-state figures use only `mid +240s` and `mid +300s`.\n",
    ]
    lines.extend(table("Estimated Transfer Volume Buckets", raw))
    lines.extend(table("Normalized Transfer Volume Buckets", norm))
    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-root", type=Path, default=Path("playground/out/overlap/e2e_agrs_nixl"))
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or args.window_root / "telemetry_5k_starttime_union_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for dirname, label in WINDOWS:
        step_path = args.window_root / dirname / "overlap_engine_step_rows.csv"
        xfer_path = args.window_root / dirname / "overlap_xfer_rows.csv"
        if not step_path.exists() or not xfer_path.exists():
            print(f"warning: skipping missing window {dirname}")
            continue
        xfers = _read_csv(xfer_path)
        for step in _read_csv(step_path):
            row: dict[str, Any] = dict(step)
            row["window"] = label
            row.update(_pressure_for_step(step, xfers))
            duration_ms = _float(row, "duration_ms")
            row["normalized_transfer_MB_per_ms"] = row["bytes_during_step_MB"] / max(duration_ms, 1e-9)
            rows.append(row)

    if not rows:
        raise SystemExit(f"no rows found under {args.window_root}")

    _write_csv(out_dir / "engine_step_pressure_rows.csv", rows)
    steady = [row for row in rows if row["window"] in STEADY_WINDOWS]

    raw_buckets = [
        ("0", 0.0, 0.0),
        ("(0, 4]", 0.0, 4.0),
        ("(4, 8]", 4.0, 8.0),
        ("(8, 16]", 8.0, 16.0),
        ("(16, 32]", 16.0, 32.0),
        ("(32, 64]", 32.0, 64.0),
        (">64", 64.0, float("inf")),
    ]
    norm_buckets = [
        ("0", 0.0, 0.0),
        ("(0, 0.5]", 0.0, 0.5),
        ("(0.5, 1]", 0.5, 1.0),
        ("(1, 2]", 1.0, 2.0),
        ("(2, 3]", 2.0, 3.0),
        ("(3, 4]", 3.0, 4.0),
        (">4", 4.0, float("inf")),
    ]
    raw_summary = _summarize_buckets(steady, "bytes_during_step_MB", raw_buckets)
    norm_summary = _summarize_buckets(steady, "normalized_transfer_MB_per_ms", norm_buckets)
    _write_csv(out_dir / "engine_step_estimated_transfer_mb_bucket_summary.csv", raw_summary)
    _write_csv(out_dir / "engine_step_normalized_transfer_volume_bucket_summary.csv", norm_summary)

    _plot_quantiles(
        out_dir / "engine_step_tail_by_estimated_transfer_mb_steady.png",
        raw_summary,
        "Steady Windows: Engine-Step Tail vs Estimated In-Step Transfer Volume",
        "estimated NIXL data transferred during engine step (MB)",
    )
    _plot_slowdown(
        out_dir / "engine_step_tail_slowdown_by_estimated_transfer_mb_steady.png",
        raw_summary,
        "Steady Windows: Tail Slowdown vs Estimated In-Step Transfer Volume",
        "estimated NIXL data transferred during engine step (MB)",
    )
    _plot_survival(out_dir / "engine_step_tail_survival_by_estimated_transfer_mb_steady.png", steady)
    _plot_quantiles(
        out_dir / "engine_step_tail_by_normalized_transfer_volume_steady.png",
        norm_summary,
        "Steady Windows: Engine-Step Tail vs Normalized In-Step Transfer Volume",
        "estimated NIXL data per engine-step ms (MB/ms, equivalent to GB/s)",
    )
    _plot_slowdown(
        out_dir / "engine_step_tail_slowdown_by_normalized_transfer_volume_steady.png",
        norm_summary,
        "Steady Windows: Tail Slowdown vs Normalized In-Step Transfer Volume",
        "estimated NIXL data per engine-step ms (MB/ms, equivalent to GB/s)",
    )
    _report(out_dir / "e2e_agrs_nixl_slowdown_report.md", raw_summary, norm_summary)
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
