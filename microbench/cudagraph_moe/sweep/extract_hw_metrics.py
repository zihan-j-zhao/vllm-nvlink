#!/usr/bin/env python3
"""Extract HBM and NVLink utilization from an nsys-exported SQLite DB.

Usage:
    nsys export -t sqlite --force-overwrite=true \\
        -o report.sqlite report.nsys-rep
    python extract_hw_metrics.py report.sqlite
    python extract_hw_metrics.py report.sqlite --filter-range iter_
    python extract_hw_metrics.py report.sqlite --dump-samples

Schema notes (nsys 2024+ exports)
---------------------------------
GPU_METRICS rows: (timestamp, typeId, metricId, value)
  - typeId identifies a (device, metric-group) bucket
  - metricId is the field index inside that group

Human-readable metric names live in GENERIC_EVENT_TYPE_FIELDS, joined by
(typeId, fieldIdx == metricId) -> fieldNameId -> StringIds.value.

NVTX_EVENTS holds explicit-text NVTX ranges (`text` column is non-NULL),
with start/end nanoseconds on the same clock as GPU_METRICS.timestamp.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path


# Substring match (case-insensitive) for metrics we care about.
METRIC_NAME_PATTERNS = [
    "DRAM",
    "NVLINK",
    "PCIE",
    "SMS ACTIVE",
    "TENSOR",
    "L2",
    "WARPS IN FLIGHT",
]


def _matches(name: str) -> bool:
    n = name.upper()
    return any(p in n for p in METRIC_NAME_PATTERNS)


def list_metric_fields(
    conn: sqlite3.Connection,
) -> dict[tuple[int, int], str]:
    """Return {(typeId, fieldIdx): metric_name} for matching metrics.

    Only includes metrics that actually appear in GPU_METRICS.
    """
    cur = conn.execute(
        """
        SELECT DISTINCT f.typeId, f.fieldIdx, s.value
        FROM GENERIC_EVENT_TYPE_FIELDS f
        JOIN StringIds s ON f.fieldNameId = s.id
        WHERE f.typeId IN (SELECT DISTINCT typeId FROM GPU_METRICS)
        """
    )
    return {(tid, fidx): name for tid, fidx, name in cur if _matches(name)}


def list_typeid_devices(conn: sqlite3.Connection) -> dict[int, str]:
    """Best-effort: label each typeId with the source name (usually 'GpuMetrics').

    Different GPUs show up as different typeIds; we include the typeId itself
    in the output rows so multi-GPU runs are distinguishable.
    """
    cur = conn.execute(
        """
        SELECT t.typeId, s.value
        FROM GENERIC_EVENT_TYPES t
        JOIN GENERIC_EVENT_SOURCES src ON t.sourceId = src.sourceId
        JOIN StringIds s ON src.nameId = s.id
        WHERE t.typeId IN (SELECT DISTINCT typeId FROM GPU_METRICS)
        """
    )
    return {tid: src for tid, src in cur}


def list_nvtx_ranges(
    conn: sqlite3.Connection,
    text_filter: str | None,
) -> list[tuple[str, int, int]]:
    """Explicit-text NVTX ranges (`text` non-NULL, `end` non-NULL)."""
    sql = (
        "SELECT text, start, end FROM NVTX_EVENTS "
        "WHERE text IS NOT NULL AND end IS NOT NULL"
    )
    params: list[str] = []
    if text_filter:
        sql += " AND text LIKE ?"
        params.append(f"%{text_filter}%")
    sql += " ORDER BY start"
    return list(conn.execute(sql, params))


def aggregate(
    conn: sqlite3.Connection,
    metric_fields: dict[tuple[int, int], str],
    type_labels: dict[int, str],
    ranges: list[tuple[str, int, int]],
) -> list[dict[str, object]]:
    """For each (range, typeId, metric_name): compute count/mean/min/max/sum."""
    if not metric_fields or not ranges:
        return []
    fields_by_type: dict[int, dict[int, str]] = defaultdict(dict)
    for (tid, fidx), name in metric_fields.items():
        fields_by_type[tid][fidx] = name

    rows: list[dict[str, object]] = []
    for text, start_ns, end_ns in ranges:
        cur = conn.execute(
            """
            SELECT typeId, metricId, value
            FROM GPU_METRICS
            WHERE timestamp >= ? AND timestamp <= ?
            """,
            [start_ns, end_ns],
        )
        bucket: dict[tuple[int, int], list[float]] = defaultdict(list)
        for tid, fidx, val in cur:
            if fidx in fields_by_type.get(tid, ()):
                bucket[(tid, fidx)].append(float(val))
        for (tid, fidx), vals in bucket.items():
            name = fields_by_type[tid][fidx]
            rows.append(
                {
                    "nvtx_range": text,
                    "typeId": tid,
                    "source": type_labels.get(tid, ""),
                    "metric": name,
                    "count": len(vals),
                    "min": min(vals),
                    "mean": sum(vals) / len(vals),
                    "max": max(vals),
                    "sum": sum(vals),
                    "duration_ns": end_ns - start_ns,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    cols = [
        "nvtx_range", "typeId", "source", "metric",
        "count", "min", "mean", "max", "sum", "duration_ns",
    ]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in sorted(
            rows, key=lambda r: (r["nvtx_range"], r["typeId"], r["metric"])
        ):
            fmt = dict(r)
            for k in ("min", "mean", "max", "sum"):
                fmt[k] = f"{float(r[k]):.3f}"
            w.writerow(fmt)


def write_samples_csv(
    path: Path,
    conn: sqlite3.Connection,
    metric_fields: dict[tuple[int, int], str],
) -> None:
    """Dump raw per-sample stream for the metrics we tracked."""
    fields_by_type: dict[int, dict[int, str]] = defaultdict(dict)
    for (tid, fidx), name in metric_fields.items():
        fields_by_type[tid][fidx] = name

    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp_ns", "typeId", "metric", "value"])
        cur = conn.execute(
            "SELECT timestamp, typeId, metricId, value FROM GPU_METRICS "
            "ORDER BY timestamp"
        )
        for ts, tid, fidx, val in cur:
            name = fields_by_type.get(tid, {}).get(fidx)
            if name is not None:
                w.writerow([ts, tid, name, val])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite", type=Path, help="report.sqlite from `nsys export`")
    ap.add_argument(
        "--out", type=Path, default=None,
        help="Per-range CSV path (default: <sqlite>.metrics.csv).",
    )
    ap.add_argument(
        "--dump-samples", action="store_true",
        help="Also write raw per-sample stream as <sqlite>.samples.csv.",
    )
    ap.add_argument(
        "--filter-range", default=None,
        help="Only aggregate NVTX ranges whose text contains this substring.",
    )
    ap.add_argument(
        "--list-metrics", action="store_true",
        help="Print every metric name found in the DB and exit.",
    )
    args = ap.parse_args()

    if not args.sqlite.exists():
        print(f"no such file: {args.sqlite}", file=sys.stderr)
        return 1

    out_csv = args.out or args.sqlite.with_suffix(".metrics.csv")
    with sqlite3.connect(args.sqlite) as conn:
        if args.list_metrics:
            cur = conn.execute(
                """
                SELECT DISTINCT s.value
                FROM GENERIC_EVENT_TYPE_FIELDS f
                JOIN StringIds s ON f.fieldNameId = s.id
                WHERE f.typeId IN (SELECT DISTINCT typeId FROM GPU_METRICS)
                ORDER BY s.value
                """
            )
            for (name,) in cur:
                print(name)
            return 0

        metric_fields = list_metric_fields(conn)
        if not metric_fields:
            print(
                "No matching metrics in GPU_METRICS. "
                "Run with --list-metrics to see what nsys actually recorded "
                "and adjust METRIC_NAME_PATTERNS at the top of this script.",
                file=sys.stderr,
            )
            return 2

        type_labels = list_typeid_devices(conn)
        print(
            f"[metrics] {len(metric_fields)} tracked metric/typeId pairs "
            f"across {len({tid for tid, _ in metric_fields})} GPU sources"
        )
        for name in sorted(set(metric_fields.values())):
            print(f"    {name}")

        ranges = list_nvtx_ranges(conn, args.filter_range)
        print(f"[nvtx] aggregating across {len(ranges)} ranges")

        rows = aggregate(conn, metric_fields, type_labels, ranges)
        write_csv(out_csv, rows)
        print(f"[wrote] {out_csv}  ({len(rows)} rows)")

        if args.dump_samples:
            samples_csv = args.sqlite.with_suffix(".samples.csv")
            write_samples_csv(samples_csv, conn, metric_fields)
            print(f"[wrote] {samples_csv}  (raw per-sample stream)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
