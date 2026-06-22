"""Per-iter HW distribution + kernel duration stats for a sweep cell.

Two reports produced from one nsys-exported SQLite DB:

  - ``<sqlite>.iter.csv`` — one row per (iter, metric) with min/p50/p90/p99/max
    of the GPU-metric samples that fell inside that iter's NVTX range. At
    1 kHz this is intentionally a small sample set (a single decode iter
    is typically 3-30 ms, so 3-30 samples per iter per metric).

  - ``<sqlite>.kernels.csv`` — duration stats for a small allow-list of
    "hot" kernel name substrings (FMHA, allreduce-fusion, fused_moe).
    Per-kernel HW counters can't be reliably attached at 1 kHz, so we
    only report duration distributions and counts.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path


# Metrics whose distribution we care about per iter.  Substring match,
# case-insensitive.  Keep the list short so the CSV stays readable.
PER_ITER_METRIC_PATTERNS = [
    "DRAM READ BANDWIDTH",
    "DRAM WRITE BANDWIDTH",
    "SMS ACTIVE",
    "TENSOR ACTIVE",
    "NVLINK RX REQUESTS USER DATA",
    "NVLINK TX REQUESTS USER DATA",
]

# Kernel-name substrings we want duration stats for.  Match against the
# demangled kernel name in CUPTI_ACTIVITY_KIND_KERNEL.
KERNEL_NAME_PATTERNS = {
    "fmha":       "fmhaSm",
    "allreduce":  "allreduce_fusion_kernel",
    "fused_moe":  "fused_moe_kernel",
}


def _matches(name: str, patterns: list[str]) -> bool:
    n = name.upper()
    return any(p in n for p in patterns)


# ---------------------------------------------------------------------------
# Per-iter HW distribution
# ---------------------------------------------------------------------------
def list_metric_fields(
    conn: sqlite3.Connection, patterns: list[str],
) -> dict[tuple[int, int], str]:
    cur = conn.execute(
        """
        SELECT DISTINCT f.typeId, f.fieldIdx, s.value
        FROM GENERIC_EVENT_TYPE_FIELDS f
        JOIN StringIds s ON f.fieldNameId = s.id
        WHERE f.typeId IN (SELECT DISTINCT typeId FROM GPU_METRICS)
        """
    )
    return {(tid, fidx): name for tid, fidx, name in cur if _matches(name, patterns)}


def list_iter_ranges(
    conn: sqlite3.Connection, name_prefix: str = "iter_",
) -> list[tuple[str, int, int]]:
    cur = conn.execute(
        "SELECT text, start, end FROM NVTX_EVENTS "
        "WHERE text LIKE ? AND end IS NOT NULL ORDER BY start",
        [f"{name_prefix}%"],
    )
    return list(cur)


def per_iter_distribution(
    conn: sqlite3.Connection,
    metric_fields: dict[tuple[int, int], str],
    ranges: list[tuple[str, int, int]],
) -> list[dict[str, object]]:
    if not metric_fields or not ranges:
        return []
    fields_by_type: dict[int, dict[int, str]] = defaultdict(dict)
    for (tid, fidx), name in metric_fields.items():
        fields_by_type[tid][fidx] = name

    # Stable GPU index: typeIds in GPU_METRICS are integers; sort them
    # ascending and assign 0..N-1. This matches nsys-ui's "GPU N" labels.
    sorted_typeids = sorted(fields_by_type.keys())
    typeid_to_gpu = {tid: i for i, tid in enumerate(sorted_typeids)}

    rows: list[dict[str, object]] = []
    for text, start_ns, end_ns in ranges:
        # Pull metric samples inside this iter's window.
        cur = conn.execute(
            "SELECT typeId, metricId, value FROM GPU_METRICS "
            "WHERE timestamp >= ? AND timestamp <= ?",
            [start_ns, end_ns],
        )
        bucket: dict[tuple[int, int], list[float]] = defaultdict(list)
        for tid, fidx, val in cur:
            if fidx in fields_by_type.get(tid, ()):
                bucket[(tid, fidx)].append(float(val))
        iter_dur_ms = (end_ns - start_ns) / 1e6
        for (tid, fidx), vals in bucket.items():
            name = fields_by_type[tid][fidx]
            vals_sorted = sorted(vals)
            rows.append({
                "iter": text,
                "gpu": typeid_to_gpu[tid],
                "typeId": tid,
                "metric": name,
                "samples": len(vals),
                "iter_dur_ms": round(iter_dur_ms, 3),
                "min": min(vals),
                "p50": vals_sorted[len(vals_sorted) // 2],
                "p90": vals_sorted[min(len(vals_sorted) - 1,
                                       max(0, round(0.90 * (len(vals_sorted) - 1))))],
                "p99": vals_sorted[min(len(vals_sorted) - 1,
                                       max(0, round(0.99 * (len(vals_sorted) - 1))))],
                "max": max(vals),
                "mean": sum(vals) / len(vals),
            })
    return rows


# ---------------------------------------------------------------------------
# Kernel duration stats
# ---------------------------------------------------------------------------
def list_kernel_durations(
    conn: sqlite3.Connection, window_start: int, window_end: int,
) -> list[dict[str, object]]:
    """Find kernels matching our allow-list inside [start, end] ns and
    collect their duration distribution.
    """
    rows: list[dict[str, object]] = []
    for label, substr in KERNEL_NAME_PATTERNS.items():
        cur = conn.execute(
            """
            SELECT s.value, (k.end - k.start) AS dur_ns
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.demangledName = s.id
            WHERE k.start >= ? AND k.end <= ? AND s.value LIKE ?
            """,
            [window_start, window_end, f"%{substr}%"],
        )
        durations = [int(d) for _, d in cur]
        if not durations:
            continue
        d_us = [d / 1e3 for d in durations]
        d_sorted = sorted(d_us)

        # Also capture an example name (truncated) for traceability.
        cur2 = conn.execute(
            """
            SELECT s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.demangledName = s.id
            WHERE k.start >= ? AND k.end <= ? AND s.value LIKE ? LIMIT 1
            """,
            [window_start, window_end, f"%{substr}%"],
        )
        example = (cur2.fetchone() or ("",))[0]

        rows.append({
            "kernel_label": label,
            "match_substr": substr,
            "example_name": example[:120],
            "count": len(d_us),
            "total_us": round(sum(d_us), 1),
            "mean_us": round(statistics.fmean(d_us), 3),
            "min_us": round(min(d_us), 3),
            "p50_us": round(d_sorted[len(d_sorted) // 2], 3),
            "p90_us": round(d_sorted[max(0, round(0.90 * (len(d_sorted) - 1)))], 3),
            "p99_us": round(d_sorted[max(0, round(0.99 * (len(d_sorted) - 1)))], 3),
            "max_us": round(max(d_us), 3),
        })
    return rows


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------
def write_iter_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    cols = ["iter", "gpu", "typeId", "metric", "samples", "iter_dur_ms",
            "min", "p50", "p90", "p99", "max", "mean"]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["iter"], r["gpu"], r["metric"])):
            w.writerow({k: r[k] for k in cols})


def write_kernel_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    cols = ["kernel_label", "match_substr", "example_name", "count",
            "total_us", "mean_us", "min_us", "p50_us", "p90_us",
            "p99_us", "max_us"]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["kernel_label"]):
            w.writerow(r)


# ---------------------------------------------------------------------------
# Per-kernel HW distribution
# ---------------------------------------------------------------------------
# IMPORTANT CAVEAT:
#   At 1 kHz sampling each PM sample integrates over ~1 ms while most
#   kernels we track are 5–168 us. A sample whose timestamp falls inside
#   a kernel's [start, end] really reflects ~1 ms of GPU activity, of
#   which only a fraction was that kernel. For long kernels (FMHA at
#   B=1024p2048 = 168 us) the bias is small; for short kernels
#   (allreduce = 11 us) the number reflects mostly neighbouring kernels
#   on the decode hot path. Use these numbers as *trends*, not exact
#   per-kernel attributions.  For exact per-kernel HW counters use
#   Nsight Compute (`ncu`).

def per_kernel_hw_distribution(
    conn: sqlite3.Connection,
    metric_fields: dict[tuple[int, int], str],
    window_start: int,
    window_end: int,
) -> list[dict[str, object]]:
    """For each tracked kernel name x metric, distribution of samples
    whose timestamp fell inside any invocation of that kernel.

    Returns rows with the same shape as per_iter_distribution() but keyed
    by (kernel_label, gpu, metric) instead of (iter, gpu, metric).
    """
    if not metric_fields:
        return []
    fields_by_type: dict[int, dict[int, str]] = defaultdict(dict)
    for (tid, fidx), name in metric_fields.items():
        fields_by_type[tid][fidx] = name
    sorted_typeids = sorted(fields_by_type.keys())
    typeid_to_gpu = {tid: i for i, tid in enumerate(sorted_typeids)}

    rows: list[dict[str, object]] = []
    for label, substr in KERNEL_NAME_PATTERNS.items():
        # Pull all (start, end) for matching kernel invocations in window.
        cur = conn.execute(
            """
            SELECT k.start, k.end
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.demangledName = s.id
            WHERE k.start >= ? AND k.end <= ? AND s.value LIKE ?
            ORDER BY k.start
            """,
            [window_start, window_end, f"%{substr}%"],
        )
        intervals = [(int(a), int(b)) for a, b in cur]
        if not intervals:
            continue
        n_invocations = len(intervals)

        # Bucket samples by (gpu, metric). We use a single GPU_METRICS
        # scan and a sorted-interval bisect for the membership test —
        # naive O(N*M) joins are slow when both sides are tens of thousands.
        # `intervals` is sorted by start; we also keep an array of ends
        # for the bisect.
        starts = [a for a, _ in intervals]
        ends = [b for _, b in intervals]

        import bisect
        bucket: dict[tuple[int, int], list[float]] = defaultdict(list)
        cur = conn.execute(
            "SELECT timestamp, typeId, metricId, value FROM GPU_METRICS"
            " WHERE timestamp >= ? AND timestamp <= ?",
            [window_start, window_end],
        )
        for ts, tid, fidx, val in cur:
            if fidx not in fields_by_type.get(tid, ()):
                continue
            # Find first interval with start > ts; the candidate kernel
            # is the one before it (if any).
            idx = bisect.bisect_right(starts, ts) - 1
            if idx >= 0 and ends[idx] >= ts:
                bucket[(tid, fidx)].append(float(val))

        for (tid, fidx), vals in bucket.items():
            vals_sorted = sorted(vals)
            n = len(vals_sorted)
            rows.append({
                "kernel_label": label,
                "gpu": typeid_to_gpu[tid],
                "typeId": tid,
                "metric": fields_by_type[tid][fidx],
                "invocations": n_invocations,
                "samples": n,
                "min": min(vals_sorted),
                "p50": vals_sorted[n // 2],
                "p90": vals_sorted[min(n - 1, max(0, round(0.90 * (n - 1))))],
                "p99": vals_sorted[min(n - 1, max(0, round(0.99 * (n - 1))))],
                "max": max(vals_sorted),
                "mean": sum(vals_sorted) / n,
            })
    return rows


def write_kernel_hw_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    cols = ["kernel_label", "gpu", "typeId", "metric", "invocations",
            "samples", "min", "p50", "p90", "p99", "max", "mean"]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["kernel_label"], r["gpu"], r["metric"])):
            w.writerow({k: r[k] for k in cols})


def get_bench_window(conn: sqlite3.Connection) -> tuple[int, int] | None:
    cur = conn.execute(
        "SELECT MIN(start), MAX(end) FROM NVTX_EVENTS "
        "WHERE text LIKE 'realistic_bench%' AND end IS NOT NULL"
    )
    row = cur.fetchone()
    if row is None or row[0] is None:
        return None
    return int(row[0]), int(row[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite", type=Path)
    ap.add_argument(
        "--iter-out", type=Path, default=None,
        help="Path for per-iter CSV (default: <sqlite>.iter.csv).",
    )
    ap.add_argument(
        "--kernel-out", type=Path, default=None,
        help="Path for kernel-stats CSV (default: <sqlite>.kernels.csv).",
    )
    ap.add_argument(
        "--kernel-hw-out", type=Path, default=None,
        help="Path for per-kernel HW distribution CSV "
             "(default: <sqlite>.kernel_hw.csv).",
    )
    args = ap.parse_args()

    if not args.sqlite.exists():
        print(f"no such file: {args.sqlite}", file=sys.stderr)
        return 1

    iter_csv = args.iter_out or args.sqlite.with_suffix(".iter.csv")
    kernel_csv = args.kernel_out or args.sqlite.with_suffix(".kernels.csv")
    kernel_hw_csv = args.kernel_hw_out or args.sqlite.with_suffix(".kernel_hw.csv")

    with sqlite3.connect(args.sqlite) as conn:
        metric_fields = list_metric_fields(conn, PER_ITER_METRIC_PATTERNS)
        ranges = list_iter_ranges(conn)
        iter_rows = per_iter_distribution(conn, metric_fields, ranges)
        write_iter_csv(iter_csv, iter_rows)
        print(f"[wrote] {iter_csv}  ({len(iter_rows)} rows, "
              f"{len(ranges)} iters, {len(metric_fields)} metric/typeId pairs)")

        window = get_bench_window(conn)
        if window is None:
            print("[warn] no 'realistic_bench' NVTX range found; "
                  "kernel report empty")
            write_kernel_csv(kernel_csv, [])
            write_kernel_hw_csv(kernel_hw_csv, [])
        else:
            kernel_rows = list_kernel_durations(conn, *window)
            write_kernel_csv(kernel_csv, kernel_rows)
            print(f"[wrote] {kernel_csv}  ({len(kernel_rows)} kernel labels)")

            khw_rows = per_kernel_hw_distribution(conn, metric_fields, *window)
            write_kernel_hw_csv(kernel_hw_csv, khw_rows)
            print(f"[wrote] {kernel_hw_csv}  ({len(khw_rows)} rows)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
