#!/usr/bin/env python3
"""Fast steady-state mean extractor for the 30B grid nsys sweeps.

Replaces the slow per-(GPU,metric) timeseries scans + matplotlib plotting
in postprocess_nsys.py with a single grouped ``AVG`` SQL scan per cell.
Writes a combined summary CSV (same schema aggregate_grid.py expects).

GPU role mapping (both servers): nsys relative GPU index 0,1 = prefill,
2,3 = decode.

Usage:
    python fast_summary.py <out_csv> <cell_dir> [<cell_dir> ...]
"""
from __future__ import annotations

import csv
import json
import re
import sqlite3
import sys
from pathlib import Path

PREFILL = {0, 1}
DECODE = {2, 3}


def _profiling_window(cell_dir: Path, session_start_ns: int) -> tuple[int, int] | None:
    """Return (lo, hi) in GPU_METRICS session-relative ns for aiperf's
    profiling phase, parsed from aiperf.log's PhaseRecordsStats line (epoch
    ns), or None if not determinable."""
    log = cell_dir / "aiperf.log"
    if not log.is_file():
        return None
    txt = log.read_text(errors="ignore")
    m = re.search(r"phase=CreditPhase\.PROFILING[^)]*", txt)
    if not m:
        return None
    seg = m.group(0)
    ms = re.search(r"start_ns=(\d+)", seg)
    if not ms:
        return None
    start = int(ms.group(1))
    end = None
    for key in ("requests_end_ns", "sent_end_ns", "records_end_ns"):
        me = re.search(rf"{key}=(\d+)", seg)
        if me:
            end = int(me.group(1))
            break
    if end is None or end <= start:
        return None
    return (start - session_start_ns, end - session_start_ns)


def process_cell(cell_dir: Path) -> list[dict]:
    cj = cell_dir / "cell.json"
    sq = cell_dir / "nsys_report.sqlite"
    if not cj.is_file() or not sq.is_file():
        return []
    meta = json.loads(cj.read_text())
    warmup_s = float(meta.get("warmup_s", 0))
    isl = meta.get("isl_mean")
    osl = meta.get("osl_mean")
    conc = meta.get("concurrency")
    tag = meta.get("cell_tag", cell_dir.name)

    conn = sqlite3.connect(str(sq))
    cur = conn.cursor()
    # metricId -> name
    names: dict[int, str] = {}
    cur.execute("SELECT DISTINCT metricId, metricName FROM TARGET_INFO_GPU_METRICS")
    for mid, mn in cur.fetchall():
        names[int(mid)] = str(mn)
    # typeId base + capture extent
    cur.execute("SELECT MIN(typeId), MIN(timestamp), MAX(timestamp) FROM GPU_METRICS")
    base_tid, t0, tmax = cur.fetchone()
    base_tid = int(base_tid)
    t0 = int(t0); tmax = int(tmax)
    # peak HBM (B200: identical across GPUs); take first
    peak_hbm_gbps = 0.0
    try:
        cur.execute("SELECT memoryBandwidth FROM TARGET_INFO_GPU LIMIT 1")
        r = cur.fetchone()
        if r and r[0]:
            peak_hbm_gbps = float(r[0]) / 1e9
    except sqlite3.OperationalError:
        pass

    # --- Choose the averaging window ------------------------------------
    # Prefer aiperf's *profiling* phase window (the measured run), mapped from
    # epoch ns to GPU_METRICS session time via TARGET_INFO_SESSION_START_TIME.
    # This excludes the warmup wave and the inter-wave idle. Fall back to a
    # data-driven active window (first..last SM>thr) if the profiling window
    # can't be parsed.
    sess_ns = None
    try:
        cur.execute("SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME LIMIT 1")
        r = cur.fetchone()
        if r and r[0]:
            sess_ns = int(r[0])
    except sqlite3.OperationalError:
        pass

    win_mode = "profiling"
    win = _profiling_window(cell_dir, sess_ns) if sess_ns is not None else None
    if win is not None:
        win_lo, win_hi = max(win[0], t0), min(win[1], tmax)
    if win is None or (win_hi - win_lo) < int(1 * 1e9):
        # Fallback: data-driven active window (first..last SM>thr).
        win_mode = "active"
        sm_mid = next((mid for mid, mn in names.items() if "SMs Active" in mn), None)
        win_lo, win_hi = t0, tmax
        if sm_mid is not None:
            cur.execute(
                "SELECT timestamp, MAX(value) FROM GPU_METRICS "
                "WHERE metricId=? GROUP BY timestamp ORDER BY timestamp",
                (sm_mid,),
            )
            active_ts = [int(ts) for ts, v in cur.fetchall() if v is not None and v > 2.0]
            if len(active_ts) >= 2:
                win_lo, win_hi = active_ts[0], active_ts[-1]
        if win_hi - win_lo < int(3 * 1e9):
            win_lo = t0 + int(warmup_s * 1e9)
            win_hi = tmax
    win_s = (win_hi - win_lo) / 1e9

    # Grouped scan: mean per (gpu, metric) over the steady-state window.
    cur.execute(
        "SELECT typeId, metricId, AVG(value) FROM GPU_METRICS "
        "WHERE timestamp BETWEEN ? AND ? GROUP BY typeId, metricId",
        (win_lo, win_hi),
    )
    # gpu_id -> {metricName: mean}
    per_gpu: dict[int, dict[str, float]] = {}
    for tid, mid, avg in cur.fetchall():
        gid = int(tid) - base_tid
        per_gpu.setdefault(gid, {})[names.get(int(mid), str(mid))] = float(avg)
    conn.close()

    rows = []
    for gid in sorted(per_gpu):
        role = "prefill" if gid in PREFILL else "decode" if gid in DECODE else "unknown"
        row = {
            "cell_tag": tag, "isl": isl, "osl": osl, "concurrency": conc,
            "gpu_id": gid, "role": role,
            "peak_hbm_gbps": f"{peak_hbm_gbps:.1f}",
            "window_mode": win_mode,
            "window_s": f"{win_s:.1f}",
            "capture_span_s": f"{(tmax - t0) / 1e9:.1f}",
        }
        for mn, v in per_gpu[gid].items():
            row[mn] = f"{v:.3f}"
        rows.append(row)
    return rows


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    out_csv = Path(sys.argv[1])
    cell_dirs = [Path(p) for p in sys.argv[2:]]

    all_rows: list[dict] = []
    metric_cols: list[str] = []
    for i, cd in enumerate(cell_dirs):
        rows = process_cell(cd)
        for r in rows:
            for k in r:
                if k not in ("cell_tag", "isl", "osl", "concurrency", "gpu_id",
                             "role", "peak_hbm_gbps", "window_mode", "window_s",
                             "capture_span_s") and k not in metric_cols:
                    metric_cols.append(k)
        all_rows.extend(rows)
        print(f"[fast_summary] {i+1}/{len(cell_dirs)} {cd.name}: {len(rows)} gpu-rows",
              flush=True)

    fieldnames = ["cell_tag", "isl", "osl", "concurrency", "gpu_id", "role",
                  "peak_hbm_gbps", "window_mode", "window_s", "capture_span_s",
                  *metric_cols]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"[fast_summary] wrote {len(all_rows)} rows -> {out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
