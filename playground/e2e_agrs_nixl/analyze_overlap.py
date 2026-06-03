#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Analyze high-level P/D overlap from the env-gated JSONL traces.

This intentionally avoids CUPTI/Nsight. It uses CPU wall-clock events emitted
outside CUDA graph capture to answer a service-level question:

    Do decode engine-step intervals that overlap active NIXL KV receives
    show higher latency / ITL than intervals without overlap?

Inputs: role=*.jsonl files written under VLLM_PD_TRACE_DIR.
Outputs:
  * overlap_summary.md
    * overlap_engine_step_rows.csv
    * overlap_forward_rows.csv
  * overlap_itl_rows.csv
  * overlap_timeline.png

Usage:
    python playground/e2e_agrs_nixl/analyze_overlap.py \
        --trace-dir playground/log/e2e_agrs_nixl/<TS>/pd_trace \
        [--out-dir playground/out/overlap/e2e_agrs_nixl/<TS>]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    print(f"error: matplotlib is required ({exc})", file=sys.stderr)
    raise

_FNAME_RE = re.compile(
    r"^role=(?P<role>[^_]+)_dp=(?P<dp>\d+)_tp=(?P<tp>\d+)_pid=(?P<pid>\d+)"
)


@dataclass
class ProcessTrace:
    path: Path
    role: str
    dp: int
    tp: int
    pid: int
    offset: float = 0.0
    events: list[dict[str, Any]] = field(default_factory=list)


def _parse_trace(path: Path) -> ProcessTrace:
    match = _FNAME_RE.match(path.name)
    if not match:
        raise ValueError(f"unparseable trace filename: {path.name}")
    trace = ProcessTrace(
        path=path,
        role=match.group("role"),
        dp=int(match.group("dp")),
        tp=int(match.group("tp")),
        pid=int(match.group("pid")),
    )
    raw_events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for raw in fp:
            raw = raw.strip()
            if not raw:
                continue
            try:
                ev = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if ev.get("ev") == "boot":
                trace.offset = float(ev["wall"]) - float(ev["ts"])
            raw_events.append(ev)
    for ev in raw_events:
        if "ts" in ev:
            ev = dict(ev)
            ev["ts"] = float(ev["ts"]) + trace.offset
        trace.events.append(ev)
    return trace


def _gather(trace_dir: Path) -> list[ProcessTrace]:
    files = sorted(trace_dir.glob("role=*.jsonl"))
    if not files:
        raise SystemExit(f"no trace files found under {trace_dir}")
    traces = []
    for path in files:
        try:
            traces.append(_parse_trace(path))
        except Exception as exc:
            print(f"warning: skipping {path}: {exc}", file=sys.stderr)
    return traces


def _pair_intervals(
    trace: ProcessTrace,
    start_ev: str,
    done_ev: str,
) -> list[dict[str, Any]]:
    pending: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for ev in trace.events:
        kind = ev.get("ev")
        if kind == start_ev:
            pending.append(ev)
        elif kind == done_ev and pending:
            start = pending.pop(0)
            row = dict(start)
            row.update({
                "start": start["ts"],
                "end": ev["ts"],
                "duration_ms": (ev["ts"] - start["ts"]) * 1000.0,
                "role": trace.role,
                "dp": trace.dp,
                "tp": trace.tp,
                "pid": trace.pid,
            })
            rows.append(row)
    return rows


def _recv_intervals(decode_traces: list[ProcessTrace]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace in decode_traces:
        pending_by_req: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for ev in trace.events:
            kind = ev.get("ev")
            if kind == "recv_start":
                pending_by_req[str(ev.get("req"))].append(ev)
            elif kind == "recv_done":
                req_id = str(ev.get("req"))
                if not pending_by_req.get(req_id):
                    continue
                start = pending_by_req[req_id].pop(0)
                rows.append({
                    "start": start["ts"],
                    "end": ev["ts"],
                    "duration_ms": (ev["ts"] - start["ts"]) * 1000.0,
                    "req": req_id,
                    "role": trace.role,
                    "dp": trace.dp,
                    "tp": trace.tp,
                    "pid": trace.pid,
                    "n_blocks": start.get("n_blocks"),
                    "remote_engine": start.get("remote_engine"),
                    "remote_rank": start.get("remote_rank"),
                })
    return rows


def _xfer_intervals(
    decode_traces: list[ProcessTrace],
    start_kind: str,
) -> list[dict[str, Any]]:
    def _float_or_nan(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    rows: list[dict[str, Any]] = []
    for trace in decode_traces:
        post_begin: dict[tuple[str, str], dict[str, Any]] = {}
        post_done: dict[tuple[str, str], dict[str, Any]] = {}
        for ev in trace.events:
            kind = ev.get("ev")
            if kind not in {"recv_post_begin", "recv_post_done", "recv_xfer_telemetry"}:
                continue
            req_id = str(ev.get("req"))
            handle_obj = ev.get("handle")
            if handle_obj is None:
                continue
            handle = str(handle_obj)
            key = (req_id, handle)
            if kind == "recv_post_begin":
                post_begin[key] = ev
            elif kind == "recv_post_done":
                post_done[key] = ev
            elif kind == "recv_xfer_telemetry":
                try:
                    xfer_duration_us = float(ev["xfer_duration_us"])
                except (KeyError, TypeError, ValueError):
                    continue
                xfer_duration_s = xfer_duration_us / 1e6
                begin_ev = post_begin.get(key)
                done_ev = post_done.get(key)
                start_time_us = _float_or_nan(ev.get("start_time_us"))
                start_time_raw_s = start_time_us / 1e6
                start_time_s = start_time_raw_s + trace.offset
                if start_kind == "telemetry_end":
                    start = ev["ts"] - xfer_duration_s
                elif start_kind == "start_time" and not np.isnan(start_time_s):
                    start = start_time_s
                elif start_kind == "post_begin" and begin_ev is not None:
                    start = begin_ev["ts"]
                elif done_ev is not None:
                    start = done_ev["ts"]
                elif begin_ev is not None:
                    start = begin_ev["ts"]
                else:
                    start = ev["ts"] - xfer_duration_s
                end = ev["ts"] if start_kind == "telemetry_end" else start + xfer_duration_s
                post_begin_ts = begin_ev.get("ts") if begin_ev is not None else None
                post_done_ts = done_ev.get("ts") if done_ev is not None else None
                post_cpu_ms = (
                    (post_done_ts - post_begin_ts) * 1000.0
                    if post_begin_ts is not None and post_done_ts is not None
                    else None
                )
                bytes_transferred = ev.get("bytes_transferred")
                bytes_float = _float_or_nan(bytes_transferred)
                post_duration_us = _float_or_nan(ev.get("post_duration_us"))
                throughput_gbps = (
                    bytes_float / xfer_duration_s / 1e9
                    if xfer_duration_s > 0 and not np.isnan(bytes_float)
                    else float("nan")
                )
                rows.append({
                    "start": start,
                    "end": end,
                    "duration_ms": xfer_duration_s * 1000.0,
                    "poll_ts": ev["ts"],
                    "poll_slack_ms": (ev["ts"] - end) * 1000.0,
                    "start_time_us": ev.get("start_time_us"),
                    "start_time_raw_s": start_time_raw_s,
                    "start_time_s": start_time_s,
                    "start_time_to_poll_ms": (
                        (ev["ts"] - start_time_s) * 1000.0
                        if not np.isnan(start_time_s)
                        else float("nan")
                    ),
                    "post_begin_to_start_time_ms": (
                        (start_time_s - post_begin_ts) * 1000.0
                        if post_begin_ts is not None and not np.isnan(start_time_s)
                        else float("nan")
                    ),
                    "post_cpu_ms": post_cpu_ms,
                    "post_duration_ms": post_duration_us / 1000.0,
                    "bytes_transferred": bytes_transferred,
                    "throughput_gbps": throughput_gbps,
                    "desc_count": ev.get("desc_count"),
                    "req": req_id,
                    "handle": handle,
                    "role": trace.role,
                    "dp": trace.dp,
                    "tp": trace.tp,
                    "pid": trace.pid,
                    "n_blocks": (done_ev or begin_ev or {}).get("n_blocks"),
                    "remote_engine": (done_ev or begin_ev or {}).get("remote_engine"),
                    "remote_rank": (done_ev or begin_ev or {}).get("remote_rank"),
                    "interval_source": f"nixl_{start_kind}",
                })
    rows.sort(key=lambda row: row["start"])
    return rows


def _overlap_ms(start: float, end: float, intervals: list[dict[str, Any]]) -> float:
    total = 0.0
    for interval in intervals:
        lo = max(start, interval["start"])
        hi = min(end, interval["end"])
        if hi > lo:
            total += hi - lo
    return total * 1000.0


def _union_overlap_ms(
    start: float,
    end: float,
    intervals: list[dict[str, Any]],
) -> float:
    spans: list[tuple[float, float]] = []
    for interval in intervals:
        lo = max(start, interval["start"])
        hi = min(end, interval["end"])
        if hi > lo:
            spans.append((lo, hi))
    if not spans:
        return 0.0
    spans.sort()
    total = 0.0
    cur_lo, cur_hi = spans[0]
    for lo, hi in spans[1:]:
        if lo <= cur_hi:
            cur_hi = max(cur_hi, hi)
        else:
            total += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
    total += cur_hi - cur_lo
    return total * 1000.0


def _max_active(start: float, end: float, intervals: list[dict[str, Any]]) -> int:
    points: list[tuple[float, int]] = []
    for interval in intervals:
        lo = max(start, interval["start"])
        hi = min(end, interval["end"])
        if hi > lo:
            points.append((lo, 1))
            points.append((hi, -1))
    active = 0
    best = 0
    for _, delta in sorted(points):
        active += delta
        best = max(best, active)
    return best


def _active_series(
    intervals: list[dict[str, Any]], t0: float, t1: float, bin_s: float
) -> tuple[np.ndarray, np.ndarray]:
    n_bins = max(1, int(np.ceil((t1 - t0) / bin_s)))
    edges = t0 + bin_s * np.arange(n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    active = np.zeros(n_bins, dtype=np.float64)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # Mean active transfers in this bin, using interval overlap time.
        acc = 0.0
        for interval in intervals:
            a = max(lo, interval["start"])
            b = min(hi, interval["end"])
            if b > a:
                acc += b - a
        active[i] = acc / bin_s
    return centers, active


def _itl_rows(decode_traces: list[ProcessTrace], recv_intervals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    per_req: dict[str, list[tuple[float, int, int]]] = defaultdict(list)
    for trace in decode_traces:
        for ev in trace.events:
            if ev.get("ev") != "step":
                continue
            for req_id, n_new in ev.get("tokens", []):
                if int(n_new) > 0:
                    per_req[str(req_id)].append((ev["ts"], int(n_new), trace.dp))
    rows: list[dict[str, Any]] = []
    for req_id, samples in per_req.items():
        samples.sort(key=lambda item: item[0])
        prev_ts: float | None = None
        for ts, n_new, dp in samples:
            if prev_ts is None:
                prev_ts = ts
                continue
            itl_ms = (ts - prev_ts) * 1000.0 / max(n_new, 1)
            overlap = _overlap_ms(prev_ts, ts, recv_intervals)
            union_overlap = _union_overlap_ms(prev_ts, ts, recv_intervals)
            duration_ms = max((ts - prev_ts) * 1000.0, 1e-9)
            rows.append({
                "req": req_id,
                "dp": dp,
                "start": prev_ts,
                "end": ts,
                "itl_ms": itl_ms,
                "overlap_ms": overlap,
                "overlap_ratio": overlap / duration_ms,
                "union_overlap_ms": union_overlap,
                "union_overlap_ratio": union_overlap / duration_ms,
                "overlap_intensity": overlap / max(union_overlap, 1e-9),
                "max_active_recv": _max_active(prev_ts, ts, recv_intervals),
            })
            prev_ts = ts
    return rows


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {k: float("nan") for k in ("n", "mean", "p50", "p90", "p95", "p99", "max")}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": float(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(arr.max()),
    }


def _fmt(value: float) -> str:
    if np.isnan(value):
        return "nan"
    if value == int(value):
        return str(int(value))
    return f"{value:.3f}"


def _group_summary(rows: list[dict[str, Any]], metric: str) -> list[tuple[str, dict[str, float]]]:
    groups = {
        "no_recv_overlap": [r[metric] for r in rows if r["overlap_ms"] <= 0.0],
        "recv_overlap": [r[metric] for r in rows if r["overlap_ms"] > 0.0],
        "max_active_recv_1": [r[metric] for r in rows if int(r["max_active_recv"]) == 1],
        "max_active_recv_ge2": [r[metric] for r in rows if int(r["max_active_recv"]) >= 2],
    }
    return [(name, _quantiles(vals)) for name, vals in groups.items()]


def _attach_overlap_fields(
    rows: list[dict[str, Any]],
    recv_intervals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    for row in rows:
        overlap = _overlap_ms(row["start"], row["end"], recv_intervals)
        union_overlap = _union_overlap_ms(row["start"], row["end"], recv_intervals)
        row["overlap_ms"] = overlap
        row["overlap_ratio"] = overlap / max(row["duration_ms"], 1e-9)
        row["union_overlap_ms"] = union_overlap
        row["union_overlap_ratio"] = union_overlap / max(row["duration_ms"], 1e-9)
        row["overlap_intensity"] = overlap / max(union_overlap, 1e-9)
        row["max_active_recv"] = _max_active(row["start"], row["end"], recv_intervals)
    return rows


def _phase_rows(
    decode_traces: list[ProcessTrace],
    recv_intervals: list[dict[str, Any]],
    phase: str,
    start_ev: str,
    done_ev: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace in decode_traces:
        for row in _pair_intervals(trace, start_ev, done_ev):
            row["phase"] = phase
            rows.append(row)
    rows.sort(key=lambda row: row["start"])
    return _attach_overlap_fields(rows, recv_intervals)


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


def _plot(
    out: Path,
    step_rows: list[dict[str, Any]],
    itl_rows: list[dict[str, Any]],
    recv_intervals: list[dict[str, Any]],
    t0: float,
    t1: float,
    bin_ms: float,
) -> None:
    bin_s = bin_ms / 1000.0
    centers, active = _active_series(recv_intervals, t0, t1, bin_s)
    fig, axes = plt.subplots(
        3, 1, figsize=(13, 8), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 2.0, 2.0]},
    )
    ax0, ax1, ax2 = axes
    ax0.step(centers - t0, active, where="mid", color="tab:purple")
    ax0.set_ylabel("active KV\nrecvs")
    ax0.grid(True, alpha=0.3)

    if step_rows:
        xs = np.asarray([r["start"] - t0 for r in step_rows])
        ys = np.asarray([r["duration_ms"] for r in step_rows])
        colors = np.asarray([r["max_active_recv"] for r in step_rows])
        sc = ax1.scatter(xs, ys, c=colors, cmap="viridis", s=14, alpha=0.75)
        fig.colorbar(sc, ax=ax1, label="max active KV recvs")
    ax1.set_ylabel("decode engine step\nduration (ms)")
    ax1.grid(True, alpha=0.3)

    if itl_rows:
        xs = np.asarray([r["end"] - t0 for r in itl_rows])
        ys = np.asarray([r["itl_ms"] for r in itl_rows])
        colors = np.asarray([r["max_active_recv"] for r in itl_rows])
        sc = ax2.scatter(xs, ys, c=colors, cmap="viridis", s=8, alpha=0.45)
        fig.colorbar(sc, ax=ax2, label="max active KV recvs")
    ax2.set_ylabel("ITL sample (ms)")
    ax2.set_xlabel("wall-clock time since trace window start (s)")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(0.001, t1 - t0))
    fig.suptitle("P/D overlap: KV receive activity vs decode engine steps and ITL")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--bin-ms", type=float, default=100.0)
    parser.add_argument(
        "--window-start-sec",
        type=float,
        default=None,
        help="Analyze a window starting this many seconds after the first "
             "decode trace event.",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=None,
        help="Analyze only this many seconds after --window-start-sec.",
    )
    parser.add_argument(
        "--no-itl",
        action="store_true",
        help="Skip per-request ITL reconstruction. Useful for fast window sweeps.",
    )
    parser.add_argument(
        "--recv-interval-source",
        choices=("auto", "software", "nixl"),
        default="auto",
        help="Intervals used for overlap: software recv_start/done, NIXL telemetry duration, or auto prefer NIXL when present.",
    )
    parser.add_argument(
        "--nixl-start",
        choices=("telemetry_end", "post_done", "post_begin", "start_time"),
        default="start_time",
        help="How to align NIXL xferDuration on the CPU timeline. telemetry_end uses [telemetry_ts - xferDuration, telemetry_ts]; start_time uses NIXL startTime if it shares the CPU clock domain.",
    )
    args = parser.parse_args()

    if args.trace_dir is None:
        candidates = sorted(glob.glob("playground/log/e2e_agrs_nixl/*/pd_trace"), key=os.path.getmtime)
        if not candidates:
            raise SystemExit("no --trace-dir and no playground/log/e2e_agrs_nixl/*/pd_trace found")
        args.trace_dir = Path(candidates[-1])

    traces = _gather(args.trace_dir)
    decode_traces = [trace for trace in traces if trace.role == "decode"]
    if not decode_traces:
        raise SystemExit("no decode traces found")
    trace_t0_candidates = [
        ev["ts"]
        for trace in decode_traces
        for ev in trace.events
        if ev.get("ev") != "boot" and "ts" in ev
    ]
    if not trace_t0_candidates:
        raise SystemExit("no timestamped decode events found")
    trace_t0 = min(trace_t0_candidates)
    if args.window_start_sec is not None or args.window_sec is not None:
        if args.window_start_sec is None or args.window_sec is None:
            raise SystemExit("--window-start-sec and --window-sec must be set together")
        win_lo = trace_t0 + args.window_start_sec
        win_hi = win_lo + args.window_sec
        for trace in decode_traces:
            trace.events = [
                ev for ev in trace.events
                if ev.get("ev") == "boot"
                or ("ts" in ev and win_lo <= ev["ts"] <= win_hi)
            ]
        print(
            f"[overlap] window: +{args.window_start_sec:.3f}s "
            f"to +{args.window_start_sec + args.window_sec:.3f}s "
            f"({args.window_sec:.3f}s)"
        )
    if args.out_dir is None:
        args.out_dir = Path("playground/out/overlap/e2e_agrs_nixl") / args.trace_dir.parent.name
    args.out_dir.mkdir(parents=True, exist_ok=True)

    software_recv_rows = _recv_intervals(decode_traces)
    xfer_rows = _xfer_intervals(decode_traces, args.nixl_start)
    if args.recv_interval_source == "nixl":
        recv_rows = xfer_rows
        if not recv_rows:
            raise SystemExit(
                "--recv-interval-source=nixl requested, but no recv_xfer_telemetry events found"
            )
    elif args.recv_interval_source == "auto" and xfer_rows:
        recv_rows = xfer_rows
    else:
        recv_rows = software_recv_rows
    interval_source = (
        "nixl" if recv_rows is xfer_rows else "software"
    )
    step_rows: list[dict[str, Any]] = []
    for trace in decode_traces:
        for row in _pair_intervals(trace, "engine_step_start", "engine_step_done"):
            step_rows.append(row)
    _attach_overlap_fields(step_rows, recv_rows)
    step_rows.sort(key=lambda row: row["start"])

    forward_rows: list[dict[str, Any]] = []
    for trace in decode_traces:
        for row in _pair_intervals(trace, "forward_start", "forward_done"):
            forward_rows.append(row)
    _attach_overlap_fields(forward_rows, recv_rows)
    forward_rows.sort(key=lambda row: row["start"])

    phase_specs = [
        ("engine_step_fn", "engine_step_start", "engine_step_after_step_fn"),
        ("engine_post_step", "engine_step_after_step_fn", "engine_step_done"),
        ("kv_start_load", "kv_start_load_begin", "kv_start_load_done"),
        ("kv_finalize", "kv_finalize_begin", "kv_finalize_done"),
        ("kv_wait_for_save", "kv_wait_for_save_begin", "kv_wait_for_save_done"),
        ("kv_get_finished", "kv_get_finished_begin", "kv_get_finished_done"),
        ("sample_tokens", "sample_tokens_start", "sample_tokens_done"),
        ("sample", "sample_start", "sample_done"),
        ("bookkeeping", "bookkeeping_start", "bookkeeping_done"),
    ]
    phase_rows: list[dict[str, Any]] = []
    for phase, start_ev, done_ev in phase_specs:
        phase_rows.extend(
            _phase_rows(decode_traces, recv_rows, phase, start_ev, done_ev)
        )
    phase_rows.sort(key=lambda row: (row["phase"], row["start"]))

    itl_rows = [] if args.no_itl else _itl_rows(decode_traces, recv_rows)
    itl_rows.sort(key=lambda row: row["end"])

    all_times = []
    for rows in (recv_rows, step_rows, forward_rows, phase_rows, itl_rows):
        for row in rows:
            all_times.append(row.get("start", row.get("end")))
            all_times.append(row.get("end", row.get("start")))
    if not all_times:
        raise SystemExit("no analyzable events found; did you restart server after instrumentation?")
    t0 = min(float(t) for t in all_times if t is not None)
    t1 = max(float(t) for t in all_times if t is not None)

    _write_csv(args.out_dir / "overlap_engine_step_rows.csv", step_rows)
    _write_csv(args.out_dir / "overlap_forward_rows.csv", forward_rows)
    _write_csv(args.out_dir / "overlap_phase_rows.csv", phase_rows)
    _write_csv(args.out_dir / "overlap_itl_rows.csv", itl_rows)
    _write_csv(args.out_dir / "overlap_recv_rows.csv", recv_rows)
    _write_csv(args.out_dir / "overlap_software_recv_rows.csv", software_recv_rows)
    _write_csv(args.out_dir / "overlap_xfer_rows.csv", xfer_rows)
    _plot(args.out_dir / "overlap_timeline.png", step_rows, itl_rows, recv_rows, t0, t1, args.bin_ms)

    lines = []
    lines.append(f"# P/D Overlap Summary\n")
    lines.append(f"trace_dir: `{args.trace_dir}`\n")
    lines.append(f"window_s: {_fmt(t1 - t0)}\n")
    lines.append(f"decode_trace_files: {len(decode_traces)}\n")
    lines.append(f"overlap_interval_source: {interval_source}\n")
    lines.append(f"kv_recv_intervals: {len(recv_rows)}\n")
    lines.append(f"software_recv_intervals: {len(software_recv_rows)}\n")
    lines.append(f"nixl_xfer_intervals: {len(xfer_rows)}\n")
    lines.append(f"decode_engine_step_intervals: {len(step_rows)}\n")
    lines.append(f"decode_forward_intervals: {len(forward_rows)}\n")
    lines.append(f"decode_phase_intervals: {len(phase_rows)}\n")
    lines.append(f"itl_samples: {len(itl_rows)}\n")

    lines.append("\n## Decode Engine Step Duration By KV-Receive Overlap\n")
    lines.append("| group | n | mean_ms | p50_ms | p90_ms | p95_ms | p99_ms | max_ms |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for name, stats in _group_summary(step_rows, "duration_ms"):
        lines.append(
            f"| {name} | {_fmt(stats['n'])} | {_fmt(stats['mean'])} | "
            f"{_fmt(stats['p50'])} | {_fmt(stats['p90'])} | {_fmt(stats['p95'])} | "
            f"{_fmt(stats['p99'])} | {_fmt(stats['max'])} |\n"
        )

    lines.append("\n## Decode Forward Launch-Scope Duration By KV-Receive Overlap\n")
    lines.append("These CPU intervals surround `_model_forward()` without CUDA sync; they are useful as launch-scope context, not full GPU execution time.\n\n")
    lines.append("| group | n | mean_ms | p50_ms | p90_ms | p95_ms | p99_ms | max_ms |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for name, stats in _group_summary(forward_rows, "duration_ms"):
        lines.append(
            f"| {name} | {_fmt(stats['n'])} | {_fmt(stats['mean'])} | "
            f"{_fmt(stats['p50'])} | {_fmt(stats['p90'])} | {_fmt(stats['p95'])} | "
            f"{_fmt(stats['p99'])} | {_fmt(stats['max'])} |\n"
        )

    lines.append("\n## Decode Subphase Duration By KV-Receive Overlap\n")
    lines.append("Rows are CPU wall-clock intervals. `sample` and `bookkeeping` were added for the steady-state follow-up run; older traces may have zero rows for those phases.\n\n")
    lines.append("| phase | group | n | mean_ms | p50_ms | p90_ms | p95_ms | p99_ms | max_ms |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for phase, _, _ in phase_specs:
        rows = [row for row in phase_rows if row["phase"] == phase]
        if not rows:
            continue
        for name, stats in _group_summary(rows, "duration_ms"):
            lines.append(
                f"| {phase} | {name} | {_fmt(stats['n'])} | {_fmt(stats['mean'])} | "
                f"{_fmt(stats['p50'])} | {_fmt(stats['p90'])} | {_fmt(stats['p95'])} | "
                f"{_fmt(stats['p99'])} | {_fmt(stats['max'])} |\n"
            )

    lines.append("\n## ITL By KV-Receive Overlap\n")
    lines.append("| group | n | mean_ms | p50_ms | p90_ms | p95_ms | p99_ms | max_ms |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for name, stats in _group_summary(itl_rows, "itl_ms"):
        lines.append(
            f"| {name} | {_fmt(stats['n'])} | {_fmt(stats['mean'])} | "
            f"{_fmt(stats['p50'])} | {_fmt(stats['p90'])} | {_fmt(stats['p95'])} | "
            f"{_fmt(stats['p99'])} | {_fmt(stats['max'])} |\n"
        )

    recv_stats = _quantiles([row["duration_ms"] for row in recv_rows])
    lines.append("\n## KV Receive / Transfer Duration\n")
    lines.append(
        f"n={_fmt(recv_stats['n'])}, mean={_fmt(recv_stats['mean'])} ms, "
        f"p50={_fmt(recv_stats['p50'])} ms, p90={_fmt(recv_stats['p90'])} ms, "
        f"p99={_fmt(recv_stats['p99'])} ms, max={_fmt(recv_stats['max'])} ms\n"
    )
    if xfer_rows:
        xfer_stats = _quantiles([row["duration_ms"] for row in xfer_rows])
        throughput_stats = _quantiles([
            row["throughput_gbps"] for row in xfer_rows
            if not np.isnan(float(row["throughput_gbps"]))
        ])
        slack_stats = _quantiles([row["poll_slack_ms"] for row in xfer_rows])
        lines.append("\n## NIXL Transfer Telemetry\n")
        lines.append(
            f"xfer_duration_ms: n={_fmt(xfer_stats['n'])}, mean={_fmt(xfer_stats['mean'])}, "
            f"p50={_fmt(xfer_stats['p50'])}, p90={_fmt(xfer_stats['p90'])}, "
            f"p99={_fmt(xfer_stats['p99'])}, max={_fmt(xfer_stats['max'])}\n"
        )
        lines.append(
            f"throughput_GBps: mean={_fmt(throughput_stats['mean'])}, "
            f"p50={_fmt(throughput_stats['p50'])}, p90={_fmt(throughput_stats['p90'])}, "
            f"p99={_fmt(throughput_stats['p99'])}\n"
        )
        lines.append(
            f"poll_slack_ms: mean={_fmt(slack_stats['mean'])}, p50={_fmt(slack_stats['p50'])}, "
            f"p90={_fmt(slack_stats['p90'])}, p99={_fmt(slack_stats['p99'])}\n"
        )

    summary = "".join(lines)
    (args.out_dir / "overlap_summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"[overlap] wrote {args.out_dir / 'overlap_timeline.png'}")
    print(f"[overlap] wrote {args.out_dir / 'overlap_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
