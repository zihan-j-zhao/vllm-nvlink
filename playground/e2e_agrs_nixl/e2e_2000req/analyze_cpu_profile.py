#!/usr/bin/env python3
"""Analyze the e2e_2000req CPU/JSONL profiling run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"matplotlib is required: {exc}")

_FNAME_RE = re.compile(
    r"^role=(?P<role>[^_]+)_dp=(?P<dp>\d+)_tp=(?P<tp>\d+)_pid=(?P<pid>\d+)"
)
_REQ_BASE_RE = re.compile(
    r"^(?P<base>chatcmpl-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})-[0-9a-fA-F]{8}$"
)


@dataclass
class Trace:
    path: Path
    role: str
    dp: int
    tp: int
    pid: int
    offset: float = 0.0
    events: list[dict[str, Any]] = field(default_factory=list)


def _req_base(req_id: str) -> str:
    match = _REQ_BASE_RE.match(req_id)
    return match.group("base") if match else req_id


def _parse_trace(path: Path) -> Trace:
    match = _FNAME_RE.match(path.name)
    if not match:
        raise ValueError(f"unparseable trace filename: {path.name}")
    trace = Trace(
        path=path,
        role=match.group("role"),
        dp=int(match.group("dp")),
        tp=int(match.group("tp")),
        pid=int(match.group("pid")),
    )
    raw_events = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if event.get("ev") == "boot" and "wall" in event and "ts" in event:
                trace.offset = float(event["wall"]) - float(event["ts"])
            raw_events.append(event)
    for event in raw_events:
        if "ts" in event:
            event = dict(event)
            event["ts"] = float(event["ts"]) + trace.offset
        trace.events.append(event)
    return trace


def _load_traces(trace_dir: Path) -> list[Trace]:
    traces = []
    for path in sorted(trace_dir.glob("role=*.jsonl")):
        traces.append(_parse_trace(path))
    if not traces:
        raise SystemExit(f"no trace files found under {trace_dir}")
    return traces


def _load_markers(path: Path) -> tuple[float, float]:
    markers: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            row = json.loads(raw)
            markers[row["phase"]] = float(row["time_epoch_s"])
    if "main_start" not in markers or "main_done" not in markers:
        raise SystemExit(f"phase markers missing main_start/main_done in {path}")
    return markers["main_start"], markers["main_done"]


def _in_window(event: dict[str, Any], start: float, end: float) -> bool:
    ts = event.get("ts")
    return isinstance(ts, float) and start <= ts <= end


def _pair_intervals(events: list[dict[str, Any]], start_ev: str, done_ev: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pending: list[dict[str, Any]] = []
    pairs = []
    for event in events:
        if event.get("ev") == start_ev and "ts" in event:
            pending.append(event)
        elif event.get("ev") == done_ev and "ts" in event and pending:
            pairs.append((pending.pop(0), event))
    return pairs


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round((len(values) - 1) * pct / 100.0)))
    return values[idx]


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "avg": float("nan"), "p50": float("nan"), "p90": float("nan"), "p95": float("nan"), "p99": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "count": len(values),
        "avg": sum(values) / len(values),
        "p50": _percentile(values, 50),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "min": min(values),
        "max": max(values),
    }


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_hist(path: Path, title: str, values: list[float], xlabel: str, bins: int = 80) -> None:
    if not values:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.hist(values, bins=min(bins, max(10, int(math.sqrt(len(values)) * 2))), color="#2563eb", alpha=0.78)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _analyze(traces: list[Trace], start: float, end: float) -> dict[str, Any]:
    result: dict[str, Any] = {
        "engine_step_ms": defaultdict(list),
        "forward_ms": defaultdict(list),
        "batch_n_reqs": defaultdict(list),
        "batch_n_sched_tokens": defaultdict(list),
        "per_request_scheduled_tokens": defaultdict(list),
        "per_request_step_counts": defaultdict(Counter),
        "kv_transfer_bytes": defaultdict(list),
        "kv_transfer_ms": defaultdict(list),
    }

    for trace in traces:
        key = f"{trace.role}_dp{trace.dp}"
        for start_event, done_event in _pair_intervals(trace.events, "engine_step_start", "engine_step_done"):
            if not _in_window(start_event, start, end):
                continue
            duration_ms = (done_event["ts"] - start_event["ts"]) * 1000.0
            if duration_ms >= 0:
                result["engine_step_ms"][key].append(duration_ms)
        for start_event, done_event in _pair_intervals(trace.events, "forward_start", "forward_done"):
            if not _in_window(start_event, start, end):
                continue
            duration_ms = (done_event["ts"] - start_event["ts"]) * 1000.0
            if duration_ms >= 0:
                result["forward_ms"][key].append(duration_ms)
            n_reqs = start_event.get("n_reqs")
            n_sched = start_event.get("n_sched")
            if isinstance(n_reqs, int):
                result["batch_n_reqs"][key].append(float(n_reqs))
            if isinstance(n_sched, int):
                result["batch_n_sched_tokens"][key].append(float(n_sched))
            per_req = start_event.get("num_scheduled_tokens_by_req")
            if isinstance(per_req, dict):
                for req_id, num_tokens in per_req.items():
                    try:
                        token_count = int(num_tokens)
                    except (TypeError, ValueError):
                        continue
                    result["per_request_scheduled_tokens"][key].append(float(token_count))
                    result["per_request_step_counts"][key][_req_base(str(req_id))] += 1
        for event in trace.events:
            if event.get("ev") != "recv_xfer_telemetry" or not _in_window(event, start, end):
                continue
            try:
                duration_ms = float(event.get("xfer_duration_us")) / 1000.0
                bytes_transferred = float(event.get("bytes_transferred"))
            except (TypeError, ValueError):
                continue
            result["kv_transfer_ms"][key].append(duration_ms)
            result["kv_transfer_bytes"][key].append(bytes_transferred)
    return result


def _flatten_summary(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    metric_units = {
        "engine_step_ms": "ms",
        "forward_ms": "ms",
        "batch_n_reqs": "requests",
        "batch_n_sched_tokens": "tokens",
        "per_request_scheduled_tokens": "tokens",
        "kv_transfer_ms": "ms",
        "kv_transfer_bytes": "bytes",
    }
    for metric, by_key in result.items():
        if metric == "per_request_step_counts":
            for key, counter in by_key.items():
                values = [float(v) for v in counter.values()]
                row = {"metric": metric, "role_dp": key, "unit": "steps"}
                row.update(_summary(values))
                rows.append(row)
            continue
        for key, values in by_key.items():
            row = {"metric": metric, "role_dp": key, "unit": metric_units.get(metric, "")}
            row.update(_summary([float(v) for v in values]))
            rows.append(row)
    return rows


def _write_raw_csv(out_dir: Path, result: dict[str, Any]) -> None:
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for metric, by_key in result.items():
        if metric == "per_request_step_counts":
            for key, counter in by_key.items():
                rows = [{"request_id": req_id, "steps": count} for req_id, count in counter.items()]
                _write_summary_csv(raw_dir / f"{metric}_{key}.csv", rows)
            continue
        for key, values in by_key.items():
            rows = [{"value": value} for value in values]
            _write_summary_csv(raw_dir / f"{metric}_{key}.csv", rows)


def _write_plots(out_dir: Path, result: dict[str, Any]) -> None:
    plot_dir = out_dir / "plots"
    for metric, by_key in result.items():
        if metric == "per_request_step_counts":
            for key, counter in by_key.items():
                values = [float(v) for v in counter.values()]
                _plot_hist(plot_dir / f"{metric}_{key}.png", f"{metric} {key}", values, "steps per request")
            continue
        xlabel = "value"
        if metric.endswith("_ms"):
            xlabel = "milliseconds"
        elif metric.endswith("_bytes"):
            xlabel = "bytes"
        elif metric == "batch_n_reqs":
            xlabel = "requests per forward"
        elif "tokens" in metric:
            xlabel = "scheduled tokens"
        for key, values in by_key.items():
            _plot_hist(plot_dir / f"{metric}_{key}.png", f"{metric} {key}", [float(v) for v in values], xlabel)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--markers", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    run_name = args.run_name
    if run_name is None:
        run_name = Path("/tmp/e2e_2000req_run_name.txt").read_text().strip()
    trace_dir = args.trace_dir or Path("playground/log/e2e_agrs_nixl") / run_name / "pd_trace"
    markers = args.markers or Path("playground/out/aiperf_pd/e2e_agrs_nixl") / run_name / "phase_markers.jsonl"
    out_dir = args.out_dir or Path("playground/out/cpu_profile/e2e_agrs_nixl") / run_name

    start, end = _load_markers(markers)
    traces = _load_traces(trace_dir)
    result = _analyze(traces, start, end)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _flatten_summary(result)
    _write_summary_csv(out_dir / "summary.csv", rows)
    _write_raw_csv(out_dir, result)
    _write_plots(out_dir, result)

    summary_md = [
        "# CPU Profile Summary\n\n",
        f"run_name: `{run_name}`\n\n",
        f"trace_dir: `{trace_dir}`\n\n",
        f"main_window_epoch: `{start:.6f}` to `{end:.6f}` (`{end - start:.3f}s`)\n\n",
        "| metric | role_dp | count | avg | p50 | p90 | p95 | p99 | min | max | unit |\n",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n",
    ]
    for row in rows:
        summary_md.append(
            f"| {row['metric']} | {row['role_dp']} | {int(row['count'])} | "
            f"{row['avg']:.3f} | {row['p50']:.3f} | {row['p90']:.3f} | "
            f"{row['p95']:.3f} | {row['p99']:.3f} | {row['min']:.3f} | "
            f"{row['max']:.3f} | {row['unit']} |\n"
        )
    (out_dir / "summary.md").write_text("".join(summary_md), encoding="utf-8")
    print(f"[cpu-profile-analysis] wrote {out_dir / 'summary.md'}")
    print(f"[cpu-profile-analysis] wrote {out_dir / 'summary.csv'}")
    print(f"[cpu-profile-analysis] plots under {out_dir / 'plots'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
