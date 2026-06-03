#!/usr/bin/env python3
"""Plot one request's P/D lifecycle from PD JSONL traces."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
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


@dataclass
class Interval:
    start: float
    end: float
    lane: str
    kind: str
    label: str
    color: str
    alpha: float = 0.9

    @property
    def duration_ms(self) -> float:
        return max(0.0, (self.end - self.start) * 1000.0)


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
    raw_events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for raw in fp:
            raw = raw.strip()
            if not raw:
                continue
            event = json.loads(raw)
            if event.get("ev") == "boot" and "wall" in event and "ts" in event:
                trace.offset = float(event["wall"]) - float(event["ts"])
            raw_events.append(event)
    for event in raw_events:
        if "ts" in event:
            event = dict(event)
            event["ts"] = float(event["ts"]) + trace.offset
        trace.events.append(event)
    return trace


def _gather(trace_dir: Path) -> list[Trace]:
    traces = [_parse_trace(path) for path in sorted(trace_dir.glob("role=*.jsonl"))]
    if not traces:
        raise SystemExit(f"no trace files found under {trace_dir}")
    return traces


def _pair(trace: Trace, start_ev: str, done_ev: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pending: list[dict[str, Any]] = []
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for event in trace.events:
        if event.get("ev") == start_ev and "ts" in event:
            pending.append(event)
        elif event.get("ev") == done_ev and "ts" in event and pending:
            pairs.append((pending.pop(0), event))
    return pairs


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _req_base(req_id: str) -> str:
    match = _REQ_BASE_RE.match(req_id)
    return match.group("base") if match else req_id


def _has_req(event: dict[str, Any], req_id: str) -> bool:
    req_ids = event.get("req_ids")
    if not isinstance(req_ids, list):
        return False
    target = _req_base(req_id)
    return any(_req_base(str(item)) == target for item in req_ids)


def _request_transfers(traces: list[Trace], req_id: str, label_dp_base: int) -> list[Interval]:
    intervals: list[Interval] = []
    for trace in traces:
        if trace.role != "decode":
            continue
        for event in trace.events:
            if event.get("ev") != "recv_xfer_telemetry":
                continue
            if _req_base(str(event.get("req"))) != _req_base(req_id):
                continue
            duration_us = _float_or_nan(event.get("xfer_duration_us"))
            start_us = _float_or_nan(event.get("start_time_us"))
            if not math.isfinite(duration_us) or not math.isfinite(start_us) or duration_us <= 0:
                continue
            start = start_us / 1e6 + trace.offset
            intervals.append(
                Interval(
                    start=start,
                    end=start + duration_us / 1e6,
                    lane=f"D{trace.dp + label_dp_base} request KV transfer",
                    kind="kv_transfer",
                    label="request KV transfer",
                    color="#8b5cf6",
                    alpha=0.75,
                )
            )
    return intervals


def _all_kv_transfers(traces: list[Trace], label_dp_base: int) -> list[Interval]:
    intervals: list[Interval] = []
    for trace in traces:
        if trace.role != "decode":
            continue
        for event in trace.events:
            if event.get("ev") != "recv_xfer_telemetry":
                continue
            duration_us = _float_or_nan(event.get("xfer_duration_us"))
            start_us = _float_or_nan(event.get("start_time_us"))
            if not math.isfinite(duration_us) or not math.isfinite(start_us) or duration_us <= 0:
                continue
            start = start_us / 1e6 + trace.offset
            intervals.append(
                Interval(
                    start=start,
                    end=start + duration_us / 1e6,
                    lane=f"D{trace.dp + label_dp_base} all KV transfers",
                    kind="all_kv_transfer",
                    label="all KV transfers",
                    color="#a78bfa",
                    alpha=0.36,
                )
            )
    return intervals


def _request_forward_windows(traces: list[Trace], req_id: str) -> list[tuple[Trace, float, float]]:
    windows: list[tuple[Trace, float, float]] = []
    for trace in traces:
        for start, done in _pair(trace, "forward_start", "forward_done"):
            if not _has_req(start, req_id):
                continue
            if done["ts"] <= start["ts"]:
                continue
            windows.append((trace, start["ts"], done["ts"]))
    return windows


def _phase_intervals_for_windows(
    traces: list[Trace],
    windows: list[tuple[Trace, float, float]],
    label_dp_base: int,
) -> list[Interval]:
    intervals: list[Interval] = []
    windows_by_pid: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for trace, start, end in windows:
        windows_by_pid[trace.pid].append((start, end))
        prefix = "P" if trace.role == "prefill" else "D"
        intervals.append(
            Interval(
                start=start,
                end=end,
                lane=f"{prefix}{trace.dp + label_dp_base} request forward",
                kind="forward",
                label="request forward envelope",
                color="#94a3b8",
                alpha=0.35,
            )
        )
    for trace in traces:
        relevant_windows = windows_by_pid.get(trace.pid, [])
        if not relevant_windows:
            continue
        prefix = "P" if trace.role == "prefill" else "D"
        for phase, start_ev, done_ev, color in (
            ("attention", "attention_start", "attention_done", "#7c3aed"),
            ("moe", "moe_start", "moe_done", "#ef4444"),
        ):
            for start, done in _pair(trace, start_ev, done_ev):
                if done["ts"] <= start["ts"]:
                    continue
                for win_start, win_end in relevant_windows:
                    lo = max(start["ts"], win_start)
                    hi = min(done["ts"], win_end)
                    if hi <= lo:
                        continue
                    intervals.append(
                        Interval(
                            start=lo,
                            end=hi,
                            lane=f"{prefix}{trace.dp + label_dp_base} {phase}",
                            kind=phase,
                            label=phase,
                            color=color,
                            alpha=0.9,
                        )
                    )
    return intervals


def _all_phase_intervals(traces: list[Trace], label_dp_base: int) -> list[Interval]:
    intervals: list[Interval] = []
    for trace in traces:
        prefix = "P" if trace.role == "prefill" else "D"
        for phase, start_ev, done_ev, color in (
            ("attention", "attention_start", "attention_done", "#7c3aed"),
            ("moe", "moe_start", "moe_done", "#ef4444"),
        ):
            for start, done in _pair(trace, start_ev, done_ev):
                if done["ts"] <= start["ts"]:
                    continue
                intervals.append(
                    Interval(
                        start=start["ts"],
                        end=done["ts"],
                        lane=f"{prefix}{trace.dp + label_dp_base} {phase}",
                        kind=phase,
                        label=phase,
                        color=color,
                        alpha=0.86,
                    )
                )
    return intervals


def _select_request(traces: list[Trace]) -> str:
    transfer_count: dict[str, int] = defaultdict(int)
    transfer_bytes: dict[str, float] = defaultdict(float)
    forward_roles: dict[str, set[str]] = defaultdict(set)
    for trace in traces:
        for event in trace.events:
            if event.get("ev") == "recv_xfer_telemetry" and "req" in event:
                req_id = _req_base(str(event["req"]))
                transfer_count[req_id] += 1
                bytes_transferred = _float_or_nan(event.get("bytes_transferred"))
                if math.isfinite(bytes_transferred):
                    transfer_bytes[req_id] += bytes_transferred
            elif event.get("ev") == "forward_start" and isinstance(event.get("req_ids"), list):
                for req_id in event["req_ids"]:
                    forward_roles[_req_base(str(req_id))].add(trace.role)
    candidates = []
    for req_id, count in transfer_count.items():
        if not {"prefill", "decode"}.issubset(forward_roles.get(req_id, set())):
            continue
        candidates.append((count, transfer_bytes[req_id], req_id))
    if not candidates:
        raise SystemExit("no request has KV telemetry plus both prefill and decode forward req_ids")
    candidates.sort(reverse=True)
    return candidates[len(candidates) // 2][2]


def _lane_sort_key(lane: str) -> tuple[int, int, int, str]:
    role = 0 if lane.startswith("P") else 1
    match = re.match(r"[PD](\d+)", lane)
    dp = int(match.group(1)) if match else 99
    if "forward" in lane:
        kind = 0
    elif "attention" in lane:
        kind = 1
    elif "moe" in lane:
        kind = 2
    elif "request KV" in lane:
        kind = 3
    elif "all KV" in lane:
        kind = 4
    else:
        kind = 9
    return role, dp, kind, lane


def _window_bounds(intervals: list[Interval], pad_ms: float, max_window_sec: float) -> tuple[float, float]:
    if not intervals:
        raise SystemExit("no request intervals found")
    t0 = min(item.start for item in intervals) - pad_ms / 1000.0
    t1 = max(item.end for item in intervals) + pad_ms / 1000.0
    if max_window_sec > 0 and t1 - t0 > max_window_sec:
        t1 = t0 + max_window_sec
    return t0, t1


def _plot(
    out: Path,
    req_id: str,
    intervals: list[Interval],
    t0: float,
    t1: float,
) -> tuple[float, float, list[Interval]]:
    if not intervals:
        raise SystemExit(f"no intervals found for request {req_id}")
    plotted = [item for item in intervals if item.end >= t0 and item.start <= t1]
    lanes = sorted({item.lane for item in plotted}, key=_lane_sort_key)
    y_for_lane = {lane: idx for idx, lane in enumerate(lanes)}
    fig_height = max(6.0, min(12.0, 0.48 * len(lanes) + 2.0))
    fig, ax = plt.subplots(figsize=(16, fig_height), dpi=170)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfbfd")
    for item in sorted(plotted, key=lambda value: (y_for_lane[value.lane], value.start)):
        start = max(item.start, t0)
        end = min(item.end, t1)
        if end <= start:
            continue
        ax.broken_barh(
            [((start - t0) * 1000.0, max((end - start) * 1000.0, 0.025))],
            (y_for_lane[item.lane] - 0.35, 0.7),
            facecolors=item.color,
            alpha=item.alpha,
            edgecolors="none",
        )
    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels(lanes, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("time since request-lifecycle window start (ms)")
    short_req = req_id[:28] + "..." if len(req_id) > 31 else req_id
    ax.set_title(f"Request lifecycle: {short_req}", fontsize=13, pad=10)
    ax.set_xlim(0, max(1.0, (t1 - t0) * 1000.0))
    ax.grid(True, axis="x", color="#d1d5db", alpha=0.55)
    ax.grid(False, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend = [
        Patch(facecolor="#94a3b8", alpha=0.35, label="request forward envelope"),
        Patch(facecolor="#7c3aed", alpha=0.9, label="attention phase"),
        Patch(facecolor="#ef4444", alpha=0.9, label="MoE phase"),
        Patch(facecolor="#8b5cf6", alpha=0.75, label="request KV transfer"),
        Patch(facecolor="#a78bfa", alpha=0.36, label="all decode KV transfers"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=True, fontsize=9)
    fig.text(
        0.01,
        0.01,
        "Request rows are anchored by req_ids. Ambient rows show all P/D attention+MoE and all decode KV transfers in the same wall-clock window.",
        fontsize=8,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    return t0, t1, plotted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--req-id", default=None)
    parser.add_argument("--label-dp-base", type=int, default=0)
    parser.add_argument("--pad-ms", type=float, default=20.0)
    parser.add_argument("--max-window-sec", type=float, default=8.0)
    parser.add_argument("--include-environment", action="store_true")
    args = parser.parse_args()

    traces = _gather(args.trace_dir)
    req_id = args.req_id or _select_request(traces)
    transfers = _request_transfers(traces, req_id, args.label_dp_base)
    windows = _request_forward_windows(traces, req_id)
    request_intervals = _phase_intervals_for_windows(traces, windows, args.label_dp_base) + transfers
    t0, t1 = _window_bounds(request_intervals, args.pad_ms, args.max_window_sec)
    if args.include_environment:
        intervals = request_intervals + _all_phase_intervals(traces, args.label_dp_base) + _all_kv_transfers(traces, args.label_dp_base)
    else:
        intervals = request_intervals
    out = args.out_dir / "request_lifecycle.png"
    t0, t1, plotted = _plot(out, req_id, intervals, t0, t1)
    summary = args.out_dir / "request_lifecycle_summary.md"
    by_kind: dict[str, list[Interval]] = defaultdict(list)
    for item in plotted:
        by_kind[item.kind].append(item)
    lines = [
        "# Request Lifecycle Summary\n\n",
        f"req_id: `{req_id}`\n\n",
        f"trace_dir: `{args.trace_dir}`\n\n",
        f"window_epoch_start: {t0:.6f}\n\n",
        f"window_sec: {t1 - t0:.6f}\n\n",
        f"request_forward_intervals_total: {len(windows)}\n\n",
        f"request_kv_transfer_intervals_total: {len(transfers)}\n\n",
        "| kind | n | total_ms | max_ms |\n",
        "|---|---:|---:|---:|\n",
    ]
    for kind in ("forward", "attention", "moe", "kv_transfer", "all_kv_transfer"):
        values = by_kind.get(kind, [])
        if not values:
            continue
        durations = [item.duration_ms for item in values]
        lines.append(f"| {kind} | {len(values)} | {sum(durations):.3f} | {max(durations):.3f} |\n")
    summary.write_text("".join(lines), encoding="utf-8")
    print(f"[request] req_id={req_id}")
    print(f"[request] forward_intervals={len(windows)} kv_transfer_intervals={len(transfers)} plotted_intervals={len(plotted)}")
    print(f"[request] wrote {out}")
    print(f"[request] wrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
