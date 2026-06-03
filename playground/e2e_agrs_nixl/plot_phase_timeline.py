#!/usr/bin/env python3
"""Plot a fine-grained P/D timeline from PD JSONL traces.

The PD trace events are CPU wall-clock events emitted around vLLM phases. They
are useful for understanding ordering and overlap, but they are not CUDA kernel
durations unless the trace event was explicitly emitted from synchronized GPU
timing. NIXL KV transfer intervals use NIXL telemetry when available.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
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


@dataclass
class ProcessTrace:
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
    alpha: float = 0.85
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return max(0.0, (self.end - self.start) * 1000.0)


PHASE_SPECS: list[tuple[str, str, str, str]] = [
    ("engine", "engine_step_start", "engine_step_done", "#9ca3af"),
    ("forward", "forward_start", "forward_done", "#2563eb"),
    ("kv_start_load", "kv_start_load_begin", "kv_start_load_done", "#f59e0b"),
    ("kv_finalize", "kv_finalize_begin", "kv_finalize_done", "#d97706"),
    ("kv_wait_for_save", "kv_wait_for_save_begin", "kv_wait_for_save_done", "#fbbf24"),
    ("kv_get_finished", "kv_get_finished_begin", "kv_get_finished_done", "#b45309"),
    ("sample", "sample_start", "sample_done", "#10b981"),
    ("sample_tokens", "sample_tokens_start", "sample_tokens_done", "#34d399"),
    ("bookkeeping", "bookkeeping_start", "bookkeeping_done", "#6b7280"),
    # Future-proof names. Current traces usually do not contain these yet.
    ("attention", "attention_start", "attention_done", "#7c3aed"),
    ("mlp", "mlp_start", "mlp_done", "#0ea5e9"),
    ("moe", "moe_start", "moe_done", "#dc2626"),
    ("moe_router", "moe_router_start", "moe_router_done", "#f43f5e"),
    ("moe_dispatch", "moe_dispatch_start", "moe_dispatch_done", "#ef4444"),
    ("moe_experts", "moe_experts_start", "moe_experts_done", "#f97316"),
    ("moe_combine", "moe_combine_start", "moe_combine_done", "#b91c1c"),
]

KIND_ORDER = {
    "engine": 0,
    "forward": 1,
    "attention": 2,
    "mlp": 3,
    "moe": 4,
    "moe_router": 5,
    "moe_dispatch": 6,
    "moe_experts": 7,
    "moe_combine": 8,
    "kv_start_load": 9,
    "kv_transfer": 10,
    "kv_recv_cpu": 11,
    "kv_finalize": 12,
    "kv_wait_for_save": 13,
    "kv_get_finished": 14,
    "sample": 15,
    "sample_tokens": 16,
    "bookkeeping": 17,
}


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


def _gather(trace_dir: Path) -> list[ProcessTrace]:
    traces = []
    for path in sorted(trace_dir.glob("role=*.jsonl")):
        try:
            traces.append(_parse_trace(path))
        except Exception as exc:
            print(f"warning: skipping {path}: {exc}", file=sys.stderr)
    if not traces:
        raise SystemExit(f"no trace files found under {trace_dir}")
    return traces


def _pair_intervals(
    trace: ProcessTrace, start_ev: str, done_ev: str
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pending: list[dict[str, Any]] = []
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for event in trace.events:
        kind = event.get("ev")
        if kind == start_ev and "ts" in event:
            pending.append(event)
        elif kind == done_ev and "ts" in event and pending:
            pairs.append((pending.pop(0), event))
    return pairs


def _phase_intervals(traces: list[ProcessTrace]) -> list[Interval]:
    intervals: list[Interval] = []
    for trace in traces:
        lane_prefix = f"{trace.role[0].upper()}{trace.dp}"
        for phase, start_ev, done_ev, color in PHASE_SPECS:
            for start, done in _pair_intervals(trace, start_ev, done_ev):
                if done["ts"] <= start["ts"]:
                    continue
                intervals.append(
                    Interval(
                        start=start["ts"],
                        end=done["ts"],
                        lane=f"{lane_prefix} {phase}",
                        kind=phase,
                        label=phase.replace("_", " "),
                        color=color,
                        alpha=0.78 if phase == "engine" else 0.9,
                        meta={"role": trace.role, "dp": trace.dp, "pid": trace.pid},
                    )
                )
    return intervals


def _software_recv_intervals(traces: list[ProcessTrace]) -> list[Interval]:
    intervals: list[Interval] = []
    for trace in traces:
        if trace.role != "decode":
            continue
        pending_by_req: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in trace.events:
            kind = event.get("ev")
            if kind == "recv_start" and "ts" in event:
                pending_by_req[str(event.get("req"))].append(event)
            elif kind == "recv_done" and "ts" in event:
                req = str(event.get("req"))
                if not pending_by_req.get(req):
                    continue
                start = pending_by_req[req].pop(0)
                intervals.append(
                    Interval(
                        start=start["ts"],
                        end=event["ts"],
                        lane=f"D{trace.dp} kv recv cpu",
                        kind="kv_recv_cpu",
                        label="KV recv CPU scope",
                        color="#9333ea",
                        alpha=0.30,
                        meta={"req": req, "role": trace.role, "dp": trace.dp},
                    )
                )
    return intervals


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _nixl_transfer_intervals(traces: list[ProcessTrace], start_kind: str) -> list[Interval]:
    intervals: list[Interval] = []
    for trace in traces:
        if trace.role != "decode":
            continue
        post_begin: dict[tuple[str, str], dict[str, Any]] = {}
        post_done: dict[tuple[str, str], dict[str, Any]] = {}
        for event in trace.events:
            kind = event.get("ev")
            if kind not in {"recv_post_begin", "recv_post_done", "recv_xfer_telemetry"}:
                continue
            req = str(event.get("req"))
            handle_obj = event.get("handle")
            if handle_obj is None:
                continue
            handle = str(handle_obj)
            key = (req, handle)
            if kind == "recv_post_begin" and "ts" in event:
                post_begin[key] = event
            elif kind == "recv_post_done" and "ts" in event:
                post_done[key] = event
            elif kind == "recv_xfer_telemetry" and "ts" in event:
                xfer_duration_us = _float_or_nan(event.get("xfer_duration_us"))
                if not math.isfinite(xfer_duration_us) or xfer_duration_us <= 0:
                    continue
                duration_s = xfer_duration_us / 1e6
                begin_ev = post_begin.get(key)
                done_ev = post_done.get(key)
                start_time_us = _float_or_nan(event.get("start_time_us"))
                start_time_s = start_time_us / 1e6 + trace.offset
                if start_kind == "telemetry_end":
                    start = event["ts"] - duration_s
                elif start_kind == "start_time" and math.isfinite(start_time_s):
                    start = start_time_s
                elif start_kind == "post_begin" and begin_ev is not None:
                    start = begin_ev["ts"]
                elif done_ev is not None:
                    start = done_ev["ts"]
                elif begin_ev is not None:
                    start = begin_ev["ts"]
                else:
                    start = event["ts"] - duration_s
                end = event["ts"] if start_kind == "telemetry_end" else start + duration_s
                bytes_transferred = _float_or_nan(event.get("bytes_transferred"))
                intervals.append(
                    Interval(
                        start=start,
                        end=end,
                        lane=f"D{trace.dp} NIXL KV xfer",
                        kind="kv_transfer",
                        label="NIXL KV transfer",
                        color="#8b5cf6",
                        alpha=0.62,
                        meta={
                            "req": req,
                            "handle": handle,
                            "role": trace.role,
                            "dp": trace.dp,
                            "pid": trace.pid,
                            "bytes_transferred": bytes_transferred,
                            "throughput_gbps": (
                                bytes_transferred / duration_s / 1e9
                                if duration_s > 0 and math.isfinite(bytes_transferred)
                                else float("nan")
                            ),
                        },
                    )
                )
    return intervals


def _choose_window(
    intervals: list[Interval], base_t0: float, window_sec: float, requested_start: float | None
) -> tuple[float, float, str]:
    if requested_start is not None:
        start = base_t0 + requested_start
        return start, start + window_sec, f"requested +{requested_start:.3f}s"
    transfers = [item for item in intervals if item.kind == "kv_transfer"]
    if transfers:
        candidates = sorted({item.start for item in transfers})
        stride = max(1, len(candidates) // 500)
        best_start = candidates[0]
        best_score = -1.0
        for start in candidates[::stride]:
            end = start + window_sec
            score = 0.0
            for item in transfers:
                lo = max(start, item.start)
                hi = min(end, item.end)
                if hi > lo:
                    score += hi - lo
            if score > best_score:
                best_score = score
                best_start = start
        return best_start, best_start + window_sec, "auto densest NIXL-transfer window"
    starts = [item.start for item in intervals]
    if not starts:
        raise SystemExit("no intervals found to plot")
    start = min(starts)
    return start, start + window_sec, "auto first interval window"


def _clip_intervals(intervals: list[Interval], t0: float, t1: float, min_ms: float) -> list[Interval]:
    clipped: list[Interval] = []
    for item in intervals:
        original_start = item.start
        original_end = item.end
        start = max(t0, item.start)
        end = min(t1, item.end)
        if end <= start:
            continue
        if (end - start) * 1000.0 < min_ms:
            continue
        meta = dict(item.meta)
        meta["original_start"] = original_start
        meta["original_end"] = original_end
        clipped.append(Interval(start, end, item.lane, item.kind, item.label, item.color, item.alpha, meta))
    return clipped


def _lane_sort_key(lane: str) -> tuple[int, int, int, str]:
    role_order = 0 if lane.startswith("P") else 1 if lane.startswith("D") else 2
    match = re.match(r"[PD](\d+)", lane)
    dp = int(match.group(1)) if match else 99
    kind = lane.split(" ", 1)[1] if " " in lane else lane
    if kind.startswith("NIXL KV xfer"):
        normalized_kind = "kv_transfer"
    elif kind.startswith("kv recv cpu"):
        normalized_kind = "kv_recv_cpu"
    else:
        normalized_kind = kind.replace(" ", "_")
    return role_order, dp, KIND_ORDER.get(normalized_kind, 99), lane


def _parse_csv_set(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    values = {item.strip() for item in raw.split(",") if item.strip()}
    return values or None


def _filter_intervals(
    intervals: list[Interval],
    roles: set[str] | None,
    include_kinds: set[str] | None,
    exclude_kinds: set[str] | None,
) -> list[Interval]:
    filtered: list[Interval] = []
    for item in intervals:
        role = str(item.meta.get("role", ""))
        if roles is not None and role not in roles:
            continue
        if include_kinds is not None and item.kind not in include_kinds:
            continue
        if exclude_kinds is not None and item.kind in exclude_kinds:
            continue
        filtered.append(item)
    return filtered


def _spread_transfer_lanes(intervals: list[Interval], max_lanes: int) -> list[Interval]:
    if max_lanes <= 1:
        return intervals
    result: list[Interval] = []
    by_dp: dict[int, list[Interval]] = defaultdict(list)
    for item in intervals:
        if item.kind == "kv_transfer":
            by_dp[int(item.meta.get("dp", 0))].append(item)
        else:
            result.append(item)
    for dp, transfers in by_dp.items():
        lane_ends = [float("-inf")] * max_lanes
        for item in sorted(transfers, key=lambda value: (value.start, value.end)):
            lane_idx = next((idx for idx, end in enumerate(lane_ends) if end <= item.start), None)
            if lane_idx is None:
                lane_idx = min(range(max_lanes), key=lambda idx: lane_ends[idx])
            lane_ends[lane_idx] = max(lane_ends[lane_idx], item.end)
            result.append(
                Interval(
                    start=item.start,
                    end=item.end,
                    lane=f"D{dp} NIXL KV xfer {lane_idx}",
                    kind=item.kind,
                    label=item.label,
                    color=item.color,
                    alpha=item.alpha,
                    meta=item.meta,
                )
            )
    return result


def _plot_timeline(out: Path, intervals: list[Interval], t0: float, t1: float, title: str, max_bars: int) -> None:
    if not intervals:
        raise SystemExit("selected window contains no plottable intervals")
    if len(intervals) > max_bars:
        print(
            f"warning: plotting first {max_bars} of {len(intervals)} bars; "
            "reduce --window-sec or raise --max-bars",
            file=sys.stderr,
        )
        intervals = sorted(intervals, key=lambda item: (item.start, _lane_sort_key(item.lane)))[:max_bars]
    lanes = sorted({item.lane for item in intervals}, key=_lane_sort_key)
    y_for_lane = {lane: idx for idx, lane in enumerate(lanes)}
    height = max(6.0, 0.34 * len(lanes) + 1.8)
    fig, ax = plt.subplots(figsize=(16, height), dpi=170)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfbfd")
    for item in sorted(intervals, key=lambda value: (y_for_lane[value.lane], value.start, KIND_ORDER.get(value.kind, 99))):
        y = y_for_lane[item.lane] - 0.38
        left = (item.start - t0) * 1000.0
        width = max((item.end - item.start) * 1000.0, 0.02)
        ax.broken_barh([(left, width)], (y, 0.76), facecolors=item.color, alpha=item.alpha, edgecolors="none")
    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels(lanes, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("time since window start (ms)")
    ax.set_title(title, fontsize=13, pad=10)
    ax.grid(True, axis="x", color="#d1d5db", alpha=0.55)
    ax.grid(True, axis="y", color="#e5e7eb", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max(1.0, (t1 - t0) * 1000.0))
    legend_items: dict[str, Patch] = {}
    for item in intervals:
        legend_items.setdefault(item.label, Patch(facecolor=item.color, alpha=item.alpha, label=item.label))
    ax.legend(
        handles=[legend_items[key] for key in sorted(legend_items)],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=min(5, max(1, len(legend_items))),
        frameon=True,
        fontsize=8,
    )
    note = (
        "CPU phase bars show host-side launch/phase scope, not CUDA kernel time. "
        "NIXL KV transfer bars use xferDuration/startTime telemetry."
    )
    fig.text(0.01, 0.01, note, fontsize=8, color="#4b5563")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")


def _pack_lanes(items: list[Interval]) -> list[tuple[Interval, int, int]]:
    lane_ends: list[float] = []
    packed: list[tuple[Interval, int, int]] = []
    for item in sorted(items, key=lambda value: (value.start, value.end)):
        lane_idx = next((idx for idx, end in enumerate(lane_ends) if end <= item.start), None)
        if lane_idx is None:
            lane_idx = len(lane_ends)
            lane_ends.append(float("-inf"))
        lane_ends[lane_idx] = max(lane_ends[lane_idx], item.end)
        packed.append((item, lane_idx, 0))
    total = max(1, len(lane_ends))
    return [(item, lane_idx, total) for item, lane_idx, _ in packed]


def _plot_overlap_model(
    out: Path,
    intervals: list[Interval],
    t0: float,
    t1: float,
    title: str,
    max_bars: int,
    dp_label_base: int,
    forward_context: bool,
    idle_gap_ms: float,
) -> None:
    compute_kind_order = ["forward", "attention", "mlp", "moe"]
    compute_kinds = set(compute_kind_order)
    compute_colors = {
        "forward": "#2563eb",
        "attention": "#7c3aed",
        "mlp": "#0ea5e9",
        "moe": "#ef4444",
        "kv_transfer": "#8b5cf6",
        "kv_start_load": "#f59e0b",
        "kv_finalize": "#d97706",
        "kv_get_finished": "#b45309",
    }
    compute_intervals = [item for item in intervals if item.kind in compute_kinds]
    row_compute_intervals = [
        item for item in compute_intervals if not (forward_context and item.kind == "forward")
    ]
    kv_intervals = [item for item in intervals if item.kind == "kv_transfer"]
    kv_markers = [
        item
        for item in intervals
        if item.kind in {"kv_start_load", "kv_finalize", "kv_get_finished"}
    ]
    if len(row_compute_intervals) + len(kv_intervals) + len(kv_markers) > max_bars:
        print(
            f"warning: plotting first {max_bars} of "
            f"{len(row_compute_intervals) + len(kv_intervals) + len(kv_markers)} bars",
            file=sys.stderr,
        )
    roles = ["prefill", "decode"]
    role_label = {"prefill": "P", "decode": "D"}
    rows: list[tuple[str, str, int | None, str | None, float]] = []
    for role in roles:
        dps = sorted({int(item.meta.get("dp", 0)) for item in row_compute_intervals if item.meta.get("role") == role})
        for dp in dps:
            present_kinds = {
                item.kind
                for item in row_compute_intervals
                if item.meta.get("role") == role and int(item.meta.get("dp", 0)) == dp
            }
            for kind in compute_kind_order:
                if kind not in present_kinds:
                    continue
                rows.append((f"{role_label[role]}{dp + dp_label_base} {kind}", role, dp, kind, 0.70))
    for dp in sorted({int(item.meta.get("dp", 0)) for item in kv_intervals}):
        count = sum(1 for item in kv_intervals if int(item.meta.get("dp", 0)) == dp)
        rows.append((f"D{dp + dp_label_base} KV transfers (n={count})", "decode", dp, "kv_transfer", 1.25))

    if not rows:
        raise SystemExit("selected window contains no compute or KV intervals for overlap plot")
    y_base: dict[str, float] = {}
    y = 0.0
    previous_group = None
    for label, role, dp, _kind, height in rows:
        group = (role, dp)
        if previous_group is not None and group != previous_group:
            y += 0.35
        y_base[label] = y
        y += height + 0.12
        previous_group = group

    fig_height = max(6.0, min(11.0, y * 0.48 + 1.8))
    fig, ax = plt.subplots(figsize=(15.5, fig_height), dpi=170)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfbfd")

    row_by_key = {(role, dp, kind): label for label, role, dp, kind, _height in rows}
    for item in sorted(row_compute_intervals, key=lambda value: (value.start, value.kind))[:max_bars]:
        key = (str(item.meta.get("role")), int(item.meta.get("dp", 0)), item.kind)
        label = row_by_key.get(key)
        if label is None:
            continue
        left = (item.start - t0) * 1000.0
        width = max((item.end - item.start) * 1000.0, 0.025)
        ax.broken_barh(
            [(left, width)],
            (y_base[label], 0.50),
            facecolors=compute_colors[item.kind],
            alpha=0.88,
            edgecolors="none",
        )

    if forward_context and idle_gap_ms > 0:
        forward_by_group: dict[tuple[str, int], list[Interval]] = defaultdict(list)
        row_labels_by_group: dict[tuple[str, int], list[str]] = defaultdict(list)
        for item in compute_intervals:
            if item.kind == "forward":
                forward_by_group[(str(item.meta.get("role")), int(item.meta.get("dp", 0)))].append(item)
        for label, role, dp, kind, _height in rows:
            if kind != "kv_transfer" and dp is not None:
                row_labels_by_group[(role, dp)].append(label)
        for group, labels_for_group in row_labels_by_group.items():
            if not labels_for_group:
                continue
            spans = sorted((max(t0, item.start), min(t1, item.end)) for item in forward_by_group.get(group, []))
            merged: list[tuple[float, float]] = []
            for start, end in spans:
                if end <= start:
                    continue
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            cursor = t0
            gaps: list[tuple[float, float]] = []
            for start, end in merged:
                if start > cursor:
                    gaps.append((cursor, start))
                cursor = max(cursor, end)
            if cursor < t1:
                gaps.append((cursor, t1))
            y_values = [y_base[label] for label in labels_for_group]
            text_y = (min(y_values) + max(y_values)) / 2 + 0.35
            for start, end in gaps:
                gap_ms = (end - start) * 1000.0
                if gap_ms < idle_gap_ms:
                    continue
                ax.text(
                    (start + end) / 2 * 1000.0 - t0 * 1000.0,
                    text_y,
                    "no forward\n(waiting)",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#6b7280",
                    alpha=0.85,
                )

    for dp in sorted({int(item.meta.get("dp", 0)) for item in kv_intervals}):
        label = row_by_key.get(("decode", dp, "kv_transfer"))
        if label is None:
            continue
        band_y = y_base[label]
        band_height = 1.05
        ax.axhspan(band_y - 0.02, band_y + band_height + 0.02, color="#f3e8ff", alpha=0.55, linewidth=0)
        packed = _pack_lanes([item for item in kv_intervals if int(item.meta.get("dp", 0)) == dp])
        for item, lane_idx, lane_count in packed:
            lane_height = max(0.018, min(0.18, band_height / max(1, lane_count) * 0.86))
            lane_y = band_y + (lane_idx / max(1, lane_count)) * band_height
            left = (item.start - t0) * 1000.0
            width = max((item.end - item.start) * 1000.0, 0.04)
            ax.broken_barh(
                [(left, width)],
                (lane_y, lane_height),
                facecolors=compute_colors["kv_transfer"],
                alpha=0.62,
                edgecolors="none",
            )
            midpoint_y = lane_y + lane_height / 2
            if float(item.meta.get("original_start", item.start)) < t0:
                ax.scatter(
                    [0],
                    [midpoint_y],
                    marker="<",
                    s=14,
                    color=compute_colors["kv_transfer"],
                    alpha=0.85,
                    linewidths=0,
                    zorder=3,
                )
            if float(item.meta.get("original_end", item.end)) > t1:
                ax.scatter(
                    [(t1 - t0) * 1000.0],
                    [midpoint_y],
                    marker=">",
                    s=14,
                    color=compute_colors["kv_transfer"],
                    alpha=0.85,
                    linewidths=0,
                    zorder=3,
                )
        for item in [item for item in kv_markers if int(item.meta.get("dp", 0)) == dp]:
            left = (item.start - t0) * 1000.0
            width = max((item.end - item.start) * 1000.0, 0.06)
            marker_y = band_y + band_height + 0.05
            ax.broken_barh(
                [(left, width)],
                (marker_y, 0.10),
                facecolors=compute_colors[item.kind],
                alpha=0.95,
                edgecolors="none",
            )

    labels = [label for label, *_rest in rows]
    ax.set_yticks([y_base[label] + 0.25 for label in labels])
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("time since window start (ms)")
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlim(0, max(1.0, (t1 - t0) * 1000.0))
    ax.grid(True, axis="x", color="#d1d5db", alpha=0.55)
    ax.grid(False, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    present_compute_kinds = {item.kind for item in row_compute_intervals}
    legend = []
    if "attention" in present_compute_kinds:
        legend.append(Patch(facecolor=compute_colors["attention"], alpha=0.88, label="attention"))
    if "forward" in present_compute_kinds:
        legend.append(Patch(facecolor=compute_colors["forward"], alpha=0.88, label="forward"))
    if "mlp" in present_compute_kinds:
        legend.append(Patch(facecolor=compute_colors["mlp"], alpha=0.88, label="MLP"))
    if "moe" in present_compute_kinds:
        legend.append(Patch(facecolor=compute_colors["moe"], alpha=0.88, label="MoE"))
    if kv_intervals:
        legend.append(Patch(facecolor=compute_colors["kv_transfer"], alpha=0.62, label="NIXL KV transfer"))
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=5, frameon=True, fontsize=9)
    fig.text(
        0.01,
        0.01,
        "Compute rows show CPU-side per-layer phase envelopes. KV transfer band plots every overlapping NIXL transfer interval from telemetry.",
        fontsize=8,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")


def _write_summary(
    out: Path, trace_dir: Path, window_t0: float, window_t1: float, reason: str, intervals: list[Interval]
) -> None:
    by_kind: dict[str, list[Interval]] = defaultdict(list)
    for item in intervals:
        by_kind[item.kind].append(item)
    lines = [
        "# P/D Phase Timeline Summary\n\n",
        f"trace_dir: `{trace_dir}`\n\n",
        f"window_reason: {reason}\n\n",
        f"window_epoch_start: {window_t0:.6f}\n\n",
        f"window_sec: {window_t1 - window_t0:.6f}\n\n",
        "## Interval Counts\n\n",
        "| kind | n | total_ms | p50_ms | p95_ms | max_ms |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ]
    for kind in sorted(by_kind, key=lambda value: KIND_ORDER.get(value, 99)):
        values = sorted(item.duration_ms for item in by_kind[kind])
        if not values:
            continue
        p50 = values[int(0.50 * (len(values) - 1))]
        p95 = values[int(0.95 * (len(values) - 1))]
        lines.append(
            f"| {kind} | {len(values)} | {sum(values):.3f} | {p50:.3f} | {p95:.3f} | {max(values):.3f} |\n"
        )
    if "moe" not in by_kind and "moe_dispatch" not in by_kind and "moe_combine" not in by_kind:
        lines.extend([
            "\n## MoE Visibility\n\n",
            "No MoE-specific phase events were present in this trace. The `forward` bars contain attention, MoE, and other model work at CPU launch scope. To see actual MoE or attention kernel durations, collect Nsight/CUPTI data or add synchronized CUDA-event instrumentation around the relevant model submodules.\n",
        ])
    out.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--window-start-sec", type=float, default=None)
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--min-duration-ms", type=float, default=0.02)
    parser.add_argument("--max-bars", type=int, default=7000)
    parser.add_argument(
        "--style",
        choices=("timeline", "overlap"),
        default="timeline",
        help="timeline is the full diagnostic plot; overlap is a compact presentation plot.",
    )
    parser.add_argument(
        "--roles",
        type=str,
        default=None,
        help="Comma-separated roles to plot, for example: prefill,decode",
    )
    parser.add_argument(
        "--include-kinds",
        type=str,
        default=None,
        help="Comma-separated interval kinds to include, for example: engine,kv_transfer,attention",
    )
    parser.add_argument(
        "--exclude-kinds",
        type=str,
        default=None,
        help="Comma-separated interval kinds to exclude, for example: kv_recv_cpu",
    )
    parser.add_argument(
        "--kv-transfer-lanes",
        type=int,
        default=1,
        help="Spread overlapping KV transfers over this many sublanes per decode DP.",
    )
    parser.add_argument(
        "--overlap-max-compute-ms",
        type=float,
        default=0.0,
        help="For --style overlap, drop compute intervals longer than this many ms. 0 disables filtering.",
    )
    parser.add_argument(
        "--label-dp-base",
        type=int,
        default=0,
        help="Add this offset to displayed DP labels. Use 1 for D1/D2 slide labels.",
    )
    parser.add_argument(
        "--overlap-forward-context",
        action="store_true",
        help="Use forward intervals to label no-forward gaps without drawing forward rows.",
    )
    parser.add_argument(
        "--overlap-idle-gap-ms",
        type=float,
        default=25.0,
        help="Minimum no-forward gap to label when --overlap-forward-context is enabled.",
    )
    parser.add_argument(
        "--nixl-start",
        choices=("telemetry_end", "post_done", "post_begin", "start_time"),
        default="start_time",
    )
    args = parser.parse_args()

    traces = _gather(args.trace_dir)
    all_timestamped = [
        event["ts"]
        for trace in traces
        for event in trace.events
        if event.get("ev") != "boot" and "ts" in event
    ]
    if not all_timestamped:
        raise SystemExit("no timestamped events found")
    base_t0 = min(all_timestamped)
    phase_intervals = _phase_intervals(traces)
    transfer_intervals = _nixl_transfer_intervals(traces, args.nixl_start)
    software_recv_intervals = _software_recv_intervals(traces)
    all_intervals = phase_intervals + transfer_intervals + software_recv_intervals
    all_intervals = _filter_intervals(
        all_intervals,
        roles=_parse_csv_set(args.roles),
        include_kinds=_parse_csv_set(args.include_kinds),
        exclude_kinds=_parse_csv_set(args.exclude_kinds),
    )
    all_intervals = _spread_transfer_lanes(all_intervals, args.kv_transfer_lanes)
    window_t0, window_t1, reason = _choose_window(all_intervals, base_t0, args.window_sec, args.window_start_sec)
    plotted = _clip_intervals(all_intervals, window_t0, window_t1, args.min_duration_ms)
    if args.out_dir is None:
        args.out_dir = Path("playground/out/timeline/e2e_agrs_nixl") / args.trace_dir.parent.name
    args.out_dir.mkdir(parents=True, exist_ok=True)
    title = f"P/D phase timeline: {args.trace_dir.parent.name} ({reason})"
    png = args.out_dir / "pd_phase_timeline.png"
    if args.style == "overlap":
        title = f"P/D compute and KV-transfer overlap ({reason}, {(window_t1 - window_t0) * 1000:.0f} ms)"
        if args.overlap_max_compute_ms > 0:
            compute_kinds = {"forward", "attention", "mlp", "moe"}
            plotted = [
                item
                for item in plotted
                if item.kind not in compute_kinds or item.duration_ms <= args.overlap_max_compute_ms
            ]
        _plot_overlap_model(
            png,
            plotted,
            window_t0,
            window_t1,
            title,
            args.max_bars,
            args.label_dp_base,
            args.overlap_forward_context,
            args.overlap_idle_gap_ms,
        )
    else:
        _plot_timeline(png, plotted, window_t0, window_t1, title, args.max_bars)
    summary = args.out_dir / "pd_phase_timeline_summary.md"
    _write_summary(summary, args.trace_dir, window_t0, window_t1, reason, plotted)
    print(f"[timeline] trace_dir={args.trace_dir}")
    print(f"[timeline] window={reason}, +{window_t0 - base_t0:.3f}s to +{window_t1 - base_t0:.3f}s")
    print(f"[timeline] plotted_intervals={len(plotted)}")
    print(f"[timeline] nixl_transfer_intervals_in_window={sum(1 for item in plotted if item.kind == 'kv_transfer')}")
    print(f"[timeline] moe_intervals_in_window={sum(1 for item in plotted if item.kind.startswith('moe'))}")
    print(f"[timeline] wrote {png}")
    print(f"[timeline] wrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())