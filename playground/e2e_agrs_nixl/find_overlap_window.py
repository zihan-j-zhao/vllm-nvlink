#!/usr/bin/env python3
"""Find P/D timeline windows with compute and KV-transfer overlap."""

from __future__ import annotations

import argparse
import bisect
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


_FNAME_RE = re.compile(r"^role=(?P<role>[^_]+)_dp=(?P<dp>\d+)_")


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    role: str
    dp: int
    kind: str


def _load_trace(path: Path) -> tuple[str, int, float, list[dict]]:
    match = _FNAME_RE.match(path.name)
    if not match:
        raise ValueError(f"unparseable trace filename: {path.name}")
    offset = 0.0
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            event = json.loads(line)
            rows.append(event)
            if event.get("ev") == "boot":
                offset = float(event["wall"]) - float(event["ts"])
    return match.group("role"), int(match.group("dp")), offset, rows


def _pair(rows: list[dict], start_ev: str, done_ev: str, offset: float, base: float) -> list[tuple[float, float]]:
    pending: list[dict] = []
    out: list[tuple[float, float]] = []
    for event in rows:
        if event.get("ev") == start_ev and "ts" in event:
            pending.append(event)
        elif event.get("ev") == done_ev and "ts" in event and pending:
            start = pending.pop(0)
            start_ts = float(start["ts"]) + offset - base
            end_ts = float(event["ts"]) + offset - base
            if end_ts > start_ts:
                out.append((start_ts, end_ts))
    return out


def _overlap(
    items: list[tuple[float, float]],
    max_duration: float,
    t0: float,
    t1: float,
) -> tuple[int, float, int, int, float]:
    start_idx = bisect.bisect_left(items, (t0 - max_duration, float("-inf")))
    idx = bisect.bisect_right(items, (t1, float("inf")))
    count = 0
    active = 0.0
    left_clipped = 0
    right_clipped = 0
    max_clipped_duration = 0.0
    for start, end in items[start_idx:idx]:
        if end < t0:
            continue
        lo = max(start, t0)
        hi = min(end, t1)
        if hi <= lo:
            continue
        count += 1
        clipped_duration = hi - lo
        active += clipped_duration
        max_clipped_duration = max(max_clipped_duration, clipped_duration)
        if start < t0:
            left_clipped += 1
        if end > t1:
            right_clipped += 1
    return count, active, left_clipped, right_clipped, max_clipped_duration


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--window-sec", type=float, default=0.16)
    parser.add_argument("--compute-kinds", type=str, default="attention,moe")
    parser.add_argument("--min-xfers", type=int, default=1)
    parser.add_argument("--max-xfers", type=int, default=0, help="0 means no maximum")
    parser.add_argument("--max-decode-compute-ms", type=float, default=0.0, help="0 means no maximum")
    parser.add_argument(
        "--require-xfer-dps",
        type=str,
        default="",
        help="Comma-separated zero-based decode DP ids that must have KV transfers.",
    )
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    compute_kinds = {item.strip() for item in args.compute_kinds.split(",") if item.strip()}
    required_xfer_dps = {
        int(item.strip()) for item in args.require_xfer_dps.split(",") if item.strip()
    }

    traces = [_load_trace(path) for path in sorted(args.trace_dir.glob("role=*.jsonl"))]
    all_ts = [
        float(event["ts"]) + offset
        for _role, _dp, offset, rows in traces
        for event in rows
        if event.get("ev") != "boot" and "ts" in event
    ]
    base = min(all_ts)

    by_key: dict[tuple[str, int, str], list[tuple[float, float]]] = defaultdict(list)
    candidate_starts: set[float] = set()
    for role, dp, offset, rows in traces:
        for kind in sorted(compute_kinds):
            values = _pair(rows, f"{kind}_start", f"{kind}_done", offset, base)
            by_key[(role, dp, kind)].extend(values)
            for start, _end in values[:: max(1, len(values) // 800)]:
                candidate_starts.add(round(start, 3))
        if role == "decode":
            for event in rows:
                if event.get("ev") != "recv_xfer_telemetry":
                    continue
                duration = float(event.get("xfer_duration_us", 0.0)) / 1e6
                if duration <= 0:
                    continue
                start = float(event.get("start_time_us", 0.0)) / 1e6 + offset - base
                by_key[(role, dp, "kv_transfer")].append((start, start + duration))
                for delta in (-0.12, -0.08, -0.04, 0.0, 0.04, 0.08):
                    candidate_starts.add(round(start + delta, 3))

    max_duration_by_key = {}
    for key in by_key:
        by_key[key].sort()
        max_duration_by_key[key] = max((end - start for start, end in by_key[key]), default=0.0)

    rows = []
    for t0 in sorted(candidate_starts):
        t1 = t0 + args.window_sec
        p_count = d_count = xfer_count = 0
        p_active = d_active = xfer_active = 0.0
        p_dps: set[int] = set()
        d_dps: set[int] = set()
        xfer_dps: set[int] = set()
        clip_left = clip_right = 0
        compute_clip = 0
        max_d_compute = 0.0
        for (role, dp, kind), items in by_key.items():
            count, active, left, right, max_duration = _overlap(
                items, max_duration_by_key[(role, dp, kind)], t0, t1
            )
            if not count:
                continue
            if role == "prefill" and kind in compute_kinds:
                p_count += count
                p_active += active
                p_dps.add(dp)
            elif role == "decode" and kind in compute_kinds:
                d_count += count
                d_active += active
                d_dps.add(dp)
                compute_clip += left + right
                max_d_compute = max(max_d_compute, max_duration)
            elif role == "decode" and kind == "kv_transfer":
                xfer_count += count
                xfer_active += active
                xfer_dps.add(dp)
                clip_left += left
                clip_right += right
        if not (p_count and d_count and xfer_count):
            continue
        if xfer_count < args.min_xfers:
            continue
        if args.max_xfers > 0 and xfer_count > args.max_xfers:
            continue
        if args.max_decode_compute_ms > 0 and max_d_compute * 1000.0 > args.max_decode_compute_ms:
            continue
        if required_xfer_dps and not required_xfer_dps.issubset(xfer_dps):
            continue
        same_decode_dp = len(d_dps & xfer_dps)
        score = (
            p_active * 20.0
            + d_active * 45.0
            + xfer_active * 3.0
            + len(p_dps) * 2.0
            + len(d_dps) * 7.0
            + len(xfer_dps) * 5.0
            + same_decode_dp * 10.0
            - clip_right * 0.15
            - compute_clip * 1.5
            - max(0.0, max_d_compute - args.window_sec * 0.35) * 80.0
        )
        rows.append(
            (
                score,
                t0,
                t1,
                p_count,
                d_count,
                xfer_count,
                p_active,
                d_active,
                xfer_active,
                sorted(p_dps),
                sorted(d_dps),
                sorted(xfer_dps),
                clip_left,
                clip_right,
                compute_clip,
                max_d_compute,
            )
        )

    for row in sorted(rows, reverse=True)[: args.top]:
        (
            score,
            t0,
            t1,
            p_count,
            d_count,
            xfer_count,
            p_active,
            d_active,
            xfer_active,
            p_dps,
            d_dps,
            xfer_dps,
            clip_left,
            clip_right,
            compute_clip,
            max_d_compute,
        ) = row
        print(
            f"score={score:.2f} start={t0:.3f} end={t1:.3f} "
            f"p={p_count}/{p_active * 1000:.1f}ms d={d_count}/{d_active * 1000:.1f}ms "
            f"xfer={xfer_count}/{xfer_active * 1000:.1f}ms "
            f"p_dps={p_dps} d_dps={d_dps} xfer_dps={xfer_dps} "
            f"xfer_clipL/R={clip_left}/{clip_right} compute_clip={compute_clip} "
            f"max_d_compute_ms={max_d_compute * 1000:.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())