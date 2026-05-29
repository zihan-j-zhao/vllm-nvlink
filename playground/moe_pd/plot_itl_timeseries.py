#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Plot system-wide TPOT/ITL over wall-clock time for the P/D-disagg + EP
playground, with shaded regions where decode-side NIXL KV transfers were
active.

Inputs: a directory of JSONL files written by the env-gated tracer in
vllm/distributed/kv_transfer/kv_connector/v1/nixl/_trace.py.
File naming convention (one per process, see _trace.py::get_writer):
    role=<prefill|decode|engine>_dp=<i>_tp=<j>_pid=<pid>[_label=<L>].jsonl

Events:
    {"ev":"boot","ts":perf_counter,"wall":time.time(),"pid":...}
    {"ev":"step","ts":...,"n_sched":...,"tokens":[[req_id,n_new], ...]}
    {"ev":"recv_start","ts":...,"req":...,"remote_engine":..., ...}
    {"ev":"recv_done","ts":...,"req":...}

Clock alignment: each process records its perf_counter()->time.time() offset
once at startup. We translate every per-process `ts` into a common wall-clock
axis and re-zero everything to the earliest first step seen.

Output: a PNG (default playground/out/aiperf_pd/<latest>/itl_timeseries.png)
with:
  - ITL p50/p90/p99 lines aggregated across all *decode-side* requests
  - shaded grey bands over the union of decode `[recv_start, recv_done]`
    intervals (the wall-clock windows in which NIXL/UCX was actively
    pulling KV blocks)

Usage:
    python playground/moe_pd/plot_itl_timeseries.py \\
        --trace-dir playground/log/moe_pd/<TS>/pd_trace \\
        [--bin-ms 100] [--out OUT.png]

If --trace-dir is omitted, picks the newest playground/log/moe_pd/*/pd_trace
directory.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - import-time error path
    print(f"error: matplotlib is required ({exc})", file=sys.stderr)
    raise


# Filename like: role=decode_dp=1_tp=0_pid=12345.jsonl
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
    # perf_counter -> wall_time offset; wall = ts + offset.
    offset: float = 0.0
    boot_wall: float | None = None
    boot_perf: float | None = None
    # Raw events (after wall-clock translation): list of dicts with `ts` now
    # being wall time (seconds, float).
    steps: list[dict] = None  # type: ignore[assignment]
    step_dones: list[dict] = None  # type: ignore[assignment]
    recv_starts: list[dict] = None  # type: ignore[assignment]
    recv_dones: list[dict] = None  # type: ignore[assignment]

    def __post_init__(self):
        self.steps = []
        self.step_dones = []
        self.recv_starts = []
        self.recv_dones = []


def _parse_jsonl(path: Path) -> ProcessTrace:
    m = _FNAME_RE.match(path.name)
    if not m:
        raise ValueError(f"unparseable trace filename: {path.name}")
    pt = ProcessTrace(
        path=path,
        role=m.group("role"),
        dp=int(m.group("dp")),
        tp=int(m.group("tp")),
        pid=int(m.group("pid")),
    )
    with path.open("r", encoding="utf-8") as fp:
        for raw in fp:
            raw = raw.strip()
            if not raw:
                continue
            try:
                ev = json.loads(raw)
            except json.JSONDecodeError:
                continue
            kind = ev.get("ev")
            if kind == "boot":
                pt.boot_perf = float(ev["ts"])
                pt.boot_wall = float(ev["wall"])
                pt.offset = pt.boot_wall - pt.boot_perf
            elif kind == "step":
                pt.steps.append(ev)
            elif kind == "step_done":
                pt.step_dones.append(ev)
            elif kind == "recv_start":
                pt.recv_starts.append(ev)
            elif kind == "recv_done":
                pt.recv_dones.append(ev)
    # Translate `ts` from per-process perf_counter to absolute wall time.
    for arr in (pt.steps, pt.step_dones, pt.recv_starts, pt.recv_dones):
        for ev in arr:
            ev["ts"] = float(ev["ts"]) + pt.offset
    return pt


def _gather_traces(trace_dir: Path) -> list[ProcessTrace]:
    files = sorted(trace_dir.glob("role=*.jsonl"))
    if not files:
        raise SystemExit(f"no trace files found under {trace_dir}")
    traces: list[ProcessTrace] = []
    for f in files:
        try:
            traces.append(_parse_jsonl(f))
        except Exception as exc:
            print(f"warning: skipping {f}: {exc}", file=sys.stderr)
    return traces


def _build_per_request_itl(decode_traces: list[ProcessTrace]
                           ) -> dict[str, list[tuple[float, float]]]:
    """Return {req_id -> [(wall_ts_of_token_k, itl_seconds_for_token_k)]}.

    `itl_seconds_for_token_k` is wall_ts[k] - wall_ts[k-1] for the k-th
    emitted token of that request (k >= 1). We attribute each token of a
    step to that step's timestamp -- which is a slight smoothing for
    speculative decoding (multi-token steps) but does not affect the
    aggregate ITL signal.
    """
    # Collect per-request ordered (ts, n_new_this_step) across all decode DP
    # ranks. In practice a given request only lives on one decode DP rank,
    # so cross-rank ordering is trivial, but we sort defensively.
    per_req: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for tr in decode_traces:
        for ev in tr.steps:
            ts = ev["ts"]
            for rid, n in ev["tokens"]:
                if n > 0:
                    per_req[rid].append((ts, int(n)))
    out: dict[str, list[tuple[float, float]]] = {}
    for rid, samples in per_req.items():
        samples.sort(key=lambda x: x[0])
        # The first token-emission step for a P/D request is the first
        # decoded token after KV-pull; "ITL" for that token has no
        # predecessor, so we skip it. We do count all subsequent inter-step
        # gaps. For steps emitting m>1 tokens (speculative), we assign each
        # of the m gaps the average step gap (gap/m).
        timeline: list[tuple[float, float]] = []
        prev_ts: float | None = None
        for ts, n in samples:
            if prev_ts is not None:
                step_gap = ts - prev_ts
                # split evenly across n new tokens this step
                per_tok = step_gap / max(n, 1)
                for k in range(n):
                    timeline.append((ts, per_tok))
            else:
                # skip the first token of this request (no ITL)
                pass
            prev_ts = ts
        if timeline:
            out[rid] = timeline
    return out


def _bin_itl_quantiles(per_req_itl: dict[str, list[tuple[float, float]]],
                       t0: float, t1: float, bin_s: float
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                  np.ndarray]:
    """Bin all ITL samples by their wall-clock timestamp.

    Returns (bin_centers, p50, p90, p99, count) arrays.
    Bins with zero samples emit NaN for the quantiles.
    """
    n_bins = max(1, int(np.ceil((t1 - t0) / bin_s)))
    edges = t0 + bin_s * np.arange(n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    buckets: list[list[float]] = [[] for _ in range(n_bins)]
    for samples in per_req_itl.values():
        for ts, itl in samples:
            idx = int((ts - t0) / bin_s)
            if 0 <= idx < n_bins:
                buckets[idx].append(itl)
    p50 = np.full(n_bins, np.nan)
    p90 = np.full(n_bins, np.nan)
    p99 = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=np.int64)
    for i, b in enumerate(buckets):
        if not b:
            continue
        arr = np.asarray(b, dtype=np.float64)
        p50[i], p90[i], p99[i] = np.quantile(arr, [0.5, 0.9, 0.99])
        counts[i] = arr.size
    return centers, p50, p90, p99, counts


def _union_intervals(intervals: list[tuple[float, float]]
                     ) -> list[tuple[float, float]]:
    """Merge overlapping/adjacent intervals."""
    if not intervals:
        return []
    s = sorted(intervals)
    out = [list(s[0])]
    for a, b in s[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _decode_recv_intervals(decode_traces: list[ProcessTrace]
                           ) -> list[tuple[float, float]]:
    """Per-rank pair each recv_start with the next recv_done for the same
    req_id; collect all (start, done) wall-clock pairs and union them so
    the plot has one set of shaded bands.
    """
    pairs: list[tuple[float, float]] = []
    for tr in decode_traces:
        starts_by_req: dict[str, list[float]] = defaultdict(list)
        for ev in tr.recv_starts:
            starts_by_req[ev["req"]].append(ev["ts"])
        for ev in tr.recv_dones:
            rid = ev["req"]
            if not starts_by_req.get(rid):
                # Unmatched done (e.g. trace started mid-flight). Skip.
                continue
            s = starts_by_req[rid].pop(0)
            pairs.append((s, ev["ts"]))
    return _union_intervals(pairs)


def _kv_xfer_rows(decode_traces: list[ProcessTrace]) -> list[dict]:
    """Pair recv_start/recv_done per (dp_rank, req_id) and return one row
    per matched pair, suitable for CSV export.

    Columns:
      dp_rank         decode DP rank that issued the recv
      req_id          chat-completion request id
      remote_engine   prefill engine id (uuid_dp<N>) the KV was pulled from
      remote_rank     remote worker rank within that engine
      n_blocks        number of local blocks pulled (None if not recorded)
      recv_start_wall recv-issue wall-clock timestamp (UNIX seconds, float)
      recv_done_wall  recv-complete wall-clock timestamp
      latency_ms      recv_done_wall - recv_start_wall, in milliseconds

    Unmatched events (trace started mid-flight or recv failed) are skipped.
    """
    rows: list[dict] = []
    for tr in decode_traces:
        # Per (req_id) FIFO of pending recv_start metadata; queue index lets
        # us pair multiple recvs for the same req_id in order (rare with
        # NIXL but possible if the connector retries).
        pending: dict[str, list[dict]] = defaultdict(list)
        for ev in tr.recv_starts:
            pending[ev["req"]].append(ev)
        for ev in tr.recv_dones:
            rid = ev["req"]
            if not pending.get(rid):
                continue
            s = pending[rid].pop(0)
            rows.append({
                "dp_rank": tr.dp,
                "req_id": rid,
                "remote_engine": s.get("remote_engine"),
                "remote_rank": s.get("remote_rank"),
                "n_blocks": s.get("n_blocks"),
                "recv_start_wall": s["ts"],
                "recv_done_wall": ev["ts"],
                "latency_ms": (ev["ts"] - s["ts"]) * 1000.0,
            })
    return rows


def _kv_xfer_latencies_ms(decode_traces: list[ProcessTrace]
                          ) -> list[float]:
    """Convenience wrapper: just the per-pair latencies (ms) for quantile
    summaries. Pairs by (dp_rank, req_id); unmatched events dropped."""
    return [r["latency_ms"] for r in _kv_xfer_rows(decode_traces)]


def _bin_attribution(decode_traces: list[ProcessTrace],
                     t0: float, t1: float, bin_s: float
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bin the post-step `step_done` events to count, per wall-clock bin,
    how many requests finished and how many KV recvs completed (summed
    across all decode DP ranks). Returns (centers, n_finished, n_recv_done,
    n_step_done_events)."""
    n_bins = max(1, int(np.ceil((t1 - t0) / bin_s)))
    edges = t0 + bin_s * np.arange(n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    nf = np.zeros(n_bins, dtype=np.int64)
    nr = np.zeros(n_bins, dtype=np.int64)
    nev = np.zeros(n_bins, dtype=np.int64)
    for tr in decode_traces:
        for ev in tr.step_dones:
            ts = ev["ts"]
            idx = int((ts - t0) / bin_s)
            if 0 <= idx < n_bins:
                nf[idx] += int(ev.get("n_finished", 0))
                nr[idx] += int(ev.get("n_recv_done", 0))
                nev[idx] += 1
    return centers, nf, nr, nev


def _print_quantile_table(label: str, values: list[float], unit: str = "ms"
                          ) -> str:
    """Print a quantile table and return its plain-text representation."""
    if not values:
        s = f"\n=== {label} ===\n  (no samples)\n"
        print(s, end="")
        return s
    arr = np.asarray(values, dtype=np.float64)
    qs = [0.50, 0.75, 0.90, 0.95, 0.99]
    qv = np.quantile(arr, qs)
    s = (
        f"\n=== {label} (n={arr.size}, unit={unit}) ===\n"
        f"  min={arr.min():.3f}  max={arr.max():.3f}  "
        f"mean={arr.mean():.3f}  std={arr.std():.3f}\n"
        f"  p50={qv[0]:.3f}  p75={qv[1]:.3f}  p90={qv[2]:.3f}  "
        f"p95={qv[3]:.3f}  p99={qv[4]:.3f}\n"
    )
    print(s, end="")
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--trace-dir",
        type=Path,
        default=None,
        help="Directory of role=*.jsonl trace files. Defaults to the newest "
             "playground/log/moe_pd/*/pd_trace.",
    )
    ap.add_argument(
        "--bin-ms",
        type=float,
        default=100.0,
        help="Time bin width in milliseconds (default: 100).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: <trace-dir>/../itl_timeseries.png).",
    )
    ap.add_argument(
        "--max-itl-ms",
        type=float,
        default=200.0,
        help="Y-axis cap in ms; outliers are clipped for readability "
             "(default: 200).",
    )
    ap.add_argument(
        "--clip-from-aiperf",
        type=Path,
        default=None,
        help="Path to an aiperf profile_export_aiperf.json. If set, clip the "
             "plot's wall-clock window to its [start_time, end_time] range. "
             "Useful for plotting only the 'actual experiment' window after a "
             "separate warmup run on the same long-lived server (so the trace "
             "file accumulates both runs, but the plot shows only one).",
    )
    ap.add_argument(
        "--title",
        type=str,
        default=None,
        help="Override the figure title.",
    )
    args = ap.parse_args()

    if args.trace_dir is None:
        candidates = sorted(glob.glob("playground/log/moe_pd/*/pd_trace"),
                            key=os.path.getmtime)
        if not candidates:
            raise SystemExit(
                "no --trace-dir given and no playground/log/moe_pd/*/pd_trace "
                "found"
            )
        args.trace_dir = Path(candidates[-1])
        print(f"[plot] using trace dir: {args.trace_dir}")

    traces = _gather_traces(args.trace_dir)
    decode_traces = [t for t in traces if t.role == "decode"]
    if not decode_traces:
        raise SystemExit("no decode-role traces found; nothing to plot")

    print(f"[plot] found {len(traces)} trace files "
          f"({len(decode_traces)} decode-role)")

    # Optional wall-clock clipping window from an aiperf run. aiperf writes
    # ISO-8601 timestamps in local (naive) time; convert via datetime to UNIX
    # so they compare apples-to-apples with our `boot.wall` (which is
    # `time.time()` in UTC seconds).
    clip_lo: float | None = None
    clip_hi: float | None = None
    if args.clip_from_aiperf is not None:
        try:
            aj = json.loads(args.clip_from_aiperf.read_text())
        except Exception as exc:
            raise SystemExit(
                f"failed to read aiperf json {args.clip_from_aiperf}: {exc}"
            )
        try:
            clip_lo = _dt.datetime.fromisoformat(aj["start_time"]).timestamp()
            clip_hi = _dt.datetime.fromisoformat(aj["end_time"]).timestamp()
        except Exception as exc:
            raise SystemExit(
                f"aiperf json missing/invalid start_time/end_time: {exc}"
            )
        print(f"[plot] clipping to aiperf window: "
              f"{aj['start_time']} -> {aj['end_time']} "
              f"({clip_hi - clip_lo:.2f}s)")
        for tr in decode_traces:
            tr.steps = [ev for ev in tr.steps if clip_lo <= ev["ts"] <= clip_hi]
            tr.step_dones = [
                ev for ev in tr.step_dones if clip_lo <= ev["ts"] <= clip_hi
            ]
            tr.recv_starts = [
                ev for ev in tr.recv_starts if clip_lo <= ev["ts"] <= clip_hi
            ]
            tr.recv_dones = [
                ev for ev in tr.recv_dones if clip_lo <= ev["ts"] <= clip_hi
            ]

    # Determine wall-clock origin: either the clip window's start, or the
    # earliest decode `step` event.
    all_step_ts = [ev["ts"] for tr in decode_traces for ev in tr.steps]
    if not all_step_ts:
        raise SystemExit("no step events in (clipped) decode traces; "
                         "nothing to plot")
    if clip_lo is not None and clip_hi is not None:
        t0, t1 = clip_lo, clip_hi
    else:
        t0 = min(all_step_ts)
        t1 = max(all_step_ts)
    bin_s = args.bin_ms / 1000.0

    per_req_itl = _build_per_request_itl(decode_traces)
    print(f"[plot] reconstructed ITL for {len(per_req_itl)} decode requests "
          f"({sum(len(v) for v in per_req_itl.values())} samples)")

    centers, p50, p90, p99, counts = _bin_itl_quantiles(
        per_req_itl, t0=t0, t1=t1, bin_s=bin_s
    )
    # Convert to ms and to relative seconds.
    centers_rel = centers - t0
    p50_ms = p50 * 1000.0
    p90_ms = p90 * 1000.0
    p99_ms = p99 * 1000.0

    intervals = _decode_recv_intervals(decode_traces)
    print(f"[plot] {len(intervals)} KV-transfer intervals (unioned from "
          f"{sum(len(t.recv_starts) for t in decode_traces)} recv_starts)")

    # KV transfer rows (per-pair, decode side). The full per-request data
    # is persisted as a CSV under playground/out/; the quantile summary is
    # printed to stdout only (no `.txt` sidecar).
    xfer_rows = _kv_xfer_rows(decode_traces)
    _print_quantile_table(
        "KV recv latency (per-request, decode side)",
        [r["latency_ms"] for r in xfer_rows],
        unit="ms",
    )

    # Attribution panel data from `step_done` events (added by the
    # scheduler): per-bin n_finished and n_recv_done across all decode DP
    # ranks. Used to distinguish "EOS clump" vs "KV-recv bookkeeping" vs
    # "other" ITL spikes.
    centers_attr, nf, nr, nev = _bin_attribution(
        decode_traces, t0=t0, t1=t1, bin_s=bin_s
    )
    has_attribution = bool(nev.sum() > 0)
    if not has_attribution:
        print("[plot] no step_done events found (older trace?); "
              "skipping attribution panel")

    # ---------------- figure ----------------
    n_panels = 3 if has_attribution else 2
    heights = [4, 1, 1] if has_attribution else [4, 1]
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(12, 7.5 if has_attribution else 6.5),
        gridspec_kw={"height_ratios": heights},
        sharex=True,
    )
    ax = axes[0]
    ax2 = axes[1]
    ax3 = axes[2] if has_attribution else None
    # Shade KV-transfer windows.
    for a, b in intervals:
        ax.axvspan(a - t0, b - t0, color="0.85", alpha=0.6, zorder=0)
    # Decorative legend handle for the shaded band.
    if intervals:
        ax.axvspan(np.nan, np.nan, color="0.85", alpha=0.6,
                   label="decode NIXL recv active")
    ax.plot(centers_rel, p50_ms, color="tab:blue", lw=1.4, label="ITL p50")
    ax.plot(centers_rel, p90_ms, color="tab:orange", lw=1.2, label="ITL p90")
    ax.plot(centers_rel, p99_ms, color="tab:red", lw=1.0, label="ITL p99")
    ax.set_ylabel("Inter-Token Latency (ms)")
    ax.set_ylim(0, args.max_itl_ms)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    title = args.title or (
        f"System-wide TPOT/ITL — {args.trace_dir.parent.name}"
    )
    ax.set_title(title)

    # Sample density panel for context.
    ax2.bar(centers_rel, counts, width=bin_s, color="tab:gray", alpha=0.7,
            align="center")
    ax2.set_ylabel(f"tokens / {int(args.bin_ms)} ms")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(0.001, t1 - t0))

    # Attribution panel: n_finished (red) and n_recv_done (blue), both per
    # `--bin-ms` bin, both summed across decode DP ranks. Stacked side by
    # side so it is obvious which one is driving a tail spike.
    if ax3 is not None:
        centers_rel_attr = centers_attr - t0
        w = bin_s * 0.45
        ax3.bar(centers_rel_attr - w / 2, nf, width=w,
                color="tab:red", alpha=0.85, align="center",
                label="n_finished / bin")
        ax3.bar(centers_rel_attr + w / 2, nr, width=w,
                color="tab:blue", alpha=0.85, align="center",
                label="n_recv_done / bin")
        ax3.set_ylabel(f"reqs / {int(args.bin_ms)} ms")
        ax3.set_xlabel("wall-clock time since first decode step (s)")
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc="upper right", fontsize=9)
        ax3.set_xlim(0, max(0.001, t1 - t0))
    else:
        ax2.set_xlabel("wall-clock time since first decode step (s)")

    fig.tight_layout()
    # Default output: playground/out/figures/<dir-name>.png, derived from
    # the clip target when given, otherwise from the trace dir.
    if args.out is None:
        if args.clip_from_aiperf is not None:
            stem = args.clip_from_aiperf.parent.name  # e.g. actual_<TS>
            args.out = Path("playground/out/figures") / f"itl_timeseries_{stem}.png"
        else:
            args.out = (
                Path("playground/out/figures")
                / f"itl_timeseries_{args.trace_dir.parent.name}.png"
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"[plot] wrote {args.out}")
    # Persist all KV-recv pairs as a CSV under playground/out/ (one row per
    # matched recv_start/recv_done across all decode DP ranks). Filename is
    # derived from the clip target when given (so it lines up with the
    # `actual_<TS>` / `warmup_<TS>` aiperf artifact dir), otherwise from
    # the trace dir.
    if args.clip_from_aiperf is not None:
        csv_stem = f"kv_xfer_{args.clip_from_aiperf.parent.name}"
    else:
        csv_stem = f"kv_xfer_{args.trace_dir.parent.name}"
    csv_path = Path("playground/out") / f"{csv_stem}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    _columns = [
        "dp_rank", "req_id", "remote_engine", "remote_rank", "n_blocks",
        "recv_start_wall", "recv_done_wall", "latency_ms",
    ]
    import csv as _csv  # local import; stdlib
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        w = _csv.DictWriter(fp, fieldnames=_columns)
        w.writeheader()
        # Sort by start time for deterministic / browseable output.
        for row in sorted(xfer_rows, key=lambda r: r["recv_start_wall"]):
            w.writerow({k: row.get(k) for k in _columns})
    print(f"[plot] wrote {csv_path}  ({len(xfer_rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
