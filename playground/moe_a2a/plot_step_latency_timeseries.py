#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot system-view decode step latency vs wall-clock time from AIPerf
``profile_export.jsonl``.

Why this works
--------------

In OpenAI chat streaming, every in-flight request receives one decode token
per vLLM forward step, and each token is delivered as one SSE chunk. The
``inter_chunk_latency`` (ICL) array AIPerf records for a request is therefore
a sequence of per-decode-step gaps in milliseconds. Pooling ICL samples from
every concurrent request and bucketing them by wall-clock time yields an
estimate of "average decode step latency at time t" purely from client-side
data, with no server instrumentation required.

The formula for the arrival timestamp of ICL[i] (i.e., the wall-clock at
which step i+1 ended for request r) is::

    t_{r,i} = request_start_ns_r
            + 1e6 * (time_to_first_token_r + sum(ICL_r[0..i]))

We bin all ``(t_{r,i}, ICL_r[i])`` samples by ``--bin-ms`` (default 100 ms)
and plot the per-bin p50 as a line, with a shaded p10..p90 band. One line
per input JSONL (one per run); the time origin per run is the earliest
``request_start_ns`` seen in that file.

Caveats
-------

* ICL[0] is the gap between the prefill-end chunk and the first decode
  chunk; it is still a decode-step gap (the request had just joined the
  active batch), so we keep it.
* SSE ``[DONE]`` chunks can cause ``http_req_chunks_received`` to exceed
  ``len(ICL) + 1`` by one. The reconstruction only walks ICL so this is
  harmless.
* The chunk-per-token assumption breaks if the server coalesces decode
  tokens into a single SSE event. AIPerf's per-request
  ``output_sequence_length`` should roughly equal ``len(ICL) + 1``; we
  print a one-line warning if the average ratio across the run is < 0.95.
* Bin granularity of 100 ms with ~20 req/s and ~tens of in-flight
  requests gives O(tens) of samples per bin in the steady state; the
  p10/p90 band is informative only where the bin is well-populated.

Usage
-----

::

    python playground/moe_a2a/plot_step_latency_timeseries.py \\
        --jsonl run_a/profile_export.jsonl \\
                run_b/profile_export.jsonl \\
                run_c/profile_export.jsonl \\
        --label "run A" "run B" "run C" \\
        --output playground/out/figures/step_latency_timeseries.png
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_samples(
    jsonl_path: Path,
) -> tuple[list[tuple[float, float]], dict[str, float]]:
    """Return ``([(t_sec_since_run_start, itl_ms), ...], diagnostics)``.

    ``t_sec_since_run_start`` is wall-clock seconds since the earliest
    ``request_start_ns`` in this file.
    """
    rows: list[dict] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        raise SystemExit(f"{jsonl_path}: no rows")

    # Origin = earliest request_start_ns across all requests in this run.
    origin_ns = min(r["metadata"]["request_start_ns"] for r in rows)

    samples: list[tuple[float, float]] = []
    chunk_token_ratio_num = 0.0
    chunk_token_ratio_den = 0
    skipped = 0

    for r in rows:
        meta = r["metadata"]
        m = r["metrics"]
        icl = m.get("inter_chunk_latency", {}).get("value")
        if not icl:
            skipped += 1
            continue
        ttft_ms = m["time_to_first_token"]["value"]
        req_start_ns = meta["request_start_ns"]

        # Per-request t=0 is request_start_ns, but for plotting we shift
        # to run-relative seconds.
        req_start_s = (req_start_ns - origin_ns) / 1e9
        ttft_s = ttft_ms / 1e3

        # Cumulative chunk arrival, in run-relative seconds. ICL[i] is
        # the gap ending at chunk i+1.
        t_cum_s = req_start_s + ttft_s
        for gap_ms in icl:
            t_cum_s += gap_ms / 1e3
            samples.append((t_cum_s, float(gap_ms)))

        osl = m.get("output_sequence_length", {}).get("value")
        if osl is not None and osl >= 2:
            # len(icl)+1 chunks ought to map to ~osl tokens.
            chunk_token_ratio_num += (len(icl) + 1) / osl
            chunk_token_ratio_den += 1

    diag = {
        "n_rows": len(rows),
        "n_skipped": skipped,
        "n_samples": len(samples),
        "origin_ns": origin_ns,
        "chunk_per_token_ratio": (
            chunk_token_ratio_num / chunk_token_ratio_den
            if chunk_token_ratio_den
            else float("nan")
        ),
    }
    return samples, diag


def _bin_stats(
    samples: list[tuple[float, float]],
    bin_s: float,
    min_count: int,
) -> tuple[list[float], list[float], list[float], list[float], list[int]]:
    """Bucket ``(t_s, itl_ms)`` samples and return p50/p10/p90/count per bin.

    Returns ``(bin_center_s, p50, p10, p90, count)`` with one entry per
    non-empty, sufficiently-populated bin.
    """
    buckets: dict[int, list[float]] = defaultdict(list)
    for t_s, v in samples:
        buckets[int(t_s // bin_s)].append(v)
    keys = sorted(buckets)
    centers: list[float] = []
    p50: list[float] = []
    p10: list[float] = []
    p90: list[float] = []
    counts: list[int] = []
    for k in keys:
        vals = buckets[k]
        if len(vals) < min_count:
            continue
        vals.sort()
        n = len(vals)
        # Plain nearest-rank percentiles; fast and adequate for plotting.
        p50.append(vals[n // 2])
        p10.append(vals[max(0, int(0.10 * n))])
        p90.append(vals[min(n - 1, int(0.90 * n))])
        centers.append((k + 0.5) * bin_s)
        counts.append(n)
    return centers, p50, p10, p90, counts


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--jsonl",
        nargs="+",
        required=True,
        type=Path,
        help="One or more AIPerf profile_export.jsonl files (one per run).",
    )
    ap.add_argument(
        "--label",
        nargs="*",
        default=None,
        help="Per-run line labels. Defaults to parent dir name of each --jsonl.",
    )
    ap.add_argument(
        "--bin-ms",
        type=float,
        default=100.0,
        help="Bin width in milliseconds (default: 100).",
    )
    ap.add_argument(
        "--min-count",
        type=int,
        default=5,
        help=(
            "Drop bins with fewer than this many samples (default: 5). "
            "Reduces visual noise at the run's prefill ramp-up / drain tail."
        ),
    )
    ap.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output PNG path (parent dirs are created).",
    )
    ap.add_argument(
        "--title",
        type=str,
        default=(
            "System decode step latency vs wall-clock time "
            "(AIPerf inter_chunk_latency, pooled across requests)"
        ),
    )
    ap.add_argument(
        "--ymax",
        type=float,
        default=None,
        help="Optional fixed y-axis max in ms (otherwise auto).",
    )
    ap.add_argument(
        "--xmin", type=float, default=None,
        help="Optional fixed x-axis min in seconds (otherwise auto).",
    )
    ap.add_argument(
        "--xmax", type=float, default=None,
        help="Optional fixed x-axis max in seconds (otherwise auto).",
    )
    ap.add_argument(
        "--transfer-log",
        nargs="+",
        default=None,
        type=Path,
        help=(
            "Optional per-run KV-transfer session JSONL paths (one per "
            "--jsonl, same order). Each file is one or more records with "
            "start_ns/end_ns; the resulting wall-clock windows are shaded "
            "on the figure aligned to that run's origin (the run's "
            "earliest request_start_ns). Produced by "
            "simulate_kv_transfer.py."
        ),
    )
    args = ap.parse_args()

    if args.label is not None and len(args.label) != len(args.jsonl):
        print(
            f"error: --label has {len(args.label)} values but --jsonl has "
            f"{len(args.jsonl)}",
            file=sys.stderr,
        )
        return 2
    if args.transfer_log is not None and len(args.transfer_log) != len(args.jsonl):
        print(
            f"error: --transfer-log has {len(args.transfer_log)} values "
            f"but --jsonl has {len(args.jsonl)}",
            file=sys.stderr,
        )
        return 2
    labels = args.label or [p.parent.name for p in args.jsonl]

    bin_s = args.bin_ms / 1000.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)

    # Matplotlib's default property cycle gives us distinct colors per run;
    # we pin the fill color to the line color so the band matches the run.
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    summary_lines: list[str] = []
    any_data = False
    for i, (jsonl, label) in enumerate(zip(args.jsonl, labels)):
        samples, diag = _load_samples(jsonl)
        if diag["chunk_per_token_ratio"] < 0.95:
            print(
                f"warning: {jsonl}: (len(ICL)+1)/output_tokens averaged "
                f"{diag['chunk_per_token_ratio']:.3f}; some chunks may "
                "carry multiple tokens, which biases ITL low.",
                file=sys.stderr,
            )
        centers, p50, p10, p90, counts = _bin_stats(
            samples, bin_s=bin_s, min_count=args.min_count
        )
        if not centers:
            print(f"warning: {jsonl}: no bins with >= {args.min_count} samples; "
                  "skipping in plot", file=sys.stderr)
            continue
        any_data = True
        color = color_cycle[i % len(color_cycle)]
        ax.plot(centers, p50, label=label, color=color, linewidth=1.5)
        ax.fill_between(centers, p10, p90, color=color, alpha=0.15, linewidth=0)
        summary_lines.append(
            f"  {label}: rows={diag['n_rows']} samples={diag['n_samples']} "
            f"plotted_bins={len(centers)} "
            f"median_itl_ms={statistics.median(p50):.2f}"
        )

        # If a transfer log was given for this run, draw shaded vertical
        # bands for each transfer session aligned to this run's origin.
        # Each record in the JSONL is one session with {start_ns, end_ns}.
        # We draw bands in a neutral gray (not the run color) and at low
        # alpha; when multiple runs' bands stack we still want the data
        # lines to stay legible.
        if args.transfer_log is not None:
            tlog = args.transfer_log[i]
            origin_ns = diag["origin_ns"]
            sessions: list[tuple[float, float]] = []
            try:
                with open(tlog) as tf:
                    for line in tf:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        s = (rec["start_ns"] - origin_ns) / 1e9
                        e = (rec["end_ns"] - origin_ns) / 1e9
                        sessions.append((s, e))
            except FileNotFoundError:
                print(
                    f"warning: transfer log {tlog} not found; skipping band",
                    file=sys.stderr,
                )
            for s, e in sessions:
                ax.axvspan(s, e, color="0.55", alpha=0.08, linewidth=0, zorder=0)
            if sessions:
                summary_lines.append(
                    f"      transfer windows: "
                    + ", ".join(f"{s:.1f}-{e:.1f}s" for s, e in sessions)
                )

    if not any_data:
        print("error: no run had plottable data", file=sys.stderr)
        return 1

    ax.set_xlabel("wall-clock time since first request_start (s)")
    ax.set_ylabel(f"inter-token latency, per-bin p50 (ms)  [bin={args.bin_ms:g} ms]")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.3)
    # Proxy legend handle for the transfer-window shading, if any was drawn.
    if args.transfer_log is not None:
        from matplotlib.patches import Patch
        handles, lbls = ax.get_legend_handles_labels()
        handles.append(Patch(facecolor="0.55", alpha=0.30, linewidth=0))
        lbls.append("KV transfer active")
        ax.legend(handles, lbls, loc="best")
    else:
        ax.legend(loc="best")
    if args.ymax is not None:
        ax.set_ylim(top=args.ymax)
    ax.set_ylim(bottom=0)
    if args.xmin is not None or args.xmax is not None:
        ax.set_xlim(left=args.xmin, right=args.xmax)

    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"wrote {args.output}")
    print("per-run summary:")
    for ln in summary_lines:
        print(ln)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
