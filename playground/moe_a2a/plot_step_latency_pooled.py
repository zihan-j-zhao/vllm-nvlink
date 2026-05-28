#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pool multiple AIPerf ``profile_export.jsonl`` runs per condition into a
single time-binned p10/p50/p90 line, one line per condition. Companion to
``plot_step_latency_timeseries.py``.

When to use this vs ``plot_step_latency_timeseries.py``
-------------------------------------------------------

* ``plot_step_latency_timeseries.py``: one curve per ``--jsonl`` (per run);
  useful for inspecting run-to-run variability of a single condition.
* This script: one curve per **group** of jsonls (per condition); useful
  for comparing N conditions with M repeats each, where you want the
  per-condition variability folded into the shaded band rather than into
  separate lines.

Pooling semantics
-----------------

For each ``--group``, every input jsonl is loaded via
``plot_step_latency_timeseries._load_samples``, which re-bases its
``(t, itl_ms)`` samples to the earliest ``request_start_ns`` *within that
run*. All re-based samples are then dropped into a single dict keyed by
``int(t // bin_s)``; per-bin nearest-rank p10/p50/p90 are computed on the
pooled values (after sorting), and bins with fewer than ``--min-count``
samples are dropped.

This treats the M repeats of a condition as i.i.d. samples of the same
underlying wall-clock-relative distribution (a fair assumption when each
AIPerf run is an independent Poisson stream against the same warmed
server). The p10..p90 band therefore tightens by roughly ``sqrt(M)`` vs a
single-run band -- which is the whole point of having repeats.

Usage
-----

Each ``--group`` takes the form ``"LABEL=GLOB"`` (or ``LABEL=PATH``); the
glob is expanded to one or more jsonl files. Repeat ``--group`` once per
condition; the order on the command line is the order in the legend / on
the color cycle.

::

    python playground/moe_a2a/plot_step_latency_pooled.py \\
        --group "baseline=playground/out/aiperf/cgnp_base_*/profile_export.jsonl" \\
        --group "KV via copy engine=playground/out/aiperf/cgnp_withkv2way_sat_*/profile_export.jsonl" \\
        --group "KV via NCCL=playground/out/aiperf/cgnp_nccl_withkv2way_sat_*/profile_export.jsonl" \\
        --bin-ms 200 --xmax 60 --yscale log --ymin 8 --ymax 2000 \\
        --output playground/out/figures/step_latency_3way.png
"""

from __future__ import annotations

import argparse
import glob as globmod
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the AIPerf JSONL parser + per-run origin alignment from the
# single-run timeseries plot.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_step_latency_timeseries import _load_samples  # noqa: E402


def _parse_group(spec: str) -> tuple[str, list[Path]]:
    """Parse a ``LABEL=GLOB`` group spec into ``(label, [paths...])``."""
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"--group spec must be 'LABEL=GLOB', got: {spec!r}"
        )
    label, pattern = spec.split("=", 1)
    label = label.strip()
    pattern = pattern.strip()
    if not label:
        raise argparse.ArgumentTypeError(f"--group {spec!r}: empty label")
    if not pattern:
        raise argparse.ArgumentTypeError(f"--group {spec!r}: empty path/glob")
    matches = sorted(globmod.glob(pattern))
    if not matches:
        # Treat as literal path; let downstream open() raise the real error.
        matches = [pattern]
    return label, [Path(p) for p in matches]


def _per_bin_stats(
    paths: list[Path],
    bin_s: float,
    min_count: int,
    t_min: float | None,
    t_max: float | None,
    lo_q: float,
    hi_q: float,
) -> tuple[list[float], list[float], list[float], list[float], int, int]:
    """Pool samples across runs and return per-bin (centers, p50, p_lo, p_hi).

    Also returns ``(n_runs, n_samples_total)`` for diagnostics.
    """
    buckets: dict[int, list[float]] = defaultdict(list)
    n_samples_total = 0
    for p in paths:
        samples, _ = _load_samples(p)
        for t, v in samples:
            if t_min is not None and t < t_min:
                continue
            if t_max is not None and t > t_max:
                continue
            buckets[int(t // bin_s)].append(v)
            n_samples_total += 1

    centers: list[float] = []
    p50: list[float] = []
    p_lo: list[float] = []
    p_hi: list[float] = []
    for k in sorted(buckets):
        vals = buckets[k]
        if len(vals) < min_count:
            continue
        vals.sort()
        n = len(vals)
        # Nearest-rank percentiles; matches plot_step_latency_timeseries.py.
        centers.append((k + 0.5) * bin_s)
        p50.append(vals[n // 2])
        p_lo.append(vals[max(0, int(lo_q * n))])
        p_hi.append(vals[min(n - 1, int(hi_q * n))])
    return centers, p50, p_lo, p_hi, len(paths), n_samples_total


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--group",
        action="append",
        required=True,
        type=_parse_group,
        help=(
            "Repeat once per condition. Each value is 'LABEL=GLOB', where "
            "GLOB expands to one or more AIPerf profile_export.jsonl paths "
            "to pool. Order on the command line = legend / color order."
        ),
    )
    ap.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output PNG path (parent dirs are created).",
    )
    ap.add_argument(
        "--bin-ms", type=float, default=200.0,
        help="Time bin width in milliseconds (default 200).",
    )
    ap.add_argument(
        "--min-count", type=int, default=30,
        help=(
            "Drop bins with fewer than this many pooled samples (default "
            "30). Reduces visual noise at the run's prefill ramp-up / drain."
        ),
    )
    ap.add_argument(
        "--lo-quantile", type=float, default=0.10,
        help="Lower edge of the shaded band per bin (default 0.10).",
    )
    ap.add_argument(
        "--hi-quantile", type=float, default=0.90,
        help="Upper edge of the shaded band per bin (default 0.90).",
    )
    ap.add_argument(
        "--xmin", type=float, default=0.0,
        help="x-axis min in seconds (default 0).",
    )
    ap.add_argument(
        "--xmax", type=float, default=60.0,
        help="x-axis max in seconds (default 60). Use 0 to disable.",
    )
    ap.add_argument(
        "--yscale", choices=("linear", "log"), default="log",
        help="y-axis scale (default log; better when conditions span >1 decade).",
    )
    ap.add_argument("--ymin", type=float, default=None, help="y-axis min (ms).")
    ap.add_argument("--ymax", type=float, default=None, help="y-axis max (ms).")
    ap.add_argument(
        "--title", type=str, default=None,
        help=(
            "Custom figure title. Defaults to a concise description "
            "including bin width and band quantiles."
        ),
    )
    ap.add_argument(
        "--figsize", type=str, default="13,6",
        help="Matplotlib figsize as 'W,H' inches (default '13,6').",
    )
    ap.add_argument(
        "--legend-loc", type=str, default="lower right",
        help="Matplotlib legend loc string (default 'lower right').",
    )
    args = ap.parse_args()

    try:
        figsize = tuple(float(x) for x in args.figsize.split(","))
        assert len(figsize) == 2
    except Exception:
        print(f"error: --figsize {args.figsize!r} must be 'W,H'", file=sys.stderr)
        return 2

    xmax = None if args.xmax == 0 else args.xmax
    bin_s = args.bin_ms / 1000.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    summary: list[str] = []
    any_data = False
    for i, (label, paths) in enumerate(args.group):
        centers, p50, p_lo, p_hi, n_runs, n_samples = _per_bin_stats(
            paths, bin_s, args.min_count,
            t_min=args.xmin, t_max=xmax,
            lo_q=args.lo_quantile, hi_q=args.hi_quantile,
        )
        if not centers:
            print(
                f"warning: group {label!r}: no bins with >= {args.min_count} "
                "samples; skipping",
                file=sys.stderr,
            )
            continue
        any_data = True
        color = color_cycle[i % len(color_cycle)]
        ax.fill_between(centers, p_lo, p_hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(centers, p50, color=color, linewidth=1.8, label=label)
        summary.append(
            f"  {label}: pooled p50 over bins = "
            f"{statistics.median(p50):.2f} ms "
            f"({len(centers)} bins, {n_runs} runs, {n_samples} samples)"
        )

    if not any_data:
        print("error: no group had plottable data", file=sys.stderr)
        return 1

    if xmax is not None:
        ax.set_xlim(args.xmin, xmax)
    ax.set_yscale(args.yscale)
    if args.ymin is not None or args.ymax is not None:
        ax.set_ylim(bottom=args.ymin, top=args.ymax)
    ax.set_xlabel("wall-clock time since first request_start (s)")
    ax.set_ylabel(
        f"inter-token latency, per-bin p50 (ms"
        + (", log scale" if args.yscale == "log" else "")
        + ")"
    )
    if args.title is None:
        lo_pct = int(round(args.lo_quantile * 100))
        hi_pct = int(round(args.hi_quantile * 100))
        title = (
            "Decode step latency: per-condition pooled across runs "
            f"({int(args.bin_ms)} ms bins, line = p50, "
            f"shaded = p{lo_pct}..p{hi_pct})"
        )
    else:
        title = args.title
    ax.set_title(title)
    ax.legend(loc=args.legend_loc, fontsize=10)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.4)

    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"wrote {args.output}")
    for ln in summary:
        print(ln)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
