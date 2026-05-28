#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone GPU-to-GPU NVLink P2P bandwidth/latency benchmark.

Sweeps transfer sizes from a few KiB up to ~1 GiB, performing a tight
``torch.Tensor.copy_()`` (which reduces to ``cudaMemcpyPeerAsync`` over
NVLink when peer access is enabled) for each size. Per-call kernel time
is measured with paired CUDA events on a dedicated stream, the same way
``cuda_communicator`` measures MoE all-to-all calls.

What this script is NOT
-----------------------

This is **not** vLLM MoE all-to-all. There is no NCCL collective, no
peer-rank synchronization, no Python dispatch jitter per call. The
result is a clean upper bound on what a same-node NVLink P2P copy can
achieve, suitable as a reference curve for MoE-collective measurements.

Output
------

* A 1x2 PNG (left: size vs time; right: size vs effective bandwidth).
* Optional raw CSV with ``size_bytes, iter_idx, time_ms`` per sample.

Example::

    python playground/moe_a2a/benchmark_p2p_transfer.py \\
        --src-gpu 0 --dst-gpu 1 \\
        --output playground/out/figures/p2p_benchmark.png
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def _geomspace_bytes(min_kib: int, max_mib: int, points: int) -> list[int]:
    """Return ``points`` byte sizes log-spaced from ``min_kib`` KiB to ``max_mib`` MiB."""
    import math

    lo = math.log10(min_kib * 1024)
    hi = math.log10(max_mib * 1024 * 1024)
    return [int(round(10 ** (lo + (hi - lo) * i / (points - 1)))) for i in range(points)]


def benchmark(
    src_gpu: int,
    dst_gpu: int,
    sizes_bytes: list[int],
    iters: int,
    warmup: int,
) -> list[tuple[int, int, float]]:
    """Run the size sweep. Returns a list of ``(size_bytes, iter_idx, time_ms)``."""
    src_dev = torch.device("cuda", src_gpu)
    dst_dev = torch.device("cuda", dst_gpu)

    can_peer = torch.cuda.can_device_access_peer(src_gpu, dst_gpu)
    if not can_peer:
        print(
            f"warning: GPU{src_gpu} cannot peer-access GPU{dst_gpu}; copies "
            "will bounce through host memory (no NVLink path)",
            file=sys.stderr,
        )

    max_bytes = max(sizes_bytes)
    print(
        f"[bench] src=cuda:{src_gpu} dst=cuda:{dst_gpu} peer_access={can_peer} "
        f"max_size={max_bytes / 1024**2:.0f} MiB iters={iters} warmup={warmup}",
        flush=True,
    )

    # uint8 buffers: 1 byte/element. Allocate once at the max size and
    # narrow for smaller probes — avoids reallocating per size.
    src_buf = torch.empty(max_bytes, dtype=torch.uint8, device=src_dev)
    dst_buf = torch.empty(max_bytes, dtype=torch.uint8, device=dst_dev)
    src_buf.fill_(0xAB)

    # Dedicated copy stream on the destination device so events bracket
    # only the memcpy.
    copy_stream = torch.cuda.Stream(device=dst_dev)

    records: list[tuple[int, int, float]] = []
    for size in sizes_bytes:
        s_view = src_buf[:size]
        d_view = dst_buf[:size]

        # Warm up.
        with torch.cuda.stream(copy_stream):
            for _ in range(warmup):
                d_view.copy_(s_view, non_blocking=True)
        torch.cuda.synchronize(dst_dev)

        # Time `iters` copies. Allocate event pairs up front so the
        # measurement loop is uniform.
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        with torch.cuda.stream(copy_stream):
            for i in range(iters):
                starts[i].record()
                d_view.copy_(s_view, non_blocking=True)
                ends[i].record()
        torch.cuda.synchronize(dst_dev)

        for i in range(iters):
            ms = float(starts[i].elapsed_time(ends[i]))
            records.append((size, i, ms))

        # Brief sanity print as we go.
        per_size_ms = sorted([records[-iters + i][2] for i in range(iters)])
        med = per_size_ms[iters // 2]
        bw = size / 1e9 / (med / 1000.0)
        print(
            f"[bench]  size={size / 1024:>10.1f} KiB  "
            f"median_time={med:7.4f} ms  median_bw={bw:7.2f} GB/s",
            flush=True,
        )

    return records


def write_csv(records: list[tuple[int, int, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["size_bytes", "iter_idx", "time_ms"])
        for row in records:
            w.writerow(row)
    print(f"wrote {path}")


def plot(
    records: list[tuple[int, int, float]],
    output: Path,
    bw_peak_gb_s: float,
) -> None:
    # Group by size for per-size stats.
    from collections import defaultdict

    by_size: dict[int, list[float]] = defaultdict(list)
    for sz, _, ms in records:
        by_size[sz].append(ms)
    sizes = sorted(by_size)

    sizes_mib = [s / (1024 * 1024) for s in sizes]
    # Worst (slowest) and best (fastest) per size, for both panels.
    best_time_ms = [min(by_size[s]) for s in sizes]
    worst_time_ms = [max(by_size[s]) for s in sizes]
    med_time_ms = [statistics.median(by_size[s]) for s in sizes]
    # Bandwidth: best = fastest time (highest bw), worst = slowest (lowest bw).
    best_bw = [s / 1e9 / (min(by_size[s]) / 1000.0) for s in sizes]
    worst_bw = [s / 1e9 / (max(by_size[s]) / 1000.0) for s in sizes]
    med_bw = [s / 1e9 / (m / 1000.0) for s, m in zip(sizes, med_time_ms)]
    iters_per_size = len(next(iter(by_size.values())))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_t, ax_bw) = plt.subplots(
        1, 2, figsize=(12, 4.5), constrained_layout=True
    )

    # Time panel: worst-best envelope + median line. No legend.
    ax_t.fill_between(
        sizes_mib,
        best_time_ms,
        worst_time_ms,
        color="tab:blue", alpha=0.30, linewidth=0,
    )
    ax_t.plot(
        sizes_mib, med_time_ms,
        color="tab:blue", linewidth=1.8, marker="o", markersize=3,
    )
    ax_t.set_xscale("log")
    ax_t.set_xlabel("Transfer Size (MiB)")
    ax_t.set_ylabel("Transfer Time (ms)")
    ax_t.set_title(
        f"NVLink P2P: transfer time vs size\n"
        f"(line = median, band = best–worst across {iters_per_size} iters/size)"
    )
    ax_t.set_ylim(bottom=0)

    # Inset: zoom into the latency-bound region x ∈ [1, 10] MiB, where the
    # parent plot shows only a near-zero flat segment. Placed in the upper-
    # left whitespace of the left subplot.
    _zoom_lo, _zoom_hi = 1.0, 10.0
    _band_idx = [
        i for i, s in enumerate(sizes_mib) if _zoom_lo <= s <= _zoom_hi
    ]
    if _band_idx:
        ins = ax_t.inset_axes([0.08, 0.50, 0.42, 0.42])
        ins.fill_between(
            sizes_mib, best_time_ms, worst_time_ms,
            color="tab:blue", alpha=0.30, linewidth=0,
        )
        ins.plot(
            sizes_mib, med_time_ms,
            color="tab:blue", linewidth=1.4, marker="o", markersize=2.5,
        )
        ins.set_xscale("log")
        ins.set_xlim(_zoom_lo, _zoom_hi)
        _y_in_band = [worst_time_ms[i] for i in _band_idx]
        ins.set_ylim(0, max(_y_in_band) * 1.15)
        # Keep tick labels minimal so the inset stays readable.
        from matplotlib.ticker import FixedLocator, NullFormatter
        ins.xaxis.set_major_locator(FixedLocator([1.0, 10.0]))
        ins.xaxis.set_minor_formatter(NullFormatter())
        ins.set_xticklabels(["1", "10"])
        ins.tick_params(labelsize=8)
        ins.set_title("zoom: 1\u201310 MiB", fontsize=9)
        ax_t.indicate_inset_zoom(ins, edgecolor="gray", alpha=0.7)

    # Bandwidth panel: worst-best envelope + median line. No legend.
    ax_bw.fill_between(
        sizes_mib, worst_bw, best_bw,
        color="tab:orange", alpha=0.30, linewidth=0,
    )
    ax_bw.plot(
        sizes_mib, med_bw,
        color="tab:orange", linewidth=1.8, marker="o", markersize=3,
    )
    if bw_peak_gb_s > 0:
        ax_bw.axhline(
            bw_peak_gb_s, color="black", linewidth=0.8, linestyle="--",
            alpha=0.6, zorder=0,
        )
        ax_bw.text(
            sizes_mib[-1], bw_peak_gb_s,
            f"  NVLink peak ≈ {bw_peak_gb_s:g} GB/s",
            ha="right", va="bottom", fontsize=9, color="black", alpha=0.7,
        )
    ax_bw.set_xscale("log")
    ax_bw.set_xlabel("Transfer Size (MiB)")
    ax_bw.set_ylabel("Effective Bandwidth (GB/s)")
    ax_bw.set_title(
        f"NVLink P2P: effective bandwidth vs size\n"
        f"(line = median, band = worst–best across {iters_per_size} iters/size)"
    )
    ax_bw.set_ylim(bottom=0)

    fig.suptitle(
        "Standalone GPU-to-GPU torch.copy_ (cudaMemcpyPeerAsync over NVLink)",
        fontsize=12,
    )
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"wrote {output}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src-gpu", type=int, default=None)
    ap.add_argument("--dst-gpu", type=int, default=None)
    ap.add_argument(
        "--min-kib", type=int, default=1,
        help="Smallest probe size in KiB (default 1).",
    )
    ap.add_argument(
        "--max-mib", type=int, default=8192,
        help="Largest probe size in MiB (default 8192 = 8 GiB).",
    )
    ap.add_argument(
        "--points", type=int, default=28,
        help="Number of log-spaced size points (default 28).",
    )
    ap.add_argument(
        "--iters", type=int, default=200,
        help="Timed iterations per size (default 200).",
    )
    ap.add_argument(
        "--warmup", type=int, default=20,
        help="Warm-up iterations per size (default 20).",
    )
    ap.add_argument(
        "--bw-peak", type=float, default=900.0,
        help="NVLink peak reference line in GB/s (default 900 for B200 NVLink-5; 0 to hide).",
    )
    ap.add_argument("--output", type=Path, required=True, help="Output PNG path.")
    ap.add_argument(
        "--csv-output", type=Path, default=None,
        help="Optional path to dump raw per-iter measurements.",
    )
    ap.add_argument(
        "--replot-from-csv", type=Path, default=None,
        help="Skip the GPU benchmark and rebuild the plot from a CSV "
             "previously written by --csv-output (same schema: "
             "size_bytes,iter_idx,time_ms).",
    )
    args = ap.parse_args()

    if args.replot_from_csv is not None:
        records: list[tuple[int, int, float]] = []
        with open(args.replot_from_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(
                    (int(row["size_bytes"]), int(row["iter_idx"]), float(row["time_ms"]))
                )
        print(f"[bench] loaded {len(records)} samples from {args.replot_from_csv}")
        plot(records, args.output, bw_peak_gb_s=args.bw_peak)
        return 0

    if args.src_gpu is None or args.dst_gpu is None:
        print(
            "error: --src-gpu and --dst-gpu are required unless "
            "--replot-from-csv is given",
            file=sys.stderr,
        )
        return 2
    if args.src_gpu == args.dst_gpu:
        print("error: --src-gpu must differ from --dst-gpu", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("error: CUDA not available", file=sys.stderr)
        return 1
    if max(args.src_gpu, args.dst_gpu) >= torch.cuda.device_count():
        print(
            f"error: requested gpus {args.src_gpu}/{args.dst_gpu} but only "
            f"{torch.cuda.device_count()} visible",
            file=sys.stderr,
        )
        return 1

    sizes = _geomspace_bytes(args.min_kib, args.max_mib, args.points)
    t0 = time.time()
    records = benchmark(
        src_gpu=args.src_gpu, dst_gpu=args.dst_gpu,
        sizes_bytes=sizes, iters=args.iters, warmup=args.warmup,
    )
    print(f"[bench] total wall time: {time.time() - t0:.1f}s; {len(records)} samples")

    if args.csv_output is not None:
        write_csv(records, args.csv_output)
    plot(records, args.output, bw_peak_gb_s=args.bw_peak)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
