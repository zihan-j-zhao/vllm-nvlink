#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone NCCL collective benchmark for ``all_gatherv`` and
``reduce_scatterv``.

Two modes are supported.

Symmetric mode (default, ``--mode symmetric``)
----------------------------------------------

Both ranks pass the *same* per-rank size ``N`` and we sweep ``N`` log-spaced
over a wide range (default 4 KiB .. 256 MiB). This is the canonical
"performance landscape" plot: x-axis is the message size, and we report
both kernel latency (us, log-y) and effective bandwidth (GB/s) side by
side. NCCL protocol is left at ``auto`` -- the protocol/algorithm
transitions show up naturally as kinks/knees in the curves.

Asymmetric mode (``--mode asymmetric``)
---------------------------------------

The original hypothesis sweep: fix ``max_size`` on one rank, vary the
*other side*'s contribution from very small (4 KiB) to ``max_size``. If
kernel time is dominated by ``max(send_size, recv_size)`` per direction,
the resulting time-vs-size curve should be approximately **flat** at the
value for the symmetric (max==max) call. This mode keeps the multi-protocol
sweep for inspecting Simple vs LL128 vs LL behaviour.

Setup
-----

Two processes are spawned (one per GPU), bootstrapped with PyTorch
distributed using the ``gloo`` backend. We then build vLLM's
``PyNcclCommunicator`` for the actual NCCL work (this gives us the
``all_gatherv`` / ``reduce_scatterv`` primitives with per-rank variable
sizes; PyTorch's bare distributed API only exposes the uniform variants).
Per-call kernel time is measured with paired CUDA events on a dedicated
stream, exactly as :mod:`cuda_communicator` measures MoE all-to-all.

For each ``(collective, max_size, min_size)`` triple we run ``--iters``
calls (after a few warm-up) and record one row per iteration. Rank 0
aggregates and writes a JSONL plus the figure.

Examples::

    # Symmetric landscape sweep (the default).
    CUDA_VISIBLE_DEVICES=6,7 python \\
        playground/moe_a2a/benchmark_nccl_collectives.py \\
        --src-gpu 0 --dst-gpu 1 \\
        --iters 50 --warmup 5 \\
        --output playground/out/figures/nccl_collectives_symmetric.png \\
        --csv-output playground/out/nccl_collectives_symmetric.csv

    # Original asymmetric hypothesis sweep with protocol comparison.
    CUDA_VISIBLE_DEVICES=6,7 python \\
        playground/moe_a2a/benchmark_nccl_collectives.py \\
        --mode asymmetric \\
        --src-gpu 0 --dst-gpu 1 \\
        --max-sizes-mib 1,16,256 --min-size-points 12 \\
        --iters 50 --warmup 5 \\
        --output playground/out/figures/nccl_collectives_asym.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

# vLLM transformers_utils hard-imports modelscope if VLLM_USE_MODELSCOPE is
# set; we don't need it. Scrub before importing torch / vllm.
os.environ.setdefault("VLLM_USE_MODELSCOPE", "False")
for _k in ("LMDEPLOY_USE_MODELSCOPE", "MODELSCOPE_CACHE", "MEGATRON_LM_PATH"):
    os.environ.pop(_k, None)

import torch
import torch.multiprocessing as mp


# ---------------------------------------------------------------------------
# Worker (one per GPU)
# ---------------------------------------------------------------------------

def _worker(rank: int, world_size: int, cfg: dict, results_dir: str) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = cfg["master_port"]
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ.setdefault("VLLM_USE_MODELSCOPE", "False")
    # NCCL_PROTO is set in the parent before mp.spawn; children inherit it.
    proto_label = cfg.get("protocol_label", "auto")

    gpu = cfg["gpus"][rank]
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")

    # CPU group bootstrap. PyNCCL only needs the group object for rank
    # discovery; the actual collective calls are pure NCCL on `device`.
    torch.distributed.init_process_group(
        backend="gloo", rank=rank, world_size=world_size
    )
    cpu_group = torch.distributed.group.WORLD

    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    pynccl = PyNcclCommunicator(group=cpu_group, device=device)
    print(f"[rank {rank}] gpu={gpu} pynccl ready", flush=True)

    # Pre-allocate at the maximum needed buffer sizes. AG output = sum(sizes),
    # which is at most 2*max_size for ws=2. RS input = sum(sizes), output =
    # local. So a single 2*max_size buffer covers both panels and either rank.
    max_size_bytes = max(cfg["max_sizes_bytes"])
    big_a = torch.empty(2 * max_size_bytes, dtype=torch.uint8, device=device)
    big_b = torch.empty(2 * max_size_bytes, dtype=torch.uint8, device=device)
    big_a.fill_(0xAB)
    big_b.fill_(0xCD)

    nccl_stream = torch.cuda.Stream(device=device)

    records: list[dict] = []

    def _time_one(coll: str, sizes: list[int]) -> list[float]:
        """Run warmup + iters of `coll` with the given per-rank sizes.

        Returns a list of `iters` per-iter kernel times in ms.
        """
        local_size = sizes[rank]
        peer_size = sizes[1 - rank]
        total = local_size + peer_size

        # Slice the pre-allocated buffers without reallocating.
        if coll == "all_gatherv":
            inp = big_a.narrow(0, 0, local_size)
            out = big_b.narrow(0, 0, total)
        else:  # reduce_scatterv
            inp = big_a.narrow(0, 0, total)
            out = big_b.narrow(0, 0, local_size)

        # Warmup.
        with torch.cuda.stream(nccl_stream):
            for _ in range(cfg["warmup"]):
                if coll == "all_gatherv":
                    pynccl.all_gatherv(out, inp, sizes=sizes)
                else:
                    pynccl.reduce_scatterv(out, inp, sizes=sizes)
        torch.cuda.synchronize(device)

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(cfg["iters"])]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(cfg["iters"])]
        with torch.cuda.stream(nccl_stream):
            for i in range(cfg["iters"]):
                starts[i].record()
                if coll == "all_gatherv":
                    pynccl.all_gatherv(out, inp, sizes=sizes)
                else:
                    pynccl.reduce_scatterv(out, inp, sizes=sizes)
                ends[i].record()
        torch.cuda.synchronize(device)
        return [float(starts[i].elapsed_time(ends[i])) for i in range(cfg["iters"])]

    # The actual sweep.
    for coll in ("all_gatherv", "reduce_scatterv"):
        for max_b in cfg["max_sizes_bytes"]:
            for min_b in cfg["min_sizes_bytes_by_max"][str(max_b)]:
                if min_b > max_b:
                    continue
                # rank 0 = small side, rank 1 = max side.
                # (Both ranks must pass identical `sizes` to NCCL.)
                sizes = [min_b, max_b]
                times = _time_one(coll, sizes)
                for i, ms in enumerate(times):
                    records.append({
                        "rank": rank, "collective": coll,
                        "max_bytes": max_b, "min_bytes": min_b,
                        "iter": i, "time_ms": ms,
                    })
                if rank == 0:
                    med = statistics.median(times)
                    print(
                        f"[rank 0]  proto={proto_label:<8s} {coll:<16s} "
                        f"max={max_b / 1024**2:>8.2f} MiB  "
                        f"min={min_b / 1024**2:>10.5f} MiB  "
                        f"median={med:7.4f} ms",
                        flush=True,
                    )

    if rank == 0:
        with open(os.path.join(results_dir, f"results_{proto_label}.jsonl"), "w") as fp:
            for rec in records:
                rec["protocol"] = proto_label
                fp.write(json.dumps(rec) + "\n")
        print(f"[rank 0] proto={proto_label} wrote {len(records)} records", flush=True)

    torch.distributed.destroy_process_group()


# ---------------------------------------------------------------------------
# Plot (run in the parent process after spawn returns)
# ---------------------------------------------------------------------------

def _plot(
    records: list[dict],
    output: Path,
    max_sizes_bytes: list[int],
    protocols: list[str] | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Two modes:
    # (1) Protocol comparison: a single max_size, multiple protocols -> one
    #     line per protocol per panel.
    # (2) Max-size comparison (legacy): a single 'auto' protocol, multiple
    #     max_sizes -> one line per max_size per panel.
    has_protocols = protocols is not None and len(protocols) > 1
    proto_mode = has_protocols and len(max_sizes_bytes) == 1

    grouped: dict[tuple, list[float]] = defaultdict(list)
    for r in records:
        key = (r["collective"], r.get("protocol", "auto"),
               r["max_bytes"], r["min_bytes"])
        grouped[key].append(r["time_ms"])

    MIB = 1024 * 1024
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    iters_per = len(next(iter(grouped.values())))

    if proto_mode:
        max_b = max_sizes_bytes[0]
        series_keys = [(p, max_b) for p in protocols]
        legend_format = lambda p, m: f"NCCL_PROTO={p}"
        max_tag = f"larger side = {max_b / MIB:g} MiB"
    else:
        series_keys = [("auto", m) for m in max_sizes_bytes]
        legend_format = lambda p, m: f"larger side held at {m / MIB:g} MiB"
        max_tag = ""

    for row_idx, coll in enumerate(("all_gatherv", "reduce_scatterv")):
        ax_t = axes[row_idx, 0]
        ax_bw = axes[row_idx, 1]
        for ci, (proto, max_b) in enumerate(series_keys):
            min_sizes = sorted({
                m for (c, p, M, m) in grouped
                if c == coll and p == proto and M == max_b
            })
            if not min_sizes:
                continue
            xs = [m / MIB for m in min_sizes]
            med_t = [statistics.median(grouped[(coll, proto, max_b, m)]) for m in min_sizes]
            best_t = [min(grouped[(coll, proto, max_b, m)]) for m in min_sizes]
            worst_t = [max(grouped[(coll, proto, max_b, m)]) for m in min_sizes]
            med_bw = [max_b / 1e9 / (t / 1000.0) for t in med_t]
            best_bw = [max_b / 1e9 / (t / 1000.0) for t in best_t]
            worst_bw = [max_b / 1e9 / (t / 1000.0) for t in worst_t]

            color = color_cycle[ci % len(color_cycle)]
            label = legend_format(proto, max_b)

            ax_t.fill_between(xs, best_t, worst_t, color=color, alpha=0.20, linewidth=0)
            ax_t.plot(xs, med_t, color=color, linewidth=1.8, marker="o", markersize=3, label=label)

            ax_bw.fill_between(xs, worst_bw, best_bw, color=color, alpha=0.20, linewidth=0)
            ax_bw.plot(xs, med_bw, color=color, linewidth=1.8, marker="o", markersize=3, label=label)

        ax_t.set_xscale("log")
        ax_t.set_xlabel("Smaller (varied) Side Size (MiB)")
        ax_t.set_ylabel("Kernel Time (ms)")
        title_suffix = max_tag if proto_mode else ""
        ax_t.set_title(
            f"{coll}: time vs smaller-side size"
            + (f" ({title_suffix})" if title_suffix else "")
            + f"\n(line = median, band = best–worst across {iters_per} iters)"
        )
        ax_t.set_ylim(bottom=0)
        ax_t.legend(loc="best", fontsize=9)

        ax_bw.set_xscale("log")
        ax_bw.set_xlabel("Smaller (varied) Side Size (MiB)")
        ax_bw.set_ylabel("Effective BW = larger_side / time (GB/s)")
        ax_bw.set_title(
            f"{coll}: effective bandwidth vs smaller-side size"
            + (f" ({title_suffix})" if title_suffix else "")
        )
        ax_bw.set_ylim(bottom=0)
        ax_bw.legend(loc="best", fontsize=9)

    if proto_mode:
        fig.suptitle(
            "NCCL protocol comparison: 'auto' tracks Simple exactly. The "
            "fast/slow spikes are intrinsic to the chunked-transfer kernel "
            "(Simple and LL128); only LL is smooth (but uniformly slowest).",
            fontsize=10,
        )
    else:
        fig.suptitle(
            "Hypothesis: kernel time ≈ f(larger side); flat curves = supports hypothesis. "
            "Per call, rank 0 contributes the smaller (x-axis) side and rank 1 the larger.",
            fontsize=11,
        )
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"wrote {output}")


# ---------------------------------------------------------------------------
# Plot (symmetric mode: x = message size, y = latency / bandwidth)
# ---------------------------------------------------------------------------

def _plot_symmetric(records: list[dict], output: Path) -> None:
    """Performance-landscape figure for the symmetric (size_per_rank=N) sweep.

    Layout: 2 rows (all_gatherv, reduce_scatterv) x 2 cols
    (kernel latency in us, effective bandwidth in GB/s). The shared x-axis
    is message size per rank in MiB (log). One curve per NCCL protocol:
    'auto' over the full sweep range, and any fixed-protocol sub-sweeps
    (typically Simple / LL128 / LL on a focused sub-range) overlaid.
    Shaded band = best..worst across iters.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Records are keyed by (collective, protocol, N).
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for r in records:
        grouped[(r["collective"], r.get("protocol", "auto"),
                 r["max_bytes"])].append(r["time_ms"])

    # 'auto' first (most prominent), then extras in input order.
    protocols = sorted({p for (_, p, _) in grouped})
    if "auto" in protocols:
        protocols = ["auto"] + [p for p in protocols if p != "auto"]
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    proto_color = {p: color_cycle[i % len(color_cycle)]
                   for i, p in enumerate(protocols)}

    MIB = 1024 * 1024
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        2, 2, figsize=(13, 8), constrained_layout=True, sharex=True,
    )

    # Zoom-inset config: same band as benchmark_p2p_transfer.py.
    _zoom_lo, _zoom_hi = 1.0, 10.0
    from matplotlib.ticker import FixedLocator, NullFormatter

    def _add_zoom_inset(ax, xs_mib, ys, color, ylabel_unit):
        band = [i for i, x in enumerate(xs_mib) if _zoom_lo <= x <= _zoom_hi]
        if not band:
            return
        ins = ax.inset_axes([0.08, 0.50, 0.40, 0.42])
        ins.plot(xs_mib, ys, color=color, linewidth=1.4,
                 marker="o", markersize=2.5)
        ins.set_xscale("log")
        ins.set_xlim(_zoom_lo, _zoom_hi)
        ys_band = [ys[i] for i in band]
        y_lo = min(ys_band) * 0.9
        y_hi = max(ys_band) * 1.1
        ins.set_ylim(y_lo, y_hi)
        ins.xaxis.set_major_locator(FixedLocator([1.0, 10.0]))
        ins.xaxis.set_minor_formatter(NullFormatter())
        ins.set_xticklabels(["1", "10"])
        ins.tick_params(labelsize=8)
        ins.set_title(f"zoom: 1\u201310 MiB ({ylabel_unit})", fontsize=9)
        ax.indicate_inset_zoom(ins, edgecolor="gray", alpha=0.7)

    for row_idx, coll in enumerate(("all_gatherv", "reduce_scatterv")):
        ax_t = axes[row_idx, 0]
        ax_bw = axes[row_idx, 1]
        auto_xs: list[float] | None = None
        auto_med_us: list[float] | None = None
        auto_med_bw: list[float] | None = None
        for proto in protocols:
            sizes = sorted({N for (c, p, N) in grouped
                            if c == coll and p == proto})
            if not sizes:
                continue
            xs_mib = [N / MIB for N in sizes]
            med_ms = [statistics.median(grouped[(coll, proto, N)]) for N in sizes]
            best_ms = [min(grouped[(coll, proto, N)]) for N in sizes]
            worst_ms = [max(grouped[(coll, proto, N)]) for N in sizes]

            med_us = [t * 1e3 for t in med_ms]
            best_us = [t * 1e3 for t in best_ms]
            worst_us = [t * 1e3 for t in worst_ms]
            # Per-rank wire BW: each rank sends N bytes to its peer.
            med_bw = [N / 1e9 / (t / 1000.0) for N, t in zip(sizes, med_ms)]
            best_bw = [N / 1e9 / (t / 1000.0) for N, t in zip(sizes, best_ms)]
            worst_bw = [N / 1e9 / (t / 1000.0) for N, t in zip(sizes, worst_ms)]

            color = proto_color[proto]
            lw = 2.0 if proto == "auto" else 1.4
            label = f"NCCL_PROTO={proto}"

            ax_t.fill_between(xs_mib, best_us, worst_us,
                              color=color, alpha=0.15, linewidth=0)
            ax_t.plot(xs_mib, med_us, color=color, linewidth=lw,
                      marker="o", markersize=3, label=label)

            ax_bw.fill_between(xs_mib, worst_bw, best_bw,
                               color=color, alpha=0.15, linewidth=0)
            ax_bw.plot(xs_mib, med_bw, color=color, linewidth=lw,
                       marker="o", markersize=3, label=label)

            if proto == "auto":
                auto_xs, auto_med_us, auto_med_bw = xs_mib, med_us, med_bw

        ax_t.set_xscale("log")
        ax_t.set_ylabel("Kernel latency (us)")
        ax_t.set_title(coll)
        ax_t.set_ylim(bottom=0)
        ax_t.legend(loc="lower left", fontsize=9)

        ax_bw.set_xscale("log")
        ax_bw.set_ylabel("Effective bandwidth (GB/s)")
        ax_bw.set_title(coll)
        ax_bw.set_ylim(bottom=0)
        ax_bw.legend(loc="lower left", fontsize=9)

        # Zoom insets on the [1, 10] MiB band -- only 'auto' has data there.
        if auto_xs is not None:
            auto_color = proto_color["auto"]
            _add_zoom_inset(ax_t, auto_xs, auto_med_us, auto_color, "us")
            _add_zoom_inset(ax_bw, auto_xs, auto_med_bw, auto_color, "GB/s")

    # With sharex=True, only label the bottom row.
    for ax in axes[-1, :]:
        ax.set_xlabel("Message size per rank (MiB)")

    fig.suptitle("NCCL AG/RS, world_size=2", fontsize=11)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"wrote {output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _geomspace_bytes(lo_bytes: int, hi_bytes: int, n: int) -> list[int]:
    if n < 2:
        return [hi_bytes]
    lo = math.log10(max(lo_bytes, 1))
    hi = math.log10(hi_bytes)
    return [int(round(10 ** (lo + (hi - lo) * i / (n - 1)))) for i in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src-gpu", type=int, default=0,
                    help="GPU index for rank 0 (within CUDA_VISIBLE_DEVICES).")
    ap.add_argument("--dst-gpu", type=int, default=1,
                    help="GPU index for rank 1.")
    ap.add_argument(
        "--mode", choices=("symmetric", "asymmetric"), default="symmetric",
        help=(
            "symmetric (default): both ranks pass the same per-rank size N "
            "and we sweep N to map the latency/bandwidth landscape. "
            "asymmetric: fix the larger side and sweep the smaller side "
            "(the original max(send,recv) hypothesis test)."
        ),
    )
    ap.add_argument(
        "--sym-min-kib", type=int, default=4,
        help="[symmetric] smallest per-rank message size in KiB (default 4).",
    )
    ap.add_argument(
        "--sym-max-mib", type=int, default=256,
        help="[symmetric] largest per-rank message size in MiB (default 256).",
    )
    ap.add_argument(
        "--sym-points", type=int, default=24,
        help="[symmetric] number of log-spaced sizes (default 24).",
    )
    ap.add_argument(
        "--sym-extra-protocols", type=str, default="Simple,LL128,LL",
        help=(
            "[symmetric] comma-separated NCCL protocols to overlay on a "
            "focused sub-range (in addition to the default 'auto' sweep). "
            "Pass an empty string to disable. Default: 'Simple,LL128,LL'."
        ),
    )
    ap.add_argument(
        "--sym-extra-min-mib", type=int, default=32,
        help="[symmetric] smallest size in MiB for the fixed-protocol sub-sweep (default 32).",
    )
    ap.add_argument(
        "--sym-extra-max-mib", type=int, default=256,
        help="[symmetric] largest size in MiB for the fixed-protocol sub-sweep (default 256).",
    )
    ap.add_argument(
        "--sym-extra-points", type=int, default=8,
        help="[symmetric] number of log-spaced sizes in the sub-sweep (default 8).",
    )
    ap.add_argument(
        "--max-sizes-mib", type=str, default="1,16,256",
        help="[asymmetric] comma-separated max_size values in MiB. Default 1,16,256.",
    )
    ap.add_argument(
        "--min-size-points", type=int, default=12,
        help="[asymmetric] number of log-spaced 'other-side' probes per max (default 12).",
    )
    ap.add_argument("--min-other-kib", type=int, default=4,
                    help="[asymmetric] smallest 'other-side' size in KiB (default 4).")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--master-port", type=str, default="29501")
    ap.add_argument("--output", type=Path, required=True, help="Output PNG path.")
    ap.add_argument("--csv-output", type=Path, default=None,
                    help="Optional path to dump raw per-iter measurements as CSV.")
    ap.add_argument(
        "--protocols", type=str, default="auto,Simple,LL128,LL",
        help=(
            "[asymmetric] comma-separated list of NCCL protocols to sweep. "
            "Use 'auto' for NCCL's default selector (no NCCL_PROTO env). "
            "Default: 'auto,Simple,LL128,LL'. Ignored in symmetric mode "
            "(which always uses 'auto')."
        ),
    )
    args = ap.parse_args()

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

    if args.mode == "symmetric":
        sym_sizes = _geomspace_bytes(
            args.sym_min_kib * 1024,
            args.sym_max_mib * 1024 * 1024,
            args.sym_points,
        )
        # 'auto' covers the full landscape; extra protocols overlay a
        # focused sub-range so we can see how Simple / LL128 / LL each
        # behave around ~100 MiB without paying for a full sweep on each.
        proto_configs: dict[str, tuple[list[int], dict[str, list[int]]]] = {
            "auto": (sym_sizes, {str(N): [N] for N in sym_sizes}),
        }
        extra_protos = [
            p.strip() for p in args.sym_extra_protocols.split(",") if p.strip()
        ]
        if extra_protos:
            extra_sizes = _geomspace_bytes(
                args.sym_extra_min_mib * 1024 * 1024,
                args.sym_extra_max_mib * 1024 * 1024,
                args.sym_extra_points,
            )
            extra_min_by = {str(N): [N] for N in extra_sizes}
            for p in extra_protos:
                proto_configs[p] = (extra_sizes, extra_min_by)
        protocols = list(proto_configs.keys())
        # Carried only for the asymmetric plot's signature (unused here).
        max_sizes_bytes_for_plot = sym_sizes
    else:
        max_sizes_mib = [int(s) for s in args.max_sizes_mib.split(",")]
        max_sizes_bytes = [m * 1024 * 1024 for m in max_sizes_mib]

        min_sizes_by_max = {}
        for max_b in max_sizes_bytes:
            min_sizes_by_max[str(max_b)] = _geomspace_bytes(
                args.min_other_kib * 1024, max_b, args.min_size_points
            )
        protocols = [p.strip() for p in args.protocols.split(",") if p.strip()]
        proto_configs = {p: (max_sizes_bytes, min_sizes_by_max) for p in protocols}
        max_sizes_bytes_for_plot = max_sizes_bytes

    cfg_common = {
        "gpus": [args.src_gpu, args.dst_gpu],
        "iters": args.iters,
        "warmup": args.warmup,
        "master_port": args.master_port,
    }

    all_records: list[dict] = []
    with tempfile.TemporaryDirectory() as results_dir:
        for proto in protocols:
            size_max, size_min_by = proto_configs[proto]
            print(f"=== protocol: {proto} ===", flush=True)
            # Mutate env in the parent so spawn children inherit it.
            if proto == "auto":
                os.environ.pop("NCCL_PROTO", None)
            else:
                os.environ["NCCL_PROTO"] = proto
            cfg = {
                **cfg_common,
                "max_sizes_bytes": size_max,
                "min_sizes_bytes_by_max": size_min_by,
                "protocol_label": proto,
            }
            mp.spawn(_worker, nprocs=2, args=(2, cfg, results_dir), join=True)
            results_path = os.path.join(results_dir, f"results_{proto}.jsonl")
            if not os.path.exists(results_path):
                print(f"error: no results for proto={proto} at {results_path}",
                      file=sys.stderr)
                return 1
            with open(results_path) as fp:
                all_records.extend(json.loads(line) for line in fp)
        os.environ.pop("NCCL_PROTO", None)

    records = all_records

    if args.csv_output is not None:
        import csv as csv_mod
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv_output, "w", newline="") as fp:
            w = csv_mod.writer(fp)
            w.writerow(["rank", "collective", "protocol", "max_bytes", "min_bytes", "iter", "time_ms"])
            for r in records:
                w.writerow([r["rank"], r["collective"], r.get("protocol", "auto"),
                            r["max_bytes"], r["min_bytes"], r["iter"], r["time_ms"]])
        print(f"wrote {args.csv_output}")

    if args.mode == "symmetric":
        _plot_symmetric(records, args.output)
    else:
        _plot(records, args.output, max_sizes_bytes_for_plot, protocols=protocols)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
