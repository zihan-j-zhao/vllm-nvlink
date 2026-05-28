#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Simplified background KV-transfer simulator (GPU -> GPU).

Purpose
-------

Stand-in for vLLM's PD-disaggregation KV transfer worker, used by
``playground/moe_a2a`` Experiment 3 to study whether a concurrent
NVLink transfer interferes with steady-state ITL on the serving GPUs.

Faithful to the real path
-------------------------

vLLM's ``P2pNcclEngine`` (``vllm/distributed/kv_transfer/kv_connector/v1/p2p/``)
pushes KV chunks between a prefill rank and a decode rank via
``ncclSend`` / ``ncclRecv`` on a dedicated NCCL comm and stream. On a
single node with peer access enabled, those NCCL primitives reduce to
``cudaMemcpyPeerAsync`` over NVLink. This script does the same on-wire
work using ``torch.Tensor.copy_()`` across two CUDA devices: it
allocates a source buffer on ``--src-gpu`` and a destination buffer on
``--dst-gpu``, enables peer access, and copies the source buffer into
the destination buffer in a tight loop on a dedicated stream. There is
no NCCL group, no Python dispatch per chunk, and no message framing
overhead, so this is a clean upper bound on the NVLink contention a
KV-transfer worker would impose.

Operation
---------

The script runs from launch until SIGTERM (or until ``--duration-s``
elapses, whichever first), copying ``--chunk-mib`` MiB per iteration.
On shutdown it emits a single JSONL session record::

    {"start_ns": ..., "end_ns": ..., "src_gpu": 6, "dst_gpu": 7,
     "chunk_bytes": ..., "n_chunks": ..., "total_bytes": ...,
     "avg_throughput_gb_s": ...}

The start / end timestamps are wall-clock (``time.time_ns``) captured
right around the first chunk launch and the final ``cudaStreamSynchronize``,
so they bracket the actual contention window from the host's point of
view. They are aligned to the same clock AIPerf's
``request_start_ns`` / ``request_end_ns`` use, which is what
``plot_step_latency_timeseries.py --transfer-log`` consumes for the
background shading.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import torch

_stop_requested = False


def _on_sigterm(_signum, _frame) -> None:
    global _stop_requested
    _stop_requested = True


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--src-gpu", type=int, required=True,
        help="Source GPU index (within CUDA_VISIBLE_DEVICES).",
    )
    ap.add_argument(
        "--dst-gpu", type=int, required=True,
        help="Destination GPU index (within CUDA_VISIBLE_DEVICES).",
    )
    ap.add_argument(
        "--chunk-mib", type=int, default=256,
        help="Per-iteration copy size in MiB (default 256).",
    )
    ap.add_argument(
        "--duration-s", type=float, default=None,
        help=(
            "Optional hard cap on wall-clock runtime in seconds. "
            "If omitted, runs until SIGTERM."
        ),
    )
    ap.add_argument(
        "--log-path", type=Path, required=True,
        help="Output JSONL path. Parent dirs are created.",
    )
    ap.add_argument(
        "--warmup-chunks", type=int, default=2,
        help=(
            "Number of warm-up chunks to execute before the timed window "
            "begins (default 2). Excluded from start_ns/end_ns and totals."
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
            f"{torch.cuda.device_count()} visible to this process",
            file=sys.stderr,
        )
        return 1

    # Enable peer access so cudaMemcpyPeerAsync uses NVLink directly.
    # If peer access is not supported the copy still works (it'll bounce
    # through host memory), but we want the NVLink path for this study.
    src_dev = torch.device("cuda", args.src_gpu)
    dst_dev = torch.device("cuda", args.dst_gpu)
    can_src_to_dst = torch.cuda.can_device_access_peer(args.src_gpu, args.dst_gpu)
    if not can_src_to_dst:
        print(
            f"warning: GPU{args.src_gpu} cannot peer-access GPU{args.dst_gpu}; "
            "copies will fall back to staging through host memory",
            file=sys.stderr,
        )

    chunk_bytes = args.chunk_mib * 1024 * 1024
    # uint8 buffers: 1 byte/element, so numel == bytes.
    src = torch.empty(chunk_bytes, dtype=torch.uint8, device=src_dev)
    dst = torch.empty(chunk_bytes, dtype=torch.uint8, device=dst_dev)
    # Touch the source so the allocation is materialized (no first-touch
    # surprises in the timed window).
    src.fill_(0xAB)

    # Dedicated stream on the destination device for the copy launches.
    copy_stream = torch.cuda.Stream(device=dst_dev)

    # Warm-up: a few copies + sync so allocator / peer-access caches are hot.
    with torch.cuda.stream(copy_stream):
        for _ in range(args.warmup_chunks):
            dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize(dst_dev)

    # Install signal handler last so warm-up can't be interrupted half-way.
    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
        signal.signal(signal.SIGINT, _on_sigterm)
    except (ValueError, OSError):
        pass

    args.log_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[kv_sim] src=cuda:{args.src_gpu} dst=cuda:{args.dst_gpu} "
        f"chunk={args.chunk_mib} MiB peer_access={can_src_to_dst} "
        f"pid={os.getpid()}",
        flush=True,
    )

    # --- timed window -------------------------------------------------------
    start_ns = time.time_ns()
    n_chunks = 0
    deadline_ns = (
        start_ns + int(args.duration_s * 1e9) if args.duration_s else None
    )
    try:
        with torch.cuda.stream(copy_stream):
            while not _stop_requested:
                if deadline_ns is not None and time.time_ns() >= deadline_ns:
                    break
                dst.copy_(src, non_blocking=True)
                n_chunks += 1
                # We do NOT sync per chunk; we let the stream queue copies
                # back-to-back, mirroring how NCCL would saturate the link.
                # If we synced per chunk we'd serialize launches and
                # under-utilize NVLink.
        # Drain pending copies before stamping end_ns.
        torch.cuda.synchronize(dst_dev)
    finally:
        end_ns = time.time_ns()
        total_bytes = n_chunks * chunk_bytes
        duration_s = max((end_ns - start_ns) / 1e9, 1e-9)
        avg_gb_s = total_bytes / 1e9 / duration_s
        rec = {
            "start_ns": start_ns,
            "end_ns": end_ns,
            "src_gpu": args.src_gpu,
            "dst_gpu": args.dst_gpu,
            "chunk_bytes": chunk_bytes,
            "n_chunks": n_chunks,
            "total_bytes": total_bytes,
            "avg_throughput_gb_s": avg_gb_s,
            "duration_s": duration_s,
        }
        # JSONL with one record per session so concatenating logs from
        # multiple runs is trivial.
        with open(args.log_path, "w", buffering=1) as fp:
            fp.write(json.dumps(rec) + "\n")
        print(
            f"[kv_sim] wrote {args.log_path}: {n_chunks} chunks "
            f"({total_bytes / 1024**3:.2f} GiB) in {duration_s:.2f} s "
            f"-> {avg_gb_s:.1f} GB/s avg",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
