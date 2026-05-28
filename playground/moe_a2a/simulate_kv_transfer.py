#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Simplified background KV-transfer simulator (GPU -> GPU).

Purpose
-------

Stand-in for vLLM's PD-disaggregation KV transfer worker, used by
``playground/moe_a2a`` Experiment 3 to study whether a concurrent
NVLink transfer interferes with steady-state ITL on the serving GPUs.

Two transfer modes
------------------

``--mode copy`` (default, the original implementation)
    Single process. Uses ``torch.Tensor.copy_()`` between two CUDA
    devices with peer access, which lowers to ``cudaMemcpyPeerAsync``
    over NVLink. This is **copy-engine** traffic: dedicated DMA
    hardware that does not consume SMs. One invocation = one
    direction; call the script twice (fwd and rev) to saturate the
    link in both directions, as in the prior experiments.

``--mode nccl``
    Two processes (spawned via ``torch.multiprocessing``), one NCCL
    rank per GPU. Each rank runs a tight loop of
    ``dist.batch_isend_irecv`` against its peer (full-duplex
    ``ncclSend`` + ``ncclRecv`` in lock-step), so a single invocation
    already saturates both directions and matches vLLM's real
    ``P2pNcclEngine`` path (``ncclSend/ncclRecv`` on a dedicated NCCL
    comm/stream). This is **SM-kernel** traffic: each
    transfer launches a small CUDA kernel that occupies a few SMs for
    the duration of the copy. The two modes therefore have different
    interference profiles -- copy mode contends only for the NVLink
    wire and copy engines; nccl mode also contends for SM scheduling
    slots with the model kernels.

Faithful to the real path
-------------------------

vLLM's ``P2pNcclEngine`` (``vllm/distributed/kv_transfer/kv_connector/v1/p2p/``)
pushes KV chunks between a prefill rank and a decode rank via
``ncclSend`` / ``ncclRecv`` on a dedicated NCCL comm and stream. The
``nccl`` mode reproduces that exact primitive set; the ``copy`` mode
is a clean upper bound on the on-wire bandwidth a perfectly-batched
KV transfer could achieve (no kernel launch / framing cost).

Operation
---------

The script runs from launch until SIGTERM (or until ``--duration-s``
elapses, whichever first), copying ``--chunk-mib`` MiB per iteration.
On shutdown it emits a single JSONL session record per rank::

    {"start_ns": ..., "end_ns": ..., "src_gpu": 6, "dst_gpu": 7,
     "chunk_bytes": ..., "n_chunks": ..., "total_bytes": ...,
     "avg_throughput_gb_s": ..., "mode": "copy"|"nccl", "rank": ...}

In ``--mode copy`` rank is always 0 and a single file is written
(``--log-path``). In ``--mode nccl`` two files are written: rank 0
goes to ``--log-path``, rank 1 goes to ``--log-path-peer`` (or, if
omitted, is auto-derived by replacing the last ``_fwd`` in the basename
with ``_rev``, mirroring the copy-mode two-process convention).

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


def _install_stop_handlers() -> None:
    """Best-effort install of SIGTERM/SIGINT handlers (no-op under spawn
    on platforms where signal installation in child is restricted)."""
    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
        signal.signal(signal.SIGINT, _on_sigterm)
    except (ValueError, OSError):
        pass


def _derive_peer_log_path(log_path: Path) -> Path:
    """Default rank-1 path: replace last ``_fwd`` -> ``_rev``, else add ``_peer``.

    Keeps the established fwd/rev file convention used by the existing
    plot/orchestrator tooling.
    """
    name = log_path.name
    if "_fwd" in name:
        new_name = name[::-1].replace("dwf_", "ver_", 1)[::-1]
    else:
        stem = log_path.stem
        new_name = f"{stem}_peer{log_path.suffix}"
    return log_path.with_name(new_name)


def _run_copy_mode(args: argparse.Namespace) -> int:
    """Original single-direction copy_() implementation (copy-engine path)."""
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
    _install_stop_handlers()

    args.log_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[kv_sim copy] src=cuda:{args.src_gpu} dst=cuda:{args.dst_gpu} "
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
            "mode": "copy",
            "rank": 0,
        }
        # JSONL with one record per session so concatenating logs from
        # multiple runs is trivial.
        with open(args.log_path, "w", buffering=1) as fp:
            fp.write(json.dumps(rec) + "\n")
        print(
            f"[kv_sim copy] wrote {args.log_path}: {n_chunks} chunks "
            f"({total_bytes / 1024**3:.2f} GiB) in {duration_s:.2f} s "
            f"-> {avg_gb_s:.1f} GB/s avg",
            flush=True,
        )
    return 0


def _nccl_worker(
    rank: int,
    world_size: int,
    gpus: list[int],
    chunk_bytes: int,
    warmup_chunks: int,
    duration_s: float | None,
    master_port: str,
    log_paths: list[str],
) -> None:
    """NCCL rank: full-duplex ``isend``+``irecv`` against the peer rank.

    Both ranks post a paired ``(isend -> peer, irecv <- peer)`` per
    iteration via ``dist.batch_isend_irecv``; NCCL will run the two ops
    on its dedicated stream as a single P2P group, which on a node with
    peer access lowers to full-duplex NVLink kernels.
    """
    import torch.distributed as dist  # local import: parent doesn't need it.

    my_gpu = gpus[rank]
    peer_rank = 1 - rank

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("VLLM_USE_MODELSCOPE", "False")

    torch.cuda.set_device(my_gpu)
    device = torch.device("cuda", my_gpu)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # uint8 buffers: 1 byte/element, so numel == bytes. Distinct fill per
    # rank just so a hex-dump on the receiver would prove the transfer
    # actually crossed the link (not strictly required for timing).
    send_buf = torch.empty(chunk_bytes, dtype=torch.uint8, device=device)
    send_buf.fill_(0xAB + rank)
    recv_buf = torch.empty(chunk_bytes, dtype=torch.uint8, device=device)

    def _one_round_trip() -> None:
        ops = [
            dist.P2POp(dist.isend, send_buf, peer_rank),
            dist.P2POp(dist.irecv, recv_buf, peer_rank),
        ]
        reqs = dist.batch_isend_irecv(ops)
        for r in reqs:
            r.wait()

    # Warm-up (both ranks must do the same number of round trips).
    for _ in range(warmup_chunks):
        _one_round_trip()
    torch.cuda.synchronize(device)

    # Signal handlers installed after warm-up so a stray signal can't
    # interrupt NCCL init / handshake.
    _install_stop_handlers()

    # Barrier so both ranks start timing at (approximately) the same instant.
    dist.barrier()

    print(
        f"[kv_sim nccl rank{rank}] gpu=cuda:{my_gpu} peer_rank={peer_rank} "
        f"peer_gpu=cuda:{gpus[peer_rank]} chunk={chunk_bytes // 1024**2} MiB "
        f"pid={os.getpid()}",
        flush=True,
    )

    # --- timed window -------------------------------------------------------
    start_ns = time.time_ns()
    n_chunks = 0
    deadline_ns = (
        start_ns + int(duration_s * 1e9) if duration_s else None
    )
    try:
        while not _stop_requested:
            if deadline_ns is not None and time.time_ns() >= deadline_ns:
                break
            _one_round_trip()
            n_chunks += 1
        # Drain NCCL's internal stream before stamping end_ns.
        torch.cuda.synchronize(device)
    finally:
        end_ns = time.time_ns()
        total_bytes = n_chunks * chunk_bytes
        duration_s_actual = max((end_ns - start_ns) / 1e9, 1e-9)
        # Each round trip moves chunk_bytes in each direction; report the
        # per-rank wire bandwidth (send leg). Multiply by 2 for the
        # aggregate full-duplex throughput.
        avg_gb_s = total_bytes / 1e9 / duration_s_actual
        log_path = Path(log_paths[rank])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "start_ns": start_ns,
            "end_ns": end_ns,
            "src_gpu": my_gpu,
            "dst_gpu": gpus[peer_rank],
            "chunk_bytes": chunk_bytes,
            "n_chunks": n_chunks,
            "total_bytes": total_bytes,
            "avg_throughput_gb_s": avg_gb_s,
            "duration_s": duration_s_actual,
            "mode": "nccl",
            "rank": rank,
        }
        with open(log_path, "w", buffering=1) as fp:
            fp.write(json.dumps(rec) + "\n")
        print(
            f"[kv_sim nccl rank{rank}] wrote {log_path}: {n_chunks} round trips "
            f"({total_bytes / 1024**3:.2f} GiB sent) in {duration_s_actual:.2f} s "
            f"-> {avg_gb_s:.1f} GB/s per-direction",
            flush=True,
        )
        dist.destroy_process_group()


def _run_nccl_mode(args: argparse.Namespace) -> int:
    """Spawn two NCCL ranks; one invocation already covers both directions."""
    import torch.multiprocessing as mp

    if args.log_path_peer is None:
        peer_path = _derive_peer_log_path(args.log_path)
    else:
        peer_path = args.log_path_peer
    if peer_path == args.log_path:
        print(
            "error: --log-path-peer must differ from --log-path",
            file=sys.stderr,
        )
        return 2

    chunk_bytes = args.chunk_mib * 1024 * 1024
    print(
        f"[kv_sim nccl] spawning 2 ranks: "
        f"rank0=cuda:{args.src_gpu} -> {args.log_path}, "
        f"rank1=cuda:{args.dst_gpu} -> {peer_path}",
        flush=True,
    )
    mp.spawn(
        _nccl_worker,
        nprocs=2,
        args=(
            2,
            [args.src_gpu, args.dst_gpu],
            chunk_bytes,
            args.warmup_chunks,
            args.duration_s,
            args.master_port,
            [str(args.log_path), str(peer_path)],
        ),
        join=True,
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--mode", choices=("copy", "nccl"), default="copy",
        help=(
            "copy (default): single-process torch.copy_() == "
            "cudaMemcpyPeerAsync via copy engines; one direction per "
            "invocation. nccl: spawn 2 ranks doing full-duplex "
            "isend/irecv via NCCL; one invocation covers both directions."
        ),
    )
    ap.add_argument(
        "--src-gpu", type=int, required=True,
        help="Source GPU index (within CUDA_VISIBLE_DEVICES). In nccl mode this is rank 0's GPU.",
    )
    ap.add_argument(
        "--dst-gpu", type=int, required=True,
        help="Destination GPU index (within CUDA_VISIBLE_DEVICES). In nccl mode this is rank 1's GPU.",
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
        help=(
            "Output JSONL path. Parent dirs are created. In nccl mode "
            "this receives rank 0's record; see --log-path-peer for rank 1."
        ),
    )
    ap.add_argument(
        "--log-path-peer", type=Path, default=None,
        help=(
            "[nccl only] Path for rank 1's JSONL record. If omitted, "
            "derived from --log-path by replacing the last '_fwd' in the "
            "basename with '_rev' (or appending '_peer' if no '_fwd' present)."
        ),
    )
    ap.add_argument(
        "--master-port", type=str, default="29503",
        help="[nccl only] MASTER_PORT for the 2-rank process group (default 29503).",
    )
    ap.add_argument(
        "--warmup-chunks", type=int, default=2,
        help=(
            "Number of warm-up chunks (or round trips, in nccl mode) to "
            "execute before the timed window begins (default 2). Excluded "
            "from start_ns/end_ns and totals."
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

    if args.mode == "copy":
        return _run_copy_mode(args)
    return _run_nccl_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
