"""Standalone background CE traffic runner.

Run as a subprocess so its CPU/GIL load doesn't sit on top of vLLM's
worker process. This matters for collectives whose host-side launch
depends on every rank's CPU being responsive (NCCL Ring*).

Behavior matches the in-proc :class:`BackgroundTraffic` exactly:
  - Allocates src/dst buffers on (phantom, decoder) per the direction.
  - Spins a thread that issues cudaMemcpyPeerAsync at the requested
    rate. Side CUDA stream on the source GPU.
  - Stops on SIGTERM/SIGINT and writes a JSON stats blob (optional).
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from pathlib import Path

import torch

from .bg_traffic import BackgroundTraffic, make_pattern


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-rank", type=int, required=True,
                    help="Decoder local rank (0..3). Pairs with GPU "
                         "local_rank+4 as the phantom prefiller.")
    ap.add_argument("--direction", choices=["ingress", "egress", "both"],
                    required=True)
    ap.add_argument("--pattern", default="constant")
    ap.add_argument("--rate-gbps", type=float, required=True)
    ap.add_argument("--chunk-mb", type=int, default=4)
    ap.add_argument("--buffer-mb", type=int, default=64)
    ap.add_argument("--stats-out", type=Path, default=None,
                    help="Optional JSON path; final stats written on exit.")
    ap.add_argument("--ready-file", type=Path, default=None,
                    help="Touched after buffers are allocated and runner is idle.")
    ap.add_argument("--go-file", type=Path, default=None,
                    help="Runner waits for this file before issuing traffic.")
    ap.add_argument("--stop-file", type=Path, default=None,
                    help="Runner stops when this file appears.")
    ap.add_argument("--max-inflight-copies", type=int, default=4,
                    help="Synchronize after this many enqueued copies. "
                         "0 keeps the stream unbounded.")
    args = ap.parse_args()

    directions = (("ingress", "egress") if args.direction == "both"
                  else (args.direction,))
    bgs: list[BackgroundTraffic] = []
    for d in directions:
        per_rate = (args.rate_gbps / len(directions)) * 1e9
        pat = make_pattern(
            args.pattern,
            rate_bytes_per_sec=per_rate,
            chunk_bytes=args.chunk_mb * 1024 * 1024,
        )
        bg = BackgroundTraffic(
            local_rank=args.local_rank, pattern=pat,
            buffer_bytes=args.buffer_mb * 1024 * 1024,
            direction=d,
            max_inflight_copies=args.max_inflight_copies,
        )
        bgs.append(bg)

    stop = threading.Event()

    def _handler(signum, frame):  # noqa: ARG001
        stop.set()

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)

    for bg in bgs:
        bg.prepare()
        print(f"[bg_runner lr={args.local_rank}] {bg.describe()}", flush=True)

    if args.ready_file:
        args.ready_file.parent.mkdir(parents=True, exist_ok=True)
        args.ready_file.touch()

    print(f"[bg_runner lr={args.local_rank}] ready, waiting for go",
          flush=True)
    while not stop.is_set():
        if args.go_file is None or args.go_file.exists():
            break
        time.sleep(0.01)

    if not stop.is_set():
        for bg in bgs:
            bg.start()
        print(f"[bg_runner lr={args.local_rank}] running", flush=True)

    while not stop.is_set():
        if args.stop_file is not None and args.stop_file.exists():
            stop.set()
            break
        time.sleep(0.01)

    stats = [bg.stop() for bg in bgs]
    print(f"[bg_runner lr={args.local_rank}] stopped: {stats}", flush=True)
    if args.stats_out:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)
        args.stats_out.write_text(json.dumps({
            "local_rank": args.local_rank,
            "direction": args.direction,
            "stats": stats,
        }, indent=2))

    # Drop CUDA tensors before process exit to avoid noisy shutdown.
    del bgs
    torch.cuda.synchronize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
