#!/usr/bin/env python3
"""
Microbenchmark: how does background P2P `cudaMemcpyAsync` traffic (copy-engine
path) affect a single vLLM CUDA-graph decode forward?

Layout
------
- Ranks 0 and 1 run vLLM Qwen3-MoE on cuda:0 and cuda:1 (DP=2 + EP).
- A separate "noise GPU" (default cuda:2) holds a source buffer.
- A background thread on rank 0 sustains ~10 GB/s of `cudaMemcpyPeerAsync`
  reads from cuda:2 into cuda:0 on a dedicated non-blocking stream.

Phases
------
Phase A "quiet":  10 decode forwards of 100 tokens, no background traffic.
Phase B "noisy":  10 decode forwards of 100 tokens, background P2P at target Gbps.

Both phases use the same captured CUDA graph (so the difference is purely
contention on the destination GPU's copy engines / NVLink / memory subsystem).

NVTX
----
Designed to be opened in Nsight Systems. The taxonomy:

    bench/setup
        bench/setup/build_llm
        bench/setup/warmup
    bench/phase/quiet
        bench/phase/quiet/warmup
        bench/phase/quiet/timed
            bench/phase/quiet/iter_{i}    (one per forward)
    bench/p2p/start
    bench/phase/noisy
        bench/phase/noisy/warmup
        bench/phase/noisy/timed
            bench/phase/noisy/iter_{i}
    bench/p2p/stop
    bench/p2p/copy_{src}_to_{dst}        (each P2P chunk, on the noise thread)

Plus instantaneous NVTX marks at the start/end of every iteration.

Usage
-----
    bash run.sh                           # defaults: 10 GB/s P2P, src=cuda:2
    bash run.sh --target-gbps 25
    bash run.sh --no-p2p                  # phase B with P2P disabled (sanity check)
    bash run_nsys.sh                      # capture an nsys timeline
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def _parse_int_list(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round((len(values) - 1) * pct / 100.0)))
    return values[idx]


def rank_print(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Background P2P traffic generator
# ---------------------------------------------------------------------------


class P2PBackgroundTraffic:
    """Sustains a target throughput of P2P cudaMemcpyAsync reads.

    Pulls bytes from `src_device` into `dst_device` on a dedicated non-blocking
    stream owned by `dst_device`. Because peer access is enabled, the runtime
    issues `cudaMemcpyPeerAsync`, which is serviced by the GPU's copy engines
    (DMA path) over NVLink/PCIe — exactly the path we want to stress.

    Rate is held to `target_gbps` by host-side throttling between launches;
    `in_flight` bounds the queue depth so we don't pile up too much work.
    """

    def __init__(
        self,
        dst_device: int,
        src_device: int,
        target_gbps: float = 10.0,
        chunk_mb: int = 32,
        in_flight: int = 4,
    ) -> None:
        if dst_device == src_device:
            raise ValueError("dst and src devices must differ for P2P")
        self.dst_device = dst_device
        self.src_device = src_device
        self.target_bytes_per_sec = int(target_gbps * (1024 ** 3))
        self.chunk_bytes = int(chunk_mb * (1024 ** 2))
        self.in_flight = in_flight
        self.period_s = self.chunk_bytes / float(self.target_bytes_per_sec)

        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None

        # Stats populated by the worker.
        self.bytes_copied: int = 0
        self.launches: int = 0
        self.start_time: float = 0.0
        self.end_time: float = 0.0

    @staticmethod
    def _try_enable_peer(dst: int, src: int) -> bool:
        """Best-effort P2P enable. PyTorch usually does this lazily on copy_."""
        try:
            ok = torch.cuda.can_device_access_peer(dst, src) and \
                 torch.cuda.can_device_access_peer(src, dst)
        except Exception:
            return False
        return bool(ok)

    def _worker(self) -> None:
        try:
            torch.cuda.set_device(self.dst_device)
            can_peer = self._try_enable_peer(self.dst_device, self.src_device)

            num_elems = self.chunk_bytes // 2  # bfloat16 = 2 bytes
            dst_buf = torch.empty(
                num_elems, dtype=torch.bfloat16,
                device=f"cuda:{self.dst_device}",
            )
            src_buf = torch.empty(
                num_elems, dtype=torch.bfloat16,
                device=f"cuda:{self.src_device}",
            )
            src_buf.fill_(1.0)
            dst_buf.fill_(0.0)
            torch.cuda.synchronize(self.dst_device)
            torch.cuda.synchronize(self.src_device)

            stream = torch.cuda.Stream(device=self.dst_device)
            events = [
                torch.cuda.Event(enable_timing=False)
                for _ in range(self.in_flight)
            ]

            # Mark the lifecycle of the noise thread in nsys.
            torch.cuda.nvtx.range_push("bench/p2p/worker")
            try:
                start_msg = (
                    f"[p2p] dst=cuda:{self.dst_device} src=cuda:{self.src_device} "
                    f"target={self.target_bytes_per_sec / (1024 ** 3):.2f} GiB/s "
                    f"chunk={self.chunk_bytes // (1024 ** 2)} MiB "
                    f"in_flight={self.in_flight} peer_access={can_peer}"
                )
                print(start_msg, flush=True)

                self._ready.set()

                i = 0
                self.start_time = time.perf_counter()
                next_launch = self.start_time
                copy_label = f"bench/p2p/copy_{self.src_device}_to_{self.dst_device}"

                while not self._stop.is_set():
                    now = time.perf_counter()
                    if now < next_launch:
                        # Host-side rate limit. Bounded sleep so stop() is responsive.
                        time.sleep(min(next_launch - now, 0.005))
                        continue

                    slot = i % self.in_flight
                    if i >= self.in_flight:
                        events[slot].synchronize()

                    with torch.cuda.stream(stream):
                        torch.cuda.nvtx.range_push(copy_label)
                        # Tensor.copy_ between two CUDA tensors with peer access
                        # turns into cudaMemcpyPeerAsync (copy-engine path).
                        dst_buf.copy_(src_buf, non_blocking=True)
                        torch.cuda.nvtx.range_pop()
                        events[slot].record(stream)

                    self.bytes_copied += self.chunk_bytes
                    self.launches += 1
                    next_launch += self.period_s
                    i += 1

                stream.synchronize()
                self.end_time = time.perf_counter()
            finally:
                torch.cuda.nvtx.range_pop()

        except BaseException as e:  # noqa: BLE001
            self._error = e
            self._ready.set()
            print(f"[p2p worker error] {e!r}", file=sys.stderr, flush=True)

    def start(self) -> None:
        self._stop.clear()
        self._ready.clear()
        self._thread = threading.Thread(
            target=self._worker, name="P2PBackgroundTraffic", daemon=True,
        )
        torch.cuda.nvtx.mark("bench/p2p/start")
        self._thread.start()
        if not self._ready.wait(timeout=30):
            raise TimeoutError("P2P worker failed to become ready in 30s")
        if self._error is not None:
            raise self._error

    def stop(self) -> None:
        if self._thread is None:
            return
        torch.cuda.nvtx.mark("bench/p2p/stop")
        self._stop.set()
        self._thread.join(timeout=30)
        self._thread = None

    def effective_gbps(self) -> float:
        duration = self.end_time - self.start_time
        if duration <= 0:
            return 0.0
        return self.bytes_copied / duration / (1024 ** 3)


# ---------------------------------------------------------------------------
# vLLM setup
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--num-tokens", type=int, default=100,
                        help="Decode batch size per rank (uniform decode).")
    parser.add_argument("--iters", type=int, default=10,
                        help="Timed decode iterations per phase.")
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--attention-backend", default="FLASHINFER")
    parser.add_argument("--moe-backend", default="triton")
    parser.add_argument("--all2all-backend", default="allgather_reducescatter")
    parser.add_argument("--no-enable-expert-parallel", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--seed", type=int, default=1)

    # P2P noise knobs.
    parser.add_argument("--no-p2p", action="store_true",
                        help="Disable phase B noise (sanity check).")
    parser.add_argument("--p2p-src-device", type=int, default=2,
                        help="Source GPU index for noise (default cuda:2).")
    parser.add_argument("--p2p-dst-rank", type=int, default=0,
                        help="Which DP rank gets the noise (default rank 0).")
    parser.add_argument("--target-gbps", type=float, default=10.0,
                        help="Target sustained P2P read throughput (GiB/s).")
    parser.add_argument("--p2p-chunk-mb", type=int, default=32)
    parser.add_argument("--p2p-in-flight", type=int, default=4)
    parser.add_argument("--p2p-prelude-s", type=float, default=0.5,
                        help="Seconds to let P2P traffic settle before phase B.")

    parser.add_argument("--output-json", type=Path,
                        default=Path("results/p2p_decode.json"))
    return parser.parse_args()


def build_llm(args: argparse.Namespace, world_size: int):
    from vllm import LLM

    os.environ.setdefault("VLLM_ATTENTION_BACKEND", args.attention_backend)

    return LLM(
        model=args.model,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=world_size,
        enable_expert_parallel=not args.no_enable_expert_parallel,
        distributed_executor_backend="external_launcher",
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
        disable_log_stats=True,
        moe_backend=args.moe_backend,
        all2all_backend=args.all2all_backend,
    )


def get_model_runner(llm) -> Any:
    executor = llm.llm_engine.model_executor
    driver = executor.driver_worker
    worker = getattr(driver, "worker", driver)
    return worker.model_runner


# ---------------------------------------------------------------------------
# Benchmark phase
# ---------------------------------------------------------------------------


def bench_phase(
    model_runner,
    phase: str,
    num_tokens: int,
    warmup_iters: int,
    timed_iters: int,
    device: torch.device,
) -> dict[str, Any]:
    """Run `timed_iters` decode forwards and return per-iter latency stats.

    Heavy NVTX annotation so the resulting nsys timeline is searchable.
    """
    # Warmup is captured under its own NVTX range so it can be folded away.
    torch.cuda.nvtx.range_push(f"bench/phase/{phase}/warmup")
    for i in range(warmup_iters):
        torch.cuda.nvtx.range_push(f"bench/phase/{phase}/warmup_iter_{i}")
        model_runner._dummy_run(
            num_tokens=num_tokens,
            uniform_decode=True,
            skip_eplb=True,
            is_profile=False,
            remove_lora=True,
        )
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize(device)
    if dist.is_initialized():
        dist.barrier(device_ids=[device.index])
    torch.cuda.nvtx.range_pop()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(timed_iters)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(timed_iters)]

    torch.cuda.nvtx.range_push(f"bench/phase/{phase}/timed")
    host_t0 = time.perf_counter()
    for i in range(timed_iters):
        torch.cuda.nvtx.mark(f"bench/phase/{phase}/iter_{i}/start")
        torch.cuda.nvtx.range_push(f"bench/phase/{phase}/iter_{i}")
        starts[i].record()
        model_runner._dummy_run(
            num_tokens=num_tokens,
            uniform_decode=True,
            skip_eplb=True,
            is_profile=False,
            remove_lora=True,
        )
        stops[i].record()
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.mark(f"bench/phase/{phase}/iter_{i}/end")
    torch.cuda.synchronize(device)
    host_t_total = time.perf_counter() - host_t0
    torch.cuda.nvtx.range_pop()

    per_iter_ms = [s.elapsed_time(e) for s, e in zip(starts, stops)]

    return {
        "phase": phase,
        "num_tokens": num_tokens,
        "iters": timed_iters,
        "per_iter_ms": per_iter_ms,
        "mean_ms": statistics.fmean(per_iter_ms),
        "p50_ms": statistics.median(per_iter_ms),
        "p90_ms": percentile(per_iter_ms, 90),
        "p99_ms": percentile(per_iter_ms, 99),
        "min_ms": min(per_iter_ms),
        "max_ms": max(per_iter_ms),
        "wall_total_s": host_t_total,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if args.num_tokens > args.max_num_seqs:
        raise SystemExit(
            f"--num-tokens {args.num_tokens} > --max-num-seqs {args.max_num_seqs}"
        )

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    rank_print(rank, f"[rank {rank}] cuda:{local_rank}  world_size={world_size}")

    # ------------------------------------------------------------------ setup
    torch.cuda.nvtx.range_push("bench/setup")
    torch.cuda.nvtx.range_push("bench/setup/build_llm")
    t0 = time.perf_counter()
    llm = build_llm(args, world_size)
    setup_s = time.perf_counter() - t0
    torch.cuda.nvtx.range_pop()
    rank_print(rank, f"[rank {rank}] LLM ready in {setup_s:.1f}s")

    model_runner = get_model_runner(llm)
    cg_mode = model_runner.compilation_config.cudagraph_mode
    rank_print(rank, f"[rank {rank}] cudagraph_mode = {cg_mode}")
    torch.cuda.nvtx.range_pop()  # bench/setup

    # ------------------------------------------------------------ phase A
    if dist.is_initialized():
        dist.barrier(device_ids=[device.index])
    torch.cuda.nvtx.range_push("bench/phase/quiet")
    rank_print(rank, f"[rank {rank}] >>> Phase A (quiet)")
    quiet = bench_phase(
        model_runner, "quiet",
        num_tokens=args.num_tokens,
        warmup_iters=args.warmup_iters,
        timed_iters=args.iters,
        device=device,
    )
    torch.cuda.nvtx.range_pop()
    rank_print(
        rank,
        f"[rank {rank}] Phase A  p50={quiet['p50_ms']:.3f} ms "
        f"p90={quiet['p90_ms']:.3f}  p99={quiet['p99_ms']:.3f}  "
        f"min={quiet['min_ms']:.3f}  max={quiet['max_ms']:.3f}",
    )

    # ------------------------------------------------------ start noise
    p2p: P2PBackgroundTraffic | None = None
    enable_noise = (not args.no_p2p) and (rank == args.p2p_dst_rank)
    if enable_noise:
        visible = torch.cuda.device_count()
        if args.p2p_src_device >= visible:
            raise SystemExit(
                f"--p2p-src-device {args.p2p_src_device} not visible "
                f"(device_count={visible}). Set CUDA_VISIBLE_DEVICES to include it."
            )
        p2p = P2PBackgroundTraffic(
            dst_device=local_rank,
            src_device=args.p2p_src_device,
            target_gbps=args.target_gbps,
            chunk_mb=args.p2p_chunk_mb,
            in_flight=args.p2p_in_flight,
        )
        p2p.start()
        if args.p2p_prelude_s > 0:
            time.sleep(args.p2p_prelude_s)

    if dist.is_initialized():
        dist.barrier(device_ids=[device.index])

    # ------------------------------------------------------------ phase B
    torch.cuda.nvtx.range_push("bench/phase/noisy")
    rank_print(rank, f"[rank {rank}] >>> Phase B (noisy, p2p={enable_noise})")
    noisy = bench_phase(
        model_runner, "noisy",
        num_tokens=args.num_tokens,
        warmup_iters=args.warmup_iters,
        timed_iters=args.iters,
        device=device,
    )
    torch.cuda.nvtx.range_pop()
    rank_print(
        rank,
        f"[rank {rank}] Phase B  p50={noisy['p50_ms']:.3f} ms "
        f"p90={noisy['p90_ms']:.3f}  p99={noisy['p99_ms']:.3f}  "
        f"min={noisy['min_ms']:.3f}  max={noisy['max_ms']:.3f}",
    )

    # ------------------------------------------------------- stop noise
    p2p_stats: dict[str, Any] = {}
    if p2p is not None:
        p2p.stop()
        p2p_stats = {
            "dst_device": p2p.dst_device,
            "src_device": p2p.src_device,
            "target_gbps": args.target_gbps,
            "chunk_mb": args.p2p_chunk_mb,
            "in_flight": args.p2p_in_flight,
            "launches": p2p.launches,
            "bytes_copied": p2p.bytes_copied,
            "duration_s": p2p.end_time - p2p.start_time,
            "effective_gbps": p2p.effective_gbps(),
        }
        print(
            f"[rank {rank}] p2p: {p2p.launches} launches, "
            f"{p2p.bytes_copied / (1024 ** 3):.2f} GiB in "
            f"{p2p.end_time - p2p.start_time:.2f}s -> "
            f"{p2p.effective_gbps():.2f} GiB/s effective",
            flush=True,
        )

    # ----------------------------------------------------------- gather
    local_payload = {"rank": rank, "quiet": quiet, "noisy": noisy, "p2p": p2p_stats}
    gathered: list[dict[str, Any] | None] = [None] * world_size
    if dist.is_initialized():
        dist.all_gather_object(gathered, local_payload)
    else:
        gathered[0] = local_payload

    if rank == 0:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "model": args.model,
                "world_size": world_size,
                "num_tokens": args.num_tokens,
                "iters": args.iters,
                "warmup_iters": args.warmup_iters,
                "attention_backend": args.attention_backend,
                "moe_backend": args.moe_backend,
                "all2all_backend": args.all2all_backend,
                "expert_parallel": not args.no_enable_expert_parallel,
                "enforce_eager": args.enforce_eager,
                "p2p_enabled": not args.no_p2p,
                "p2p_dst_rank": args.p2p_dst_rank,
                "p2p_src_device": args.p2p_src_device,
                "target_gbps": args.target_gbps,
            },
            "ranks": [g for g in gathered if g is not None],
        }
        args.output_json.write_text(json.dumps(payload, indent=2))
        print(f"\n[wrote] {args.output_json}\n", flush=True)
        # Pretty summary table.
        print(
            f"{'rank':>4} {'phase':>6} "
            f"{'p50':>8} {'p90':>8} {'p99':>8} {'min':>8} {'max':>8}"
        )
        for g in gathered:
            if g is None:
                continue
            for phase_key in ("quiet", "noisy"):
                row = g[phase_key]
                print(
                    f"{g['rank']:>4} {phase_key:>6}  "
                    f"{row['p50_ms']:>7.3f}  {row['p90_ms']:>7.3f}  "
                    f"{row['p99_ms']:>7.3f}  {row['min_ms']:>7.3f}  "
                    f"{row['max_ms']:>7.3f}"
                )

    with contextlib.suppress(Exception):
        if dist.is_initialized():
            dist.barrier(device_ids=[device.index])

    del llm


if __name__ == "__main__":
    main()
