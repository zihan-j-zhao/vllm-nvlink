#!/usr/bin/env python3
"""
P2P bandwidth sweep over the vLLM CUDA-graph decode forward.

Topology and per-phase machinery are identical to ../moe_cuda_cpy/, but:

* The vLLM model + CUDA graphs are initialized **once** for the whole sweep.
* We run a quiet baseline, then for each target throughput in --sweep-gbps
  we start the P2P noise thread, run a noisy phase, then stop it.
* Defaults: 200 timed iterations of `num_tokens=100` per phase, sweeping
  10, 20, 40, 80, 160, 320, 640 GiB/s (doubling each step, capped at 800).

NVTX layout (search-friendly inside nsys):

    bench/setup/build_llm
    bench/phase/quiet/{warmup,timed}
    bench/phase/noisy_{gbps}gbps/{warmup,timed}
        bench/phase/.../iter_{i}
    bench/p2p/start
    bench/p2p/stop
    bench/p2p/copy_{src}_to_{dst}     (each cudaMemcpyPeerAsync chunk)
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


def _parse_float_list(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


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
# Background P2P traffic generator (same as moe_cuda_cpy/, kept local).
# ---------------------------------------------------------------------------


class P2PBackgroundTraffic:
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

        self.bytes_copied = 0
        self.launches = 0
        self.start_time = 0.0
        self.end_time = 0.0

    def _worker(self) -> None:
        try:
            torch.cuda.set_device(self.dst_device)
            try:
                can_peer = bool(
                    torch.cuda.can_device_access_peer(self.dst_device, self.src_device)
                )
            except Exception:
                can_peer = False

            num_elems = self.chunk_bytes // 2
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
            events = [torch.cuda.Event(enable_timing=False) for _ in range(self.in_flight)]

            torch.cuda.nvtx.range_push(f"bench/p2p/worker_{self.target_bytes_per_sec >> 30}gbps")
            try:
                print(
                    f"[p2p] dst=cuda:{self.dst_device} src=cuda:{self.src_device} "
                    f"target={self.target_bytes_per_sec / (1024 ** 3):.2f} GiB/s "
                    f"chunk={self.chunk_bytes // (1024 ** 2)} MiB "
                    f"in_flight={self.in_flight} peer_access={can_peer}",
                    flush=True,
                )
                self._ready.set()

                i = 0
                self.start_time = time.perf_counter()
                next_launch = self.start_time
                copy_label = f"bench/p2p/copy_{self.src_device}_to_{self.dst_device}"

                while not self._stop.is_set():
                    now = time.perf_counter()
                    if now < next_launch:
                        time.sleep(min(next_launch - now, 0.002))
                        continue

                    slot = i % self.in_flight
                    if i >= self.in_flight:
                        events[slot].synchronize()

                    with torch.cuda.stream(stream):
                        torch.cuda.nvtx.range_push(copy_label)
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
        self.bytes_copied = 0
        self.launches = 0
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
        d = self.end_time - self.start_time
        return self.bytes_copied / d / (1024 ** 3) if d > 0 else 0.0


# ---------------------------------------------------------------------------
# vLLM helpers
# ---------------------------------------------------------------------------


def build_llm(args, world_size: int):
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


def get_model_runner(llm):
    executor = llm.llm_engine.model_executor
    driver = executor.driver_worker
    worker = getattr(driver, "worker", driver)
    return worker.model_runner


def bench_phase(
    model_runner,
    phase: str,
    num_tokens: int,
    warmup_iters: int,
    timed_iters: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.cuda.nvtx.range_push(f"bench/phase/{phase}/warmup")
    for i in range(warmup_iters):
        torch.cuda.nvtx.range_push(f"bench/phase/{phase}/warmup_iter_{i}")
        model_runner._dummy_run(
            num_tokens=num_tokens, uniform_decode=True,
            skip_eplb=True, is_profile=False, remove_lora=True,
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
        torch.cuda.nvtx.range_push(f"bench/phase/{phase}/iter_{i}")
        starts[i].record()
        model_runner._dummy_run(
            num_tokens=num_tokens, uniform_decode=True,
            skip_eplb=True, is_profile=False, remove_lora=True,
        )
        stops[i].record()
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize(device)
    wall_total = time.perf_counter() - host_t0
    torch.cuda.nvtx.range_pop()

    per_iter_ms = [s.elapsed_time(e) for s, e in zip(starts, stops)]
    return {
        "phase": phase,
        "iters": timed_iters,
        "per_iter_ms": per_iter_ms,
        "mean_ms": statistics.fmean(per_iter_ms),
        "p50_ms": statistics.median(per_iter_ms),
        "p90_ms": percentile(per_iter_ms, 90),
        "p99_ms": percentile(per_iter_ms, 99),
        "min_ms": min(per_iter_ms),
        "max_ms": max(per_iter_ms),
        "wall_total_s": wall_total,
    }


# ---------------------------------------------------------------------------
# Main: single LLM init + sweep
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    p.add_argument("--num-tokens", type=int, default=100)
    p.add_argument("--iters", type=int, default=200,
                   help="Timed decode iterations per phase.")
    p.add_argument("--warmup-iters", type=int, default=10)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--attention-backend", default="FLASHINFER")
    p.add_argument("--moe-backend", default="triton")
    p.add_argument("--all2all-backend", default="allgather_reducescatter")
    p.add_argument("--no-enable-expert-parallel", action="store_true")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--seed", type=int, default=1)

    p.add_argument("--sweep-gbps", type=_parse_float_list,
                   default=[10, 20, 40, 80, 160, 320, 640],
                   help="Comma-separated list of target P2P throughputs (GiB/s).")
    p.add_argument("--p2p-src-device", type=int, default=2)
    p.add_argument("--p2p-dst-rank", type=int, default=0)
    p.add_argument("--p2p-chunk-mb", type=int, default=32)
    p.add_argument("--p2p-in-flight", type=int, default=4)
    p.add_argument("--p2p-prelude-s", type=float, default=0.5)
    p.add_argument("--cooldown-s", type=float, default=0.5,
                   help="Sleep between phases so traces are easy to separate.")
    p.add_argument("--skip-quiet", action="store_true")

    p.add_argument("--output-json", type=Path,
                   default=Path("results/p2p_sweep.json"))
    p.add_argument("--output-csv", type=Path,
                   default=Path("results/p2p_sweep.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    rank_print(rank, f"[rank {rank}] cuda:{local_rank}  world_size={world_size}  "
                     f"sweep={args.sweep_gbps} GiB/s  iters={args.iters}")

    # ---------- build LLM once ----------
    torch.cuda.nvtx.range_push("bench/setup/build_llm")
    t0 = time.perf_counter()
    llm = build_llm(args, world_size)
    setup_s = time.perf_counter() - t0
    torch.cuda.nvtx.range_pop()
    rank_print(rank, f"[rank {rank}] LLM ready in {setup_s:.1f}s")

    model_runner = get_model_runner(llm)
    cg_mode = model_runner.compilation_config.cudagraph_mode
    rank_print(rank, f"[rank {rank}] cudagraph_mode = {cg_mode}")

    # Sanity check noise GPU visibility (only on the dst rank).
    is_dst = (rank == args.p2p_dst_rank)
    if is_dst:
        if args.p2p_src_device >= torch.cuda.device_count():
            raise SystemExit(
                f"--p2p-src-device {args.p2p_src_device} not visible "
                f"(device_count={torch.cuda.device_count()}). "
                f"Update CUDA_VISIBLE_DEVICES."
            )

    phase_results: list[dict[str, Any]] = []

    # ---------- quiet baseline ----------
    if not args.skip_quiet:
        if dist.is_initialized():
            dist.barrier(device_ids=[device.index])
        torch.cuda.nvtx.range_push("bench/phase/quiet")
        rank_print(rank, f"[rank {rank}] >>> quiet baseline")
        quiet = bench_phase(
            model_runner, "quiet",
            num_tokens=args.num_tokens,
            warmup_iters=args.warmup_iters,
            timed_iters=args.iters,
            device=device,
        )
        torch.cuda.nvtx.range_pop()
        quiet["target_gbps"] = 0.0
        quiet["effective_gbps"] = 0.0
        phase_results.append(quiet)
        rank_print(
            rank,
            f"[rank {rank}] quiet  "
            f"p50={quiet['p50_ms']:.3f}  p90={quiet['p90_ms']:.3f}  "
            f"p99={quiet['p99_ms']:.3f}  min={quiet['min_ms']:.3f}  "
            f"max={quiet['max_ms']:.3f}",
        )

    # ---------- sweep ----------
    for gbps in args.sweep_gbps:
        if args.cooldown_s > 0:
            time.sleep(args.cooldown_s)

        p2p = None
        if is_dst:
            p2p = P2PBackgroundTraffic(
                dst_device=local_rank,
                src_device=args.p2p_src_device,
                target_gbps=gbps,
                chunk_mb=args.p2p_chunk_mb,
                in_flight=args.p2p_in_flight,
            )
            p2p.start()
            if args.p2p_prelude_s > 0:
                time.sleep(args.p2p_prelude_s)

        if dist.is_initialized():
            dist.barrier(device_ids=[device.index])

        phase_name = f"noisy_{int(round(gbps))}gbps"
        torch.cuda.nvtx.range_push(f"bench/phase/{phase_name}")
        rank_print(rank, f"[rank {rank}] >>> {phase_name}")
        row = bench_phase(
            model_runner, phase_name,
            num_tokens=args.num_tokens,
            warmup_iters=args.warmup_iters,
            timed_iters=args.iters,
            device=device,
        )
        torch.cuda.nvtx.range_pop()

        if p2p is not None:
            p2p.stop()
            row["effective_gbps"] = p2p.effective_gbps()
            row["p2p_launches"] = p2p.launches
            row["p2p_bytes"] = p2p.bytes_copied
        else:
            row["effective_gbps"] = 0.0
        row["target_gbps"] = gbps
        phase_results.append(row)

        rank_print(
            rank,
            f"[rank {rank}] {phase_name}  "
            f"p50={row['p50_ms']:.3f}  p90={row['p90_ms']:.3f}  "
            f"p99={row['p99_ms']:.3f}  min={row['min_ms']:.3f}  "
            f"max={row['max_ms']:.3f}  "
            f"effective={row['effective_gbps']:.2f} GiB/s",
        )

        if dist.is_initialized():
            dist.barrier(device_ids=[device.index])

    # ---------- gather & write ----------
    local_payload = {"rank": rank, "phases": phase_results}
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
                "sweep_gbps": args.sweep_gbps,
                "attention_backend": args.attention_backend,
                "moe_backend": args.moe_backend,
                "all2all_backend": args.all2all_backend,
                "expert_parallel": not args.no_enable_expert_parallel,
                "enforce_eager": args.enforce_eager,
                "p2p_dst_rank": args.p2p_dst_rank,
                "p2p_src_device": args.p2p_src_device,
            },
            "ranks": [g for g in gathered if g is not None],
        }
        args.output_json.write_text(json.dumps(payload, indent=2))
        print(f"\n[wrote] {args.output_json}", flush=True)

        # CSV: one row per (rank, phase)
        import csv as _csv
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow([
                "rank", "phase", "target_gbps", "effective_gbps",
                "iters", "mean_ms", "p50_ms", "p90_ms", "p99_ms",
                "min_ms", "max_ms", "wall_total_s",
            ])
            for g in gathered:
                if g is None:
                    continue
                for row in g["phases"]:
                    w.writerow([
                        g["rank"], row["phase"],
                        row.get("target_gbps", 0.0),
                        row.get("effective_gbps", 0.0),
                        row["iters"], f"{row['mean_ms']:.4f}",
                        f"{row['p50_ms']:.4f}", f"{row['p90_ms']:.4f}",
                        f"{row['p99_ms']:.4f}", f"{row['min_ms']:.4f}",
                        f"{row['max_ms']:.4f}", f"{row['wall_total_s']:.4f}",
                    ])
        print(f"[wrote] {args.output_csv}\n", flush=True)

        # Pretty summary table.
        print(
            f"{'rank':>4} {'phase':>20} {'tgt_gbps':>9} {'eff_gbps':>9}  "
            f"{'p50':>8} {'p90':>8} {'p99':>8} {'min':>8} {'max':>8}"
        )
        for g in gathered:
            if g is None:
                continue
            for row in g["phases"]:
                print(
                    f"{g['rank']:>4} {row['phase']:>20} "
                    f"{row.get('target_gbps', 0.0):>9.2f} "
                    f"{row.get('effective_gbps', 0.0):>9.2f}  "
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
