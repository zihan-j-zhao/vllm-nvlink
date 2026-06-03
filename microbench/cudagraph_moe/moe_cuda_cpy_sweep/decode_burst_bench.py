#!/usr/bin/env python3
"""
Sparse-bursty traffic vs. decode forward — calibrated to production.

What this models (from the per-step CSVs in
playground/out/cpu_profile/e2e_agrs_nixl/...):

    xfers_per_step  ~ 0.12     # only ~12% of decode steps have a KV transfer
    bytes_per_xfer  ~ 12 MB    # p50, with a long tail up to ~100 MB
    avg throughput  ~ 9 GB/s
    decode iters    >> noise events

So instead of holding a steady GB/s, we emit a single P2P or NIXL burst on
roughly --burst-prob fraction of decode iterations, with --burst-mb of
payload, and we record the per-iter latency time-series so spikes that align
with bursts can be seen.

Outputs:
    results/burst_timeseries.json   per-iter latency, burst indicator
    results/burst_summary.csv       overall + burst-vs-quiet conditional stats
    results/burst_iters.csv         long-form: one row per iter

NVTX:
    bench/setup/{build_llm,nixl_init}
    bench/iter_{i}                  every decode forward
    bench/burst_{j}_{mode}          each burst on the noise stream
    bench/burst_{j}/req_{k}         per request inside a NIXL burst
    bench/burst_{j}/copy            P2P copy
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import csv as _csv
import json
import os
import random
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

try:
    from nixl._api import nixl_agent, nixl_agent_config
except ImportError:
    nixl_agent = None
    nixl_agent_config = None


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
# Burst engines
# ---------------------------------------------------------------------------


class P2PBurstEngine:
    """One-shot P2P burst: copies `total_bytes` from src GPU into dst GPU
    as a few back-to-back cudaMemcpyPeerAsync launches on a dedicated stream.

    Burst is launched from the main thread (so it lines up tightly with the
    iteration being timed). Uses a dedicated non-blocking stream so the copies
    do not serialize with the compute stream the model uses.
    """

    def __init__(self, dst_device: int, src_device: int,
                 max_burst_bytes: int, chunk_bytes: int) -> None:
        if dst_device == src_device:
            raise ValueError("dst != src required for P2P")
        torch.cuda.set_device(dst_device)
        self.dst_device = dst_device
        self.src_device = src_device
        self.chunk_bytes = chunk_bytes
        n = chunk_bytes // 2
        self.dst = torch.empty(n, dtype=torch.bfloat16,
                               device=f"cuda:{dst_device}")
        self.src = torch.empty(n, dtype=torch.bfloat16,
                               device=f"cuda:{src_device}")
        self.src.fill_(1.0); self.dst.fill_(0.0)
        torch.cuda.synchronize(self.dst_device)
        torch.cuda.synchronize(self.src_device)
        self.stream = torch.cuda.Stream(device=dst_device, priority=0)
        self.bytes_emitted = 0

    def fire(self, total_bytes: int, burst_idx: int) -> None:
        n_chunks = max(1, (total_bytes + self.chunk_bytes - 1) // self.chunk_bytes)
        with torch.cuda.stream(self.stream):
            torch.cuda.nvtx.range_push(f"bench/burst_{burst_idx}_p2p")
            for c in range(n_chunks):
                torch.cuda.nvtx.range_push(f"bench/burst_{burst_idx}/copy_{c}")
                self.dst.copy_(self.src, non_blocking=True)
                torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_pop()
        self.bytes_emitted += n_chunks * self.chunk_bytes


class NixlBurstEngine:
    """One-shot NIXL READ burst: issues N prepped reads of `req_bytes` each.

    Each burst = `requests` make_prepped_xfer/transfer pairs. Mirrors the
    per-request structure of NixlConnectorWorker.transfer(). Does *not*
    block on completion — bursts can overlap with the next iteration just
    like production does.
    """

    def __init__(self, rank: int, local_rank: int, world_size: int,
                 victim_rank: int, peer_rank: int,
                 donor_blocks: int, block_size_bytes: int) -> None:
        if nixl_agent is None:
            raise RuntimeError("NIXL not installed; pip install nixl")
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.victim_rank = victim_rank
        self.peer_rank = peer_rank
        self.donor_blocks = donor_blocks
        self.block_size_bytes = block_size_bytes
        self.is_victim = (rank == victim_rank)

        self.agent: Any | None = None
        self.donor: torch.Tensor | None = None
        self.registered_descs: Any | None = None
        self.remote_agent_name: str | None = None
        self.remote_ptr: int | None = None
        self.remote_device: int | None = None
        self._cache: dict[tuple[int, int], int] = {}  # (start_block, n_blocks) -> xfer
        self._inflight: list[int] = []
        self.bytes_emitted = 0
        self.request_count = 0

    def initialize(self) -> None:
        torch.cuda.nvtx.range_push("bench/setup/nixl_init")
        try:
            self.donor = torch.empty(
                self.donor_blocks * self.block_size_bytes,
                dtype=torch.uint8, device=f"cuda:{self.local_rank}",
            )
            self.donor.fill_(self.rank % 251 + 1)
            torch.cuda.synchronize(self.local_rank)

            cfg = nixl_agent_config(capture_telemetry=False) if nixl_agent_config else None
            self.agent = nixl_agent(f"burst-r{self.rank}-{os.getpid()}", cfg)
            descs = self.agent.get_reg_descs([self.donor], "VRAM")
            self.agent.register_memory(descs)
            self.registered_descs = descs

            meta = {
                "agent": base64.b64encode(self.agent.get_agent_metadata()).decode("ascii"),
                "ptr": int(self.donor.data_ptr()),
                "nbytes": int(self.donor.numel()),
                "device": int(self.local_rank),
            }
            all_meta: list[Any] = [None] * self.world_size
            dist.all_gather_object(all_meta, meta)
            if self.is_victim:
                peer = all_meta[self.peer_rank]
                self.remote_agent_name = self.agent.add_remote_agent(
                    base64.b64decode(peer["agent"])
                )
                self.remote_ptr = int(peer["ptr"])
                self.remote_device = int(peer["device"])
        finally:
            torch.cuda.nvtx.range_pop()

    def _get_xfer(self, start_block: int, n_blocks: int) -> int:
        key = (start_block, n_blocks)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        assert self.agent is not None and self.donor is not None
        bsz = self.block_size_bytes
        local_descs = self.agent.get_xfer_descs(
            [(int(self.donor.data_ptr()) + (start_block + i) * bsz, bsz, self.local_rank)
             for i in range(n_blocks)], "VRAM",
        )
        remote_descs = self.agent.get_xfer_descs(
            [(int(self.remote_ptr) + (start_block + i) * bsz, bsz, self.remote_device)
             for i in range(n_blocks)], "VRAM",
        )
        lh = self.agent.prep_xfer_dlist("NIXL_INIT_AGENT", local_descs)
        rh = self.agent.prep_xfer_dlist(self.remote_agent_name, remote_descs)
        idx = list(range(n_blocks))
        xfer = self.agent.make_prepped_xfer("READ", lh, idx, rh, idx)
        self._cache[key] = xfer
        return xfer

    def fire(self, total_bytes: int, n_requests: int, burst_idx: int) -> None:
        assert self.is_victim and self.agent is not None
        # Reap any finished prior transfers (non-blocking).
        if self._inflight:
            kept = []
            for x in self._inflight:
                try:
                    if self.agent.check_xfer_state(x) != "DONE":
                        kept.append(x)
                except Exception:
                    kept.append(x)
            self._inflight = kept

        req_bytes = max(self.block_size_bytes, total_bytes // max(1, n_requests))
        blocks_per_req = max(1, req_bytes // self.block_size_bytes)
        if blocks_per_req > self.donor_blocks:
            blocks_per_req = self.donor_blocks
        torch.cuda.nvtx.range_push(f"bench/burst_{burst_idx}_nixl")
        for k in range(n_requests):
            start = ((burst_idx * n_requests + k) * blocks_per_req) % max(
                1, self.donor_blocks - blocks_per_req + 1)
            xfer = self._get_xfer(start, blocks_per_req)
            torch.cuda.nvtx.range_push(
                f"bench/burst_{burst_idx}/req_{k}_{blocks_per_req * self.block_size_bytes}B"
            )
            self.agent.transfer(xfer)
            torch.cuda.nvtx.range_pop()
            self._inflight.append(xfer)
            self.request_count += 1
            self.bytes_emitted += blocks_per_req * self.block_size_bytes
        torch.cuda.nvtx.range_pop()

    def drain(self, timeout_s: float = 5.0) -> None:
        if self.agent is None:
            return
        deadline = time.perf_counter() + timeout_s
        while self._inflight and time.perf_counter() < deadline:
            kept = []
            for x in self._inflight:
                try:
                    if self.agent.check_xfer_state(x) != "DONE":
                        kept.append(x)
                except Exception:
                    kept.append(x)
            self._inflight = kept
            if self._inflight:
                time.sleep(0.0005)

    def close(self) -> None:
        if self.agent is None:
            return
        self.drain(timeout_s=2.0)
        for x in self._cache.values():
            with contextlib.suppress(Exception):
                self.agent.release_xfer_handle(x)
        self._cache.clear()
        if self.registered_descs is not None:
            with contextlib.suppress(Exception):
                self.agent.deregister_memory(self.registered_descs)
            self.registered_descs = None
        self.agent = None


# ---------------------------------------------------------------------------
# vLLM helpers (same as decode_noise_sweep.py)
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


# ---------------------------------------------------------------------------
# Main timed loop with sparse bursts
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    p.add_argument("--num-tokens", type=int, default=100)
    p.add_argument("--iters", type=int, default=2000,
                   help="Total timed iters (need a lot so bursts have a tail).")
    p.add_argument("--warmup-iters", type=int, default=20)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--attention-backend", default="FLASHINFER")
    p.add_argument("--moe-backend", default="triton")
    p.add_argument("--all2all-backend", default="allgather_reducescatter")
    p.add_argument("--no-enable-expert-parallel", action="store_true")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--seed", type=int, default=1)

    p.add_argument("--noise-mode", choices=["none", "p2p", "nixl"], default="nixl")
    p.add_argument("--victim-rank", type=int, default=0)
    p.add_argument("--peer-rank", type=int, default=1)

    # Production-calibrated defaults (xfers/step=0.12, p50=12 MB).
    p.add_argument("--burst-prob", type=float, default=0.12,
                   help="Probability each decode iter fires a burst.")
    p.add_argument("--burst-mb", type=float, default=12.0,
                   help="Mean burst payload (MB).")
    p.add_argument("--burst-mb-jitter", type=float, default=0.5,
                   help="+/- fractional jitter on burst-mb (e.g. 0.5 = +/-50%%).")
    p.add_argument("--burst-lag-after-iters", type=int, default=0,
                   help="Fire the burst N iters before the timed iter "
                        "(0 = same iter; 1 = previous; -1 = next).")
    # p2p
    p.add_argument("--p2p-src-device", type=int, default=2)
    p.add_argument("--p2p-chunk-mb", type=int, default=4,
                   help="Bytes per cudaMemcpyPeerAsync chunk inside a burst.")
    # nixl
    p.add_argument("--nixl-block-kb", type=int, default=1536)
    p.add_argument("--nixl-requests-per-burst", type=int, default=4,
                   help="N requests inside one burst (transfers).")
    p.add_argument("--nixl-donor-mb", type=int, default=4096)

    p.add_argument("--output-dir", type=Path, default=Path("results"))
    p.add_argument("--tag", default="burst")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)
    is_victim = (rank == args.victim_rank)

    rank_print(rank,
        f"[rank {rank}] cuda:{local_rank} ws={world_size} noise={args.noise_mode} "
        f"burst_prob={args.burst_prob} burst_mb={args.burst_mb} iters={args.iters}")

    torch.cuda.nvtx.range_push("bench/setup/build_llm")
    t0 = time.perf_counter()
    llm = build_llm(args, world_size)
    setup_s = time.perf_counter() - t0
    torch.cuda.nvtx.range_pop()
    rank_print(rank, f"[rank {rank}] LLM ready in {setup_s:.1f}s")
    model_runner = get_model_runner(llm)

    # ------------------- noise setup -------------------
    p2p_engine: P2PBurstEngine | None = None
    nixl_engine: NixlBurstEngine | None = None
    if args.noise_mode == "nixl":
        if args.victim_rank == args.peer_rank:
            raise SystemExit("victim != peer required")
        block_bytes = args.nixl_block_kb * 1024
        donor_blocks = (args.nixl_donor_mb * 1024 * 1024) // block_bytes
        nixl_engine = NixlBurstEngine(
            rank=rank, local_rank=local_rank, world_size=world_size,
            victim_rank=args.victim_rank, peer_rank=args.peer_rank,
            donor_blocks=donor_blocks, block_size_bytes=block_bytes,
        )
        nixl_engine.initialize()
    elif args.noise_mode == "p2p" and is_victim:
        if args.p2p_src_device >= torch.cuda.device_count():
            raise SystemExit(
                f"--p2p-src-device {args.p2p_src_device} not visible "
                f"(device_count={torch.cuda.device_count()})"
            )
        p2p_engine = P2PBurstEngine(
            dst_device=local_rank, src_device=args.p2p_src_device,
            max_burst_bytes=int(args.burst_mb * 1.5 * 1024 * 1024),
            chunk_bytes=args.p2p_chunk_mb * 1024 * 1024,
        )

    # ------------------- warmup -------------------
    torch.cuda.nvtx.range_push("bench/warmup")
    for _ in range(args.warmup_iters):
        model_runner._dummy_run(
            num_tokens=args.num_tokens, uniform_decode=True,
            skip_eplb=True, is_profile=False, remove_lora=True,
        )
    torch.cuda.synchronize(device)
    if dist.is_initialized():
        dist.barrier(device_ids=[device.index])
    torch.cuda.nvtx.range_pop()

    # ------------------- timed loop -------------------
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
    stops  = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
    host_starts = [0.0] * args.iters
    host_stops  = [0.0] * args.iters
    fired_this_iter = [False] * args.iters
    burst_bytes_this_iter = [0] * args.iters
    burst_idx = 0

    rng = random.Random(0xBEEF + rank)
    fired_when = {i: rng.random() < args.burst_prob for i in range(args.iters)}
    # Apply burst_lag_after_iters by shifting the fire schedule.
    lag = args.burst_lag_after_iters

    torch.cuda.nvtx.range_push("bench/timed")
    host_t0 = time.perf_counter()
    for i in range(args.iters):
        # Decide whether to fire a burst this iter (or one timed slightly before).
        fire_now = is_victim and args.noise_mode != "none" and fired_when.get(i - lag, False)
        if fire_now:
            payload_mb = args.burst_mb * (
                1.0 + rng.uniform(-args.burst_mb_jitter, args.burst_mb_jitter)
            )
            payload_mb = max(0.5, payload_mb)
            payload_bytes = int(payload_mb * 1024 * 1024)
            burst_bytes_this_iter[i] = payload_bytes
            fired_this_iter[i] = True
            if p2p_engine is not None:
                p2p_engine.fire(payload_bytes, burst_idx)
            elif nixl_engine is not None:
                nixl_engine.fire(payload_bytes, args.nixl_requests_per_burst, burst_idx)
            burst_idx += 1

        torch.cuda.nvtx.range_push(f"bench/iter_{i}")
        host_starts[i] = time.perf_counter()
        starts[i].record()
        model_runner._dummy_run(
            num_tokens=args.num_tokens, uniform_decode=True,
            skip_eplb=True, is_profile=False, remove_lora=True,
        )
        stops[i].record()
        host_stops[i] = time.perf_counter()
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize(device)
    wall_total = time.perf_counter() - host_t0
    torch.cuda.nvtx.range_pop()

    per_iter_ms = [s.elapsed_time(e) for s, e in zip(starts, stops)]
    host_iter_ms = [(b - a) * 1000.0 for a, b in zip(host_starts, host_stops)]

    if nixl_engine is not None:
        nixl_engine.drain(timeout_s=5.0)

    # ------------------- conditional stats -------------------
    def stats(name: str, xs: list[float]) -> dict[str, float]:
        if not xs:
            return {"name": name, "n": 0}
        return {
            "name": name, "n": len(xs),
            "mean": statistics.fmean(xs),
            "p50": statistics.median(xs),
            "p90": percentile(xs, 90),
            "p99": percentile(xs, 99),
            "p999": percentile(xs, 99.9),
            "min": min(xs), "max": max(xs),
        }

    gpu_with_burst = [per_iter_ms[i] for i in range(args.iters) if fired_this_iter[i]]
    gpu_no_burst   = [per_iter_ms[i] for i in range(args.iters) if not fired_this_iter[i]]
    host_with_burst = [host_iter_ms[i] for i in range(args.iters) if fired_this_iter[i]]
    host_no_burst   = [host_iter_ms[i] for i in range(args.iters) if not fired_this_iter[i]]

    summary = {
        "all_gpu_ms": stats("all_gpu_ms", per_iter_ms),
        "all_host_ms": stats("all_host_ms", host_iter_ms),
        "burst_gpu_ms": stats("burst_gpu_ms", gpu_with_burst),
        "quiet_gpu_ms": stats("quiet_gpu_ms", gpu_no_burst),
        "burst_host_ms": stats("burst_host_ms", host_with_burst),
        "quiet_host_ms": stats("quiet_host_ms", host_no_burst),
    }

    if rank == 0:
        print()
        for k, v in summary.items():
            if v.get("n", 0) == 0:
                print(f"{k:>20}  n=0")
                continue
            print(
                f"{k:>20}  n={v['n']:5d}  mean={v['mean']:6.3f}  "
                f"p50={v['p50']:6.3f}  p90={v['p90']:6.3f}  "
                f"p99={v['p99']:6.3f}  p999={v['p999']:6.3f}  "
                f"max={v['max']:6.3f}"
            )

    # ------------------- gather + write -------------------
    local_payload = {
        "rank": rank, "summary": summary,
        "per_iter_ms": per_iter_ms, "host_iter_ms": host_iter_ms,
        "fired_this_iter": fired_this_iter,
        "burst_bytes_this_iter": burst_bytes_this_iter,
        "burst_count": sum(fired_this_iter),
        "bytes_emitted":
            (p2p_engine.bytes_emitted if p2p_engine else 0)
            + (nixl_engine.bytes_emitted if nixl_engine else 0),
        "wall_total_s": wall_total,
    }
    gathered: list[Any] = [None] * world_size
    if dist.is_initialized():
        dist.all_gather_object(gathered, local_payload)
    else:
        gathered[0] = local_payload

    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        ts_path = args.output_dir / f"{args.tag}_timeseries.json"
        ts_path.write_text(json.dumps({
            "config": {k: (str(v) if isinstance(v, Path) else v)
                       for k, v in vars(args).items()} | {"world_size": world_size},
            "ranks": [g for g in gathered if g is not None],
        }, default=str))
        print(f"\n[wrote] {ts_path}")

        iters_path = args.output_dir / f"{args.tag}_iters.csv"
        with iters_path.open("w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["rank", "iter", "gpu_ms", "host_ms", "fired", "burst_bytes"])
            for g in gathered:
                if g is None: continue
                for i, (gms, hms, f_, bb) in enumerate(zip(
                    g["per_iter_ms"], g["host_iter_ms"],
                    g["fired_this_iter"], g["burst_bytes_this_iter"]
                )):
                    w.writerow([g["rank"], i, f"{gms:.4f}", f"{hms:.4f}",
                                int(bool(f_)), bb])
        print(f"[wrote] {iters_path}")

        summary_path = args.output_dir / f"{args.tag}_summary.csv"
        with summary_path.open("w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["rank", "bucket", "n", "mean_ms", "p50_ms",
                        "p90_ms", "p99_ms", "p999_ms", "min_ms", "max_ms"])
            for g in gathered:
                if g is None: continue
                for bucket, st in g["summary"].items():
                    if st.get("n", 0) == 0:
                        w.writerow([g["rank"], bucket, 0,
                                    "", "", "", "", "", "", ""])
                        continue
                    w.writerow([
                        g["rank"], bucket, st["n"],
                        f"{st['mean']:.4f}", f"{st['p50']:.4f}",
                        f"{st['p90']:.4f}", f"{st['p99']:.4f}",
                        f"{st['p999']:.4f}", f"{st['min']:.4f}",
                        f"{st['max']:.4f}",
                    ])
        print(f"[wrote] {summary_path}\n")

    with contextlib.suppress(Exception):
        if dist.is_initialized():
            dist.barrier(device_ids=[device.index])

    if nixl_engine is not None:
        nixl_engine.close()
    del llm


if __name__ == "__main__":
    main()
