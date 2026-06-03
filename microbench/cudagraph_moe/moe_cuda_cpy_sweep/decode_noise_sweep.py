#!/usr/bin/env python3
"""
P2P / NIXL noise sweep over the vLLM CUDA-graph decode forward.

This extends ../moe_cuda_cpy_sweep/decode_p2p_sweep.py with a `nixl` noise
mode that mimics the real vLLM NIXL KV-receive pattern:

* Each DP rank registers a "donor" GPU buffer pool with NIXL on top of the
  same torch.distributed group vLLM uses (no extra torchrun ranks needed).
* The configured victim rank (default rank 0) spawns a background thread
  that issues bursty NIXL READs from the peer rank's donor buffer.
* Each burst = a fan-in of N "requests", each request = a list of
  `block_size`-byte READ descs, executed as ONE `make_prepped_xfer` —
  mirroring how `NixlConnectorWorker.transfer()` issues per-request reads
  in production (`vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py`).
* Target throughput is held by a host-side token-bucket between bursts.
* For p2p mode, the noise is the cudaMemcpyPeerAsync path from a 3rd GPU.

Noise modes
-----------
    --noise-mode none    quiet only (or skip the sweep entirely)
    --noise-mode p2p     cudaMemcpyPeerAsync from --p2p-src-device -> victim
    --noise-mode nixl    NIXL READs from the peer DP rank -> victim

NVTX taxonomy (nsys-friendly)
-----------------------------
    bench/setup/build_llm
    bench/setup/nixl_init                   (only in nixl mode)
    bench/phase/quiet/{warmup,timed,iter_i}
    bench/phase/noisy_{N}gbps/{warmup,timed,iter_i}
    bench/noise/{p2p|nixl}/worker_{N}gbps
        bench/noise/p2p/copy_{src}_to_{dst}     (each chunk)
        bench/noise/nixl/burst_{j}              (one burst = N "requests")
            bench/noise/nixl/req_{k}            (one make_prepped_xfer)
    bench/noise/{p2p|nixl}/start, .../stop      (marks)
"""

from __future__ import annotations

import argparse
import base64
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

try:
    from nixl._api import nixl_agent, nixl_agent_config
except ImportError:
    nixl_agent = None
    nixl_agent_config = None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


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
# Noise source: cudaMemcpyPeerAsync (copy engine) — same as moe_cuda_cpy_sweep
# ---------------------------------------------------------------------------


class P2PBackgroundTraffic:
    def __init__(
        self,
        dst_device: int,
        src_device: int,
        target_gbps: float,
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
        self.target_gbps = target_gbps

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
            n = self.chunk_bytes // 2
            dst_buf = torch.empty(n, dtype=torch.bfloat16, device=f"cuda:{self.dst_device}")
            src_buf = torch.empty(n, dtype=torch.bfloat16, device=f"cuda:{self.src_device}")
            src_buf.fill_(1.0); dst_buf.fill_(0.0)
            torch.cuda.synchronize(self.dst_device)
            torch.cuda.synchronize(self.src_device)

            stream = torch.cuda.Stream(device=self.dst_device)
            events = [torch.cuda.Event(enable_timing=False) for _ in range(self.in_flight)]

            label = f"bench/noise/p2p/worker_{int(self.target_gbps)}gbps"
            torch.cuda.nvtx.range_push(label)
            try:
                self._ready.set()
                i = 0
                self.start_time = time.perf_counter()
                next_launch = self.start_time
                copy_label = f"bench/noise/p2p/copy_{self.src_device}_to_{self.dst_device}"

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
            self._error = e; self._ready.set()
            print(f"[p2p worker error] {e!r}", file=sys.stderr, flush=True)

    def start(self) -> None:
        self._stop.clear(); self._ready.clear()
        self.bytes_copied = 0; self.launches = 0
        self._thread = threading.Thread(target=self._worker, daemon=True,
                                        name="P2PBackgroundTraffic")
        torch.cuda.nvtx.mark("bench/noise/p2p/start")
        self._thread.start()
        if not self._ready.wait(timeout=30):
            raise TimeoutError("P2P worker not ready in 30s")
        if self._error: raise self._error

    def stop(self) -> None:
        if self._thread is None: return
        torch.cuda.nvtx.mark("bench/noise/p2p/stop")
        self._stop.set(); self._thread.join(timeout=30); self._thread = None

    def effective_gbps(self) -> float:
        d = self.end_time - self.start_time
        return self.bytes_copied / d / (1024 ** 3) if d > 0 else 0.0


# ---------------------------------------------------------------------------
# Noise source: NIXL READ (mirrors vLLM's NixlConnectorWorker pattern)
# ---------------------------------------------------------------------------


class NixlNoiseHarness:
    """One per rank. Owns the NIXL agent + donor buffer + cached READ handles.

    Lifecycle:
        harness = NixlNoiseHarness(...)
        harness.initialize()      # all ranks: agent + donor + metadata swap
        ...
        traffic = harness.start_traffic(target_gbps, ...)  # only victim rank
        ...                                                # noisy phase runs
        traffic.stop()
        ...
        harness.close()           # all ranks

    The donor buffer is a contiguous bf16 tensor laid out as B blocks of
    `block_size_bytes` each, registered with NIXL once. Each "request" READ
    pulls K random blocks from the peer donor, mirroring how vLLM pulls a
    list of KV-cache blocks per request.
    """

    def __init__(
        self,
        rank: int,
        local_rank: int,
        world_size: int,
        victim_rank: int,
        peer_rank: int,
        donor_blocks: int,
        block_size_bytes: int,
        blocks_per_request: int,
        max_in_flight: int,
    ) -> None:
        if nixl_agent is None:
            raise RuntimeError(
                "NIXL Python package required for --noise-mode nixl. "
                "Install nixl (e.g. `pip install nixl`) inside the vllm env."
            )
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.victim_rank = victim_rank
        self.peer_rank = peer_rank
        self.donor_blocks = donor_blocks
        self.block_size_bytes = block_size_bytes
        self.blocks_per_request = blocks_per_request
        self.max_in_flight = max_in_flight
        self.is_victim = (rank == victim_rank)
        self.is_peer = (rank == peer_rank)

        self.agent: Any | None = None
        self.donor: torch.Tensor | None = None
        self.registered_descs: Any | None = None
        self.remote_agent_name: str | None = None
        self.remote_ptr: int | None = None
        self.remote_device: int | None = None
        self._handle_cache: dict[tuple[tuple[int, ...]], int] = {}

    def initialize(self) -> None:
        torch.cuda.nvtx.range_push("bench/setup/nixl_init")
        try:
            device = torch.device(f"cuda:{self.local_rank}")
            # Every rank gets a donor — keeps the protocol symmetric and lets
            # you flip victim/peer with a flag.
            self.donor = torch.empty(
                self.donor_blocks * self.block_size_bytes,
                dtype=torch.uint8, device=device,
            )
            # Use a non-trivial fill so reads aren't optimized away anywhere.
            self.donor.fill_(self.rank % 251 + 1)
            torch.cuda.synchronize(self.local_rank)

            cfg = nixl_agent_config(capture_telemetry=False) if nixl_agent_config else None
            self.agent = nixl_agent(
                f"sweep-noise-r{self.rank}-{os.getpid()}", cfg,
            )
            descs = self.agent.get_reg_descs([self.donor], "VRAM")
            self.agent.register_memory(descs)
            self.registered_descs = descs

            # Exchange (agent_metadata, donor_ptr, donor_device) across ranks
            # via the global torch.distributed group. vLLM already initialized
            # it under external_launcher.
            meta = {
                "agent": base64.b64encode(self.agent.get_agent_metadata()).decode("ascii"),
                "ptr": int(self.donor.data_ptr()),
                "nbytes": int(self.donor.numel()),
                "device": int(self.local_rank),
            }
            all_meta: list[Any] = [None] * self.world_size
            dist.all_gather_object(all_meta, meta)

            # The victim sets up the remote handle for reading from peer.
            if self.is_victim:
                peer_meta = all_meta[self.peer_rank]
                assert peer_meta is not None
                remote_md = base64.b64decode(peer_meta["agent"])
                self.remote_agent_name = self.agent.add_remote_agent(remote_md)
                self.remote_ptr = int(peer_meta["ptr"])
                self.remote_device = int(peer_meta["device"])
                if peer_meta["nbytes"] < self.donor.numel():
                    raise RuntimeError("peer donor buffer too small")
        finally:
            torch.cuda.nvtx.range_pop()

    def _make_request_handle(self, block_ids: tuple[int, ...]) -> int:
        """Create or fetch a cached prepped xfer for a list of block ids.

        Local + remote descriptor lists are at the same block offsets to keep
        addressing trivial. The cache key is the tuple of block ids so
        repeated patterns hit the fast path.
        """
        cached = self._handle_cache.get(block_ids)
        if cached is not None:
            return cached
        assert self.agent is not None and self.donor is not None
        assert self.remote_agent_name is not None

        bsz = self.block_size_bytes
        local_descs_list = [
            (int(self.donor.data_ptr()) + b * bsz, bsz, int(self.local_rank))
            for b in block_ids
        ]
        remote_descs_list = [
            (int(self.remote_ptr) + b * bsz, bsz, int(self.remote_device))
            for b in block_ids
        ]
        local_descs = self.agent.get_xfer_descs(local_descs_list, "VRAM")
        remote_descs = self.agent.get_xfer_descs(remote_descs_list, "VRAM")
        local_handle = self.agent.prep_xfer_dlist("NIXL_INIT_AGENT", local_descs)
        remote_handle = self.agent.prep_xfer_dlist(self.remote_agent_name, remote_descs)
        indices = list(range(len(block_ids)))
        xfer = self.agent.make_prepped_xfer(
            "READ", local_handle, indices, remote_handle, indices,
        )
        self._handle_cache[block_ids] = xfer
        return xfer

    def start_traffic(self, target_gbps: float,
                      burst_interval_ms_min: float,
                      burst_interval_ms_max: float,
                      requests_per_burst: int) -> "NixlBackgroundTraffic":
        if not self.is_victim:
            raise RuntimeError("only the victim rank issues NIXL noise")
        traffic = NixlBackgroundTraffic(
            harness=self, target_gbps=target_gbps,
            burst_interval_ms_min=burst_interval_ms_min,
            burst_interval_ms_max=burst_interval_ms_max,
            requests_per_burst=requests_per_burst,
        )
        traffic.start()
        return traffic

    def close(self) -> None:
        if self.agent is None:
            return
        # Release cached handles (best effort).
        for xfer in self._handle_cache.values():
            try:
                self.agent.release_xfer_handle(xfer)
            except Exception:
                pass
        self._handle_cache.clear()
        if self.registered_descs is not None:
            try:
                self.agent.deregister_memory(self.registered_descs)
            except Exception:
                pass
            self.registered_descs = None
        self.agent = None


class NixlBackgroundTraffic:
    """Bursty NIXL READ generator. One per active noise phase on the victim.

    Each burst issues `requests_per_burst` prepped transfers (each transfer
    pulls `blocks_per_request` blocks from the peer donor). Between bursts we
    sleep enough to hold the target average throughput. Mirrors the way vLLM
    decode issues per-request reads in waves driven by the scheduler.
    """

    def __init__(
        self,
        harness: NixlNoiseHarness,
        target_gbps: float,
        burst_interval_ms_min: float,
        burst_interval_ms_max: float,
        requests_per_burst: int,
    ) -> None:
        self.h = harness
        self.target_gbps = target_gbps
        self.requests_per_burst = max(1, requests_per_burst)
        self.burst_min_s = max(0.0, burst_interval_ms_min / 1000.0)
        self.burst_max_s = max(self.burst_min_s, burst_interval_ms_max / 1000.0)

        # Pre-built request block-id tuples; cycle through them to vary the
        # access pattern but keep the prepped-xfer cache warm.
        n_blocks = self.h.donor_blocks
        bp = self.h.blocks_per_request
        if bp > n_blocks:
            raise ValueError("blocks_per_request > donor_blocks")
        # 32 distinct request patterns (rolling windows).
        self._patterns: list[tuple[int, ...]] = []
        stride = max(1, n_blocks // 32)
        for k in range(32):
            start = (k * stride) % max(1, n_blocks - bp + 1)
            self._patterns.append(tuple(start + j for j in range(bp)))

        self.bytes_per_request = bp * self.h.block_size_bytes
        self.bytes_per_burst = self.requests_per_burst * self.bytes_per_request
        self.target_bytes_per_sec = int(target_gbps * (1024 ** 3))
        # Average inter-burst period to hit target throughput.
        self.period_s = self.bytes_per_burst / float(self.target_bytes_per_sec)

        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None
        self.bytes_xferred = 0
        self.burst_count = 0
        self.request_count = 0
        self.start_time = 0.0
        self.end_time = 0.0
        # Sliding window of in-flight handles (so multiple bursts can overlap).
        self._inflight: list[int] = []

    def _drain_some(self) -> None:
        """Pop completed handles from in_flight; non-blocking."""
        if not self._inflight:
            return
        agent = self.h.agent
        assert agent is not None
        kept: list[int] = []
        for x in self._inflight:
            try:
                st = agent.check_xfer_state(x)
            except Exception:
                kept.append(x); continue
            if st == "DONE":
                continue
            kept.append(x)
        self._inflight = kept

    def _worker(self) -> None:
        try:
            agent = self.h.agent
            assert agent is not None
            label = f"bench/noise/nixl/worker_{int(self.target_gbps)}gbps"
            torch.cuda.nvtx.range_push(label)
            try:
                self._ready.set()
                self.start_time = time.perf_counter()
                next_burst = self.start_time
                burst_idx = 0
                import random
                rng = random.Random(0xC0FFEE)

                while not self._stop.is_set():
                    now = time.perf_counter()
                    if now < next_burst:
                        time.sleep(min(next_burst - now, 0.002))
                        continue

                    # Cap in-flight to avoid runaway queueing.
                    while (len(self._inflight)
                           >= self.h.max_in_flight * self.requests_per_burst):
                        self._drain_some()
                        if self._stop.is_set():
                            break
                        if (len(self._inflight)
                                >= self.h.max_in_flight * self.requests_per_burst):
                            time.sleep(0.0005)

                    torch.cuda.nvtx.range_push(f"bench/noise/nixl/burst_{burst_idx}")
                    for k in range(self.requests_per_burst):
                        pattern = self._patterns[
                            (burst_idx * self.requests_per_burst + k)
                            % len(self._patterns)
                        ]
                        xfer = self.h._make_request_handle(pattern)
                        torch.cuda.nvtx.range_push(
                            f"bench/noise/nixl/req_{k}_{self.bytes_per_request}B"
                        )
                        agent.transfer(xfer)
                        torch.cuda.nvtx.range_pop()
                        self._inflight.append(xfer)
                        self.request_count += 1
                        self.bytes_xferred += self.bytes_per_request
                    torch.cuda.nvtx.range_pop()

                    self.burst_count += 1
                    burst_idx += 1

                    # Jittered next-burst time to avoid lock-step with vLLM.
                    if self.burst_max_s > 0:
                        jitter = rng.uniform(self.burst_min_s, self.burst_max_s)
                    else:
                        jitter = 0.0
                    next_burst = max(next_burst + self.period_s, now + jitter)

                    self._drain_some()

                # Drain remaining transfers.
                deadline = time.perf_counter() + 5.0
                while self._inflight and time.perf_counter() < deadline:
                    self._drain_some()
                    if self._inflight:
                        time.sleep(0.0005)
                self.end_time = time.perf_counter()
            finally:
                torch.cuda.nvtx.range_pop()
        except BaseException as e:  # noqa: BLE001
            self._error = e; self._ready.set()
            print(f"[nixl worker error] {e!r}", file=sys.stderr, flush=True)

    def start(self) -> None:
        self._stop.clear(); self._ready.clear()
        self.bytes_xferred = 0; self.burst_count = 0; self.request_count = 0
        self._inflight = []
        self._thread = threading.Thread(target=self._worker, daemon=True,
                                        name="NixlBackgroundTraffic")
        torch.cuda.nvtx.mark("bench/noise/nixl/start")
        self._thread.start()
        if not self._ready.wait(timeout=30):
            raise TimeoutError("nixl worker not ready in 30s")
        if self._error: raise self._error

    def stop(self) -> None:
        if self._thread is None: return
        torch.cuda.nvtx.mark("bench/noise/nixl/stop")
        self._stop.set(); self._thread.join(timeout=60); self._thread = None

    def effective_gbps(self) -> float:
        d = self.end_time - self.start_time
        return self.bytes_xferred / d / (1024 ** 3) if d > 0 else 0.0


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


def populate_realistic_block_table(model_runner, num_reqs: int,
                                   max_seq_len: int, seed: int = 0x5EED) -> int:
    """Fill input_batch.block_table with distinct random block ids so
    attention reads scatter across the KV cache instead of hitting the
    same low-numbered blocks every iter.

    Returns the number of distinct blocks the test uses.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    ib = model_runner.input_batch
    mg = ib.block_table  # MultiGroupBlockTable

    total_distinct = 0
    for g_idx in range(len(mg.block_tables)):
        bt = mg.block_tables[g_idx]
        block_size = bt.block_size
        blocks_needed = max(1, (max_seq_len + block_size - 1) // block_size)
        cpu = bt.block_table.cpu  # shape (max_num_reqs, max_num_blocks_per_req)
        max_blocks_per_req = cpu.shape[1]
        blocks_needed = min(blocks_needed, max_blocks_per_req)

        # KV cache total block count: infer from the per-layer kv_cache tensor
        # by reading max(block_id) the runner has registered, otherwise fall
        # back to a generous number.
        kv_blocks = getattr(model_runner, "num_total_kv_cache_blocks", None)
        if kv_blocks is None:
            kv_blocks = 4096
        kv_blocks = max(blocks_needed * num_reqs, kv_blocks)

        # Distinct ids per request (sampled w/o replacement across all reqs).
        # NOTE: cpu is a torch tensor whose .dtype is e.g. torch.int32 — numpy
        # cannot interpret that, so we go via torch.from_numpy and .to(dtype).
        flat_ids_np = rng.choice(
            kv_blocks, size=num_reqs * blocks_needed, replace=False
        )
        flat_ids = torch.from_numpy(
            flat_ids_np.reshape(num_reqs, blocks_needed)
        ).to(cpu.dtype)
        cpu[:num_reqs, :blocks_needed] = flat_ids
        # Zero out anything past blocks_needed (kernel ignores it but cleaner).
        if blocks_needed < max_blocks_per_req:
            cpu[:num_reqs, blocks_needed:] = 0
        bt.block_table.copy_to_gpu(num_reqs)
        total_distinct += num_reqs * blocks_needed
    return total_distinct


def bench_phase(model_runner, phase: str, num_tokens: int,
                warmup_iters: int, timed_iters: int,
                device: torch.device,
                profile_seq_lens: int | None = None) -> dict[str, Any]:
    # Build dummy_run kwargs once.
    dr_kwargs: dict[str, Any] = dict(
        num_tokens=num_tokens, uniform_decode=True,
        skip_eplb=True, is_profile=False, remove_lora=True,
    )
    if profile_seq_lens is not None and profile_seq_lens > 1:
        # Pass as torch.int tensor sized to num_reqs (== num_tokens for
        # uniform decode max_query_len=1).
        dr_kwargs["profile_seq_lens"] = torch.full(
            (num_tokens,), int(profile_seq_lens), dtype=torch.int,
        )

    torch.cuda.nvtx.range_push(f"bench/phase/{phase}/warmup")
    for i in range(warmup_iters):
        torch.cuda.nvtx.range_push(f"bench/phase/{phase}/warmup_iter_{i}")
        model_runner._dummy_run(**dr_kwargs)
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize(device)
    if dist.is_initialized():
        dist.barrier(device_ids=[device.index])
    torch.cuda.nvtx.range_pop()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(timed_iters)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(timed_iters)]
    host_starts = [0.0] * timed_iters
    host_stops = [0.0] * timed_iters

    torch.cuda.nvtx.range_push(f"bench/phase/{phase}/timed")
    host_t0 = time.perf_counter()
    for i in range(timed_iters):
        torch.cuda.nvtx.range_push(f"bench/phase/{phase}/iter_{i}")
        host_starts[i] = time.perf_counter()
        starts[i].record()
        model_runner._dummy_run(**dr_kwargs)
        stops[i].record()
        host_stops[i] = time.perf_counter()
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize(device)
    wall_total = time.perf_counter() - host_t0
    torch.cuda.nvtx.range_pop()

    per_iter_ms = [s.elapsed_time(e) for s, e in zip(starts, stops)]
    host_iter_ms = [(b - a) * 1000.0 for a, b in zip(host_starts, host_stops)]
    return {
        "phase": phase,
        "iters": timed_iters,
        "per_iter_ms": per_iter_ms,
        "host_iter_ms": host_iter_ms,
        "mean_ms": statistics.fmean(per_iter_ms),
        "p50_ms": statistics.median(per_iter_ms),
        "p90_ms": percentile(per_iter_ms, 90),
        "p99_ms": percentile(per_iter_ms, 99),
        "min_ms": min(per_iter_ms),
        "max_ms": max(per_iter_ms),
        "host_p50_ms": statistics.median(host_iter_ms),
        "host_p99_ms": percentile(host_iter_ms, 99),
        "host_max_ms": max(host_iter_ms),
        "wall_total_s": wall_total,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    p.add_argument("--num-tokens", type=int, default=100)
    p.add_argument("--iters", type=int, default=200)
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

    # Noise selection.
    p.add_argument("--noise-mode", choices=["none", "p2p", "nixl"], default="nixl",
                   help="Background noise during noisy phases.")
    p.add_argument("--sweep-gbps", type=_parse_float_list,
                   default=[10, 20, 40, 80, 160, 320, 640])
    p.add_argument("--victim-rank", type=int, default=0,
                   help="DP rank that receives the noise.")
    p.add_argument("--peer-rank", type=int, default=1,
                   help="(NIXL only) DP rank that serves as donor.")
    p.add_argument("--skip-quiet", action="store_true")
    p.add_argument("--cooldown-s", type=float, default=0.5)
    p.add_argument("--prelude-s", type=float, default=0.5)

    # p2p mode
    p.add_argument("--p2p-src-device", type=int, default=2)
    p.add_argument("--p2p-chunk-mb", type=int, default=32)
    p.add_argument("--p2p-in-flight", type=int, default=4)

    # nixl mode
    p.add_argument("--nixl-block-kb", type=int, default=1536,
                   help="Bytes per KV block (default 1.5 MiB ~ vLLM page).")
    p.add_argument("--nixl-blocks-per-request", type=int, default=8,
                   help="Blocks read per 'request' (one make_prepped_xfer).")
    p.add_argument("--nixl-requests-per-burst", type=int, default=4,
                   help="Requests issued back-to-back per burst.")
    p.add_argument("--nixl-burst-interval-ms-min", type=float, default=0.0,
                   help="Min jitter between bursts (avg target set by --sweep-gbps).")
    p.add_argument("--nixl-burst-interval-ms-max", type=float, default=0.0,
                   help="Max jitter between bursts (0 = strict rate).")
    p.add_argument("--nixl-donor-mb", type=int, default=4096,
                   help="Donor buffer size per rank (MiB). Must hold all "
                        "concurrent blocks across patterns.")
    p.add_argument("--nixl-max-in-flight", type=int, default=4,
                   help="Max in-flight bursts.")

    p.add_argument("--output-json", type=Path, default=Path("results/sweep.json"))
    p.add_argument("--output-csv", type=Path, default=Path("results/sweep.csv"))

    # Realism knobs — make _dummy_run's attention look like real decode.
    p.add_argument("--profile-seq-lens", type=int, default=None,
                   help="Override per-request seq_lens (in tokens). When set, "
                        "attention scans this many KV slots per request "
                        "instead of the default 1. Use a realistic context "
                        "length to make attention reads hit HBM and expose "
                        "memory-subsystem contention with background noise.")
    p.add_argument("--realistic-kv-reads", action="store_true",
                   help="Also populate input_batch.block_table with distinct "
                        "random block ids across the KV cache so the reads "
                        "scatter (instead of all hitting the same low-id "
                        "blocks).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    rank_print(rank,
        f"[rank {rank}] cuda:{local_rank}  ws={world_size}  "
        f"noise={args.noise_mode}  sweep={args.sweep_gbps}  iters={args.iters}")

    # ---------- build LLM once ----------
    torch.cuda.nvtx.range_push("bench/setup/build_llm")
    t0 = time.perf_counter()
    llm = build_llm(args, world_size)
    setup_s = time.perf_counter() - t0
    torch.cuda.nvtx.range_pop()
    rank_print(rank, f"[rank {rank}] LLM ready in {setup_s:.1f}s")

    model_runner = get_model_runner(llm)
    rank_print(rank,
        f"[rank {rank}] cudagraph_mode = {model_runner.compilation_config.cudagraph_mode}")

    # ---------- realism patch ----------
    if args.realistic_kv_reads:
        seq_for_bt = args.profile_seq_lens or 1
        try:
            n_distinct = populate_realistic_block_table(
                model_runner,
                num_reqs=args.num_tokens,
                max_seq_len=seq_for_bt,
                seed=0x5EED + rank,
            )
            rank_print(rank,
                f"[rank {rank}] realistic_kv_reads: populated block_table "
                f"with {n_distinct} distinct block ids "
                f"(seq_lens={seq_for_bt})")
        except Exception as e:
            rank_print(rank,
                f"[rank {rank}] WARN: realistic_kv_reads failed: {e!r}; "
                f"continuing with default block_table")

    # ---------- noise harness ----------
    nixl_harness: NixlNoiseHarness | None = None
    if args.noise_mode == "nixl":
        if args.victim_rank == args.peer_rank:
            raise SystemExit("--victim-rank and --peer-rank must differ")
        if not (0 <= args.victim_rank < world_size and 0 <= args.peer_rank < world_size):
            raise SystemExit("victim/peer rank out of range")
        block_bytes = args.nixl_block_kb * 1024
        donor_bytes = args.nixl_donor_mb * 1024 * 1024
        donor_blocks = donor_bytes // block_bytes
        if donor_blocks < args.nixl_blocks_per_request:
            raise SystemExit("donor too small for blocks_per_request")
        rank_print(rank,
            f"[rank {rank}] NIXL: donor_blocks={donor_blocks} "
            f"block={block_bytes}B req_blocks={args.nixl_blocks_per_request} "
            f"req_per_burst={args.nixl_requests_per_burst}")
        nixl_harness = NixlNoiseHarness(
            rank=rank, local_rank=local_rank, world_size=world_size,
            victim_rank=args.victim_rank, peer_rank=args.peer_rank,
            donor_blocks=donor_blocks, block_size_bytes=block_bytes,
            blocks_per_request=args.nixl_blocks_per_request,
            max_in_flight=args.nixl_max_in_flight,
        )
        nixl_harness.initialize()
    elif args.noise_mode == "p2p":
        if rank == args.victim_rank and args.p2p_src_device >= torch.cuda.device_count():
            raise SystemExit(
                f"--p2p-src-device {args.p2p_src_device} not visible "
                f"(device_count={torch.cuda.device_count()})"
            )

    is_victim = (rank == args.victim_rank)
    phase_results: list[dict[str, Any]] = []

    # ---------- quiet baseline ----------
    if not args.skip_quiet:
        if dist.is_initialized():
            dist.barrier(device_ids=[device.index])
        torch.cuda.nvtx.range_push("bench/phase/quiet")
        rank_print(rank, f"[rank {rank}] >>> quiet baseline")
        quiet = bench_phase(model_runner, "quiet",
                            num_tokens=args.num_tokens,
                            warmup_iters=args.warmup_iters,
                            timed_iters=args.iters, device=device,
                            profile_seq_lens=args.profile_seq_lens)
        torch.cuda.nvtx.range_pop()
        quiet["target_gbps"] = 0.0
        quiet["effective_gbps"] = 0.0
        phase_results.append(quiet)
        rank_print(rank,
            f"[rank {rank}] quiet  p50={quiet['p50_ms']:.3f}  "
            f"p90={quiet['p90_ms']:.3f}  p99={quiet['p99_ms']:.3f}  "
            f"host_p99={quiet['host_p99_ms']:.3f}  "
            f"min={quiet['min_ms']:.3f}  max={quiet['max_ms']:.3f}")

    # ---------- sweep ----------
    for gbps in args.sweep_gbps:
        if args.cooldown_s > 0:
            time.sleep(args.cooldown_s)
        if args.noise_mode == "none":
            break

        traffic_p2p: P2PBackgroundTraffic | None = None
        traffic_nixl: NixlBackgroundTraffic | None = None
        if is_victim:
            if args.noise_mode == "p2p":
                traffic_p2p = P2PBackgroundTraffic(
                    dst_device=local_rank, src_device=args.p2p_src_device,
                    target_gbps=gbps, chunk_mb=args.p2p_chunk_mb,
                    in_flight=args.p2p_in_flight,
                )
                traffic_p2p.start()
            elif args.noise_mode == "nixl":
                assert nixl_harness is not None
                traffic_nixl = nixl_harness.start_traffic(
                    target_gbps=gbps,
                    burst_interval_ms_min=args.nixl_burst_interval_ms_min,
                    burst_interval_ms_max=args.nixl_burst_interval_ms_max,
                    requests_per_burst=args.nixl_requests_per_burst,
                )
            if args.prelude_s > 0:
                time.sleep(args.prelude_s)

        if dist.is_initialized():
            dist.barrier(device_ids=[device.index])

        phase_name = f"noisy_{int(round(gbps))}gbps_{args.noise_mode}"
        torch.cuda.nvtx.range_push(f"bench/phase/{phase_name}")
        rank_print(rank, f"[rank {rank}] >>> {phase_name}")
        row = bench_phase(model_runner, phase_name,
                          num_tokens=args.num_tokens,
                          warmup_iters=args.warmup_iters,
                          timed_iters=args.iters, device=device,
                          profile_seq_lens=args.profile_seq_lens)
        torch.cuda.nvtx.range_pop()

        if traffic_p2p is not None:
            traffic_p2p.stop()
            row["effective_gbps"] = traffic_p2p.effective_gbps()
            row["launches"] = traffic_p2p.launches
        elif traffic_nixl is not None:
            traffic_nixl.stop()
            row["effective_gbps"] = traffic_nixl.effective_gbps()
            row["nixl_bursts"] = traffic_nixl.burst_count
            row["nixl_requests"] = traffic_nixl.request_count
        else:
            row["effective_gbps"] = 0.0
        row["target_gbps"] = gbps
        phase_results.append(row)

        rank_print(rank,
            f"[rank {rank}] {phase_name}  p50={row['p50_ms']:.3f}  "
            f"p90={row['p90_ms']:.3f}  p99={row['p99_ms']:.3f}  "
            f"host_p99={row['host_p99_ms']:.3f}  "
            f"min={row['min_ms']:.3f}  max={row['max_ms']:.3f}  "
            f"eff={row['effective_gbps']:.2f} GiB/s")

        if dist.is_initialized():
            dist.barrier(device_ids=[device.index])

    # ---------- gather + write ----------
    local_payload = {"rank": rank, "phases": phase_results}
    gathered: list[Any] = [None] * world_size
    if dist.is_initialized():
        dist.all_gather_object(gathered, local_payload)
    else:
        gathered[0] = local_payload

    if rank == 0:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": vars(args) | {"world_size": world_size},
            "ranks": [g for g in gathered if g is not None],
        }
        # Strip non-serializable Path objects.
        payload["config"] = {k: (str(v) if isinstance(v, Path) else v)
                             for k, v in payload["config"].items()}
        args.output_json.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\n[wrote] {args.output_json}", flush=True)

        import csv as _csv
        with args.output_csv.open("w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow([
                "rank", "phase", "noise_mode", "target_gbps", "effective_gbps",
                "iters", "mean_ms", "p50_ms", "p90_ms", "p99_ms",
                "min_ms", "max_ms", "host_p50_ms", "host_p99_ms", "host_max_ms",
                "wall_total_s",
            ])
            for g in gathered:
                if g is None: continue
                for row in g["phases"]:
                    w.writerow([
                        g["rank"], row["phase"], args.noise_mode,
                        row.get("target_gbps", 0.0),
                        row.get("effective_gbps", 0.0),
                        row["iters"], f"{row['mean_ms']:.4f}",
                        f"{row['p50_ms']:.4f}", f"{row['p90_ms']:.4f}",
                        f"{row['p99_ms']:.4f}", f"{row['min_ms']:.4f}",
                        f"{row['max_ms']:.4f}",
                        f"{row['host_p50_ms']:.4f}",
                        f"{row['host_p99_ms']:.4f}",
                        f"{row['host_max_ms']:.4f}",
                        f"{row['wall_total_s']:.4f}",
                    ])
        print(f"[wrote] {args.output_csv}\n", flush=True)

        # Summary.
        print(f"{'rank':>4} {'phase':>28} {'tgt':>7} {'eff':>7}  "
              f"{'p50':>7} {'p90':>7} {'p99':>7} {'hp99':>7} {'max':>7}")
        for g in gathered:
            if g is None: continue
            for row in g["phases"]:
                print(
                    f"{g['rank']:>4} {row['phase']:>28} "
                    f"{row.get('target_gbps', 0.0):>7.1f} "
                    f"{row.get('effective_gbps', 0.0):>7.1f}  "
                    f"{row['p50_ms']:>7.3f} {row['p90_ms']:>7.3f} "
                    f"{row['p99_ms']:>7.3f} {row['host_p99_ms']:>7.3f} "
                    f"{row['max_ms']:>7.3f}"
                )

    with contextlib.suppress(Exception):
        if dist.is_initialized():
            dist.barrier(device_ids=[device.index])

    if nixl_harness is not None:
        nixl_harness.close()
    del llm


if __name__ == "__main__":
    main()
