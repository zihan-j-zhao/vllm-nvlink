#!/usr/bin/env python3
"""
Microbenchmark: time vLLM's CUDA-graph decode forward path in isolation.

Pattern mirrors how vLLM behaves at steady state during decoding:
- The framework warms up + captures CUDA graphs for each decode batch size.
- After that, every decode forward is a graph replay through
  `GPUModelRunner._dummy_run(num_tokens=N, uniform_decode=True)`.

We piggyback on `LLM(..., distributed_executor_backend="external_launcher")`
so vLLM does all the heavy lifting: distributed init, model load, KV cache,
kernel warmup, and CUDA graph capture. We then reach into the runner and
time exactly the captured replay path for N decode steps.

Launch with torchrun on 2 GPUs:

    bash run.sh
    bash run.sh --batch-sizes 1,4,16 --iters 50

Defaults match the AgRS + MoE Qwen3-30B-A3B setup
(start_server_cudagraph.sh / start_server.sh).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="HF model id or local path.",
    )
    parser.add_argument(
        "--batch-sizes", type=_parse_int_list, default=[1, 4, 16, 64],
        help="Comma-separated decode batch sizes (num_tokens per rank).",
    )
    parser.add_argument(
        "--warmup-iters", type=int, default=5,
        help="Warmup _dummy_run iterations per batch size before timing.",
    )
    parser.add_argument(
        "--iters", type=int, default=5,
        help='How many "5 tokens forward" decode steps to time per batch size.',
    )
    parser.add_argument(
        "--max-model-len", type=int, default=8192,
    )
    parser.add_argument(
        "--max-num-seqs", type=int, default=256,
        help="Must be >= the largest decode batch size you benchmark.",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.85,
    )
    parser.add_argument(
        "--attention-backend", default="FLASHINFER",
    )
    parser.add_argument(
        "--moe-backend", default="triton",
    )
    parser.add_argument(
        "--all2all-backend", default="allgather_reducescatter",
        help='vLLM EP all-to-all backend. "allgather_reducescatter" matches AgRS.',
    )
    parser.add_argument(
        "--no-enable-expert-parallel", action="store_true",
        help="Disable expert parallel (default: enabled).",
    )
    parser.add_argument(
        "--enforce-eager", action="store_true",
        help="Disable CUDA graphs; useful as a baseline.",
    )
    parser.add_argument(
        "--seed", type=int, default=1,
    )
    parser.add_argument(
        "--output-json", type=Path, default=Path("results/cudagraph_decode.json"),
        help="Where rank 0 writes the timing summary.",
    )
    return parser.parse_args()


def build_llm(args: argparse.Namespace, world_size: int):
    # vLLM uses RANK / LOCAL_RANK / WORLD_SIZE / MASTER_* from torchrun.
    from vllm import LLM

    # These knobs are not stable LLM constructor args across vLLM versions,
    # so set them via env vars to match the start_server scripts.
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", args.attention_backend)

    llm_kwargs: dict[str, Any] = dict(
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
        # Forwarded as extra engine args; matches AgRS + MoE config.
        moe_backend=args.moe_backend,
        all2all_backend=args.all2all_backend,
    )
    return LLM(**llm_kwargs)


def get_model_runner(llm) -> Any:
    """Reach into vLLM internals to grab the GPUModelRunner."""
    executor = llm.llm_engine.model_executor
    driver = executor.driver_worker
    # WorkerWrapperBase.worker -> Worker, Worker.model_runner -> GPUModelRunner
    worker = getattr(driver, "worker", driver)
    return worker.model_runner


def cuda_graph_supported(model_runner) -> bool:
    from vllm.config import CUDAGraphMode

    mode = model_runner.compilation_config.cudagraph_mode
    return mode != CUDAGraphMode.NONE


def bench_one_batch_size(
    model_runner,
    num_tokens: int,
    warmup_iters: int,
    timed_iters: int,
    device: torch.device,
) -> dict[str, float]:
    """Time `timed_iters` decode forwards at `num_tokens` per rank.

    All ranks must call this with the same `num_tokens` so DP allgather
    of `num_tokens_across_dp` is consistent.
    """
    # Warmup. _dummy_run with no explicit cudagraph_runtime_mode lets the
    # dispatcher pick FULL/PIECEWISE for the captured shape — i.e. graph replay.
    for _ in range(warmup_iters):
        model_runner._dummy_run(
            num_tokens=num_tokens,
            uniform_decode=True,
            skip_eplb=True,
            is_profile=False,
            remove_lora=True,
        )
    torch.cuda.synchronize(device)
    if dist.is_initialized():
        dist.barrier(device_ids=[device.index])

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(timed_iters)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(timed_iters)]

    # Tight timing loop: one CUDA event pair per replayed decode step.
    for i in range(timed_iters):
        starts[i].record()
        model_runner._dummy_run(
            num_tokens=num_tokens,
            uniform_decode=True,
            skip_eplb=True,
            is_profile=False,
            remove_lora=True,
        )
        stops[i].record()

    torch.cuda.synchronize(device)
    per_iter_ms = [s.elapsed_time(e) for s, e in zip(starts, stops)]

    return {
        "num_tokens": num_tokens,
        "iters": timed_iters,
        "per_iter_ms": per_iter_ms,
        "mean_ms": statistics.fmean(per_iter_ms),
        "p50_ms": statistics.median(per_iter_ms),
        "p90_ms": percentile(per_iter_ms, 90),
        "p99_ms": percentile(per_iter_ms, 99),
        "min_ms": min(per_iter_ms),
        "max_ms": max(per_iter_ms),
    }


def gather_results(local: list[dict[str, Any]], world_size: int) -> list[list[dict[str, Any]]]:
    gathered: list[list[dict[str, Any]] | None] = [None] * world_size
    if dist.is_initialized():
        dist.all_gather_object(gathered, local)
    else:
        gathered[0] = local
    return [g if g is not None else [] for g in gathered]


def main() -> None:
    args = parse_args()

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if max(args.batch_sizes) > args.max_num_seqs:
        raise SystemExit(
            f"max batch size {max(args.batch_sizes)} exceeds --max-num-seqs "
            f"{args.max_num_seqs}; uniform decode needs one seq per token."
        )

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    rank_print(rank, f"[rank {rank}] building vLLM LLM (this captures CUDA graphs) ...")
    t0 = time.perf_counter()
    llm = build_llm(args, world_size)
    setup_s = time.perf_counter() - t0
    rank_print(rank, f"[rank {rank}] LLM ready in {setup_s:.1f}s")

    model_runner = get_model_runner(llm)
    cg_on = cuda_graph_supported(model_runner)
    rank_print(rank, f"[rank {rank}] cudagraph_mode = "
                     f"{model_runner.compilation_config.cudagraph_mode} "
                     f"(graphs {'ENABLED' if cg_on else 'DISABLED'})")

    if not cg_on and not args.enforce_eager:
        rank_print(rank, "[warn] CUDA graphs are disabled by config; timing the eager path.")

    local_results: list[dict[str, Any]] = []
    for batch_size in args.batch_sizes:
        rank_print(rank, f"[rank {rank}] >>> bench decode batch_size={batch_size}")
        # All ranks must enter `_dummy_run` in lockstep for DP.
        if dist.is_initialized():
            dist.barrier(device_ids=[device.index])
        stats = bench_one_batch_size(
            model_runner,
            num_tokens=batch_size,
            warmup_iters=args.warmup_iters,
            timed_iters=args.iters,
            device=device,
        )
        stats.update(
            rank=rank,
            world_size=world_size,
            model=args.model,
            cudagraph=bool(cg_on),
            attention_backend=args.attention_backend,
            moe_backend=args.moe_backend,
            all2all_backend=args.all2all_backend,
            expert_parallel=not args.no_enable_expert_parallel,
        )
        local_results.append(stats)
        rank_print(
            rank,
            f"[rank {rank}]   p50={stats['p50_ms']:.3f} ms "
            f"p90={stats['p90_ms']:.3f} ms p99={stats['p99_ms']:.3f} ms "
            f"min={stats['min_ms']:.3f} max={stats['max_ms']:.3f} "
            f"(per_iter={['%.3f' % x for x in stats['per_iter_ms']]})",
        )

    gathered = gather_results(local_results, world_size)

    if rank == 0:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "model": args.model,
                "world_size": world_size,
                "batch_sizes": args.batch_sizes,
                "warmup_iters": args.warmup_iters,
                "iters": args.iters,
                "attention_backend": args.attention_backend,
                "moe_backend": args.moe_backend,
                "all2all_backend": args.all2all_backend,
                "expert_parallel": not args.no_enable_expert_parallel,
                "enforce_eager": args.enforce_eager,
            },
            "ranks": [{"rank": i, "rows": rows} for i, rows in enumerate(gathered)],
        }
        args.output_json.write_text(json.dumps(payload, indent=2))
        print(f"\n[wrote] {args.output_json}\n", flush=True)
        print(f"{'batch':>6} {'rank':>4}  {'p50_ms':>8}  {'p90_ms':>8}  {'p99_ms':>8}  {'min_ms':>8}  {'max_ms':>8}")
        for rank_idx, rows in enumerate(gathered):
            for row in rows:
                print(
                    f"{row['num_tokens']:>6} {rank_idx:>4}  "
                    f"{row['p50_ms']:>8.3f}  {row['p90_ms']:>8.3f}  "
                    f"{row['p99_ms']:>8.3f}  {row['min_ms']:>8.3f}  "
                    f"{row['max_ms']:>8.3f}",
                    flush=True,
                )

    with contextlib.suppress(Exception):
        if dist.is_initialized():
            dist.barrier(device_ids=[device.index])

    # Let vLLM tear down cleanly.
    del llm


if __name__ == "__main__":
    main()
