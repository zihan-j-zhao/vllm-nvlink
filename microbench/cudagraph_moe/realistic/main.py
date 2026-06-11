#!/usr/bin/env python3
"""Realistic CUDA-graph decode microbenchmark.

Same model + same captured graphs as ``base/main.py``, but each timed
step decodes a workload of requests with *heterogeneous* prefill lengths
and ages so that:

  - per-request attention seq_lens vary (real paged-attn page walks),
  - block tables point at real, disjoint KV blocks,
  - ``reshape_and_cache`` actually writes K/V into the pool.

Launch with torchrun on 2 GPUs:

    bash run.sh
    bash run.sh --batch-size 16 --prefill-dist uniform:128:4096 --iters 100
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from .driver import RealisticDecodeDriver, prime_captured_graph
from .fake_scheduler import build_workload, parse_int_dist
from .bg_traffic import BackgroundTraffic, PATTERN_REGISTRY, make_pattern


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="HF model id or local path.",
    )
    p.add_argument(
        "--batch-size", type=int, default=16,
        help="Number of concurrent decode requests (per DP rank).",
    )
    p.add_argument(
        "--prefill-dist", default="uniform:128:2048",
        help="Per-request prefill length distribution. "
             "Examples: fixed:1024, uniform:128:8192, geometric:1500, file:lens.json",
    )
    p.add_argument(
        "--age-dist", default="uniform:0:1024",
        help="Per-request starting decode-age distribution. "
             "Examples: fixed:0, uniform:0:4096",
    )
    p.add_argument(
        "--max-decode-steps", type=int, default=32768,
        help="Max decode steps a request runs before its age recycles to 0. "
             "Determines how many KV blocks we pre-allocate per request.",
    )
    p.add_argument(
        "--warmup-iters", type=int, default=10,
    )
    p.add_argument(
        "--iters", type=int, default=50,
    )

    # vLLM bootstrap (matches base/main.py).
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--attention-backend", default="FLASHINFER")
    p.add_argument("--moe-backend", default="triton")
    p.add_argument("--all2all-backend", default="allgather_reducescatter")
    p.add_argument("--no-enable-expert-parallel", action="store_true")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--cudagraph-capture-sizes",
        default=None,
        help="Comma-separated batch sizes to capture CUDA graphs for. "
             "Default: vLLM's built-in list (1,2,4,...,512). Use this if you "
             "need to bench arbitrary B values (e.g. '64,128,256,512,768,1024') "
             "without falling back to eager. Each value triggers one extra "
             "capture at LLM init (~0.2-1s per size).",
    )

    # Background traffic. Fixed mapping: rank R receives CE memcpy from
    # GPU R+4 (decoders 0-3, phantom prefillers 4-7). Only --bg-pattern "off"
    # vs a registered pattern name needs to change.
    p.add_argument(
        "--bg-pattern", default="off",
        choices=["off", *sorted(PATTERN_REGISTRY)],
        help="Background CE-traffic pattern between decoder GPU R and "
             "phantom GPU R+4 (NIXL-style). 'off' = no bg load. Add new "
             "patterns by registering in bg_traffic.PATTERN_REGISTRY.",
    )
    p.add_argument(
        "--bg-direction", default="ingress",
        choices=["ingress", "egress", "both"],
        help="Direction of CE traffic relative to the decoder GPU. "
             "'ingress' (default, NIXL prefill->decoder push): CE engine "
             "on phantom GPU, decoder GPU only writes -> barely contends "
             "with decode forward. 'egress' (decoder->phantom push): CE "
             "engine on decoder GPU, decoder HBM reads contend with the "
             "MoE attention's HBM reads -> shifts p50 for memory-bound "
             "decodes. 'both' runs one of each simultaneously, "
             "approximating bidirectional KV transfer.",
    )
    p.add_argument(
        "--bg-rate-gbps", type=float, default=50.0,
        help="Target rate for the 'constant' pattern, in GB/s (default 50).",
    )
    p.add_argument(
        "--bg-chunk-mb", type=int, default=4,
        help="Chunk size per CE memcpy in MiB (default 4). NIXL-like.",
    )
    p.add_argument(
        "--bg-buffer-mb", type=int, default=64,
        help="Persistent src/dst buffer size in MiB (default 64). Ops larger "
             "than this are clamped.",
    )

    # Parallelism. WORLD_SIZE (from torchrun) must equal tp * pp * dp.
    # Common choices on 2-node-of-4 / 4-GPU setups:
    #   --tp 1 --dp N           : pure data-parallel (current default)
    #   --tp N --dp 1           : pure tensor-parallel (lights up NVLink)
    #   --tp 2 --dp 2           : mixed; production-like for MoE serving
    # `--ep` toggles expert-parallel inside the MoE; when on, expert dispatch
    # uses the `--all2all-backend` over the same NVLink mesh.
    p.add_argument("--tp", type=int, default=1,
                   help="Tensor-parallel size.")
    p.add_argument("--pp", type=int, default=1,
                   help="Pipeline-parallel size.")
    p.add_argument("--dp", type=int, default=None,
                   help="Data-parallel size. Defaults to WORLD_SIZE/(tp*pp).")
    p.add_argument(
        "--validate", action="store_true",
        help="After the first timed step, read back seq_lens/block_table/"
             "slot_mapping and assert they match the workload.",
    )
    p.add_argument(
        "--output-json", type=Path,
        default=Path("results/cudagraph_decode_realistic.json"),
    )
    p.add_argument(
        "--log-dir", type=Path, default=None,
        help="If set, redirect each rank's stdout+stderr to "
             "<log-dir>/rank<R>.log instead of the terminal. The file is "
             "line-buffered so `tail -f` works during the run.",
    )
    return p.parse_args()


def rank_print(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def _redirect_to_logfile(log_dir: Path, rank: int) -> None:
    """Send this process's stdout and stderr to ``<log_dir>/rank<R>.log``.

    Uses os.dup2 on the underlying file descriptors so output from C/CUDA
    libraries (vLLM, NCCL, CUDA driver) is captured too, not just Python
    ``print``. Line-buffered so ``tail -f`` shows live progress.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"rank{rank}.log"
    fh = open(path, "w", buffering=1)  # line-buffered
    fd = fh.fileno()
    os.dup2(fd, sys.stdout.fileno())
    os.dup2(fd, sys.stderr.fileno())
    # Re-bind Python's text wrappers so ``print`` flushes correctly.
    sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)
    # Keep the original handle alive; closing it would close the dup'd fd.
    _redirect_to_logfile._handle = fh  # type: ignore[attr-defined]
    print(f"[rank {rank}] log redirected to {path}", flush=True)


# ---------------------------------------------------------------------------
# vLLM bootstrap (matches base/main.py)
# ---------------------------------------------------------------------------
def build_llm(args: argparse.Namespace, world_size: int):
    from vllm import LLM

    os.environ.setdefault("VLLM_ATTENTION_BACKEND", args.attention_backend)

    # Optionally override which batch sizes vLLM captures CUDA graphs for.
    # vLLM's default capture list maxes out around 512; anything past that
    # falls back to eager (3-10x slower for decode). Bench batch size must
    # be in this list (or pad to one) to stay on the graph fast-path.
    compilation_config: dict[str, Any] | None = None
    if args.cudagraph_capture_sizes is not None:
        sizes = sorted({int(s) for s in args.cudagraph_capture_sizes.split(",") if s.strip()})
        if not sizes:
            raise SystemExit("--cudagraph-capture-sizes parsed to empty list")
        compilation_config = {"cudagraph_capture_sizes": sizes}

    return LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        pipeline_parallel_size=args.pp,
        data_parallel_size=args.dp,
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
        compilation_config=compilation_config,
    )


def get_model_runner(llm) -> Any:
    executor = llm.llm_engine.model_executor
    driver = executor.driver_worker
    worker = getattr(driver, "worker", driver)
    return worker.model_runner


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # Redirect this rank's output *before* importing vLLM so its init
    # logs land in the file too.
    if args.log_dir is not None:
        _redirect_to_logfile(args.log_dir, rank)

    # Resolve / sanity-check parallelism config.
    if args.dp is None:
        denom = args.tp * args.pp
        if world_size % denom != 0:
            raise SystemExit(
                f"WORLD_SIZE={world_size} not divisible by tp*pp={denom}; "
                f"pass --dp explicitly."
            )
        args.dp = world_size // denom
    expected_world = args.tp * args.pp * args.dp
    if expected_world != world_size:
        raise SystemExit(
            f"tp*pp*dp = {args.tp}*{args.pp}*{args.dp} = {expected_world} "
            f"!= WORLD_SIZE={world_size}. Either fix --tp/--pp/--dp or "
            f"relaunch with NPROC={expected_world}."
        )
    rank_print(
        rank,
        f"[rank {rank}] parallelism: tp={args.tp} pp={args.pp} dp={args.dp} "
        f"ep={'on' if not args.no_enable_expert_parallel else 'off'} "
        f"(world_size={world_size})",
    )

    if args.batch_size > args.max_num_seqs:
        raise SystemExit(
            f"--batch-size {args.batch_size} > --max-num-seqs {args.max_num_seqs}"
        )

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    rank_print(rank, f"[rank {rank}] building vLLM LLM (captures CUDA graphs) ...")
    t0 = time.perf_counter()
    llm = build_llm(args, world_size)
    rank_print(rank, f"[rank {rank}] LLM ready in {time.perf_counter() - t0:.1f}s")

    model_runner = get_model_runner(llm)

    # Read KV-cache shape after init.
    kv_cfg = model_runner.kv_cache_config
    block_sizes = [g.kv_cache_spec.block_size for g in kv_cfg.kv_cache_groups]
    num_blocks_per_group = [kv_cfg.num_blocks for _ in kv_cfg.kv_cache_groups]
    # Per-row width of InputBatch.block_table: hard cap on blocks/request.
    ib_block_tables = model_runner.input_batch.block_table.block_tables
    max_blocks_per_req = [bt.max_num_blocks_per_req for bt in ib_block_tables]
    rank_print(
        rank,
        f"[rank {rank}] kv_cache: num_blocks={kv_cfg.num_blocks}, "
        f"groups={len(kv_cfg.kv_cache_groups)}, block_sizes={block_sizes}, "
        f"max_blocks_per_req={max_blocks_per_req}",
    )
    rank_print(
        rank,
        f"[rank {rank}] cudagraph_mode = "
        f"{model_runner.compilation_config.cudagraph_mode}",
    )

    # Build per-rank workload with deterministic seed.
    workload = build_workload(
        batch_size=args.batch_size,
        prefill_sampler=parse_int_dist(args.prefill_dist),
        age_sampler=parse_int_dist(args.age_dist),
        max_decode_steps=args.max_decode_steps,
        block_sizes_per_group=block_sizes,
        num_blocks_per_group=num_blocks_per_group,
        max_blocks_per_req_per_group=max_blocks_per_req,
        seed=args.seed + rank,
    )

    # Quick stats for the report.
    prefill_lens = [r.prefill_len for r in workload]
    rank_print(
        rank,
        f"[rank {rank}] workload: B={args.batch_size}, "
        f"prefill_lens min/median/max = "
        f"{min(prefill_lens)}/{sorted(prefill_lens)[len(prefill_lens)//2]}/"
        f"{max(prefill_lens)}",
    )

    driver = RealisticDecodeDriver(
        model_runner, workload, device=device, validate=args.validate
    )

    # Prime the captured graph for our batch size, then seed input batch.
    rank_print(rank, f"[rank {rank}] priming captured graph for B={args.batch_size} ...")
    prime_captured_graph(model_runner, args.batch_size)
    driver.setup()

    if dist.is_initialized():
        dist.barrier(device_ids=[device.index])

    rank_print(rank, f"[rank {rank}] warmup ({args.warmup_iters} iters) ...")
    driver.warmup(args.warmup_iters)

    if dist.is_initialized():
        dist.barrier(device_ids=[device.index])

    # Optional background CE traffic.
    bgs: list[BackgroundTraffic] = []
    if args.bg_pattern != "off":
        directions = (
            ("ingress", "egress") if args.bg_direction == "both"
            else (args.bg_direction,)
        )
        for direction in directions:
            # Each bg instance owns its own pacer state. With 'both' we
            # split the requested rate evenly between directions so the
            # total wire-time pressure stays comparable.
            per_dir_rate = (
                args.bg_rate_gbps / len(directions)
            ) * 1e9
            pattern = make_pattern(
                args.bg_pattern,
                rate_bytes_per_sec=per_dir_rate,
                chunk_bytes=args.bg_chunk_mb * 1024 * 1024,
            )
            bg = BackgroundTraffic(
                local_rank=local_rank,
                pattern=pattern,
                buffer_bytes=args.bg_buffer_mb * 1024 * 1024,
                direction=direction,
            )
            print(f"[rank {rank}] {bg.describe()}", flush=True)
            bgs.append(bg)

    rank_print(rank, f"[rank {rank}] bench ({args.iters} iters) ...")
    stats = driver.bench(args.iters, bgs=bgs)
    stats.update(
        rank=rank,
        world_size=world_size,
        batch_size=args.batch_size,
        model=args.model,
        prefill_dist=args.prefill_dist,
        age_dist=args.age_dist,
        prefill_min=min(prefill_lens),
        prefill_max=max(prefill_lens),
        prefill_median=sorted(prefill_lens)[len(prefill_lens) // 2],
    )
    rank_print(
        rank,
        f"[rank {rank}] p50={stats['p50_ms']:.3f}ms p90={stats['p90_ms']:.3f}ms "
        f"p99={stats['p99_ms']:.3f}ms min={stats['min_ms']:.3f}ms "
        f"max={stats['max_ms']:.3f}ms",
    )

    # Gather across ranks.
    if dist.is_initialized():
        gathered: list[Any] = [None] * world_size
        dist.all_gather_object(gathered, stats)
    else:
        gathered = [stats]

    if rank == 0:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": vars(args) | {"world_size": world_size},
            "ranks": gathered,
        }
        # argparse Namespace contains Path() which json can't serialize.
        payload["config"]["output_json"] = str(args.output_json)
        args.output_json.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\n[wrote] {args.output_json}", flush=True)
        print(
            f"\n{'rank':>4}  {'p50_ms':>8}  {'p90_ms':>8}  {'p99_ms':>8}  "
            f"{'min_ms':>8}  {'max_ms':>8}  prefill_median",
            flush=True,
        )
        for s in gathered:
            print(
                f"{s['rank']:>4}  {s['p50_ms']:>8.3f}  {s['p90_ms']:>8.3f}  "
                f"{s['p99_ms']:>8.3f}  {s['min_ms']:>8.3f}  {s['max_ms']:>8.3f}  "
                f"{s['prefill_median']}",
                flush=True,
            )

    with contextlib.suppress(Exception):
        if dist.is_initialized():
            dist.barrier(device_ids=[device.index])

    del llm


if __name__ == "__main__":
    main()
