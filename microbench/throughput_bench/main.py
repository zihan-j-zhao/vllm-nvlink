#!/usr/bin/env python3
"""Stage-1 decode stability benchmark.

This is intentionally no-background-traffic only. It reuses the realistic
CUDA-graph decode driver from ``cudagraph_moe/sweep`` but freezes request ages
by default so all timed forwards use the same per-request seq_len.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

MICROBENCH_DIR = Path(__file__).resolve().parents[1]
BENCH_DIR = Path(__file__).resolve().parent
if str(MICROBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(MICROBENCH_DIR))

from cudagraph_moe.sweep.driver import RealisticDecodeDriver, prime_captured_graph
from cudagraph_moe.sweep.fake_scheduler import build_workload, parse_int_dist


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round((len(values) - 1) * pct / 100.0)))
    return values[idx]


def summarize_latencies(values_ms: list[float], tokens_per_step: int) -> dict[str, float]:
    mean_ms = statistics.fmean(values_ms)
    stdev_ms = statistics.stdev(values_ms) if len(values_ms) > 1 else 0.0
    p50_ms = statistics.median(values_ms)
    throughput = [tokens_per_step * 1000.0 / v for v in values_ms]
    return {
        "mean_ms": mean_ms,
        "stdev_ms": stdev_ms,
        "cv_pct": 100.0 * stdev_ms / mean_ms if mean_ms else 0.0,
        "p50_ms": p50_ms,
        "p90_ms": percentile(values_ms, 90),
        "p99_ms": percentile(values_ms, 99),
        "min_ms": min(values_ms),
        "max_ms": max(values_ms),
        "max_abs_delta_from_p50_pct": (
            100.0 * max(abs(v - p50_ms) for v in values_ms) / p50_ms
            if p50_ms else 0.0
        ),
        "throughput_mean_tok_s": statistics.fmean(throughput),
        "throughput_p50_tok_s": statistics.median(throughput),
        "throughput_min_tok_s": min(throughput),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument(
        "--seq-len", type=int, default=2048,
        help="Fixed context length for every request in every timed forward.",
    )
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup-iters", type=int, default=10)
    p.add_argument(
        "--advance-age", action="store_true",
        help="Let seq_len grow by one after each forward, matching the older "
             "realistic driver behavior. Default keeps seq_len fixed.",
    )
    p.add_argument(
        "--block-layout",
        choices=["contiguous", "interleaved"],
        default="contiguous",
        help="Physical KV block layout. 'contiguous' is the original layout; "
             "'interleaved' assigns block 0 for all requests, then block 1 "
             "for all requests, creating strided per-request KV chains.",
    )
    p.add_argument("--max-decode-steps", type=int, default=100)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--max-num-seqs", type=int, default=None)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--attention-backend", default="FLASHINFER")
    p.add_argument("--moe-backend", default="triton")
    p.add_argument("--all2all-backend", default="allgather_reducescatter")
    p.add_argument(
        "--quantization",
        default=None,
        help="Optional vLLM quantization method, e.g. experts_int8, fp8, awq. "
             "experts_int8 quantizes MoE experts while leaving Linear layers "
             "unquantized.",
    )
    p.add_argument("--no-enable-expert-parallel", action="store_true")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--cudagraph-capture-sizes",
        default=None,
        help="Comma-separated extra graph capture sizes. The benchmark batch "
             "size is always included.",
    )
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--pp", type=int, default=1)
    p.add_argument("--dp", type=int, default=None)
    p.add_argument("--validate", action="store_true")
    p.add_argument(
        "--lmcache-kv-traffic", action="store_true",
        help="Spawn LMCache-shaped prefill->decode cuMemcpyDtoDAsync traffic. "
             "Currently intended for TP=1, DP=EP, one prefill GPU per decoder GPU.",
    )
    p.add_argument(
        "--lmcache-prefill-ranks", type=int, default=None,
        help="Number of extra torchrun ranks that act as prefill KV senders. "
             "Default: infer from NPROC - (tp*pp*dp). If this is 0, the "
             "legacy rank-0 subprocess sender mode is used.",
    )
    p.add_argument(
        "--lmcache-prefill-gpu-offset", type=int, default=None,
        help="Source GPU offset for synthetic prefill GPUs. Default: world_size, "
             "so decoder rank r receives from GPU r+world_size.",
    )
    p.add_argument(
        "--lmcache-hotspot-rank", type=int, default=None,
        help="Route all LMCache prefill sender ranks to this decoder local rank "
             "instead of the default 1:1 mapping. Example for DP=2: "
             "--lmcache-hotspot-rank 0 makes GPU2->GPU0 and GPU3->GPU0.",
    )
    p.add_argument(
        "--lmcache-direction",
        choices=["ingress", "egress", "both"],
        default="ingress",
        help="'ingress' copies prefill->decode, matching KV push. 'egress' "
             "copies decode->prefill, useful for multi-turn/prefix-cache "
             "style stress and direct decoder HBM-read contention. 'both' "
             "launches both directions from each prefill sender rank.",
    )
    p.add_argument(
        "--lmcache-chunk-tokens", type=int, default=256,
        help="Tokens per transferred KV chunk. LMCache default is 256. Ignored "
             "when --lmcache-chunk-bytes is set.",
    )
    p.add_argument(
        "--lmcache-chunk-bytes", type=int, default=None,
        help="Direct chunk size in bytes. If set, this overrides "
             "--lmcache-chunk-tokens * KV_bytes_per_token. Use this with "
             "--lmcache-chunks-per-burst to tune traffic by chunk size/count.",
    )
    p.add_argument(
        "--lmcache-kv-bytes-per-token", type=int, default=None,
        help="Override KV bytes per token. Default is inferred from model config "
             "for TP=1.",
    )
    p.add_argument(
        "--lmcache-prefill-tokens-per-burst", type=int, default=None,
        help="Prefill tokens emitted per burst. chunks_per_burst is "
             "ceil(tokens / --lmcache-chunk-tokens).",
    )
    p.add_argument(
        "--lmcache-chunks-per-burst", type=int, default=1,
        help="Directly set chunks per burst when "
             "--lmcache-prefill-tokens-per-burst is not provided.",
    )
    p.add_argument(
        "--lmcache-burst-interval-ms", type=float, default=1.0,
        help="Interval between bursts. 0 means issue bursts as fast as possible.",
    )
    p.add_argument(
        "--lmcache-max-inflight-copies", type=int, default=64,
        help="Maximum in-flight D2D copies per runner before synchronizing events.",
    )
    p.add_argument(
        "--output-json", type=Path,
        default=BENCH_DIR / "results" / "decode_stability.json",
    )
    p.add_argument(
        "--output-csv", type=Path,
        default=None,
        help="Per-iteration CSV path. Default: output JSON with .iter.csv suffix.",
    )
    p.add_argument(
        "--log-dir", type=Path, default=None,
        help="If set, redirect each rank's stdout+stderr to <log-dir>/rank<R>.log.",
    )
    p.add_argument(
        "--skip-final-barrier", action="store_true",
        help="Skip the post-write distributed barrier. Useful when vLLM teardown "
             "hangs after the JSON/CSV have already been written.",
    )
    p.add_argument(
        "--hard-exit-after-write", action="store_true",
        help="Exit each rank with os._exit(0) immediately after rank 0 writes "
             "outputs. This bypasses Python/vLLM/CUDA teardown and should only "
             "be used for microbenchmark runs where persisted JSON/CSV are the "
             "only required outputs.",
    )
    args = p.parse_args()

    if args.max_num_seqs is None:
        args.max_num_seqs = args.batch_size
    if args.max_model_len is None:
        args.max_model_len = max(8192, args.seq_len + args.max_decode_steps + 1024)
    if args.iters <= 0:
        raise SystemExit("--iters must be > 0")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")
    if args.seq_len <= 0:
        raise SystemExit("--seq-len must be > 0")
    if args.batch_size > args.max_num_seqs:
        raise SystemExit(
            f"--batch-size {args.batch_size} > --max-num-seqs {args.max_num_seqs}"
        )
    if args.lmcache_kv_traffic:
        if args.lmcache_chunk_tokens <= 0:
            raise SystemExit("--lmcache-chunk-tokens must be > 0")
        if args.lmcache_chunk_bytes is not None and args.lmcache_chunk_bytes <= 0:
            raise SystemExit("--lmcache-chunk-bytes must be > 0")
        if args.tp != 1 and args.lmcache_chunk_bytes is None:
            raise SystemExit(
                "--lmcache-kv-traffic with TP>1 requires direct "
                "--lmcache-chunk-bytes because KV bytes/token inference is "
                "currently TP=1-only"
            )
        if args.lmcache_chunks_per_burst <= 0:
            raise SystemExit("--lmcache-chunks-per-burst must be > 0")
        if args.lmcache_prefill_tokens_per_burst is not None:
            if args.lmcache_chunk_bytes is not None:
                raise SystemExit(
                    "--lmcache-chunk-bytes cannot be combined with "
                    "--lmcache-prefill-tokens-per-burst; use "
                    "--lmcache-chunks-per-burst instead"
                )
            if args.lmcache_prefill_tokens_per_burst <= 0:
                raise SystemExit("--lmcache-prefill-tokens-per-burst must be > 0")
        if args.lmcache_burst_interval_ms < 0:
            raise SystemExit("--lmcache-burst-interval-ms must be >= 0")
        if args.lmcache_max_inflight_copies <= 0:
            raise SystemExit("--lmcache-max-inflight-copies must be > 0")
        if args.lmcache_prefill_ranks is not None and args.lmcache_prefill_ranks < 0:
            raise SystemExit("--lmcache-prefill-ranks must be >= 0")
        if args.lmcache_hotspot_rank is not None and args.lmcache_hotspot_rank < 0:
            raise SystemExit("--lmcache-hotspot-rank must be >= 0")
    return args


def rank_print(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def redirect_to_logfile(log_dir: Path, rank: int) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"rank{rank}.log"
    fh = open(path, "w", buffering=1)
    fd = fh.fileno()
    os.dup2(fd, sys.stdout.fileno())
    os.dup2(fd, sys.stderr.fileno())
    sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)
    redirect_to_logfile._handle = fh  # type: ignore[attr-defined]
    print(f"[rank {rank}] log redirected to {path}", flush=True)


def capture_sizes(args: argparse.Namespace) -> list[int]:
    sizes = {1, 4, 16, 64, args.batch_size}
    if args.cudagraph_capture_sizes:
        sizes.update(
            int(s) for s in args.cudagraph_capture_sizes.split(",") if s.strip()
        )
    return sorted(sizes)


def build_llm(args: argparse.Namespace, world_size: int):
    from vllm import LLM

    os.environ.setdefault("VLLM_ATTENTION_BACKEND", args.attention_backend)
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
        quantization=args.quantization,
        compilation_config={"cudagraph_capture_sizes": capture_sizes(args)},
    )


def get_model_runner(llm) -> Any:
    executor = llm.llm_engine.model_executor
    driver = executor.driver_worker
    worker = getattr(driver, "worker", driver)
    return worker.model_runner


def infer_kv_bytes_per_token(model_runner: Any, tp: int) -> int:
    if tp != 1:
        raise ValueError("automatic KV bytes/token inference is only wired for TP=1")
    hf_config = model_runner.model_config.hf_config
    n_layers = int(getattr(hf_config, "num_hidden_layers"))
    n_kv_heads = int(
        getattr(
            hf_config,
            "num_key_value_heads",
            getattr(hf_config, "num_attention_heads"),
        )
    )
    head_dim = int(
        getattr(
            hf_config,
            "head_dim",
            getattr(hf_config, "hidden_size") // getattr(hf_config, "num_attention_heads"),
        )
    )
    dtype = getattr(model_runner.model_config, "dtype", torch.bfloat16)
    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    return n_layers * 2 * n_kv_heads * head_dim * dtype_bytes


def ensure_lmcache_runner() -> Path:
    runner = BENCH_DIR / "lmcache_kv_runner"
    source = BENCH_DIR / "lmcache_kv_runner.cu"
    if runner.exists() and runner.stat().st_mtime >= source.stat().st_mtime:
        return runner
    nvcc = shutil_which("nvcc")
    if nvcc is None:
        raise FileNotFoundError(
            "nvcc not found; build LMCache runner manually with: "
            f"nvcc -O3 -std=c++17 -o {runner} {source} -lcuda"
        )
    subprocess.run(
        [nvcc, "-O3", "-std=c++17", "-o", str(runner), str(source), "-lcuda"],
        check=True,
    )
    return runner


def shutil_which(name: str) -> str | None:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(directory) / name
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def lmcache_base_dir(args: argparse.Namespace) -> Path:
    return args.log_dir or args.output_json.parent / f"{args.output_json.stem}_lmcache_kv"


def lmcache_signal_dir(args: argparse.Namespace) -> Path:
    # torchrun --standalone chooses a fresh MASTER_PORT per run. Include it in
    # the rendezvous directory so sender ranks cannot consume stale config/stop
    # files from a previous failed attempt with the same output path.
    run_key = os.environ.get("MASTER_PORT", "default")
    restart = os.environ.get("TORCHELASTIC_RESTART_COUNT", "0")
    return lmcache_base_dir(args) / f"signals_{run_key}_{restart}"


def lmcache_chunks_per_burst(args: argparse.Namespace) -> int:
    return (
        math.ceil(args.lmcache_prefill_tokens_per_burst / args.lmcache_chunk_tokens)
        if args.lmcache_prefill_tokens_per_burst is not None
        else args.lmcache_chunks_per_burst
    )


def lmcache_chunk_bytes(args: argparse.Namespace, kv_bytes_per_token: int) -> int:
    return (
        args.lmcache_chunk_bytes
        if args.lmcache_chunk_bytes is not None
        else args.lmcache_chunk_tokens * kv_bytes_per_token
    )


def lmcache_decoder_device(
    args: argparse.Namespace,
    prefill_idx: int,
    decoder_world_size: int,
) -> int:
    if args.lmcache_hotspot_rank is not None:
        if args.lmcache_hotspot_rank >= decoder_world_size:
            raise ValueError(
                f"--lmcache-hotspot-rank={args.lmcache_hotspot_rank} must be "
                f"< decoder_world_size={decoder_world_size}"
            )
        return args.lmcache_hotspot_rank
    return prefill_idx


def lmcache_transfer_specs(
    args: argparse.Namespace,
    *,
    prefill_idx: int,
    prefill_device: int,
    decoder_world_size: int,
) -> list[tuple[str, int, int]]:
    decoder_device = lmcache_decoder_device(args, prefill_idx, decoder_world_size)
    if args.lmcache_direction == "ingress":
        return [("ingress", prefill_device, decoder_device)]
    if args.lmcache_direction == "egress":
        return [("egress", decoder_device, prefill_device)]
    return [
        ("ingress", prefill_device, decoder_device),
        ("egress", decoder_device, prefill_device),
    ]


def wait_for_path(path: Path, timeout_s: float, what: str) -> None:
    deadline = time.monotonic() + timeout_s
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"timed out waiting for {what}: {path}")
        time.sleep(0.05)


class TorchrunLmcacheKvTraffic:
    """Rank-0 controller for torchrun-owned LMCache KV sender ranks."""

    start_after_profiler = False

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        decoder_world_size: int,
        prefill_world_size: int,
        kv_bytes_per_token: int,
    ):
        self.args = args
        self.decoder_world_size = decoder_world_size
        self.prefill_world_size = prefill_world_size
        self.kv_bytes_per_token = kv_bytes_per_token
        self.chunk_bytes = lmcache_chunk_bytes(args, kv_bytes_per_token)
        self.chunks_per_burst = lmcache_chunks_per_burst(args)
        self.signal_dir = lmcache_signal_dir(args)
        self.steps_dir = self.signal_dir / "steps"
        self.config_file = self.signal_dir / "config.json"
        self.go_file = self.signal_dir / "go"
        self.stop_file = self.signal_dir / "stop"
        self.stats_files: list[Path] = []

    def prepare(self) -> None:
        ensure_lmcache_runner()
        self.steps_dir.mkdir(parents=True, exist_ok=True)
        for stale in self.signal_dir.glob("ready_*"):
            stale.unlink(missing_ok=True)
        for stale in self.steps_dir.glob("step_*"):
            stale.unlink(missing_ok=True)
        for path in (self.config_file, self.go_file, self.stop_file):
            path.unlink(missing_ok=True)

        stats_dir = lmcache_base_dir(self.args)
        stats_dir.mkdir(parents=True, exist_ok=True)
        self.stats_files = []
        for prefill_idx in range(self.prefill_world_size):
            for direction, src_device, dst_device in lmcache_transfer_specs(
                self.args,
                prefill_idx=prefill_idx,
                prefill_device=self.decoder_world_size + prefill_idx,
                decoder_world_size=self.decoder_world_size,
            ):
                self.stats_files.append(
                    stats_dir / (
                        f"lmcache_kv_{direction}_src{src_device}_dst{dst_device}.json"
                    )
                )
        for path in self.stats_files:
            path.unlink(missing_ok=True)

        config = {
            "chunk_bytes": self.chunk_bytes,
            "chunks_per_burst": self.chunks_per_burst,
            "burst_interval_us": int(self.args.lmcache_burst_interval_ms * 1000.0),
            "max_inflight_copies": self.args.lmcache_max_inflight_copies,
            "decoder_world_size": self.decoder_world_size,
            "prefill_world_size": self.prefill_world_size,
            "hotspot_rank": self.args.lmcache_hotspot_rank,
            "direction": self.args.lmcache_direction,
        }
        self.config_file.write_text(json.dumps(config, indent=2))

        ready_files = [
            self.signal_dir / f"ready_{direction}_src{src_device}_dst{dst_device}"
            for i in range(self.prefill_world_size)
            for direction, src_device, dst_device in lmcache_transfer_specs(
                self.args,
                prefill_idx=i,
                prefill_device=self.decoder_world_size + i,
                decoder_world_size=self.decoder_world_size,
            )
        ]
        deadline = time.monotonic() + 120.0
        while any(not path.exists() for path in ready_files):
            if time.monotonic() > deadline:
                missing = [str(p) for p in ready_files if not p.exists()]
                raise TimeoutError(f"timed out waiting for LMCache senders: {missing}")
            time.sleep(0.05)

    def start(self) -> None:
        self.go_file.touch()

    def before_step(self, i: int) -> None:
        (self.steps_dir / f"step_{i}").touch()

    def stop(self) -> dict[str, Any]:
        self.stop_file.touch()
        deadline = time.monotonic() + 15.0
        while any(not path.exists() for path in self.stats_files):
            if time.monotonic() > deadline:
                break
            time.sleep(0.05)
        per_runner: list[dict[str, Any]] = []
        missing: list[str] = []
        for stats_file in self.stats_files:
            if not stats_file.exists():
                missing.append(str(stats_file))
                continue
            payload = json.loads(stats_file.read_text())
            per_runner.extend(payload.get("stats", []))
        achieved = [float(s["achieved_gbps"]) for s in per_runner]
        chunks_per_s = [float(s["chunks_per_s"]) for s in per_runner]
        return {
            "direction": "lmcache_kv",
            "external": False,
            "torchrun_owned_prefill_ranks": True,
            "step_synchronized": True,
            "runners": self.prefill_world_size,
            "transfer_direction": self.args.lmcache_direction,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "chunk_tokens": self.args.lmcache_chunk_tokens,
            "chunk_bytes": self.chunk_bytes,
            "chunk_bytes_override": self.args.lmcache_chunk_bytes is not None,
            "chunks_per_burst": self.chunks_per_burst,
            "prefill_ranks": self.prefill_world_size,
            "hotspot_rank": self.args.lmcache_hotspot_rank,
            "achieved_gbps_mean": (
                statistics.fmean(achieved) if achieved else 0.0
            ),
            "achieved_gbps_total": sum(achieved),
            "chunks_per_s_mean": (
                statistics.fmean(chunks_per_s) if chunks_per_s else 0.0
            ),
            "per_runner": per_runner,
            "missing_stats": missing,
        }


def run_lmcache_prefill_rank(
    args: argparse.Namespace,
    rank: int,
    local_rank: int,
    decoder_world_size: int,
) -> None:
    signal_dir = lmcache_signal_dir(args)
    config_file = signal_dir / "config.json"
    wait_for_path(config_file, timeout_s=1800.0, what="LMCache config")
    config = json.loads(config_file.read_text())
    prefill_idx = rank - decoder_world_size
    if prefill_idx < 0:
        raise SystemExit(f"rank {rank} is not a prefill rank")

    runner = ensure_lmcache_runner()
    stats_dir = lmcache_base_dir(args)
    steps_dir = signal_dir / "steps"
    stats_dir.mkdir(parents=True, exist_ok=True)
    procs: list[tuple[subprocess.Popen, Any]] = []
    for direction, src_device, dst_device in lmcache_transfer_specs(
        args,
        prefill_idx=prefill_idx,
        prefill_device=local_rank,
        decoder_world_size=decoder_world_size,
    ):
        suffix = f"{direction}_src{src_device}_dst{dst_device}"
        log_path = stats_dir / f"lmcache_kv_{suffix}.log"
        ready_file = signal_dir / f"ready_{suffix}"
        stats_file = stats_dir / f"lmcache_kv_{suffix}.json"
        cmd = [
            str(runner),
            "--src-device", str(src_device),
            "--dst-device", str(dst_device),
            "--chunk-bytes", str(config["chunk_bytes"]),
            "--buffer-bytes", str(config["chunk_bytes"]),
            "--chunks-per-burst", str(config["chunks_per_burst"]),
            "--burst-interval-us", str(config["burst_interval_us"]),
            "--max-inflight-copies", str(config["max_inflight_copies"]),
            "--ready-file", str(ready_file),
            "--go-file", str(signal_dir / "go"),
            "--stop-file", str(signal_dir / "stop"),
            "--step-signal-dir", str(steps_dir),
            "--stats-out", str(stats_file),
        ]
        fh = open(log_path, "w", buffering=1)
        procs.append((subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT), fh))
    exit_code = 0
    for proc, fh in procs:
        rc = proc.wait()
        fh.close()
        if rc != 0 and exit_code == 0:
            exit_code = rc
    raise SystemExit(exit_code)


class ExternalLmcacheKvTraffic:
    """Rank-0 controller for one LMCache D2D runner per decoder rank."""

    start_after_profiler = False

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        world_size: int,
        kv_bytes_per_token: int,
    ):
        self.args = args
        self.world_size = world_size
        self.kv_bytes_per_token = kv_bytes_per_token
        self.chunk_bytes = lmcache_chunk_bytes(args, kv_bytes_per_token)
        self.chunks_per_burst = (
            math.ceil(
                args.lmcache_prefill_tokens_per_burst
                / args.lmcache_chunk_tokens
            )
            if args.lmcache_prefill_tokens_per_burst is not None
            else args.lmcache_chunks_per_burst
        )
        self.burst_interval_us = int(args.lmcache_burst_interval_ms * 1000.0)
        self.src_offset = (
            args.lmcache_prefill_gpu_offset
            if args.lmcache_prefill_gpu_offset is not None
            else world_size
        )
        base_dir = args.log_dir or args.output_json.parent / "lmcache_kv"
        self.signal_dir = base_dir / "lmcache_signals"
        self.go_file = self.signal_dir / "go"
        self.stop_file = self.signal_dir / "stop"
        self.procs: list[subprocess.Popen] = []
        self.stats_files: list[Path] = []

    def prepare(self) -> None:
        runner = ensure_lmcache_runner()
        self.signal_dir.mkdir(parents=True, exist_ok=True)
        for path in (self.go_file, self.stop_file):
            path.unlink(missing_ok=True)
        for local_rank in range(self.world_size):
            for direction, src_device, dst_device in lmcache_transfer_specs(
                self.args,
                prefill_idx=local_rank,
                prefill_device=local_rank + self.src_offset,
                decoder_world_size=self.world_size,
            ):
                suffix = f"{direction}_src{src_device}_dst{dst_device}"
                log_path = self.signal_dir.parent / f"lmcache_kv_{suffix}.log"
                ready_file = self.signal_dir / f"ready_{suffix}"
                stats_file = self.signal_dir.parent / f"lmcache_kv_{suffix}.json"
                ready_file.unlink(missing_ok=True)
                stats_file.unlink(missing_ok=True)
                self.stats_files.append(stats_file)
                cmd = [
                    str(runner),
                    "--src-device", str(src_device),
                    "--dst-device", str(dst_device),
                    "--chunk-bytes", str(self.chunk_bytes),
                    "--buffer-bytes", str(self.chunk_bytes),
                    "--chunks-per-burst", str(self.chunks_per_burst),
                    "--burst-interval-us", str(self.burst_interval_us),
                    "--max-inflight-copies", str(self.args.lmcache_max_inflight_copies),
                    "--ready-file", str(ready_file),
                    "--go-file", str(self.go_file),
                    "--stop-file", str(self.stop_file),
                    "--stats-out", str(stats_file),
                ]
                fh = open(log_path, "w", buffering=1)
                proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
                proc._log_fh = fh  # type: ignore[attr-defined]
                self.procs.append(proc)

        ready_deadline = time.monotonic() + 120.0
        ready_files = [
            self.signal_dir / f"ready_{direction}_src{src_device}_dst{dst_device}"
            for i in range(self.world_size)
            for direction, src_device, dst_device in lmcache_transfer_specs(
                self.args,
                prefill_idx=i,
                prefill_device=i + self.src_offset,
                decoder_world_size=self.world_size,
            )
        ]
        while any(not path.exists() for path in ready_files):
            failed = [p.returncode for p in self.procs if p.poll() is not None]
            if failed:
                raise RuntimeError(f"LMCache KV runner exited before ready: {failed}")
            if time.monotonic() > ready_deadline:
                raise TimeoutError("timed out waiting for LMCache KV runners")
            time.sleep(0.05)

    def start(self) -> None:
        self.go_file.touch()
        time.sleep(0.02)

    def stop(self) -> dict[str, Any]:
        self.stop_file.touch()
        for proc in self.procs:
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5.0)
            fh = getattr(proc, "_log_fh", None)
            if fh is not None:
                fh.close()

        per_runner: list[dict[str, Any]] = []
        missing: list[str] = []
        for stats_file in self.stats_files:
            if not stats_file.exists():
                missing.append(str(stats_file))
                continue
            payload = json.loads(stats_file.read_text())
            per_runner.extend(payload.get("stats", []))
        achieved = [float(s["achieved_gbps"]) for s in per_runner]
        chunks_per_s = [float(s["chunks_per_s"]) for s in per_runner]
        return {
            "direction": "lmcache_kv",
            "external": True,
            "runners": len(self.procs),
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "chunk_tokens": self.args.lmcache_chunk_tokens,
            "chunk_bytes": self.chunk_bytes,
            "chunk_bytes_override": self.args.lmcache_chunk_bytes is not None,
            "chunks_per_burst": self.chunks_per_burst,
            "burst_interval_ms": self.args.lmcache_burst_interval_ms,
            "prefill_gpu_offset": self.src_offset,
            "hotspot_rank": self.args.lmcache_hotspot_rank,
            "transfer_direction": self.args.lmcache_direction,
            "achieved_gbps_mean": (
                statistics.fmean(achieved) if achieved else 0.0
            ),
            "achieved_gbps_total": sum(achieved),
            "chunks_per_s_mean": (
                statistics.fmean(chunks_per_s) if chunks_per_s else 0.0
            ),
            "per_runner": per_runner,
            "missing_stats": missing,
        }


class FixedSeqLenDecodeDriver(RealisticDecodeDriver):
    def __init__(self, *args: Any, advance_age: bool, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.advance_age = advance_age

    @torch.inference_mode()
    def step(self) -> None:
        if self.advance_age:
            super().step()
            return
        ages = [r.age_steps for r in self.workload]
        super().step()
        for req, age in zip(self.workload, ages):
            req.age_steps = age


def resolve_parallelism(args: argparse.Namespace, world_size: int) -> None:
    if args.dp is None:
        if args.lmcache_kv_traffic and args.lmcache_prefill_ranks is None:
            raise SystemExit(
                "--dp must be explicit when using --lmcache-kv-traffic with "
                "extra torchrun sender ranks"
            )
        denom = args.tp * args.pp
        if world_size % denom != 0:
            raise SystemExit(
                f"WORLD_SIZE={world_size} not divisible by tp*pp={denom}; "
                "pass --dp explicitly."
            )
        args.dp = world_size // denom
    expected_world = args.tp * args.pp * args.dp
    extra_ranks = world_size - expected_world
    if extra_ranks != 0:
        if extra_ranks < 0:
            raise SystemExit(
                f"tp*pp*dp = {expected_world} exceeds torchrun WORLD_SIZE={world_size}"
            )
        if not args.lmcache_kv_traffic:
            raise SystemExit(
                f"tp*pp*dp = {expected_world} != WORLD_SIZE={world_size}. "
                "Extra ranks are only supported with --lmcache-kv-traffic."
            )
        if args.lmcache_prefill_ranks is not None and args.lmcache_prefill_ranks != extra_ranks:
            raise SystemExit(
                f"--lmcache-prefill-ranks={args.lmcache_prefill_ranks} but "
                f"torchrun has {extra_ranks} extra ranks"
            )
        args.lmcache_prefill_ranks = extra_ranks
    elif args.lmcache_prefill_ranks is None:
        args.lmcache_prefill_ranks = 0
    elif args.lmcache_prefill_ranks != 0:
        raise SystemExit(
            f"--lmcache-prefill-ranks={args.lmcache_prefill_ranks}, but "
            f"torchrun WORLD_SIZE={world_size} has no extra ranks beyond "
            f"tp*pp*dp={expected_world}"
        )
    args.decoder_world_size = expected_world
    args.torchrun_world_size = world_size


def write_iter_csv(path: Path, config: dict[str, Any], ranks: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank", "iter", "latency_ms", "logical_tokens_per_s",
        "batch_size", "seq_len", "tp", "pp", "dp", "world_size",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for rank_row in ranks:
            rank = int(rank_row["rank"])
            for i, latency_ms in enumerate(rank_row["per_iter_ms"]):
                writer.writerow({
                    "rank": rank,
                    "iter": i,
                    "latency_ms": latency_ms,
                    "logical_tokens_per_s": (
                        config["batch_size"] * 1000.0 / latency_ms
                    ),
                    "batch_size": config["batch_size"],
                    "seq_len": config["seq_len"],
                    "tp": config["tp"],
                    "pp": config["pp"],
                    "dp": config["dp"],
                    "world_size": config["world_size"],
                })


def aggregate_summary(
    gathered: list[dict[str, Any]], args: argparse.Namespace,
) -> dict[str, Any]:
    iters = min(len(r["per_iter_ms"]) for r in gathered)
    wall_ms = [
        max(float(r["per_iter_ms"][i]) for r in gathered)
        for i in range(iters)
    ]
    logical_tokens_per_step = args.batch_size * args.dp
    summary = summarize_latencies(wall_ms, logical_tokens_per_step)
    summary["logical_tokens_per_step"] = logical_tokens_per_step
    summary["wall_time_source"] = "max_latency_across_ranks_per_iter"
    return summary


def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if args.log_dir is not None:
        redirect_to_logfile(args.log_dir, rank)

    resolve_parallelism(args, world_size)
    decoder_world_size = args.decoder_world_size
    if args.lmcache_kv_traffic and rank >= decoder_world_size:
        run_lmcache_prefill_rank(args, rank, local_rank, decoder_world_size)
        return
    if rank >= decoder_world_size:
        raise SystemExit(
            f"rank {rank} is outside decoder world size {decoder_world_size}"
        )

    # vLLM should only see decoder ranks. Extra torchrun ranks are owned by
    # the LMCache sender path and do not join vLLM's process group.
    os.environ["WORLD_SIZE"] = str(decoder_world_size)
    world_size = decoder_world_size

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    rank_print(
        rank,
        f"[rank {rank}] stability bench: B={args.batch_size} seq_len={args.seq_len} "
        f"iters={args.iters} tp={args.tp} pp={args.pp} dp={args.dp}",
    )
    rank_print(rank, f"[rank {rank}] capture sizes: {capture_sizes(args)}")

    t0 = time.perf_counter()
    llm = build_llm(args, world_size)
    rank_print(rank, f"[rank {rank}] LLM ready in {time.perf_counter() - t0:.1f}s")
    model_runner = get_model_runner(llm)

    kv_cfg = model_runner.kv_cache_config
    block_sizes = [g.kv_cache_spec.block_size for g in kv_cfg.kv_cache_groups]
    num_blocks_per_group = [kv_cfg.num_blocks for _ in kv_cfg.kv_cache_groups]
    max_blocks_per_req = [
        bt.max_num_blocks_per_req
        for bt in model_runner.input_batch.block_table.block_tables
    ]
    workload = build_workload(
        batch_size=args.batch_size,
        prefill_sampler=parse_int_dist(f"fixed:{args.seq_len}"),
        age_sampler=parse_int_dist("fixed:0"),
        max_decode_steps=args.max_decode_steps,
        block_sizes_per_group=block_sizes,
        num_blocks_per_group=num_blocks_per_group,
        max_blocks_per_req_per_group=max_blocks_per_req,
        seed=args.seed + rank,
        block_layout=args.block_layout,
    )

    driver = FixedSeqLenDecodeDriver(
        model_runner,
        workload,
        device=device,
        validate=args.validate,
        advance_age=args.advance_age,
    )
    prime_captured_graph(model_runner, args.batch_size)
    driver.setup()

    if dist.is_initialized():
        dist.barrier(device_ids=[device.index])
    rank_print(rank, f"[rank {rank}] warmup ({args.warmup_iters} iters)")
    driver.warmup(args.warmup_iters)
    if dist.is_initialized():
        dist.barrier(device_ids=[device.index])

    bgs: list[Any] = []
    if args.lmcache_kv_traffic:
        kv_bytes_per_token = (
            args.lmcache_kv_bytes_per_token
            if args.lmcache_kv_bytes_per_token is not None
            else (
                0
                if args.lmcache_chunk_bytes is not None and args.tp != 1
                else infer_kv_bytes_per_token(model_runner, args.tp)
            )
        )
        chunks_per_burst = (
            math.ceil(
                args.lmcache_prefill_tokens_per_burst
                / args.lmcache_chunk_tokens
            )
            if args.lmcache_prefill_tokens_per_burst is not None
            else args.lmcache_chunks_per_burst
        )
        rank_print(
            rank,
            "[rank 0] LMCache KV traffic: "
            f"chunk_tokens={args.lmcache_chunk_tokens} "
            f"kv_bytes/token={kv_bytes_per_token} "
            f"chunk_bytes={lmcache_chunk_bytes(args, kv_bytes_per_token)} "
            f"chunks_per_burst={chunks_per_burst} "
            f"burst_interval_ms={args.lmcache_burst_interval_ms}",
        )
        if rank == 0:
            if args.lmcache_prefill_ranks:
                lmcache_bg = TorchrunLmcacheKvTraffic(
                    args=args,
                    decoder_world_size=world_size,
                    prefill_world_size=args.lmcache_prefill_ranks,
                    kv_bytes_per_token=kv_bytes_per_token,
                )
            else:
                lmcache_bg = ExternalLmcacheKvTraffic(
                    args=args,
                    world_size=world_size,
                    kv_bytes_per_token=kv_bytes_per_token,
                )
            lmcache_bg.prepare()
            bgs.append(lmcache_bg)
        if dist.is_initialized():
            dist.barrier(device_ids=[device.index])

    rank_print(rank, f"[rank {rank}] timed forwards ({args.iters} iters)")
    stats = driver.bench(args.iters, bgs=bgs)
    stats.update(
        summarize_latencies(stats["per_iter_ms"], args.batch_size),
        rank=rank,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        fixed_seq_len=not args.advance_age,
        block_layout=args.block_layout,
        prefill_dist=f"fixed:{args.seq_len}",
        age_dist="fixed:0",
        throughput_unit="logical decode tokens/s per DP group",
    )

    if dist.is_initialized():
        gathered: list[Any] = [None] * world_size
        dist.all_gather_object(gathered, stats)
    else:
        gathered = [stats]

    if rank == 0:
        output_csv = args.output_csv or args.output_json.with_suffix(".iter.csv")
        config = vars(args) | {
            "world_size": world_size,
            "capture_sizes": capture_sizes(args),
            "output_json": str(args.output_json),
            "output_csv": str(output_csv),
        }
        payload = {
            "config": config,
            "aggregate": aggregate_summary(gathered, args),
            "ranks": gathered,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, default=str))
        write_iter_csv(output_csv, config, gathered)
        agg = payload["aggregate"]
        print(
            f"[wrote] {args.output_json}\n"
            f"[wrote] {output_csv}\n"
            f"aggregate p50={agg['p50_ms']:.3f}ms "
            f"cv={agg['cv_pct']:.3f}% "
            f"throughput_p50={agg['throughput_p50_tok_s']:.1f} tok/s",
            flush=True,
        )

    if args.hard_exit_after_write:
        if dist.is_initialized() and not args.skip_final_barrier:
            dist.barrier(device_ids=[device.index])
        os._exit(0)

    if dist.is_initialized() and not args.skip_final_barrier:
        dist.barrier(device_ids=[device.index])
    del llm


if __name__ == "__main__":
    main()
