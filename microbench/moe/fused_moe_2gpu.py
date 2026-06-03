#!/usr/bin/env python3
"""Two-rank vLLM FusedMoE layer runner.

This intentionally starts from vLLM's Qwen3 MoE block instead of a standalone
kernel call, so the benchmark uses vLLM's gate, FusedMoE layer, backend
selection, expert-parallel layout, and MoE runner path.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from transformers import Qwen3MoeConfig

from vllm.config import ModelConfig, ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock

try:
    from nixl._api import nixl_agent, nixl_agent_config
except ImportError:  # Optional unless --kv-transfer-mb > 0.
    nixl_agent = None
    nixl_agent_config = None


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round((len(values) - 1) * pct / 100.0)))
    return values[idx]


def rank_print(rank: int, message: str) -> None:
    if rank == 0:
        print(message, flush=True)


def barrier(device: torch.device) -> None:
    device_index = device.index
    assert device_index is not None
    dist.barrier(device_ids=[device_index])


def kv_enabled(args: argparse.Namespace) -> bool:
    return args.kv_transfer_mb > 0.0


def wait_for_path(path: Path, timeout_s: float = 120.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(0.05)


def write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class NixlKvTransfer:
    """Real NIXL READ traffic from producer ranks into decode ranks.

    torchrun ranks 0-1 run vLLM MoE. Ranks 2-3 only allocate producer GPU
    buffers and keep their NIXL agents alive. Decode ranks issue READs from
    those producer buffers before each MoE forward.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        rank: int,
        local_rank: int,
        world_size: int,
    ) -> None:
        if nixl_agent is None:
            raise RuntimeError("NIXL Python package is required for --kv-transfer-mb")
        self.args = args
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.decode_ranks = list(range(args.decode_ranks))
        self.producer_ranks = list(
            range(args.decode_ranks, args.decode_ranks + args.kv_producer_ranks)
        )
        self.is_decode = rank in self.decode_ranks
        self.is_producer = rank in self.producer_ranks
        self.rendezvous_dir = (
            args.work_dir
            / ".nixl_rendezvous"
            / os.environ.get("MASTER_PORT", "standalone")
        )
        self.memory_type = "VRAM"
        self.chunk_bytes = max(1, int(args.kv_chunk_mb * 1024 * 1024))
        self.chunks_per_read = max(1, math.ceil(args.kv_transfer_mb / args.kv_chunk_mb))
        self.max_slots = max(
            1, args.kv_waves * args.kv_concurrent_reads * self.chunks_per_read
        )
        config = nixl_agent_config(capture_telemetry=True) if nixl_agent_config else None
        self.agent = nixl_agent(f"fused-moe-kv-r{rank}-{os.getpid()}", config)
        self.registered_descs: Any | None = None
        self.remote_agents: dict[int, str] = {}
        self.remote_meta: dict[int, dict[str, Any]] = {}
        self.handles: dict[tuple[int, int, int], Any] = {}

    def initialize(self) -> None:
        if self.rank == 0:
            shutil.rmtree(self.rendezvous_dir, ignore_errors=True)
            self.rendezvous_dir.mkdir(parents=True, exist_ok=True)
            (self.rendezvous_dir / "ready").write_text("1", encoding="utf-8")
        else:
            wait_for_path(self.rendezvous_dir / "ready")

        self._allocate_and_register()
        self._publish_metadata()
        self._load_remote_metadata()

    def _allocate_and_register(self) -> None:
        device = torch.device(f"cuda:{self.local_rank}")
        tensors: list[torch.Tensor] = []
        if self.is_decode:
            self.recv_buffers = [
                torch.empty(self.chunk_bytes, device=device, dtype=torch.uint8)
                for _ in range(self.max_slots)
            ]
            for buffer in self.recv_buffers:
                buffer.zero_()
            tensors.extend(self.recv_buffers)
        else:
            self.recv_buffers = []

        if self.is_producer:
            self.send_buffers = [
                torch.empty(self.chunk_bytes, device=device, dtype=torch.uint8)
                for _ in range(self.max_slots)
            ]
            for buffer in self.send_buffers:
                buffer.fill_(self.rank)
            tensors.extend(self.send_buffers)
        else:
            self.send_buffers = []

        if tensors:
            torch.cuda.synchronize(self.local_rank)
            descs = self.agent.get_reg_descs(tensors, self.memory_type)
            self.agent.register_memory(descs)
            self.registered_descs = descs

    def _publish_metadata(self) -> None:
        agent_metadata = self.agent.get_agent_metadata()
        write_json(
            self.rendezvous_dir / f"agent_rank{self.rank}.json",
            {"agent": base64.b64encode(agent_metadata).decode("ascii")},
        )
        if self.is_producer:
            write_json(
                self.rendezvous_dir / f"producer_rank{self.rank}.json",
                {
                    "device": self.local_rank,
                    "slots": [
                        {"ptr": int(buffer.data_ptr()), "nbytes": int(buffer.numel())}
                        for buffer in self.send_buffers
                    ],
                },
            )

    def _load_remote_metadata(self) -> None:
        if not self.is_decode:
            return
        for peer in self.producer_ranks:
            if peer == self.rank:
                continue
            wait_for_path(self.rendezvous_dir / f"agent_rank{peer}.json")
            agent_data = read_json(self.rendezvous_dir / f"agent_rank{peer}.json")
            metadata = base64.b64decode(agent_data["agent"])
            self.remote_agents[peer] = self.agent.add_remote_agent(metadata)
            wait_for_path(self.rendezvous_dir / f"producer_rank{peer}.json")
            self.remote_meta[peer] = read_json(
                self.rendezvous_dir / f"producer_rank{peer}.json"
            )

    def wait_for_done(self) -> None:
        for decode_rank in self.decode_ranks:
            wait_for_path(
                self.rendezvous_dir / f"done_rank{decode_rank}",
                timeout_s=self.args.producer_timeout_s,
            )

    def mark_done(self) -> None:
        (self.rendezvous_dir / f"done_rank{self.rank}").write_text("1", encoding="utf-8")

    def _get_handle(self, src_rank: int, slot: int, nbytes: int) -> Any:
        key = (src_rank, slot, nbytes)
        cached = self.handles.get(key)
        if cached is not None:
            return cached

        local_buffer = self.recv_buffers[slot]
        local_desc = self.agent.get_xfer_descs(
            [(local_buffer.data_ptr(), nbytes, self.local_rank)],
            self.memory_type,
        )
        local_handle = self.agent.prep_xfer_dlist("NIXL_INIT_AGENT", local_desc)

        remote = self.remote_meta[src_rank]
        remote_slot = remote["slots"][slot]
        if nbytes > int(remote_slot["nbytes"]):
            raise RuntimeError("producer send buffer is too small")
        remote_desc = self.agent.get_xfer_descs(
            [(int(remote_slot["ptr"]), nbytes, int(remote["device"]))],
            self.memory_type,
        )
        remote_handle = self.agent.prep_xfer_dlist(
            self.remote_agents[src_rank], remote_desc
        )
        xfer = self.agent.make_prepped_xfer("READ", local_handle, [0], remote_handle, [0])
        self.handles[key] = (local_handle, remote_handle, xfer)
        return self.handles[key]

    def launch(self) -> list[Any]:
        if not self.is_decode or self.args.kv_transfer_mb <= 0.0:
            return []
        handles = []
        slot = 0
        bytes_per_read = int(self.args.kv_transfer_mb * 1024 * 1024)
        for wave_idx in range(self.args.kv_waves):
            torch.cuda.nvtx.range_push(f"nixl_wave_{wave_idx}")
            for read_idx in range(self.args.kv_concurrent_reads):
                src_rank = self.producer_ranks[
                    (self.rank + wave_idx + read_idx) % len(self.producer_ranks)
                ]
                remaining = bytes_per_read
                while remaining > 0:
                    nbytes = min(self.chunk_bytes, remaining)
                    _, _, xfer = self._get_handle(src_rank, slot, nbytes)
                    torch.cuda.nvtx.range_push(
                        f"nixl_read_rank{self.rank}_src{src_rank}_slot{slot}_{nbytes}B"
                    )
                    self.agent.transfer(xfer)
                    torch.cuda.nvtx.range_pop()
                    handles.append(xfer)
                    remaining -= nbytes
                    slot += 1
            torch.cuda.nvtx.range_pop()
        return handles

    def wait(self, handles: list[Any]) -> None:
        pending = set(handles)
        while pending:
            done = []
            for handle in pending:
                state = self.agent.check_xfer_state(handle)
                if state == "DONE":
                    done.append(handle)
                elif state != "PROC":
                    raise RuntimeError(f"NIXL transfer failed: state={state}")
            for handle in done:
                pending.remove(handle)
            if pending:
                time.sleep(0.0002)

    def close(self) -> None:
        for local_handle, remote_handle, xfer in self.handles.values():
            self.agent.release_xfer_handle(xfer)
            self.agent.release_dlist_handle(local_handle)
            self.agent.release_dlist_handle(remote_handle)
        self.handles.clear()
        if self.registered_descs is not None:
            self.agent.deregister_memory(self.registered_descs)
            self.registered_descs = None


def make_synthetic_qwen_config(args: argparse.Namespace, rank: int) -> Path:
    config_dir = args.work_dir / ".synthetic_qwen3_moe" / f"rank{rank}"
    config_dir.mkdir(parents=True, exist_ok=True)
    hf_config = Qwen3MoeConfig(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        moe_intermediate_size=args.moe_intermediate_size,
        num_experts=args.num_experts,
        num_experts_per_tok=args.top_k,
        hidden_act="silu",
        norm_topk_prob=True,
        shared_expert_intermediate_size=0,
        decoder_sparse_step=1,
        num_hidden_layers=args.layers,
    )
    hf_config.architectures = ["Qwen3MoeForCausalLM"]
    hf_config.save_pretrained(config_dir)
    return config_dir


def make_vllm_config(args: argparse.Namespace, rank: int, world_size: int) -> VllmConfig:
    config_dir = make_synthetic_qwen_config(args, rank)
    model_config = ModelConfig(
        model=str(config_dir),
        tokenizer=str(config_dir),
        dtype=torch.bfloat16,
        max_model_len=args.max_model_len,
        skip_tokenizer_init=True,
        enforce_eager=not args.cuda_graph,
        trust_remote_code=False,
    )
    parallel_config = ParallelConfig(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=world_size,
        data_parallel_size_local=world_size,
        data_parallel_rank=rank,
        enable_expert_parallel=True,
        all2all_backend=args.all2all_backend,
        distributed_executor_backend="external_launcher",
    )
    return VllmConfig(model_config=model_config, parallel_config=parallel_config)


def setup_distributed(rank: int, local_rank: int, world_size: int) -> None:
    torch.cuda.set_device(local_rank)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        backend="nccl",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        backend="nccl",
    )


def init_random_weights(module: nn.Module, seed: int, device: torch.device) -> None:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    with torch.no_grad():
        for param in module.parameters():
            if param.is_floating_point():
                param.normal_(mean=0.0, std=0.02, generator=generator)
            else:
                param.zero_()


def build_layers(
    args: argparse.Namespace,
    vllm_config: VllmConfig,
    device: torch.device,
) -> nn.ModuleList:
    layers = nn.ModuleList()
    for layer_idx in range(args.layers):
        block = Qwen3MoeSparseMoeBlock(
            vllm_config,
            prefix=f"model.layers.{layer_idx}.mlp",
        ).to(device=device, dtype=torch.bfloat16)
        init_random_weights(block, args.weight_seed + layer_idx, device)
        block.experts.quant_method.process_weights_after_loading(block.experts)
        if block.experts.quant_method.moe_kernel is None:
            raise RuntimeError("vLLM FusedMoEKernel was not initialized")
        layers.append(block)
    return layers


def make_hidden(args: argparse.Namespace, rank: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(args.input_seed + rank)
    return torch.randn(
        args.tokens,
        args.hidden_size,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )


def forward_layers(
    args: argparse.Namespace,
    layers: nn.ModuleList,
    hidden_states: torch.Tensor,
    vllm_config: VllmConfig,
    world_size: int,
    device: torch.device,
) -> torch.Tensor:
    num_tokens_across_dp = torch.full(
        (world_size,),
        hidden_states.shape[0],
        device="cpu",
        dtype=torch.int,
    )
    with set_forward_context(
        None,
        vllm_config,
        num_tokens=hidden_states.shape[0],
        num_tokens_across_dp=num_tokens_across_dp,
    ):
        out = hidden_states
        for layer_idx, layer in enumerate(layers):
            if args.sync_per_layer and not args.cuda_graph:
                barrier(device)
            torch.cuda.nvtx.range_push(f"fused_moe_layer_{layer_idx}")
            out = layer(out)
            torch.cuda.nvtx.range_pop()
        return out


class LayerGraphRunner:
    def __init__(
        self,
        args: argparse.Namespace,
        layers: nn.ModuleList,
        hidden_states: torch.Tensor,
        vllm_config: VllmConfig,
        world_size: int,
        device: torch.device,
    ) -> None:
        self.args = args
        self.layers = layers
        self.static_hidden_states = hidden_states
        self.vllm_config = vllm_config
        self.world_size = world_size
        self.device = device
        self.graph: torch.cuda.CUDAGraph | None = None
        self.static_output: torch.Tensor | None = None
        if args.cuda_graph:
            self.capture()

    def eager(self) -> torch.Tensor:
        return forward_layers(
            self.args,
            self.layers,
            self.static_hidden_states,
            self.vllm_config,
            self.world_size,
            self.device,
        )

    def capture(self) -> None:
        if self.args.sync_per_layer:
            raise RuntimeError("--cuda-graph cannot be combined with --sync-per-layer")
        barrier(self.device)
        torch.cuda.synchronize(self.device)
        warmup_stream = torch.cuda.Stream(device=self.device)
        warmup_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(warmup_stream):
            for _ in range(self.args.graph_warmup):
                self.static_output = self.eager()
        torch.cuda.current_stream(self.device).wait_stream(warmup_stream)
        barrier(self.device)
        torch.cuda.synchronize(self.device)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = self.eager()
        barrier(self.device)
        torch.cuda.synchronize(self.device)

    def run(self) -> torch.Tensor:
        if self.graph is None:
            return self.eager()
        self.graph.replay()
        assert self.static_output is not None
        return self.static_output


def time_one(
    runner: LayerGraphRunner,
    device: torch.device,
    kv: NixlKvTransfer | None,
    phase: str,
    iteration: int,
) -> tuple[float, float]:
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    barrier(device)
    torch.cuda.synchronize(device)
    torch.cuda.nvtx.range_push(f"iter_{iteration}_{phase}")
    handles = kv.launch() if kv is not None else []
    torch.cuda.nvtx.range_push(f"moe_forward_{phase}")
    start.record()
    out = runner.run()
    stop.record()
    stop.synchronize()
    torch.cuda.nvtx.range_pop()
    wait_start = time.perf_counter()
    if kv is not None:
        torch.cuda.nvtx.range_push(f"nixl_wait_{phase}")
        kv.wait(handles)
        torch.cuda.nvtx.range_pop()
    kv_wait_ms = (time.perf_counter() - wait_start) * 1000.0
    # Keep the output live and make consecutive iterations resemble decode steps.
    runner.static_hidden_states.copy_(out)
    torch.cuda.nvtx.range_pop()
    return start.elapsed_time(stop), kv_wait_ms


def run_phase(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    phase: str,
    runner: LayerGraphRunner,
    device: torch.device,
    kv: NixlKvTransfer | None,
    kernel_names: list[str],
    moe_backends: list[str],
) -> dict[str, object]:
    times = []
    kv_wait_times = []
    torch.cuda.nvtx.range_push(f"phase_{phase}")
    for iteration in range(args.warmup + args.iters):
        ms, kv_wait_ms = time_one(runner, device, kv, phase, iteration)
        if iteration >= args.warmup:
            times.append(ms)
            kv_wait_times.append(kv_wait_ms)
    torch.cuda.nvtx.range_pop()

    kv_active = kv is not None
    return {
        "phase": phase,
        "rank": rank,
        "world_size": world_size,
        "decode_ranks": args.decode_ranks,
        "tokens": args.tokens,
        "layers": args.layers,
        "hidden_size": args.hidden_size,
        "moe_intermediate_size": args.moe_intermediate_size,
        "num_experts": args.num_experts,
        "top_k": args.top_k,
        "all2all_backend": args.all2all_backend,
        "kv_transfer_mb": args.kv_transfer_mb if kv_active else 0.0,
        "kv_concurrent_reads": args.kv_concurrent_reads if kv_active else 0,
        "kv_waves": args.kv_waves if kv_active else 0,
        "kv_chunk_mb": args.kv_chunk_mb if kv_active else 0.0,
        "kv_total_mb_per_decode_rank": (
            args.kv_transfer_mb * args.kv_concurrent_reads * args.kv_waves
            if kv_active
            else 0.0
        ),
        "cuda_graph": int(args.cuda_graph),
        "sync_per_layer": int(args.sync_per_layer),
        "kernel": ",".join(kernel_names),
        "moe_backend": ",".join(moe_backends),
        "p50_ms": statistics.median(times),
        "p90_ms": percentile(times, 90),
        "p99_ms": percentile(times, 99),
        "min_ms": min(times),
        "max_ms": max(times),
        "kv_wait_p50_ms": statistics.median(kv_wait_times),
        "kv_wait_p90_ms": percentile(kv_wait_times, 90),
        "kv_wait_max_ms": max(kv_wait_times),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--out", type=Path, default=Path("results/fused_moe_2gpu.csv"))
    parser.add_argument("--tokens", type=int, default=55)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=6144)
    parser.add_argument("--moe-intermediate-size", type=int, default=768)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--all2all-backend", default="allgather_reducescatter")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument(
        "--allow-graph-collectives",
        action="store_true",
        help=(
            "Try raw torch CUDA graph capture with AGRS collectives. This is "
            "known to hang in this standalone harness today, so it is opt-in."
        ),
    )
    parser.add_argument("--graph-warmup", type=int, default=3)
    parser.add_argument("--sync-per-layer", action="store_true")
    parser.add_argument(
        "--decode-ranks",
        type=int,
        default=2,
        help="Number of ranks that run vLLM MoE. This benchmark expects 2.",
    )
    parser.add_argument(
        "--kv-producer-ranks",
        type=int,
        default=2,
        help="Number of extra ranks that only host NIXL source KV buffers.",
    )
    parser.add_argument(
        "--kv-transfer-mb",
        type=float,
        default=0.0,
        help=(
            "Per READ transfer size in MiB. Use values like 12.5, 64.5, or "
            "88.1 from the e2e transfer-volume distribution. 0 disables NIXL."
        ),
    )
    parser.add_argument(
        "--kv-concurrent-reads",
        type=int,
        default=1,
        help="Concurrent NIXL READs launched by each decode rank per MoE forward.",
    )
    parser.add_argument(
        "--kv-waves",
        type=int,
        default=1,
        help=(
            "Number of e2e-sized NIXL READ waves launched before each MoE forward. "
            "Keep 1 for realism; increase to make overlap visible in nsys."
        ),
    )
    parser.add_argument(
        "--kv-chunk-mb",
        type=float,
        default=32.0,
        help="Maximum chunk size for splitting a larger per-read transfer.",
    )
    parser.add_argument(
        "--phases",
        choices=("auto", "moe", "moe+nixl", "both"),
        default="auto",
        help=(
            "Which measurement phase to run. 'auto' runs MoE-only without "
            "--kv-transfer-mb and MoE+NIXL with it. 'both' emits a MoE-only "
            "phase followed by a MoE+NIXL phase in one nsys capture."
        ),
    )
    parser.add_argument(
        "--producer-timeout-s",
        type=float,
        default=1800.0,
        help="How long NIXL producer ranks wait for decode ranks to finish.",
    )
    parser.add_argument("--weight-seed", type=int, default=1234)
    parser.add_argument("--input-seed", type=int, default=5678)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])
    if args.decode_ranks != 2:
        raise SystemExit("This benchmark currently expects exactly two decode ranks")
    if args.kv_concurrent_reads < 1:
        raise SystemExit("--kv-concurrent-reads must be >= 1")
    if args.kv_waves < 1:
        raise SystemExit("--kv-waves must be >= 1")
    if args.kv_chunk_mb <= 0:
        raise SystemExit("--kv-chunk-mb must be > 0")
    if args.phases == "auto":
        args.phases = "moe+nixl" if kv_enabled(args) else "moe"
    if args.phases in ("moe+nixl", "both") and not kv_enabled(args):
        raise SystemExit("--phases moe+nixl/both requires --kv-transfer-mb > 0")
    expected_world_size = args.decode_ranks + args.kv_producer_ranks if kv_enabled(args) else args.decode_ranks
    if world_size != expected_world_size:
        raise SystemExit(
            f"expected WORLD_SIZE={expected_world_size} for this configuration, got {world_size}"
        )
    if args.cuda_graph and not args.allow_graph_collectives:
        raise SystemExit(
            "--cuda-graph over two-rank AGRS collectives is not enabled by default: "
            "raw torch graph replay hangs in this standalone harness. Pass "
            "--allow-graph-collectives only to debug that path."
        )

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    kv = None
    if kv_enabled(args):
        kv = NixlKvTransfer(args, rank, local_rank, world_size)
        kv.initialize()
        if rank >= args.decode_ranks:
            rank_print(0, f"NIXL producer rank {rank} ready on cuda:{local_rank}")
            kv.wait_for_done()
            kv.close()
            return

    vllm_config = make_vllm_config(args, rank, args.decode_ranks)

    with set_current_vllm_config(vllm_config):
        setup_distributed(rank, local_rank, args.decode_ranks)
        rank_print(rank, "building vLLM Qwen3MoeSparseMoeBlock/FusedMoE layers")
        layers = build_layers(args, vllm_config, device)
        hidden_states = make_hidden(args, rank, device)
        runner = LayerGraphRunner(
            args,
            layers,
            hidden_states,
            vllm_config,
            args.decode_ranks,
            device,
        )

        kernel_names = [
            layer.experts.quant_method.moe_kernel.__class__.__name__ for layer in layers
        ]
        moe_backends = [
            str(getattr(layer.experts.quant_method, "unquantized_backend", "unknown"))
            for layer in layers
        ]
        phase_specs: list[tuple[str, NixlKvTransfer | None]]
        if args.phases == "moe":
            phase_specs = [("moe", None)]
        elif args.phases == "moe+nixl":
            phase_specs = [("moe+nixl", kv)]
        else:
            phase_specs = [("moe", None), ("moe+nixl", kv)]

        local_rows = [
            run_phase(
                args,
                rank,
                world_size,
                phase,
                runner,
                device,
                phase_kv,
                kernel_names,
                moe_backends,
            )
            for phase, phase_kv in phase_specs
        ]
        gathered: list[list[dict[str, object]] | None] = [None] * args.decode_ranks
        dist.all_gather_object(gathered, local_rows)

    destroy_model_parallel()
    destroy_distributed_environment()
    if kv is not None:
        kv.mark_done()
        kv.close()

    if rank == 0:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            row
            for rank_rows in gathered
            if rank_rows is not None
            for row in rank_rows
        ]
        with args.out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(args.out, flush=True)
        for item in rows:
            print(
                f"phase={item['phase']} rank={item['rank']} p50={item['p50_ms']:.3f} ms "
                f"p90={item['p90_ms']:.3f} ms p99={item['p99_ms']:.3f} ms "
                f"kernel={item['kernel']} backend={item['moe_backend']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
