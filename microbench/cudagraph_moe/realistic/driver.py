"""Drive a captured CUDA-graph decode replay with realistic per-request state.

The hot path (`step`) is a stripped-down copy of `GPUModelRunner._dummy_run`
that keeps only the bits that matter for timing the decode forward, but
swaps in our own per-iteration buffer programming so that:

  - ``seq_lens`` / ``positions`` vary per request (not scalar),
  - ``block_table`` GPU view points at real, disjoint blocks,
  - ``slot_mapping`` lands in those blocks so ``reshape_and_cache`` runs,
  - everything else (sampler, drafter, bookkeeping, EPLB) is skipped.
"""

from __future__ import annotations

import contextlib
import statistics
import time
from typing import Any

import numpy as np
import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import set_forward_context

from .fake_scheduler import (
    FakeRequest,
    maybe_grow_block_tables,
    populate_input_batch,
    validate_state,
)
from .bg_traffic import BackgroundTraffic


def _pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round((len(s) - 1) * p / 100.0)))
    return s[idx]


class RealisticDecodeDriver:
    """Per-rank harness that owns a workload and steps the model on it."""

    def __init__(
        self,
        model_runner: Any,
        workload: list[FakeRequest],
        *,
        device: torch.device,
        validate: bool = False,
    ):
        self.runner = model_runner
        self.workload = workload
        self.device = device
        self.validate = validate
        self.batch_size = len(workload)
        self._validated = False

        # Sanity: capture-time uniform_decode_query_len must be 1 for the
        # plain decode case (no speculative decoding).
        assert model_runner.uniform_decode_query_len == 1, (
            "Realistic bench currently supports pure decode "
            "(uniform_decode_query_len == 1); spec decode is out of scope."
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(self) -> None:
        """One-shot: seed the input batch with the workload.

        Mirrors what _update_states would do, minus the parts of vLLM's
        scheduler bookkeeping the bench doesn't care about.
        """
        populate_input_batch(self.runner, self.workload)
        # Pre-fill input_ids with arbitrary non-zero tokens. The decode
        # graph reads from `input_ids.gpu[:B]` so we want valid token IDs;
        # value doesn't matter for timing.
        runner = self.runner
        vocab = runner.model_config.get_vocab_size()
        runner.input_ids.gpu[: self.batch_size] = torch.randint(
            1, vocab, (self.batch_size,), dtype=runner.input_ids.gpu.dtype, device=self.device
        )

    # ------------------------------------------------------------------
    # Per-iteration buffer programming
    # ------------------------------------------------------------------
    def _program_per_iter_buffers(self) -> None:
        """Write per-request seq_lens / positions / slot_mapping into the
        runner's persistent CUDA-graph buffers.

        These are the same buffers that ``_prepare_inputs`` writes during a
        real `execute_model` call — captured graphs read them at fixed
        addresses, so overwriting them in place before replay is what makes
        the bench realistic without recapturing.
        """
        runner = self.runner
        B = self.batch_size

        # 1. Per-request seq_lens (varies!).  Pinned CPU buffer first, then
        #    non_blocking copy to the GPU buffer the graph reads.
        seq_lens_np = np.fromiter(
            (r.seq_len for r in self.workload), dtype=np.int32, count=B
        )
        runner.optimistic_seq_lens_cpu[:B] = torch.from_numpy(seq_lens_np)
        runner.optimistic_seq_lens_cpu[B:].fill_(0)
        runner.seq_lens.copy_(runner.optimistic_seq_lens_cpu, non_blocking=True)

        # 2. query_start_loc = [0, 1, 2, ..., B] (1 query token per request).
        runner.query_start_loc.np[0] = 0
        runner.query_start_loc.np[1 : B + 1] = np.arange(1, B + 1, dtype=np.int32)
        # Pad like vLLM does — non-decreasing required by FlashAttention.
        runner.query_start_loc.np[B + 1 :].fill(B)
        runner.query_start_loc.copy_to_gpu()

        # 3. Positions = prefill_len + age_steps (one token per request).
        positions_np = np.fromiter(
            (r.position for r in self.workload), dtype=np.int64, count=B
        )
        runner.positions[:B].copy_(
            torch.from_numpy(positions_np).pin_memory()
            if runner.pin_memory
            else torch.from_numpy(positions_np),
            non_blocking=True,
        )

        # 4. Has any request crossed a block boundary since last commit?
        #    If so, append a new block id to its row and re-commit.
        if maybe_grow_block_tables(runner, self.workload):
            runner.input_batch.block_table.commit_block_table(B)

        # 5. Slot mapping: triton kernel computes per-token KV slot ids
        #    from block_table[req, pos // bs] * bs + pos % bs.
        runner.input_batch.block_table.compute_slot_mapping(
            B,
            runner.query_start_loc.gpu[: B + 1],
            runner.positions[:B],
        )

        # 6. discard_request_mask: never discard. Required because
        #    _build_attention_metadata doesn't touch it; downstream
        #    bookkeeping would but we skip bookkeeping entirely.
        runner.discard_request_mask.np[:B] = False
        runner.discard_request_mask.copy_to_gpu(B)

        # 7. num_accepted_tokens: 1 per req (no spec decode).
        runner.num_accepted_tokens.np[:B] = 1
        runner.num_accepted_tokens.copy_to_gpu(B)

    # ------------------------------------------------------------------
    # Forward replay
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def step(self) -> None:
        """One realistic decode forward (graph replay if captured)."""
        runner = self.runner
        B = self.batch_size

        with runner.synchronize_input_prep():
            self._program_per_iter_buffers()

            # Dispatch to the captured graph for this (B, uniform-decode) key.
            (
                cudagraph_mode,
                batch_desc,
                should_ubatch,
                num_tokens_across_dp,
                _cg_stats,
            ) = runner._determine_batch_execution_and_padding(
                num_tokens=B,
                num_reqs=B,
                num_scheduled_tokens_np=np.ones(B, dtype=np.int32),
                max_num_scheduled_tokens=1,
                use_cascade_attn=False,
                allow_microbatching=False,
                force_uniform_decode=True,
            )
            assert not should_ubatch, "ubatching not supported in realistic bench"

            num_tokens_padded = batch_desc.num_tokens
            num_reqs_padded = batch_desc.num_reqs or B

            pad_attn = cudagraph_mode == CUDAGraphMode.FULL

            slot_mappings_by_group, slot_mappings = runner._get_slot_mappings(
                num_tokens_padded=num_tokens_padded if pad_attn else B,
                num_reqs_padded=num_reqs_padded if pad_attn else B,
                num_tokens_unpadded=B,
                ubatch_slices=None,
            )

            attn_metadata, _ = runner._build_attention_metadata(
                num_tokens=B,
                num_tokens_padded=num_tokens_padded if pad_attn else None,
                num_reqs=B,
                num_reqs_padded=num_reqs_padded if pad_attn else None,
                max_query_len=1,
                ubatch_slices=None,
                for_cudagraph_capture=False,
                slot_mappings=slot_mappings_by_group,
                use_spec_decode=False,
            )

        # Forward.
        input_ids = runner.input_ids.gpu[:num_tokens_padded]
        positions = runner.positions[:num_tokens_padded]

        with set_forward_context(
            attn_metadata,
            runner.vllm_config,
            num_tokens=num_tokens_padded,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_mode,
            batch_descriptor=batch_desc,
            ubatch_slices=None,
            slot_mapping=slot_mappings,
        ):
            runner.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=None,
                inputs_embeds=None,
            )

        if self.validate and not self._validated:
            torch.cuda.synchronize(self.device)
            validate_state(runner, self.workload)
            self._validated = True

        # Advance lifecycle. Recycle requests that hit max_decode_steps so
        # we can run arbitrarily many iters without exhausting blocks.
        for r in self.workload:
            r.age_steps += 1
            if r.age_steps >= r.max_decode_steps:
                r.age_steps = 0

    # ------------------------------------------------------------------
    # Benchmark loop
    # ------------------------------------------------------------------
    def warmup(self, n: int) -> None:
        for _ in range(n):
            self.step()
        torch.cuda.synchronize(self.device)

    def bench(
        self, iters: int, bgs: list[BackgroundTraffic] | None = None,
    ) -> dict[str, Any]:
        # Wrap the timed loop in cudaProfilerStart/Stop + an NVTX range so
        # nsys captures *only* these iters when launched with
        # `--capture-range=cudaProfilerApi`. Outside that flag the calls
        # are cheap no-ops.
        #
        # Each ``bgs`` entry runs on its own CPU thread + side CUDA
        # stream during the timed window. They contend for hardware
        # (NVLink + CE engines + HBM, depending on direction) without
        # entering the model's kernel queue.
        bgs = bgs or []
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        stops = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        torch.cuda.synchronize(self.device)
        for bg in bgs:
            bg.start()
        torch.cuda.profiler.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push(f"realistic_bench iters={iters} B={self.batch_size}")
        try:
            for i in range(iters):
                torch.cuda.nvtx.range_push(f"iter_{i}")
                starts[i].record()
                self.step()
                stops[i].record()
                torch.cuda.nvtx.range_pop()
        finally:
            torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize(self.device)
            torch.cuda.profiler.cudart().cudaProfilerStop()
            bg_stats = [bg.stop() for bg in bgs]
        per_iter_ms = [s.elapsed_time(e) for s, e in zip(starts, stops)]
        result: dict[str, Any] = {
            "iters": iters,
            "per_iter_ms": per_iter_ms,
            "mean_ms": statistics.fmean(per_iter_ms),
            "p50_ms": statistics.median(per_iter_ms),
            "p90_ms": _pct(per_iter_ms, 90),
            "p99_ms": _pct(per_iter_ms, 99),
            "min_ms": min(per_iter_ms),
            "max_ms": max(per_iter_ms),
        }
        if bg_stats:
            result["bg_traffic"] = bg_stats
        return result


# ---------------------------------------------------------------------------
# Helper: pre-touch the captured graph at this batch size.
# ---------------------------------------------------------------------------
def prime_captured_graph(model_runner: Any, batch_size: int) -> None:
    """Run vLLM's own `_dummy_run` once to make sure the FULL graph (or
    PIECEWISE pieces) for ``num_tokens=batch_size, uniform_decode=True``
    have been compiled/JITed.

    Capture itself happened during `LLM(...)` init via ``capture_model``;
    this just primes any per-shape JIT caches before timed iterations.
    """
    with contextlib.suppress(Exception):
        model_runner._dummy_run(
            num_tokens=batch_size,
            uniform_decode=True,
            skip_eplb=True,
            is_profile=False,
            remove_lora=True,
        )
        torch.cuda.synchronize()
        time.sleep(0)  # yield so any background DMAs settle
