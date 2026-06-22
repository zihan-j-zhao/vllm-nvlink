"""Synthetic scheduler / workload generator.

Builds a list of `FakeRequest`s, allocates KV-cache block IDs for them,
and seeds vLLM's `InputBatch` with the per-request state needed to make
attention metadata realistic. Intentionally does NOT call
`_update_states` — the realistic bench replaces that scheduler-driven
update path with a static, pre-allocated workload.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import torch


# Block 0 is reserved by vLLM (see vllm.v1.attention.backends.utils.NULL_BLOCK_ID).
RESERVED_BLOCK_ID = 0


# ---------------------------------------------------------------------------
# Workload spec
# ---------------------------------------------------------------------------
@dataclass
class FakeRequest:
    """One synthetic decode request.

    Attributes mirror the subset of `CachedRequestState` /
    `InputBatch` per-row state that the realistic bench actually needs to
    program for attention metadata to be faithful.
    """

    req_id: str
    req_idx: int                    # row in InputBatch (0..max_num_seqs-1)
    prefill_len: int                # initial context length
    max_decode_steps: int           # how many decode steps before recycling
    age_steps: int = 0              # decode steps already done
    # block_ids[g] is the per-kv-cache-group list of block IDs, sized for
    # the full lifetime (prefill + max_decode_steps + 1 token).
    block_ids: tuple[list[int], ...] = field(default_factory=tuple)

    @property
    def seq_len(self) -> int:
        """seq_lens value for attention metadata (context + 1 new token)."""
        return self.prefill_len + self.age_steps + 1

    @property
    def position(self) -> int:
        """Position id for the single decoded token this step."""
        return self.prefill_len + self.age_steps


# ---------------------------------------------------------------------------
# Distributions for prefill length / starting age
# ---------------------------------------------------------------------------
SamplerFn = Callable[[np.random.Generator], int]
BlockLayout = Literal["contiguous", "interleaved"]


def parse_int_dist(spec: str) -> SamplerFn:
    """Parse a tiny distribution DSL into a sampler ``rng -> int``.

    Supported:
      ``fixed:N``             always returns N
      ``uniform:lo:hi``       uniform integer in [lo, hi]
      ``geometric:mean``      geometric (>=1) with given mean
      ``file:path.json``      JSON list of ints — sampled uniformly with replacement
    """
    if ":" not in spec:
        raise ValueError(f"distribution spec must contain ':' — got {spec!r}")
    kind, _, rest = spec.partition(":")
    kind = kind.lower()

    if kind == "fixed":
        v = int(rest)
        return lambda rng: v

    if kind == "uniform":
        lo_s, _, hi_s = rest.partition(":")
        lo, hi = int(lo_s), int(hi_s)
        if lo > hi:
            raise ValueError(f"uniform: lo > hi ({lo} > {hi})")
        return lambda rng: int(rng.integers(lo, hi + 1))

    if kind == "geometric":
        mean = float(rest)
        # numpy.geometric uses p; mean = 1/p, so p = 1/mean.
        if mean < 1.0:
            raise ValueError("geometric mean must be >= 1")
        p = 1.0 / mean
        return lambda rng: int(rng.geometric(p))

    if kind == "file":
        path = Path(rest).expanduser()
        with path.open() as fh:
            data = json.load(fh)
        if not (isinstance(data, list) and all(isinstance(x, int) for x in data)):
            raise ValueError(f"{path} must contain a JSON list of ints")
        arr = np.array(data, dtype=np.int64)
        return lambda rng: int(arr[rng.integers(0, len(arr))])

    raise ValueError(f"unknown distribution kind {kind!r} in {spec!r}")


# ---------------------------------------------------------------------------
# KV-block allocator
# ---------------------------------------------------------------------------
class BlockPool:
    """Sequential, non-recycling allocator over the KV-cache block range.

    For a bench we don't care about correctness of attention outputs, only
    that block IDs are valid (in ``[1, num_blocks)``) and disjoint across
    requests so prefetch/TLB/page-table behaviour is realistic.

    Per-group: vLLM models with multiple KV-cache groups (e.g. hybrid
    attention/mamba) get a separate allocator per group.
    """

    def __init__(self, num_blocks_per_group: list[int]):
        # cursor starts at 1 to avoid colliding with NULL_BLOCK_ID==0.
        self._cursors = [1] * len(num_blocks_per_group)
        self._caps = list(num_blocks_per_group)

    def alloc(self, n_per_group: list[int]) -> tuple[list[int], ...]:
        out: list[list[int]] = []
        for gid, n in enumerate(n_per_group):
            start = self._cursors[gid]
            end = start + n
            if end > self._caps[gid]:
                raise RuntimeError(
                    f"BlockPool exhausted for group {gid}: tried to alloc {n} "
                    f"more starting at {start}, cap is {self._caps[gid]}. "
                    f"Lower --batch-size or --max-decode-steps, or raise "
                    f"--gpu-memory-utilization."
                )
            out.append(list(range(start, end)))
            self._cursors[gid] = end
        return tuple(out)

    def alloc_interleaved(
        self, n_per_group_per_req: list[list[int]]
    ) -> list[tuple[list[int], ...]]:
        """Allocate block IDs round-robin across requests.

        With equal block counts this produces:

            req0: [1,  B+1, 2B+1, ...]
            req1: [2,  B+2, 2B+2, ...]

        This keeps each request's block order valid while avoiding the
        perfectly contiguous per-request layout of ``alloc``.
        """
        if not n_per_group_per_req:
            return []
        n_groups = len(n_per_group_per_req[0])
        if any(len(req_counts) != n_groups for req_counts in n_per_group_per_req):
            raise ValueError("all requests must have the same number of KV groups")

        out: list[list[list[int]]] = [
            [[] for _ in range(n_groups)]
            for _ in n_per_group_per_req
        ]
        for gid in range(n_groups):
            total = sum(req_counts[gid] for req_counts in n_per_group_per_req)
            start = self._cursors[gid]
            end = start + total
            if end > self._caps[gid]:
                raise RuntimeError(
                    f"BlockPool exhausted for group {gid}: tried to alloc {total} "
                    f"more starting at {start}, cap is {self._caps[gid]}. "
                    f"Lower --batch-size or --max-decode-steps, or raise "
                    f"--gpu-memory-utilization."
                )
            next_block = start
            max_blocks = max(req_counts[gid] for req_counts in n_per_group_per_req)
            for block_idx in range(max_blocks):
                for req_idx, req_counts in enumerate(n_per_group_per_req):
                    if block_idx < req_counts[gid]:
                        out[req_idx][gid].append(next_block)
                        next_block += 1
            self._cursors[gid] = end
        return [tuple(group_ids) for group_ids in out]

    @property
    def used(self) -> list[int]:
        return [c - 1 for c in self._cursors]


# ---------------------------------------------------------------------------
# Workload build
# ---------------------------------------------------------------------------
def build_workload(
    batch_size: int,
    prefill_sampler: SamplerFn,
    age_sampler: SamplerFn,
    max_decode_steps: int,
    block_sizes_per_group: list[int],
    num_blocks_per_group: list[int],
    max_blocks_per_req_per_group: list[int] | None = None,
    seed: int = 1,
    block_layout: BlockLayout = "contiguous",
) -> list[FakeRequest]:
    """Sample a list of FakeRequests and pre-allocate KV blocks for each.

    ``max_blocks_per_req_per_group`` is the per-row width of vLLM's
    ``BlockTable.block_table`` (== ``cdiv(max_model_len, block_size)``).
    If a request would need more blocks than that, we fail fast with a
    clear message instead of letting numpy raise a confusing broadcast
    error inside ``append_row``.

    ``block_layout`` controls the synthetic KV physical layout:
      - ``contiguous`` gives each request a contiguous block chain.
      - ``interleaved`` allocates block 0 for all requests, then block 1 for
        all requests, etc., creating strided per-request block chains.
    """
    if block_layout not in ("contiguous", "interleaved"):
        raise ValueError(
            f"unknown block_layout {block_layout!r}; expected 'contiguous' "
            "or 'interleaved'"
        )
    rng = np.random.default_rng(seed)
    pool = BlockPool(num_blocks_per_group)

    reqs: list[FakeRequest] = []
    req_block_counts: list[list[int]] = []
    for i in range(batch_size):
        prefill_len = max(1, prefill_sampler(rng))
        starting_age = max(0, min(age_sampler(rng), max_decode_steps - 1))
        # Total token range we'll ever index into this request:
        max_total = prefill_len + max_decode_steps
        n_per_group = [
            (max_total + bs - 1) // bs  # ceil_div
            for bs in block_sizes_per_group
        ]
        if max_blocks_per_req_per_group is not None:
            for gid, n in enumerate(n_per_group):
                cap = max_blocks_per_req_per_group[gid]
                if n > cap:
                    bs = block_sizes_per_group[gid]
                    raise ValueError(
                        f"Request {i} would need {n} blocks in kv group {gid} "
                        f"(prefill={prefill_len} + max_decode_steps={max_decode_steps} "
                        f"= {max_total} tokens, block_size={bs}), but the "
                        f"InputBatch BlockTable row only holds {cap} blocks "
                        f"(= cdiv(max_model_len, block_size)). "
                        f"Either lower --prefill-dist / --max-decode-steps, "
                        f"or raise --max-model-len to at least {max_total}."
                    )
        reqs.append(
            FakeRequest(
                req_id=f"fake-{i}",
                req_idx=i,
                prefill_len=prefill_len,
                max_decode_steps=max_decode_steps,
                age_steps=starting_age,
            )
        )
        req_block_counts.append(n_per_group)

    if block_layout == "contiguous":
        block_ids_by_req = [pool.alloc(counts) for counts in req_block_counts]
    else:
        block_ids_by_req = pool.alloc_interleaved(req_block_counts)
    for req, block_ids in zip(reqs, block_ids_by_req):
        req.block_ids = block_ids
    return reqs


# ---------------------------------------------------------------------------
# InputBatch seeding (replaces what _update_states would do)
# ---------------------------------------------------------------------------
def populate_input_batch(model_runner: Any, workload: list[FakeRequest]) -> None:
    """Directly write the per-request fields the attention metadata path reads.

    This bypasses `model_runner._update_states` and writes the minimum
    subset of `InputBatch` state required for `_build_attention_metadata`,
    `_get_slot_mappings`, and `compute_slot_mapping` to produce realistic
    outputs:

      - ``input_batch.req_id_to_index`` / ``_req_ids``  (slot mapping)
      - ``input_batch.num_prompt_tokens_cpu_tensor[i]`` (drives ``is_prefilling``)
      - ``input_batch.num_computed_tokens_cpu_tensor[i]`` (drives ``seq_lens`` math)
      - ``input_batch.num_tokens_no_spec[i]``           (avoids stale token bookkeeping)
      - ``input_batch.block_table[gid]`` rows           (real, disjoint block IDs)

    These are exactly the fields ``_update_states`` would write based on
    a ``SchedulerOutput``.
    """
    ib = model_runner.input_batch

    max_req = ib.max_num_reqs
    if len(workload) > max_req:
        raise ValueError(
            f"workload size {len(workload)} > InputBatch.max_num_reqs "
            f"{max_req}; rerun the LLM with a larger --max-num-seqs."
        )

    # The realistic bench currently supports only homogeneous block sizes
    # (kernel block size == kv-manager block size). When they differ vLLM
    # expands each kv-manager block id into N kernel ids inside add_row,
    # which would invalidate our hand-computed `need_blocks` slicing below.
    for gid, bt in enumerate(ib.block_table.block_tables):
        if getattr(bt, "blocks_per_kv_block", 1) != 1:
            raise NotImplementedError(
                f"kv-cache group {gid} uses hybrid blocks "
                f"(kv_block_size != kernel_block_size); realistic bench "
                f"does not yet support this. Pick a backend where they match."
            )

    # Clear any prior state from warmup / previous workload.
    ib.req_id_to_index.clear()
    ib._req_ids = []  # type: ignore[attr-defined]
    ib.req_output_token_ids = []
    # `spec_token_ids` is pre-allocated to length max_num_reqs in __init__;
    # truncate to our batch and reuse the existing inner lists.
    ib.spec_token_ids = [[] for _ in range(len(workload))]
    ib.block_table.clear()

    # Per-request static state.
    for req in workload:
        idx = req.req_idx
        assert len(req.block_ids) == len(ib.block_table.block_tables), (
            f"req.block_ids has {len(req.block_ids)} groups but the input "
            f"batch has {len(ib.block_table.block_tables)} kv-cache groups"
        )
        # Initially seed with enough blocks to cover the starting seq_len.
        # `bt.block_size == kv-manager block size` here (asserted above) so
        # slicing the same list works.
        n_per_group_now: list[list[int]] = []
        for gid, bt in enumerate(ib.block_table.block_tables):
            need_blocks = (req.seq_len + bt.block_size - 1) // bt.block_size
            n_per_group_now.append(req.block_ids[gid][:need_blocks])
        ib.block_table.add_row(tuple(n_per_group_now), idx)

        # Bookkeeping vLLM otherwise derives from CachedRequestState:
        ib.num_prompt_tokens[idx] = req.prefill_len
        ib.num_computed_tokens_cpu[idx] = req.prefill_len + req.age_steps
        ib.num_tokens_no_spec[idx] = req.prefill_len + req.age_steps
        # _req_ids is grown in lockstep with req_idx; we rely on req_idx == i
        # in build_workload, so this stays consistent.
        ib._req_ids.append(req.req_id)  # type: ignore[attr-defined]
        ib.req_output_token_ids.append([])
        ib.req_id_to_index[req.req_id] = idx

    # Commit block table CPU->GPU so the attention kernels can read it.
    ib.block_table.commit_block_table(len(workload))


def maybe_grow_block_tables(
    model_runner: Any, workload: list[FakeRequest]
) -> bool:
    """Append a new block row whenever a request crossed a block boundary.

    Returns True if any append happened (caller should ``commit_block_table``).
    """
    ib = model_runner.input_batch
    grew = False
    for req in workload:
        for gid, bt in enumerate(ib.block_table.block_tables):
            have = int(bt.num_blocks_per_row[req.req_idx])
            need = (req.seq_len + bt.block_size - 1) // bt.block_size
            if need > have:
                extra = req.block_ids[gid][have:need]
                bt.append_row(extra, req.req_idx)
                grew = True
    return grew


# ---------------------------------------------------------------------------
# Cheap sanity checks
# ---------------------------------------------------------------------------
def validate_state(model_runner: Any, workload: list[FakeRequest]) -> None:
    """Read back GPU state and assert it matches what we asked for.

    Called once after the first step when ``--validate`` is set.
    """
    ib = model_runner.input_batch
    B = len(workload)

    # seq_lens: post-step the workload was already advanced by +1, so we
    # check against (prefill_len + age_steps) which equals seq_len of the
    # *just-executed* step (=prev seq_len, since current age was bumped).
    # We just confirm the GPU buffer is non-zero per-request and matches
    # the optimistic CPU buffer (which we wrote ourselves).
    cpu = model_runner.optimistic_seq_lens_cpu[:B]
    gpu = model_runner.seq_lens[:B].cpu()
    if not torch.equal(cpu, gpu):
        raise AssertionError(
            f"seq_lens cpu/gpu disagree: cpu={cpu.tolist()} gpu={gpu.tolist()}"
        )
    if (cpu <= 0).any():
        raise AssertionError(f"seq_lens has non-positive entries: {cpu.tolist()}")

    # block table row 0 of each request: must equal the first block we assigned.
    bt0 = ib.block_table[0]
    bt_gpu = bt0.get_device_tensor(B).cpu()
    for r in workload:
        first_block = r.block_ids[0][0]
        if int(bt_gpu[r.req_idx, 0]) != first_block:
            raise AssertionError(
                f"block_table[{r.req_idx}, 0]={int(bt_gpu[r.req_idx, 0])} "
                f"but workload wanted {first_block}"
            )

    # slot_mapping must land in *this request's* block range.
    sm = bt0.slot_mapping.gpu[:B].cpu().numpy()
    block_size = bt0.block_size
    for i, r in enumerate(workload):
        slot = int(sm[i])
        block = slot // block_size
        if block not in r.block_ids[0]:
            raise AssertionError(
                f"req {r.req_id} (idx {i}): slot {slot} -> block {block}, "
                f"not in assigned blocks {r.block_ids[0]}"
            )


def ib_pinned_or_gpu_seq_lens(model_runner: Any) -> torch.Tensor:
    """Return whichever copy of seq_lens is authoritative this step.

    `_build_attention_metadata` reads either ``self.seq_lens`` (GPU) or
    ``self.optimistic_seq_lens_cpu`` (pinned) depending on mode; both are
    written by `RealisticDecodeDriver._program_per_iter_buffers`. Either
    one is fine for the post-step validation read-back.
    """
    return model_runner.seq_lens
