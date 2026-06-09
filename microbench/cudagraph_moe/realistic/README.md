# Realistic CUDA-graph decode microbenchmark

Sibling of [base/main.py](../base/main.py). Same model bootstrap, same
captured CUDA graphs — but each timed step decodes a *batch of synthetic
requests* with realistic KV-cache state instead of the all-`-1` placeholder
batch that `_dummy_run` produces.

## Why this exists

`base/main.py` calls `GPUModelRunner._dummy_run(num_tokens=B, uniform_decode=True)`
in its tight loop. That is enough to time the dense GEMMs + MoE all-to-all,
but it is **not** a faithful decode forward:

1. `_dummy_run` fills every `slot_mapping` entry with `-1`, so the captured
   `reshape_and_cache` kernel becomes a no-op. The decode forward never
   writes K/V into the pool, so HBM write traffic and L2 dirty-line churn
   are invisible.
2. `_dummy_run` sets `seq_lens` to a single scalar (`max_query_len == 1`).
   Every "request" looks like context-length-1 and the block-table rows in
   `self.input_batch.block_table` are zeros, so attention reads block 0 for
   every request. Paged-attention page walks, page-table TLB pressure, and
   per-request work imbalance are all invisible.

The realistic harness fixes both by **programming the persistent
CUDA-graph buffers in-place before each replay**. The captured graphs read
`seq_lens`, `block_table`, `slot_mapping`, `query_start_loc`, `positions`,
and `input_ids` at fixed addresses; overwriting those buffers with real
per-request values is what makes the bench realistic *without recapturing
graphs*.

## Quick start

```bash
# 2-way DP, defaults: B=16, prefill∈Uniform(128..2048), age∈Uniform(0..1024).
bash run.sh

# Larger / heterogeneous workload
bash run.sh --batch-size 64 --prefill-dist uniform:512:8192 --iters 100

# Replay a fixed prefill length per request
bash run.sh --batch-size 16 --prefill-dist fixed:4096 --age-dist fixed:0

# Replay a recorded ShareGPT distribution (JSON list of ints)
bash run.sh --prefill-dist file:/path/to/sharegpt_lens.json

# Parallelism sweep — main.py auto-derives --dp from WORLD_SIZE / (tp*pp).
NPROC=4 bash run.sh --tp 4                   # pure TP, lights up NVLink
NPROC=4 bash run.sh --tp 2 --dp 2            # mixed TP+DP, production-like
NPROC=4 bash run.sh --tp 4 --no-enable-expert-parallel  # TP-only, no EP
```

Results land in `realistic/results/cudagraph_decode_realistic.json`.

### CLI knobs

| Flag                     | What it controls                                                |
|--------------------------|-----------------------------------------------------------------|
| `--batch-size`           | Concurrent decode requests per DP rank                          |
| `--prefill-dist`         | Per-req prefill length (`fixed:N`, `uniform:lo:hi`, `geometric:mean`, `file:path.json`) |
| `--age-dist`             | Per-req starting decode-age — controls how varied seq_lens are  |
| `--max-decode-steps`     | Lifetime cap; sets how many KV blocks we pre-allocate per req   |
| `--warmup-iters`/`--iters` | Standard timing knobs                                         |
| `--tp` / `--pp` / `--dp` | Parallelism dims. `--dp` auto-derives from `WORLD_SIZE/(tp*pp)`. Asserted: `tp*pp*dp == WORLD_SIZE` |
| `--no-enable-expert-parallel` | Disable EP inside the MoE                                 |
| `--all2all-backend`      | EP dispatch backend (`allgather_reducescatter`, `deepep_high_throughput`, …) |
| `--validate`             | After first step, read back seq_lens / block_table / slot_mapping and assert they match |
| Everything else          | Forwarded to `LLM(...)` (model, max-num-seqs, attn/MoE backend …) — same as `base/main.py` |

### Parallelism cheat-sheet

Decode is HBM-bound; NVLink utilization is dominated by what crosses the link
*every* layer. Use this to pick a config:

| Config                            | What it stresses                            |
|-----------------------------------|---------------------------------------------|
| `--tp 1 --dp N` (default)         | Per-rank HBM only; NVLink near-idle         |
| `--tp N --dp 1`                   | NVLink all-reduces every transformer layer  |
| `--tp 2 --dp N/2`                 | Production-like serving (TP within node, DP across) |
| `--tp N --no-enable-expert-parallel` | Isolate TP traffic from EP all-to-all     |
| `--all2all-backend deepep_high_throughput` | Bigger EP packets — only matters when EP is on |

## Code path (one step)

```
main.py                         build_workload + populate_input_batch (once)
  └─ driver.RealisticDecodeDriver.bench(iters)
       └─ for i in iters:
            step()
              ├─ synchronize_input_prep()         # reuse vLLM event protocol
              ├─ _program_per_iter_buffers()      # writes the per-iter state
              │     ├─ seq_lens          (varies per req)
              │     ├─ positions         (varies per req)
              │     ├─ query_start_loc   ([0..B] for uniform decode)
              │     ├─ block_table.commit (if a req crossed a block boundary)
              │     ├─ block_table.compute_slot_mapping (Triton kernel)
              │     ├─ discard_request_mask = False
              │     └─ num_accepted_tokens = 1
              ├─ runner._determine_batch_execution_and_padding()  # picks captured graph
              ├─ runner._get_slot_mappings(...)                    # per-layer view of slot_mapping
              ├─ runner._build_attention_metadata(...)             # builds AttentionMetadata
              └─ set_forward_context(...): runner.model(...)       # ★ graph replay ★
            # post-step: advance r.age_steps; recycle when reaching max_decode_steps
```

What `step()` deliberately **omits** vs. the real `execute_model`:

| Real `execute_model` phase     | In `step()`? | Why |
|--------------------------------|--------------|-----|
| `_update_states`               | No (replaced by one-shot `populate_input_batch`) | Synthetic scheduler writes state directly |
| `_prepare_inputs`              | Replaced by `_program_per_iter_buffers`           | We only need the buffer writes, not token/embed gathering |
| attn metadata + slot mapping   | Yes (reuses `_build_attention_metadata` / `_get_slot_mappings`) | This is the only "real" thing in the forward path |
| `set_forward_context` + model  | Yes                                              | The thing we want to time |
| `compute_logits` / sampler     | No                                               | Not part of "decode forward"; adds CPU sync |
| `_bookkeeping_sync`            | No                                               | Same |
| `propose_draft_token_ids`      | No                                               | Spec decode out of scope (first version) |
| `eplb_step`                    | No                                               | Adds nondeterministic rebalancing |

The omissions are why we can drive arbitrarily long runs without burning
through KV blocks or accumulating sampled-token bookkeeping; the inclusions
are exactly what's needed to keep the captured graphs valid.

## What `populate_input_batch` writes (the "skip `_update_states`" path)

`_update_states` normally consumes a `SchedulerOutput` and writes a much
larger subset of vLLM state (mrope, generators, output_token_ids, mamba,
ngram, …). For a pure decode bench we only need the fields that
`_build_attention_metadata` + `compute_slot_mapping` actually read:

| Field                                                  | Source                | Used by                                |
|--------------------------------------------------------|-----------------------|----------------------------------------|
| `input_batch.req_id_to_index` / `_req_ids`             | one slot per req      | `num_reqs` (property), iterators       |
| `input_batch.num_prompt_tokens[i]`                     | `req.prefill_len`     | `is_prefilling` flag for mamba/GDN     |
| `input_batch.num_computed_tokens_cpu[i]`               | `prefill + age`       | `is_prefilling`, kernel seq_lens math  |
| `input_batch.num_tokens_no_spec[i]`                    | `prefill + age`       | avoids stale sampler bookkeeping       |
| `input_batch.block_table[gid].block_table` (numpy+GPU) | real, disjoint IDs    | every attention kernel                 |

Block IDs are handed out by a sequential `BlockPool` over `[1, num_blocks)`
(block 0 is vLLM's `NULL_BLOCK_ID`) so per-request KV ranges are disjoint
and prefetch / TLB behaviour is realistic. When a request crosses a block
boundary during decoding, `step()` appends the next pre-allocated block
ID to its row and re-commits the block table.

## Future plan: distribution-driven arrivals/completions

The current setup admits the whole workload once and never evicts. The
shape of the synthetic scheduler is already in place to support a
Poisson-arrival / variable-completion model in the future:

```python
class SyntheticScheduler:
    def tick(self) -> tuple[list[FakeRequest], list[FakeRequest]]:
        # 1. arrivals: sample 0+ new requests from arrival_rate
        # 2. completions: any req at age >= max_decode → free slot + blocks
        # 3. update self.active list
        return admitted, evicted

    def commit_to_input_batch(self, admitted, evicted):
        # for r in evicted:  ib.block_table[*].clear_row(r.req_idx)
        # for r in admitted: ib.block_table[*].add_row(...); set num_*_tokens
```

Adding it later is a 60-line drop-in that does **not** require resurrecting
`_update_states`. Watch out for:

- **CUDA-graph batch-size mismatch.** Graphs are captured for a fixed set
  of `num_tokens` values. Quantize active count to the nearest captured
  size; pad with dummy requests whose `slot_mapping = -1` so the dummies
  don't write KV (same trick `_dummy_run` uses with `NULL_BLOCK_ID`).
- **Slot fragmentation.** Without `condense()`, holes form between active
  rows. Either run `condense()` periodically, or only recycle when a slot
  is fully free.
- **Block re-allocation after free.** vLLM normally calls `_zero_block_ids`
  to flush in-flight `reshape_and_cache` writes before reusing freed
  blocks. For a bench, over-allocate (`num_blocks >> sum(max_blocks_per_req)`)
  to avoid the issue entirely.

## Tests

Pure-python self-tests (no CUDA / no vLLM needed):

```bash
cd microbench/cudagraph_moe
python -m unittest realistic.test_fake_scheduler -v
```

These cover distribution parsing (`fixed` / `uniform` / `geometric` /
`file`), `BlockPool` accounting (disjoint IDs, exhaustion, multi-group),
`build_workload` (per-req block counts, `seq_len` / `position` math,
cross-request block disjointness), and the per-row block-table cap.

## Troubleshooting

**`ValueError: ... would need N blocks ... only holds M blocks`** — your
chosen `--prefill-dist` (+ `--max-decode-steps`) exceeds vLLM's per-row
block-table cap, which is `cdiv(max_model_len, block_size)`. The fix is
to raise `--max-model-len` to at least `prefill + max_decode_steps` (and,
typically, the matching env var for the served model). For example:

```bash
bash run.sh --batch-size 16 --prefill-dist fixed:10000 \
            --max-model-len 16384
```

Run-time validation (cheap, ~one extra D2H per run):

```bash
bash run.sh --validate
```

After the first timed step, this synchronously reads back
`seq_lens` (CPU vs GPU view), `block_table[*, 0]` (matches the first
assigned block), and `slot_mapping[i]` (falls in this request's blocks)
and raises if any disagree with the workload.

## Files

| File                         | Role                                                                 |
|------------------------------|----------------------------------------------------------------------|
| [main.py](main.py)           | CLI + vLLM bootstrap + result aggregation                             |
| [driver.py](driver.py)       | `RealisticDecodeDriver.step()`, `warmup()`, `bench()`                 |
| [fake_scheduler.py](fake_scheduler.py) | `FakeRequest`, `BlockPool`, `build_workload`, `populate_input_batch`, `validate_state` |
| [run.sh](run.sh)             | `torchrun --nproc_per_node=NPROC ... -m realistic.main`              |
| [test_fake_scheduler.py](test_fake_scheduler.py) | Pure-python unit tests                                |
