# `moe_cuda_cpy_sweep/` — P2P bandwidth sweep over the decode forward

Variant of `../moe_cuda_cpy/` that initializes the vLLM model **once** and then
sweeps the background P2P throughput over a list of targets, recording the
decode latency distribution at each level.

## What it runs

1. `bench/setup/build_llm` — one-time vLLM model load + CUDA-graph capture.
2. `bench/phase/quiet` — 200 timed forwards of `num_tokens=100`, no noise.
3. For each `g` in `--sweep-gbps` (default `10,20,40,80,160,320,640` GiB/s):
   - start the P2P noise thread targeting `g` GiB/s,
   - settle for `--p2p-prelude-s` seconds,
   - run `bench/phase/noisy_{g}gbps` — 200 timed forwards,
   - stop the noise thread, cooldown, repeat.

Each `bench/phase/...` range contains a `warmup` sub-range and a `timed`
sub-range; each iteration is `bench/phase/.../iter_{i}`. P2P chunks on the
noise thread are `bench/p2p/copy_{src}_to_{dst}`, wrapped in
`bench/p2p/worker_{g}gbps` per sweep step.

## Outputs

- `results/p2p_sweep.json` — full per-rank, per-phase payload (incl. the raw
  per-iteration latency list).
- `results/p2p_sweep.csv` — one row per `(rank, phase)` with `target_gbps`,
  `effective_gbps`, `iters`, `mean_ms`, `p50/p90/p99_ms`, `min/max_ms`,
  `wall_total_s`.

`effective_gbps` is the measured throughput the noise thread actually
achieved over the timed phase — verify it tracks the target before reading
too much into the latency numbers, especially at high bandwidths.

## Run

```bash
bash run.sh                                # full default sweep
bash run.sh --sweep-gbps 10,40,160,640     # custom subset
bash run.sh --iters 500 --warmup-iters 20  # more iters per point
bash run.sh --skip-quiet                   # skip the quiet baseline
```

## Knobs

| Flag | Default | Notes |
| --- | --- | --- |
| `--num-tokens` | 100 | Decode batch size per rank |
| `--iters` | 200 | Timed forwards per phase |
| `--warmup-iters` | 10 | Untimed forwards per phase |
| `--sweep-gbps` | `10,20,40,80,160,320,640` | Comma-separated targets |
| `--p2p-src-device` | 2 | Source GPU index (inside `CUDA_VISIBLE_DEVICES`) |
| `--p2p-dst-rank` | 0 | DP rank that receives the noise |
| `--p2p-chunk-mb` | 32 | Bytes per `cudaMemcpyPeerAsync` |
| `--p2p-in-flight` | 4 | Max concurrent P2P copies |
| `--p2p-prelude-s` | 0.5 | Settle time after starting noise |
| `--cooldown-s` | 0.5 | Sleep between phases |

## Notes

- The noise generator is the same `cudaMemcpyPeerAsync`-via-`Tensor.copy_`
  pattern as in `../moe_cuda_cpy/`. At very high targets (≥ 300 GiB/s) the
  thread may not keep up — check `effective_gbps` in the CSV.
- All ranks must enter each phase together (the bench `dist.barrier`s
  between phases), so phase boundaries are clean in nsys traces.
- nsys: the per-phase NVTX names include the target throughput
  (e.g. `bench/phase/noisy_160gbps`), so it's easy to jump straight to a
  particular sweep point.
