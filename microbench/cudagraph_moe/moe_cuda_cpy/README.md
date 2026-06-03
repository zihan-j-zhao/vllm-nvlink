# `moe_cuda_cpy/` — decode forward vs. background P2P copy-engine traffic

This bench answers: *how much does a sustained P2P `cudaMemcpyAsync` flow on
the destination GPU's copy engines slow down a vLLM Qwen3-MoE decode forward?*

It builds on `../base/decode_cudagraph_bench.py` (same setup pattern, same
CUDA-graph dispatch), and adds a controlled background noise generator.

## Topology

```
+--------+         MoE (DP=2 + EP)         +--------+
| cuda:0 |  <--- NCCL all-to-all --->      | cuda:1 |
+--------+                                 +--------+
   ^
   |  background cudaMemcpyPeerAsync (~10 GiB/s, copy engines)
   |
+--------+
| cuda:2 |   noise source (not part of vLLM)
+--------+
```

`cuda:0` and `cuda:1` run the MoE forward as DP ranks. `cuda:2` is **not** seen
by vLLM but lives in `CUDA_VISIBLE_DEVICES`. A daemon thread on rank 0 issues
`cudaMemcpyPeerAsync` from `cuda:2 → cuda:0` on a dedicated non-blocking stream,
rate-limited to the target GiB/s.

## Phases

| Phase | What | NVTX range |
| --- | --- | --- |
| Setup | LLM build + CUDA-graph capture | `bench/setup` |
| A — quiet | 10 forwards of `num_tokens=100`, no noise | `bench/phase/quiet` |
| — | start noise thread | mark `bench/p2p/start` |
| B — noisy | 10 forwards of `num_tokens=100`, sustained P2P | `bench/phase/noisy` |
| — | stop noise thread | mark `bench/p2p/stop` |

Per-iteration NVTX ranges are emitted as `bench/phase/{quiet,noisy}/iter_{i}`,
plus start/end marks. Each P2P chunk is wrapped in
`bench/p2p/copy_{src}_to_{dst}` so the copy engine activity is searchable.

## Files

| File | What |
| --- | --- |
| `decode_p2p_bench.py` | torchrun-launched bench driver + `P2PBackgroundTraffic` thread |
| `run.sh`              | Default 2-rank torchrun launcher (`CUDA_VISIBLE_DEVICES=0,1,2`) |
| `run_nsys.sh`         | Wraps `run.sh` under `nsys profile` with NVTX capture |

## Run

```bash
# defaults: 10 GiB/s P2P from cuda:2 -> cuda:0, 10 forwards of 100 tokens
bash run.sh

# heavier noise
bash run.sh --target-gbps 25

# control: phase B with noise disabled (should match phase A)
bash run.sh --no-p2p

# noise into rank 1 instead
bash run.sh --p2p-dst-rank 1 --p2p-src-device 3
CUDA_VISIBLE_DEVICES=0,1,3 bash run.sh --p2p-dst-rank 1 --p2p-src-device 3

# different GPU triple
CUDA_VISIBLE_DEVICES=4,5,6 bash run.sh
```

JSON summary is written to `results/p2p_decode.json` (per-rank, per-phase
latency distribution + measured effective P2P throughput).

## nsys profiling

```bash
bash run_nsys.sh                         # default 10 GiB/s trace
NSYS_OUTPUT=results/trace_25g bash run_nsys.sh --target-gbps 25
bash run_nsys.sh --no-p2p                # baseline trace
```

The wrapper uses `--capture-range=nvtx --nvtx-capture='bench/phase/quiet'`, so
capture starts when phase A begins (skipping the slow CUDA-graph capture).
Open the resulting `.nsys-rep` and search for:

- `bench/phase/quiet` and `bench/phase/noisy` to compare the two phases side
  by side.
- `bench/phase/.../iter_*` to zoom into a single forward.
- `bench/p2p/copy_*` to see the copy-engine bursts and how they interleave
  with NCCL kernels in the MoE all-to-all on `cuda:0`.

## Knobs

| Flag | Default | Notes |
| --- | --- | --- |
| `--num-tokens` | 100 | Uniform decode batch size per rank |
| `--iters` | 10 | Timed forwards per phase |
| `--warmup-iters` | 5 | Untimed forwards before each phase |
| `--target-gbps` | 10.0 | Sustained P2P throughput target (GiB/s) |
| `--p2p-chunk-mb` | 32 | Bytes per `cudaMemcpyPeerAsync` |
| `--p2p-in-flight` | 4 | Max concurrent P2P copies (queue depth) |
| `--p2p-src-device` | 2 | Source GPU (CUDA index inside `CUDA_VISIBLE_DEVICES`) |
| `--p2p-dst-rank` | 0 | Which DP rank gets the noise |
| `--p2p-prelude-s` | 0.5 | Settle time after starting noise, before phase B |
| `--no-p2p` | off | Disable the noise thread (control) |

`--target-gbps` is *sustained average* throughput. The bench prints the actual
effective rate from observed launches; verify it lands near the target.

## Caveats

- The noise GPU **must** be in `CUDA_VISIBLE_DEVICES`. vLLM picks devices via
  `LOCAL_RANK`, so extra visible devices don't disturb model placement.
- `cudaMemcpyPeerAsync` requires P2P access between dst and src. On NVLinked
  hardware (e.g., B200 + NVSwitch) it's automatic; on PCIe-only systems the
  effective throughput cap may be lower than the target.
- The noise is *asymmetric* by default (rank 0 only). Phase B latency on
  rank 1 may also rise because EP all-to-all is gated on the slowest rank.
- Per-iter latency is measured with `cuda.Event` pairs — pure GPU time on the
  compute stream, no host overhead in the timing loop itself.
