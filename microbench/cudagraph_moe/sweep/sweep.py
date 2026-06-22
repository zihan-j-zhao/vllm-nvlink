#!/usr/bin/env python3
"""Sweep orchestrator: launches one nsys+torchrun job per grid cell.

Layout for each cell ``<tag> = b{B}_p{P}_{bg}``:
    <sweep_dir>/
        nsys/<tag>.nsys-rep
        nsys/<tag>.sqlite
        json/<tag>.json              # latency stats from the bench
        iter_csv/<tag>.iter.csv      # per-iter HW distribution
        kernel_csv/<tag>.kernels.csv # kernel duration stats
        logs/<tag>/rank{0..3}.log
        sweep_results.csv            # one summary row per cell

Resumable: cells whose .json output already exists are skipped. Failed
cells are recorded in ``failed.csv`` so the sweep keeps going.

Important invariants (all sanity-checked before launching):
  - max-num-seqs == batch_size  (avoids implicit padding)
  - cudagraph-capture-sizes contains every batch_size in the grid
  - KV-pool budget can hold batch_size * ceil((prefill+max_dec)/16) blocks
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]  # vllm-nvlink/
BENCH_DIR = Path(__file__).resolve().parent     # .../sweep/


# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------
DEFAULT_BATCH_SIZES = [128, 512, 1024]
DEFAULT_PREFILL_LENS = [128, 512, 1024, 2048]
DEFAULT_BG_LIST = ["off"]


# Background-traffic profiles. Each name maps to the bg flags appended
# to the bench CLI. Add new shapes here, then pass --bg-profiles
# <name>,<name>,... on the sweep command line.
#
# Empty dict => no bg traffic (passes --bg-pattern off implicitly).
BG_PROFILES: dict[str, dict[str, str]] = {
    "off": {},
    # Saturate the link: large chunks, deep buffer, ask for more than
    # the link can deliver so the achieved rate is bounded by hardware.
    "ingress_max": {
        "direction": "ingress",
        "chunk_mb": "64", "buffer_mb": "128", "rate_gbps": "9999",
    },
    "egress_max": {
        "direction": "egress",
        "chunk_mb": "64", "buffer_mb": "128", "rate_gbps": "9999",
    },
    "both_max": {
        "direction": "both",
        "chunk_mb": "64", "buffer_mb": "128", "rate_gbps": "9999",
    },
    # Hotspot topology for EP synchronization stress: all phantom traffic
    # GPUs 4..7 send ingress traffic into one decoder GPU (default rank 0).
    # Only supported with --external-bg and the C++ bg runner.
    "ingress_hotspot_max": {
        "direction": "ingress",
        "chunk_mb": "64", "buffer_mb": "128", "rate_gbps": "9999",
        "hotspot_rank": "0",
    },
    # NIXL-page-shaped traffic: each chunk roughly the size of a KV
    # page being pushed (256 KiB per layer * a handful of layers ~= a
    # few MiB). Use a smaller buffer so back-to-back chunks have time
    # to drain, ensuring per-chunk pacing dominates rather than
    # buffer-deep continuous saturation. 50 GB/s is the target rate;
    # the bg thread will sleep between chunks to honour it.
    "ingress_med": {
        "direction": "ingress",
        "chunk_mb": "4", "buffer_mb": "64", "rate_gbps": "50",
    },
    "egress_med": {
        "direction": "egress",
        "chunk_mb": "4", "buffer_mb": "64", "rate_gbps": "50",
    },
    # Apples-to-apples profiles for --external-bg validation: the
    # GIL-throttled in-proc 'ingress_max' achieved ~30 Gbps in practice.
    # Rate-limit external bg to that level for fair comparison.
    "ingress_30g": {
        "direction": "ingress",
        "chunk_mb": "4", "buffer_mb": "64", "rate_gbps": "30",
    },
    "egress_30g": {
        "direction": "egress",
        "chunk_mb": "4", "buffer_mb": "64", "rate_gbps": "30",
    },
}


def _parse_pct_token(raw: str) -> tuple[float, str]:
    """Parse one percentage token and return (value, normalized_tag_piece)."""
    token = raw.strip()
    if not token:
        raise ValueError("empty percentage token")
    pct = float(token)
    if pct <= 0 or pct >= 100:
        raise ValueError(f"percentage must be in (0, 100), got {token!r}")
    if pct.is_integer():
        norm = str(int(pct))
    else:
        # Keep file/tag names path-safe and deterministic.
        norm = f"{pct:g}".replace(".", "p")
    return pct, norm


def _extract_ingress_max_from_json(path: Path) -> float:
    data = json.loads(path.read_text())
    ranks = data.get("ranks", []) or []
    vals: list[float] = []
    for rank in ranks:
        for entry in rank.get("bg_traffic", []) or []:
            if entry.get("direction") != "ingress":
                continue
            if "achieved_gbps" in entry:
                vals.append(float(entry["achieved_gbps"]))
    if not vals:
        raise ValueError(
            f"{path} has no ingress achieved_gbps in ranks[].bg_traffic[]"
        )
    return sum(vals) / len(vals)


def _extract_ingress_max_from_csv(path: Path, tag: str | None) -> float:
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path} has no rows")

    # Prefer an explicit tag match if provided.
    cands = rows
    if tag:
        cands = [r for r in rows if r.get("tag") == tag]
        if not cands:
            raise ValueError(f"{path} has no row with tag={tag!r}")

    # Then prefer bg=ingress_max rows (same convention as existing sweeps).
    ingress_rows = [r for r in cands if r.get("bg") == "ingress_max"]
    if ingress_rows:
        cands = ingress_rows

    vals: list[float] = []
    for r in cands:
        v = r.get("bg_ingress_gbps_mean")
        if v not in (None, ""):
            vals.append(float(v))
    if not vals:
        raise ValueError(
            f"{path} has no bg_ingress_gbps_mean in candidate rows"
        )
    return sum(vals) / len(vals)


def resolve_ingress_max_gbps(args: argparse.Namespace) -> float:
    """Resolve the max-ingress reference throughput used for pct targets."""
    if args.bg_ingress_max_gbps is not None:
        if args.bg_ingress_max_gbps <= 0:
            raise ValueError("--bg-ingress-max-gbps must be > 0")
        return float(args.bg_ingress_max_gbps)

    if args.bg_ingress_max_source is not None:
        src = args.bg_ingress_max_source
        if not src.exists():
            raise FileNotFoundError(f"--bg-ingress-max-source not found: {src}")
        if src.suffix.lower() == ".json":
            return _extract_ingress_max_from_json(src)
        if src.suffix.lower() == ".csv":
            return _extract_ingress_max_from_csv(src, args.bg_ingress_max_tag)
        raise ValueError(
            "--bg-ingress-max-source must point to .json or .csv"
        )

    raise ValueError(
        "Need ingress max reference for --bg-ingress-pcts. "
        "Pass either --bg-ingress-max-gbps or --bg-ingress-max-source."
    )


def build_dynamic_bg_profiles(
    base_profiles: dict[str, dict[str, str]],
    *,
    ingress_pcts_raw: str | None,
    ingress_max_gbps: float | None,
) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Return (profile_map, generated_bg_names_to_append_to_grid)."""
    profiles = dict(base_profiles)
    generated: list[str] = []

    if not ingress_pcts_raw:
        return profiles, generated
    if ingress_max_gbps is None:
        raise ValueError("internal error: ingress_max_gbps missing")

    for tok in (t for t in ingress_pcts_raw.split(",") if t.strip()):
        pct, norm = _parse_pct_token(tok)
        name = f"ingress_pct{norm}"
        rate = ingress_max_gbps * (pct / 100.0)
        profiles[name] = {
            "direction": "ingress",
            "chunk_mb": "64",
            "buffer_mb": "128",
            "rate_gbps": f"{rate:.3f}",
        }
        generated.append(name)
    return profiles, generated


@dataclass
class Cell:
    batch_size: int
    prefill_len: int
    bg: str  # profile name, must be a key in BG_PROFILES

    @property
    def tag(self) -> str:
        return f"b{self.batch_size}_p{self.prefill_len}_{self.bg}"

    @property
    def max_model_len(self) -> int:
        # Comfortably > prefill + max_decode_steps so block table rows fit.
        return max(8192, 2 * self.prefill_len + 1024)

    @property
    def kv_blocks_needed(self) -> int:
        # block_size=16 assumed (Qwen3-30B default); max_decode_steps=100.
        per_req = math.ceil((self.prefill_len + 100) / 16)
        return self.batch_size * per_req


def build_grid(
    batch_sizes: list[int], prefill_lens: list[int], bgs: list[str]
) -> list[Cell]:
    """Stable ordering: outer = max_model_len, inner = batch, inner = bg.

    Sorting by max_model_len groups cells that have the same KV-pool
    sizing together, which gives us roughly even per-cell setup time.
    """
    cells = [
        Cell(b, p, bg)
        for p in prefill_lens
        for b in batch_sizes
        for bg in bgs
    ]
    cells.sort(key=lambda c: (c.max_model_len, c.prefill_len, c.batch_size, c.bg))
    return cells


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
def build_cmd(
    cell: Cell,
    args: argparse.Namespace,
    profile_map: dict[str, dict[str, str]],
    nsys_out_no_suffix: Path,
    json_out: Path,
    log_dir: Path,
    all_capture_sizes: list[int],
) -> list[str]:
    env_cmd = [
        "env",
        "VLLM_PROFILE_KIND=cuda",
        "VLLM_WORKER_MULTIPROC_METHOD=spawn",
    ]
    # With --external-bg we want bg-runner subprocesses traced too, so
    # capture metrics on all 8 GPUs (decoders 0-3 + phantoms 4-7).
    metrics_devices = "all" if args.external_bg else "0,1,2,3"
    nsys_cmd = [
        "nsys", "profile",
            f"--gpu-metrics-devices={metrics_devices}",
            f"--gpu-metrics-frequency={args.gpu_metrics_frequency}",
            "--trace-fork-before-exec=true",
            "--cuda-graph-trace=node",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--force-overwrite=true",
            "-o", str(nsys_out_no_suffix),
    ]
    bg_flags: list[str] = []
    profile = profile_map.get(cell.bg)
    if profile is None:
        raise ValueError(
            f"unknown bg profile {cell.bg!r}; available: {sorted(profile_map)}"
        )
    if profile:  # 'off' has empty dict
        bg_flags = [
            "--bg-pattern", "constant",
            "--bg-direction", profile["direction"],
            "--bg-chunk-mb", profile["chunk_mb"],
            "--bg-buffer-mb", profile["buffer_mb"],
            "--bg-rate-gbps", profile["rate_gbps"],
        ]
        if "hotspot_rank" in profile:
            bg_flags.extend(["--bg-hotspot-rank", profile["hotspot_rank"]])
        if args.external_bg:
            bg_flags.append("--bg-external")
    run_cmd = [
        "bash", "run.sh",
            "--tp", str(args.tp),
            "--dp", str(args.dp),
            *(["--no-enable-expert-parallel"] if args.no_ep else []),
            "--batch-size", str(cell.batch_size),
            "--max-num-seqs", str(cell.batch_size),
            "--prefill-dist", f"fixed:{cell.prefill_len}",
            "--age-dist", "fixed:0",
            "--max-decode-steps", str(args.max_decode_steps),
            "--max-model-len", str(cell.max_model_len),
            "--cudagraph-capture-sizes",
                ",".join(str(s) for s in all_capture_sizes),
            "--warmup-iters", str(args.warmup_iters),
            "--iters", str(args.iters),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--output-json", str(json_out),
            "--log-dir", str(log_dir),
            *bg_flags,
    ]
    if args.no_nsys:
        return env_cmd + run_cmd
    return ["sudo", "-E", *env_cmd, f"PATH={os.environ['PATH']}",
            f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}",
            *nsys_cmd, *run_cmd]


# ---------------------------------------------------------------------------
# Per-cell post-processing
# ---------------------------------------------------------------------------
def export_sqlite(nsys_rep: Path) -> Path:
    sqlite_path = nsys_rep.with_suffix(".sqlite")
    subprocess.run(
        ["nsys", "export", "-t", "sqlite",
         "--force-overwrite=true",
         "-o", str(sqlite_path),
         str(nsys_rep)],
        check=True,
    )
    return sqlite_path


def run_extractor(sqlite_path: Path, iter_csv: Path, kernel_csv: Path) -> None:
    subprocess.run(
        [sys.executable, "-m", "sweep.extract_per_iter",
         str(sqlite_path),
         "--iter-out", str(iter_csv),
         "--kernel-out", str(kernel_csv)],
        check=True,
        cwd=BENCH_DIR.parent,  # so `-m sweep.extract_per_iter` resolves
    )


# ---------------------------------------------------------------------------
# Summary row
# ---------------------------------------------------------------------------
def summarize_cell(
    cell: Cell, json_out: Path, iter_csv: Path, kernel_csv: Path,
    nsys_rep: Path, elapsed_s: float,
) -> dict[str, Any]:
    """Read the cell's outputs into a flat dict for sweep_results.csv."""
    data = json.loads(json_out.read_text())
    # Average p50/p90 across ranks (TP=4 -> 4 ranks all see the same step).
    per_rank = [r for r in data["ranks"] if "p50_ms" in r]
    if not per_rank:
        raise RuntimeError(f"no rank stats in {json_out}")
    p50 = sum(r["p50_ms"] for r in per_rank) / len(per_rank)
    p90 = sum(r["p90_ms"] for r in per_rank) / len(per_rank)
    p99 = sum(r["p99_ms"] for r in per_rank) / len(per_rank)

    # Background-traffic stats: per-rank list of {direction, achieved_gbps,
    # ...}. Average achieved_gbps across ranks for the headline; one column
    # per direction since 'both' produces two entries per rank.
    bg_stats: dict[str, float] = {}
    bg_by_dir: dict[str, list[float]] = {}
    bg_total_by_dir: dict[str, list[float]] = {}
    for r in per_rank:
        for entry in r.get("bg_traffic", []) or []:
            d = entry.get("direction", "?")
            bg_by_dir.setdefault(d, []).append(float(entry["achieved_gbps"]))
            if "achieved_total_gbps" in entry:
                bg_total_by_dir.setdefault(d, []).append(
                    float(entry["achieved_total_gbps"])
                )
    for d, vals in bg_by_dir.items():
        bg_stats[f"bg_{d}_gbps_mean"] = round(sum(vals) / len(vals), 2)
        bg_stats[f"bg_{d}_gbps_max"]  = round(max(vals), 2)
    for d, vals in bg_total_by_dir.items():
        bg_stats[f"bg_{d}_gbps_total_mean"] = round(sum(vals) / len(vals), 2)
        bg_stats[f"bg_{d}_gbps_total_max"] = round(max(vals), 2)

    # Per-iter HW: take the median across all iters of mean-of-iter for
    # each metric. Two-level reduction so a single outlier iter doesn't
    # swing the headline number.
    metric_med: dict[str, float] = {}
    metric_max: dict[str, float] = {}
    if iter_csv.exists() and iter_csv.stat().st_size > 0:
        with iter_csv.open() as fh:
            rdr = csv.DictReader(fh)
            by_metric: dict[str, list[float]] = {}
            by_metric_max: dict[str, list[float]] = {}
            for row in rdr:
                m = row["metric"]
                by_metric.setdefault(m, []).append(float(row["mean"]))
                by_metric_max.setdefault(m, []).append(float(row["max"]))
            for m, vals in by_metric.items():
                metric_med[m] = sorted(vals)[len(vals) // 2]
            for m, vals in by_metric_max.items():
                metric_max[m] = max(vals)

    # Kernel medians (us): take median of p50_us across ranks (one
    # extract per .sqlite, which already aggregates per-rank kernels).
    kernels: dict[str, float] = {}
    kernel_counts: dict[str, int] = {}
    if kernel_csv.exists() and kernel_csv.stat().st_size > 0:
        with kernel_csv.open() as fh:
            for row in csv.DictReader(fh):
                kernels[f"{row['kernel_label']}_p50_us"] = float(row["p50_us"])
                kernels[f"{row['kernel_label']}_mean_us"] = float(row["mean_us"])
                kernel_counts[f"{row['kernel_label']}_count"] = int(row["count"])

    return {
        "tag": cell.tag,
        "batch_size": cell.batch_size,
        "prefill_len": cell.prefill_len,
        "bg": cell.bg,
        "max_model_len": cell.max_model_len,
        "p50_ms": round(p50, 4),
        "p90_ms": round(p90, 4),
        "p99_ms": round(p99, 4),
        "elapsed_s": round(elapsed_s, 1),
        "nsys_rep": str(nsys_rep),
        **{f"{k}_med": round(v, 2) for k, v in metric_med.items()},
        **{f"{k}_max": round(v, 2) for k, v in metric_max.items()},
        **{k: round(v, 3) for k, v in kernels.items()},
        **kernel_counts,
        **bg_stats,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sweep-dir", type=Path,
        default=Path("sweep_runs") / datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Output directory (default: sweep_runs/<timestamp>).",
    )
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--dp", type=int, default=1,
                   help="Data-parallel size. World size = tp*dp; sets NPROC.")
    p.add_argument("--no-ep", action="store_true",
                   help="Disable expert-parallel (default: EP on).")
    p.add_argument("--external-bg", action="store_true",
                   help="Run bg traffic as separate processes (avoids "
                        "GIL/host-jitter contention with the worker). "
                        "Tradeoff: bg memcpys won't appear in the nsys "
                        "report since they're outside the profiled proc.")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup-iters", type=int, default=10)
    p.add_argument("--max-decode-steps", type=int, default=100)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--gpu-metrics-frequency", type=int, default=1000)
    p.add_argument(
        "--batch-sizes", type=str,
        default=",".join(str(b) for b in DEFAULT_BATCH_SIZES),
        help="Comma-separated batch sizes to sweep.",
    )
    p.add_argument(
        "--prefill-lens", type=str,
        default=",".join(str(p_) for p_ in DEFAULT_PREFILL_LENS),
        help="Comma-separated prefill lengths to sweep.",
    )
    p.add_argument(
        "--bgs", type=str, default="off",
        help="Comma-separated bg-profile names (keys of BG_PROFILES). "
             "Built-in: off, ingress_max, egress_max, both_max, "
             "ingress_hotspot_max, ingress_med, egress_med.",
    )
    p.add_argument(
        "--bg-ingress-pcts", type=str, default=None,
        help="Optional comma-separated ingress-load targets as percentages "
             "of a max-ingress reference. Example: '20,40' generates "
             "profiles ingress_pct20 and ingress_pct40 and appends them "
             "to the sweep grid.",
    )
    p.add_argument(
        "--bg-ingress-max-gbps", type=float, default=None,
        help="Explicit max-ingress reference throughput (GB/s) used by "
             "--bg-ingress-pcts.",
    )
    p.add_argument(
        "--bg-ingress-max-source", type=Path, default=None,
        help="Path to prior ingress-max result (.json from one cell or "
             "sweep_results.csv) used to calibrate --bg-ingress-pcts.",
    )
    p.add_argument(
        "--bg-ingress-max-tag", type=str, default=None,
        help="Optional tag filter when --bg-ingress-max-source points to "
             "a sweep_results.csv (for example b256_p2048_ingress_max).",
    )
    p.add_argument(
        "--only", type=str, default=None,
        help="If set, only run cell tags matching this substring (e.g. 'b128_p128').",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print planned cells and commands but don't execute.",
    )
    p.add_argument(
        "--no-nsys", action="store_true",
        help="Run cells directly without Nsight Systems profiling/export. "
             "Only JSON latency/background stats and sweep_results.csv are written.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    prefill_lens = [int(x) for x in args.prefill_lens.split(",") if x.strip()]
    bgs = [x.strip() for x in args.bgs.split(",") if x.strip()]
    ingress_max_gbps: float | None = None
    if args.bg_ingress_pcts:
        ingress_max_gbps = resolve_ingress_max_gbps(args)
    profile_map, generated_bgs = build_dynamic_bg_profiles(
        BG_PROFILES,
        ingress_pcts_raw=args.bg_ingress_pcts,
        ingress_max_gbps=ingress_max_gbps,
    )
    if generated_bgs:
        existing = set(bgs)
        bgs.extend(bg for bg in generated_bgs if bg not in existing)

    unknown = [bg for bg in bgs if bg not in profile_map]
    if unknown:
        raise ValueError(
            f"Unknown --bgs entries: {unknown}; available: {sorted(profile_map)}"
        )

    cells = build_grid(batch_sizes, prefill_lens, bgs)
    if args.only:
        cells = [c for c in cells if args.only in c.tag]

    # Always include every swept batch size so vLLM captures graphs for
    # it. Plus a few small ones for warmup compatibility.
    all_capture_sizes = sorted(
        set([1, 4, 16, 64] + batch_sizes + [max(batch_sizes)])
    )

    # All paths must be absolute: subprocess cwd is BENCH_DIR (so `-m
    # sweep.main` resolves), and run.sh internally cds to the package
    # parent, so relative paths would resolve to the wrong place.
    sweep_dir = args.sweep_dir.resolve()
    nsys_dir = sweep_dir / "nsys"
    json_dir = sweep_dir / "json"
    iter_csv_dir = sweep_dir / "iter_csv"
    kernel_csv_dir = sweep_dir / "kernel_csv"
    logs_dir = sweep_dir / "logs"
    for d in (nsys_dir, json_dir, iter_csv_dir, kernel_csv_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    summary_csv = sweep_dir / "sweep_results.csv"
    failed_csv = sweep_dir / "failed.csv"

    print(f"[sweep] dir: {sweep_dir}")
    print(f"[sweep] cells: {len(cells)}")
    print(f"[sweep] capture sizes: {all_capture_sizes}")
    if generated_bgs:
        print("[sweep] generated ingress load profiles:")
        for name in generated_bgs:
            p = profile_map[name]
            print(f"  - {name}: direction={p['direction']} "
                  f"chunk={p['chunk_mb']}MiB "
                  f"buffer={p['buffer_mb']}MiB "
                  f"rate={p['rate_gbps']} GB/s")
        if ingress_max_gbps is not None:
            print(f"  (max-ingress reference: {ingress_max_gbps:.3f} GB/s)")
    print(f"[sweep] tp={args.tp}, dp={args.dp}, "
          f"ep={'off' if args.no_ep else 'on'}, iters={args.iters}, "
          f"freq={args.gpu_metrics_frequency} Hz, "
          f"nsys={'off' if args.no_nsys else 'on'}")
    print()

    summary_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []

    for i, cell in enumerate(cells, 1):
        # Resumability: nsys runs need json + iter csv; raw runs only
        # produce json and summarize directly from it.
        json_out = json_dir / f"{cell.tag}.json"
        iter_csv = iter_csv_dir / f"{cell.tag}.iter.csv"
        kernel_csv = kernel_csv_dir / f"{cell.tag}.kernels.csv"
        done = json_out.exists() if args.no_nsys else (
            json_out.exists() and iter_csv.exists()
        )
        if done:
            print(f"[{i}/{len(cells)}] {cell.tag}: SKIP (already done)")
            try:
                nsys_rep = nsys_dir / f"{cell.tag}.nsys-rep"
                row = summarize_cell(cell, json_out, iter_csv, kernel_csv,
                                     nsys_rep, elapsed_s=0.0)
                summary_rows.append(row)
            except Exception as e:
                print(f"  (failed to re-summarize: {e})")
            continue

        nsys_rep = nsys_dir / f"{cell.tag}.nsys-rep"
        log_dir = logs_dir / cell.tag
        cmd = build_cmd(cell, args, profile_map, nsys_rep.with_suffix(""), json_out,
                        log_dir, all_capture_sizes)

        print(f"[{i}/{len(cells)}] {cell.tag}")
        print(f"  KV budget: need ~{cell.kv_blocks_needed} blocks, "
              f"max_model_len={cell.max_model_len}")
        print(f"  cmd: {' '.join(shlex.quote(c) for c in cmd)}")
        if args.dry_run:
            continue

        t0 = time.perf_counter()
        try:
            env = dict(os.environ, NPROC=str(args.tp * args.dp))
            subprocess.run(cmd, cwd=BENCH_DIR, env=env, check=True)
        except subprocess.CalledProcessError as e:
            elapsed = time.perf_counter() - t0
            print(f"  FAIL after {elapsed:.1f}s: exit={e.returncode}")
            failed_rows.append({
                "tag": cell.tag, "exit_code": e.returncode,
                "elapsed_s": round(elapsed, 1),
            })
            continue
        elapsed = time.perf_counter() - t0

        # Export sqlite + run analysis, unless this is a raw no-nsys run.
        try:
            if not args.no_nsys:
                sqlite_path = export_sqlite(nsys_rep)
                run_extractor(sqlite_path, iter_csv, kernel_csv)
            row = summarize_cell(cell, json_out, iter_csv, kernel_csv,
                                  nsys_rep, elapsed)
            summary_rows.append(row)
            print(f"  ok in {elapsed:.1f}s  p50={row['p50_ms']:.2f}ms")
        except Exception as e:
            print(f"  POST-PROCESS FAIL: {e}")
            failed_rows.append({
                "tag": cell.tag, "exit_code": -1,
                "elapsed_s": round(elapsed, 1),
                "error": str(e)[:200],
            })
            continue

        # Persist after each cell so a Ctrl-C doesn't lose results.
        write_csv(summary_csv, summary_rows)
        if failed_rows:
            write_csv(failed_csv, failed_rows)

    print()
    print(f"[sweep] done: {len(summary_rows)} ok, {len(failed_rows)} failed")
    # Always write the summary at the end so a sweep that consists
    # entirely of SKIP cells (or finishes its loop without going through
    # the per-cell post-process branch) still produces sweep_results.csv.
    write_csv(summary_csv, summary_rows)
    if failed_rows:
        write_csv(failed_csv, failed_rows)
    print(f"[sweep] summary: {summary_csv}")
    return 0 if not failed_rows else 1


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    # Build union of keys to handle rows with different metric columns.
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    sys.exit(main())
