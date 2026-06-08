#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Plot ITL CDFs from one or more `vllm bench serve --save-detailed` JSON files.

The benchmark stores per-request inter-token latencies under the top-level
`itls` key as seconds. This script flattens those samples, converts to ms,
and writes a CDF plot for quick tail comparison across runs.

Usage:
    python playground/moe_pd/plot_vllm_bench_itl_cdf.py \
        playground/out/vllm_bench_pd/run_1/pd_bench_kv.json \
        --out playground/out/vllm_bench_pd/run_1/itl_cdf.png

Compare runs:
    python playground/moe_pd/plot_vllm_bench_itl_cdf.py \
        --input no_kv=playground/out/vllm_bench_pd/run_0/pd_bench_no_kv.json \
        --input kv=playground/out/vllm_bench_pd/run_1/pd_bench_kv.json \
        --out playground/out/vllm_bench_pd/itl_cdf_compare.png

Pool repeated runs into one curve per condition:
    python playground/moe_pd/plot_vllm_bench_itl_cdf.py \
        --group 'no_kv=playground/out/vllm_bench_pd/run_*/pd_bench.json' \
        --group 'kv=playground/out/vllm_bench_pd/run_*/pd_bench_kv.json' \
        --out playground/out/vllm_bench_pd/itl_cdf_pooled.png
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - import-time error path
    print(f"error: matplotlib is required ({exc})", file=sys.stderr)
    raise


def _parse_labeled_path(value: str) -> tuple[str | None, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        label = label.strip()
        if label:
            return label, Path(path)
    return None, Path(value)


def _parse_group(value: str) -> tuple[str, list[Path]]:
    if "=" not in value:
        raise ValueError("--group must be LABEL=GLOB")
    label, pattern = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError("--group label must be non-empty")
    paths = [Path(path) for path in sorted(glob.glob(pattern))]
    if not paths:
        raise ValueError(f"--group {label!r} matched no files: {pattern}")
    return label, paths


def _load_itl_ms(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    raw_itls = data.get("itls")
    if raw_itls is None:
        raise ValueError(
            f"{path} has no 'itls' field; rerun vllm bench with "
            "--save-result --save-detailed"
        )

    values: list[float] = []
    for request_itls in raw_itls:
        if not request_itls:
            continue
        values.extend(float(x) * 1000.0 for x in request_itls if x is not None)

    if not values:
        raise ValueError(f"{path} has no ITL samples")
    return np.asarray(values, dtype=np.float64)


def _default_label(path: Path) -> str:
    parent = path.parent.name
    stem = path.stem
    if parent and parent not in {".", ""}:
        return f"{parent}/{stem}"
    return stem


def _cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = np.sort(values)
    ys = np.arange(1, len(xs) + 1, dtype=np.float64) / len(xs)
    return xs, ys


def _summary(values: np.ndarray) -> str:
    return (
        f"n={len(values)} "
        f"p50={np.percentile(values, 50):.3f}ms "
        f"p90={np.percentile(values, 90):.3f}ms "
        f"p99={np.percentile(values, 99):.3f}ms "
        f"max={np.max(values):.3f}ms"
    )


def _load_group_itl_ms(paths: Iterable[Path]) -> np.ndarray:
    arrays = [_load_itl_ms(path) for path in paths]
    if not arrays:
        raise ValueError("group has no input files")
    return np.concatenate(arrays)


def plot_cdfs(
    inputs: Iterable[tuple[str | None, Path]],
    groups: Iterable[tuple[str, list[Path]]],
    out: Path,
    title: str,
    x_max_ms: float | None,
    log_x: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    for label, path in inputs:
        values = _load_itl_ms(path)
        xs, ys = _cdf(values)
        display_label = label or _default_label(path)
        ax.plot(xs, ys, linewidth=1.8, label=f"{display_label} ({_summary(values)})")
        print(f"{display_label}: {_summary(values)}")

    for label, paths in groups:
        values = _load_group_itl_ms(paths)
        xs, ys = _cdf(values)
        ax.plot(xs, ys, linewidth=2.2, label=f"{label} ({_summary(values)})")
        print(f"{label}: {_summary(values)}")
        for path in paths:
            print(f"  included {path}")

    ax.set_title(title)
    ax.set_xlabel("Inter-token latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, which="both", alpha=0.3)
    if x_max_ms is not None:
        ax.set_xlim(left=0.0 if not log_x else None, right=x_max_ms)
    if log_x:
        ax.set_xscale("log")
    ax.legend(loc="lower right", fontsize="small")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    print(f"wrote {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        help="vLLM bench JSON paths, optionally LABEL=PATH.",
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Additional input as PATH or LABEL=PATH. Can be repeated.",
    )
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        metavar="LABEL=GLOB",
        help="Pool all JSONs matching GLOB into one CDF curve. Can be repeated.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("playground/out/vllm_bench_pd/itl_cdf.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--title",
        default="vLLM Bench ITL CDF",
        help="Plot title.",
    )
    parser.add_argument(
        "--x-max-ms",
        type=float,
        default=None,
        help="Optional x-axis upper bound in ms.",
    )
    parser.add_argument(
        "--log-x",
        action="store_true",
        help="Use a log-scaled x axis.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_specs = [*args.paths, *args.input]
    if not input_specs and not args.group:
        print("error: provide at least one vLLM bench JSON or --group", file=sys.stderr)
        return 2

    inputs = [_parse_labeled_path(spec) for spec in input_specs]
    groups = [_parse_group(spec) for spec in args.group]
    plot_cdfs(
        inputs=inputs,
        groups=groups,
        out=args.out,
        title=args.title,
        x_max_ms=args.x_max_ms,
        log_x=args.log_x,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
