#!/usr/bin/env python3
"""Plot stage-1 decode stability JSON output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib missing", file=sys.stderr)
    sys.exit(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    payload = json.loads(args.json_path.read_text())
    cfg = payload["config"]
    out_dir = args.out_dir or args.json_path.parent / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for rank in payload["ranks"]:
        ys = rank["per_iter_ms"]
        ax.plot(range(len(ys)), ys, marker="o", markersize=3, label=f"rank {rank['rank']}")
    ax.set_xlabel("timed forward iteration")
    ax.set_ylabel("forward latency [ms]")
    ax.set_title(
        f"Decode stability: B={cfg['batch_size']}, seq_len={cfg['seq_len']}, "
        f"tp={cfg['tp']}, dp={cfg['dp']}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    latency_path = out_dir / "latency_by_iter.png"
    fig.savefig(latency_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for rank in payload["ranks"]:
        ys = [cfg["batch_size"] * 1000.0 / x for x in rank["per_iter_ms"]]
        ax.plot(range(len(ys)), ys, marker="o", markersize=3, label=f"rank {rank['rank']}")
    ax.set_xlabel("timed forward iteration")
    ax.set_ylabel("logical decode throughput [tokens/s per DP group]")
    ax.set_title("Throughput derived from per-forward latency")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    throughput_path = out_dir / "throughput_by_iter.png"
    fig.savefig(throughput_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[wrote] {latency_path}")
    print(f"[wrote] {throughput_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

