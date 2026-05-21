#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fold raw MoE a2a JSONL traces into a long-format per-step CSV.

Schema (one row per (rank, step_id, layer_idx)):

    rank, step_id, layer_idx, scheduled_tokens, world_size,
    dispatch_in_tokens,  dispatch_in_bytes,
    dispatch_out_tokens, dispatch_out_bytes,
    combine_in_tokens,   combine_in_bytes,
    combine_out_tokens,  combine_out_bytes

The user-facing interpretations the profiler was designed for are derivable
from this:

* Bytes transferred out during dispatch on this rank (wire bytes, AG, W=2):
      ``dispatch_in_bytes``
* Bytes transferred in during combine on this rank  (wire bytes, RS, W=2):
      ``combine_out_bytes``
* Tokens dispatched out:  ``dispatch_in_tokens``
* Tokens combined in:     ``combine_out_tokens``

Sanity asserts (enabled by default; pass --no-assert to skip):

* Each (rank, step_id) has exactly ``num_layers`` dispatch and combine rows.
* For W=2:  dispatch_out_tokens == 2 * dispatch_in_tokens  and
            combine_in_tokens  == 2 * combine_out_tokens.
* All rows in the same (rank, step_id) carry the same scheduled_tokens.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path


def _read_jsonl(paths: list[str]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        with open(path) as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                records.append(json.loads(ln))
    return records


def _fold(records: list[dict]) -> dict[tuple[int, int, int], dict]:
    """Index by (rank, step_id, layer_idx) and merge dispatch+combine."""
    rows: dict[tuple[int, int, int], dict] = {}
    for r in records:
        key = (r["rank"], r["step_id"], r["layer_idx"])
        if key not in rows:
            rows[key] = {
                "rank": r["rank"],
                "step_id": r["step_id"],
                "layer_idx": r["layer_idx"],
                "scheduled_tokens": r["scheduled_tokens"],
                "world_size": r["world_size"],
                "dispatch_in_tokens": None,
                "dispatch_in_bytes": None,
                "dispatch_out_tokens": None,
                "dispatch_out_bytes": None,
                "combine_in_tokens": None,
                "combine_in_bytes": None,
                "combine_out_tokens": None,
                "combine_out_bytes": None,
            }
        row = rows[key]
        if r["kind"] == "dispatch":
            row["dispatch_in_tokens"] = r["in_tokens"]
            row["dispatch_in_bytes"] = r["in_bytes"]
            row["dispatch_out_tokens"] = r["out_tokens"]
            row["dispatch_out_bytes"] = r["out_bytes"]
        elif r["kind"] == "combine":
            row["combine_in_tokens"] = r["in_tokens"]
            row["combine_in_bytes"] = r["in_bytes"]
            row["combine_out_tokens"] = r["out_tokens"]
            row["combine_out_bytes"] = r["out_bytes"]
        # else: ignore unknown kinds
    return rows


def _assert_invariants(rows: dict[tuple[int, int, int], dict], num_layers: int) -> None:
    # Group by (rank, step_id) for shape checks.
    per_step: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for (rank, step, _), row in rows.items():
        per_step[(rank, step)].append(row)

    problems: list[str] = []
    for (rank, step), step_rows in per_step.items():
        if len(step_rows) != num_layers:
            problems.append(
                f"rank={rank} step={step}: got {len(step_rows)} layers, "
                f"expected {num_layers}"
            )
            continue
        sched = step_rows[0]["scheduled_tokens"]
        for row in step_rows:
            if row["scheduled_tokens"] != sched:
                problems.append(
                    f"rank={rank} step={step} layer={row['layer_idx']}: "
                    f"scheduled_tokens={row['scheduled_tokens']} differs from "
                    f"first row's {sched}"
                )
            for k in (
                "dispatch_in_tokens",
                "dispatch_out_tokens",
                "combine_in_tokens",
                "combine_out_tokens",
            ):
                if row[k] is None:
                    problems.append(
                        f"rank={rank} step={step} layer={row['layer_idx']}: "
                        f"missing {k}"
                    )
            # Per-row AG/RS sanity: dispatch output includes own + peer tokens
            # (must be >= own input); combine input includes peer contributions
            # to own shard (must be >= own output shard).
            # NOTE: we deliberately do NOT check `out == W * in` because under
            # chunked-prefill with sequence parallelism, ranks can have
            # unequal local chunk sizes, so AG total = sum_over_ranks(in),
            # not W * in.
            d_in, d_out = row["dispatch_in_tokens"], row["dispatch_out_tokens"]
            c_in, c_out = row["combine_in_tokens"], row["combine_out_tokens"]
            if None not in (d_in, d_out) and d_out < d_in:
                problems.append(
                    f"rank={rank} step={step} layer={row['layer_idx']}: "
                    f"dispatch_out_tokens({d_out}) < dispatch_in_tokens({d_in})"
                )
            if None not in (c_in, c_out) and c_in < c_out:
                problems.append(
                    f"rank={rank} step={step} layer={row['layer_idx']}: "
                    f"combine_in_tokens({c_in}) < combine_out_tokens({c_out})"
                )

    # NOTE: cross-rank checks (e.g. "ranks should agree on dispatch_out_tokens
    # for the same (step_id, layer_idx)") are intentionally omitted. `step_id`
    # is a per-worker monotonic counter incremented inside execute_model on
    # each rank independently; DP idle bubbles / dummy_runs cause rank 0 and
    # rank 1 to advance at different rates, so the same step_id does not
    # represent the same forward pass across ranks. This profiler is scoped
    # to per-rank trajectories.

    if problems:
        print(
            f"[extract_per_step] {len(problems)} invariant violations "
            "(showing up to 20):",
            file=sys.stderr,
        )
        for p in problems[:20]:
            print(f"  {p}", file=sys.stderr)
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--jsonl-glob",
        required=True,
        help="Glob pattern for raw per-rank JSONL files "
        "(e.g. '/tmp/moe_a2a_rank*.jsonl').",
    )
    ap.add_argument(
        "--num-layers",
        type=int,
        required=True,
        help="Number of MoE layers in the model (e.g. 48 for Qwen3-30B-A3B).",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Output CSV path.",
    )
    ap.add_argument(
        "--no-assert",
        action="store_true",
        help="Skip per-step invariant assertions (still report on stderr).",
    )
    ap.add_argument(
        "--keep-warmup",
        action="store_true",
        help=(
            "Keep warmup / idle rows in the output CSV. By default rows are "
            "dropped if their (rank, step_id) group did not have exactly "
            "num_layers * 2 raw records (kernel warmup / profiling "
            "phases that re-run the model multiple times under one step_id), "
            "or if scheduled_tokens <= 0 (pre-execute_model warmup with "
            "step_id=-1, or DP idle bubbles). Use this flag to retain them "
            "for inspection."
        ),
    )
    args = ap.parse_args()

    paths = sorted(glob.glob(args.jsonl_glob))
    if not paths:
        raise SystemExit(f"No files match: {args.jsonl_glob}")
    print(f"[extract_per_step] reading {len(paths)} files", file=sys.stderr)

    records = _read_jsonl(paths)
    print(f"[extract_per_step] read {len(records)} records", file=sys.stderr)

    # Identify well-formed (rank, step_id) groups: exactly num_layers * 2
    # records (dispatch + combine for every MoE layer). Anything else is a
    # warmup / profiling repetition that re-ran the model multiple times
    # under the same step_id; folding those would silently overwrite same-key
    # rows and yield meaningless single-shot values.
    expected_per_step = args.num_layers * 2
    group_counts: dict[tuple[int, int], int] = {}
    for r in records:
        key = (r["rank"], r["step_id"])
        group_counts[key] = group_counts.get(key, 0) + 1
    well_formed_keys = {
        k for k, c in group_counts.items() if c == expected_per_step
    }
    print(
        f"[extract_per_step] (rank,step) groups: "
        f"{len(well_formed_keys)} well-formed, "
        f"{len(group_counts) - len(well_formed_keys)} abnormal",
        file=sys.stderr,
    )

    rows = _fold(records)
    print(
        f"[extract_per_step] folded into {len(rows)} (rank,step,layer) rows",
        file=sys.stderr,
    )

    if not args.keep_warmup:
        before = len(rows)
        rows = {
            k: v for k, v in rows.items()
            if (v["rank"], v["step_id"]) in well_formed_keys
            and v["scheduled_tokens"] > 0
        }
        print(
            f"[extract_per_step] dropped {before - len(rows)} warmup/idle "
            f"rows (use --keep-warmup to retain)",
            file=sys.stderr,
        )

    if not args.no_assert:
        _assert_invariants(rows, args.num_layers)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank", "step_id", "layer_idx", "scheduled_tokens", "world_size",
        "dispatch_in_tokens", "dispatch_in_bytes",
        "dispatch_out_tokens", "dispatch_out_bytes",
        "combine_in_tokens", "combine_in_bytes",
        "combine_out_tokens", "combine_out_bytes",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        # Sort by (rank, step_id, layer_idx) for stable output.
        for key in sorted(rows.keys()):
            w.writerow(rows[key])
    print(f"[extract_per_step] wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
