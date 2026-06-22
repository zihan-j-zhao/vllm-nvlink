"""Lightweight unit tests for the synthetic scheduler.

Runs without vLLM / CUDA so it can be exercised on any machine. Tests the
pure-python pieces: distribution parsing, block-pool accounting, and
workload assembly.

    python -m unittest microbench.cudagraph_moe.realistic.test_fake_scheduler
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .fake_scheduler import (
    BlockPool,
    build_workload,
    parse_int_dist,
)


class ParseIntDistTest(unittest.TestCase):
    def test_fixed(self):
        f = parse_int_dist("fixed:7")
        rng = np.random.default_rng(0)
        for _ in range(20):
            self.assertEqual(f(rng), 7)

    def test_uniform_range(self):
        f = parse_int_dist("uniform:3:5")
        rng = np.random.default_rng(0)
        vals = {f(rng) for _ in range(200)}
        self.assertTrue(vals.issubset({3, 4, 5}))
        # Should hit every value with enough draws.
        self.assertEqual(vals, {3, 4, 5})

    def test_uniform_inverted_range(self):
        with self.assertRaises(ValueError):
            parse_int_dist("uniform:10:5")

    def test_geometric_mean(self):
        f = parse_int_dist("geometric:5")
        rng = np.random.default_rng(0)
        samples = [f(rng) for _ in range(10_000)]
        # Geometric mean should be close to 5 (loose tolerance).
        self.assertAlmostEqual(np.mean(samples), 5.0, delta=0.3)
        self.assertTrue(all(v >= 1 for v in samples))

    def test_geometric_too_small(self):
        with self.assertRaises(ValueError):
            parse_int_dist("geometric:0.5")

    def test_file(self):
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False
        ) as fh:
            json.dump([10, 20, 30], fh)
            path = fh.name
        try:
            f = parse_int_dist(f"file:{path}")
            rng = np.random.default_rng(0)
            vals = {f(rng) for _ in range(200)}
            self.assertEqual(vals, {10, 20, 30})
        finally:
            Path(path).unlink()

    def test_unknown_kind(self):
        with self.assertRaises(ValueError):
            parse_int_dist("poisson:5")

    def test_missing_colon(self):
        with self.assertRaises(ValueError):
            parse_int_dist("fixed7")


class BlockPoolTest(unittest.TestCase):
    def test_disjoint_alloc(self):
        pool = BlockPool([100])
        a = pool.alloc([10])
        b = pool.alloc([20])
        self.assertEqual(a, (list(range(1, 11)),))
        self.assertEqual(b, (list(range(11, 31)),))
        # IDs must be disjoint.
        self.assertTrue(set(a[0]).isdisjoint(set(b[0])))
        # Block 0 is reserved and must never be handed out.
        self.assertNotIn(0, a[0] + b[0])

    def test_exhaustion(self):
        pool = BlockPool([10])
        pool.alloc([5])
        with self.assertRaisesRegex(RuntimeError, "exhausted"):
            pool.alloc([10])

    def test_multi_group(self):
        pool = BlockPool([50, 100])
        a = pool.alloc([3, 7])
        self.assertEqual(a, (list(range(1, 4)), list(range(1, 8))))
        b = pool.alloc([2, 4])
        # Per-group cursors are independent.
        self.assertEqual(b, (list(range(4, 6)), list(range(8, 12))))

    def test_interleaved_alloc(self):
        pool = BlockPool([100])
        reqs = pool.alloc_interleaved([[3], [3], [3]])
        self.assertEqual(reqs[0], ([1, 4, 7],))
        self.assertEqual(reqs[1], ([2, 5, 8],))
        self.assertEqual(reqs[2], ([3, 6, 9],))

    def test_interleaved_alloc_ragged(self):
        pool = BlockPool([100])
        reqs = pool.alloc_interleaved([[3], [1], [2]])
        self.assertEqual(reqs[0], ([1, 4, 6],))
        self.assertEqual(reqs[1], ([2],))
        self.assertEqual(reqs[2], ([3, 5],))


class BuildWorkloadTest(unittest.TestCase):
    def test_sizes_and_block_counts(self):
        workload = build_workload(
            batch_size=4,
            prefill_sampler=parse_int_dist("fixed:1024"),
            age_sampler=parse_int_dist("fixed:0"),
            max_decode_steps=512,
            block_sizes_per_group=[16],
            num_blocks_per_group=[10_000],
            seed=42,
        )
        self.assertEqual(len(workload), 4)
        # Need ceil((1024 + 512) / 16) = 96 blocks per request, single group.
        for req in workload:
            self.assertEqual(len(req.block_ids), 1)
            self.assertEqual(len(req.block_ids[0]), 96)
        # req_idx should be 0..B-1, monotonic.
        self.assertEqual([r.req_idx for r in workload], [0, 1, 2, 3])

    def test_seq_len_advances_with_age(self):
        workload = build_workload(
            batch_size=1,
            prefill_sampler=parse_int_dist("fixed:100"),
            age_sampler=parse_int_dist("fixed:0"),
            max_decode_steps=10,
            block_sizes_per_group=[16],
            num_blocks_per_group=[10_000],
            seed=0,
        )
        r = workload[0]
        self.assertEqual(r.seq_len, 101)  # prefill + age + 1 (new tok)
        self.assertEqual(r.position, 100)  # prefill + age (0-indexed)
        r.age_steps += 5
        self.assertEqual(r.seq_len, 106)
        self.assertEqual(r.position, 105)

    def test_blocks_disjoint_across_reqs(self):
        workload = build_workload(
            batch_size=8,
            prefill_sampler=parse_int_dist("uniform:128:512"),
            age_sampler=parse_int_dist("uniform:0:100"),
            max_decode_steps=128,
            block_sizes_per_group=[16, 32],
            num_blocks_per_group=[5000, 5000],
            seed=7,
        )
        for gid in range(2):
            seen: set[int] = set()
            for r in workload:
                ids = set(r.block_ids[gid])
                self.assertTrue(
                    seen.isdisjoint(ids),
                    f"group {gid}: req {r.req_id} overlaps prior",
                )
                seen.update(ids)

    def test_interleaved_layout(self):
        workload = build_workload(
            batch_size=3,
            prefill_sampler=parse_int_dist("fixed:32"),
            age_sampler=parse_int_dist("fixed:0"),
            max_decode_steps=0,
            block_sizes_per_group=[16],
            num_blocks_per_group=[100],
            seed=0,
            block_layout="interleaved",
        )
        # Need ceil(32 / 16) = 2 blocks per request.
        self.assertEqual(workload[0].block_ids[0], [1, 4])
        self.assertEqual(workload[1].block_ids[0], [2, 5])
        self.assertEqual(workload[2].block_ids[0], [3, 6])

    def test_exhaustion_raises_clearly(self):
        # Cap too small to fit 4 requests * 16 blocks each.
        with self.assertRaisesRegex(RuntimeError, "exhausted"):
            build_workload(
                batch_size=4,
                prefill_sampler=parse_int_dist("fixed:128"),
                age_sampler=parse_int_dist("fixed:0"),
                max_decode_steps=0,
                block_sizes_per_group=[16],
                num_blocks_per_group=[20],  # only ~19 usable blocks
                seed=0,
            )

    def test_max_blocks_per_req_cap(self):
        # prefill=10000 + decode=4096 = 14096 tokens → 881 blocks (bs=16),
        # but max_blocks_per_req=512 — should fail fast with a clear message
        # that mentions --max-model-len.
        with self.assertRaisesRegex(ValueError, "max-model-len"):
            build_workload(
                batch_size=1,
                prefill_sampler=parse_int_dist("fixed:10000"),
                age_sampler=parse_int_dist("fixed:0"),
                max_decode_steps=4096,
                block_sizes_per_group=[16],
                num_blocks_per_group=[100_000],
                max_blocks_per_req_per_group=[512],  # = cdiv(8192, 16)
                seed=0,
            )

    def test_max_blocks_per_req_passes_when_fits(self):
        # 1024 + 1024 = 2048 tokens → 128 blocks (bs=16), well under cap.
        workload = build_workload(
            batch_size=2,
            prefill_sampler=parse_int_dist("fixed:1024"),
            age_sampler=parse_int_dist("fixed:0"),
            max_decode_steps=1024,
            block_sizes_per_group=[16],
            num_blocks_per_group=[10_000],
            max_blocks_per_req_per_group=[512],
            seed=0,
        )
        self.assertEqual(len(workload), 2)


if __name__ == "__main__":
    unittest.main()
