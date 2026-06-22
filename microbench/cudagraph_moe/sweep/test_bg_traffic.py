"""Lightweight tests for bg_traffic patterns.

Covers the pure-python pieces (Pattern subclasses, registry). Doesn't
touch CUDA or threads.

    python -m unittest microbench.cudagraph_moe.realistic.test_bg_traffic
"""

from __future__ import annotations

import threading
import time
import unittest

from .bg_traffic import (
    ConstantRate,
    PATTERN_REGISTRY,
    make_pattern,
)


class ConstantRateTest(unittest.TestCase):
    def test_validation(self):
        with self.assertRaises(ValueError):
            ConstantRate(rate_bytes_per_sec=0, chunk_bytes=1024)
        with self.assertRaises(ValueError):
            ConstantRate(rate_bytes_per_sec=1e9, chunk_bytes=0)

    def test_period(self):
        # 1 MB chunks at 1 GB/s -> 1 ms per op (SI units).
        p = ConstantRate(rate_bytes_per_sec=1_000_000_000, chunk_bytes=1_000_000)
        self.assertAlmostEqual(p._period_s, 1e-3, delta=1e-6)

    def test_pacing(self):
        # 4 MiB chunks at 4 GiB/s ~ 1 ms per op. 5 ops -> ~4 ms.
        p = ConstantRate(
            rate_bytes_per_sec=4 * 1024 ** 3, chunk_bytes=4 * 1024 ** 2
        )
        stop = threading.Event()
        p.reset()
        t0 = time.monotonic()
        for _ in range(5):
            n = p.next_chunk_bytes(stop)
            self.assertEqual(n, 4 * 1024 ** 2)
        elapsed = time.monotonic() - t0
        # Generous slack for OS scheduling jitter.
        self.assertGreater(elapsed, 3e-3)
        self.assertLess(elapsed, 50e-3)

    def test_stop_event_aborts_sleep(self):
        # Long inter-op delay; setting stop should abort within ~ms.
        p = ConstantRate(rate_bytes_per_sec=1.0, chunk_bytes=1)  # 1s per op
        stop = threading.Event()
        p.reset()
        # First call schedules the next op 1s in the future...
        n = p.next_chunk_bytes(stop)
        self.assertEqual(n, 1)
        # ...so the next call would sleep ~1s; fire stop after 20ms.
        def _fire():
            time.sleep(0.02)
            stop.set()
        threading.Thread(target=_fire, daemon=True).start()
        t0 = time.monotonic()
        n2 = p.next_chunk_bytes(stop)
        elapsed = time.monotonic() - t0
        self.assertIsNone(n2)
        self.assertLess(elapsed, 0.3)  # well under the 1s nominal period

    def test_describe(self):
        p = ConstantRate(rate_bytes_per_sec=100e9, chunk_bytes=4 * 1024 * 1024)
        d = p.describe()
        self.assertIn("ConstantRate", d)
        self.assertIn("GB/s", d)


class RegistryTest(unittest.TestCase):
    def test_constant_registered(self):
        self.assertIn("constant", PATTERN_REGISTRY)
        self.assertIs(PATTERN_REGISTRY["constant"], ConstantRate)

    def test_make_pattern_constant(self):
        p = make_pattern("constant", rate_bytes_per_sec=1e9, chunk_bytes=1024)
        self.assertIsInstance(p, ConstantRate)

    def test_make_pattern_unknown(self):
        with self.assertRaisesRegex(ValueError, "Unknown background traffic pattern"):
            make_pattern("nonexistent")

    def test_extensibility(self):
        # New patterns plug in via the registry without touching the
        # driver or main.py.
        class CustomPattern(ConstantRate):
            pass

        PATTERN_REGISTRY["custom_test"] = CustomPattern
        try:
            p = make_pattern("custom_test", rate_bytes_per_sec=1e9, chunk_bytes=64)
            self.assertIsInstance(p, CustomPattern)
        finally:
            del PATTERN_REGISTRY["custom_test"]


if __name__ == "__main__":
    unittest.main()
