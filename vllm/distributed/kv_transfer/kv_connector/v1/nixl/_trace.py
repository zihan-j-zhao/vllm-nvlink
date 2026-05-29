# SPDX-License-Identifier: Apache-2.0
"""
Lightweight, env-gated JSONL tracer for the P/D-disagg + EP experiment.

Records:
  - decode-side NIXL recv start/done intervals (for shading "KV xfer active"
    regions on the wall-clock axis), and
  - per-engine-step token emissions (for reconstructing system-wide TPOT/ITL
    timeseries from any role/DP rank).

Hard requirements:
  - **CUDA-graph safe**: only `time.perf_counter()`, list append, queue put.
    Never imports torch in the hot path. Never allocates GPU memory or runs
    GPU ops. Invoked strictly outside the captured `forward()` region.
  - **Zero overhead when disabled**: a single `is_enabled()` check returns
    False if `VLLM_PD_TRACE_DIR` is unset, and call sites guard on it.

Activation:
  export VLLM_PD_TRACE_DIR=/abs/path/to/dir
  (Optional) export VLLM_PD_TRACE_LABEL=my_run   # tag added to filenames

File layout: one JSONL file per process under $VLLM_PD_TRACE_DIR, named
  role=<prefill|decode|engine>_dp=<i>_tp=<j>_pid=<pid>[_label=<L>].jsonl
Each line is one JSON object; see TraceWriter.recv_start/recv_done/step.

Origin clock is `time.perf_counter()` (process-monotonic). On startup we also
write a `boot` event that records `time.time()` and `perf_counter()` together
so post-processing can align ranks via the wall-clock pair.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from typing import Any

_DISABLED_SENTINEL = object()

# Process-global singleton; we only ever have one writer per process.
_writer: "TraceWriter | None | object" = _DISABLED_SENTINEL


def _trace_dir() -> str | None:
    d = os.environ.get("VLLM_PD_TRACE_DIR")
    if not d:
        return None
    return d


def is_enabled() -> bool:
    return _trace_dir() is not None


class TraceWriter:
    """Background-thread JSONL writer. Main thread only does q.put_nowait()."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._q: queue.SimpleQueue[Any] = queue.SimpleQueue()
        # daemon=True so the thread does not block process exit; we also
        # register an atexit-style flush via a stop sentinel.
        self._stop = object()
        self._thread = threading.Thread(
            target=self._run,
            name=f"pd-trace[{os.path.basename(path)}]",
            daemon=True,
        )
        # Open file in line-buffered text mode so each flush hits disk; the
        # background thread is the only writer.
        # buffering=1 (line buffering) only applies to text mode, which is
        # what we want here.
        self._fp = open(path, "w", buffering=1, encoding="utf-8")
        self._thread.start()
        # Record clock alignment (wall <-> perf_counter) up front.
        self._emit(
            {
                "ev": "boot",
                "ts": time.perf_counter(),
                "wall": time.time(),
                "pid": os.getpid(),
            }
        )

    # ---------------- hot-path API ----------------

    def recv_start(self, req_id: str, dst_engine_id: str | None = None,
                   remote_rank: int | None = None,
                   n_local_blocks: int | None = None) -> None:
        self._emit({
            "ev": "recv_start",
            "ts": time.perf_counter(),
            "req": req_id,
            "remote_engine": dst_engine_id,
            "remote_rank": remote_rank,
            "n_blocks": n_local_blocks,
        })

    def recv_done(self, req_id: str) -> None:
        self._emit({
            "ev": "recv_done",
            "ts": time.perf_counter(),
            "req": req_id,
        })

    def step(self, tokens: list[tuple[str, int]],
             num_scheduled: int | None = None) -> None:
        """Per-scheduler-step token emission summary.

        `tokens` is a list of (request_id, num_new_tokens_this_step) pairs,
        filtered to those that actually got >=1 new sampled token. The list
        format keeps the per-line JSON small even for batches of hundreds.
        """
        self._emit({
            "ev": "step",
            "ts": time.perf_counter(),
            "n_sched": num_scheduled,
            "tokens": tokens,
        })

    def step_done(self, n_finished: int, n_recv_done: int) -> None:
        """Post-step bookkeeping summary, emitted at the end of
        `Scheduler.update_from_output` (so its timestamp is after per-request
        finalization). Pairs with the immediately preceding `step` event.

        Lets the plotter attribute ITL spikes:
          * `n_finished` > 0 with no recv activity nearby -> EOS clump
            (per-request finalization in scheduler)
          * `n_recv_done` > 0 -> KV-recv completion bookkeeping (block-table
            integration, post-process kernels)
          * both zero -> something else (DP sync, GC, allocator, ...).
        """
        self._emit({
            "ev": "step_done",
            "ts": time.perf_counter(),
            "n_finished": int(n_finished),
            "n_recv_done": int(n_recv_done),
        })

    # ---------------- internals ----------------

    def _emit(self, obj: Any) -> None:
        try:
            self._q.put_nowait(obj)
        except Exception:
            # Tracing must never raise into the engine; drop on overflow.
            pass

    def _run(self) -> None:
        # Single thread, single file -> no locking needed.
        while True:
            try:
                obj = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if obj is self._stop:
                break
            try:
                self._fp.write(json.dumps(obj, separators=(",", ":")) + "\n")
            except Exception:
                # Best-effort tracer; never crash the engine on disk errors.
                pass

    def close(self) -> None:
        try:
            self._q.put_nowait(self._stop)
            self._thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            self._fp.flush()
            self._fp.close()
        except Exception:
            pass


def get_writer(role: str, dp_rank: int, tp_rank: int = 0) -> "TraceWriter | None":
    """Return the process-wide writer, creating it on first use.

    Returns None if VLLM_PD_TRACE_DIR is unset. The (role, dp_rank, tp_rank)
    tuple is encoded in the filename; the first call wins (we only have one
    writer per process). Subsequent calls with different labels are ignored,
    which is fine because each process owns exactly one (role, dp, tp) slot.
    """
    global _writer
    if _writer is _DISABLED_SENTINEL:
        d = _trace_dir()
        if d is None:
            _writer = None
            return None
        os.makedirs(d, exist_ok=True)
        label = os.environ.get("VLLM_PD_TRACE_LABEL")
        suffix = f"_label={label}" if label else ""
        fname = (
            f"role={role}_dp={dp_rank}_tp={tp_rank}_pid={os.getpid()}{suffix}.jsonl"
        )
        _writer = TraceWriter(os.path.join(d, fname))
    return _writer  # type: ignore[return-value]
