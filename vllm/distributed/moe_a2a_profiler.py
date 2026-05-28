# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight per-rank, per-layer, per-step profiler for the MoE all-to-all
boundary.

Scope (refuses to start if any assumption is violated; see ``_enforce_scope``):

* Single-node, DP=EP=2, ``enforce_eager=True``.
* ``pcp_size == 1``.
* ``all2all`` backend is the naive AG/RS path (``AgRsAll2AllManager``).
  Other backends (DeepEP / Mori / NIXL / FlashInfer) are out of scope: the
  EP choke point in :mod:`cuda_communicator` is not the call site for them.

Records one JSONL row per call into the choke point with::

    {seq, rank, step_id, layer_idx, layer_name, scheduled_tokens, kind,
     world_size, in_tokens, in_bytes, out_tokens, out_bytes, time_ms}

``step_id`` and ``scheduled_tokens`` are published once per forward step from
``GPUModelRunner.execute_model``; ``layer_name`` is published once per MoE
layer from ``MoeRunner._forward_impl``. Records are written in execution
order; cross-rank alignment is by ``step_id``, not by record index. ``seq``
is a per-process monotonic counter; kept for backward compatibility with
older traces that joined a separate timing sidecar.

Timing
------

The caller (cuda_communicator) brackets each collective with
``torch.cuda.synchronize()`` and ``time.perf_counter()`` and passes the
resulting ``time_ms`` to ``record()``, which writes it directly into the
JSONL row. No separate sidecar is produced. The synchronization is a
device-wide drain, so enabling the profiler measurably slows the run and
removes any compute/comm overlap; this is acceptable for the use case
(offline characterization of the AG/RS choke point) but means the trace
is not representative of unprofiled wall-clock performance.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import signal
import threading
from collections.abc import Iterable, Sequence
from typing import Any

import torch

_LAYER_IDX_RE = re.compile(r"layers\.(\d+)\.")


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _parse_layer_idx(layer_name: str | None) -> int:
    if not layer_name:
        return -1
    m = _LAYER_IDX_RE.search(layer_name)
    return int(m.group(1)) if m else -1


def _sum_bytes(tensors: Iterable[Any]) -> int:
    total = 0
    for t in tensors:
        if t is None:
            continue
        if isinstance(t, torch.Tensor):
            total += t.numel() * t.element_size()
        elif isinstance(t, (list, tuple)):
            total += _sum_bytes(t)
    return total


def _first_dim(tensors: Sequence[Any]) -> int:
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.dim() >= 1:
            return int(t.shape[0])
    return -1


class _Profiler:
    """Process-local singleton.

    Thread-safety: ``set_step`` / ``set_layer`` / ``record`` are guarded by a
    single ``_lock`` since the GPU worker thread is the only writer in
    practice but the JSONL file may also be flushed from ``atexit`` /
    ``SIGTERM`` handlers.
    """

    def __init__(self) -> None:
        self._enabled = _bool_env("VLLM_MOE_A2A_PROFILE", False)
        self._path_template = os.environ.get(
            "VLLM_MOE_A2A_PROFILE_PATH", "/tmp/moe_a2a_rank{rank}.jsonl"
        )
        self._lock = threading.Lock()
        self._fp = None  # type: ignore[var-annotated]
        self._path: str | None = None
        self._rank: int | None = None
        self._step_id: int = -1
        self._scheduled_tokens: int = -1
        self._layer_name: str | None = None
        # Set true after _enforce_scope succeeds; profile records only then.
        self._scope_ok: bool = False
        self._scope_checked: bool = False
        # Monotonic per-process record counter; retained for backward
        # compatibility with older traces (some downstream tooling joins on it).
        self._seq: int = 0

        if self._enabled:
            atexit.register(self.close)
            try:
                signal.signal(signal.SIGTERM, self._on_sigterm)
            except (ValueError, OSError):
                # Not main thread or signal unsupported; atexit still fires.
                pass

    # ------------------------------------------------------------------ scope

    def enforce_scope(
        self,
        *,
        all2all_backend: str,
        data_parallel_size: int,
        enable_expert_parallel: bool,
        pcp_size: int,
    ) -> None:
        """Validate hard preconditions once per process.

        Called from the MoE runner on first forward (which is the first place
        that has a fully-populated MoE config). Disables the profiler with a
        clear error line on stderr if any check fails, so a misconfigured
        run produces no silently-wrong trace.

        Note: enforce_eager is *not* checked here because the global vLLM
        config is not always reachable from the worker forward path. The
        launch script (playground/moe_a2a/start_server.sh) is the
        authoritative source for that requirement.
        """
        if not self._enabled or self._scope_checked:
            return
        self._scope_checked = True
        problems: list[str] = []
        if all2all_backend not in ("naive", "allgather_reducescatter"):
            problems.append(
                f"all2all_backend={all2all_backend!r} (need naive or "
                "allgather_reducescatter; other backends bypass the EP "
                "communicator choke point)"
            )
        if not enable_expert_parallel:
            problems.append("enable_expert_parallel=False")
        if data_parallel_size != 2:
            problems.append(
                f"data_parallel_size={data_parallel_size} (this profiler is "
                "scoped to DP=EP=2)"
            )
        if pcp_size != 1:
            problems.append(f"pcp_size={pcp_size} (need 1)")
        if problems:
            self._enabled = False
            self._scope_ok = False
            msg = (
                "[moe_a2a_profiler] disabled because preconditions failed: "
                + "; ".join(problems)
            )
            print(msg, flush=True)
            return
        self._scope_ok = True
        print(
            f"[moe_a2a_profiler] enabled, pid={os.getpid()} "
            f"backend={all2all_backend} dp={data_parallel_size} "
            f"ep={enable_expert_parallel} pcp={pcp_size}",
            flush=True,
        )

    # ----------------------------------------------------------------- public

    def is_enabled(self) -> bool:
        return self._enabled and self._scope_ok

    def set_step(self, step_id: int, scheduled_tokens: int) -> None:
        if not self._enabled:
            return
        with self._lock:
            self._step_id = int(step_id)
            self._scheduled_tokens = int(scheduled_tokens)

    def set_layer(self, layer_name: str | None) -> None:
        if not self._enabled:
            return
        with self._lock:
            self._layer_name = layer_name

    def record(
        self,
        *,
        kind: str,
        rank: int,
        world_size: int,
        in_tensors: Sequence[Any],
        out_tensors: Sequence[Any],
        time_ms: float | None = None,
    ) -> None:
        if not self.is_enabled():
            return
        in_bytes = _sum_bytes(in_tensors)
        out_bytes = _sum_bytes(out_tensors)
        in_tokens = _first_dim(in_tensors)
        out_tokens = _first_dim(out_tensors)
        with self._lock:
            seq = self._seq
            self._seq += 1
            rec = {
                "seq": seq,
                "rank": int(rank),
                "step_id": self._step_id,
                "layer_idx": _parse_layer_idx(self._layer_name),
                "layer_name": self._layer_name,
                "scheduled_tokens": self._scheduled_tokens,
                "kind": kind,
                "world_size": int(world_size),
                "in_tokens": in_tokens,
                "in_bytes": in_bytes,       # sizeof in_tensors (not necessarily the bytes over the wire)
                "out_tokens": out_tokens,
                "out_bytes": out_bytes,     # sizeof out_tensors (not necessarily the bytes over the wire)
            }
            if time_ms is not None:
                rec["time_ms"] = float(time_ms)
            self._ensure_open(rank)
            assert self._fp is not None
            self._fp.write(json.dumps(rec, separators=(",", ":")) + "\n")

    def close(self) -> None:
        with self._lock:
            if self._fp is not None:
                try:
                    self._fp.flush()
                    self._fp.close()
                finally:
                    self._fp = None

    # ---------------------------------------------------------------- private

    def _ensure_open(self, rank: int) -> None:
        if self._fp is not None:
            return
        self._rank = int(rank)
        self._path = self._path_template.replace("{rank}", str(self._rank))
        parent = os.path.dirname(self._path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # Line-buffered so a crash leaves a complete, parseable JSONL.
        self._fp = open(self._path, "a", buffering=1)

    def _on_sigterm(self, _signum, _frame) -> None:
        self.close()
        # Re-raise default behavior so the worker actually exits.
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGTERM)


_instance: _Profiler | None = None


def get_profiler() -> _Profiler:
    global _instance
    if _instance is None:
        _instance = _Profiler()
    return _instance
