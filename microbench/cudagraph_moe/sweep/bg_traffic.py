"""Background CE traffic from GPUs 4-7 -> GPUs 0-3.

Fixed assumptions for this microbench:
  - 8 GPUs total on the node.
  - Vllm decode runs on local_rank 0..3.
  - GPUs 4..7 are idle and used only to source background traffic.
  - Mapping is 1:1: rank R receives bytes from GPU (R + 4).
  - Transport is cudaMemcpyPeerAsync (CE over NVLink), in the
    src-on-prefiller, dst-on-decoder direction. This matches NIXL's
    intra-node KV transfer path.

What is *not* fixed (so we can add new shapes later):
  - The rate / chunk-size pattern. ``Pattern`` is a one-method ABC; add
    new shapes (bursty, poisson, trace replay) by subclassing and
    registering in ``PATTERN_REGISTRY``. The driver code never touches
    pattern internals.

Threading model: the background thread loops:
    n = pattern.next_chunk_bytes(stop_event)  # may sleep, may return None
    if n is None: break
    issue cudaMemcpyPeerAsync(n bytes) on side stream

Side stream + side thread = no contention with vLLM's kernel queue, only
with the underlying hardware (NVLink CE engines).
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Pattern: WHEN / HOW MUCH (the only extension point)
# ---------------------------------------------------------------------------
class Pattern(ABC):
    """Generates one chunk size per call. Blocks for pacing as needed.

    Implementations should use ``stop_event.wait(timeout)`` rather than
    ``time.sleep`` so shutdown is prompt. Return ``None`` to indicate
    "stop the run".
    """

    @abstractmethod
    def next_chunk_bytes(self, stop_event: threading.Event) -> int | None: ...

    def reset(self) -> None:
        """Called once before the first ``next_chunk_bytes`` call."""

    def describe(self) -> str:
        return self.__class__.__name__


class ConstantRate(Pattern):
    """Steady ``rate_bytes_per_sec`` of ``chunk_bytes``-sized ops.

    Inter-op delay = chunk_bytes / rate. If the link can't sustain the
    requested rate, ``cudaMemcpyAsync`` queues onto the side stream and
    the achieved rate plateaus at the link's actual capacity.
    """

    def __init__(self, rate_bytes_per_sec: float, chunk_bytes: int):
        if rate_bytes_per_sec <= 0:
            raise ValueError("rate_bytes_per_sec must be > 0")
        if chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be > 0")
        self.rate_bytes_per_sec = rate_bytes_per_sec
        self.chunk_bytes = chunk_bytes
        self._period_s = chunk_bytes / rate_bytes_per_sec
        self._next_t = 0.0

    def reset(self) -> None:
        self._next_t = time.monotonic()

    def next_chunk_bytes(self, stop_event: threading.Event) -> int | None:
        delay = self._next_t - time.monotonic()
        if delay > 0 and stop_event.wait(timeout=delay):
            return None  # asked to stop mid-sleep
        self._next_t += self._period_s
        return self.chunk_bytes

    def describe(self) -> str:
        return (
            f"ConstantRate(rate={self.rate_bytes_per_sec / 1e9:.1f} GB/s, "
            f"chunk={self.chunk_bytes // 1024} KiB, "
            f"period={self._period_s * 1e6:.1f} us)"
        )


# Registry. Add new patterns here; main.py picks them up by name.
# Each value is a (factory_callable, help_string) pair.
PATTERN_REGISTRY: dict[str, type[Pattern]] = {
    "constant": ConstantRate,
}


def make_pattern(name: str, **kwargs: Any) -> Pattern:
    if name not in PATTERN_REGISTRY:
        raise ValueError(
            f"Unknown background traffic pattern {name!r}; "
            f"available: {sorted(PATTERN_REGISTRY)}"
        )
    return PATTERN_REGISTRY[name](**kwargs)


# ---------------------------------------------------------------------------
# BackgroundTraffic: holds buffers + side stream, runs a worker thread
# ---------------------------------------------------------------------------
# Fixed mapping for this bench: decoder local_rank R pairs with GPU R+4.
DECODER_RANK_COUNT = 4
SRC_DEVICE_OFFSET = 4
REQUIRED_VISIBLE_GPUS = 8

# Supported traffic directions, from the decoder's perspective:
#   ingress: phantom(R+4) -> decoder(R)   [CE engine on phantom, decoder HBM=write]
#   egress : decoder(R)   -> phantom(R+4) [CE engine on decoder, decoder HBM=read]
# 'both' is built by the caller from two BackgroundTraffic instances.
DIRECTIONS = ("ingress", "egress")


class BackgroundTraffic:
    """Per-decoder-rank CE traffic to/from its phantom prefill GPU.

    Spawned once per decoder rank.  ``direction`` controls which GPU owns
    the CE engine (always the *source*) and therefore where contention
    lands:

      - ``ingress`` (default): src=phantom, dst=decoder. CE runs on the
        phantom; decoder GPU only does HBM **writes** (~1% of decode's
        HBM activity), so this mostly stresses the link, not the
        decoder's HBM read path.
      - ``egress``: src=decoder, dst=phantom. CE runs on the **decoder**,
        which means HBM reads on the decoder fight the decode forward's
        HBM reads. This is what shifts p50 for memory-bound decodes.

    For full bidirectional KV-transfer-like load, create two instances
    with different directions and start both.
    """

    def __init__(
        self,
        local_rank: int,
        pattern: Pattern,
        buffer_bytes: int,
        direction: str = "ingress",
        max_inflight_copies: int = 0,
    ):
        if local_rank >= DECODER_RANK_COUNT:
            raise ValueError(
                f"BackgroundTraffic only configured for local_rank "
                f"0..{DECODER_RANK_COUNT - 1}, got {local_rank}"
            )
        if direction not in DIRECTIONS:
            raise ValueError(
                f"direction must be one of {DIRECTIONS}, got {direction!r}"
            )
        visible = torch.cuda.device_count()
        if visible < REQUIRED_VISIBLE_GPUS:
            raise RuntimeError(
                f"BackgroundTraffic requires {REQUIRED_VISIBLE_GPUS} visible "
                f"GPUs (4 decoders + 4 phantom prefills); only {visible} "
                "visible. Launch with CUDA_VISIBLE_DEVICES unset or =0..7."
            )
        decoder = torch.device(f"cuda:{local_rank}")
        phantom = torch.device(f"cuda:{local_rank + SRC_DEVICE_OFFSET}")
        if direction == "ingress":
            self.src_device, self.dst_device = phantom, decoder
        else:  # egress
            self.src_device, self.dst_device = decoder, phantom
        self.direction = direction
        if not torch.cuda.can_device_access_peer(
            self.dst_device.index, self.src_device.index
        ):
            raise RuntimeError(
                f"GPU {self.dst_device.index} cannot access peer "
                f"{self.src_device.index}; intra-node NVLink P2P required."
            )

        self.pattern = pattern
        self.buffer_bytes = buffer_bytes
        if max_inflight_copies < 0:
            raise ValueError("max_inflight_copies must be >= 0")
        self.max_inflight_copies = max_inflight_copies
        # CE engine on the *source* GPU drives the copy, so put the
        # stream there. (Putting the stream on the dst still works but
        # would record the cudaMemcpyPeerAsync against the dst's stream
        # queue, which is misleading in nsys.)
        with torch.cuda.device(self.src_device):
            self._stream = torch.cuda.Stream(device=self.src_device)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._src: torch.Tensor | None = None
        self._dst: torch.Tensor | None = None
        # Stats
        self._ops = 0
        self._bytes = 0
        self._t_start = 0.0
        self._t_stop = 0.0

    # ------------------------------------------------------------------
    def describe(self) -> str:
        return (
            f"BackgroundTraffic[{self.direction}](src=cuda:{self.src_device.index} "
            f"-> dst=cuda:{self.dst_device.index}, "
            f"buffer={self.buffer_bytes // (1024 * 1024)} MiB) | "
            f"{self.pattern.describe()}"
        )

    # ------------------------------------------------------------------
    def prepare(self) -> None:
        # Allocate src on the phantom GPU and dst on the decoder GPU. The
        # decoder process opens a context on the phantom GPU briefly to
        # allocate the src buffer; the context persists for the run.
        if self._src is None:
            with torch.cuda.device(self.src_device):
                self._src = torch.empty(
                    self.buffer_bytes, dtype=torch.uint8, device=self.src_device
                )
        if self._dst is None:
            with torch.cuda.device(self.dst_device):
                self._dst = torch.empty(
                    self.buffer_bytes, dtype=torch.uint8, device=self.dst_device
                )

    def start(self) -> None:
        self.prepare()
        self._stop.clear()
        self._ops = 0
        self._bytes = 0
        self.pattern.reset()
        self._t_start = time.monotonic()

        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name=f"bg-traffic-r{self.dst_device.index}-{self.direction}",
        )
        self._thread.start()

    def _loop(self) -> None:
        # Run the issuer on the *source* GPU so the CE engine and stream
        # match — same device as self._stream was created on.
        torch.cuda.set_device(self.src_device)
        while not self._stop.is_set():
            n = self.pattern.next_chunk_bytes(self._stop)
            if n is None or self._stop.is_set():
                return
            n = min(n, self.buffer_bytes)
            assert self._src is not None and self._dst is not None
            try:
                with torch.cuda.stream(self._stream):
                    # tensor.copy_ across distinct CUDA devices ->
                    # cudaMemcpyPeerAsync -> CE over NVLink.
                    self._dst[:n].copy_(self._src[:n], non_blocking=True)
            except RuntimeError:
                if self._stop.is_set():
                    return
                raise
            self._ops += 1
            self._bytes += n
            if (
                self.max_inflight_copies
                and self._ops % self.max_inflight_copies == 0
            ):
                self._stream.synchronize()

    def stop(self) -> dict[str, Any]:
        if self._thread is None:
            return {}
        self._stop.set()
        self._thread.join(timeout=5.0)
        try:
            with torch.cuda.device(self.src_device):
                self._stream.synchronize()
        except RuntimeError:
            pass
        # Drop buffer refs *before* vLLM tears down CUDA contexts to avoid
        # "terminate called without an active exception" at shutdown.
        self._src = None
        self._dst = None
        self._t_stop = time.monotonic()
        elapsed = max(self._t_stop - self._t_start, 1e-9)
        return {
            "direction": self.direction,
            "src_device": self.src_device.index,
            "dst_device": self.dst_device.index,
            "ops": self._ops,
            "bytes": self._bytes,
            "elapsed_s": elapsed,
            "achieved_gbps": (self._bytes / elapsed) / 1e9,
        }
