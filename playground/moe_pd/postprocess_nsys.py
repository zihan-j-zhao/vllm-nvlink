#!/usr/bin/env python3
"""Post-process nsys GPU-metrics SQLite exports from run_nsys_sweep.sh.

For each sweep cell (ISL/OSL pair) and each GPU, produces a figure with
stacked timeseries subplots showing resource utilisation over time.

The nsys ``gb10x`` metric set (auto-selected for B200 GPUs) reports all
values as **throughput %** of peak.  This script can plot them raw (as %)
or convert to absolute units (GB/s) using peak specs from the nsys
``TARGET_INFO_GPU`` table.

Default subplots (one figure per GPU):
    1. HBM Read + Write Bandwidth  (%  or  GB/s)
    2. NVLink TX + RX User Data    (% of peak NVLink BW)
    3. SM Activity + Tensor Core   (%)
    4. NVLink Protocol overhead    (%)   [with --all-metrics]

Usage
-----
    # Process full sweep (one figure per GPU per cell):
    python playground/moe_pd/postprocess_nsys.py <sweep_dir>

    # Single cell directory:
    python playground/moe_pd/postprocess_nsys.py <sweep_dir>/cell00_isl1024_osl128_c64

    # Show absolute GB/s instead of throughput %:
    python playground/moe_pd/postprocess_nsys.py <sweep_dir> --absolute

    # List all metrics in the sqlite:
    python playground/moe_pd/postprocess_nsys.py <cell_dir>/nsys_report.sqlite --dump-metrics

Inputs (per cell directory)
---------------------------
    nsys_report.sqlite   ``nsys export --type=sqlite`` output
    cell.json            timing + workload metadata from run_nsys_sweep.sh
    topology.json        prefill_gpus / decode_gpus  (optional)

nsys 2025 SQLite schema (B200 / GB10x)
---------------------------------------
    GPU_METRICS: rawTimestamp, timestamp, typeId, metricId, value
    TARGET_INFO_GPU_METRICS: typeId, sourceId, typeName, metricId, metricName
    TARGET_INFO_GPU: id, memoryBandwidth, ...
    GENERIC_EVENT_SOURCES: sourceId, ...

GPU identity is encoded as ``typeId = sourceId_base + gpu_index`` where
``sourceId_base`` comes from ``GENERIC_EVENT_SOURCES``.

All metric values are integers in the original nsys units:
    - Throughput % metrics: 0–100 (integer)
    - Clock frequencies: MHz
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
except ImportError as exc:
    print(f"error: matplotlib is required ({exc})", file=sys.stderr)
    raise

# ---------------------------------------------------------------------------
# Subplot definitions
# ---------------------------------------------------------------------------
# Each subplot can contain multiple traces (metric lines).  We match
# metrics by substring of the metricName from TARGET_INFO_GPU_METRICS.


@dataclass
class TraceDef:
    """One line within a subplot."""
    match: str          # substring to match against metricName
    label: str          # legend label
    color: str
    is_bandwidth: bool = False  # True → can be converted to GB/s


@dataclass
class SubplotDef:
    title: str
    traces: list[TraceDef]
    y_label_pct: str = "Throughput %"
    y_label_abs: str = "GB/s"
    default: bool = True   # included without --all-metrics


SUBPLOTS: list[SubplotDef] = [
    SubplotDef(
        title="HBM Bandwidth",
        traces=[
            TraceDef("DRAM Read Bandwidth", "HBM Read", "#1f77b4", is_bandwidth=True),
            TraceDef("DRAM Write Bandwidth", "HBM Write", "#ff7f0e", is_bandwidth=True),
        ],
    ),
    SubplotDef(
        title="NVLink User Data",
        traces=[
            TraceDef("NVLink TX Requests User Data", "TX Req User", "#2ca02c"),
            TraceDef("NVLink TX Responses User Data", "TX Resp User", "#98df8a"),
            TraceDef("NVLink RX Requests User Data", "RX Req User", "#d62728"),
            TraceDef("NVLink RX Responses User Data", "RX Resp User", "#ff9896"),
        ],
    ),
    SubplotDef(
        title="Compute Activity",
        traces=[
            TraceDef("SMs Active", "SMs Active", "#9467bd"),
            TraceDef("Tensor Active", "Tensor Active", "#e377c2"),
        ],
    ),
    SubplotDef(
        title="NVLink Protocol Overhead",
        traces=[
            TraceDef("NVLink TX Requests Protocol", "TX Req Proto", "#7f7f7f"),
            TraceDef("NVLink TX Responses Protocol", "TX Resp Proto", "#c7c7c7"),
            TraceDef("NVLink RX Requests Protocol", "RX Req Proto", "#bcbd22"),
            TraceDef("NVLink RX Responses Protocol", "RX Resp Proto", "#dbdb8d"),
        ],
        default=False,
    ),
    SubplotDef(
        title="PCIe Throughput",
        traces=[
            TraceDef("PCIe TX Throughput", "PCIe TX", "#17becf"),
            TraceDef("PCIe RX Throughput", "PCIe RX", "#9edae5"),
        ],
        default=False,
    ),
    SubplotDef(
        title="Clock Frequencies",
        y_label_pct="MHz",
        y_label_abs="MHz",
        traces=[
            TraceDef("GPC Clock Frequency", "GPC Clock", "#8c564b"),
            TraceDef("SYS Clock Frequency", "SYS Clock", "#c49c94"),
        ],
        default=False,
    ),
    SubplotDef(
        title="Warps in Flight",
        traces=[
            TraceDef("Compute Warps in Flight [Throughput %]", "Warps (Tput %)", "#e377c2"),
        ],
        default=False,
    ),
]


# ---------------------------------------------------------------------------
# nsys SQLite reader (2025.x schema)
# ---------------------------------------------------------------------------

class NsysSqlite:
    """Read GPU metrics from an nsys 2025.x SQLite export."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        cur = self.conn.cursor()

        # Validate tables exist.
        tables = self._list_tables(cur)
        if "GPU_METRICS" not in tables:
            raise RuntimeError(
                f"No GPU_METRICS table in {db_path}. "
                f"Tables: {sorted(tables)}"
            )

        # Build metricId → metricName map.
        self._metric_names: dict[int, str] = {}
        if "TARGET_INFO_GPU_METRICS" in tables:
            cur.execute(
                "SELECT DISTINCT metricId, metricName "
                "FROM TARGET_INFO_GPU_METRICS "
                "ORDER BY metricId"
            )
            for row in cur.fetchall():
                self._metric_names[int(row[0])] = str(row[1])

        # Build typeId → gpu_index map.
        # typeId = sourceId_base + gpu_index
        self._type_to_gpu: dict[int, int] = {}
        cur.execute("SELECT DISTINCT typeId FROM GPU_METRICS ORDER BY typeId")
        type_ids = [int(r[0]) for r in cur.fetchall()]
        if type_ids:
            base = min(type_ids)
            for tid in type_ids:
                self._type_to_gpu[tid] = tid - base

        # Read peak HBM bandwidth per GPU from TARGET_INFO_GPU.
        self._peak_hbm_bps: dict[int, float] = {}
        if "TARGET_INFO_GPU" in tables:
            cur.execute(
                "SELECT id, memoryBandwidth FROM TARGET_INFO_GPU "
                "ORDER BY id"
            )
            for row in cur.fetchall():
                self._peak_hbm_bps[int(row[0])] = float(row[1])

    @staticmethod
    def _list_tables(cur: sqlite3.Cursor) -> set[str]:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return {r[0] for r in cur.fetchall()}

    # --- public API --------------------------------------------------------

    def metric_names(self) -> dict[int, str]:
        """Return {metricId: metricName} for all known metrics."""
        return dict(self._metric_names)

    def gpu_ids(self) -> list[int]:
        """Return sorted list of GPU indices present in the data."""
        return sorted(set(self._type_to_gpu.values()))

    def peak_hbm_bps(self, gpu_id: int) -> float:
        """Peak HBM bandwidth in B/s for a GPU, or 0 if unknown."""
        return self._peak_hbm_bps.get(gpu_id, 0.0)

    def timeseries(
        self,
        gpu_id: int,
        metric_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (timestamps_ns, values) for one GPU and metric.

        Arrays are sorted by timestamp.
        """
        # Find typeIds for this gpu_id.
        type_ids = [
            tid for tid, gid in self._type_to_gpu.items()
            if gid == gpu_id
        ]
        if not type_ids:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

        cur = self.conn.cursor()
        placeholders = ",".join("?" * len(type_ids))
        cur.execute(
            f"SELECT timestamp, value FROM GPU_METRICS "
            f"WHERE typeId IN ({placeholders}) AND metricId = ? "
            f"ORDER BY timestamp",
            (*type_ids, metric_id),
        )
        rows = cur.fetchall()
        if not rows:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
        ts = np.array([r[0] for r in rows], dtype=np.int64)
        vals = np.array([r[1] for r in rows], dtype=np.float64)
        return ts, vals

    def close(self) -> None:
        self.conn.close()


def dump_schema(db_path: Path) -> None:
    """Print the full SQLite schema for debugging."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT name, type, sql FROM sqlite_master ORDER BY name")
    for name, kind, sql in cur.fetchall():
        if kind != "table":
            continue
        print(f"-- {kind}: {name}")
        if sql:
            print(sql)
        print()
    conn.close()


# ---------------------------------------------------------------------------
# Binning / resampling
# ---------------------------------------------------------------------------

def bin_timeseries(
    ts_ns: np.ndarray,
    values: np.ndarray,
    bin_ms: float | None = None,
    t0_ns: int | None = None,
    smooth_window: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw nsys samples to (t_s, y).

    If ``bin_ms`` is None (the default), returns the native nsys samples
    unchanged — nsys already controls the sample rate via
    ``--gpu-metrics-frequency`` and binning on top of that throws away
    real signal.  Use a non-None ``bin_ms`` only when you explicitly
    want to downsample (e.g. very long traces).

    ``smooth_window`` optionally applies a centred rolling-mean filter
    of that many samples on top — useful for the integer-quantised
    throughput-% metrics from the ``gb10x`` set.
    """
    if len(ts_ns) == 0:
        return np.array([]), np.array([])

    if t0_ns is None:
        t0_ns = int(ts_ns[0])

    if bin_ms is None:
        # Native: just translate timestamps to seconds-from-t0.
        t_s = (ts_ns - t0_ns) / 1e9
        y = values.astype(np.float64)
    else:
        bin_ns = int(bin_ms * 1e6)
        bin_idx = ((ts_ns - t0_ns) // bin_ns).astype(np.int64)
        lo, hi = int(bin_idx.min()), int(bin_idx.max())
        n_bins = hi - lo + 1

        sums = np.zeros(n_bins, dtype=np.float64)
        counts = np.zeros(n_bins, dtype=np.int64)
        for i, bi in enumerate(bin_idx):
            j = int(bi) - lo
            if 0 <= j < n_bins:
                v = values[i]
                if not np.isnan(v):
                    sums[j] += v
                    counts[j] += 1

        mask = counts > 0
        t_s = (np.arange(lo, hi + 1) + 0.5) * bin_ns / 1e9
        y = np.where(mask, sums / np.maximum(counts, 1), np.nan)

    if smooth_window and smooth_window > 1 and len(y) > smooth_window:
        # Centred rolling mean via cumsum trick; pad with NaN where the
        # window doesn't fully overlap.
        w = int(smooth_window)
        # Replace NaNs with 0 for the cumulative-sum pass, then track
        # the valid-sample count separately to renormalise.
        valid = (~np.isnan(y)).astype(np.float64)
        yz = np.where(np.isnan(y), 0.0, y)
        csum = np.concatenate(([0.0], np.cumsum(yz)))
        cvalid = np.concatenate(([0.0], np.cumsum(valid)))
        win_sum = csum[w:] - csum[:-w]
        win_cnt = cvalid[w:] - cvalid[:-w]
        with np.errstate(invalid="ignore", divide="ignore"):
            y_sm = win_sum / np.where(win_cnt > 0, win_cnt, np.nan)
        # Centre the smoothed array within the original.
        pad_lo = (len(y) - len(y_sm)) // 2
        out = np.full_like(y, np.nan, dtype=np.float64)
        out[pad_lo:pad_lo + len(y_sm)] = y_sm
        y = out

    return t_s, y


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _format_time(x: float, _pos: Any) -> str:
    if x >= 60:
        return f"{int(x)//60}:{int(x)%60:02d}"
    return f"{x:.0f}"


def plot_gpu_figure(
    gpu_id: int,
    role: str,
    subplot_data: list[tuple[SubplotDef, list[tuple[TraceDef, np.ndarray, np.ndarray]]]],
    cell_tag: str,
    out_path: Path,
    absolute: bool = False,
    peak_hbm_gbps: float = 0.0,
) -> None:
    """Plot one figure for a GPU with stacked subplots."""
    n = len(subplot_data)
    if n == 0:
        return

    fig, axes = plt.subplots(
        n, 1,
        figsize=(14, 2.5 * n + 1.2),
        sharex=True,
        squeeze=False,
    )
    fig.suptitle(
        f"GPU {gpu_id} ({role})  —  {cell_tag}",
        fontsize=13,
        fontweight="bold",
    )

    for idx, (sp_def, traces) in enumerate(subplot_data):
        ax = axes[idx, 0]
        for tr_def, t_s, y in traces:
            plot_y = y.copy()
            # Convert throughput % → GB/s for HBM bandwidth metrics.
            if absolute and tr_def.is_bandwidth and peak_hbm_gbps > 0:
                plot_y = y * peak_hbm_gbps / 100.0

            ax.plot(t_s, plot_y, linewidth=0.7, color=tr_def.color,
                    alpha=0.85, label=tr_def.label)
            if tr_def.is_bandwidth or "NVLink" in sp_def.title:
                ax.fill_between(t_s, 0, plot_y, alpha=0.1, color=tr_def.color)

        if absolute and any(td.is_bandwidth for td, _, _ in traces) and peak_hbm_gbps > 0:
            ylabel = sp_def.y_label_abs
        else:
            ylabel = sp_def.y_label_pct
        ax.set_ylabel(f"{sp_def.title}\n({ylabel})", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        if len(traces) > 1:
            ax.legend(fontsize=7, loc="upper right", ncol=min(len(traces), 4))

    axes[-1, 0].set_xlabel("Time (s)", fontsize=10)
    axes[-1, 0].xaxis.set_major_formatter(FuncFormatter(_format_time))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {out_path.name}")


# ---------------------------------------------------------------------------
# Cell processing
# ---------------------------------------------------------------------------

def _find_metric_id(
    metric_names: dict[int, str],
    match_substr: str,
) -> int | None:
    """Find the metricId whose name contains match_substr."""
    for mid, name in metric_names.items():
        if match_substr in name:
            return mid
    return None


def process_cell(
    cell_dir: Path,
    *,
    cli_prefill: list[int] | None,
    cli_decode: list[int] | None,
    bin_ms: float | None,
    smooth_window: int,
    skip_warmup: bool,
    all_metrics: bool,
    absolute: bool,
    out_dir: Path | None,
) -> bool:
    """Process one sweep cell.  Returns True if successful."""
    cell_json = cell_dir / "cell.json"
    if not cell_json.is_file():
        print(f"[postprocess] skip {cell_dir.name}: no cell.json",
              file=sys.stderr)
        return False

    meta = json.loads(cell_json.read_text())
    cell_tag = meta.get("cell_tag", cell_dir.name)

    # Find nsys sqlite.
    sqlite_path = cell_dir / "nsys_report.sqlite"
    if not sqlite_path.is_file():
        candidates = list(cell_dir.glob("*.sqlite"))
        sqlite_path = candidates[0] if candidates else None
    if sqlite_path is None or not sqlite_path.is_file():
        print(f"[postprocess] skip {cell_tag}: no .sqlite in {cell_dir}",
              file=sys.stderr)
        return False

    # Topology.
    prefill_gpus = cli_prefill
    decode_gpus = cli_decode
    topo_path = cell_dir / "topology.json"
    if topo_path.is_file():
        topo = json.loads(topo_path.read_text())
        if prefill_gpus is None:
            prefill_gpus = list(topo.get("prefill_gpus", []))
        if decode_gpus is None:
            decode_gpus = list(topo.get("decode_gpus", []))
    prefill_set = set(prefill_gpus or [])
    decode_set = set(decode_gpus or [])

    try:
        db = NsysSqlite(sqlite_path)
    except Exception as exc:
        print(f"[postprocess] skip {cell_tag}: {exc}", file=sys.stderr)
        return False

    metric_names = db.metric_names()
    if not metric_names:
        print(f"[postprocess] skip {cell_tag}: no metrics", file=sys.stderr)
        db.close()
        return False

    print(f"[postprocess] {cell_tag}: {len(metric_names)} metrics, "
          f"GPUs {db.gpu_ids()}")

    # Determine global t0 across all GPUs.
    t0_ns: int | None = None
    for gid in db.gpu_ids():
        for mid in metric_names:
            ts, _ = db.timeseries(gid, mid)
            if len(ts) > 0:
                t0_ns = min(t0_ns, int(ts[0])) if t0_ns is not None else int(ts[0])
            break  # only need one metric per GPU to find earliest ts

    if t0_ns is None:
        print(f"[postprocess] skip {cell_tag}: no data", file=sys.stderr)
        db.close()
        return False

    warmup_s = float(meta.get("warmup_s", 0)) if skip_warmup else 0.0
    filter_gpus = set(meta.get("gpu_ids", db.gpu_ids()))

    active_subplots = [sp for sp in SUBPLOTS if all_metrics or sp.default]

    plot_dir = out_dir or (cell_dir / "plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    for gid in db.gpu_ids():
        if gid not in filter_gpus:
            continue
        role = ("prefill" if gid in prefill_set
                else "decode" if gid in decode_set
                else "gpu")

        peak_hbm_gbps = db.peak_hbm_bps(gid) / 1e9  # B/s → GB/s

        subplot_data: list[tuple[SubplotDef, list[tuple[TraceDef, np.ndarray, np.ndarray]]]] = []

        for sp_def in active_subplots:
            traces: list[tuple[TraceDef, np.ndarray, np.ndarray]] = []
            for tr_def in sp_def.traces:
                mid = _find_metric_id(metric_names, tr_def.match)
                if mid is None:
                    continue
                ts, vals = db.timeseries(gid, mid)
                if len(ts) == 0:
                    continue
                t_s, y = bin_timeseries(
                    ts, vals,
                    bin_ms=bin_ms,
                    t0_ns=t0_ns,
                    smooth_window=smooth_window,
                )
                if warmup_s > 0 and len(t_s) > 0:
                    mask = t_s >= warmup_s
                    t_s, y = t_s[mask], y[mask]
                if len(t_s) > 0:
                    traces.append((tr_def, t_s, y))
            if traces:
                subplot_data.append((sp_def, traces))

        if subplot_data:
            fname = f"gpu{gid}_{role}_{cell_tag}.png"
            plot_gpu_figure(
                gid, role, subplot_data, cell_tag, plot_dir / fname,
                absolute=absolute, peak_hbm_gbps=peak_hbm_gbps,
            )

    db.close()
    return True


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def write_summary_csv(
    sweep_dir: Path,
    cell_dirs: list[Path],
    *,
    cli_prefill: list[int] | None,
    cli_decode: list[int] | None,
    bin_ms: float | None,
) -> None:
    """Write a tidy CSV with mean metric values per cell per GPU."""
    rows: list[dict] = []

    for cell_dir in cell_dirs:
        cell_json = cell_dir / "cell.json"
        if not cell_json.is_file():
            continue
        meta = json.loads(cell_json.read_text())
        cell_tag = meta.get("cell_tag", cell_dir.name)

        sqlite_path = cell_dir / "nsys_report.sqlite"
        if not sqlite_path.is_file():
            candidates = list(cell_dir.glob("*.sqlite"))
            sqlite_path = candidates[0] if candidates else None
        if sqlite_path is None or not sqlite_path.is_file():
            continue

        prefill_gpus = cli_prefill
        decode_gpus = cli_decode
        topo_path = cell_dir / "topology.json"
        if topo_path.is_file():
            topo = json.loads(topo_path.read_text())
            if prefill_gpus is None:
                prefill_gpus = list(topo.get("prefill_gpus", []))
            if decode_gpus is None:
                decode_gpus = list(topo.get("decode_gpus", []))
        prefill_set = set(prefill_gpus or [])
        decode_set = set(decode_gpus or [])

        try:
            db = NsysSqlite(sqlite_path)
        except Exception:
            continue

        metric_names = db.metric_names()
        warmup_s = float(meta.get("warmup_s", 0))

        for gid in db.gpu_ids():
            role = ("prefill" if gid in prefill_set
                    else "decode" if gid in decode_set
                    else "unknown")
            peak_hbm_gbps = db.peak_hbm_bps(gid) / 1e9
            row: dict = {
                "cell_tag": cell_tag,
                "isl": meta.get("isl_mean"),
                "osl": meta.get("osl_mean"),
                "mode": meta.get("mode"),
                "concurrency": meta.get("concurrency"),
                "request_rate": meta.get("request_rate"),
                "gpu_id": gid,
                "role": role,
                "peak_hbm_gbps": f"{peak_hbm_gbps:.1f}",
            }
            for mid, mname in metric_names.items():
                ts, vals = db.timeseries(gid, mid)
                if len(ts) == 0:
                    row[mname] = ""
                    continue
                t_s, y = bin_timeseries(ts, vals, bin_ms=bin_ms)
                if warmup_s > 0 and len(t_s) > 0:
                    mask = t_s >= warmup_s
                    y = y[mask]
                row[mname] = f"{float(np.nanmean(y)):.2f}" if len(y) > 0 else ""
            rows.append(row)
        db.close()

    if not rows:
        return
    out_path = sweep_dir / "summary_nsys.csv"
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[postprocess] summary CSV -> {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_int_csv(s: str | None) -> list[int] | None:
    if s is None:
        return None
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Post-process nsys GPU metrics into per-GPU timeseries plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "path", type=Path,
        help="Sweep root dir (cellNN_* subdirs), a single cell dir, "
             "or a .sqlite file.",
    )
    ap.add_argument("--bin-ms", type=float, default=None,
                    help="Time-bin width in ms. Default: None (plot at "
                         "the native nsys sample rate; use a value here "
                         "only to downsample very long traces).")
    ap.add_argument("--smooth", type=int, default=0, metavar="N",
                    help="Centred rolling-mean window in samples (e.g. "
                         "--smooth 50 averages 50 native samples ≈ 50 ms "
                         "at 1 kHz nsys). 0 disables smoothing (default).")
    ap.add_argument("--no-warmup-skip", action="store_true",
                    help="Do not skip the warmup period from cell.json.")
    ap.add_argument("--all-metrics", action="store_true",
                    help="Plot all metric groups (adds NVLink protocol, "
                         "PCIe, clocks, warps).")
    ap.add_argument("--absolute", action="store_true",
                    help="Convert HBM throughput %% to GB/s using peak from "
                         "TARGET_INFO_GPU.")
    ap.add_argument("--prefill-gpus", type=str, default=None,
                    help="Comma-separated GPU ids (overrides topology.json).")
    ap.add_argument("--decode-gpus", type=str, default=None,
                    help="Comma-separated GPU ids (overrides topology.json).")
    ap.add_argument("-o", "--output-dir", type=Path, default=None,
                    help="Override plot output directory.")
    ap.add_argument("--dump-schema", action="store_true",
                    help="Print SQLite schema and exit.")
    ap.add_argument("--dump-metrics", action="store_true",
                    help="Print all GPU metric names and exit.")
    ap.add_argument("--no-summary-csv", action="store_true",
                    help="Skip writing summary_nsys.csv.")
    args = ap.parse_args()
    path: Path = args.path
    cli_prefill = _parse_int_csv(args.prefill_gpus)
    cli_decode = _parse_int_csv(args.decode_gpus)

    # Direct sqlite file.
    if path.suffix == ".sqlite" and path.is_file():
        if args.dump_schema:
            dump_schema(path)
            return 0
        if args.dump_metrics:
            db = NsysSqlite(path)
            for mid, name in db.metric_names().items():
                print(f"  {mid:3d}  {name}")
            db.close()
            return 0
        cell_dir = path.parent
        process_cell(
            cell_dir,
            cli_prefill=cli_prefill, cli_decode=cli_decode,
            bin_ms=args.bin_ms, smooth_window=args.smooth,
            skip_warmup=not args.no_warmup_skip,
            all_metrics=args.all_metrics, absolute=args.absolute,
            out_dir=args.output_dir,
        )
        return 0

    if not path.is_dir():
        print(f"error: not a directory or .sqlite file: {path}",
              file=sys.stderr)
        return 2

    # Single cell dir or sweep root?
    if (path / "cell.json").is_file():
        cell_dirs = [path]
        sweep_dir = path.parent
    else:
        cell_dirs = sorted(
            p for p in path.iterdir()
            if p.is_dir() and (p / "cell.json").is_file()
        )
        sweep_dir = path

    if not cell_dirs:
        print(f"error: no cells found under {path}", file=sys.stderr)
        return 1

    if args.dump_schema:
        for cd in cell_dirs:
            for sq in cd.glob("*.sqlite"):
                print(f"=== {sq} ===")
                dump_schema(sq)
                return 0
        print("error: no .sqlite files", file=sys.stderr)
        return 1

    if args.dump_metrics:
        for cd in cell_dirs:
            for sq in cd.glob("*.sqlite"):
                db = NsysSqlite(sq)
                mn = db.metric_names()
                print(f"=== {sq.name} ({len(mn)} metrics) ===")
                for mid, name in mn.items():
                    print(f"  {mid:3d}  {name}")
                db.close()
                return 0
        print("error: no .sqlite files", file=sys.stderr)
        return 1

    ok = 0
    for cd in cell_dirs:
        if process_cell(
            cd,
            cli_prefill=cli_prefill, cli_decode=cli_decode,
            bin_ms=args.bin_ms, smooth_window=args.smooth,
            skip_warmup=not args.no_warmup_skip,
            all_metrics=args.all_metrics, absolute=args.absolute,
            out_dir=args.output_dir,
        ):
            ok += 1

    if not args.no_summary_csv:
        write_summary_csv(
            sweep_dir, cell_dirs,
            cli_prefill=cli_prefill, cli_decode=cli_decode,
            bin_ms=args.bin_ms,
        )

    print(f"\n[postprocess] done. {ok}/{len(cell_dirs)} cells processed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
