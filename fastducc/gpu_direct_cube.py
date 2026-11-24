# -*- coding: utf-8 -*-
"""
fastducc.gpu_direct_cube: Double-buffered pinned-host MS streaming and direct
in-VRAM Hermitian half-plane rasterisation into consolidated CSR stores.

Architecture:
  1. Single metadata pre-scan over Measurement Set(s) using ``python-casacore``
     to inspect timestamps, scans, baselines, channel frequencies, correlation
     mappings, phase centre, ``WEIGHT_SPECTRUM`` availability, and row ordering.
  2. Time-grid analysis or regularisation (:func:`fastducc.timeshift.regularize_time_grid`
     or :func:`fastducc.cube.analyze_time_axis`), integer channel dispersion
     delays (:func:`fastducc.dedisp.get_dm_delays`), and scan-aware timestamp-aligned
     chunk partitioning (default ``row_chunk_size=300_000``, clamping
     ``nt_emit <= 4096`` so ``uint16`` fragment counts never overflow).
  3. On CUDA: two pinned host buffer pools (``cp.cuda.alloc_pinned_memory``)
     fed asynchronously by a background Casacore I/O thread while ``h2d_stream``
     and ``compute_stream`` transfer, stage (:func:`stage_ms_rows_gpu`),
     optionally shift sub-sample block offsets on GPU, maintain the causal
     dedispersion lookahead buffer across chunks within each scan, rasterise
     (:func:`dedisperse_and_rasterise_chunk_gpu` with direct-to-sparse output),
     and pack active UV samples in VRAM (:func:`pack_chunk_to_ragged_gpu`).
  4. Zero-fragment VRAM consolidation via
     :func:`fastducc.ragged_gpu.consolidate_device_chunks_to_gpu_store`,
     returning a :class:`GPURaggedDGAStore` (and optional
     ``cupyx.scipy.sparse.csr_matrix``) without writing intermediate
     ``.csr.bin`` fragment files.
  5. Complete CPU fallback executing identical chunking and lookahead logic
     via :func:`stage_ms_rows_cpu`, :func:`dedisperse_and_rasterise_chunk_cpu`,
     and :func:`pack_chunk_to_ragged_csr`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import logging
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import cupy as cp

    HAS_CUPY = True
    _CUPY_IMPORT_ERROR: Optional[BaseException] = None
except Exception as exc:
    cp = None  # type: ignore[assignment]
    HAS_CUPY = False
    _CUPY_IMPORT_ERROR = exc

try:
    from casacore.tables import table
except ImportError:
    table = None

from fastducc import cube as fd_cube
from fastducc import dedisp, ms_utils, timeshift
from fastducc.gpu_nufft import (
    build_gpu_csr_matrix,
    from_gpu_buffers,
    resolve_device as _resolve_device,
    write_ragged_store_to_disk,
)
from fastducc.gpu_raster import (
    _normalise_mode,
    _resolve_normalise_mode,
    compute_raster_grid_geometry,
    dedisperse_active_cell_matrix_cpu,
    dedisperse_active_cell_matrix_gpu,
    dedisperse_and_rasterise_chunk_cpu,
    dedisperse_and_rasterise_chunk_gpu,
    dedisperse_and_rasterise_subband_cpu,
    dedisperse_and_rasterise_subband_gpu,
    stage_ms_rows_cpu,
    stage_ms_rows_gpu,
)
from fastducc.ragged import (
    RaggedDGAStore,
    _scatter_fragment_samples_numba,
    pack_chunk_to_ragged_csr,
)
from fastducc.ragged_gpu import pack_chunk_to_ragged_gpu, pack_uv_chunk_to_ragged_gpu

logger = logging.getLogger("fastducc.gpu_direct_cube")

_MAX_NT_EMIT = 4096
_SHIFT_HALO_SAMPLES = 8


def _require_casacore() -> None:
    if table is None:
        raise ImportError(
            "python-casacore is required for streaming Measurement Sets in gpu_direct_cube."
        )


@dataclass
class DirectRasterResult:
    """
    Container returned by :func:`stream_ms_to_ragged_csr` supporting both
    attribute access and mapping-style indexing.
    """

    gpu_store: Optional[Any]
    store: Optional[RaggedDGAStore]
    csr_matrix: Optional[Any]
    attrs: Dict[str, Any]
    time_info: Dict[str, Any]
    io_s: float
    gpu_s: float
    consolidate_s: float
    total_s: float
    nrows: int
    m_total: int
    nnz_total: int
    device: str

    def to_csr_matrix(self, validate: bool = False) -> Any:
        """
        Return the cached ``cupyx.scipy.sparse.csr_matrix`` or build it on demand
        from ``gpu_store`` (or ``store`` if on CPU with CUDA available).
        """
        if self.csr_matrix is not None:
            return self.csr_matrix
        src = self.gpu_store if self.gpu_store is not None else self.store
        if src is None:
            raise RuntimeError("No CSR store available to convert to cupyx csr_matrix.")
        total_planes = int(self.attrs.get("total_planes", self.time_info.get("total_planes", 0)))
        dt = float(self.attrs.get("tsamp", self.time_info.get("dt_nominal", 1.0)))
        self.csr_matrix = build_gpu_csr_matrix(src, total_planes=total_planes, dt=dt, validate=validate)
        return self.csr_matrix

    def _as_dict(self) -> Dict[str, Any]:
        return {
            "gpu_store": self.gpu_store,
            "store": self.store,
            "csr_matrix": self.csr_matrix,
            "attrs": self.attrs,
            "time_info": self.time_info,
            "io_s": self.io_s,
            "gpu_s": self.gpu_s,
            "consolidate_s": self.consolidate_s,
            "total_s": self.total_s,
            "nrows": self.nrows,
            "m_total": self.m_total,
            "nnz_total": self.nnz_total,
            "device": self.device,
        }

    def __getitem__(self, key: str) -> Any:
        return self._as_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._as_dict().get(key, default)

    def keys(self):
        return self._as_dict().keys()

    def __contains__(self, key: object) -> bool:
        return key in self._as_dict()

    def __iter__(self) -> Iterator[str]:
        return iter(self._as_dict())


@dataclass
class _MSMetaEntry:
    ms_idx: int
    ms_path: str
    nrows: int
    time_col: str
    has_scan: bool
    has_flag_row: bool
    has_weight_spectrum: bool
    is_time_sorted: bool
    t_vals: np.ndarray
    scan_vals: np.ndarray
    bl_keys: np.ndarray
    t_min: float
    ts_starts: Optional[np.ndarray] = None
    ts_ends: Optional[np.ndarray] = None
    is_uniform_bl: bool = False
    nbl_step: int = 0
    bl_keys_step: Optional[np.ndarray] = None


@dataclass
class _MSReadSegment:
    ms_idx: int
    ms_path: str
    startrow: int
    nrow: int
    row_ids: Optional[np.ndarray]
    bl_idx: np.ndarray
    t_rel: np.ndarray
    has_flag_row: bool
    has_weight_spectrum: bool
    time_col: str


@dataclass
class _ChunkSpec:
    chunk_idx: int
    block_idx: int
    shift_samps: float
    t_idx_start: int
    t_idx_end: int
    nt_new: int
    is_last_in_block: bool
    segments: List[_MSReadSegment] = field(default_factory=list)
    total_rows: int = 0


def _normalise_ms_list(msname: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]]) -> List[str]:
    if isinstance(msname, (str, os.PathLike)):
        paths = [str(msname)]
    else:
        paths = [str(p) for p in msname]
    if not paths:
        raise ValueError("At least one Measurement Set path must be provided.")
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Measurement Set '{p}' not found.")
    return [str(Path(p).resolve()) for p in paths]


def _resolve_spw_freqs(ms_path: str, spw: int, chan_sel: Optional[Tuple[int, Optional[int]]]) -> np.ndarray:
    spw_path = os.path.join(ms_path, "SPECTRAL_WINDOW")
    with table(spw_path, readonly=True, ack=False) as t_spw:
        is_empty = int(t_spw.nrows()) == 0
    if is_empty and ms_path.endswith(".uvsub.ms"):
        alt_ms = ms_path[:-9] + ".ms"
        if os.path.exists(os.path.join(alt_ms, "SPECTRAL_WINDOW")):
            spw_path = os.path.join(alt_ms, "SPECTRAL_WINDOW")
    with table(spw_path, readonly=True, ack=False) as t_spw:
        if int(spw) >= int(t_spw.nrows()):
            raise ValueError(f"SPW {spw} out of range for '{spw_path}' (nrows={t_spw.nrows()}).")
        freqs_full = np.asarray(t_spw.getcell("CHAN_FREQ", int(spw)), dtype=np.float64).ravel()
    if chan_sel is not None:
        c0, c1 = chan_sel
        return np.ascontiguousarray(freqs_full[c0:c1], dtype=np.float64)
    return np.ascontiguousarray(freqs_full, dtype=np.float64)


def _check_weight_spectrum_usable(t_main: Any, colnames: set[str], use_weight_spectrum: bool) -> bool:
    if not use_weight_spectrum or "WEIGHT_SPECTRUM" not in colnames:
        return False
    try:
        nrows = int(t_main.nrows())
        if nrows <= 0:
            return False
        if hasattr(t_main, "iscelldefined") and not t_main.iscelldefined("WEIGHT_SPECTRUM", 0):
            return False
        cell = t_main.getcell("WEIGHT_SPECTRUM", 0)
        if cell is None or np.asarray(cell).ndim != 2:
            return False
        if "WEIGHT" in colnames:
            n_sample = min(128, nrows)
            ws_0 = np.asarray(t_main.getcol("WEIGHT_SPECTRUM", startrow=0, nrow=n_sample), dtype=np.float32)
            w_0 = np.asarray(t_main.getcol("WEIGHT", startrow=0, nrow=n_sample), dtype=np.float32)
            if w_0.ndim == 1:
                w_0 = w_0[:, None]
            if ws_0.ndim == 3 and ws_0.shape[0] == w_0.shape[0] and ws_0.shape[2] == w_0.shape[1]:
                if float(np.max(np.abs(ws_0 - w_0[:, None, :]))) < 1e-6:
                    mid = max(0, (nrows // 2) - (n_sample // 2))
                    ws_m = np.asarray(t_main.getcol("WEIGHT_SPECTRUM", startrow=mid, nrow=n_sample), dtype=np.float32)
                    w_m = np.asarray(t_main.getcol("WEIGHT", startrow=mid, nrow=n_sample), dtype=np.float32)
                    if w_m.ndim == 1:
                        w_m = w_m[:, None]
                    if float(np.max(np.abs(ws_m - w_m[:, None, :]))) < 1e-6:
                        return False
        return True
    except Exception:
        return False


def _prescan_ms_metadata(
    ms_paths: List[str],
    *,
    spw: int,
    chan_sel: Optional[Tuple[int, Optional[int]]],
    data_column: str,
    use_weight_spectrum: bool,
    field_name: Optional[str],
    start_time_idx: Optional[int],
    end_time_idx: Optional[int],
    start_time: Optional[float],
    end_time: Optional[float],
    time_step: Optional[float],
    time_tolerance: float,
    max_blank_factor: float,
    regularize_time: bool,
    sync_tolerance: float,
) -> Dict[str, Any]:
    """
    Open each Measurement Set once to inspect columns, ordering, baselines,
    spectral window frequencies, correlation labels, phase centre, and time axis.
    """
    _require_casacore()
    entries: List[_MSMetaEntry] = []
    chan_freq: Optional[np.ndarray] = None
    nchan_full: Optional[int] = None
    ncorr: Optional[int] = None
    labels: Optional[List[str]] = None
    lbl2idx: Optional[Dict[str, int]] = None

    for raw_idx, ms_path in enumerate(ms_paths):
        with table(ms_path, readonly=True, ack=False) as t_main:
            nrows = int(t_main.nrows())
            if nrows == 0:
                continue
            colnames = set(t_main.colnames())
            if data_column not in colnames:
                raise KeyError(f"Column '{data_column}' not found in Measurement Set '{ms_path}'.")
            time_col = "TIME_CENTROID" if "TIME_CENTROID" in colnames else "TIME"
            has_scan = "SCAN_NUMBER" in colnames
            has_flag_row = "FLAG_ROW" in colnames
            has_wgt_spec = _check_weight_spectrum_usable(t_main, colnames, use_weight_spectrum)

            t_vals = np.asarray(t_main.getcol(time_col), dtype=np.float64)
            is_sorted = bool(nrows <= 1 or np.all(t_vals[1:] >= t_vals[:-1]))
            t_min = float(t_vals[0]) if is_sorted else float(np.min(t_vals))

            ts_starts: Optional[np.ndarray] = None
            ts_ends: Optional[np.ndarray] = None
            is_uniform_bl = False
            nbl_step = 0
            bl_keys_step: Optional[np.ndarray] = None

            if is_sorted:
                chg = np.flatnonzero(t_vals[1:] != t_vals[:-1]) + 1
                ts_starts = np.empty(chg.size + 1, dtype=np.int64)
                ts_starts[0] = 0
                ts_starts[1:] = chg
                ts_ends = np.empty(chg.size + 1, dtype=np.int64)
                ts_ends[:-1] = chg
                ts_ends[-1] = nrows

                n_ts_ent = int(ts_starts.size)
                nbl_0 = int(ts_ends[0] - ts_starts[0]) if n_ts_ent > 0 else 0
                if nbl_0 > 0 and nrows == n_ts_ent * nbl_0 and bool(np.all((ts_ends - ts_starts) == nbl_0)):
                    sample_ts = sorted({0, n_ts_ent // 2, n_ts_ent - 1})
                    sample_bls = []
                    for ti_s in sample_ts:
                        sr_s = int(ts_starts[ti_s])
                        a1_s = np.asarray(t_main.getcol("ANTENNA1", startrow=sr_s, nrow=nbl_0), dtype=np.int64)
                        a2_s = np.asarray(t_main.getcol("ANTENNA2", startrow=sr_s, nrow=nbl_0), dtype=np.int64)
                        sample_bls.append((a1_s << 32) | (a2_s & 0xFFFFFFFF))
                    if all(np.array_equal(sample_bls[0], b) for b in sample_bls[1:]) and (
                        np.unique(sample_bls[0]).size == nbl_0
                    ):
                        is_uniform_bl = True
                        nbl_step = nbl_0
                        bl_keys_step = sample_bls[0]

            if is_uniform_bl and bl_keys_step is not None and ts_starts is not None:
                bl_keys = bl_keys_step
                if has_scan:
                    scan_vals = np.asarray(
                        t_main.getcol("SCAN_NUMBER", startrow=0, nrow=nrows, rowincr=nbl_step),
                        dtype=np.int64,
                    )
                else:
                    scan_vals = np.zeros(ts_starts.size, dtype=np.int64)
            elif is_sorted and ts_starts is not None and int(ts_starts.size) > 0:
                if has_scan:
                    scan_all = np.asarray(t_main.getcol("SCAN_NUMBER"), dtype=np.int64)
                    scan_vals = scan_all[ts_starts]
                    del scan_all
                else:
                    scan_vals = np.zeros(ts_starts.size, dtype=np.int64)
                a1 = np.asarray(t_main.getcol("ANTENNA1"), dtype=np.int64)
                a2 = np.asarray(t_main.getcol("ANTENNA2"), dtype=np.int64)
                bl_keys = (a1 << 32) | (a2 & 0xFFFFFFFF)
                del a1, a2
            else:
                scan_vals = (
                    np.asarray(t_main.getcol("SCAN_NUMBER"), dtype=np.int64)
                    if has_scan
                    else np.zeros(nrows, dtype=np.int64)
                )
                a1 = np.asarray(t_main.getcol("ANTENNA1"), dtype=np.int64)
                a2 = np.asarray(t_main.getcol("ANTENNA2"), dtype=np.int64)
                bl_keys = (a1 << 32) | (a2 & 0xFFFFFFFF)
                del a1, a2

            sample_cell = np.asarray(t_main.getcell(data_column, 0))
            if sample_cell.ndim == 1:
                cell_nchan, cell_ncorr = int(sample_cell.shape[0]), 1
            elif sample_cell.ndim == 2:
                cell_nchan, cell_ncorr = int(sample_cell.shape[0]), int(sample_cell.shape[1])
            else:
                raise ValueError(
                    f"Unexpected {data_column} cell shape {sample_cell.shape} in '{ms_path}'."
                )

        ms_freqs = _resolve_spw_freqs(ms_path, spw=spw, chan_sel=chan_sel)
        if chan_freq is None:
            chan_freq = ms_freqs
            nchan_full = cell_nchan
            ncorr = cell_ncorr
            try:
                labels, lbl2idx = ms_utils.get_corr_label_indices(ms_path)
            except Exception:
                labels = [str(i) for i in range(ncorr)]
                lbl2idx = {lab: i for i, lab in enumerate(labels)}
        else:
            if len(ms_freqs) != len(chan_freq) or cell_nchan != nchan_full or cell_ncorr != ncorr:
                raise ValueError(
                    f"Inconsistent SPW/correlation shape across MS list: '{ms_path}' "
                    f"({cell_nchan}x{cell_ncorr}) vs expected ({nchan_full}x{ncorr})."
                )

        entries.append(
            _MSMetaEntry(
                ms_idx=raw_idx,
                ms_path=ms_path,
                nrows=nrows,
                time_col=time_col,
                has_scan=has_scan,
                has_flag_row=has_flag_row,
                has_weight_spectrum=has_wgt_spec,
                is_time_sorted=is_sorted,
                t_vals=t_vals,
                scan_vals=scan_vals,
                bl_keys=bl_keys,
                t_min=t_min,
                ts_starts=ts_starts,
                ts_ends=ts_ends,
                is_uniform_bl=is_uniform_bl,
                nbl_step=nbl_step,
                bl_keys_step=bl_keys_step,
            )
        )

    if not entries or chan_freq is None or nchan_full is None or ncorr is None:
        raise RuntimeError(f"No valid visibility rows found in Measurement Set(s): {ms_paths}")

    entries.sort(key=lambda e: e.t_min)
    for new_i, ent in enumerate(entries):
        ent.ms_idx = new_i

    first_ms = entries[0].ms_path
    ra0_rad, dec0_rad, used_field = ms_utils.get_phase_center(first_ms, field_name)

    all_bl_keys = np.concatenate([e.bl_keys for e in entries])
    unique_bl_keys = np.unique(all_bl_keys)
    nbl = int(unique_bl_keys.size)
    del all_bl_keys

    # Fast-path when all MS entries are time-sorted and disjoint in time
    all_sorted_disjoint = all(e.is_time_sorted and e.ts_starts is not None for e in entries) and all(
        float(entries[i].t_vals[-1]) < float(entries[i + 1].t_vals[0]) for i in range(len(entries) - 1)
    )
    if all_sorted_disjoint:
        unique_times = np.concatenate([e.t_vals[e.ts_starts] for e in entries])
        scan_per_time = np.concatenate([e.scan_vals for e in entries])
    else:
        all_times = np.concatenate([e.t_vals for e in entries])
        all_scans = np.concatenate(
            [
                (
                    np.repeat(e.scan_vals, e.nbl_step)
                    if e.is_uniform_bl
                    else (
                        np.repeat(e.scan_vals, e.ts_ends - e.ts_starts)
                        if (e.is_time_sorted and e.ts_starts is not None and e.ts_ends is not None)
                        else e.scan_vals
                    )
                )
                for e in entries
            ]
        )
        unique_times, u_idx = np.unique(all_times, return_index=True)
        scan_per_time = all_scans[u_idx]
        del all_times, all_scans

    start_idx, end_idx = fd_cube.resolve_time_bounds(
        unique_times,
        start_time_idx=start_time_idx,
        end_time_idx=end_time_idx,
        start_time=start_time,
        end_time=end_time,
    )
    sel_times = np.ascontiguousarray(unique_times[start_idx : end_idx + 1], dtype=np.float64)
    sel_scans = np.ascontiguousarray(scan_per_time[start_idx : end_idx + 1], dtype=np.int64)

    if regularize_time:
        time_info = timeshift.regularize_time_grid(
            sel_times,
            dt_nominal=time_step,
            sync_tolerance=sync_tolerance,
            scan_numbers=sel_scans,
            max_blank_factor=max_blank_factor,
        )
    else:
        time_info = fd_cube.analyze_time_axis(
            sel_times,
            create_blanks=True,
            dt_nominal=time_step,
            tolerance=time_tolerance,
            max_blank_factor=max_blank_factor,
        )
        blocks, _ = timeshift.identify_time_blocks(
            sel_times,
            scan_numbers=sel_scans,
            dt_nominal=float(time_info["dt_nominal"]),
        )
        for blk in blocks:
            blk.update(
                {
                    "k_start": int(time_info["grid_indices"][blk["start_sample"]]),
                    "k_end": int(time_info["grid_indices"][blk["end_sample"]]),
                    "offset_sec": 0.0,
                    "offset_frac": 0.0,
                    "shift_samps": 0.0,
                    "is_shifted": False,
                }
            )
        time_info["blocks"] = blocks
        time_info["shift_samps_per_time"] = np.zeros(len(sel_times), dtype=np.float64)
        time_info["n_shifted_blocks"] = 0

    return {
        "entries": entries,
        "first_ms": first_ms,
        "chan_freq": chan_freq,
        "nchan_full": int(nchan_full),
        "nchan": int(len(chan_freq)),
        "ncorr": int(ncorr),
        "labels": labels,
        "lbl2idx": lbl2idx,
        "ra0_rad": float(ra0_rad),
        "dec0_rad": float(dec0_rad),
        "used_field": str(used_field),
        "unique_bl_keys": unique_bl_keys,
        "nbl": nbl,
        "sel_times": sel_times,
        "sel_scans": sel_scans,
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "time_info": time_info,
    }


def _plan_streaming_chunks(
    meta: Dict[str, Any],
    *,
    row_chunk_size: int,
    max_nt_new: int,
) -> Tuple[List[_ChunkSpec], int, int]:
    """
    Partition rows within each scan/time-block into timestamp-aligned chunks
    respecting ``row_chunk_size`` and ``max_nt_new <= 4096``.

    Handles both monotonically time-sorted MSs (contiguous ``startrow``/``nrow``
    reads) and unsorted MSs (sorted row-ID gather) cleanly.
    """
    entries: List[_MSMetaEntry] = meta["entries"]
    sel_times: np.ndarray = meta["sel_times"]
    unique_bl_keys: np.ndarray = meta["unique_bl_keys"]
    blocks: List[Dict[str, Any]] = meta["time_info"]["blocks"]

    n_sel = int(sel_times.size)
    if n_sel == 0 or not blocks:
        return [], 0, 0

    t_first = float(sel_times[0])
    t_last = float(sel_times[-1])

    eff_row_cap = max(1, int(row_chunk_size))
    eff_nt_cap = max(1, min(_MAX_NT_EMIT, int(max_nt_new)))
    chunks: List[_ChunkSpec] = []
    chunk_counter = 0
    max_chunk_rows = 0
    total_rows_planned = 0

    # Fast-path for single time-sorted MS with uniform baseline step per timestamp
    if (
        len(entries) == 1
        and entries[0].is_time_sorted
        and entries[0].is_uniform_bl
        and entries[0].ts_starts is not None
        and entries[0].bl_keys_step is not None
    ):
        ent0 = entries[0]
        nbl_step = int(ent0.nbl_step)
        start_idx_meta = int(meta["start_idx"])
        bl_step_idx = np.searchsorted(unique_bl_keys, ent0.bl_keys_step).astype(np.int32)
        nt_per_chunk = max(1, min(eff_nt_cap, eff_row_cap // max(1, nbl_step)))

        for b_idx, blk in enumerate(blocks):
            b_start = int(blk["start_sample"])
            b_end = int(blk["end_sample"])
            b_shift = float(blk.get("shift_samps", 0.0))

            ti = b_start
            while ti <= b_end:
                ti_end = min(b_end + 1, ti + nt_per_chunk)
                nt_new = ti_end - ti
                nrow_seg = nt_new * nbl_step
                startrow = int(ent0.ts_starts[start_idx_meta + ti])
                bl_arr = np.tile(bl_step_idx, nt_new)
                t_rel_arr = np.repeat(np.arange(nt_new, dtype=np.int32), nbl_step)

                seg = _MSReadSegment(
                    ms_idx=ent0.ms_idx,
                    ms_path=ent0.ms_path,
                    startrow=startrow,
                    nrow=nrow_seg,
                    row_ids=None,
                    bl_idx=bl_arr,
                    t_rel=t_rel_arr,
                    has_flag_row=ent0.has_flag_row,
                    has_weight_spectrum=ent0.has_weight_spectrum,
                    time_col=ent0.time_col,
                )
                is_last = ti_end > b_end
                chunks.append(
                    _ChunkSpec(
                        chunk_idx=chunk_counter,
                        block_idx=b_idx,
                        shift_samps=b_shift,
                        t_idx_start=ti,
                        t_idx_end=ti_end,
                        nt_new=nt_new,
                        is_last_in_block=is_last,
                        segments=[seg],
                        total_rows=nrow_seg,
                    )
                )
                chunk_counter += 1
                max_chunk_rows = max(max_chunk_rows, nrow_seg)
                total_rows_planned += nrow_seg
                ti = ti_end

        return chunks, max_chunk_rows, total_rows_planned

    # Build per-timestamp row runs per MS (disjoint MSs -> fast contiguous slices).
    per_ms_runs: List[Dict[str, Any]] = []
    for ent in entries:
        t_vals = ent.t_vals
        if ent.is_time_sorted:
            r0 = int(np.searchsorted(t_vals, t_first, side="left"))
            r1 = int(np.searchsorted(t_vals, t_last, side="right"))
            if r1 <= r0:
                continue
            sub_t = t_vals[r0:r1]
            if ent.is_uniform_bl and ent.bl_keys_step is not None:
                bl_step = np.searchsorted(unique_bl_keys, ent.bl_keys_step).astype(np.int32)
                sub_bl = np.tile(bl_step, (r1 - r0) // max(1, ent.nbl_step))
            else:
                sub_bl = np.searchsorted(unique_bl_keys, ent.bl_keys[r0:r1]).astype(np.int32)
            t_idx_arr = np.searchsorted(sel_times, sub_t).astype(np.int64)
            # Boundaries where timestamp index changes
            chg = np.flatnonzero(t_idx_arr[1:] != t_idx_arr[:-1]) + 1
            starts = np.concatenate(([0], chg))
            ends = np.concatenate((chg, [len(t_idx_arr)]))
            ts_map: Dict[int, Tuple[int, int]] = {}
            for s_pos, e_pos in zip(starts, ends):
                ti = int(t_idx_arr[s_pos])
                ts_map[ti] = (r0 + int(s_pos), r0 + int(e_pos))
            per_ms_runs.append(
                {
                    "entry": ent,
                    "is_sorted": True,
                    "r0": r0,
                    "sub_bl": sub_bl,
                    "ts_map": ts_map,
                }
            )
        else:
            mask = (t_vals >= t_first) & (t_vals <= t_last)
            raw_ids = np.flatnonzero(mask).astype(np.int64)
            if raw_ids.size == 0:
                continue
            sub_t = t_vals[raw_ids]
            order = np.argsort(sub_t, kind="mergesort")
            sorted_ids = raw_ids[order]
            sorted_t = sub_t[order]
            sorted_bl = np.searchsorted(unique_bl_keys, ent.bl_keys[sorted_ids]).astype(np.int32)
            t_idx_arr = np.searchsorted(sel_times, sorted_t).astype(np.int64)
            chg = np.flatnonzero(t_idx_arr[1:] != t_idx_arr[:-1]) + 1
            starts = np.concatenate(([0], chg))
            ends = np.concatenate((chg, [len(t_idx_arr)]))
            ts_map_unsorted: Dict[int, Tuple[int, int]] = {}
            for s_pos, e_pos in zip(starts, ends):
                ti = int(t_idx_arr[s_pos])
                ts_map_unsorted[ti] = (int(s_pos), int(e_pos))
            per_ms_runs.append(
                {
                    "entry": ent,
                    "is_sorted": False,
                    "sorted_ids": sorted_ids,
                    "sub_bl": sorted_bl,
                    "ts_map": ts_map_unsorted,
                }
            )

    for b_idx, blk in enumerate(blocks):
        b_start = int(blk["start_sample"])
        b_end = int(blk["end_sample"])
        b_shift = float(blk.get("shift_samps", 0.0))

        # Assign each timestamp in [b_start, b_end] to a primary MS run; if multiple
        # MSs interleave at the same block, flush chunk whenever the MS switches.
        block_ts_info: List[Tuple[int, int, int, int]] = []  # (ti, run_idx, s_pos, e_pos)
        for ti in range(b_start, b_end + 1):
            matches = [
                (r_i, r_info["ts_map"][ti][0], r_info["ts_map"][ti][1])
                for r_i, r_info in enumerate(per_ms_runs)
                if ti in r_info["ts_map"]
            ]
            if not matches:
                continue
            # Standard disjoint MSs: 1 match per timestamp
            for r_i, s_pos, e_pos in matches:
                block_ts_info.append((ti, r_i, s_pos, e_pos))

        if not block_ts_info:
            continue

        # Partition timestamps in this block into chunks
        # Note: if an MS spans multiple timestamps consecutively, group them up to caps.
        pos = 0
        n_items = len(block_ts_info)
        while pos < n_items:
            first_ti, first_run_idx, _, _ = block_ts_info[pos]
            cur_rows = 0
            cur_nt = 0
            end_pos = pos
            prev_ti = first_ti - 1

            while end_pos < n_items:
                ti, run_idx, s_pos, e_pos = block_ts_info[end_pos]
                if run_idx != first_run_idx:
                    break
                if ti != prev_ti + 1 and cur_nt > 0:
                    # Non-contiguous MS timestamps within block: start new chunk
                    break
                n_ts_rows = e_pos - s_pos
                if cur_nt > 0 and (cur_rows + n_ts_rows > eff_row_cap or (cur_nt + 1) > eff_nt_cap):
                    break
                cur_rows += n_ts_rows
                cur_nt += 1
                prev_ti = ti
                end_pos += 1

            run_info = per_ms_runs[first_run_idx]
            ent = run_info["entry"]
            t_idx_start = block_ts_info[pos][0]
            t_idx_end = block_ts_info[end_pos - 1][0] + 1
            nt_new = t_idx_end - t_idx_start

            # Build single segment for this chunk
            seg_s = block_ts_info[pos][2]
            seg_e = block_ts_info[end_pos - 1][3]
            nrow_seg = seg_e - seg_s
            t_rel_parts = [
                np.full(block_ts_info[k][3] - block_ts_info[k][2], block_ts_info[k][0] - t_idx_start, dtype=np.int32)
                for k in range(pos, end_pos)
            ]
            t_rel_arr = np.concatenate(t_rel_parts) if len(t_rel_parts) > 1 else t_rel_parts[0]

            if run_info["is_sorted"]:
                r0 = run_info["r0"]
                bl_arr = np.ascontiguousarray(run_info["sub_bl"][seg_s - r0 : seg_e - r0], dtype=np.int32)
                row_ids_arr = None
                startrow = seg_s
            else:
                bl_arr = np.ascontiguousarray(run_info["sub_bl"][seg_s:seg_e], dtype=np.int32)
                row_ids_arr = np.ascontiguousarray(run_info["sorted_ids"][seg_s:seg_e], dtype=np.int64)
                startrow = -1

            seg = _MSReadSegment(
                ms_idx=ent.ms_idx,
                ms_path=ent.ms_path,
                startrow=startrow,
                nrow=nrow_seg,
                row_ids=row_ids_arr,
                bl_idx=bl_arr,
                t_rel=t_rel_arr,
                has_flag_row=ent.has_flag_row,
                has_weight_spectrum=ent.has_weight_spectrum,
                time_col=ent.time_col,
            )

            is_last = end_pos >= n_items
            spec = _ChunkSpec(
                chunk_idx=chunk_counter,
                block_idx=b_idx,
                shift_samps=b_shift,
                t_idx_start=t_idx_start,
                t_idx_end=t_idx_end,
                nt_new=nt_new,
                is_last_in_block=is_last,
                segments=[seg],
                total_rows=nrow_seg,
            )
            chunks.append(spec)
            chunk_counter += 1
            max_chunk_rows = max(max_chunk_rows, nrow_seg)
            total_rows_planned += nrow_seg
            pos = end_pos

    return chunks, max_chunk_rows, total_rows_planned


def _read_segment_columns(
    tbl: Any,
    seg: _MSReadSegment,
    *,
    data_column: str,
    nchan: int,
    nchan_full: int,
    ncorr: int,
    chan_sel: Optional[Tuple[int, Optional[int]]],
    out_uvw: np.ndarray,
    out_data: np.ndarray,
    out_wgt: np.ndarray,
    out_flag: np.ndarray,
    out_flag_row: Optional[np.ndarray],
    out_bl_idx: np.ndarray,
    out_t_rel: np.ndarray,
) -> None:
    """
    Read Casacore columns for ``seg`` directly into pre-allocated (pinned)
    arrays using ``getcolnp`` / ``getcolslicenp`` on sorted MSs and
    ``selectrows`` + ``getcolnp`` when unsorted.
    """
    nrow = seg.nrow
    if nrow == 0:
        return

    out_bl_idx[:nrow] = seg.bl_idx
    out_t_rel[:nrow] = seg.t_rel

    sub_tbl = None
    src_tbl = tbl
    startrow = seg.startrow
    try:
        if seg.row_ids is not None:
            sub_tbl = tbl.selectrows(seg.row_ids.tolist())
            src_tbl = sub_tbl
            startrow = 0

        # Resolve explicit channel slice bounds [c0, c1) for getcolslicenp
        if chan_sel is not None:
            c0_raw, c1_raw = chan_sel
            c0 = 0 if c0_raw is None else (nchan_full + int(c0_raw) if int(c0_raw) < 0 else int(c0_raw))
            c1 = nchan_full if c1_raw is None else (nchan_full + int(c1_raw) if int(c1_raw) < 0 else int(c1_raw))
            need_slice = (c0 != 0) or (c1 != nchan_full)
        else:
            c0, c1 = 0, nchan_full
            need_slice = False

        # UVW (nrow, 3) float64
        try:
            src_tbl.getcolnp("UVW", out_uvw[:nrow], startrow=startrow, nrow=nrow)
        except Exception:
            out_uvw[:nrow] = np.asarray(src_tbl.getcol("UVW", startrow=startrow, nrow=nrow), dtype=np.float64)

        # DATA (nrow, nchan, ncorr) complex64
        data_dst = out_data[:nrow]
        try:
            if need_slice:
                src_tbl.getcolslicenp(
                    data_column,
                    data_dst,
                    blc=[int(c0), 0],
                    trc=[int(c1 - 1), int(ncorr - 1)],
                    startrow=startrow,
                    nrow=nrow,
                )
            else:
                src_tbl.getcolnp(data_column, data_dst, startrow=startrow, nrow=nrow)
        except Exception:
            raw_d = src_tbl.getcol(data_column, startrow=startrow, nrow=nrow)
            if raw_d.ndim == 2:
                raw_d = raw_d[:, :, None]
            if need_slice:
                raw_d = raw_d[:, c0:c1, :]
            data_dst[:] = np.asarray(raw_d, dtype=np.complex64)

        # FLAG (nrow, nchan, ncorr) uint8 (view as bool for zero-copy getcolnp)
        flag_dst_u8 = out_flag[:nrow]
        try:
            flag_dst_bool = flag_dst_u8.view(np.bool_)
            if need_slice:
                src_tbl.getcolslicenp(
                    "FLAG",
                    flag_dst_bool,
                    blc=[int(c0), 0],
                    trc=[int(c1 - 1), int(ncorr - 1)],
                    startrow=startrow,
                    nrow=nrow,
                )
            else:
                src_tbl.getcolnp("FLAG", flag_dst_bool, startrow=startrow, nrow=nrow)
        except Exception:
            raw_f = src_tbl.getcol("FLAG", startrow=startrow, nrow=nrow)
            if raw_f.ndim == 2:
                raw_f = raw_f[:, :, None]
            if need_slice:
                raw_f = raw_f[:, c0:c1, :]
            flag_dst_u8[:] = np.asarray(raw_f, dtype=np.uint8)

        # WEIGHT / WEIGHT_SPECTRUM
        if seg.has_weight_spectrum:
            wgt_dst = out_wgt[:nrow]
            try:
                if need_slice:
                    src_tbl.getcolslicenp(
                        "WEIGHT_SPECTRUM",
                        wgt_dst,
                        blc=[int(c0), 0],
                        trc=[int(c1 - 1), int(ncorr - 1)],
                        startrow=startrow,
                        nrow=nrow,
                    )
                else:
                    src_tbl.getcolnp("WEIGHT_SPECTRUM", wgt_dst, startrow=startrow, nrow=nrow)
            except Exception:
                raw_w = src_tbl.getcol("WEIGHT_SPECTRUM", startrow=startrow, nrow=nrow)
                if raw_w.ndim == 2:
                    raw_w = raw_w[:, :, None]
                if need_slice:
                    raw_w = raw_w[:, c0:c1, :]
                wgt_dst[:] = np.asarray(raw_w, dtype=np.float32)
        else:
            if out_wgt.ndim == 2:
                try:
                    src_tbl.getcolnp("WEIGHT", out_wgt[:nrow], startrow=startrow, nrow=nrow)
                except Exception:
                    raw_w_row = np.asarray(src_tbl.getcol("WEIGHT", startrow=startrow, nrow=nrow), dtype=np.float32)
                    if raw_w_row.ndim == 1:
                        raw_w_row = raw_w_row[:, None]
                    out_wgt[:nrow] = raw_w_row
            else:
                raw_w_row = np.asarray(src_tbl.getcol("WEIGHT", startrow=startrow, nrow=nrow), dtype=np.float32)
                if raw_w_row.ndim == 1:
                    raw_w_row = raw_w_row[:, None]
                out_wgt[:nrow] = raw_w_row[:, None, :]

        # Optional FLAG_ROW (nrow,) uint8
        if out_flag_row is not None:
            if seg.has_flag_row:
                fr_dst = out_flag_row[:nrow]
                try:
                    src_tbl.getcolnp("FLAG_ROW", fr_dst.view(np.bool_), startrow=startrow, nrow=nrow)
                except Exception:
                    fr_dst[:] = np.asarray(
                        src_tbl.getcol("FLAG_ROW", startrow=startrow, nrow=nrow), dtype=np.uint8
                    )
            else:
                out_flag_row[:nrow].fill(0)
    finally:
        if sub_tbl is not None:
            sub_tbl.close()


def _alloc_pinned_array(shape: Tuple[int, ...], dtype: Any) -> Tuple[Any, np.ndarray]:
    """Allocate a page-locked NumPy array via ``cp.cuda.alloc_pinned_memory``."""
    dt_obj = np.dtype(dtype)
    count = int(np.prod(shape))
    nbytes = max(1, count * dt_obj.itemsize)
    mem = cp.cuda.alloc_pinned_memory(nbytes)
    arr = np.frombuffer(mem, dtype=dt_obj, count=count).reshape(shape)
    return mem, arr


def _alloc_pinned_pool(
    max_rows: int,
    nchan: int,
    ncorr: int,
    has_wgt_spec: bool,
    has_flag_row: bool,
) -> Dict[str, Any]:
    """Allocate one pinned host buffer pool for asynchronous H2D streaming."""
    wgt_shape = (max_rows, nchan, ncorr) if has_wgt_spec else (max_rows, ncorr)
    specs: Dict[str, Tuple[Tuple[int, ...], Any]] = {
        "uvw": ((max_rows, 3), np.float64),
        "data": ((max_rows, nchan, ncorr), np.complex64),
        "wgt": (wgt_shape, np.float32),
        "flag": ((max_rows, nchan, ncorr), np.uint8),
        "bl_idx": ((max_rows,), np.int32),
        "t_rel": ((max_rows,), np.int32),
    }
    if has_flag_row:
        specs["flag_row"] = ((max_rows,), np.uint8)

    pool: Dict[str, Any] = {"_mem": []}
    for name, (shape, dtype) in specs.items():
        mem, arr = _alloc_pinned_array(shape, dtype)
        pool["_mem"].append(mem)
        pool[name] = arr
    return pool


def _shift_time_series_gpu(
    d_arr: "cp.ndarray",
    shift_samps: float,
    *,
    method: str = "fft",
    axis: int = -1,
    lanczos_radius: int = 4,
    pad_samples: int = 8,
) -> "cp.ndarray":
    """
    GPU-accelerated sub-sample time shift matching
    :func:`fastducc.timeshift.shift_time_series`.
    """
    if abs(float(shift_samps)) < 1e-7 or int(d_arr.shape[axis]) <= 1:
        return d_arr.copy()

    eff_axis = axis if axis >= 0 else d_arr.ndim + axis
    n = int(d_arr.shape[eff_axis])
    meth = str(method).lower().strip()

    if meth == "fft":
        pad = min(int(pad_samples), n)
        npad = n + 2 * pad
        if pad > 0:
            pad_shape = list(d_arr.shape)
            pad_shape[eff_axis] = npad
            padded = cp.empty(tuple(pad_shape), dtype=d_arr.dtype)
            slc_l_src = [slice(None)] * d_arr.ndim
            slc_r_src = [slice(None)] * d_arr.ndim
            slc_l_src[eff_axis] = slice(0, 1)
            slc_r_src[eff_axis] = slice(n - 1, n)

            slc_l_dst = [slice(None)] * d_arr.ndim
            slc_c_dst = [slice(None)] * d_arr.ndim
            slc_r_dst = [slice(None)] * d_arr.ndim
            slc_l_dst[eff_axis] = slice(0, pad)
            slc_c_dst[eff_axis] = slice(pad, pad + n)
            slc_r_dst[eff_axis] = slice(pad + n, npad)

            padded[tuple(slc_l_dst)] = d_arr[tuple(slc_l_src)]
            padded[tuple(slc_c_dst)] = d_arr
            padded[tuple(slc_r_dst)] = d_arr[tuple(slc_r_src)]
        else:
            padded = d_arr

        if cp.iscomplexobj(d_arr):
            c_dtype = cp.complex64 if d_arr.dtype == cp.complex64 else cp.complex128
            freqs = cp.fft.fftfreq(npad)
            phase = cp.exp((-2j * math.pi * float(shift_samps)) * freqs).astype(c_dtype)
            shape_phase = [1] * d_arr.ndim
            shape_phase[eff_axis] = npad
            shifted = cp.fft.ifft(cp.fft.fft(padded, axis=eff_axis) * phase.reshape(shape_phase), axis=eff_axis)
            del freqs, phase
        else:
            c_dtype = cp.complex64 if d_arr.dtype == cp.float32 else cp.complex128
            freqs_r = cp.fft.rfftfreq(npad)
            phase_r = cp.exp((-2j * math.pi * float(shift_samps)) * freqs_r).astype(c_dtype)
            shape_phase = [1] * d_arr.ndim
            shape_phase[eff_axis] = int(freqs_r.size)
            shifted = cp.fft.irfft(
                cp.fft.rfft(padded, axis=eff_axis) * phase_r.reshape(shape_phase),
                n=npad,
                axis=eff_axis,
            )
            del freqs_r, phase_r

        del padded
        slc = [slice(None)] * d_arr.ndim
        slc[eff_axis] = slice(pad, pad + n)
        out = cp.ascontiguousarray(shifted[tuple(slc)], dtype=d_arr.dtype)
        del shifted
        return out

    if meth == "sinc":
        radius = min(int(lanczos_radius), max(1, n // 2))
        k = np.arange(-radius, radius + 1, dtype=np.float64)
        u = k - float(shift_samps)
        h = np.sinc(u) * np.sinc(u / float(radius))
        h_sum = float(np.sum(h))
        if abs(h_sum) > 1e-12:
            h /= h_sum
        out = cp.zeros_like(d_arr)
        for tap_i, weight in enumerate(h):
            offset = tap_i - radius
            idx = np.clip(np.arange(n) - offset, 0, n - 1)
            slc_src = [slice(None)] * d_arr.ndim
            slc_src[eff_axis] = cp.asarray(idx, dtype=cp.int32)
            out += d_arr[tuple(slc_src)] * d_arr.dtype.type(weight)
        return out

    raise ValueError(f"Unknown shift method '{method}'. Choose 'fft' or 'sinc'.")


def _build_common_attrs(
    *,
    meta: Dict[str, Any],
    geom: Dict[str, Any],
    eff_grid_nu: int,
    eff_nv: int,
    u_half: int,
    npix_x: int,
    npix_y: int,
    pixsize_arcsec: float,
    dm: float,
    mode: str,
    m_padded: int,
    dedisp_mode: str = "fullband",
    nsubbands: int = 1,
) -> Dict[str, Any]:
    time_info = meta["time_info"]
    return {
        "format": "ragged_csr",
        "engine": "direct_raster",
        "raster_mode": str(mode),
        "dedisp_mode": str(dedisp_mode),
        "nsubbands": int(nsubbands),
        "tsamp": float(time_info["dt_nominal"]),
        "t0": float(time_info["grid_times"][0]),
        "du": float(geom["du"]),
        "dv": float(geom["dv"]),
        "ra0_rad": float(meta["ra0_rad"]),
        "dec0_rad": float(meta["dec0_rad"]),
        "pixsize_arcsec": float(pixsize_arcsec),
        "pix_rad": float(geom["pix_rad"]),
        "npix_x": int(npix_x),
        "npix_y": int(npix_y),
        "nu": int(u_half),
        "nv": int(eff_nv),
        "grid_nu": int(eff_grid_nu),
        "dm": float(dm),
        "hermitian_half": True,
        "total_planes": int(time_info["total_planes"]),
        "m_total": int(m_padded),
        "original_m_padded": int(m_padded),
    }


def _resolve_plan_buffers(
    sparse_grid_plan: Optional[Dict[str, Any]],
    eff_nv: int,
    u_half: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return host ``(sparse_plan_uv (M, 2) int32, flat_indices (M,) int64)``."""
    if sparse_grid_plan is not None:
        uv_arr = np.ascontiguousarray(sparse_grid_plan["uv_indices"], dtype=np.int32)
        flat_arr = np.ascontiguousarray(sparse_grid_plan["flat_indices"], dtype=np.int64)
        return uv_arr, flat_arr
    v_grid, u_grid = np.divmod(np.arange(eff_nv * u_half, dtype=np.int64), u_half)
    uv_arr = np.ascontiguousarray(np.column_stack((v_grid, u_grid)), dtype=np.int32)
    flat_arr = np.arange(eff_nv * u_half, dtype=np.int64)
    return uv_arr, flat_arr


def _stream_gpu(
    *,
    meta: Dict[str, Any],
    geom: Dict[str, Any],
    eff_grid_nu: int,
    eff_nv: int,
    u_half: int,
    npix_x: int,
    npix_y: int,
    pixsize_arcsec: float,
    delays_host: np.ndarray,
    max_delay: int,
    dedisp_mode: str,
    intra_delays_host: Optional[np.ndarray],
    inter_delays_host: Optional[np.ndarray],
    max_inter_delay: int,
    data_column: str,
    chan_sel: Optional[Tuple[int, Optional[int]]],
    corr_mode: str,
    basis: str,
    single_pol: str,
    average_correlations: bool,
    corr_index: Optional[int],
    dm: float,
    align_to: str,
    collapse_channels: bool,
    nsubbands: int,
    exact_uvw: bool,
    mode: str,
    wgridder_kwargs: Optional[Dict[str, Any]],
    min_valid_channels: int,
    min_uv_distance_m: Optional[float],
    uv_threshold: float,
    row_chunk_size: int,
    regularize_time: bool,
    shift_method: str,
    shift_domain: str,
    sparse_grid_plan: Optional[Dict[str, Any]],
    active_only: bool,
    out_dir: Optional[str],
    write_to_disk: bool,
    device_id: int,
    return_csr_matrix: bool,
    t_prescan: float,
    t_wall0: float,
) -> DirectRasterResult:
    """Execute double-buffered pinned-host async CUDA streaming and VRAM consolidation."""
    from fastducc.ragged_gpu import (
        _write_store_from_gpu,
        consolidate_device_chunks_to_gpu_store,
        pack_uv_chunk_to_ragged_gpu,
    )

    nbl = int(meta["nbl"])
    nchan = int(meta["nchan"])
    nchan_full = int(meta["nchan_full"])
    ncorr = int(meta["ncorr"])
    chan_freq = meta["chan_freq"]
    time_info = meta["time_info"]
    dt_nominal = float(time_info["dt_nominal"])
    total_planes = int(time_info["total_planes"])
    grid_indices = time_info["grid_indices"]

    need_shift_halo = bool(
        regularize_time
        and any(abs(float(b.get("shift_samps", 0.0))) > 1e-7 for b in time_info["blocks"])
    )
    req_overlap = max(max_delay, _SHIFT_HALO_SAMPLES) if need_shift_halo else max_delay

    sparse_plan_uv_host, flat_indices_host = _resolve_plan_buffers(sparse_grid_plan, eff_nv, u_half)
    m_padded = int(sparse_plan_uv_host.shape[0])
    common_attrs = _build_common_attrs(
        meta=meta,
        geom=geom,
        eff_grid_nu=eff_grid_nu,
        eff_nv=eff_nv,
        u_half=u_half,
        npix_x=npix_x,
        npix_y=npix_y,
        pixsize_arcsec=pixsize_arcsec,
        dm=dm,
        mode=mode,
        m_padded=m_padded,
        dedisp_mode=dedisp_mode,
        nsubbands=nsubbands,
    )

    with cp.cuda.Device(int(device_id)):
        free_vram, _ = cp.cuda.Device(int(device_id)).mem_info

        has_any_wgt_spec = any(e.has_weight_spectrum for e in meta["entries"])
        has_any_flag_row = any(e.has_flag_row for e in meta["entries"])
        row_bytes_est = (
            24
            + nchan * ncorr * 9
            + (nchan * ncorr * 4 if has_any_wgt_spec else ncorr * 4)
            + 9
        )
        # Bound row buffer + staging + dense UV chunk within safe VRAM budget
        max_rows_vram = max(1, int(0.25 * free_vram) // max(1, row_bytes_est))
        eff_row_chunk = min(int(row_chunk_size), max_rows_vram)

        stage_bytes_per_t = max(1, (24 * nbl * nchan) + (48 * nbl) + (16 * eff_nv * u_half))
        vram_nt_cap = max(64, int(0.35 * free_vram) // stage_bytes_per_t)
        max_nt_new = min(_MAX_NT_EMIT, max(16, vram_nt_cap - req_overlap))

        chunks, max_chunk_rows, total_rows = _plan_streaming_chunks(
            meta,
            row_chunk_size=eff_row_chunk,
            max_nt_new=max_nt_new,
        )

        if not chunks:
            d_off = cp.zeros(1, dtype=cp.uint64)
            d_uv = cp.empty((0, 2), dtype=cp.int32)
            d_vis = cp.empty(0, dtype=cp.complex64)
            d_times = cp.empty(0, dtype=cp.float64)
            common_attrs.update({"m_total": 0, "nnz_total": 0, "pruned": True})
            gpu_store = from_gpu_buffers(
                d_off, d_uv, d_vis, d_times, attrs=common_attrs, store_dir=out_dir or "", copy_to_host=False, device_id=device_id
            )
            host_store = None
            if write_to_disk and out_dir:
                host_store = write_ragged_store_to_disk(out_dir, d_off, d_uv, d_vis, d_times, attrs=common_attrs)
            csr_mat = build_gpu_csr_matrix(gpu_store, total_planes, dt_nominal) if return_csr_matrix else None
            return DirectRasterResult(
                gpu_store=gpu_store,
                store=host_store,
                csr_matrix=csr_mat,
                attrs=common_attrs,
                time_info=time_info,
                io_s=t_prescan,
                gpu_s=0.0,
                consolidate_s=0.0,
                total_s=time.time() - t_wall0,
                nrows=0,
                m_total=0,
                nnz_total=0,
                device="cuda",
            )

        max_nt_stage = max(req_overlap + max(c.nt_new for c in chunks), req_overlap + 1)
        tail_cap = max(1, req_overlap)

        h2d_stream = cp.cuda.Stream(non_blocking=True)
        compute_stream = cp.cuda.Stream(non_blocking=True)
        slot_h2d_done: List[Optional[Any]] = [None, None]
        ev_stage_done: Optional[Any] = None

        host_pools = [
            _alloc_pinned_pool(max_chunk_rows, nchan, ncorr, has_any_wgt_spec, has_any_flag_row),
            _alloc_pinned_pool(max_chunk_rows, nchan, ncorr, has_any_wgt_spec, has_any_flag_row),
        ]

        # Flat device row-transfer pool (single allocation reused across chunks)
        wgt_row_shape = (max_chunk_rows, nchan, ncorr) if has_any_wgt_spec else (max_chunk_rows, ncorr)
        d_rows = {
            "uvw": cp.empty((max_chunk_rows, 3), dtype=cp.float64),
            "data": cp.empty((max_chunk_rows, nchan, ncorr), dtype=cp.complex64),
            "wgt": cp.empty(wgt_row_shape, dtype=cp.float32),
            "flag": cp.empty((max_chunk_rows, nchan, ncorr), dtype=cp.uint8),
            "bl_idx": cp.empty((max_chunk_rows,), dtype=cp.int32),
            "t_rel": cp.empty((max_chunk_rows,), dtype=cp.int32),
            "flag_row": cp.empty((max_chunk_rows,), dtype=cp.uint8) if has_any_flag_row else None,
        }

        # Linear device pools for contiguous staging + lookahead tail slicing
        d_vis_pool = cp.empty(nbl * nchan * max_nt_stage, dtype=cp.complex64)
        d_wgt_pool = cp.empty(nbl * nchan * max_nt_stage, dtype=cp.float32)
        d_uvw_pool = cp.empty(nbl * 3 * max_nt_stage, dtype=cp.float64)

        d_vis_tail_pool = cp.empty(nbl * nchan * tail_cap, dtype=cp.complex64)
        d_wgt_tail_pool = cp.empty(nbl * nchan * tail_cap, dtype=cp.float32)
        d_uvw_tail_pool = cp.empty(nbl * 3 * tail_cap, dtype=cp.float64)

        d_flat_indices = cp.asarray(flat_indices_host, dtype=cp.int64)
        d_sparse_uv = cp.asarray(sparse_plan_uv_host, dtype=cp.int32)

        device_chunks: List[Tuple[Any, Any, Any, Any]] = []
        io_wait_s = t_prescan
        gpu_compute_s = 0.0

        table_cache: Dict[str, Any] = {}

        def _io_read_chunk(chunk_spec: _ChunkSpec, slot: int) -> Tuple[_ChunkSpec, int, float]:
            t0_io = time.time()
            ev = slot_h2d_done[slot]
            if ev is not None:
                ev.synchronize()

            pool = host_pools[slot]
            seg = chunk_spec.segments[0]
            tbl = table_cache.get(seg.ms_path)
            if tbl is None:
                tbl = table(seg.ms_path, readonly=True, ack=False)
                table_cache[seg.ms_path] = tbl

            _read_segment_columns(
                tbl,
                seg,
                data_column=data_column,
                nchan=nchan,
                nchan_full=nchan_full,
                ncorr=ncorr,
                chan_sel=chan_sel,
                out_uvw=pool["uvw"],
                out_data=pool["data"],
                out_wgt=pool["wgt"],
                out_flag=pool["flag"],
                out_flag_row=pool.get("flag_row"),
                out_bl_idx=pool["bl_idx"],
                out_t_rel=pool["t_rel"],
            )
            return chunk_spec, seg.nrow, float(time.time() - t0_io)

        def _io_close_tables() -> None:
            close_errs = []
            for path, tbl in table_cache.items():
                try:
                    tbl.close()
                except Exception as exc:
                    close_errs.append((path, exc))
            table_cache.clear()
            if close_errs:
                err_msgs = "; ".join(f"{p}: {e}" for p, e in close_errs)
                logger.warning("Errors while closing casacore tables: %s", err_msgs)

        def _emit_from_staged(
            d_vis_stage: "cp.ndarray",
            d_wgt_stage: "cp.ndarray",
            d_uvw_stage: "cp.ndarray",
            nt_emit: int,
            emit_cursor: int,
            shift_samps: float,
            already_shifted: bool = False,
        ) -> None:
            if nt_emit <= 0:
                return
            if (not already_shifted) and regularize_time and abs(shift_samps) > 1e-7 and shift_domain == "native":
                w_vis = _shift_time_series_gpu(d_vis_stage, shift_samps, method=shift_method, axis=-1)
                w_wgt = cp.maximum(
                    _shift_time_series_gpu(d_wgt_stage, shift_samps, method=shift_method, axis=-1),
                    np.float32(0.0),
                )
                w_uvw = _shift_time_series_gpu(d_uvw_stage, shift_samps, method=shift_method, axis=-1)
            else:
                w_vis, w_wgt, w_uvw = d_vis_stage, d_wgt_stage, d_uvw_stage

            if dedisp_mode == "subbanded":
                mode_id = _normalise_mode(mode)
                wg_kw = dict(wgridder_kwargs or {})
                wg_kw.pop("do_wgridding", None)
                use_snap_norm, _ = _resolve_normalise_mode(mode_id, wg_kw.get("normalise_mode"))
                nt_tier1 = min(int(w_vis.shape[2]), nt_emit + max_inter_delay)
                tier1_res = dedisperse_and_rasterise_subband_gpu(
                    vis_3d=w_vis,
                    wgt_3d=w_wgt,
                    uvw_3d=w_uvw,
                    freqs_hz=chan_freq,
                    intra_delays=intra_delays_host,
                    n_subbands=int(nsubbands),
                    nt_out=nt_tier1,
                    dm_coarse=float(dm),
                    dt=dt_nominal,
                    align_to=align_to,
                    exact_uvw=bool(exact_uvw),
                    npix_x=int(npix_x),
                    npix_y=int(npix_y),
                    pixsize_arcsec=float(pixsize_arcsec),
                    pix_rad=float(geom["pix_rad"]),
                    grid_nu=int(eff_grid_nu),
                    nv=int(eff_nv),
                    mode=mode,
                    min_valid_channels=int(min_valid_channels),
                    min_uv_distance_m=min_uv_distance_m,
                    flat_indices=d_flat_indices,
                    m_active=m_padded,
                    return_weights=(not use_snap_norm),
                    return_host=False,
                    **wg_kw,
                )
                if use_snap_norm:
                    d_active_dga, d_active_wgt = tier1_res, None
                else:
                    d_active_dga, d_active_wgt = tier1_res
                d_sparse_chunk = dedisperse_active_cell_matrix_gpu(
                    d_active_dga,
                    inter_delays_host,
                    nt_out=nt_emit,
                    active_wgt=d_active_wgt,
                    uv_threshold=float(uv_threshold),
                    return_host=False,
                )
                del d_active_dga
                if d_active_wgt is not None:
                    del d_active_wgt
            else:
                d_sparse_chunk = dedisperse_and_rasterise_chunk_gpu(
                    vis_3d=w_vis,
                    wgt_3d=w_wgt,
                    uvw_3d=w_uvw,
                    freqs_hz=chan_freq,
                    delays=delays_host,
                    nt_out=nt_emit,
                    dm=float(dm),
                    dt=dt_nominal,
                    align_to=align_to,
                    collapse_channels=bool(collapse_channels),
                    nsubbands=int(nsubbands),
                    exact_uvw=bool(exact_uvw),
                    npix_x=int(npix_x),
                    npix_y=int(npix_y),
                    pixsize_arcsec=float(pixsize_arcsec),
                    pix_rad=float(geom["pix_rad"]),
                    grid_nu=int(eff_grid_nu),
                    nv=int(eff_nv),
                    mode=mode,
                    min_valid_channels=int(min_valid_channels),
                    min_uv_distance_m=min_uv_distance_m,
                    uv_threshold=float(uv_threshold),
                    flat_indices=d_flat_indices,
                    m_active=m_padded,
                    return_sparse=True,
                    return_host=False,
                    **(wgridder_kwargs or {}),
                )
            if regularize_time and abs(shift_samps) > 1e-7 and shift_domain == "gridded":
                d_sparse_chunk = _shift_time_series_gpu(d_sparse_chunk, shift_samps, method=shift_method, axis=-1)

            t_chunk_host = np.asarray(
                grid_indices[emit_cursor : emit_cursor + nt_emit], dtype=np.float64
            ) * dt_nominal
            d_t_chunk = cp.asarray(t_chunk_host, dtype=cp.float64)

            d_c_idx, d_c_cnt, d_v_data, d_v_times = pack_chunk_to_ragged_gpu(
                d_sparse_chunk,
                d_t_chunk,
                threshold=float(uv_threshold),
                return_host=False,
            )
            if int(d_v_data.size) > 0:
                device_chunks.append((d_c_idx, d_c_cnt, d_v_data, d_v_times))
            del d_sparse_chunk, d_t_chunk
            if w_vis is not d_vis_stage:
                del w_vis, w_wgt, w_uvw

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            cur_block_idx = -1
            n_tail = 0
            emit_cursor = 0

            next_fut = executor.submit(_io_read_chunk, chunks[0], 0)
            n_chunks = len(chunks)

            for k in range(n_chunks):
                t_w0 = time.time()
                chunk_spec, nrow_k, _chunk_io_s = next_fut.result()
                io_wait_s += max(0.0, time.time() - t_w0)
                slot = k % 2
                if k + 1 < n_chunks:
                    next_fut = executor.submit(_io_read_chunk, chunks[k + 1], (k + 1) % 2)

                t_gpu0 = time.time()
                pool = host_pools[slot]

                if chunk_spec.block_idx != cur_block_idx:
                    cur_block_idx = chunk_spec.block_idx
                    n_tail = 0
                    emit_cursor = chunk_spec.t_idx_start

                nt_staged = n_tail + chunk_spec.nt_new
                # Ensure staging pool covers nt_staged
                if nt_staged > max_nt_stage:
                    max_nt_stage = nt_staged
                    d_vis_pool = cp.empty(nbl * nchan * max_nt_stage, dtype=cp.complex64)
                    d_wgt_pool = cp.empty(nbl * nchan * max_nt_stage, dtype=cp.float32)
                    d_uvw_pool = cp.empty(nbl * 3 * max_nt_stage, dtype=cp.float64)

                # 1. Asynchronous H2D on h2d_stream (waits for previous stage_ms_rows_gpu)
                with h2d_stream:
                    if ev_stage_done is not None:
                        h2d_stream.wait_event(ev_stage_done)
                    if nrow_k > 0:
                        d_rows["uvw"][:nrow_k].set(pool["uvw"][:nrow_k], stream=h2d_stream)
                        d_rows["data"][:nrow_k].set(pool["data"][:nrow_k], stream=h2d_stream)
                        d_rows["wgt"][:nrow_k].set(pool["wgt"][:nrow_k], stream=h2d_stream)
                        d_rows["flag"][:nrow_k].set(pool["flag"][:nrow_k], stream=h2d_stream)
                        d_rows["bl_idx"][:nrow_k].set(pool["bl_idx"][:nrow_k], stream=h2d_stream)
                        d_rows["t_rel"][:nrow_k].set(pool["t_rel"][:nrow_k], stream=h2d_stream)
                        if has_any_flag_row and d_rows["flag_row"] is not None:
                            d_rows["flag_row"][:nrow_k].set(pool["flag_row"][:nrow_k], stream=h2d_stream)
                    ev_h2d = cp.cuda.Event()
                    ev_h2d.record(h2d_stream)
                    slot_h2d_done[slot] = ev_h2d

                # 2. Compute stream waits for H2D, stages rows, manages overlap, rasterises, packs
                with compute_stream:
                    compute_stream.wait_event(ev_h2d)

                    d_vis_stage = d_vis_pool[: nbl * nchan * nt_staged].reshape((nbl, nchan, nt_staged))
                    d_wgt_stage = d_wgt_pool[: nbl * nchan * nt_staged].reshape((nbl, nchan, nt_staged))
                    d_uvw_stage = d_uvw_pool[: nbl * 3 * nt_staged].reshape((nbl, 3, nt_staged))
                    d_vis_stage.fill(0)
                    d_wgt_stage.fill(0)
                    d_uvw_stage.fill(0)

                    if n_tail > 0:
                        d_vis_stage[:, :, :n_tail] = d_vis_tail_pool[: nbl * nchan * n_tail].reshape((nbl, nchan, n_tail))
                        d_wgt_stage[:, :, :n_tail] = d_wgt_tail_pool[: nbl * nchan * n_tail].reshape((nbl, nchan, n_tail))
                        d_uvw_stage[:, :, :n_tail] = d_uvw_tail_pool[: nbl * 3 * n_tail].reshape((nbl, 3, n_tail))

                    if nrow_k > 0:
                        stage_ms_rows_gpu(
                            bl_idx=d_rows["bl_idx"][:nrow_k],
                            t_rel=d_rows["t_rel"][:nrow_k],
                            uvw_rows=d_rows["uvw"][:nrow_k],
                            data_rows=d_rows["data"][:nrow_k],
                            wgt_rows=d_rows["wgt"][:nrow_k],
                            flag_rows=d_rows["flag"][:nrow_k],
                            flag_row=(
                                d_rows["flag_row"][:nrow_k]
                                if (has_any_flag_row and chunk_spec.segments[0].has_flag_row and d_rows["flag_row"] is not None)
                                else None
                            ),
                            nbl=nbl,
                            nt_stage=nt_staged,
                            t_offset=n_tail,
                            vis_3d=d_vis_stage,
                            wgt_3d=d_wgt_stage,
                            uvw_3d=d_uvw_stage,
                            corr_mode=corr_mode,
                            basis=basis,
                            single_pol=single_pol,
                            average_correlations=average_correlations,
                            corr_index=corr_index,
                            labels=meta["labels"],
                            lbl2idx=meta["lbl2idx"],
                            min_uv_distance_m=min_uv_distance_m,
                            return_host=False,
                        )
                    ev_stage_done = cp.cuda.Event()
                    ev_stage_done.record(compute_stream)

                    if chunk_spec.is_last_in_block:
                        # Emit all remaining timestamps in <= 4096-sample slices
                        need_pre_shift = (
                            nt_staged > _MAX_NT_EMIT
                            and regularize_time
                            and abs(chunk_spec.shift_samps) > 1e-7
                            and shift_domain == "native"
                        )
                        if need_pre_shift:
                            src_vis = _shift_time_series_gpu(d_vis_stage, chunk_spec.shift_samps, method=shift_method, axis=-1)
                            src_wgt = cp.maximum(
                                _shift_time_series_gpu(d_wgt_stage, chunk_spec.shift_samps, method=shift_method, axis=-1),
                                np.float32(0.0),
                            )
                            src_uvw = _shift_time_series_gpu(d_uvw_stage, chunk_spec.shift_samps, method=shift_method, axis=-1)
                        else:
                            src_vis, src_wgt, src_uvw = d_vis_stage, d_wgt_stage, d_uvw_stage

                        rem = nt_staged
                        offset_rem = 0
                        while rem > 0:
                            nt_emit = min(_MAX_NT_EMIT, rem)
                            if offset_rem == 0:
                                cur_vis, cur_wgt, cur_uvw = src_vis, src_wgt, src_uvw
                            else:
                                cur_vis = cp.ascontiguousarray(src_vis[:, :, offset_rem:])
                                cur_wgt = cp.ascontiguousarray(src_wgt[:, :, offset_rem:])
                                cur_uvw = cp.ascontiguousarray(src_uvw[:, :, offset_rem:])
                            _emit_from_staged(
                                cur_vis,
                                cur_wgt,
                                cur_uvw,
                                nt_emit,
                                emit_cursor,
                                chunk_spec.shift_samps,
                                already_shifted=need_pre_shift,
                            )
                            emit_cursor += nt_emit
                            offset_rem += nt_emit
                            rem -= nt_emit
                        if need_pre_shift:
                            del src_vis, src_wgt, src_uvw
                        n_tail = 0
                    else:
                        nt_emit = min(_MAX_NT_EMIT, max(0, nt_staged - req_overlap))
                        n_next_tail = nt_staged - nt_emit
                        if n_next_tail > tail_cap:
                            tail_cap = n_next_tail
                            d_vis_tail_pool = cp.empty(nbl * nchan * tail_cap, dtype=cp.complex64)
                            d_wgt_tail_pool = cp.empty(nbl * nchan * tail_cap, dtype=cp.float32)
                            d_uvw_tail_pool = cp.empty(nbl * 3 * tail_cap, dtype=cp.float64)
                        if n_next_tail > 0:
                            d_vis_tail_pool[: nbl * nchan * n_next_tail].reshape((nbl, nchan, n_next_tail))[:] = d_vis_stage[:, :, nt_emit:]
                            d_wgt_tail_pool[: nbl * nchan * n_next_tail].reshape((nbl, nchan, n_next_tail))[:] = d_wgt_stage[:, :, nt_emit:]
                            d_uvw_tail_pool[: nbl * 3 * n_next_tail].reshape((nbl, 3, n_next_tail))[:] = d_uvw_stage[:, :, nt_emit:]

                        if nt_emit > 0:
                            _emit_from_staged(
                                d_vis_stage,
                                d_wgt_stage,
                                d_uvw_stage,
                                nt_emit,
                                emit_cursor,
                                chunk_spec.shift_samps,
                            )
                            emit_cursor += nt_emit
                        n_tail = n_next_tail

                gpu_compute_s += max(0.0, time.time() - t_gpu0)

            h2d_stream.synchronize()
            compute_stream.synchronize()
        finally:
            try:
                executor.submit(_io_close_tables).result(timeout=30)
            except Exception as exc:
                logger.warning("ThreadPoolExecutor failed closing tables cleanly: %s", exc)
            executor.shutdown(wait=True)
            # Release pinned host pools and streaming scratch buffers to CuPy pool before consolidation
            d_vis_stage = d_wgt_stage = d_uvw_stage = None
            src_vis = src_wgt = src_uvw = None
            cur_vis = cur_wgt = cur_uvw = None
            host_pools.clear()
            d_rows.clear()
            del d_vis_pool, d_wgt_pool, d_uvw_pool
            del d_vis_tail_pool, d_wgt_tail_pool, d_uvw_tail_pool
            del d_flat_indices

        # 3. Consolidate completely in VRAM (zero intermediate .csr.bin files)
        t_cons0 = time.time()
        gpu_dict = consolidate_device_chunks_to_gpu_store(
            device_chunks=device_chunks,
            sparse_plan_uv=d_sparse_uv,
            active_only=bool(active_only),
            store_dir=None,
            attrs=common_attrs,
        )
        del device_chunks, d_sparse_uv

        out_attrs = dict(gpu_dict["attrs"])
        m_active = int(out_attrs["m_total"])
        nnz_total = int(out_attrs["nnz_total"])

        gpu_store = from_gpu_buffers(
            cell_offsets=gpu_dict["cell_offsets"],
            cell_uv=gpu_dict["cell_uv"],
            vis_data=gpu_dict["vis_data"],
            vis_times=gpu_dict["vis_times"],
            attrs=out_attrs,
            store_dir=out_dir or "",
            copy_to_host=False,
            device_id=int(device_id),
        )

        host_store: Optional[RaggedDGAStore] = None
        if write_to_disk and out_dir:
            _write_store_from_gpu(
                store_dir=out_dir,
                d_cell_offsets=gpu_store.cell_offsets,
                d_cell_uv=gpu_store.cell_uv,
                d_vis_data=gpu_store.vis_data,
                d_vis_times=gpu_store.vis_times,
                m_active=m_active,
                m_padded=m_padded,
                nnz_total=nnz_total,
                active_only=bool(active_only),
                attrs=out_attrs,
                t_start=t_cons0,
            )
            host_store = RaggedDGAStore.open(out_dir, mode="r")

        csr_mat = (
            build_gpu_csr_matrix(gpu_store, total_planes=total_planes, dt=dt_nominal)
            if return_csr_matrix
            else None
        )
        consolidate_s = float(time.time() - t_cons0)
        cp.get_default_memory_pool().free_all_blocks()

    total_s = float(time.time() - t_wall0)
    logger.info(
        "[Direct Raster GPU] Consolidated M=%d (padded=%d), NNZ=%d across %d rows | "
        "io=%.2fs gpu=%.2fs consolidate=%.2fs total=%.2fs",
        m_active,
        m_padded,
        nnz_total,
        total_rows,
        io_wait_s,
        gpu_compute_s,
        consolidate_s,
        total_s,
    )
    return DirectRasterResult(
        gpu_store=gpu_store,
        store=host_store,
        csr_matrix=csr_mat,
        attrs=out_attrs,
        time_info=time_info,
        io_s=io_wait_s,
        gpu_s=gpu_compute_s,
        consolidate_s=consolidate_s,
        total_s=total_s,
        nrows=total_rows,
        m_total=m_active,
        nnz_total=nnz_total,
        device="cuda",
    )


def _consolidate_host_chunks_in_memory(
    cpu_chunks: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    sparse_plan_uv: np.ndarray,
    *,
    active_only: bool,
    attrs: Dict[str, Any],
    out_dir: Optional[str],
    write_to_disk: bool,
) -> RaggedDGAStore:
    """Consolidate host CSR chunk tuples in memory without writing intermediate fragment files."""
    t_start = time.time()
    m_padded = int(sparse_plan_uv.shape[0])
    counts_per_cell = np.zeros(m_padded, dtype=np.uint64)
    for c_idx, c_cnt, _, _ in cpu_chunks:
        if len(c_idx) > 0:
            counts_per_cell[c_idx] += c_cnt

    if active_only:
        active_cells = np.flatnonzero(counts_per_cell > 0).astype(np.uint32)
        m_active = int(active_cells.size)
        cell_uv = np.ascontiguousarray(sparse_plan_uv[active_cells], dtype=np.int32)
        cell_offsets = np.concatenate([[0], np.cumsum(counts_per_cell[active_cells], dtype=np.uint64)])
        cell_to_active = np.full(m_padded, -1, dtype=np.int32)
        cell_to_active[active_cells] = np.arange(m_active, dtype=np.int32)
    else:
        m_active = m_padded
        cell_uv = np.ascontiguousarray(sparse_plan_uv, dtype=np.int32)
        cell_offsets = np.concatenate([[0], np.cumsum(counts_per_cell, dtype=np.uint64)])
        cell_to_active = np.arange(m_padded, dtype=np.int32)

    nnz_total = int(cell_offsets[-1]) if cell_offsets.size > 0 else 0
    vis_data = np.empty(nnz_total, dtype=np.complex64)
    vis_times = np.empty(nnz_total, dtype=np.float64)
    cur_offsets = cell_offsets[:-1].copy()

    for c_idx, c_cnt, f_vis, f_times in cpu_chunks:
        if f_vis.size == 0:
            continue
        _scatter_fragment_samples_numba(
            c_idx=c_idx,
            c_cnt=c_cnt,
            f_vis=f_vis,
            f_times=f_times,
            cell_to_active=cell_to_active,
            cur_offsets=cur_offsets,
            vis_data=vis_data,
            vis_times=vis_times,
        )

    store_attrs = dict(attrs)
    store_attrs.update(
        {
            "format": "ragged_csr",
            "m_total": int(m_active),
            "original_m_padded": int(m_padded),
            "nnz_total": int(nnz_total),
            "pruned": bool(active_only or m_active < m_padded),
            "consolidation_duration_s": float(time.time() - t_start),
        }
    )

    if write_to_disk and out_dir:
        return write_ragged_store_to_disk(
            store_dir=out_dir,
            cell_offsets=cell_offsets,
            cell_uv=cell_uv,
            vis_data=vis_data,
            vis_times=vis_times,
            attrs=store_attrs,
        )

    return RaggedDGAStore(
        store_dir=str(out_dir or ""),
        attrs=store_attrs,
        cell_offsets=cell_offsets,
        cell_uv=cell_uv,
        vis_data=vis_data,
        vis_times=vis_times,
    )


def _stream_cpu(
    *,
    meta: Dict[str, Any],
    geom: Dict[str, Any],
    eff_grid_nu: int,
    eff_nv: int,
    u_half: int,
    npix_x: int,
    npix_y: int,
    pixsize_arcsec: float,
    delays_host: np.ndarray,
    max_delay: int,
    dedisp_mode: str,
    intra_delays_host: Optional[np.ndarray],
    inter_delays_host: Optional[np.ndarray],
    max_inter_delay: int,
    data_column: str,
    chan_sel: Optional[Tuple[int, Optional[int]]],
    corr_mode: str,
    basis: str,
    single_pol: str,
    average_correlations: bool,
    corr_index: Optional[int],
    dm: float,
    align_to: str,
    collapse_channels: bool,
    nsubbands: int,
    exact_uvw: bool,
    mode: str,
    wgridder_kwargs: Optional[Dict[str, Any]],
    min_valid_channels: int,
    min_uv_distance_m: Optional[float],
    uv_threshold: float,
    row_chunk_size: int,
    regularize_time: bool,
    shift_method: str,
    shift_domain: str,
    sparse_grid_plan: Optional[Dict[str, Any]],
    active_only: bool,
    out_dir: Optional[str],
    write_to_disk: bool,
    use_numba: bool,
    t_prescan: float,
    t_wall0: float,
) -> DirectRasterResult:
    """Execute identical scan/overlap chunked streaming and in-memory consolidation on CPU."""
    nbl = int(meta["nbl"])
    nchan = int(meta["nchan"])
    nchan_full = int(meta["nchan_full"])
    ncorr = int(meta["ncorr"])
    chan_freq = meta["chan_freq"]
    time_info = meta["time_info"]
    dt_nominal = float(time_info["dt_nominal"])
    grid_indices = time_info["grid_indices"]

    need_shift_halo = bool(
        regularize_time
        and any(abs(float(b.get("shift_samps", 0.0))) > 1e-7 for b in time_info["blocks"])
    )
    req_overlap = max(max_delay, _SHIFT_HALO_SAMPLES) if need_shift_halo else max_delay

    sparse_plan_uv_host, flat_indices_host = _resolve_plan_buffers(sparse_grid_plan, eff_nv, u_half)
    m_padded = int(sparse_plan_uv_host.shape[0])
    common_attrs = _build_common_attrs(
        meta=meta,
        geom=geom,
        eff_grid_nu=eff_grid_nu,
        eff_nv=eff_nv,
        u_half=u_half,
        npix_x=npix_x,
        npix_y=npix_y,
        pixsize_arcsec=pixsize_arcsec,
        dm=dm,
        mode=mode,
        m_padded=m_padded,
        dedisp_mode=dedisp_mode,
        nsubbands=nsubbands,
    )

    chunks, max_chunk_rows, total_rows = _plan_streaming_chunks(
        meta,
        row_chunk_size=int(row_chunk_size),
        max_nt_new=_MAX_NT_EMIT,
    )
    if not chunks:
        empty_store = _consolidate_host_chunks_in_memory(
            [],
            sparse_plan_uv_host,
            active_only=bool(active_only),
            attrs=common_attrs,
            out_dir=out_dir,
            write_to_disk=bool(write_to_disk),
        )
        return DirectRasterResult(
            gpu_store=None,
            store=empty_store,
            csr_matrix=None,
            attrs=empty_store.attrs,
            time_info=time_info,
            io_s=t_prescan,
            gpu_s=0.0,
            consolidate_s=0.0,
            total_s=time.time() - t_wall0,
            nrows=0,
            m_total=empty_store.m_total,
            nnz_total=empty_store.nnz,
            device="cpu",
        )

    has_any_wgt_spec = any(e.has_weight_spectrum for e in meta["entries"])
    has_any_flag_row = any(e.has_flag_row for e in meta["entries"])
    wgt_shape = (max_chunk_rows, nchan, ncorr) if has_any_wgt_spec else (max_chunk_rows, ncorr)
    buf_uvw = np.empty((max_chunk_rows, 3), dtype=np.float64)
    buf_data = np.empty((max_chunk_rows, nchan, ncorr), dtype=np.complex64)
    buf_wgt = np.empty(wgt_shape, dtype=np.float32)
    buf_flag = np.empty((max_chunk_rows, nchan, ncorr), dtype=np.uint8)
    buf_flag_row = np.empty((max_chunk_rows,), dtype=np.uint8) if has_any_flag_row else None
    buf_bl_idx = np.empty((max_chunk_rows,), dtype=np.int32)
    buf_t_rel = np.empty((max_chunk_rows,), dtype=np.int32)

    cpu_chunks: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    table_cache: Dict[str, Any] = {}
    io_s = t_prescan
    compute_s = 0.0

    cur_block_idx = -1
    tail_vis: Optional[np.ndarray] = None
    tail_wgt: Optional[np.ndarray] = None
    tail_uvw: Optional[np.ndarray] = None
    emit_cursor = 0

    def _emit_cpu(
        vis_in: np.ndarray,
        wgt_in: np.ndarray,
        uvw_in: np.ndarray,
        nt_emit: int,
        cur_emit: int,
        shift_samps: float,
        already_shifted: bool = False,
    ) -> None:
        if nt_emit <= 0:
            return
        if (not already_shifted) and regularize_time and abs(shift_samps) > 1e-7 and shift_domain == "native":
            w_vis = timeshift.shift_time_series(vis_in, shift_samps, method=shift_method, axis=-1)
            w_wgt = np.maximum(
                timeshift.shift_time_series(wgt_in, shift_samps, method=shift_method, axis=-1),
                np.float32(0.0),
            )
            w_uvw = timeshift.shift_time_series(uvw_in, shift_samps, method=shift_method, axis=-1)
        else:
            w_vis, w_wgt, w_uvw = vis_in, wgt_in, uvw_in

        if dedisp_mode == "subbanded":
            mode_id = _normalise_mode(mode)
            wg_kw = dict(wgridder_kwargs or {})
            wg_kw.pop("do_wgridding", None)
            use_snap_norm, _ = _resolve_normalise_mode(mode_id, wg_kw.get("normalise_mode"))
            nt_tier1 = min(int(w_vis.shape[2]), nt_emit + max_inter_delay)
            tier1_res = dedisperse_and_rasterise_subband_cpu(
                vis_3d=w_vis,
                wgt_3d=w_wgt,
                uvw_3d=w_uvw,
                freqs_hz=chan_freq,
                intra_delays=intra_delays_host,
                n_subbands=int(nsubbands),
                nt_out=nt_tier1,
                dm_coarse=float(dm),
                dt=dt_nominal,
                align_to=align_to,
                exact_uvw=bool(exact_uvw),
                npix_x=int(npix_x),
                npix_y=int(npix_y),
                pixsize_arcsec=float(pixsize_arcsec),
                pix_rad=float(geom["pix_rad"]),
                grid_nu=int(eff_grid_nu),
                nv=int(eff_nv),
                mode=mode,
                min_valid_channels=int(min_valid_channels),
                min_uv_distance_m=min_uv_distance_m,
                flat_indices=flat_indices_host,
                m_active=m_padded,
                return_weights=(not use_snap_norm),
                use_numba=bool(use_numba),
                **wg_kw,
            )
            if use_snap_norm:
                active_dga, active_wgt = tier1_res, None
            else:
                active_dga, active_wgt = tier1_res
            dga_sub = dedisperse_active_cell_matrix_cpu(
                active_dga,
                inter_delays_host,
                nt_out=nt_emit,
                active_wgt=active_wgt,
                uv_threshold=float(uv_threshold),
                use_numba=bool(use_numba),
            )
        else:
            dga_sub = dedisperse_and_rasterise_chunk_cpu(
                vis_3d=w_vis,
                wgt_3d=w_wgt,
                uvw_3d=w_uvw,
                freqs_hz=chan_freq,
                delays=delays_host,
                nt_out=nt_emit,
                dm=float(dm),
                dt=dt_nominal,
                align_to=align_to,
                collapse_channels=bool(collapse_channels),
                nsubbands=int(nsubbands),
                exact_uvw=bool(exact_uvw),
                npix_x=int(npix_x),
                npix_y=int(npix_y),
                pixsize_arcsec=float(pixsize_arcsec),
                pix_rad=float(geom["pix_rad"]),
                grid_nu=int(eff_grid_nu),
                nv=int(eff_nv),
                mode=mode,
                min_valid_channels=int(min_valid_channels),
                min_uv_distance_m=min_uv_distance_m,
                uv_threshold=float(uv_threshold),
                flat_indices=flat_indices_host,
                m_active=m_padded,
                return_sparse=True,
                use_numba=bool(use_numba),
                **(wgridder_kwargs or {}),
            )
        if regularize_time and abs(shift_samps) > 1e-7 and shift_domain == "gridded":
            dga_sub = timeshift.shift_time_series(dga_sub, shift_samps, method=shift_method, axis=-1)

        t_chunk = np.asarray(grid_indices[cur_emit : cur_emit + nt_emit], dtype=np.float64) * dt_nominal
        c_idx, c_cnt, f_vis, f_times = pack_chunk_to_ragged_csr(
            np.ascontiguousarray(dga_sub, dtype=np.complex64),
            t_chunk,
            threshold=float(uv_threshold),
            device="cpu",
        )
        if f_vis.size > 0:
            cpu_chunks.append((c_idx, c_cnt, f_vis, f_times))

    try:
        for spec in chunks:
            t_io0 = time.time()
            seg = spec.segments[0]
            tbl = table_cache.get(seg.ms_path)
            if tbl is None:
                tbl = table(seg.ms_path, readonly=True, ack=False)
                table_cache[seg.ms_path] = tbl
            _read_segment_columns(
                tbl,
                seg,
                data_column=data_column,
                nchan=nchan,
                nchan_full=nchan_full,
                ncorr=ncorr,
                chan_sel=chan_sel,
                out_uvw=buf_uvw,
                out_data=buf_data,
                out_wgt=buf_wgt,
                out_flag=buf_flag,
                out_flag_row=buf_flag_row,
                out_bl_idx=buf_bl_idx,
                out_t_rel=buf_t_rel,
            )
            io_s += float(time.time() - t_io0)

            t_cmp0 = time.time()
            if spec.block_idx != cur_block_idx:
                cur_block_idx = spec.block_idx
                tail_vis = None
                tail_wgt = None
                tail_uvw = None
                emit_cursor = spec.t_idx_start

            n_tail = int(tail_vis.shape[2]) if tail_vis is not None else 0
            nt_staged = n_tail + spec.nt_new
            vis_stage = np.zeros((nbl, nchan, nt_staged), dtype=np.complex64)
            wgt_stage = np.zeros((nbl, nchan, nt_staged), dtype=np.float32)
            uvw_stage = np.zeros((nbl, 3, nt_staged), dtype=np.float64)

            if n_tail > 0 and tail_vis is not None and tail_wgt is not None and tail_uvw is not None:
                vis_stage[:, :, :n_tail] = tail_vis
                wgt_stage[:, :, :n_tail] = tail_wgt
                uvw_stage[:, :, :n_tail] = tail_uvw

            nrow_k = seg.nrow
            if nrow_k > 0:
                stage_ms_rows_cpu(
                    bl_idx=buf_bl_idx[:nrow_k],
                    t_rel=buf_t_rel[:nrow_k],
                    uvw_rows=buf_uvw[:nrow_k],
                    data_rows=buf_data[:nrow_k],
                    wgt_rows=buf_wgt[:nrow_k],
                    flag_rows=buf_flag[:nrow_k],
                    flag_row=(
                        buf_flag_row[:nrow_k]
                        if (has_any_flag_row and seg.has_flag_row and buf_flag_row is not None)
                        else None
                    ),
                    nbl=nbl,
                    nt_stage=nt_staged,
                    t_offset=n_tail,
                    vis_3d=vis_stage,
                    wgt_3d=wgt_stage,
                    uvw_3d=uvw_stage,
                    corr_mode=corr_mode,
                    basis=basis,
                    single_pol=single_pol,
                    average_correlations=average_correlations,
                    corr_index=corr_index,
                    labels=meta["labels"],
                    lbl2idx=meta["lbl2idx"],
                    min_uv_distance_m=min_uv_distance_m,
                )

            if spec.is_last_in_block:
                need_pre_shift = (
                    nt_staged > _MAX_NT_EMIT
                    and regularize_time
                    and abs(spec.shift_samps) > 1e-7
                    and shift_domain == "native"
                )
                if need_pre_shift:
                    src_v = timeshift.shift_time_series(vis_stage, spec.shift_samps, method=shift_method, axis=-1)
                    src_w = np.maximum(
                        timeshift.shift_time_series(wgt_stage, spec.shift_samps, method=shift_method, axis=-1),
                        np.float32(0.0),
                    )
                    src_u = timeshift.shift_time_series(uvw_stage, spec.shift_samps, method=shift_method, axis=-1)
                else:
                    src_v, src_w, src_u = vis_stage, wgt_stage, uvw_stage

                rem = nt_staged
                off_rem = 0
                while rem > 0:
                    nt_emit = min(_MAX_NT_EMIT, rem)
                    cur_v = np.ascontiguousarray(src_v[:, :, off_rem:])
                    cur_w = np.ascontiguousarray(src_w[:, :, off_rem:])
                    cur_u = np.ascontiguousarray(src_u[:, :, off_rem:])
                    _emit_cpu(
                        cur_v, cur_w, cur_u, nt_emit, emit_cursor, spec.shift_samps, already_shifted=need_pre_shift
                    )
                    emit_cursor += nt_emit
                    off_rem += nt_emit
                    rem -= nt_emit
                tail_vis = tail_wgt = tail_uvw = None
            else:
                nt_emit = min(_MAX_NT_EMIT, max(0, nt_staged - req_overlap))
                n_next_tail = nt_staged - nt_emit
                if n_next_tail > 0:
                    tail_vis = np.ascontiguousarray(vis_stage[:, :, nt_emit:])
                    tail_wgt = np.ascontiguousarray(wgt_stage[:, :, nt_emit:])
                    tail_uvw = np.ascontiguousarray(uvw_stage[:, :, nt_emit:])
                else:
                    tail_vis = tail_wgt = tail_uvw = None

                if nt_emit > 0:
                    _emit_cpu(vis_stage, wgt_stage, uvw_stage, nt_emit, emit_cursor, spec.shift_samps)
                    emit_cursor += nt_emit

            compute_s += float(time.time() - t_cmp0)
    finally:
        close_errs = []
        for path, tbl in table_cache.items():
            try:
                tbl.close()
            except Exception as exc:
                close_errs.append((path, exc))
        table_cache.clear()
        if close_errs:
            err_msgs = "; ".join(f"{p}: {e}" for p, e in close_errs)
            logger.warning("Errors while closing casacore tables in _stream_cpu: %s", err_msgs)

    t_cons0 = time.time()
    host_store = _consolidate_host_chunks_in_memory(
        cpu_chunks=cpu_chunks,
        sparse_plan_uv=sparse_plan_uv_host,
        active_only=bool(active_only),
        attrs=common_attrs,
        out_dir=out_dir,
        write_to_disk=bool(write_to_disk),
    )
    consolidate_s = float(time.time() - t_cons0)
    total_s = float(time.time() - t_wall0)
    logger.info(
        "[Direct Raster CPU] Consolidated M=%d (padded=%d), NNZ=%d across %d rows | "
        "io=%.2fs cpu=%.2fs consolidate=%.2fs total=%.2fs",
        host_store.m_total,
        m_padded,
        host_store.nnz,
        total_rows,
        io_s,
        compute_s,
        consolidate_s,
        total_s,
    )
    return DirectRasterResult(
        gpu_store=None,
        store=host_store,
        csr_matrix=None,
        attrs=host_store.attrs,
        time_info=time_info,
        io_s=io_s,
        gpu_s=compute_s,
        consolidate_s=consolidate_s,
        total_s=total_s,
        nrows=total_rows,
        m_total=host_store.m_total,
        nnz_total=host_store.nnz,
        device="cpu",
    )


def stream_ms_to_ragged_csr(
    msname: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]],
    *,
    out_dir: Optional[str] = None,
    write_to_disk: bool = False,
    overwrite: bool = True,
    start_time_idx: Optional[int] = None,
    end_time_idx: Optional[int] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    time_step: Optional[float] = None,
    time_tolerance: float = 1e-3,
    max_blank_factor: float = 10.0,
    regularize_time: bool = True,
    sync_tolerance: float = 0.05,
    shift_method: str = "fft",
    shift_domain: str = "native",
    spw: int = 0,
    chan_sel: Optional[Tuple[int, Optional[int]]] = None,
    data_column: str = "DATA",
    corr_mode: str = "single",
    single_pol: str = "XX",
    basis: str = "linear",
    average_correlations: bool = True,
    corr_index: Optional[int] = None,
    use_weight_spectrum: bool = True,
    field_name: Optional[str] = None,
    npix_x: int = 384,
    npix_y: int = 384,
    pixsize_arcsec: float = 22.0,
    pix_rad: Optional[float] = None,
    grid_nu: Optional[int] = None,
    nv: Optional[int] = None,
    dm: float = 0.0,
    dedisp_mode: str = "fullband",
    align_to: str = "fmax",
    collapse_channels: bool = False,
    nsubbands: int = 1,
    grid_freq_bins: Optional[int] = None,
    exact_uvw: bool = True,
    mode: str = "wgridder",
    epsilon: float = 1e-5,
    do_wgridding: bool = False,
    sigma_min: float = 1.1,
    sigma_max: float = 2.6,
    center_x: float = 0.0,
    center_y: float = 0.0,
    allow_nshift: bool = True,
    apply_w_phase: bool = True,
    flip_u: bool = False,
    flip_v: bool = False,
    flip_w: bool = False,
    divide_by_n: bool = True,
    normalise_mode: Optional[str] = None,
    oversample: Optional[bool] = None,
    supp: Optional[int] = None,
    min_valid_channels: int = 24,
    min_uv_distance_m: Optional[float] = None,
    uv_threshold: float = 6e-6,
    row_chunk_size: int = 300_000,
    sparse_grid_plan: Optional[Dict[str, Any]] = None,
    active_only: bool = True,
    device: str = "auto",
    device_id: int = 0,
    return_csr_matrix: bool = False,
    use_numba: bool = True,
) -> DirectRasterResult:
    """
    Stream Measurement Set(s) with double-buffered pinned host memory and
    asynchronous CUDA streams directly into a consolidated ragged CSR UV store.

    Parameters
    ----------
    msname : str, PathLike, or sequence of str/PathLike
        Input Measurement Set path(s) for a single beam.
    out_dir : str, optional
        Output directory when ``write_to_disk=True``.
    write_to_disk : bool, default False
        Persist final consolidated ``RaggedDGAStore`` files to ``out_dir``.
        Intermediate ``.csr.bin`` fragment files are never written.
    overwrite : bool, default True
        Replace existing ``out_dir`` if ``write_to_disk=True``.
    npix_x, npix_y : int
        Image-plane dimensions in pixels.
    pixsize_arcsec : float, default 22.0
        Angular pixel scale in arcseconds/pixel.
    dm : float, default 0.0
        Dispersion measure in pc cm^-3.
    mode : {'nearest', 'bilinear'}, default 'nearest'
        Direct-to-grid Hermitian half-plane splatting kernel.
    row_chunk_size : int, default 300_000
        Target MS rows per timestamp-aligned chunk (``nt_emit`` is clamped
        ``<= 4096`` so ``uint16`` fragment cell counts never overflow).
    device : {'auto', 'cuda', 'cpu'}, default 'auto'
        Execution backend selector.
    return_csr_matrix : bool, default False
        When running on CUDA, also construct and attach a zero-copy
        ``cupyx.scipy.sparse.csr_matrix`` of shape ``(M_active, total_planes)``.

    Returns
    -------
    DirectRasterResult
        Result container (also accessible as a dict) with ``gpu_store``,
        ``store``, ``csr_matrix``, ``attrs``, ``time_info``, and timing telemetry.
    """
    t_wall0 = time.time()
    eff_device = _resolve_device(device)
    ms_paths = _normalise_ms_list(msname)

    if write_to_disk and out_dir and os.path.exists(out_dir):
        if not overwrite:
            raise FileExistsError(f"Output directory '{out_dir}' already exists and overwrite is False.")
        shutil.rmtree(out_dir, ignore_errors=True)

    t_pre0 = time.time()
    meta = _prescan_ms_metadata(
        ms_paths,
        spw=int(spw),
        chan_sel=chan_sel,
        data_column=str(data_column),
        use_weight_spectrum=bool(use_weight_spectrum),
        field_name=field_name,
        start_time_idx=start_time_idx,
        end_time_idx=end_time_idx,
        start_time=start_time,
        end_time=end_time,
        time_step=time_step,
        time_tolerance=float(time_tolerance),
        max_blank_factor=float(max_blank_factor),
        regularize_time=bool(regularize_time),
        sync_tolerance=float(sync_tolerance),
    )
    t_prescan = float(time.time() - t_pre0)

    eff_pix_rad = float(pix_rad) if pix_rad is not None else (float(pixsize_arcsec) / 206265.0)
    eff_pix_arcsec = eff_pix_rad * 206265.0
    geom = compute_raster_grid_geometry(
        npix_x=int(npix_x),
        npix_y=int(npix_y),
        pix_rad=eff_pix_rad,
        grid_nu=grid_nu,
        nv=nv,
        mode=str(mode),
        epsilon=float(epsilon),
        do_wgridding=bool(do_wgridding),
        sigma_min=float(sigma_min),
        sigma_max=float(sigma_max),
        oversample=oversample,
    )
    eff_grid_nu = int(geom["grid_nu"])
    eff_nv = int(geom["nv"])
    u_half = int(geom["u_half"])

    wgridder_kwargs = {
        "epsilon": float(epsilon),
        "do_wgridding": bool(do_wgridding),
        "sigma_min": float(sigma_min),
        "sigma_max": float(sigma_max),
        "center_x": float(center_x),
        "center_y": float(center_y),
        "allow_nshift": bool(allow_nshift),
        "apply_w_phase": bool(apply_w_phase),
        "flip_u": bool(flip_u),
        "flip_v": bool(flip_v),
        "flip_w": bool(flip_w),
        "divide_by_n": bool(divide_by_n),
        "normalise_mode": normalise_mode,
        "oversample": oversample,
        "supp": supp,
    }

    eff_dedisp_mode = str(dedisp_mode).strip().lower()
    if eff_dedisp_mode not in ("subbanded", "fullband"):
        raise ValueError(
            f"Unsupported dedisp_mode='{dedisp_mode}'; expected 'subbanded' or 'fullband'."
        )

    chan_freq = meta["chan_freq"]
    dt_nominal = float(meta["time_info"]["dt_nominal"])
    if eff_dedisp_mode == "fullband" and grid_freq_bins is not None and int(grid_freq_bins) > 1:
        eff_nsubbands = int(grid_freq_bins)
    else:
        eff_nsubbands = int(nsubbands)
    intra_delays_host: Optional[np.ndarray] = None
    inter_delays_host: Optional[np.ndarray] = None
    max_inter_delay = 0

    if eff_dedisp_mode == "subbanded":
        eff_nsubbands = max(1, min(eff_nsubbands, len(chan_freq))) if len(chan_freq) > 0 else 1
        plan = dedisp.DedispersionPlan(
            freqs_hz=chan_freq,
            tsamp_s=dt_nominal,
            dm_min=float(dm),
            dm_max=float(dm),
            dedisp_mode="subbanded",
            n_subbands=eff_nsubbands,
        )
        if float(dm) != 0.0 and len(chan_freq) > 0:
            intra_delays_host = plan.get_intra_delays(float(dm), align_to=align_to).astype(np.int32)
            inter_delays_host = plan.get_inter_delays(float(dm), align_to=align_to).astype(np.int32)
            if np.any(intra_delays_host < 0) or np.any(inter_delays_host < 0):
                raise ValueError(
                    "Streaming two-tier subbanded dedispersion requires non-negative causal delays "
                    "(use align_to='fmax' with dm >= 0)."
                )
            max_inter_delay = int(np.max(inter_delays_host)) if inter_delays_host.size > 0 else 0
            max_intra_delay = int(np.max(intra_delays_host)) if intra_delays_host.size > 0 else 0
            max_delay = max_intra_delay + max_inter_delay
        else:
            intra_delays_host = np.zeros(len(chan_freq), dtype=np.int32)
            inter_delays_host = np.zeros(eff_nsubbands, dtype=np.int32)
            max_inter_delay = 0
            max_delay = 0
        delays_host = intra_delays_host
    else:
        if float(dm) != 0.0 and len(chan_freq) > 0:
            delays_host = dedisp.get_dm_delays(
                chan_freq, float(dm), dt_nominal, align_to=align_to
            ).astype(np.int32)
            if np.any(delays_host < 0):
                raise ValueError(
                    "Streaming lookahead dedispersion requires non-negative causal delays "
                    "(use align_to='fmax' with dm >= 0)."
                )
            max_delay = int(np.max(delays_host))
        else:
            delays_host = np.zeros(len(chan_freq), dtype=np.int32)
            max_delay = 0

    if eff_device == "cuda":
        return _stream_gpu(
            meta=meta,
            geom=geom,
            eff_grid_nu=eff_grid_nu,
            eff_nv=eff_nv,
            u_half=u_half,
            npix_x=int(npix_x),
            npix_y=int(npix_y),
            pixsize_arcsec=eff_pix_arcsec,
            delays_host=delays_host,
            max_delay=max_delay,
            dedisp_mode=eff_dedisp_mode,
            intra_delays_host=intra_delays_host,
            inter_delays_host=inter_delays_host,
            max_inter_delay=max_inter_delay,
            data_column=str(data_column),
            chan_sel=chan_sel,
            corr_mode=str(corr_mode),
            basis=str(basis),
            single_pol=str(single_pol),
            average_correlations=bool(average_correlations),
            corr_index=corr_index,
            dm=float(dm),
            align_to=str(align_to),
            collapse_channels=bool(collapse_channels),
            nsubbands=eff_nsubbands,
            exact_uvw=bool(exact_uvw),
            mode=str(mode),
            wgridder_kwargs=wgridder_kwargs,
            min_valid_channels=int(min_valid_channels),
            min_uv_distance_m=min_uv_distance_m,
            uv_threshold=float(uv_threshold),
            row_chunk_size=int(row_chunk_size),
            regularize_time=bool(regularize_time),
            shift_method=str(shift_method),
            shift_domain=str(shift_domain),
            sparse_grid_plan=sparse_grid_plan,
            active_only=bool(active_only),
            out_dir=out_dir,
            write_to_disk=bool(write_to_disk),
            device_id=int(device_id),
            return_csr_matrix=bool(return_csr_matrix),
            t_prescan=t_prescan,
            t_wall0=t_wall0,
        )

    return _stream_cpu(
        meta=meta,
        geom=geom,
        eff_grid_nu=eff_grid_nu,
        eff_nv=eff_nv,
        u_half=u_half,
        npix_x=int(npix_x),
        npix_y=int(npix_y),
        pixsize_arcsec=eff_pix_arcsec,
        delays_host=delays_host,
        max_delay=max_delay,
        dedisp_mode=eff_dedisp_mode,
        intra_delays_host=intra_delays_host,
        inter_delays_host=inter_delays_host,
        max_inter_delay=max_inter_delay,
        data_column=str(data_column),
        chan_sel=chan_sel,
        corr_mode=str(corr_mode),
        basis=str(basis),
        single_pol=str(single_pol),
        average_correlations=bool(average_correlations),
        corr_index=corr_index,
        dm=float(dm),
        align_to=str(align_to),
        collapse_channels=bool(collapse_channels),
        nsubbands=eff_nsubbands,
        exact_uvw=bool(exact_uvw),
        mode=str(mode),
        wgridder_kwargs=wgridder_kwargs,
        min_valid_channels=int(min_valid_channels),
        min_uv_distance_m=min_uv_distance_m,
        uv_threshold=float(uv_threshold),
        row_chunk_size=int(row_chunk_size),
        regularize_time=bool(regularize_time),
        shift_method=str(shift_method),
        shift_domain=str(shift_domain),
        sparse_grid_plan=sparse_grid_plan,
        active_only=bool(active_only),
        out_dir=out_dir,
        write_to_disk=bool(write_to_disk),
        use_numba=bool(use_numba),
        t_prescan=t_prescan,
        t_wall0=t_wall0,
    )


def iter_ms_to_ragged_csr_subbanded(
    msname: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]],
    dm_trials: Sequence[float],
    *,
    dm_indices: Optional[Sequence[int]] = None,
    tolerance_samples: float = 1.25,
    start_time_idx: Optional[int] = None,
    end_time_idx: Optional[int] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    time_step: Optional[float] = None,
    time_tolerance: float = 1e-3,
    max_blank_factor: float = 10.0,
    regularize_time: bool = True,
    sync_tolerance: float = 0.05,
    shift_method: str = "fft",
    shift_domain: str = "native",
    spw: int = 0,
    chan_sel: Optional[Tuple[int, Optional[int]]] = None,
    data_column: str = "DATA",
    corr_mode: str = "single",
    single_pol: str = "XX",
    basis: str = "linear",
    average_correlations: bool = True,
    corr_index: Optional[int] = None,
    use_weight_spectrum: bool = True,
    field_name: Optional[str] = None,
    npix_x: int = 384,
    npix_y: int = 384,
    pixsize_arcsec: float = 22.0,
    pix_rad: Optional[float] = None,
    grid_nu: Optional[int] = None,
    nv: Optional[int] = None,
    align_to: str = "fmax",
    nsubbands: int = 16,
    exact_uvw: bool = True,
    mode: str = "wgridder",
    epsilon: float = 1e-5,
    sigma_min: float = 1.1,
    sigma_max: float = 2.6,
    center_x: float = 0.0,
    center_y: float = 0.0,
    allow_nshift: bool = True,
    apply_w_phase: bool = True,
    flip_u: bool = False,
    flip_v: bool = False,
    flip_w: bool = False,
    divide_by_n: bool = True,
    normalise_mode: Optional[str] = None,
    oversample: Optional[bool] = None,
    supp: Optional[int] = None,
    min_valid_channels: int = 24,
    min_uv_distance_m: Optional[float] = None,
    uv_threshold: float = 6e-6,
    row_chunk_size: int = 300_000,
    sparse_grid_plan: Optional[Dict[str, Any]] = None,
    active_only: bool = True,
    device: str = "auto",
    device_id: int = 0,
    return_csr_matrix: bool = False,
    use_numba: bool = True,
    scratch_dir: Optional[str] = None,
    max_fine_per_pass: Optional[int] = None,
) -> Iterator[Tuple[int, float, DirectRasterResult]]:
    """
    Two-Tier Subbanded multi-DM streaming generator.

    Streams the Measurement Set **once per coarse DM trial** (rather than once
    per fine DM trial):
      - Tier 1: Coarse intra-subband dedispersion & active UV-cell rasterisation
        into ``V_active[M_active, N_sub, nt_tier1]`` once per chunk.
      - Tier 2: Vectorised active-cell dedispersion across all fine DM trials
        assigned to the coarse bin, followed by CSR fragment packing.
      - Yields ``(dm_idx, dm_val, DirectRasterResult)`` one fine DM at a time
        so downstream GPU NUFFT search can process and release each store
        within bounded VRAM.
    """
    dm_list = [float(d) for d in dm_trials]
    if not dm_list:
        return
    idx_list = list(range(len(dm_list))) if dm_indices is None else [int(i) for i in dm_indices]
    if len(idx_list) != len(dm_list):
        raise ValueError("dm_indices length must match dm_trials length.")

    t_wall0 = time.time()
    eff_device = _resolve_device(device)
    ms_paths = _normalise_ms_list(msname)

    t_pre0 = time.time()
    meta = _prescan_ms_metadata(
        ms_paths,
        spw=int(spw),
        chan_sel=chan_sel,
        data_column=str(data_column),
        use_weight_spectrum=bool(use_weight_spectrum),
        field_name=field_name,
        start_time_idx=start_time_idx,
        end_time_idx=end_time_idx,
        start_time=start_time,
        end_time=end_time,
        time_step=time_step,
        time_tolerance=float(time_tolerance),
        max_blank_factor=float(max_blank_factor),
        regularize_time=bool(regularize_time),
        sync_tolerance=float(sync_tolerance),
    )
    t_prescan = float(time.time() - t_pre0)

    eff_pix_rad = float(pix_rad) if pix_rad is not None else (float(pixsize_arcsec) / 206265.0)
    eff_pix_arcsec = eff_pix_rad * 206265.0
    geom = compute_raster_grid_geometry(
        npix_x=int(npix_x),
        npix_y=int(npix_y),
        pix_rad=eff_pix_rad,
        grid_nu=grid_nu,
        nv=nv,
        mode=str(mode),
        epsilon=float(epsilon),
        do_wgridding=False,
        sigma_min=float(sigma_min),
        sigma_max=float(sigma_max),
        oversample=oversample,
    )
    eff_grid_nu = int(geom["grid_nu"])
    eff_nv = int(geom["nv"])
    u_half = int(geom["u_half"])

    wgridder_kwargs = {
        "epsilon": float(epsilon),
        "sigma_min": float(sigma_min),
        "sigma_max": float(sigma_max),
        "center_x": float(center_x),
        "center_y": float(center_y),
        "allow_nshift": bool(allow_nshift),
        "apply_w_phase": bool(apply_w_phase),
        "flip_u": bool(flip_u),
        "flip_v": bool(flip_v),
        "flip_w": bool(flip_w),
        "divide_by_n": bool(divide_by_n),
        "normalise_mode": normalise_mode,
        "oversample": oversample,
        "supp": supp,
    }

    chan_freq = meta["chan_freq"]
    time_info = meta["time_info"]
    dt_nominal = float(time_info["dt_nominal"])
    total_planes = int(time_info["total_planes"])
    grid_indices = time_info["grid_indices"]
    nbl = int(meta["nbl"])
    nchan = int(meta["nchan"])
    nchan_full = int(meta["nchan_full"])
    ncorr = int(meta["ncorr"])

    eff_nsub = max(1, min(int(nsubbands), nchan)) if nchan > 0 else 1
    plan = dedisp.DedispersionPlan(
        freqs_hz=chan_freq,
        tsamp_s=dt_nominal,
        dm_min=float(min(dm_list)),
        dm_max=float(max(dm_list)),
        tolerance_samples=float(tolerance_samples),
        dedisp_mode="subbanded",
        n_subbands=eff_nsub,
    )

    # Group supplied fine DM trials onto the plan's coarse DM centres, splitting
    # large coarse bins into memory-safe sub-passes for long whole-observation runs.
    coarse_groups_raw: Dict[int, List[int]] = {}
    for pos_i, dm_f in enumerate(dm_list):
        ic = int(np.argmin(np.abs(plan.coarse_dms - dm_f)))
        coarse_groups_raw.setdefault(ic, []).append(pos_i)

    eff_max_fine = max_fine_per_pass
    if eff_max_fine is None:
        # Cap concurrent fine DMs per pass on long (>20k plane) observations so
        # in-memory CSR fragments stay comfortably within host RAM with zero disk I/O.
        eff_max_fine = 2 if total_planes > 20_000 else max(1, len(dm_list))
    eff_max_fine = max(1, int(eff_max_fine))

    coarse_passes: List[Tuple[int, List[int]]] = []
    for ic, positions in sorted(coarse_groups_raw.items()):
        for s_i in range(0, len(positions), eff_max_fine):
            coarse_passes.append((ic, positions[s_i : s_i + eff_max_fine]))

    logger.info(
        "[Subbanded Multi-DM] Partitioned %d fine DM trials into %d coarse MS passes "
        "(n_subbands=%d, coarse_dm_step=%.2f pc cm^-3, max_fine_per_pass=%d, prescan=%.2fs).",
        len(dm_list),
        len(coarse_passes),
        eff_nsub,
        plan.coarse_dm_step,
        eff_max_fine,
        t_prescan,
    )

    sparse_plan_uv_host, flat_indices_host = _resolve_plan_buffers(sparse_grid_plan, eff_nv, u_half)
    m_padded = int(sparse_plan_uv_host.shape[0])
    mode_id = _normalise_mode(mode)
    use_snap_norm, _ = _resolve_normalise_mode(mode_id, normalise_mode)

    need_shift_halo = bool(
        regularize_time
        and any(abs(float(b.get("shift_samps", 0.0))) > 1e-7 for b in time_info["blocks"])
    )
    has_any_wgt_spec = any(e.has_weight_spectrum for e in meta["entries"])
    has_any_flag_row = any(e.has_flag_row for e in meta["entries"])

    for coarse_pass_i, (ic, fine_positions) in enumerate(coarse_passes):
        t_pass0 = time.time()
        dm_coarse = float(plan.coarse_dms[ic])
        fine_dms_bin = [dm_list[p] for p in fine_positions]
        fine_ids_bin = [idx_list[p] for p in fine_positions]
        n_fine_bin = len(fine_dms_bin)

        intra_delays_host = plan.get_intra_delays(dm_coarse, align_to=align_to).astype(np.int32)
        inter_delays_bin = np.stack(
            [plan.get_inter_delays(dm_f, align_to=align_to).astype(np.int32) for dm_f in fine_dms_bin],
            axis=0,
        )
        if np.any(intra_delays_host < 0) or np.any(inter_delays_bin < 0):
            raise ValueError(
                "Two-tier subbanded multi-DM streaming requires non-negative causal delays "
                "(use align_to='fmax' with dm >= 0)."
            )
        max_intra = int(np.max(intra_delays_host)) if intra_delays_host.size > 0 else 0
        max_inter = int(np.max(inter_delays_bin)) if inter_delays_bin.size > 0 else 0
        max_delay = max_intra + max_inter
        req_overlap = max(max_delay, _SHIFT_HALO_SAMPLES) if need_shift_halo else max_delay

        logger.info(
            "[Subbanded Pass %d/%d] Coarse DM=%.2f pc cm^-3 covering %d fine trial(s): %s "
            "(max_intra=%d, max_inter=%d samps)",
            coarse_pass_i + 1,
            len(coarse_passes),
            dm_coarse,
            n_fine_bin,
            [round(d, 2) for d in fine_dms_bin],
            max_intra,
            max_inter,
        )

        if eff_device == "cuda":
            from fastducc.ragged_gpu import consolidate_device_chunks_to_gpu_store

            host_chunks_per_dm: List[List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = [
                [] for _ in range(n_fine_bin)
            ]

            with cp.cuda.Device(int(device_id)):
                free_vram, _ = cp.cuda.Device(int(device_id)).mem_info
                row_bytes_est = (
                    24 + nchan * ncorr * 9 + (nchan * ncorr * 4 if has_any_wgt_spec else ncorr * 4) + 9
                )
                max_rows_vram = max(1, int(0.20 * free_vram) // max(1, row_bytes_est))
                eff_row_chunk = min(int(row_chunk_size), max_rows_vram)

                # Account for Tier 1 active_dga (M * N_sub * 12B) + Tier 2 batch (n_fine * M * 8B)
                stage_bytes_per_t = max(
                    1,
                    (24 * nbl * nchan)
                    + (48 * nbl)
                    + (16 * eff_nsub * m_padded)
                    + (10 * n_fine_bin * m_padded),
                )
                vram_nt_cap = max(32, int(0.30 * free_vram) // stage_bytes_per_t)
                max_nt_new = min(_MAX_NT_EMIT, max(16, vram_nt_cap - req_overlap))

                chunks, max_chunk_rows, total_rows = _plan_streaming_chunks(
                    meta, row_chunk_size=eff_row_chunk, max_nt_new=max_nt_new
                )

                io_wait_s = t_prescan if coarse_pass_i == 0 else 0.0
                gpu_compute_s = 0.0

                if chunks:
                    max_nt_stage = max(req_overlap + max(c.nt_new for c in chunks), req_overlap + 1)
                    tail_cap = max(1, req_overlap)

                    h2d_stream = cp.cuda.Stream(non_blocking=True)
                    compute_stream = cp.cuda.Stream(non_blocking=True)
                    slot_h2d_done: List[Optional[Any]] = [None, None]
                    ev_stage_done: Optional[Any] = None

                    host_pools = [
                        _alloc_pinned_pool(max_chunk_rows, nchan, ncorr, has_any_wgt_spec, has_any_flag_row),
                        _alloc_pinned_pool(max_chunk_rows, nchan, ncorr, has_any_wgt_spec, has_any_flag_row),
                    ]
                    wgt_row_shape = (max_chunk_rows, nchan, ncorr) if has_any_wgt_spec else (max_chunk_rows, ncorr)
                    d_rows = {
                        "uvw": cp.empty((max_chunk_rows, 3), dtype=cp.float64),
                        "data": cp.empty((max_chunk_rows, nchan, ncorr), dtype=cp.complex64),
                        "wgt": cp.empty(wgt_row_shape, dtype=cp.float32),
                        "flag": cp.empty((max_chunk_rows, nchan, ncorr), dtype=cp.uint8),
                        "bl_idx": cp.empty((max_chunk_rows,), dtype=cp.int32),
                        "t_rel": cp.empty((max_chunk_rows,), dtype=cp.int32),
                        "flag_row": cp.empty((max_chunk_rows,), dtype=cp.uint8) if has_any_flag_row else None,
                    }
                    d_vis_pool = cp.empty(nbl * nchan * max_nt_stage, dtype=cp.complex64)
                    d_wgt_pool = cp.empty(nbl * nchan * max_nt_stage, dtype=cp.float32)
                    d_uvw_pool = cp.empty(nbl * 3 * max_nt_stage, dtype=cp.float64)
                    d_vis_tail_pool = cp.empty(nbl * nchan * tail_cap, dtype=cp.complex64)
                    d_wgt_tail_pool = cp.empty(nbl * nchan * tail_cap, dtype=cp.float32)
                    d_uvw_tail_pool = cp.empty(nbl * 3 * tail_cap, dtype=cp.float64)
                    d_flat_indices = cp.asarray(flat_indices_host, dtype=cp.int64)
                    d_inter_delays_bin = cp.asarray(inter_delays_bin, dtype=cp.int32)

                    table_cache: Dict[str, Any] = {}

                    def _io_read_chunk(chunk_spec: _ChunkSpec, slot: int) -> Tuple[_ChunkSpec, int, float]:
                        t0_io = time.time()
                        ev = slot_h2d_done[slot]
                        if ev is not None:
                            ev.synchronize()
                        pool = host_pools[slot]
                        seg = chunk_spec.segments[0]
                        tbl = table_cache.get(seg.ms_path)
                        if tbl is None:
                            tbl = table(seg.ms_path, readonly=True, ack=False)
                            table_cache[seg.ms_path] = tbl
                        _read_segment_columns(
                            tbl,
                            seg,
                            data_column=data_column,
                            nchan=nchan,
                            nchan_full=nchan_full,
                            ncorr=ncorr,
                            chan_sel=chan_sel,
                            out_uvw=pool["uvw"],
                            out_data=pool["data"],
                            out_wgt=pool["wgt"],
                            out_flag=pool["flag"],
                            out_flag_row=pool.get("flag_row"),
                            out_bl_idx=pool["bl_idx"],
                            out_t_rel=pool["t_rel"],
                        )
                        return chunk_spec, seg.nrow, float(time.time() - t0_io)

                    def _io_close_tables() -> None:
                        for path, tbl in table_cache.items():
                            try:
                                tbl.close()
                            except Exception as exc:
                                logger.warning("Error closing table %s: %s", path, exc)
                        table_cache.clear()

                    def _emit_multi_gpu(
                        d_vis_stage: "cp.ndarray",
                        d_wgt_stage: "cp.ndarray",
                        d_uvw_stage: "cp.ndarray",
                        nt_emit: int,
                        emit_cursor: int,
                        shift_samps: float,
                        already_shifted: bool = False,
                    ) -> None:
                        if nt_emit <= 0:
                            return
                        if (not already_shifted) and regularize_time and abs(shift_samps) > 1e-7 and shift_domain == "native":
                            w_vis = _shift_time_series_gpu(d_vis_stage, shift_samps, method=shift_method, axis=-1)
                            w_wgt = cp.maximum(
                                _shift_time_series_gpu(d_wgt_stage, shift_samps, method=shift_method, axis=-1),
                                np.float32(0.0),
                            )
                            w_uvw = _shift_time_series_gpu(d_uvw_stage, shift_samps, method=shift_method, axis=-1)
                        else:
                            w_vis, w_wgt, w_uvw = d_vis_stage, d_wgt_stage, d_uvw_stage

                        nt_tier1 = min(int(w_vis.shape[2]), nt_emit + max_inter)
                        tier1_res = dedisperse_and_rasterise_subband_gpu(
                            vis_3d=w_vis,
                            wgt_3d=w_wgt,
                            uvw_3d=w_uvw,
                            freqs_hz=chan_freq,
                            intra_delays=intra_delays_host,
                            n_subbands=eff_nsub,
                            nt_out=nt_tier1,
                            dm_coarse=dm_coarse,
                            dt=dt_nominal,
                            align_to=align_to,
                            exact_uvw=bool(exact_uvw),
                            npix_x=int(npix_x),
                            npix_y=int(npix_y),
                            pixsize_arcsec=float(eff_pix_arcsec),
                            pix_rad=float(geom["pix_rad"]),
                            grid_nu=int(eff_grid_nu),
                            nv=int(eff_nv),
                            mode=mode,
                            min_valid_channels=int(min_valid_channels),
                            min_uv_distance_m=min_uv_distance_m,
                            flat_indices=d_flat_indices,
                            m_active=m_padded,
                            return_weights=(not use_snap_norm),
                            return_host=False,
                            **wgridder_kwargs,
                        )
                        if use_snap_norm:
                            d_active_dga, d_active_wgt = tier1_res, None
                        else:
                            d_active_dga, d_active_wgt = tier1_res

                        # Tier 2: dedisperse ALL fine DMs in this coarse bin in a single kernel launch
                        d_dedisp_batch = dedisperse_active_cell_matrix_gpu(
                            d_active_dga,
                            d_inter_delays_bin,
                            nt_out=nt_emit,
                            active_wgt=d_active_wgt,
                            uv_threshold=float(uv_threshold),
                            return_host=False,
                        )
                        del d_active_dga
                        if d_active_wgt is not None:
                            del d_active_wgt

                        t_chunk_host = np.asarray(
                            grid_indices[emit_cursor : emit_cursor + nt_emit], dtype=np.float64
                        ) * dt_nominal
                        d_t_chunk = cp.asarray(t_chunk_host, dtype=cp.float64)

                        for j_dm in range(n_fine_bin):
                            d_slice = d_dedisp_batch[j_dm]
                            if regularize_time and abs(shift_samps) > 1e-7 and shift_domain == "gridded":
                                d_slice = _shift_time_series_gpu(d_slice, shift_samps, method=shift_method, axis=-1)
                            c_idx, c_cnt, v_data, v_times = pack_chunk_to_ragged_gpu(
                                d_slice,
                                d_t_chunk,
                                threshold=float(uv_threshold),
                                return_host=True,
                            )
                            if int(v_data.size) > 0:
                                host_chunks_per_dm[j_dm].append((c_idx, c_cnt, v_data, v_times))

                        del d_dedisp_batch, d_t_chunk
                        if w_vis is not d_vis_stage:
                            del w_vis, w_wgt, w_uvw

                    executor = ThreadPoolExecutor(max_workers=1)
                    try:
                        cur_block_idx = -1
                        n_tail = 0
                        emit_cursor = 0
                        next_fut = executor.submit(_io_read_chunk, chunks[0], 0)
                        n_chunks = len(chunks)

                        for k in range(n_chunks):
                            t_w0 = time.time()
                            chunk_spec, nrow_k, _ = next_fut.result()
                            io_wait_s += max(0.0, time.time() - t_w0)
                            slot = k % 2
                            if k + 1 < n_chunks:
                                next_fut = executor.submit(_io_read_chunk, chunks[k + 1], (k + 1) % 2)

                            t_gpu0 = time.time()
                            pool = host_pools[slot]
                            if chunk_spec.block_idx != cur_block_idx:
                                cur_block_idx = chunk_spec.block_idx
                                n_tail = 0
                                emit_cursor = chunk_spec.t_idx_start

                            nt_staged = n_tail + chunk_spec.nt_new
                            if nt_staged > max_nt_stage:
                                max_nt_stage = nt_staged
                                d_vis_pool = cp.empty(nbl * nchan * max_nt_stage, dtype=cp.complex64)
                                d_wgt_pool = cp.empty(nbl * nchan * max_nt_stage, dtype=cp.float32)
                                d_uvw_pool = cp.empty(nbl * 3 * max_nt_stage, dtype=cp.float64)

                            with h2d_stream:
                                if ev_stage_done is not None:
                                    h2d_stream.wait_event(ev_stage_done)
                                if nrow_k > 0:
                                    d_rows["uvw"][:nrow_k].set(pool["uvw"][:nrow_k], stream=h2d_stream)
                                    d_rows["data"][:nrow_k].set(pool["data"][:nrow_k], stream=h2d_stream)
                                    d_rows["wgt"][:nrow_k].set(pool["wgt"][:nrow_k], stream=h2d_stream)
                                    d_rows["flag"][:nrow_k].set(pool["flag"][:nrow_k], stream=h2d_stream)
                                    d_rows["bl_idx"][:nrow_k].set(pool["bl_idx"][:nrow_k], stream=h2d_stream)
                                    d_rows["t_rel"][:nrow_k].set(pool["t_rel"][:nrow_k], stream=h2d_stream)
                                    if has_any_flag_row and d_rows["flag_row"] is not None:
                                        d_rows["flag_row"][:nrow_k].set(pool["flag_row"][:nrow_k], stream=h2d_stream)
                                ev_h2d = cp.cuda.Event()
                                ev_h2d.record(h2d_stream)
                                slot_h2d_done[slot] = ev_h2d

                            with compute_stream:
                                compute_stream.wait_event(ev_h2d)
                                d_vis_stage = d_vis_pool[: nbl * nchan * nt_staged].reshape((nbl, nchan, nt_staged))
                                d_wgt_stage = d_wgt_pool[: nbl * nchan * nt_staged].reshape((nbl, nchan, nt_staged))
                                d_uvw_stage = d_uvw_pool[: nbl * 3 * nt_staged].reshape((nbl, 3, nt_staged))
                                d_vis_stage.fill(0)
                                d_wgt_stage.fill(0)
                                d_uvw_stage.fill(0)
                                if n_tail > 0:
                                    d_vis_stage[:, :, :n_tail] = d_vis_tail_pool[: nbl * nchan * n_tail].reshape((nbl, nchan, n_tail))
                                    d_wgt_stage[:, :, :n_tail] = d_wgt_tail_pool[: nbl * nchan * n_tail].reshape((nbl, nchan, n_tail))
                                    d_uvw_stage[:, :, :n_tail] = d_uvw_tail_pool[: nbl * 3 * n_tail].reshape((nbl, 3, n_tail))

                                if nrow_k > 0:
                                    stage_ms_rows_gpu(
                                        bl_idx=d_rows["bl_idx"][:nrow_k],
                                        t_rel=d_rows["t_rel"][:nrow_k],
                                        uvw_rows=d_rows["uvw"][:nrow_k],
                                        data_rows=d_rows["data"][:nrow_k],
                                        wgt_rows=d_rows["wgt"][:nrow_k],
                                        flag_rows=d_rows["flag"][:nrow_k],
                                        flag_row=(
                                            d_rows["flag_row"][:nrow_k]
                                            if (has_any_flag_row and chunk_spec.segments[0].has_flag_row and d_rows["flag_row"] is not None)
                                            else None
                                        ),
                                        nbl=nbl,
                                        nt_stage=nt_staged,
                                        t_offset=n_tail,
                                        vis_3d=d_vis_stage,
                                        wgt_3d=d_wgt_stage,
                                        uvw_3d=d_uvw_stage,
                                        corr_mode=corr_mode,
                                        basis=basis,
                                        single_pol=single_pol,
                                        average_correlations=average_correlations,
                                        corr_index=corr_index,
                                        labels=meta["labels"],
                                        lbl2idx=meta["lbl2idx"],
                                        min_uv_distance_m=min_uv_distance_m,
                                        return_host=False,
                                    )
                                ev_stage_done = cp.cuda.Event()
                                ev_stage_done.record(compute_stream)

                                if chunk_spec.is_last_in_block:
                                    need_pre_shift = (
                                        nt_staged > _MAX_NT_EMIT
                                        and regularize_time
                                        and abs(chunk_spec.shift_samps) > 1e-7
                                        and shift_domain == "native"
                                    )
                                    if need_pre_shift:
                                        src_vis = _shift_time_series_gpu(d_vis_stage, chunk_spec.shift_samps, method=shift_method, axis=-1)
                                        src_wgt = cp.maximum(
                                            _shift_time_series_gpu(d_wgt_stage, chunk_spec.shift_samps, method=shift_method, axis=-1),
                                            np.float32(0.0),
                                        )
                                        src_uvw = _shift_time_series_gpu(d_uvw_stage, chunk_spec.shift_samps, method=shift_method, axis=-1)
                                    else:
                                        src_vis, src_wgt, src_uvw = d_vis_stage, d_wgt_stage, d_uvw_stage

                                    rem = nt_staged
                                    offset_rem = 0
                                    while rem > 0:
                                        nt_emit = min(_MAX_NT_EMIT, rem)
                                        if offset_rem == 0:
                                            cur_vis, cur_wgt, cur_uvw = src_vis, src_wgt, src_uvw
                                        else:
                                            cur_vis = cp.ascontiguousarray(src_vis[:, :, offset_rem:])
                                            cur_wgt = cp.ascontiguousarray(src_wgt[:, :, offset_rem:])
                                            cur_uvw = cp.ascontiguousarray(src_uvw[:, :, offset_rem:])
                                        _emit_multi_gpu(
                                            cur_vis, cur_wgt, cur_uvw, nt_emit, emit_cursor,
                                            chunk_spec.shift_samps, already_shifted=need_pre_shift,
                                        )
                                        emit_cursor += nt_emit
                                        offset_rem += nt_emit
                                        rem -= nt_emit
                                    if need_pre_shift:
                                        del src_vis, src_wgt, src_uvw
                                    n_tail = 0
                                else:
                                    nt_emit = min(_MAX_NT_EMIT, max(0, nt_staged - req_overlap))
                                    n_next_tail = nt_staged - nt_emit
                                    if n_next_tail > tail_cap:
                                        tail_cap = n_next_tail
                                        d_vis_tail_pool = cp.empty(nbl * nchan * tail_cap, dtype=cp.complex64)
                                        d_wgt_tail_pool = cp.empty(nbl * nchan * tail_cap, dtype=cp.float32)
                                        d_uvw_tail_pool = cp.empty(nbl * 3 * tail_cap, dtype=cp.float64)
                                    if n_next_tail > 0:
                                        d_vis_tail_pool[: nbl * nchan * n_next_tail].reshape((nbl, nchan, n_next_tail))[:] = d_vis_stage[:, :, nt_emit:]
                                        d_wgt_tail_pool[: nbl * nchan * n_next_tail].reshape((nbl, nchan, n_next_tail))[:] = d_wgt_stage[:, :, nt_emit:]
                                        d_uvw_tail_pool[: nbl * 3 * n_next_tail].reshape((nbl, 3, n_next_tail))[:] = d_uvw_stage[:, :, nt_emit:]
                                    if nt_emit > 0:
                                        _emit_multi_gpu(
                                            d_vis_stage, d_wgt_stage, d_uvw_stage, nt_emit, emit_cursor,
                                            chunk_spec.shift_samps,
                                        )
                                        emit_cursor += nt_emit
                                    n_tail = n_next_tail

                            gpu_compute_s += max(0.0, time.time() - t_gpu0)

                        h2d_stream.synchronize()
                        compute_stream.synchronize()
                    finally:
                        try:
                            executor.submit(_io_close_tables).result(timeout=30)
                        except Exception as exc:
                            logger.warning("ThreadPoolExecutor failed closing tables cleanly: %s", exc)
                        executor.shutdown(wait=True)
                        host_pools.clear()
                        d_rows.clear()
                        del d_vis_pool, d_wgt_pool, d_uvw_pool
                        del d_vis_tail_pool, d_wgt_tail_pool, d_uvw_tail_pool
                        del d_flat_indices, d_inter_delays_bin
                        cp.get_default_memory_pool().free_all_blocks()

                pass_stream_s = float(time.time() - t_pass0)
                per_dm_stream_s = pass_stream_s / float(max(1, n_fine_bin))

                # Consolidate and yield each fine DM store one at a time to minimise peak VRAM
                d_sparse_uv = cp.asarray(sparse_plan_uv_host, dtype=cp.int32)
                for j_dm in range(n_fine_bin):
                    dm_val = fine_dms_bin[j_dm]
                    dm_idx_val = fine_ids_bin[j_dm]
                    t_cons0 = time.time()
                    common_attrs = _build_common_attrs(
                        meta=meta,
                        geom=geom,
                        eff_grid_nu=eff_grid_nu,
                        eff_nv=eff_nv,
                        u_half=u_half,
                        npix_x=npix_x,
                        npix_y=npix_y,
                        pixsize_arcsec=eff_pix_arcsec,
                        dm=dm_val,
                        mode=mode,
                        m_padded=m_padded,
                        dedisp_mode="subbanded",
                        nsubbands=eff_nsub,
                    )
                    cp.get_default_memory_pool().free_all_blocks()
                    dev_chunks_j = [
                        (cp.asarray(ci), cp.asarray(cc), cp.asarray(vd), cp.asarray(vt))
                        for ci, cc, vd, vt in host_chunks_per_dm[j_dm]
                    ]
                    host_chunks_per_dm[j_dm].clear()

                    gpu_dict = consolidate_device_chunks_to_gpu_store(
                        device_chunks=dev_chunks_j,
                        sparse_plan_uv=d_sparse_uv,
                        active_only=bool(active_only),
                        store_dir=None,
                        attrs=common_attrs,
                    )
                    del dev_chunks_j
                    out_attrs = dict(gpu_dict["attrs"])
                    m_active_j = int(out_attrs["m_total"])
                    nnz_j = int(out_attrs["nnz_total"])
                    gpu_store = from_gpu_buffers(
                        cell_offsets=gpu_dict["cell_offsets"],
                        cell_uv=gpu_dict["cell_uv"],
                        vis_data=gpu_dict["vis_data"],
                        vis_times=gpu_dict["vis_times"],
                        attrs=out_attrs,
                        store_dir="",
                        copy_to_host=False,
                        device_id=int(device_id),
                    )
                    csr_mat = (
                        build_gpu_csr_matrix(gpu_store, total_planes=total_planes, dt=dt_nominal)
                        if return_csr_matrix
                        else None
                    )
                    cons_s = float(time.time() - t_cons0)
                    yield dm_idx_val, dm_val, DirectRasterResult(
                        gpu_store=gpu_store,
                        store=None,
                        csr_matrix=csr_mat,
                        attrs=out_attrs,
                        time_info=time_info,
                        io_s=io_wait_s / float(max(1, n_fine_bin)),
                        gpu_s=gpu_compute_s / float(max(1, n_fine_bin)),
                        consolidate_s=cons_s,
                        total_s=per_dm_stream_s + cons_s,
                        nrows=total_rows,
                        m_total=m_active_j,
                        nnz_total=nnz_j,
                        device="cuda",
                    )
                del d_sparse_uv
                cp.get_default_memory_pool().free_all_blocks()
        else:
            # CPU fallback: stream each fine DM via stream_ms_to_ragged_csr(dedisp_mode="subbanded")
            for dm_idx_val, dm_val in zip(fine_ids_bin, fine_dms_bin):
                res = stream_ms_to_ragged_csr(
                    msname,
                    start_time_idx=start_time_idx,
                    end_time_idx=end_time_idx,
                    start_time=start_time,
                    end_time=end_time,
                    time_step=time_step,
                    time_tolerance=time_tolerance,
                    max_blank_factor=max_blank_factor,
                    regularize_time=regularize_time,
                    sync_tolerance=sync_tolerance,
                    shift_method=shift_method,
                    shift_domain=shift_domain,
                    spw=spw,
                    chan_sel=chan_sel,
                    data_column=data_column,
                    corr_mode=corr_mode,
                    single_pol=single_pol,
                    basis=basis,
                    average_correlations=average_correlations,
                    corr_index=corr_index,
                    use_weight_spectrum=use_weight_spectrum,
                    field_name=field_name,
                    npix_x=npix_x,
                    npix_y=npix_y,
                    pixsize_arcsec=pixsize_arcsec,
                    pix_rad=pix_rad,
                    grid_nu=grid_nu,
                    nv=nv,
                    dm=dm_val,
                    dedisp_mode="subbanded",
                    align_to=align_to,
                    nsubbands=eff_nsub,
                    exact_uvw=exact_uvw,
                    mode=mode,
                    epsilon=epsilon,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                    center_x=center_x,
                    center_y=center_y,
                    allow_nshift=allow_nshift,
                    apply_w_phase=apply_w_phase,
                    flip_u=flip_u,
                    flip_v=flip_v,
                    flip_w=flip_w,
                    divide_by_n=divide_by_n,
                    normalise_mode=normalise_mode,
                    oversample=oversample,
                    supp=supp,
                    min_valid_channels=min_valid_channels,
                    min_uv_distance_m=min_uv_distance_m,
                    uv_threshold=uv_threshold,
                    row_chunk_size=row_chunk_size,
                    sparse_grid_plan=sparse_grid_plan,
                    active_only=active_only,
                    device="cpu",
                    use_numba=use_numba,
                )
                yield dm_idx_val, dm_val, res


__all__ = [
    "DirectRasterResult",
    "iter_ms_to_ragged_csr_subbanded",
    "stream_ms_to_ragged_csr",
]
