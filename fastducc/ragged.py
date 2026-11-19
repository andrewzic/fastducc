# fastducc/ragged.py
"""
Direct Sharded Ragged Compressed Sparse Row (CSR) Dynamic Grid Array (DGA)
storage and streaming primitives for FastDUCC.

Replaces dense 2D/3D Zarr storage with independent sharded CSR fragments written
by Stage 1 workers, and provides lightweight zero-copy consolidation and reading
for Stage 2 temporal NUFFT processing.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import struct
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import numba as nb
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

logger = logging.getLogger(__name__)

# Fragment binary layout constants:
# 64-byte binary header:
# magic (4s: 'RCSR'), version (I: 1), chunk_idx (I), n_active_cells (I),
# nnz (Q: uint64), t_start (d: float64), dt (d: float64), threshold (f: float32), reserved (20s)
HEADER_MAGIC = b"RCSR"
HEADER_VERSION = 1
HEADER_FORMAT = "<4sIIIQddf20s"
HEADER_SIZE = 64


@dataclass
class RaggedDGAStore:
    """
    On-disk Compressed Sparse Row (CSR) ragged storage container for UV cell time-series.
    """
    store_dir: str
    attrs: Dict[str, Any]
    cell_offsets: np.ndarray  # (M + 1,) uint64
    cell_uv: np.ndarray       # (M, 2) int32 [v, u]
    vis_data: np.ndarray      # (NNZ,) complex64 (memmap)
    vis_times: np.ndarray     # (NNZ,) float64 (memmap), relative to t0

    @property
    def m_total(self) -> int:
        return int(self.cell_uv.shape[0])

    @property
    def nnz(self) -> int:
        return int(self.cell_offsets[-1]) if self.cell_offsets.size > 0 else 0

    @classmethod
    def open(cls, store_dir: str, mode: str = "r") -> "RaggedDGAStore":
        """
        Open a consolidated RaggedDGAStore. If fragments exist without consolidation,
        automatically consolidates before opening.
        """
        attrs_path = os.path.join(store_dir, "attrs.json")
        offsets_path = os.path.join(store_dir, "cell_offsets.bin")
        fragments_dir = os.path.join(store_dir, "fragments")

        if not os.path.exists(offsets_path) and os.path.isdir(fragments_dir):
            logger.info(f"Consolidating unstitched fragments in {store_dir}...")
            frag_files = [
                os.path.join(fragments_dir, f)
                for f in sorted(os.listdir(fragments_dir))
                if f.endswith(".csr.bin")
            ]
            sparse_uv = None
            uv_path = os.path.join(store_dir, "cell_uv.bin")
            if os.path.exists(uv_path):
                sparse_uv = np.fromfile(uv_path, dtype=np.int32).reshape(-1, 2)
            if sparse_uv is not None and len(frag_files) > 0:
                consolidate_ragged_fragments(
                    fragment_files=frag_files,
                    store_dir=store_dir,
                    sparse_plan_uv=sparse_uv,
                    cleanup_fragments=False,
                )

        if not os.path.exists(attrs_path):
            raise FileNotFoundError(f"RaggedDGAStore metadata missing at {attrs_path}")

        with open(attrs_path, "r", encoding="utf-8") as f:
            attrs = json.load(f)

        m_total = int(attrs["m_total"])
        nnz = int(attrs["nnz_total"])

        offsets = np.memmap(
            os.path.join(store_dir, "cell_offsets.bin"),
            mode=mode,
            dtype=np.uint64,
            shape=(m_total + 1,),
        )
        uv = np.memmap(
            os.path.join(store_dir, "cell_uv.bin"),
            mode=mode,
            dtype=np.int32,
            shape=(m_total, 2),
        )
        vis = np.memmap(
            os.path.join(store_dir, "vis_data.bin"),
            mode=mode,
            dtype=np.complex64,
            shape=(nnz,),
        )
        times = np.memmap(
            os.path.join(store_dir, "vis_times.bin"),
            mode=mode,
            dtype=np.float64,
            shape=(nnz,),
        )
        return cls(
            store_dir=store_dir,
            attrs=attrs,
            cell_offsets=offsets,
            cell_uv=uv,
            vis_data=vis,
            vis_times=times,
        )

    @classmethod
    def from_gpu_buffers(
        cls,
        cell_offsets: Any,
        cell_uv: Any,
        vis_data: Any,
        vis_times: Any,
        attrs: Optional[Dict[str, Any]] = None,
        store_dir: str = "",
        copy_to_host: bool = False,
        device_id: Optional[int] = None,
        write_to_disk: bool = False,
    ) -> Any:
        """
        Construct a GPU or host ragged CSR store from CuPy (or NumPy) CSR buffers.

        Parameters
        ----------
        cell_offsets : array_like (uint64), shape (M + 1,)
            CSR row pointers per active UV cell.
        cell_uv : array_like (int32), shape (M, 2)
            Integer grid coordinates ``[v, u]``.
        vis_data : array_like (complex64), shape (NNZ,)
            Non-zero complex visibility samples.
        vis_times : array_like (float64), shape (NNZ,)
            Sample timestamps in seconds relative to ``t0``.
        attrs : dict, optional
            Store metadata dictionary.
        store_dir : str, optional
            Directory label (and output directory when ``write_to_disk=True``).
        copy_to_host : bool, default False
            If False, retain buffers in VRAM as a :class:`GPURaggedDGAStore`.
            If True, copy buffers to host memory and return :class:`RaggedDGAStore`.
        device_id : int, optional
            Target CUDA device ID when ``copy_to_host=False``.
        write_to_disk : bool, default False
            When ``copy_to_host=True`` and ``store_dir`` is non-empty, serialise
            CSR binary files to disk.
        """
        from fastducc.gpu_nufft import from_gpu_buffers as _from_gpu_buffers

        return _from_gpu_buffers(
            cell_offsets=cell_offsets,
            cell_uv=cell_uv,
            vis_data=vis_data,
            vis_times=vis_times,
            attrs=attrs,
            store_dir=store_dir,
            copy_to_host=copy_to_host,
            device_id=device_id,
            write_to_disk=write_to_disk,
            store_cls=cls,
        )


if HAS_NUMBA:
    @nb.njit(fastmath=True)
    def _pack_chunk_ragged_numba(
        dga_sub: np.ndarray,
        t_chunk: np.ndarray,
        thr_sq: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        M, nt = dga_sub.shape
        n_active = 0
        total_nnz = 0
        for m in range(M):
            c = 0
            for t in range(nt):
                val = dga_sub[m, t]
                if (val.real * val.real + val.imag * val.imag) > thr_sq:
                    c += 1
            if c > 0:
                n_active += 1
                total_nnz += c

        cell_indices = np.empty(n_active, dtype=np.uint32)
        cell_counts = np.empty(n_active, dtype=np.uint16)
        vis_data = np.empty(total_nnz, dtype=np.complex64)
        vis_times = np.empty(total_nnz, dtype=np.float64)

        act_idx = 0
        nnz_idx = 0
        for m in range(M):
            c = 0
            for t in range(nt):
                val = dga_sub[m, t]
                if (val.real * val.real + val.imag * val.imag) > thr_sq:
                    vis_data[nnz_idx] = val
                    vis_times[nnz_idx] = t_chunk[t]
                    nnz_idx += 1
                    c += 1
            if c > 0:
                cell_indices[act_idx] = m
                cell_counts[act_idx] = c
                act_idx += 1

        return cell_indices, cell_counts, vis_data, vis_times


    @nb.njit(fastmath=True)
    def _scatter_fragment_samples_numba(
        c_idx: np.ndarray,
        c_cnt: np.ndarray,
        f_vis: np.ndarray,
        f_times: np.ndarray,
        cell_to_active: np.ndarray,
        cur_offsets: np.ndarray,
        vis_data: np.ndarray,
        vis_times: np.ndarray,
    ) -> None:
        src_off = 0
        n_act = len(c_idx)
        for i in range(n_act):
            m = c_idx[i]
            cnt = int(c_cnt[i])
            act_idx = int(cell_to_active[m])
            if act_idx >= 0:
                dst_off = int(cur_offsets[act_idx])
                for j in range(cnt):
                    vis_data[dst_off + j] = f_vis[src_off + j]
                    vis_times[dst_off + j] = f_times[src_off + j]
                cur_offsets[act_idx] = dst_off + cnt
            src_off += cnt
else:
    def _pack_chunk_ragged_numba(dga_sub, t_chunk, thr_sq):
        pass

    def _scatter_fragment_samples_numba(
        c_idx, c_cnt, f_vis, f_times, cell_to_active, cur_offsets, vis_data, vis_times
    ):
        src_off = 0
        n_act = len(c_idx)
        for i in range(n_act):
            m = c_idx[i]
            cnt = int(c_cnt[i])
            act_idx = cell_to_active[m]
            if act_idx >= 0:
                dst_off = cur_offsets[act_idx]
                vis_data[dst_off : dst_off + cnt] = f_vis[src_off : src_off + cnt]
                vis_times[dst_off : dst_off + cnt] = f_times[src_off : src_off + cnt]
                cur_offsets[act_idx] = dst_off + cnt
            src_off += cnt


def _has_cuda_backend() -> bool:
    """Return True if the optional CuPy/CUDA ragged backend is available."""
    try:
        from fastducc.ragged_gpu import is_cuda_available
        return bool(is_cuda_available())
    except ImportError as exc:
        logger.debug("fastducc.ragged_gpu import failed; CUDA backend unavailable: %s", exc)
        return False


def pack_chunk_to_ragged_csr(
    dga_sub: np.ndarray,
    t_chunk: np.ndarray,
    threshold: float = 1e-4,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter samples exceeding noise threshold |V| > threshold and pack into ragged CSR arrays.

    Returns:
        cell_indices: (n_active,) uint32 array of active cell IDs in this chunk.
        cell_counts: (n_active,) uint16 array of sample counts per active cell.
        vis_data: (nnz,) complex64 array of non-zero visibility samples.
        vis_times: (nnz,) float64 timestamps.
    """
    dev = str(device).lower()
    if dev not in ("cpu", "cuda", "auto"):
        raise ValueError(f"Unsupported device='{device}'; expected 'cpu', 'cuda', or 'auto'.")
    if dev == "cuda":
        if not _has_cuda_backend():
            raise RuntimeError(
                "device='cuda' requested for pack_chunk_to_ragged_csr, but CuPy/CUDA is unavailable."
            )
        from fastducc.ragged_gpu import pack_chunk_to_ragged_gpu as _pack_gpu
        return _pack_gpu(dga_sub, t_chunk, threshold=threshold, return_host=True)
    if dev == "auto":
        if _has_cuda_backend():
            from fastducc.ragged_gpu import pack_chunk_to_ragged_gpu as _pack_gpu
            return _pack_gpu(dga_sub, t_chunk, threshold=threshold, return_host=True)
        logger.debug("pack_chunk_to_ragged_csr(device='auto') falling back to CPU: CUDA backend unavailable.")

    M, nt = dga_sub.shape
    thr_sq = float(threshold) * float(threshold)

    if HAS_NUMBA:
        return _pack_chunk_ragged_numba(
            np.ascontiguousarray(dga_sub, dtype=np.complex64),
            np.ascontiguousarray(t_chunk, dtype=np.float64),
            thr_sq,
        )

    # Vectorised NumPy fallback
    re = dga_sub.real
    im = dga_sub.imag
    mag_sq = re * re + im * im
    mask = mag_sq > thr_sq

    counts_full = np.count_nonzero(mask, axis=1)
    active_m = np.flatnonzero(counts_full > 0).astype(np.uint32)

    if len(active_m) == 0:
        return (
            np.empty(0, dtype=np.uint32),
            np.empty(0, dtype=np.uint16),
            np.empty(0, dtype=np.complex64),
            np.empty(0, dtype=np.float64),
        )

    cell_counts = counts_full[active_m].astype(np.uint16)
    sub_mask = mask[active_m, :]
    row_idx, col_idx = np.where(sub_mask)

    selected_m = active_m[row_idx]
    vis_data = np.ascontiguousarray(dga_sub[selected_m, col_idx], dtype=np.complex64)
    vis_times = np.ascontiguousarray(t_chunk[col_idx], dtype=np.float64)

    return active_m, cell_counts, vis_data, vis_times


def write_ragged_fragment(
    filepath: str,
    chunk_idx: int,
    cell_indices: np.ndarray,
    cell_counts: np.ndarray,
    vis_data: np.ndarray,
    vis_times: np.ndarray,
    t_start: float = 0.0,
    dt: float = 0.0,
    threshold: float = 1e-4,
) -> None:
    """
    Write a sharded CSR fragment as a single contiguous binary file.
    """
    n_active = len(cell_indices)
    nnz = len(vis_data)
    if len(cell_counts) != n_active:
        raise ValueError(f"cell_counts length ({len(cell_counts)}) != n_active ({n_active})")
    if len(vis_times) != nnz:
        raise ValueError(f"vis_times length ({len(vis_times)}) != nnz ({nnz})")

    pad_len = (8 - (n_active * 6) % 8) % 8
    pad_bytes = b"\x00" * pad_len

    header = struct.pack(
        HEADER_FORMAT,
        HEADER_MAGIC,
        HEADER_VERSION,
        int(chunk_idx),
        int(n_active),
        int(nnz),
        float(t_start),
        float(dt),
        float(threshold),
        b"\x00" * 20,
    )

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    tmp_path = f"{filepath}.tmp.{os.getpid()}"
    with open(tmp_path, "wb") as f:
        f.write(header)
        if n_active > 0:
            np.asarray(cell_indices, dtype=np.uint32).tofile(f)
            np.asarray(cell_counts, dtype=np.uint16).tofile(f)
            if pad_len > 0:
                f.write(pad_bytes)
        if nnz > 0:
            np.asarray(vis_data, dtype=np.complex64).tofile(f)
            np.asarray(vis_times, dtype=np.float64).tofile(f)

    os.replace(tmp_path, filepath)


def read_ragged_fragment(filepath: str) -> Dict[str, Any]:
    """
    Read full contents of a sharded CSR fragment file.
    """
    with open(filepath, "rb") as f:
        header_bytes = f.read(HEADER_SIZE)
        if len(header_bytes) < HEADER_SIZE:
            raise ValueError(f"Truncated header in fragment {filepath}")
        magic, version, chunk_idx, n_active, nnz, t_start, dt, threshold, _ = struct.unpack(
            HEADER_FORMAT, header_bytes
        )
        if magic != HEADER_MAGIC:
            raise ValueError(f"Invalid magic {magic} in fragment {filepath}")

        if n_active > 0:
            cell_indices = np.fromfile(f, dtype=np.uint32, count=n_active)
            cell_counts = np.fromfile(f, dtype=np.uint16, count=n_active)
            pad_len = (8 - (n_active * 6) % 8) % 8
            if pad_len > 0:
                f.read(pad_len)
        else:
            cell_indices = np.empty(0, dtype=np.uint32)
            cell_counts = np.empty(0, dtype=np.uint16)

        if nnz > 0:
            vis_data = np.fromfile(f, dtype=np.complex64, count=nnz)
            vis_times = np.fromfile(f, dtype=np.float64, count=nnz)
        else:
            vis_data = np.empty(0, dtype=np.complex64)
            vis_times = np.empty(0, dtype=np.float64)

    return {
        "chunk_idx": int(chunk_idx),
        "n_active_cells": int(n_active),
        "nnz": int(nnz),
        "t_start": float(t_start),
        "dt": float(dt),
        "threshold": float(threshold),
        "cell_indices": cell_indices,
        "cell_counts": cell_counts,
        "vis_data": vis_data,
        "vis_times": vis_times,
    }


def read_ragged_fragment_header(
    filepath: str,
) -> Tuple[int, int, int, float, float, np.ndarray, np.ndarray]:
    """
    Read chunk metadata and (cell_indices, cell_counts) without reading vis_data/vis_times.
    Returns: (chunk_idx, n_active, nnz, t_start, dt, cell_indices, cell_counts)
    """
    with open(filepath, "rb") as f:
        header_bytes = f.read(HEADER_SIZE)
        if len(header_bytes) < HEADER_SIZE:
            raise ValueError(f"Truncated header in fragment {filepath}")
        magic, version, chunk_idx, n_active, nnz, t_start, dt, threshold, _ = struct.unpack(
            HEADER_FORMAT, header_bytes
        )
        if magic != HEADER_MAGIC:
            raise ValueError(f"Invalid magic {magic} in fragment {filepath}")

        if n_active > 0:
            cell_indices = np.fromfile(f, dtype=np.uint32, count=n_active)
            cell_counts = np.fromfile(f, dtype=np.uint16, count=n_active)
        else:
            cell_indices = np.empty(0, dtype=np.uint32)
            cell_counts = np.empty(0, dtype=np.uint16)

    return int(chunk_idx), int(n_active), int(nnz), float(t_start), float(dt), cell_indices, cell_counts


def consolidate_ragged_fragments(
    fragment_files: List[str],
    store_dir: str,
    sparse_plan_uv: np.ndarray,
    attrs: Optional[Dict[str, Any]] = None,
    active_only: bool = True,
    cleanup_fragments: bool = False,
    device: str = "cpu",
) -> RaggedDGAStore:
    """
    Stitch and transpose sharded CSR fragments into a consolidated RaggedDGAStore.

    Parameters:
    -----------
    fragment_files : list of fragment file paths (.csr.bin)
    store_dir : destination directory for consolidated store
    sparse_plan_uv : (M_padded, 2) int32 array of [v, u] spatial coordinates
    attrs : optional attributes dict (tsamp, t0, du, dv, etc.)
    active_only : bool, if True compact store to only cells with non-zero samples (auto-pruned)
    cleanup_fragments : bool, whether to delete fragment files and fragments/ subdirectory
    device : str, execution backend ("cpu", "cuda", or "auto"; default: "cpu")
    """
    dev = str(device).lower()
    if dev not in ("cpu", "cuda", "auto"):
        raise ValueError(f"Unsupported device='{device}'; expected 'cpu', 'cuda', or 'auto'.")
    if dev == "cuda":
        if not _has_cuda_backend():
            raise RuntimeError(
                "device='cuda' requested for consolidate_ragged_fragments, but CuPy/CUDA is unavailable."
            )
        from fastducc.ragged_gpu import consolidate_ragged_fragments_gpu as _consolidate_gpu
        return _consolidate_gpu(
            fragment_files=fragment_files,
            store_dir=store_dir,
            sparse_plan_uv=sparse_plan_uv,
            attrs=attrs,
            active_only=active_only,
            cleanup_fragments=cleanup_fragments,
        )
    if dev == "auto":
        if _has_cuda_backend():
            from fastducc.ragged_gpu import consolidate_ragged_fragments_gpu as _consolidate_gpu
            return _consolidate_gpu(
                fragment_files=fragment_files,
                store_dir=store_dir,
                sparse_plan_uv=sparse_plan_uv,
                attrs=attrs,
                active_only=active_only,
                cleanup_fragments=cleanup_fragments,
            )
        logger.debug("consolidate_ragged_fragments(device='auto') falling back to CPU: CUDA backend unavailable.")

    t_start = time.time()
    os.makedirs(store_dir, exist_ok=True)
    m_padded = int(sparse_plan_uv.shape[0])

    # 1. Inspect fragment headers and sort fragments chronologically by chunk_idx
    frag_metas = []
    for fpath in fragment_files:
        if not os.path.exists(fpath):
            continue
        c_idx, n_act, nnz, t_s, dt, c_indices, c_counts = read_ragged_fragment_header(fpath)
        frag_metas.append((c_idx, fpath, n_act, nnz, t_s, dt, c_indices, c_counts))

    frag_metas.sort(key=lambda x: x[0])  # sort by chunk_idx

    # 2. Count non-zero samples per cell across all chunks
    counts_per_cell = np.zeros(m_padded, dtype=np.uint64)
    total_frag_nnz = 0
    for _, _, _, nnz, _, _, c_indices, c_counts in frag_metas:
        total_frag_nnz += nnz
        if len(c_indices) > 0:
            counts_per_cell[c_indices] += c_counts

    # 3. Form global cell offsets and active cell selection
    if active_only:
        active_cell_mask = counts_per_cell > 0
        active_cells = np.flatnonzero(active_cell_mask).astype(np.uint32)
        m_active = len(active_cells)
        cell_uv = np.ascontiguousarray(sparse_plan_uv[active_cells], dtype=np.int32)
        active_counts = counts_per_cell[active_cells]
        cell_offsets = np.concatenate([[0], np.cumsum(active_counts, dtype=np.uint64)])

        cell_to_active = np.full(m_padded, -1, dtype=np.int32)
        cell_to_active[active_cells] = np.arange(m_active, dtype=np.int32)
    else:
        m_active = m_padded
        active_cells = np.arange(m_padded, dtype=np.uint32)
        cell_uv = np.ascontiguousarray(sparse_plan_uv, dtype=np.int32)
        cell_offsets = np.concatenate([[0], np.cumsum(counts_per_cell, dtype=np.uint64)])
        cell_to_active = active_cells.astype(np.int32)

    nnz_total = int(cell_offsets[-1]) if len(cell_offsets) > 1 else 0

    # 4. Transpose and assemble vis_data and vis_times
    vis_data = np.empty(nnz_total, dtype=np.complex64)
    vis_times = np.empty(nnz_total, dtype=np.float64)
    cur_offsets = cell_offsets[:-1].copy()

    for _, fpath, _, nnz, _, _, _, _ in frag_metas:
        if nnz == 0:
            continue
        frag = read_ragged_fragment(fpath)
        c_idx = frag["cell_indices"]
        c_cnt = frag["cell_counts"]
        f_vis = frag["vis_data"]
        f_times = frag["vis_times"]

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

    if len(cell_offsets) > 1 and nnz_total > 0:
        np.testing.assert_array_equal(cur_offsets, cell_offsets[1:])

    # 5. Write binary container files
    offsets_path = os.path.join(store_dir, "cell_offsets.bin")
    uv_path = os.path.join(store_dir, "cell_uv.bin")
    vis_path = os.path.join(store_dir, "vis_data.bin")
    times_path = os.path.join(store_dir, "vis_times.bin")
    attrs_path = os.path.join(store_dir, "attrs.json")

    cell_offsets.astype(np.uint64).tofile(offsets_path)
    cell_uv.astype(np.int32).tofile(uv_path)
    vis_data.tofile(vis_path)
    vis_times.tofile(times_path)

    # 6. Save attrs
    store_attrs = dict(attrs) if attrs else {}
    store_attrs.update({
        "format": "ragged_csr",
        "m_total": int(m_active),
        "original_m_padded": int(m_padded),
        "nnz_total": int(nnz_total),
        "pruned": bool(active_only or m_active < m_padded),
        "consolidation_duration_s": float(time.time() - t_start),
    })
    with open(attrs_path, "w", encoding="utf-8") as f:
        json.dump(store_attrs, f, indent=2)

    logger.info(
        f"[Consolidate CSR] Assembled {len(frag_metas)} fragments -> {store_dir} "
        f"(M: {m_padded} -> {m_active}, NNZ: {nnz_total}, {time.time() - t_start:.2f}s)"
    )

    # 7. Optionally clean up fragments
    if cleanup_fragments:
        for _, fpath, _, _, _, _, _, _ in frag_metas:
            try:
                os.remove(fpath)
            except OSError:
                pass
        frag_dir = os.path.join(store_dir, "fragments")
        if os.path.isdir(frag_dir):
            shutil.rmtree(frag_dir, ignore_errors=True)

    return RaggedDGAStore.open(store_dir, mode="r")


def pack_chunk_to_ragged_gpu(*args: Any, **kwargs: Any):
    """Thin forwarding export to :func:`fastducc.ragged_gpu.pack_chunk_to_ragged_gpu`."""
    from fastducc.ragged_gpu import pack_chunk_to_ragged_gpu as _impl
    return _impl(*args, **kwargs)


def consolidate_ragged_fragments_gpu(*args: Any, **kwargs: Any) -> RaggedDGAStore:
    """Thin forwarding export to :func:`fastducc.ragged_gpu.consolidate_ragged_fragments_gpu`."""
    from fastducc.ragged_gpu import consolidate_ragged_fragments_gpu as _impl
    return _impl(*args, **kwargs)

