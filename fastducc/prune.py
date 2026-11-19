# fastducc/prune.py
"""
Post-gridding pruning of direct sparse DGA Zarr stores.

In ``direct_dga`` mode the sparse UV plan (``fastducc.grid_plan.compute_sparse_uv_grid_plan``)
is dilated by the gridding-kernel support (``pad_radius``) so that w-gridding kernels are never
truncated. After gridding, a large fraction of these padded cells never receive any power.
``prune_direct_dga_store`` removes those empty rows so downstream FFA / imaging stages only
allocate memory for the truly active cells.

Algorithm (memory-bounded, two passes):
  1. Activity scan: for each block of ``chunk_cells`` rows, stream over time strips and evaluate
     ``(|dga| > threshold).any(axis=1)``.
  2. Compaction: write the active rows into scratch arrays ``dga_pruned`` / ``uv_pruned``
     inside the same group. Output row blocks are aligned to output chunk boundaries so tasks
     can run concurrently without write races.
  3. Atomic swap: the scratch arrays are renamed onto the canonical ``dga`` / ``uv`` keys and
     the group attributes are updated (``m_total``, ``pruned``, ``original_m_padded``).
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import zarr

logger = logging.getLogger(__name__)

__all__ = ["prune_direct_dga_store"]

_SCRATCH_DGA = "dga_pruned"
_SCRATCH_UV = "uv_pruned"
_OLD_DGA = "dga_unpruned_old"
_OLD_UV = "uv_unpruned_old"

# Target uncompressed bytes for one (rows x time-strip) read buffer
_STRIP_TARGET_BYTES = 256 * 1024 * 1024


def _time_strips(n_time: int, t_chunk: int, n_rows: int, itemsize: int) -> List[Tuple[int, int]]:
    """Split [0, n_time) into chunk-aligned strips whose buffer stays near _STRIP_TARGET_BYTES."""
    n_rows = max(1, int(n_rows))
    per_chunk_bytes = n_rows * t_chunk * itemsize
    n_chunks_per_strip = max(1, _STRIP_TARGET_BYTES // max(1, per_chunk_bytes))
    strip = int(n_chunks_per_strip * t_chunk)
    return [(t0, min(t0 + strip, n_time)) for t0 in range(0, n_time, strip)]


def _scan_block_active(dga_zarr_path: str, r0: int, r1: int, threshold: float) -> Tuple[int, np.ndarray]:
    """Return (r0, boolean activity mask) for rows [r0, r1) of root['dga']."""
    dga = zarr.open_group(dga_zarr_path, mode="r")["dga"]
    n_time = int(dga.shape[1])
    t_chunk = int(dga.chunks[1])
    thr_sq = float(threshold) * float(threshold)
    active = np.zeros(r1 - r0, dtype=bool)
    for t0, t1 in _time_strips(n_time, t_chunk, r1 - r0, dga.dtype.itemsize):
        pending = ~active
        if not pending.any():
            break
        block = dga[r0:r1, t0:t1]
        if np.iscomplexobj(block):
            mag_sq = block.real * block.real + block.imag * block.imag
        else:
            mag_sq = block * block
        active |= (mag_sq > thr_sq).any(axis=1)
        del block, mag_sq
    return r0, active


def _copy_block(
    dga_zarr_path: str, src_rows: np.ndarray, o0: int, o1: int
) -> int:
    """Copy source rows ``src_rows`` of root['dga'] into rows [o0, o1) of root['dga_pruned']."""
    root = zarr.open_group(dga_zarr_path, mode="r+")
    src = root["dga"]
    dst = root[_SCRATCH_DGA]
    n_time = int(src.shape[1])
    t_chunk = int(dst.chunks[1])
    src_rows = np.asarray(src_rows, dtype=np.int64)
    r_lo, r_hi = int(src_rows[0]), int(src_rows[-1]) + 1
    local = src_rows - r_lo
    # Contiguous slab read is cheaper than fancy indexing when rows are dense in the span
    use_slab = (r_hi - r_lo) <= 2 * len(src_rows)
    n_read_rows = (r_hi - r_lo) if use_slab else len(src_rows)
    for t0, t1 in _time_strips(n_time, t_chunk, n_read_rows, src.dtype.itemsize):
        if use_slab:
            data = src[r_lo:r_hi, t0:t1][local]
        else:
            data = src.oindex[src_rows, t0:t1]
        dst[o0:o1, t0:t1] = data
        del data
    return o1 - o0


def _create_array(grp: Any, name: str, shape: Tuple[int, ...], chunks: Tuple[int, ...], dtype: Any) -> Any:
    if name in grp:
        del grp[name]
    if hasattr(grp, "create_array"):
        return grp.create_array(name, shape=shape, chunks=chunks, dtype=dtype)
    return grp.create_dataset(name, shape=shape, chunks=chunks, dtype=dtype)


def _run_tasks(func, arg_list: List[Tuple[Any, ...]], client: Optional[Any], n_threads: int) -> List[Any]:
    if not arg_list:
        return []
    if client is not None:
        futures = [client.submit(func, *args, pure=False) for args in arg_list]
        return client.gather(futures)
    if n_threads <= 1:
        return [func(*args) for args in arg_list]
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        return list(ex.map(lambda a: func(*a), arg_list))


def _finish_interrupted_swap(store_path: str, threshold: float) -> None:
    """
    Recover from a crash during a previous prune.

    - If the directory swap had started (``dga_unpruned_old`` exists), complete it and
      finalise the attributes.
    - Otherwise, any leftover scratch arrays are from an incomplete compaction and are removed
      (they are rebuilt from scratch).
    """
    p = lambda n: os.path.join(store_path, n)  # noqa: E731
    if os.path.exists(p(_OLD_DGA)):
        m_orig = int(zarr.open_array(p(_OLD_DGA), mode="r").shape[0])
        if not os.path.exists(p("dga")):
            os.rename(p(_SCRATCH_DGA), p("dga"))
        if os.path.exists(p(_SCRATCH_UV)):
            if os.path.exists(p("uv")):
                if os.path.exists(p(_OLD_UV)):
                    shutil.rmtree(p(_OLD_UV), ignore_errors=True)
                os.rename(p("uv"), p(_OLD_UV))
            os.rename(p(_SCRATCH_UV), p("uv"))
        root = zarr.open_group(store_path, mode="r+")
        root.attrs.update({
            "m_total": int(root["dga"].shape[0]),
            "pruned": True,
            "original_m_padded": m_orig,
            "prune_threshold": float(root.attrs.get("prune_threshold", threshold)),
            "prune_swap_in_progress": False,
        })
        logger.warning(f"[DGA Prune] Completed interrupted swap in {store_path}.")
    else:
        for n in (_SCRATCH_DGA, _SCRATCH_UV):
            if os.path.exists(p(n)):
                shutil.rmtree(p(n), ignore_errors=True)
    for n in (_OLD_DGA, _OLD_UV):
        if os.path.exists(p(n)):
            shutil.rmtree(p(n), ignore_errors=True)


def prune_direct_dga_store(
    dga_zarr_path: str,
    threshold: float = 1e-4,
    chunk_cells: int = 4096,
    client: Optional[Any] = None,
    n_threads: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Scans a direct DGA Zarr store (root['dga']: (M_padded, n_time), root['uv']: (M_padded, 2)),
    identifies cells with non-zero power (any time sample with |V| > threshold),
    and compacts root['dga'] and root['uv'] via an atomic scratch swap down to
    only the true active cells (M_active, n_time).

    Updates store metadata:
      - root.attrs['m_total'] = M_active
      - root.attrs['pruned'] = True
      - root.attrs['original_m_padded'] = M_padded
      - root.attrs['prune_threshold'] = threshold

    The threshold matches the downstream FFA convention (samples with |V| <= threshold are
    zeroed before folding), so pruned rows would have contributed exactly zero.

    Parameters
    ----------
    dga_zarr_path : path to the direct DGA Zarr group (local / POSIX filesystem).
    threshold : amplitude cutoff; a cell is kept if any sample has |V| > threshold.
    chunk_cells : number of rows per scan / copy task (rounded up to the dga row-chunk size).
    client : optional dask.distributed Client; if given, block tasks are distributed to workers
        (workers must see the same filesystem). Otherwise a local thread pool is used.
    n_threads : local thread count when ``client`` is None (default: min(8, cpu_count)).

    Returns
    -------
    dict with keys: m_original, m_pruned, active_fraction, n_time, skipped, elapsed_s.
    """
    t_start = time.time()
    store_path = os.path.abspath(str(dga_zarr_path))
    if not os.path.isdir(store_path):
        raise FileNotFoundError(f"Direct DGA store not found: {store_path}")

    _finish_interrupted_swap(store_path, float(threshold))
    root = zarr.open_group(store_path, mode="r+")
    if "dga" not in root or "uv" not in root:
        raise ValueError(f"{store_path} is not a direct DGA store (missing 'dga' or 'uv').")

    dga = root["dga"]
    uv = root["uv"]
    m_orig, n_time = int(dga.shape[0]), int(dga.shape[1])
    if int(uv.shape[0]) != m_orig:
        raise ValueError(f"'uv' rows ({uv.shape[0]}) do not match 'dga' rows ({m_orig}).")

    if bool(root.attrs.get("pruned", False)):
        logger.info(f"[DGA Prune] {store_path} already pruned (M={m_orig}); skipping.")
        return {
            "m_original": int(root.attrs.get("original_m_padded", m_orig)),
            "m_pruned": m_orig,
            "active_fraction": m_orig / max(1, int(root.attrs.get("original_m_padded", m_orig))),
            "n_time": n_time,
            "skipped": True,
            "elapsed_s": time.time() - t_start,
        }

    if n_threads is None:
        n_threads = max(1, min(8, os.cpu_count() or 1))
    row_chunk = int(dga.chunks[0])
    t_chunk = int(dga.chunks[1])
    blk = max(row_chunk, int(np.ceil(max(1, int(chunk_cells)) / row_chunk)) * row_chunk)

    # ---- Pass 1: activity scan -------------------------------------------------
    t0 = time.time()
    scan_args = [(store_path, r0, min(r0 + blk, m_orig), float(threshold)) for r0 in range(0, m_orig, blk)]
    results = _run_tasks(_scan_block_active, scan_args, client, n_threads)
    active = np.zeros(m_orig, dtype=bool)
    for r0, mask in results:
        active[r0 : r0 + len(mask)] = mask
    active_idx = np.flatnonzero(active).astype(np.int64)
    m_active = int(active_idx.size)
    logger.info(
        f"[DGA Prune] Scan of {store_path}: {m_active}/{m_orig} active cells "
        f"({m_active / max(1, m_orig) * 100:.1f}%) in {time.time() - t0:.1f}s"
    )

    stats = {
        "m_original": m_orig,
        "m_pruned": m_active,
        "active_fraction": m_active / max(1, m_orig),
        "n_time": n_time,
        "skipped": False,
    }

    if m_active == m_orig:
        root.attrs.update({"pruned": True, "original_m_padded": m_orig, "m_total": m_orig,
                           "prune_threshold": float(threshold)})
        stats["elapsed_s"] = time.time() - t_start
        return stats

    # ---- Pass 2: compaction into scratch arrays --------------------------------
    t0 = time.time()
    out_row_chunk = max(1, min(row_chunk, m_active)) if m_active > 0 else 1
    dga_shape = (m_active, n_time)
    _create_array(root, _SCRATCH_DGA, dga_shape, (out_row_chunk, min(t_chunk, max(1, n_time))), dga.dtype)
    uv_all = np.asarray(uv[:])
    uv_pruned = np.ascontiguousarray(uv_all[active_idx])
    del uv_all
    ds_uv = _create_array(root, _SCRATCH_UV, uv_pruned.shape, (max(1, m_active), 2), uv.dtype)
    if m_active > 0:
        ds_uv[:] = uv_pruned

        out_blk = max(out_row_chunk, (blk // out_row_chunk) * out_row_chunk)
        copy_args = [
            (store_path, active_idx[o0 : min(o0 + out_blk, m_active)], o0, min(o0 + out_blk, m_active))
            for o0 in range(0, m_active, out_blk)
        ]
        copied = int(sum(_run_tasks(_copy_block, copy_args, client, n_threads)))
        if copied != m_active:
            raise RuntimeError(f"[DGA Prune] Copied {copied} rows, expected {m_active}.")
    logger.info(f"[DGA Prune] Compacted {m_active} rows into scratch arrays in {time.time() - t0:.1f}s")

    # ---- Pass 3: atomic swap ----------------------------------------------------
    root.attrs.update({"prune_swap_in_progress": True})
    p = lambda n: os.path.join(store_path, n)  # noqa: E731
    os.rename(p("dga"), p(_OLD_DGA))
    os.rename(p(_SCRATCH_DGA), p("dga"))
    os.rename(p("uv"), p(_OLD_UV))
    os.rename(p(_SCRATCH_UV), p("uv"))

    root = zarr.open_group(store_path, mode="r+")
    root.attrs.update({
        "m_total": m_active,
        "pruned": True,
        "original_m_padded": m_orig,
        "prune_threshold": float(threshold),
        "prune_swap_in_progress": False,
    })

    # Sanity: canonical keys must have the compacted shape and stay row-synchronised
    chk = zarr.open_group(store_path, mode="r")
    if tuple(chk["dga"].shape) != dga_shape or int(chk["uv"].shape[0]) != m_active:
        raise RuntimeError(
            f"[DGA Prune] Post-swap shape mismatch: dga={chk['dga'].shape}, uv={chk['uv'].shape}, "
            f"expected ({m_active}, {n_time})."
        )

    shutil.rmtree(p(_OLD_DGA), ignore_errors=True)
    shutil.rmtree(p(_OLD_UV), ignore_errors=True)

    stats["elapsed_s"] = time.time() - t_start
    logger.info(
        f"[DGA Prune] {store_path}: M {m_orig} -> {m_active} "
        f"({stats['active_fraction'] * 100:.1f}% kept) in {stats['elapsed_s']:.1f}s"
    )
    return stats
