# fastducc/ragged_gpu.py
"""
CUDA/CuPy GPU acceleration for ragged CSR Dynamic Grid Array (DGA) packing
and fragment consolidation in FastDUCC.

Preserves identical mathematical thresholds, ascending cell order, chronological
sample order, and on-disk binary CSR layout as :mod:`fastducc.ragged`.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import cupy as cp

    _CUPY_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as exc:
    cp = None  # type: ignore[assignment]
    _CUPY_IMPORT_ERROR = exc

from fastducc.ragged import (
    RaggedDGAStore,
    read_ragged_fragment,
    read_ragged_fragment_header,
)

logger = logging.getLogger(__name__)

_WARP_SIZE = 32
_BLOCK_THREADS = 256
_WARPS_PER_BLOCK = _BLOCK_THREADS // _WARP_SIZE
_TILE_M = 32
_TILE_BLOCK_Y = 8
_UINT16_MAX = 65535

_CUDA_KERNEL_SRC = r"""
extern "C" __global__ void pack_chunk_count_kernel(
    const float2* __restrict__ d_dga_sub,
    unsigned short* __restrict__ d_cell_counts_full,
    int M,
    int nt_chunk,
    float thr_sq
) {
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int m = blockIdx.x * (blockDim.x >> 5) + warp_in_block;
    if (m >= M) {
        return;
    }

    const size_t row_off = static_cast<size_t>(m) * static_cast<size_t>(nt_chunk);
    unsigned int cnt = 0;

    for (int t_base = 0; t_base < nt_chunk; t_base += 32) {
        const int t = t_base + lane;
        bool pred = false;
        if (t < nt_chunk) {
            const float2 v = d_dga_sub[row_off + static_cast<size_t>(t)];
            pred = (v.x * v.x + v.y * v.y) > thr_sq;
        }
        const unsigned int mask = __ballot_sync(0xffffffffu, pred);
        cnt += static_cast<unsigned int>(__popc(mask));
    }

    if (lane == 0) {
        d_cell_counts_full[m] = static_cast<unsigned short>(cnt);
    }
}

extern "C" __global__ void pack_chunk_scatter_kernel(
    const float2* __restrict__ d_dga_sub,
    const double* __restrict__ d_t_chunk,
    const unsigned int* __restrict__ d_cell_indices,
    const unsigned long long* __restrict__ d_write_offsets,
    float2* __restrict__ d_vis_data,
    double* __restrict__ d_vis_times,
    int n_active,
    int nt_chunk,
    float thr_sq
) {
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int a = blockIdx.x * (blockDim.x >> 5) + warp_in_block;
    if (a >= n_active) {
        return;
    }

    const unsigned int m = d_cell_indices[a];
    const size_t row_off = static_cast<size_t>(m) * static_cast<size_t>(nt_chunk);
    unsigned long long dst_base = d_write_offsets[a];
    const unsigned int lane_prefix_mask = (1u << lane) - 1u;

    for (int t_base = 0; t_base < nt_chunk; t_base += 32) {
        const int t = t_base + lane;
        bool pred = false;
        float2 v = make_float2(0.0f, 0.0f);
        if (t < nt_chunk) {
            v = d_dga_sub[row_off + static_cast<size_t>(t)];
            pred = (v.x * v.x + v.y * v.y) > thr_sq;
        }
        const unsigned int mask = __ballot_sync(0xffffffffu, pred);
        if (pred) {
            const unsigned long long out_idx =
                dst_base + static_cast<unsigned long long>(__popc(mask & lane_prefix_mask));
            d_vis_data[out_idx] = v;
            d_vis_times[out_idx] = d_t_chunk[t];
        }
        dst_base += static_cast<unsigned long long>(__popc(mask));
    }
}

extern "C" __global__ void pack_uv_chunk_count_kernel(
    const float2* __restrict__ d_uv_chunk,
    const long long* __restrict__ d_flat_indices,
    unsigned short* __restrict__ d_cell_counts_full,
    int M,
    int nt_chunk,
    size_t spatial_stride,
    float thr_sq
) {
    __shared__ float2 smem_v[32][33];
    __shared__ size_t smem_fidx[32];
    __shared__ unsigned int smem_cnt[32];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int m0 = blockIdx.x * 32;

    if (ty == 0) {
        const int m = m0 + tx;
        smem_fidx[tx] = (m < M) ? static_cast<size_t>(d_flat_indices[m]) : 0ULL;
        smem_cnt[tx] = 0u;
    }
    __syncthreads();

    for (int t0 = 0; t0 < nt_chunk; t0 += 32) {
        for (int dt = ty; dt < 32; dt += 8) {
            const int t = t0 + dt;
            const int m = m0 + tx;
            if (m < M && t < nt_chunk) {
                smem_v[tx][dt] = d_uv_chunk[static_cast<size_t>(t) * spatial_stride + smem_fidx[tx]];
            }
        }
        __syncthreads();

        for (int cm = ty; cm < 32; cm += 8) {
            const int m = m0 + cm;
            const int t = t0 + tx;
            bool pred = false;
            if (m < M && t < nt_chunk) {
                const float2 v = smem_v[cm][tx];
                pred = (v.x * v.x + v.y * v.y) > thr_sq;
            }
            const unsigned int mask = __ballot_sync(0xffffffffu, pred);
            if (tx == 0 && m < M) {
                smem_cnt[cm] += static_cast<unsigned int>(__popc(mask));
            }
        }
        __syncthreads();
    }

    if (ty == 0) {
        const int m = m0 + tx;
        if (m < M) {
            d_cell_counts_full[m] = static_cast<unsigned short>(smem_cnt[tx]);
        }
    }
}

extern "C" __global__ void pack_uv_chunk_scatter_kernel(
    const float2* __restrict__ d_uv_chunk,
    const long long* __restrict__ d_flat_indices,
    const double* __restrict__ d_t_chunk,
    const unsigned int* __restrict__ d_cell_indices,
    const unsigned long long* __restrict__ d_write_offsets,
    float2* __restrict__ d_vis_data,
    double* __restrict__ d_vis_times,
    int n_active,
    int nt_chunk,
    size_t spatial_stride,
    float thr_sq
) {
    __shared__ float2 smem_v[32][33];
    __shared__ size_t smem_fidx[32];
    __shared__ unsigned long long smem_dst[32];
    __shared__ double smem_t[32];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int a0 = blockIdx.x * 32;
    const unsigned int lane_prefix_mask = (1u << tx) - 1u;

    if (ty == 0) {
        const int a = a0 + tx;
        if (a < n_active) {
            const unsigned int m = d_cell_indices[a];
            smem_fidx[tx] = static_cast<size_t>(d_flat_indices[m]);
            smem_dst[tx] = d_write_offsets[a];
        } else {
            smem_fidx[tx] = 0ULL;
            smem_dst[tx] = 0ULL;
        }
    }
    __syncthreads();

    for (int t0 = 0; t0 < nt_chunk; t0 += 32) {
        if (ty == 0) {
            const int t = t0 + tx;
            smem_t[tx] = (t < nt_chunk) ? d_t_chunk[t] : 0.0;
        }
        for (int dt = ty; dt < 32; dt += 8) {
            const int t = t0 + dt;
            const int a = a0 + tx;
            if (a < n_active && t < nt_chunk) {
                smem_v[tx][dt] = d_uv_chunk[static_cast<size_t>(t) * spatial_stride + smem_fidx[tx]];
            }
        }
        __syncthreads();

        for (int cm = ty; cm < 32; cm += 8) {
            const int a = a0 + cm;
            const int t = t0 + tx;
            bool pred = false;
            float2 v = make_float2(0.0f, 0.0f);
            if (a < n_active && t < nt_chunk) {
                v = smem_v[cm][tx];
                pred = (v.x * v.x + v.y * v.y) > thr_sq;
            }
            const unsigned int mask = __ballot_sync(0xffffffffu, pred);
            const unsigned long long dst_base = smem_dst[cm];
            if (pred) {
                const unsigned long long out_idx =
                    dst_base + static_cast<unsigned long long>(__popc(mask & lane_prefix_mask));
                d_vis_data[out_idx] = v;
                d_vis_times[out_idx] = smem_t[tx];
            }
            __syncwarp();
            if (tx == 0 && a < n_active) {
                smem_dst[cm] = dst_base + static_cast<unsigned long long>(__popc(mask));
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ void accumulate_cell_counts_kernel(
    const unsigned int* __restrict__ d_c_idx,
    const unsigned short* __restrict__ d_c_cnt,
    unsigned long long* __restrict__ d_counts_per_cell,
    int n_act,
    int m_padded,
    int* __restrict__ d_oob_flag
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_act) {
        const unsigned int cid = d_c_idx[i];
        if (cid >= static_cast<unsigned int>(m_padded)) {
            atomicExch(d_oob_flag, 1);
            return;
        }
        atomicAdd(&d_counts_per_cell[cid], static_cast<unsigned long long>(d_c_cnt[i]));
    }
}

extern "C" __global__ void scatter_fragment_to_csr_kernel(
    const unsigned int* __restrict__ d_c_idx,
    const unsigned short* __restrict__ d_c_cnt,
    const unsigned long long* __restrict__ d_src_offsets,
    const float2* __restrict__ d_f_vis,
    const double* __restrict__ d_f_times,
    const int* __restrict__ d_cell_to_active,
    unsigned long long* __restrict__ d_write_heads,
    float2* __restrict__ d_vis_data,
    double* __restrict__ d_vis_times,
    int n_act
) {
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.x * (blockDim.x >> 5) + warp_in_block;
    if (i >= n_act) {
        return;
    }

    const unsigned int m = d_c_idx[i];
    const int act_idx = d_cell_to_active[m];
    if (act_idx < 0) {
        return;
    }

    const unsigned long long cnt = static_cast<unsigned long long>(d_c_cnt[i]);
    if (cnt == 0ULL) {
        return;
    }

    const unsigned long long src_off = d_src_offsets[i];
    unsigned long long old_head = 0ULL;
    if (lane == 0) {
        old_head = atomicAdd(&d_write_heads[act_idx], cnt);
    }
    const unsigned long long dst_off = __shfl_sync(0xffffffffu, old_head, 0);

    for (unsigned long long j = static_cast<unsigned long long>(lane); j < cnt; j += 32ULL) {
        d_vis_data[dst_off + j] = d_f_vis[src_off + j];
        d_vis_times[dst_off + j] = d_f_times[src_off + j];
    }
}
"""

_MODULE_CACHE: Dict[int, Any] = {}
_MODULE_LOCK = threading.Lock()


def is_cuda_available() -> bool:
    """Return True if CuPy is importable and at least one CUDA device is accessible."""
    if cp is None:
        logger.debug("CuPy unavailable for ragged_gpu: %s", _CUPY_IMPORT_ERROR)
        return False
    try:
        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except cp.cuda.runtime.CUDARuntimeError as exc:
        logger.warning("CUDA device query failed in ragged_gpu: %s", exc)
        return False


def _require_cuda() -> None:
    """Raise an explicit RuntimeError if CuPy or a CUDA device is unavailable."""
    if cp is None:
        raise RuntimeError(
            f"CuPy is required for CUDA ragged CSR operations, but failed to import: {_CUPY_IMPORT_ERROR}"
        )
    try:
        n_dev = int(cp.cuda.runtime.getDeviceCount())
    except cp.cuda.runtime.CUDARuntimeError as exc:
        raise RuntimeError(f"CUDA device query failed: {exc}") from exc
    if n_dev <= 0:
        raise RuntimeError("CUDA ragged CSR operations requested, but no CUDA devices were detected.")


def _get_module() -> Any:
    """Lazily compile and cache the CUDA RawModule for the active CUDA device."""
    _require_cuda()
    dev_id = int(cp.cuda.Device().id)
    with _MODULE_LOCK:
        mod = _MODULE_CACHE.get(dev_id)
        if mod is None:
            mod = cp.RawModule(code=_CUDA_KERNEL_SRC, options=("--std=c++17",))
            _MODULE_CACHE[dev_id] = mod
        return mod


def _empty_csr_arrays(return_host: bool) -> Tuple[Any, Any, Any, Any]:
    """Return correctly typed empty CSR fragment arrays on host or device."""
    if return_host:
        return (
            np.empty(0, dtype=np.uint32),
            np.empty(0, dtype=np.uint16),
            np.empty(0, dtype=np.complex64),
            np.empty(0, dtype=np.float64),
        )
    return (
        cp.empty(0, dtype=cp.uint32),
        cp.empty(0, dtype=cp.uint16),
        cp.empty(0, dtype=cp.complex64),
        cp.empty(0, dtype=cp.float64),
    )


def _compact_from_counts_gpu(
    counts_full: Any,
    nt_chunk: int,
    scatter_fn: Any,
    return_host: bool = True,
) -> Tuple[Any, Any, Any, Any]:
    """
    Identify active cells from per-cell sample counts, compute exclusive CSR
    write offsets, invoke ``scatter_fn``, and optionally copy outputs to host.
    """
    if nt_chunk > _UINT16_MAX:
        raise ValueError(
            f"nt_chunk ({nt_chunk}) exceeds uint16 maximum ({_UINT16_MAX}) for ragged CSR cell_counts."
        )
    if counts_full.size == 0 or nt_chunk == 0:
        return _empty_csr_arrays(return_host)

    active_m = cp.flatnonzero(counts_full > 0).astype(cp.uint32)
    n_active = int(active_m.size)
    if n_active == 0:
        return _empty_csr_arrays(return_host)

    cell_counts = counts_full[active_m].astype(cp.uint16, copy=False)
    counts_u64 = cell_counts.astype(cp.uint64)
    cumsum_u64 = cp.cumsum(counts_u64, dtype=cp.uint64)
    nnz = int(cumsum_u64[-1].item())
    write_offsets = cumsum_u64 - counts_u64

    vis_data = cp.empty(nnz, dtype=cp.complex64)
    vis_times = cp.empty(nnz, dtype=cp.float64)

    scatter_fn(active_m, write_offsets, vis_data, vis_times, n_active)

    if return_host:
        return (
            cp.asnumpy(active_m),
            cp.asnumpy(cell_counts),
            cp.asnumpy(vis_data),
            cp.asnumpy(vis_times),
        )
    return active_m, cell_counts, vis_data, vis_times


def pack_chunk_to_ragged_gpu(
    dga_sub: Any,
    t_chunk: Any,
    threshold: float = 1e-4,
    return_host: bool = True,
) -> Tuple[Any, Any, Any, Any]:
    """
    Filter ``|V| > threshold`` samples from a 2D chunk ``(M, nt_chunk)`` on CUDA
    and pack into ragged CSR arrays preserving ascending ``m`` and chronological ``t``.
    """
    _require_cuda()
    d_dga = cp.ascontiguousarray(cp.asarray(dga_sub), dtype=cp.complex64)
    d_t = cp.ascontiguousarray(cp.asarray(t_chunk), dtype=cp.float64)

    if d_dga.ndim != 2:
        raise ValueError(f"dga_sub must be 2D (M, nt_chunk); got shape {d_dga.shape}")
    if d_t.ndim != 1 or int(d_t.shape[0]) != int(d_dga.shape[1]):
        raise ValueError(
            f"t_chunk length ({d_t.shape}) must match dga_sub.shape[1] ({d_dga.shape[1]})"
        )

    M = int(d_dga.shape[0])
    nt_chunk = int(d_dga.shape[1])
    if nt_chunk > _UINT16_MAX:
        raise ValueError(
            f"nt_chunk ({nt_chunk}) exceeds uint16 maximum ({_UINT16_MAX}) for ragged CSR cell_counts."
        )
    if M == 0 or nt_chunk == 0:
        return _empty_csr_arrays(return_host)

    thr_sq = np.float32(float(threshold) * float(threshold))
    mod = _get_module()
    count_kernel = mod.get_function("pack_chunk_count_kernel")
    scatter_kernel = mod.get_function("pack_chunk_scatter_kernel")

    d_counts_full = cp.empty(M, dtype=cp.uint16)
    blocks_count = (M + _WARPS_PER_BLOCK - 1) // _WARPS_PER_BLOCK
    count_kernel(
        (blocks_count,),
        (_BLOCK_THREADS,),
        (d_dga, d_counts_full, np.int32(M), np.int32(nt_chunk), thr_sq),
    )

    def _scatter(active_m: Any, write_offsets: Any, vis_data: Any, vis_times: Any, n_active: int) -> None:
        blocks_scatter = (n_active + _WARPS_PER_BLOCK - 1) // _WARPS_PER_BLOCK
        scatter_kernel(
            (blocks_scatter,),
            (_BLOCK_THREADS,),
            (
                d_dga,
                d_t,
                active_m,
                write_offsets,
                vis_data,
                vis_times,
                np.int32(n_active),
                np.int32(nt_chunk),
                thr_sq,
            ),
        )

    return _compact_from_counts_gpu(d_counts_full, nt_chunk, _scatter, return_host=return_host)


def pack_uv_chunk_to_ragged_gpu(
    uv_chunk: Any,
    flat_indices: Any,
    t_chunk: Any,
    threshold: float = 1e-4,
    return_host: bool = True,
    check_bounds: bool = True,
) -> Tuple[Any, Any, Any, Any]:
    """
    Pack ``|V| > threshold`` samples directly from a 3D UV cube chunk ``(nt_chunk, nv, u_half)``
    indexed by ``flat_indices`` on CUDA, avoiding any intermediate transposed ``(M, nt_chunk)`` copy.
    """
    _require_cuda()
    d_uv = (
        uv_chunk
        if (isinstance(uv_chunk, cp.ndarray) and uv_chunk.dtype == cp.complex64 and uv_chunk.flags.c_contiguous)
        else cp.ascontiguousarray(cp.asarray(uv_chunk), dtype=cp.complex64)
    )
    d_flat = (
        flat_indices
        if (isinstance(flat_indices, cp.ndarray) and flat_indices.dtype == cp.int64 and flat_indices.flags.c_contiguous)
        else cp.ascontiguousarray(cp.asarray(flat_indices), dtype=cp.int64)
    )
    d_t = (
        t_chunk
        if (isinstance(t_chunk, cp.ndarray) and t_chunk.dtype == cp.float64 and t_chunk.flags.c_contiguous)
        else cp.ascontiguousarray(cp.asarray(t_chunk), dtype=cp.float64)
    )

    if d_uv.ndim != 3:
        raise ValueError(f"uv_chunk must be 3D (nt_chunk, nv, u_half); got shape {d_uv.shape}")
    if d_flat.ndim != 1:
        raise ValueError(f"flat_indices must be 1D; got shape {d_flat.shape}")
    if d_t.ndim != 1 or int(d_t.shape[0]) != int(d_uv.shape[0]):
        raise ValueError(
            f"t_chunk length ({d_t.shape}) must match uv_chunk.shape[0] ({d_uv.shape[0]})"
        )

    nt_chunk = int(d_uv.shape[0])
    spatial_stride = int(d_uv.shape[1] * d_uv.shape[2])
    M = int(d_flat.size)

    if nt_chunk > _UINT16_MAX:
        raise ValueError(
            f"nt_chunk ({nt_chunk}) exceeds uint16 maximum ({_UINT16_MAX}) for ragged CSR cell_counts."
        )
    if M == 0 or nt_chunk == 0:
        return _empty_csr_arrays(return_host)
    if spatial_stride == 0:
        raise ValueError("uv_chunk has zero spatial plane size with non-empty flat_indices.")

    if check_bounds and bool(cp.any((d_flat < 0) | (d_flat >= spatial_stride)).item()):
        raise ValueError(
            f"flat_indices contains out-of-bounds indices for spatial_stride={spatial_stride}."
        )

    thr_sq = np.float32(float(threshold) * float(threshold))
    mod = _get_module()
    count_kernel = mod.get_function("pack_uv_chunk_count_kernel")
    scatter_kernel = mod.get_function("pack_uv_chunk_scatter_kernel")

    d_counts_full = cp.empty(M, dtype=cp.uint16)
    blocks_count = (M + _TILE_M - 1) // _TILE_M
    count_kernel(
        (blocks_count, 1, 1),
        (_TILE_M, _TILE_BLOCK_Y, 1),
        (
            d_uv,
            d_flat,
            d_counts_full,
            np.int32(M),
            np.int32(nt_chunk),
            np.uint64(spatial_stride),
            thr_sq,
        ),
    )

    def _scatter(active_m: Any, write_offsets: Any, vis_data: Any, vis_times: Any, n_active: int) -> None:
        blocks_scatter = (n_active + _TILE_M - 1) // _TILE_M
        scatter_kernel(
            (blocks_scatter, 1, 1),
            (_TILE_M, _TILE_BLOCK_Y, 1),
            (
                d_uv,
                d_flat,
                d_t,
                active_m,
                write_offsets,
                vis_data,
                vis_times,
                np.int32(n_active),
                np.int32(nt_chunk),
                np.uint64(spatial_stride),
                thr_sq,
            ),
        )

    return _compact_from_counts_gpu(d_counts_full, nt_chunk, _scatter, return_host=return_host)


def _normalise_sparse_plan_uv(sparse_plan_uv: Any) -> Any:
    """Validate and move ``sparse_plan_uv`` of shape ``(M_padded, 2)`` onto device as int32."""
    if isinstance(sparse_plan_uv, dict):
        sparse_plan_uv = sparse_plan_uv["uv_indices"]
    d_uv = cp.ascontiguousarray(cp.asarray(sparse_plan_uv), dtype=cp.int32)
    if d_uv.ndim != 2 or int(d_uv.shape[1]) != 2:
        raise ValueError(f"sparse_plan_uv must have shape (M_padded, 2); got {d_uv.shape}")
    return d_uv


def _build_csr_layout_gpu(
    d_counts_per_cell: Any,
    d_sparse_uv: Any,
    active_only: bool = True,
) -> Dict[str, Any]:
    """
    Build global CSR ``cell_offsets``, ``cell_uv``, and ``cell_to_active`` mapping on GPU.
    """
    m_padded = int(d_sparse_uv.shape[0])
    if active_only:
        d_active_cells = cp.flatnonzero(d_counts_per_cell > 0).astype(cp.uint32)
        m_active = int(d_active_cells.size)
        if m_active > 0:
            d_cell_uv = cp.ascontiguousarray(d_sparse_uv[d_active_cells], dtype=cp.int32)
            d_active_counts = d_counts_per_cell[d_active_cells]
            d_cell_offsets = cp.zeros(m_active + 1, dtype=cp.uint64)
            d_cell_offsets[1:] = cp.cumsum(d_active_counts, dtype=cp.uint64)
            d_cell_to_active = cp.full(m_padded, -1, dtype=cp.int32)
            d_cell_to_active[d_active_cells] = cp.arange(m_active, dtype=cp.int32)
        else:
            d_cell_uv = cp.empty((0, 2), dtype=cp.int32)
            d_cell_offsets = cp.zeros(1, dtype=cp.uint64)
            d_cell_to_active = cp.full(m_padded, -1, dtype=cp.int32)
    else:
        m_active = m_padded
        d_cell_uv = cp.ascontiguousarray(d_sparse_uv, dtype=cp.int32)
        d_cell_offsets = cp.zeros(m_active + 1, dtype=cp.uint64)
        if m_active > 0:
            d_cell_offsets[1:] = cp.cumsum(d_counts_per_cell, dtype=cp.uint64)
        d_cell_to_active = cp.arange(m_padded, dtype=cp.int32)

    nnz_total = int(d_cell_offsets[-1].item())
    return {
        "m_padded": m_padded,
        "m_active": m_active,
        "nnz_total": nnz_total,
        "cell_offsets": d_cell_offsets,
        "cell_uv": d_cell_uv,
        "cell_to_active": d_cell_to_active,
    }


def _scatter_one_fragment_gpu(
    mod: Any,
    d_c_idx: Any,
    d_c_cnt: Any,
    d_f_vis: Any,
    d_f_times: Any,
    d_cell_to_active: Any,
    d_write_heads: Any,
    d_vis_data: Any,
    d_vis_times: Any,
) -> None:
    """Scatter one chunk fragment's samples into global CSR arrays on GPU."""
    n_act = int(d_c_idx.size)
    if n_act == 0 or int(d_f_vis.size) == 0:
        return

    d_cnt_u64 = d_c_cnt.astype(cp.uint64)
    d_src_offsets = cp.cumsum(d_cnt_u64, dtype=cp.uint64) - d_cnt_u64

    scatter_kernel = mod.get_function("scatter_fragment_to_csr_kernel")
    blocks = (n_act + _WARPS_PER_BLOCK - 1) // _WARPS_PER_BLOCK
    scatter_kernel(
        (blocks,),
        (_BLOCK_THREADS,),
        (
            d_c_idx,
            d_c_cnt,
            d_src_offsets,
            d_f_vis,
            d_f_times,
            d_cell_to_active,
            d_write_heads,
            d_vis_data,
            d_vis_times,
            np.int32(n_act),
        ),
    )


def _write_store_from_gpu(
    store_dir: str,
    d_cell_offsets: Any,
    d_cell_uv: Any,
    d_vis_data: Any,
    d_vis_times: Any,
    m_active: int,
    m_padded: int,
    nnz_total: int,
    active_only: bool,
    attrs: Optional[Dict[str, Any]],
    t_start: float,
) -> Dict[str, Any]:
    """Serialise consolidated GPU CSR arrays to ``store_dir`` matching :class:`RaggedDGAStore`."""
    os.makedirs(store_dir, exist_ok=True)
    cp.asnumpy(d_cell_offsets).astype(np.uint64, copy=False).tofile(
        os.path.join(store_dir, "cell_offsets.bin")
    )
    cp.asnumpy(d_cell_uv).astype(np.int32, copy=False).tofile(
        os.path.join(store_dir, "cell_uv.bin")
    )
    cp.asnumpy(d_vis_data).astype(np.complex64, copy=False).tofile(
        os.path.join(store_dir, "vis_data.bin")
    )
    cp.asnumpy(d_vis_times).astype(np.float64, copy=False).tofile(
        os.path.join(store_dir, "vis_times.bin")
    )

    store_attrs = dict(attrs) if attrs else {}
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
    with open(os.path.join(store_dir, "attrs.json"), "w", encoding="utf-8") as f:
        json.dump(store_attrs, f, indent=2)
    return store_attrs


def _validate_and_to_device_fragment(
    c_idx: Any,
    c_cnt: Any,
    f_vis: Any,
    f_times: Any,
) -> Tuple[Any, Any, Any, Any]:
    """Validate 1D CSR fragment shapes/counts and transfer arrays to GPU."""
    already_device = (
        isinstance(c_idx, cp.ndarray)
        and isinstance(c_cnt, cp.ndarray)
        and isinstance(f_vis, cp.ndarray)
        and isinstance(f_times, cp.ndarray)
        and c_idx.dtype == cp.uint32
        and c_cnt.dtype == cp.uint16
        and f_vis.dtype == cp.complex64
        and f_times.dtype == cp.float64
    )
    if already_device:
        d_c_idx = c_idx if c_idx.flags.c_contiguous else cp.ascontiguousarray(c_idx)
        d_c_cnt = c_cnt if c_cnt.flags.c_contiguous else cp.ascontiguousarray(c_cnt)
        d_f_vis = f_vis if f_vis.flags.c_contiguous else cp.ascontiguousarray(f_vis)
        d_f_times = f_times if f_times.flags.c_contiguous else cp.ascontiguousarray(f_times)
    else:
        d_c_idx = cp.ascontiguousarray(cp.asarray(c_idx), dtype=cp.uint32)
        d_c_cnt = cp.ascontiguousarray(cp.asarray(c_cnt), dtype=cp.uint16)
        d_f_vis = cp.ascontiguousarray(cp.asarray(f_vis), dtype=cp.complex64)
        d_f_times = cp.ascontiguousarray(cp.asarray(f_times), dtype=cp.float64)

    if d_c_idx.ndim != 1 or d_c_cnt.ndim != 1 or d_f_vis.ndim != 1 or d_f_times.ndim != 1:
        raise ValueError(
            "Fragment arrays must all be 1D; got shapes "
            f"c_idx={d_c_idx.shape}, c_cnt={d_c_cnt.shape}, vis={d_f_vis.shape}, times={d_f_times.shape}."
        )
    if int(d_c_idx.size) != int(d_c_cnt.size):
        raise ValueError(
            f"cell_indices length ({d_c_idx.size}) != cell_counts length ({d_c_cnt.size})."
        )
    if int(d_f_vis.size) != int(d_f_times.size):
        raise ValueError(
            f"vis_data length ({d_f_vis.size}) != vis_times length ({d_f_times.size})."
        )
    if not already_device:
        total_cnt = int(cp.sum(d_c_cnt, dtype=cp.uint64).item()) if int(d_c_cnt.size) > 0 else 0
        if total_cnt != int(d_f_vis.size):
            raise ValueError(
                f"Sum of cell_counts ({total_cnt}) != vis_data length ({d_f_vis.size})."
            )
    return d_c_idx, d_c_cnt, d_f_vis, d_f_times


def _unpack_chunk_entry(chunk: Any) -> Tuple[Any, Any, Any, Any]:
    """Normalise and validate a fragment tuple or dict into device arrays."""
    if isinstance(chunk, dict):
        c_idx = chunk["cell_indices"]
        c_cnt = chunk["cell_counts"]
        f_vis = chunk["vis_data"]
        f_times = chunk["vis_times"]
    elif isinstance(chunk, (tuple, list)) and len(chunk) == 4:
        c_idx, c_cnt, f_vis, f_times = chunk
    else:
        raise ValueError(
            "Each device chunk must be a 4-tuple (cell_indices, cell_counts, vis_data, vis_times) "
            "or a fragment dict."
        )
    return _validate_and_to_device_fragment(c_idx, c_cnt, f_vis, f_times)


def consolidate_device_chunks_to_gpu_store(
    device_chunks: Iterable[Any],
    sparse_plan_uv: Any,
    active_only: bool = True,
    store_dir: Optional[str] = None,
    attrs: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Consolidate an ordered sequence of ragged chunk fragments on GPU into a
    global CSR store.

    Returns a :class:`RaggedDGAStore` when ``store_dir`` is provided, or a dict
    of GPU CuPy arrays (``cell_offsets``, ``cell_uv``, ``vis_data``, ``vis_times``,
    ``attrs``) when ``store_dir`` is ``None``.
    """
    _require_cuda()
    t_start = time.time()
    mod = _get_module()
    accum_kernel = mod.get_function("accumulate_cell_counts_kernel")

    d_sparse_uv = _normalise_sparse_plan_uv(sparse_plan_uv)
    m_padded = int(d_sparse_uv.shape[0])

    chunks_list = [_unpack_chunk_entry(ch) for ch in device_chunks]
    d_counts_per_cell = cp.zeros(m_padded, dtype=cp.uint64)
    d_oob_flag = cp.zeros(1, dtype=cp.int32)

    for d_c_idx, d_c_cnt, _, _ in chunks_list:
        n_act = int(d_c_idx.size)
        if n_act == 0:
            continue
        blocks = (n_act + _BLOCK_THREADS - 1) // _BLOCK_THREADS
        accum_kernel(
            (blocks,),
            (_BLOCK_THREADS,),
            (d_c_idx, d_c_cnt, d_counts_per_cell, np.int32(n_act), np.int32(m_padded), d_oob_flag),
        )

    if int(d_oob_flag[0].item()) != 0:
        raise ValueError(f"Fragment cell_indices exceed sparse_plan_uv length ({m_padded}).")

    layout = _build_csr_layout_gpu(d_counts_per_cell, d_sparse_uv, active_only=active_only)
    m_active = layout["m_active"]
    nnz_total = layout["nnz_total"]
    d_cell_offsets = layout["cell_offsets"]
    d_cell_uv = layout["cell_uv"]
    d_cell_to_active = layout["cell_to_active"]

    d_vis_data = cp.empty(nnz_total, dtype=cp.complex64)
    d_vis_times = cp.empty(nnz_total, dtype=cp.float64)
    d_write_heads = d_cell_offsets[:-1].copy() if m_active > 0 else cp.empty(0, dtype=cp.uint64)

    for d_c_idx, d_c_cnt, d_f_vis, d_f_times in chunks_list:
        _scatter_one_fragment_gpu(
            mod,
            d_c_idx,
            d_c_cnt,
            d_f_vis,
            d_f_times,
            d_cell_to_active,
            d_write_heads,
            d_vis_data,
            d_vis_times,
        )

    if m_active > 0 and nnz_total > 0:
        if not bool(cp.all(d_write_heads == d_cell_offsets[1:]).item()):
            raise RuntimeError("GPU CSR consolidation write-head mismatch against cell_offsets.")

    if store_dir is not None:
        _write_store_from_gpu(
            store_dir=store_dir,
            d_cell_offsets=d_cell_offsets,
            d_cell_uv=d_cell_uv,
            d_vis_data=d_vis_data,
            d_vis_times=d_vis_times,
            m_active=m_active,
            m_padded=m_padded,
            nnz_total=nnz_total,
            active_only=active_only,
            attrs=attrs,
            t_start=t_start,
        )
        return RaggedDGAStore.open(store_dir, mode="r")

    store_attrs = dict(attrs) if attrs else {}
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
    return {
        "cell_offsets": d_cell_offsets,
        "cell_uv": d_cell_uv,
        "vis_data": d_vis_data,
        "vis_times": d_vis_times,
        "attrs": store_attrs,
    }


def consolidate_ragged_fragments_gpu(
    fragment_files: List[str],
    store_dir: str,
    sparse_plan_uv: Any,
    attrs: Optional[Dict[str, Any]] = None,
    active_only: bool = True,
    cleanup_fragments: bool = False,
) -> RaggedDGAStore:
    """
    GPU-accelerated consolidation of sharded ``.csr.bin`` fragment files into a
    consolidated :class:`RaggedDGAStore` on disk.
    """
    _require_cuda()
    t_start = time.time()
    os.makedirs(store_dir, exist_ok=True)
    mod = _get_module()
    accum_kernel = mod.get_function("accumulate_cell_counts_kernel")

    d_sparse_uv = _normalise_sparse_plan_uv(sparse_plan_uv)
    m_padded = int(d_sparse_uv.shape[0])

    # 1. Inspect fragment headers and sort chronologically by chunk_idx
    frag_metas = []
    for fpath in fragment_files:
        if not os.path.exists(fpath):
            continue
        c_idx, n_act, nnz, t_s, dt, c_indices, c_counts = read_ragged_fragment_header(fpath)
        frag_metas.append((c_idx, fpath, n_act, nnz, t_s, dt, c_indices, c_counts))
    frag_metas.sort(key=lambda x: x[0])

    # 2. Accumulate non-zero sample counts per cell on GPU
    d_counts_per_cell = cp.zeros(m_padded, dtype=cp.uint64)
    d_oob_flag = cp.zeros(1, dtype=cp.int32)
    for _, _, n_act, _, _, _, c_indices, c_counts in frag_metas:
        if n_act == 0:
            continue
        d_c_idx = cp.asarray(c_indices, dtype=cp.uint32)
        d_c_cnt = cp.asarray(c_counts, dtype=cp.uint16)
        blocks = (n_act + _BLOCK_THREADS - 1) // _BLOCK_THREADS
        accum_kernel(
            (blocks,),
            (_BLOCK_THREADS,),
            (d_c_idx, d_c_cnt, d_counts_per_cell, np.int32(n_act), np.int32(m_padded), d_oob_flag),
        )

    if int(d_oob_flag[0].item()) != 0:
        raise ValueError(f"Fragment cell_indices exceed sparse_plan_uv length ({m_padded}).")

    # 3. Form global cell offsets and active cell selection on GPU
    layout = _build_csr_layout_gpu(d_counts_per_cell, d_sparse_uv, active_only=active_only)
    m_active = layout["m_active"]
    nnz_total = layout["nnz_total"]
    d_cell_offsets = layout["cell_offsets"]
    d_cell_uv = layout["cell_uv"]
    d_cell_to_active = layout["cell_to_active"]

    # 4. Stream fragment payloads to GPU and transpose into CSR storage
    d_vis_data = cp.empty(nnz_total, dtype=cp.complex64)
    d_vis_times = cp.empty(nnz_total, dtype=cp.float64)
    d_write_heads = d_cell_offsets[:-1].copy() if m_active > 0 else cp.empty(0, dtype=cp.uint64)

    for _, fpath, _, nnz, _, _, _, _ in frag_metas:
        if nnz == 0:
            continue
        frag = read_ragged_fragment(fpath)
        d_c_idx, d_c_cnt, d_f_vis, d_f_times = _validate_and_to_device_fragment(
            frag["cell_indices"],
            frag["cell_counts"],
            frag["vis_data"],
            frag["vis_times"],
        )
        _scatter_one_fragment_gpu(
            mod,
            d_c_idx,
            d_c_cnt,
            d_f_vis,
            d_f_times,
            d_cell_to_active,
            d_write_heads,
            d_vis_data,
            d_vis_times,
        )

    if m_active > 0 and nnz_total > 0:
        if not bool(cp.all(d_write_heads == d_cell_offsets[1:]).item()):
            raise RuntimeError("GPU CSR consolidation write-head mismatch against cell_offsets.")

    # 5. Serialise consolidated CSR store to disk
    _write_store_from_gpu(
        store_dir=store_dir,
        d_cell_offsets=d_cell_offsets,
        d_cell_uv=d_cell_uv,
        d_vis_data=d_vis_data,
        d_vis_times=d_vis_times,
        m_active=m_active,
        m_padded=m_padded,
        nnz_total=nnz_total,
        active_only=active_only,
        attrs=attrs,
        t_start=t_start,
    )

    logger.info(
        f"[Consolidate CSR GPU] Assembled {len(frag_metas)} fragments -> {store_dir} "
        f"(M: {m_padded} -> {m_active}, NNZ: {nnz_total}, {time.time() - t_start:.2f}s)"
    )

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


__all__ = [
    "is_cuda_available",
    "pack_chunk_to_ragged_gpu",
    "pack_uv_chunk_to_ragged_gpu",
    "consolidate_device_chunks_to_gpu_store",
    "consolidate_ragged_fragments_gpu",
]
