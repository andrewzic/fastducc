# -*- coding: utf-8 -*-
"""
fastducc.gpu_c2r: Batched 2D Hermitian half-plane Complex-to-Real (C2R) imaging.

Transforms gridded Hermitian half-plane visibilities ``(u >= 0)`` — from dense
half-plane grids ``(nt, nv, u_half)``, compact active-cell matrices
``(M_active, nt)``, or ragged CSR stores — into cropped real dirty image cubes
``(nt, ny, nx)`` with optional DUCC polynomial deapodisation.

Conforms to fastducc grid and sign conventions:
  - Input half-plane grid has shape ``(B, nv, u_half)`` with ``u_half = grid_nu // 2 + 1``.
  - Self-conjugate columns ``u = 0`` and (for even ``grid_nu``) ``u = grid_nu // 2``
    are symmetrised before ``irfft2`` / ``ducc0.fft.c2r``.
  - Output dirty images are centred via FFT shift, cropped to ``(ny, nx)``,
    and optionally multiplied by separable DUCC deapodisation factors
    ``cfu[|dx|] * cfv[|dy|]``, matching ``ducc0.wgridder.vis2dirty`` flux scale.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import logging
import threading
from typing import Any, Optional, Tuple

import numpy as np

try:
    import cupy as cp
    import cupyx.scipy.fft as cufft

    HAS_CUPY = True
    _CUPY_IMPORT_ERROR: Optional[BaseException] = None
except Exception as exc:
    cp = None  # type: ignore[assignment]
    cufft = None  # type: ignore[assignment]
    HAS_CUPY = False
    _CUPY_IMPORT_ERROR = exc

try:
    import ducc0.fft as dfft

    HAS_DUCC_FFT = True
except ImportError:
    dfft = None  # type: ignore[assignment]
    HAS_DUCC_FFT = False

logger = logging.getLogger(__name__)

_CUFFT_C2R_PLAN_CACHE: "OrderedDict[Tuple[int, int, int, int, str], Any]" = OrderedDict()
_CUFFT_PLAN_CACHE_LOCK = threading.Lock()
_CUFFT_C2R_PLAN_CACHE_MAX = 16


def resolve_halfplane_dims(nv: int, nu: Optional[int]) -> Tuple[int, int]:
    """Resolve full U-dimension ``grid_nu`` and Hermitian half-plane width ``u_half``."""
    nv_i = int(nv)
    if nu is None:
        grid_nu = nv_i
        u_half = grid_nu // 2 + 1
    elif int(nu) <= nv_i // 2 + 1:
        u_half = int(nu)
        grid_nu = (u_half - 1) * 2
    else:
        grid_nu = int(nu)
        u_half = grid_nu // 2 + 1
    if nv_i <= 0 or grid_nu <= 0 or u_half <= 0:
        raise ValueError(
            f"Invalid C2R grid dimensions: nv={nv}, nu={nu} -> grid_nu={grid_nu}, u_half={u_half}."
        )
    return grid_nu, u_half


def compute_deapod_vectors(
    npix_x: int,
    npix_y: int,
    grid_nu: int,
    nv: int,
    *,
    raster_mode: str = "wgridder",
    epsilon: float = 1e-5,
    do_wgridding: bool = False,
    sigma_min: float = 1.1,
    sigma_max: float = 2.6,
    supp: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute 1D separable DUCC deapodisation vectors ``(corfunc_u, corfunc_v)``
    for ``raster_mode='wgridder'``, or ``(None, None)`` for nearest/bilinear modes.

    Returns ``cfu`` of shape ``(npix_x // 2 + 1,)`` and ``cfv`` of shape
    ``(npix_y // 2 + 1,)`` as ``float32`` arrays.
    """
    mode_norm = str(raster_mode).strip().lower()
    if mode_norm not in ("wgridder", "ducc_wgridder"):
        return None, None

    from fastducc.gpu_raster import build_ducc_corfunc, eval_ducc_corfunc, select_ducc_kernel_params

    kpar = select_ducc_kernel_params(
        epsilon=float(epsilon),
        do_wgridding=bool(do_wgridding),
        sigma_min=float(sigma_min),
        sigma_max=float(sigma_max),
        supp=supp,
    )
    cor_x, cor_wgt = build_ducc_corfunc(int(kpar["W"]), float(kpar["beta"]), float(kpar["e0"]))
    cfu = eval_ducc_corfunc(
        np.arange(int(npix_x) // 2 + 1, dtype=np.float64) / float(grid_nu),
        cor_x,
        cor_wgt,
    ).astype(np.float32)
    cfv = eval_ducc_corfunc(
        np.arange(int(npix_y) // 2 + 1, dtype=np.float64) / float(nv),
        cor_x,
        cor_wgt,
    ).astype(np.float32)
    return cfu, cfv


def _require_cupy(device_id: int = 0) -> None:
    if not HAS_CUPY or cp is None or cufft is None:
        raise RuntimeError(
            "CuPy with cupyx.scipy.fft is required for GPU C2R imaging."
        ) from _CUPY_IMPORT_ERROR
    cp.cuda.Device(int(device_id)).use()


def _get_c2r_plan(grid: Any, nv: int, grid_nu: int) -> Any:
    """Retrieve or construct a cached 2D batched cuFFT C2R plan."""
    key = (int(grid.device.id), int(grid.shape[0]), int(nv), int(grid_nu), str(grid.dtype))
    with _CUFFT_PLAN_CACHE_LOCK:
        plan = _CUFFT_C2R_PLAN_CACHE.get(key)
        if plan is not None:
            _CUFFT_C2R_PLAN_CACHE.move_to_end(key)
            return plan
    plan = cufft.get_fft_plan(
        grid, shape=(int(nv), int(grid_nu)), axes=(1, 2), value_type="C2R"
    )
    with _CUFFT_PLAN_CACHE_LOCK:
        _CUFFT_C2R_PLAN_CACHE[key] = plan
        _CUFFT_C2R_PLAN_CACHE.move_to_end(key)
        while len(_CUFFT_C2R_PLAN_CACHE) > _CUFFT_C2R_PLAN_CACHE_MAX:
            _CUFFT_C2R_PLAN_CACHE.popitem(last=False)
    return plan


_SCATTER_ACTIVE_TIME_BLOCK_SRC = r"""
extern "C" __global__
void scatter_active_time_block_kernel(
    const float2* __restrict__ vis_mt,
    const int* __restrict__ v_idx,
    const int* __restrict__ u_idx,
    float2* __restrict__ grid,
    int M,
    int nt_total,
    int t0,
    int B,
    int nv,
    int u_half
) {
    unsigned long long total = (unsigned long long)B * (unsigned long long)M;
    unsigned long long gid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;

    for (unsigned long long idx = gid; idx < total; idx += stride) {
        int b = (int)(idx / (unsigned long long)M);
        int m = (int)(idx - (unsigned long long)b * (unsigned long long)M);
        int v = v_idx[m];
        int u = u_idx[m];
        if (v >= 0 && v < nv && u >= 0 && u < u_half) {
            unsigned long long src = (unsigned long long)m * (unsigned long long)nt_total
                                     + (unsigned long long)(t0 + b);
            unsigned long long dst = ((unsigned long long)b * (unsigned long long)nv + (unsigned long long)v)
                                     * (unsigned long long)u_half + (unsigned long long)u;
            grid[dst] = vis_mt[src];
        }
    }
}
"""

_ENFORCE_HERMITIAN_COL_SRC = r"""
extern "C" __global__
void enforce_hermitian_col_kernel(
    float2* __restrict__ grid,
    int B,
    int nv,
    int u_half,
    int grid_nu
) {
    int v_half = nv / 2 + 1;
    int u_nyq = ((grid_nu & 1) == 0) ? (grid_nu / 2) : -1;
    int n_cols = (u_nyq > 0 && u_nyq < u_half) ? 2 : 1;
    unsigned long long total = (unsigned long long)B * (unsigned long long)v_half * (unsigned long long)n_cols;
    unsigned long long gid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;

    for (unsigned long long idx = gid; idx < total; idx += stride) {
        int col_i = (int)(idx % (unsigned long long)n_cols);
        unsigned long long rem = idx / (unsigned long long)n_cols;
        int v = (int)(rem % (unsigned long long)v_half);
        int b = (int)(rem / (unsigned long long)v_half);
        int u = (col_i == 0) ? 0 : u_nyq;

        unsigned long long row_stride = (unsigned long long)u_half;
        unsigned long long base = (unsigned long long)b * (unsigned long long)nv * row_stride
                                  + (unsigned long long)u;

        if (v == 0 || (2 * v == nv)) {
            unsigned long long p = base + (unsigned long long)v * row_stride;
            float2 val = grid[p];
            grid[p] = make_float2(val.x, 0.0f);
        } else {
            int v_conj = nv - v;
            unsigned long long p1 = base + (unsigned long long)v * row_stride;
            unsigned long long p2 = base + (unsigned long long)v_conj * row_stride;
            float2 a = grid[p1];
            float2 bv = grid[p2];
            float re = 0.5f * (a.x + bv.x);
            float im = 0.5f * (a.y - bv.y);
            grid[p1] = make_float2(re, im);
            grid[p2] = make_float2(re, -im);
        }
    }
}
"""

_CROP_REAL_FOV_DEAPOD_SRC = r"""
extern "C" __global__
void crop_real_fov_deapod_kernel(
    const float* __restrict__ dirty,
    float* __restrict__ cropped,
    const float* __restrict__ cfu,
    const float* __restrict__ cfv,
    int B,
    int nv,
    int grid_nu,
    int ny,
    int nx,
    float scale
) {
    int half_ny = ny / 2;
    int half_nx = nx / 2;
    unsigned long long total = (unsigned long long)B * (unsigned long long)ny * (unsigned long long)nx;
    unsigned long long gid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;

    for (unsigned long long idx = gid; idx < total; idx += stride) {
        int x = (int)(idx % (unsigned long long)nx);
        unsigned long long rem = idx / (unsigned long long)nx;
        int y = (int)(rem % (unsigned long long)ny);
        int b = (int)(rem / (unsigned long long)ny);

        int dy = y - half_ny;
        int dx = x - half_nx;
        int y_src = dy % nv;
        if (y_src < 0) y_src += nv;
        int x_src = dx % grid_nu;
        if (x_src < 0) x_src += grid_nu;

        unsigned long long src_idx = ((unsigned long long)b * (unsigned long long)nv + (unsigned long long)y_src)
                                     * (unsigned long long)grid_nu + (unsigned long long)x_src;
        float fac = scale;
        if (cfu != nullptr && cfv != nullptr) {
            int icfu = (dx < 0) ? -dx : dx;
            int icfv = (dy < 0) ? -dy : dy;
            fac *= cfu[icfu] * cfv[icfv];
        }
        cropped[idx] = dirty[src_idx] * fac;
    }
}
"""

if HAS_CUPY:
    _SCATTER_ACTIVE_TIME_BLOCK_KERNEL = cp.RawKernel(
        _SCATTER_ACTIVE_TIME_BLOCK_SRC, "scatter_active_time_block_kernel"
    )
    _ENFORCE_HERMITIAN_COL_KERNEL = cp.RawKernel(
        _ENFORCE_HERMITIAN_COL_SRC, "enforce_hermitian_col_kernel"
    )
    _CROP_REAL_FOV_DEAPOD_KERNEL = cp.RawKernel(
        _CROP_REAL_FOV_DEAPOD_SRC, "crop_real_fov_deapod_kernel"
    )
else:
    _SCATTER_ACTIVE_TIME_BLOCK_KERNEL = None
    _ENFORCE_HERMITIAN_COL_KERNEL = None
    _CROP_REAL_FOV_DEAPOD_KERNEL = None


@dataclass
class HalfPlaneC2RWorkspace:
    """Reusable GPU workspace for batched half-plane C2R dirty imaging."""

    max_batch: int
    nv: int
    grid_nu: int
    u_half: int
    ny: int
    nx: int
    device_id: int = 0
    grid_buf: Optional[Any] = None
    out_buf: Optional[Any] = None

    @classmethod
    def create(
        cls,
        max_batch: int,
        nv: int,
        nu: Optional[int],
        ny: int,
        nx: int,
        device_id: int = 0,
    ) -> "HalfPlaneC2RWorkspace":
        _require_cupy(device_id)
        grid_nu, u_half = resolve_halfplane_dims(int(nv), nu)
        b = max(1, int(max_batch))
        with cp.cuda.Device(int(device_id)):
            grid_buf = cp.empty((b, int(nv), u_half), dtype=cp.complex64)
            out_buf = cp.empty((b, int(ny), int(nx)), dtype=cp.float32)
        return cls(
            max_batch=b,
            nv=int(nv),
            grid_nu=grid_nu,
            u_half=u_half,
            ny=int(ny),
            nx=int(nx),
            device_id=int(device_id),
            grid_buf=grid_buf,
            out_buf=out_buf,
        )

    def ensure_buffers(self, batch_size: int) -> Tuple[Any, Any]:
        _require_cupy(self.device_id)
        b = int(batch_size)
        if (
            self.grid_buf is None
            or self.out_buf is None
            or b > self.max_batch
        ):
            new_b = max(b, self.max_batch)
            with cp.cuda.Device(int(self.device_id)):
                self.grid_buf = cp.empty((new_b, self.nv, self.u_half), dtype=cp.complex64)
                self.out_buf = cp.empty((new_b, self.ny, self.nx), dtype=cp.float32)
            self.max_batch = new_b
        return self.grid_buf[:b], self.out_buf[:b]

    def release(self) -> None:
        self.grid_buf = None
        self.out_buf = None
        if HAS_CUPY and cp is not None:
            with cp.cuda.Device(int(self.device_id)):
                cp.get_default_memory_pool().free_all_blocks()


def _c2r_crop_block_gpu(
    grid_blk: Any,
    out_blk: Any,
    nv: int,
    u_half: int,
    grid_nu: int,
    ny: int,
    nx: int,
    d_cfu_ptr: Any,
    d_cfv_ptr: Any,
) -> Any:
    """Enforce Hermitian columns, run batched cuFFT C2R, crop FOV and apply deapodisation."""
    B = int(grid_blk.shape[0])
    threads = 256
    v_half = nv // 2 + 1
    n_self_cols = 2 if ((grid_nu & 1) == 0 and (grid_nu // 2) > 0 and (grid_nu // 2) < u_half) else 1
    total_herm = B * v_half * n_self_cols
    herm_blocks = min(65535, (total_herm + threads - 1) // threads)
    _ENFORCE_HERMITIAN_COL_KERNEL(
        (herm_blocks,),
        (threads,),
        (
            grid_blk,
            np.int32(B),
            np.int32(nv),
            np.int32(u_half),
            np.int32(grid_nu),
        ),
    )

    plan = _get_c2r_plan(grid_blk, nv, grid_nu)
    dirty_full = cufft.irfft2(
        grid_blk,
        s=(int(nv), int(grid_nu)),
        axes=(1, 2),
        norm="forward",
        overwrite_x=True,
        plan=plan,
    )

    scale = np.float32(1.0)
    crop_total = B * ny * nx
    crop_blocks = min(65535, (crop_total + threads - 1) // threads)
    _CROP_REAL_FOV_DEAPOD_KERNEL(
        (crop_blocks,),
        (threads,),
        (
            dirty_full,
            out_blk,
            d_cfu_ptr,
            d_cfv_ptr,
            np.int32(B),
            np.int32(nv),
            np.int32(grid_nu),
            np.int32(ny),
            np.int32(nx),
            scale,
        ),
    )
    del dirty_full
    return out_blk


def halfplane_grid_to_dirty_cube_gpu(
    uv_half: Any,
    grid_nu: int,
    npix_y: int,
    npix_x: int,
    *,
    corfunc_u: Optional[Any] = None,
    corfunc_v: Optional[Any] = None,
    c2r_batch_size: int = 256,
    workspace: Optional[HalfPlaneC2RWorkspace] = None,
    device_id: int = 0,
    return_host: bool = True,
) -> Any:
    """
    Invert a dense Hermitian half-plane UV cube ``(nt, nv, u_half)`` into a real
    dirty image cube ``(nt, npix_y, npix_x)`` on GPU.
    """
    _require_cupy(device_id)
    with cp.cuda.Device(int(device_id)):
        d_uv = cp.ascontiguousarray(cp.asarray(uv_half), dtype=cp.complex64)
        if d_uv.ndim != 3:
            raise ValueError(f"uv_half must be 3D (nt, nv, u_half); got shape {d_uv.shape}.")
        nt, nv, u_half = (int(d_uv.shape[0]), int(d_uv.shape[1]), int(d_uv.shape[2]))
        ny, nx = int(npix_y), int(npix_x)
        eff_grid_nu = int(grid_nu)
        if u_half != eff_grid_nu // 2 + 1:
            raise ValueError(f"u_half ({u_half}) != grid_nu // 2 + 1 ({eff_grid_nu // 2 + 1}).")

        if nt == 0:
            empty = np.empty((0, ny, nx), dtype=np.float32) if return_host else cp.empty((0, ny, nx), dtype=cp.float32)
            return empty

        d_cfu_ptr: Any = (
            cp.ascontiguousarray(cp.asarray(corfunc_u, dtype=cp.float32)).ravel()
            if (corfunc_u is not None and corfunc_v is not None)
            else 0
        )
        d_cfv_ptr: Any = (
            cp.ascontiguousarray(cp.asarray(corfunc_v, dtype=cp.float32)).ravel()
            if (corfunc_u is not None and corfunc_v is not None)
            else 0
        )

        batch_sz = max(1, min(int(c2r_batch_size), nt))
        owned_ws = workspace is None
        ws = workspace if workspace is not None else HalfPlaneC2RWorkspace.create(
            max_batch=batch_sz, nv=nv, nu=eff_grid_nu, ny=ny, nx=nx, device_id=device_id
        )

        out_cube = np.empty((nt, ny, nx), dtype=np.float32) if return_host else cp.empty((nt, ny, nx), dtype=cp.float32)
        try:
            for t0 in range(0, nt, batch_sz):
                t1 = min(nt, t0 + batch_sz)
                b_curr = t1 - t0
                grid_blk, out_blk = ws.ensure_buffers(b_curr)
                cp.copyto(grid_blk, d_uv[t0:t1])
                _c2r_crop_block_gpu(
                    grid_blk, out_blk, nv, u_half, eff_grid_nu, ny, nx, d_cfu_ptr, d_cfv_ptr
                )
                if return_host:
                    out_cube[t0:t1] = cp.asnumpy(out_blk)
                else:
                    out_cube[t0:t1] = out_blk
        finally:
            if owned_ws:
                ws.release()
        return out_cube


def sparse_active_to_dirty_cube_gpu(
    vis_active: Any,
    cell_uv: Any,
    nv: int,
    grid_nu: int,
    npix_y: int,
    npix_x: int,
    *,
    corfunc_u: Optional[Any] = None,
    corfunc_v: Optional[Any] = None,
    c2r_batch_size: int = 256,
    workspace: Optional[HalfPlaneC2RWorkspace] = None,
    device_id: int = 0,
    return_host: bool = True,
) -> Any:
    """
    Scatter a compact active-cell matrix ``(M_active, nt)`` onto Hermitian
    half-plane grids and invert via batched GPU 2D C2R FFT into a dirty image
    cube ``(nt, npix_y, npix_x)``.
    """
    _require_cupy(device_id)
    with cp.cuda.Device(int(device_id)):
        d_vis = cp.ascontiguousarray(cp.asarray(vis_active), dtype=cp.complex64)
        if d_vis.ndim != 2:
            raise ValueError(f"vis_active must be 2D (M_active, nt); got shape {d_vis.shape}.")
        m_active, nt = int(d_vis.shape[0]), int(d_vis.shape[1])
        ny, nx = int(npix_y), int(npix_x)
        nv_i, eff_grid_nu = int(nv), int(grid_nu)
        u_half = eff_grid_nu // 2 + 1

        if nt == 0 or m_active == 0:
            if return_host:
                return np.zeros((nt, ny, nx), dtype=np.float32)
            return cp.zeros((nt, ny, nx), dtype=cp.float32)

        d_uv_idx = cp.ascontiguousarray(cp.asarray(cell_uv, dtype=cp.int32))
        if d_uv_idx.ndim != 2 or int(d_uv_idx.shape[0]) != m_active or int(d_uv_idx.shape[1]) != 2:
            raise ValueError(f"cell_uv must have shape ({m_active}, 2); got {d_uv_idx.shape}.")
        d_v_idx = cp.ascontiguousarray(d_uv_idx[:, 0], dtype=cp.int32)
        d_u_idx = cp.ascontiguousarray(d_uv_idx[:, 1], dtype=cp.int32)

        d_cfu_ptr: Any = (
            cp.ascontiguousarray(cp.asarray(corfunc_u, dtype=cp.float32)).ravel()
            if (corfunc_u is not None and corfunc_v is not None)
            else 0
        )
        d_cfv_ptr: Any = (
            cp.ascontiguousarray(cp.asarray(corfunc_v, dtype=cp.float32)).ravel()
            if (corfunc_u is not None and corfunc_v is not None)
            else 0
        )

        batch_sz = max(1, min(int(c2r_batch_size), nt))
        owned_ws = workspace is None
        ws = workspace if workspace is not None else HalfPlaneC2RWorkspace.create(
            max_batch=batch_sz, nv=nv_i, nu=eff_grid_nu, ny=ny, nx=nx, device_id=device_id
        )

        out_cube = np.empty((nt, ny, nx), dtype=np.float32) if return_host else cp.empty((nt, ny, nx), dtype=cp.float32)
        threads = 256
        try:
            for t0 in range(0, nt, batch_sz):
                t1 = min(nt, t0 + batch_sz)
                b_curr = t1 - t0
                grid_blk, out_blk = ws.ensure_buffers(b_curr)
                grid_blk.fill(0.0 + 0.0j)
                total_scatter = b_curr * m_active
                scatter_blocks = min(65535, (total_scatter + threads - 1) // threads)
                _SCATTER_ACTIVE_TIME_BLOCK_KERNEL(
                    (scatter_blocks,),
                    (threads,),
                    (
                        d_vis,
                        d_v_idx,
                        d_u_idx,
                        grid_blk,
                        np.int32(m_active),
                        np.int32(nt),
                        np.int32(t0),
                        np.int32(b_curr),
                        np.int32(nv_i),
                        np.int32(u_half),
                    ),
                )
                _c2r_crop_block_gpu(
                    grid_blk, out_blk, nv_i, u_half, eff_grid_nu, ny, nx, d_cfu_ptr, d_cfv_ptr
                )
                if return_host:
                    out_cube[t0:t1] = cp.asnumpy(out_blk)
                else:
                    out_cube[t0:t1] = out_blk
        finally:
            if owned_ws:
                ws.release()
        return out_cube


def _enforce_hermitian_columns_cpu(grid: np.ndarray, grid_nu: int) -> None:
    """In-place Hermitian symmetry enforcement on u=0 and u=grid_nu//2 (if even)."""
    nv = int(grid.shape[1])
    u_half = int(grid.shape[2])
    cols = [0]
    if (grid_nu & 1) == 0 and 0 < (grid_nu // 2) < u_half:
        cols.append(grid_nu // 2)
    v_mid = nv // 2
    for u in cols:
        grid[:, 0, u] = grid[:, 0, u].real + 0.0j
        if (nv & 1) == 0:
            grid[:, v_mid, u] = grid[:, v_mid, u].real + 0.0j
        v_pos = np.arange(1, (nv + 1) // 2)
        v_neg = nv - v_pos
        a = grid[:, v_pos, u]
        b = grid[:, v_neg, u]
        re = 0.5 * (a.real + b.real)
        im = 0.5 * (a.imag - b.imag)
        grid[:, v_pos, u] = re + 1j * im
        grid[:, v_neg, u] = re - 1j * im


def halfplane_grid_to_dirty_cube_cpu(
    uv_half: np.ndarray,
    grid_nu: int,
    npix_y: int,
    npix_x: int,
    *,
    corfunc_u: Optional[np.ndarray] = None,
    corfunc_v: Optional[np.ndarray] = None,
    nthreads: int = 1,
) -> np.ndarray:
    """
    CPU fallback: invert dense Hermitian half-plane UV cube ``(nt, nv, u_half)``
    to real dirty image cube ``(nt, npix_y, npix_x)`` with optional deapodisation.
    """
    uv = np.array(uv_half, dtype=np.complex64, copy=True)
    if uv.ndim != 3:
        raise ValueError(f"uv_half must be 3D (nt, nv, u_half); got shape {uv.shape}.")
    nt, nv, u_half = uv.shape
    ny, nx = int(npix_y), int(npix_x)
    eff_grid_nu = int(grid_nu)
    if nt == 0:
        return np.empty((0, ny, nx), dtype=np.float32)

    _enforce_hermitian_columns_cpu(uv, eff_grid_nu)

    if HAS_DUCC_FFT and dfft is not None:
        dirty_full = np.empty((nt, nv, eff_grid_nu), dtype=np.float32)
        dfft.c2r(
            uv,
            axes=(1, 2),
            lastsize=eff_grid_nu,
            forward=False,
            inorm=0,
            out=dirty_full,
            nthreads=max(1, int(nthreads)),
        )
    else:
        dirty_full = np.fft.irfft2(
            uv, s=(nv, eff_grid_nu), axes=(1, 2), norm="forward"
        ).astype(np.float32, copy=False)

    half_ny = ny // 2
    half_nx = nx // 2
    idx_y = (np.arange(ny, dtype=np.int64) - half_ny) % nv
    idx_x = (np.arange(nx, dtype=np.int64) - half_nx) % eff_grid_nu
    cropped = dirty_full[:, idx_y[:, None], idx_x[None, :]].astype(np.float32, copy=False)

    if corfunc_u is not None and corfunc_v is not None:
        dx_abs = np.abs(np.arange(nx, dtype=np.int64) - half_nx)
        dy_abs = np.abs(np.arange(ny, dtype=np.int64) - half_ny)
        deapod_2d = (
            np.asarray(corfunc_v, dtype=np.float32)[dy_abs][:, None]
            * np.asarray(corfunc_u, dtype=np.float32)[dx_abs][None, :]
        )
        cropped = (cropped * deapod_2d[None, :, :]).astype(np.float32, copy=False)

    return np.ascontiguousarray(cropped, dtype=np.float32)


def sparse_active_to_dirty_cube_cpu(
    vis_active: np.ndarray,
    cell_uv: np.ndarray,
    nv: int,
    grid_nu: int,
    npix_y: int,
    npix_x: int,
    *,
    corfunc_u: Optional[np.ndarray] = None,
    corfunc_v: Optional[np.ndarray] = None,
    nthreads: int = 1,
) -> np.ndarray:
    """
    CPU fallback: scatter compact active-cell matrix ``(M_active, nt)`` onto
    Hermitian half-plane grids and invert to ``(nt, npix_y, npix_x)`` dirty cube.
    """
    vis = np.asarray(vis_active, dtype=np.complex64)
    if vis.ndim != 2:
        raise ValueError(f"vis_active must be 2D (M_active, nt); got shape {vis.shape}.")
    m_active, nt = vis.shape
    ny, nx = int(npix_y), int(npix_x)
    nv_i, eff_grid_nu = int(nv), int(grid_nu)
    u_half = eff_grid_nu // 2 + 1
    if nt == 0 or m_active == 0:
        return np.zeros((nt, ny, nx), dtype=np.float32)

    uv_idx = np.asarray(cell_uv, dtype=np.int64)
    if uv_idx.ndim != 2 or uv_idx.shape != (m_active, 2):
        raise ValueError(f"cell_uv must have shape ({m_active}, 2); got {uv_idx.shape}.")

    grid = np.zeros((nt, nv_i, u_half), dtype=np.complex64)
    v_idx = uv_idx[:, 0]
    u_idx = uv_idx[:, 1]
    valid = (v_idx >= 0) & (v_idx < nv_i) & (u_idx >= 0) & (u_idx < u_half)
    if np.any(valid):
        grid[:, v_idx[valid], u_idx[valid]] = vis[valid, :].T

    return halfplane_grid_to_dirty_cube_cpu(
        grid,
        grid_nu=eff_grid_nu,
        npix_y=ny,
        npix_x=nx,
        corfunc_u=corfunc_u,
        corfunc_v=corfunc_v,
        nthreads=nthreads,
    )


__all__ = [
    "HAS_CUPY",
    "HalfPlaneC2RWorkspace",
    "compute_deapod_vectors",
    "halfplane_grid_to_dirty_cube_cpu",
    "halfplane_grid_to_dirty_cube_gpu",
    "resolve_halfplane_dims",
    "sparse_active_to_dirty_cube_cpu",
    "sparse_active_to_dirty_cube_gpu",
]
