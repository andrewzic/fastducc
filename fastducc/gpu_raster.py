# -*- coding: utf-8 -*-
"""
fastducc.gpu_raster: Fused CUDA visibility staging, brute-force channel
dedispersion, and direct-to-grid Hermitian half-plane rasterisation with a
Numba/NumPy CPU fallback.

Conforms strictly to fastducc grid, sign, and dedispersion conventions:
  - No sign inversion on MS ``(u_m, v_m)`` coordinates (matching
    ``ducc0.wgridder.vis2grid(..., flip_u=False, flip_v=False)`` and Stage 3
    ``c2r`` imaging).
  - Un-oversampled default grid geometry: ``grid_nu = npix_x``,
    ``nv = npix_y``, ``u_half = grid_nu // 2 + 1``,
    ``du = 1.0 / (grid_nu * pix_rad)``, ``dv = 1.0 / (nv * pix_rad)``.
  - Non-negative causal delays ``delays[c] >= 0`` from
    ``fastducc.dedisp.get_dm_delays(..., align_to='fmax')``.
  - Degrades gracefully when CuPy or a CUDA device is unavailable.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cupy as cp

    _CUPY_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as exc:
    cp = None  # type: ignore[assignment]
    _CUPY_IMPORT_ERROR = exc

try:
    import numba as nb

    HAS_NUMBA = True
except ImportError:
    nb = None  # type: ignore[assignment]
    HAS_NUMBA = False

from fastducc.constants import C_LIGHT
from fastducc.dedisp import get_dm_delays

logger = logging.getLogger(__name__)

_CUDA_KERNEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "cuda",
    "streaming_dedisp_grid.cu",
)
_MODULE_CACHE: Dict[int, Any] = {}
_MODULE_LOCK = threading.Lock()
_MAX_GRID_Y = 65535


def _ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def is_cuda_available() -> bool:
    """
    Check whether CuPy is importable and at least one CUDA device is accessible.

    Returns
    -------
    available : bool
        True if CUDA execution via CuPy is available, False otherwise.
    """
    if cp is None:
        logger.debug("CuPy unavailable for gpu_raster: %s", _CUPY_IMPORT_ERROR)
        return False
    try:
        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except cp.cuda.runtime.CUDARuntimeError as exc:
        logger.warning("CUDA device query failed in gpu_raster: %s", exc)
        return False


def _require_cuda() -> None:
    """Raise RuntimeError if CuPy or a CUDA device is unavailable."""
    if cp is None:
        raise RuntimeError(
            f"CuPy is required for CUDA rasterisation, but failed to import: {_CUPY_IMPORT_ERROR}"
        )
    try:
        n_dev = int(cp.cuda.runtime.getDeviceCount())
    except cp.cuda.runtime.CUDARuntimeError as exc:
        raise RuntimeError(f"CUDA device query failed: {exc}") from exc
    if n_dev <= 0:
        raise RuntimeError("CUDA rasterisation requested, but no CUDA devices were detected.")


def resolve_device(device: str = "auto") -> str:
    """
    Resolve compute backend selector ``device`` into ``'cuda'`` or ``'cpu'``.

    Parameters
    ----------
    device : {'auto', 'cuda', 'cpu'}
        Requested execution backend. ``'auto'`` selects ``'cuda'`` when a
        CUDA device is accessible via CuPy, otherwise falls back to ``'cpu'``.

    Returns
    -------
    resolved : {'cuda', 'cpu'}
        Selected execution backend.
    """
    dev = str("auto" if device is None else device).strip().lower()
    if dev not in ("auto", "cuda", "cpu"):
        raise ValueError(f"Unsupported device='{device}'; expected 'auto', 'cuda', or 'cpu'.")
    if dev == "cpu":
        return "cpu"
    if dev == "cuda":
        _require_cuda()
        return "cuda"
    if is_cuda_available():
        return "cuda"
    logger.debug("gpu_raster(device='auto') falling back to CPU: CUDA/CuPy unavailable.")
    return "cpu"


def _get_module() -> Any:
    """Lazily compile and cache the CUDA RawModule for the active CUDA device."""
    _require_cuda()
    dev_id = int(cp.cuda.Device().id)
    with _MODULE_LOCK:
        mod = _MODULE_CACHE.get(dev_id)
        if mod is None:
            if not os.path.exists(_CUDA_KERNEL_PATH):
                raise FileNotFoundError(
                    f"CUDA kernel source file not found at '{_CUDA_KERNEL_PATH}'."
                )
            with open(_CUDA_KERNEL_PATH, "r", encoding="utf-8") as fh:
                src = fh.read()
            mod = cp.RawModule(
                code=src,
                options=(
                    "--std=c++17",
                    "--generate-code=arch=compute_75,code=sm_75",
                    "--generate-code=arch=compute_80,code=sm_80",
                    "--generate-code=arch=compute_80,code=compute_80",
                ),
            )
            _MODULE_CACHE[dev_id] = mod
        return mod


def compute_raster_grid_geometry(
    npix_x: int = 384,
    npix_y: int = 384,
    pix_rad: Optional[float] = None,
    pixsize_arcsec: Optional[float] = None,
    grid_nu: Optional[int] = None,
    nv: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute Hermitian half-plane direct rasterisation grid dimensions and cell spacings.

    Mathematical definition:
        grid_nu = npix_x (unless overridden)
        nv      = npix_y (unless overridden)
        u_half  = grid_nu // 2 + 1
        pix_rad = pixsize_arcsec / 206265.0 (unless passed explicitly)
        du      = 1.0 / (grid_nu * pix_rad)  [wavelengths per u cell]
        dv      = 1.0 / (nv * pix_rad)       [wavelengths per v cell]

    Parameters
    ----------
    npix_x, npix_y : int
        Default un-oversampled image dimensions in pixels.
    pix_rad : float, optional
        Angular pixel scale in radians/pixel. Overrides ``pixsize_arcsec``.
    pixsize_arcsec : float, optional
        Angular pixel scale in arcseconds/pixel (defaults to 22.0 if ``pix_rad`` is None).
    grid_nu : int, optional
        Explicit full u-axis grid size override.
    nv : int, optional
        Explicit v-axis grid size override.

    Returns
    -------
    geom : dict
        Dictionary with keys ``grid_nu``, ``nv``, ``u_half``, ``pix_rad``, ``du``, ``dv``.
    """
    eff_nu = int(npix_x if grid_nu is None else grid_nu)
    eff_nv = int(npix_y if nv is None else nv)
    if eff_nu <= 0 or eff_nv <= 0:
        raise ValueError(f"Grid dimensions must be positive; got grid_nu={eff_nu}, nv={eff_nv}.")

    if pix_rad is not None:
        eff_pix_rad = float(pix_rad)
    else:
        arcsec = 22.0 if pixsize_arcsec is None else float(pixsize_arcsec)
        eff_pix_rad = arcsec / 206265.0

    if not math.isfinite(eff_pix_rad) or eff_pix_rad <= 0.0:
        raise ValueError(f"pix_rad must be finite and positive; got {eff_pix_rad}.")

    u_half = eff_nu // 2 + 1
    du = 1.0 / (float(eff_nu) * eff_pix_rad)
    dv = 1.0 / (float(eff_nv) * eff_pix_rad)
    return {
        "grid_nu": eff_nu,
        "nv": eff_nv,
        "u_half": u_half,
        "pix_rad": eff_pix_rad,
        "du": du,
        "dv": dv,
    }


def compute_subband_slices(
    nchan: int,
    collapse_channels: bool = False,
    nsubbands: int = 1,
) -> Tuple[np.ndarray, int]:
    """
    Compute channel index boundaries matching ``np.array_split(np.arange(nchan), nsub_eff)``.

    Regimes:
      - ``collapse_channels=True`` and ``nsubbands <= 1``: ``nsub_eff = 1`` (full collapse).
      - ``collapse_channels=True`` and ``nsubbands > 1``: ``nsub_eff = min(nsubbands, nchan)``.
      - ``collapse_channels=False`` and (``nsubbands <= 1`` or ``nsubbands >= nchan``):
        ``nsub_eff = nchan`` (per-channel).
      - ``collapse_channels=False`` and ``1 < nsubbands < nchan``: ``nsub_eff = nsubbands``.

    Parameters
    ----------
    nchan : int
        Total number of frequency channels.
    collapse_channels : bool
        Whether channel collapsing is enabled.
    nsubbands : int
        Requested number of subbands.

    Returns
    -------
    sub_offsets : np.ndarray (int32)
        Monotonic channel boundaries of shape ``(nsub_eff + 1,)``.
    nsub_eff : int
        Effective number of subbands.
    """
    nchan_int = int(nchan)
    if nchan_int <= 0:
        return np.zeros(1, dtype=np.int32), 0

    nsub_req = int(nsubbands)
    if collapse_channels:
        nsub_eff = 1 if nsub_req <= 1 else min(nsub_req, nchan_int)
    else:
        nsub_eff = nchan_int if (nsub_req <= 1 or nsub_req >= nchan_int) else nsub_req

    q, r = divmod(nchan_int, nsub_eff)
    sizes = np.empty(nsub_eff, dtype=np.int32)
    sizes[:r] = q + 1
    sizes[r:] = q
    offsets = np.zeros(nsub_eff + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(sizes, dtype=np.int32)
    return offsets, nsub_eff


def build_uv_to_m_lookup(
    nv: int,
    u_half: int,
    flat_indices: Optional[Any] = None,
    uv_indices: Optional[Any] = None,
    device: str = "cpu",
) -> Any:
    """
    Construct a 2D ``(nv, u_half)`` ``int32`` active-cell lookup table ``uv_to_m``.

    Maps active grid coordinates ``(v, u)`` to dense sparse-row indices
    ``0 .. M - 1`` and marks inactive cells with ``-1``.

    Parameters
    ----------
    nv : int
        Number of grid cells along the v axis.
    u_half : int
        Number of Hermitian half-plane cells along the u axis.
    flat_indices : array_like, optional
        1D array of linear indices ``v * u_half + u`` of shape ``(M,)``.
    uv_indices : array_like, optional
        2D array of ``[v, u]`` integer coordinates of shape ``(M, 2)``.
    device : {'cpu', 'cuda', 'auto'}
        Target array placement.

    Returns
    -------
    uv_to_m : np.ndarray or cupy.ndarray (int32)
        Lookup table of shape ``(nv, u_half)``.
    """
    nv_i = int(nv)
    uh_i = int(u_half)
    if nv_i <= 0 or uh_i <= 0:
        raise ValueError(f"nv and u_half must be positive; got nv={nv_i}, u_half={uh_i}.")
    if flat_indices is None and uv_indices is None:
        raise ValueError("Either flat_indices or uv_indices must be provided to build_uv_to_m_lookup.")

    eff_dev = resolve_device(device)
    table_cpu = np.full((nv_i, uh_i), -1, dtype=np.int32)

    if flat_indices is not None:
        f_idx = np.asarray(cp.asnumpy(flat_indices) if (cp is not None and isinstance(flat_indices, cp.ndarray)) else flat_indices, dtype=np.int64).ravel()
        spatial = nv_i * uh_i
        if f_idx.size > 0 and (np.any(f_idx < 0) or np.any(f_idx >= spatial)):
            raise ValueError(f"flat_indices contains out-of-bounds indices for spatial plane ({nv_i}, {uh_i}).")
        table_cpu.ravel()[f_idx] = np.arange(f_idx.size, dtype=np.int32)
    else:
        uv_arr = np.asarray(cp.asnumpy(uv_indices) if (cp is not None and isinstance(uv_indices, cp.ndarray)) else uv_indices, dtype=np.int64)
        if uv_arr.ndim != 2 or uv_arr.shape[1] != 2:
            raise ValueError(f"uv_indices must have shape (M, 2); got {uv_arr.shape}.")
        if uv_arr.shape[0] > 0:
            v_arr = uv_arr[:, 0]
            u_arr = uv_arr[:, 1]
            if np.any(v_arr < 0) or np.any(v_arr >= nv_i) or np.any(u_arr < 0) or np.any(u_arr >= uh_i):
                raise ValueError(f"uv_indices contains coordinates outside (nv={nv_i}, u_half={uh_i}).")
            table_cpu[v_arr, u_arr] = np.arange(uv_arr.shape[0], dtype=np.int32)

    if eff_dev == "cuda":
        return cp.asarray(table_cpu, dtype=cp.int32)
    return table_cpu


def _normalise_mode(mode: str) -> int:
    m = str(mode).strip().lower()
    if m == "nearest":
        return 0
    if m == "bilinear":
        return 1
    raise ValueError(f"Unsupported rasterisation mode='{mode}'; expected 'nearest' or 'bilinear'.")


def _resolve_corr_spec(
    ncorr: int,
    corr_mode: str,
    basis: str,
    single_pol: str,
    average_correlations: bool,
    corr_index: Optional[int],
    labels: Optional[List[str]],
    lbl2idx: Optional[Dict[str, int]],
) -> Tuple[int, int, int]:
    """
    Resolve correlation collapse configuration to ``(corr_mode_id, idx_0, idx_1)``
    matching ``fastducc.imaging._collapse_correlations``.
    """
    mode = str(corr_mode).strip()
    if mode == "average":
        if average_correlations:
            return 1, 0, -1
        ci = 0 if corr_index is None else int(corr_index)
        if ci < 0 or ci >= ncorr:
            raise ValueError(f"corr_index={ci} out of bounds for ncorr={ncorr}.")
        return 0, ci, -1

    if mode == "single":
        if lbl2idx is not None:
            if single_pol not in lbl2idx:
                raise ValueError(
                    f"Requested single_pol='{single_pol}' not present in MS correlations: {labels}"
                )
            ci = int(lbl2idx[single_pol])
        else:
            ci = 0 if corr_index is None else int(corr_index)
        if ci < 0 or ci >= ncorr:
            raise ValueError(f"Correlation index {ci} out of bounds for ncorr={ncorr}.")
        return 0, ci, -1

    if mode == "stokesI":
        if lbl2idx is not None:
            have_linear = ("XX" in lbl2idx) and ("YY" in lbl2idx)
            have_circular = ("RR" in lbl2idx) and ("LL" in lbl2idx)
            use_linear = False
            if basis == "linear":
                if not have_linear:
                    raise ValueError("basis='linear' requested but XX/YY not found in MS correlations")
                use_linear = True
            elif basis == "circular":
                if not have_circular:
                    raise ValueError("basis='circular' requested but RR/LL not found in MS correlations")
            else:
                if have_linear:
                    use_linear = True
                elif not have_circular:
                    raise ValueError("Cannot form Stokes I: XX/YY or RR/LL not present in MS correlations")
            i1, i2 = (lbl2idx["XX"], lbl2idx["YY"]) if use_linear else (lbl2idx["RR"], lbl2idx["LL"])
        else:
            if ncorr < 2:
                raise ValueError(f"Cannot form Stokes I with ncorr={ncorr} without parallel-hand correlations.")
            i1, i2 = (0, 3) if ncorr >= 4 else (0, 1)
        return 2, int(i1), int(i2)

    raise ValueError(f"Unknown corr_mode='{corr_mode}'. Use 'average', 'stokesI', or 'single'.")


def _to_host_f64(arr: Any) -> np.ndarray:
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr).astype(np.float64, copy=False)
    return np.asarray(arr, dtype=np.float64)


def _to_host_i32(arr: Any) -> np.ndarray:
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr).astype(np.int32, copy=False)
    return np.asarray(arr, dtype=np.int32)


def _resolve_delays_and_subbands(
    freqs_hz: Any,
    delays: Optional[Any],
    dm: float,
    dt: Optional[float],
    align_to: str,
    collapse_channels: bool,
    nsubbands: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Validate frequencies and channel delays, and compute subband partitions,
    subband mean frequencies (Hz), and representative subband delays (samples).
    """
    freqs_host = np.ascontiguousarray(_to_host_f64(freqs_hz), dtype=np.float64)
    if freqs_host.ndim != 1:
        raise ValueError(f"freqs_hz must be 1D; got shape {freqs_host.shape}.")
    nchan = int(freqs_host.shape[0])

    if delays is None:
        if float(dm) != 0.0:
            if dt is None or float(dt) <= 0.0:
                raise ValueError("dt > 0 must be provided when dm != 0.0 and delays is None.")
            delays_host = get_dm_delays(freqs_host, float(dm), float(dt), align_to=align_to).astype(np.int32)
        else:
            delays_host = np.zeros(nchan, dtype=np.int32)
    else:
        delays_host = np.ascontiguousarray(_to_host_i32(delays), dtype=np.int32)
        if delays_host.ndim != 1 or int(delays_host.shape[0]) != nchan:
            raise ValueError(
                f"delays length ({delays_host.shape}) must match freqs_hz length ({nchan})."
            )

    sub_offsets, nsub_eff = compute_subband_slices(
        nchan=nchan,
        collapse_channels=collapse_channels,
        nsubbands=nsubbands,
    )
    sub_freqs_host = np.empty(nsub_eff, dtype=np.float64)
    sub_delays_host = np.empty(nsub_eff, dtype=np.int32)
    for s in range(nsub_eff):
        c0 = int(sub_offsets[s])
        c1 = int(sub_offsets[s + 1])
        sub_freqs_host[s] = float(np.mean(freqs_host[c0:c1]))
        if c1 - c0 == 1:
            sub_delays_host[s] = delays_host[c0]
        else:
            sub_delays_host[s] = np.int32(np.rint(np.mean(delays_host[c0:c1])))

    return freqs_host, delays_host, sub_offsets, sub_freqs_host, sub_delays_host, nsub_eff


def _resolve_uv_to_m_and_m_active(
    uv_to_m: Optional[Any],
    flat_indices: Optional[Any],
    nv: int,
    u_half: int,
    m_active: Optional[int],
    return_sparse: bool,
    target_device: str,
) -> Tuple[Optional[Any], int]:
    """Normalise optional ``uv_to_m`` lookup table and infer active cell count M."""
    if uv_to_m is None and flat_indices is not None:
        uv_to_m = build_uv_to_m_lookup(
            nv=nv,
            u_half=u_half,
            flat_indices=flat_indices,
            device=target_device,
        )
        if m_active is None:
            m_active = int(len(flat_indices))

    if return_sparse and uv_to_m is None:
        raise ValueError("return_sparse=True requires either uv_to_m or flat_indices.")

    if uv_to_m is None:
        return None, 0

    if target_device == "cuda":
        arr = cp.ascontiguousarray(cp.asarray(uv_to_m), dtype=cp.int32)
    else:
        arr = np.ascontiguousarray(
            cp.asnumpy(uv_to_m) if (cp is not None and isinstance(uv_to_m, cp.ndarray)) else uv_to_m,
            dtype=np.int32,
        )

    if arr.shape != (nv, u_half):
        raise ValueError(f"uv_to_m must have shape ({nv}, {u_half}); got {arr.shape}.")

    if m_active is None:
        if arr.size == 0:
            m_val = 0
        else:
            max_id = int(arr.max().item() if target_device == "cuda" else np.max(arr))
            m_val = max(0, max_id + 1)
    else:
        m_val = int(m_active)
        if m_val < 0:
            raise ValueError(f"m_active must be >= 0; got {m_val}.")

    return arr, m_val


# =============================================================================
# Raw Row Staging & Correlation Collapse (CPU & GPU)
# =============================================================================

def stage_ms_rows_cpu(
    bl_idx: np.ndarray,
    t_rel: np.ndarray,
    uvw_rows: np.ndarray,
    data_rows: np.ndarray,
    wgt_rows: np.ndarray,
    flag_rows: np.ndarray,
    *,
    flag_row: Optional[np.ndarray] = None,
    nbl: Optional[int] = None,
    nt_stage: Optional[int] = None,
    t_offset: int = 0,
    vis_3d: Optional[np.ndarray] = None,
    wgt_3d: Optional[np.ndarray] = None,
    uvw_3d: Optional[np.ndarray] = None,
    corr_mode: str = "average",
    basis: str = "auto",
    single_pol: str = "XX",
    average_correlations: bool = True,
    corr_index: Optional[int] = None,
    labels: Optional[List[str]] = None,
    lbl2idx: Optional[Dict[str, int]] = None,
    min_uv_distance_m: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stage MS row buffers on CPU into ``(nbl, nchan, nt_stage)`` visibility/weight
    tensors and ``(nbl, 3, nt_stage)`` baseline UVW tensors.

    Parameters
    ----------
    bl_idx : np.ndarray (int32), shape (nrow,)
        Baseline index per MS row in ``[0, nbl)``.
    t_rel : np.ndarray (int32), shape (nrow,)
        Relative time index per MS row; written at ``t_offset + t_rel[r]``.
    uvw_rows : np.ndarray (float64), shape (nrow, 3)
        Baseline coordinates ``(u, v, w)`` in metres.
    data_rows : np.ndarray (complex64), shape (nrow, nchan, ncorr) or (nrow, nchan)
        Complex visibility data.
    wgt_rows : np.ndarray (float32), shape (nrow, nchan, ncorr), (nrow, ncorr), or (nrow, nchan)
        Channel/row weights.
    flag_rows : np.ndarray (bool or uint8), shape (nrow, nchan, ncorr) or (nrow, nchan)
        Boolean correlation flags (True/non-zero means flagged).
    flag_row : np.ndarray (bool or uint8), shape (nrow,), optional
        Row-level flag mask.
    nbl : int, optional
        Number of baselines. Inferred from ``max(bl_idx) + 1`` or ``vis_3d.shape[0]`` if omitted.
    nt_stage : int, optional
        Time length of staging tensors. Inferred from ``t_offset + max(t_rel) + 1`` or ``vis_3d.shape[2]``.
    t_offset : int
        Starting time offset inside the staging tensors.
    vis_3d, wgt_3d, uvw_3d : np.ndarray, optional
        Optional pre-allocated staging buffers of shapes ``(nbl, nchan, nt_stage)`` complex64,
        ``(nbl, nchan, nt_stage)`` float32, and ``(nbl, 3, nt_stage)`` float64.
    corr_mode, basis, single_pol, average_correlations, corr_index, labels, lbl2idx :
        Correlation collapse controls matching ``fastducc.imaging._collapse_correlations``.
    min_uv_distance_m : float, optional
        Minimum baseline UV distance in metres; shorter baselines are zero-weighted.

    Returns
    -------
    vis_3d : np.ndarray (complex64), shape (nbl, nchan, nt_stage)
    wgt_3d : np.ndarray (float32), shape (nbl, nchan, nt_stage)
    uvw_3d : np.ndarray (float64), shape (nbl, 3, nt_stage)
    """
    bl_arr = np.asarray(bl_idx, dtype=np.int32)
    tr_arr = np.asarray(t_rel, dtype=np.int32)
    uvw_arr = np.asarray(uvw_rows, dtype=np.float64)
    data_arr = np.asarray(data_rows, dtype=np.complex64)
    wgt_arr = np.asarray(wgt_rows, dtype=np.float32)
    flag_arr = np.asarray(flag_rows, dtype=bool)

    if data_arr.ndim == 2:
        data_arr = data_arr[:, :, None]
        flag_arr = flag_arr[:, :, None]
        if wgt_arr.ndim == 2:
            wgt_arr = wgt_arr[:, :, None]
        elif wgt_arr.ndim == 1:
            wgt_arr = wgt_arr[:, None]

    if data_arr.ndim != 3:
        raise ValueError(f"data_rows must be 2D or 3D; got shape {data_arr.shape}.")

    nrow, nchan, ncorr = data_arr.shape
    if wgt_arr.ndim == 2:
        wgt_arr = np.broadcast_to(wgt_arr[:, None, :], (nrow, nchan, ncorr))
    elif wgt_arr.shape != (nrow, nchan, ncorr):
        raise ValueError(f"wgt_rows shape {wgt_arr.shape} incompatible with data_rows {data_arr.shape}.")
    if flag_arr.shape != (nrow, nchan, ncorr):
        raise ValueError(f"flag_rows shape {flag_arr.shape} must match data_rows {data_arr.shape}.")

    eff_nbl = (
        int(vis_3d.shape[0])
        if vis_3d is not None
        else (int(nbl) if nbl is not None else (int(np.max(bl_arr)) + 1 if nrow > 0 else 0))
    )
    eff_nt = (
        int(vis_3d.shape[2])
        if vis_3d is not None
        else (int(nt_stage) if nt_stage is not None else (int(t_offset) + int(np.max(tr_arr)) + 1 if nrow > 0 else int(t_offset)))
    )

    if vis_3d is None:
        vis_3d = np.zeros((eff_nbl, nchan, eff_nt), dtype=np.complex64)
    else:
        if vis_3d.shape != (eff_nbl, nchan, eff_nt) or vis_3d.dtype != np.complex64 or not vis_3d.flags["C_CONTIGUOUS"]:
            raise ValueError(
                f"Pre-allocated vis_3d must be C-contiguous complex64 of shape ({eff_nbl}, {nchan}, {eff_nt}); "
                f"got shape={vis_3d.shape}, dtype={vis_3d.dtype}."
            )
    if wgt_3d is None:
        wgt_3d = np.zeros((eff_nbl, nchan, eff_nt), dtype=np.float32)
    else:
        if wgt_3d.shape != (eff_nbl, nchan, eff_nt) or wgt_3d.dtype != np.float32 or not wgt_3d.flags["C_CONTIGUOUS"]:
            raise ValueError(
                f"Pre-allocated wgt_3d must be C-contiguous float32 of shape ({eff_nbl}, {nchan}, {eff_nt}); "
                f"got shape={wgt_3d.shape}, dtype={wgt_3d.dtype}."
            )
    if uvw_3d is None:
        uvw_3d = np.zeros((eff_nbl, 3, eff_nt), dtype=np.float64)
    else:
        if uvw_3d.shape != (eff_nbl, 3, eff_nt) or uvw_3d.dtype != np.float64 or not uvw_3d.flags["C_CONTIGUOUS"]:
            raise ValueError(
                f"Pre-allocated uvw_3d must be C-contiguous float64 of shape ({eff_nbl}, 3, {eff_nt}); "
                f"got shape={uvw_3d.shape}, dtype={uvw_3d.dtype}."
            )

    if nrow == 0 or eff_nbl == 0 or nchan == 0 or eff_nt == 0:
        return vis_3d, wgt_3d, uvw_3d

    t_dst = int(t_offset) + tr_arr
    in_bounds = (bl_arr >= 0) & (bl_arr < eff_nbl) & (t_dst >= 0) & (t_dst < eff_nt)
    if not np.all(in_bounds):
        bl_arr = bl_arr[in_bounds]
        t_dst = t_dst[in_bounds]
        uvw_arr = uvw_arr[in_bounds]
        data_arr = data_arr[in_bounds]
        wgt_arr = wgt_arr[in_bounds]
        flag_arr = flag_arr[in_bounds]
        if flag_row is not None:
            flag_row = np.asarray(flag_row, dtype=bool)[in_bounds]
    elif flag_row is not None:
        flag_row = np.asarray(flag_row, dtype=bool)

    if bl_arr.size == 0:
        return vis_3d, wgt_3d, uvw_3d

    good = (~flag_arr) & (wgt_arr > 0.0) & np.isfinite(wgt_arr) & np.isfinite(data_arr.real) & np.isfinite(data_arr.imag)
    if flag_row is not None:
        good &= (~flag_row[:, None, None])

    min_uv = 0.0 if min_uv_distance_m is None else float(min_uv_distance_m)
    if min_uv > 0.0:
        u_m = uvw_arr[:, 0]
        v_m = uvw_arr[:, 1]
        uv_ok = np.isfinite(u_m) & np.isfinite(v_m) & (np.hypot(u_m, v_m) >= min_uv)
        good &= uv_ok[:, None, None]

    clean_data = np.where(good, data_arr, np.complex64(0.0 + 0.0j))
    clean_wgt = np.where(good, wgt_arr, np.float32(0.0))

    mode_id, i0, i1 = _resolve_corr_spec(
        ncorr=ncorr,
        corr_mode=corr_mode,
        basis=basis,
        single_pol=single_pol,
        average_correlations=average_correlations,
        corr_index=corr_index,
        labels=labels,
        lbl2idx=lbl2idx,
    )

    if mode_id == 0:
        vis_flat = clean_data[:, :, i0]
        wgt_flat = clean_wgt[:, :, i0]
    elif mode_id == 1:
        wsum = clean_wgt.sum(axis=2, dtype=np.float64)
        vwsum = (clean_data.astype(np.complex128) * clean_wgt.astype(np.float64)).sum(axis=2)
        with np.errstate(invalid="ignore", divide="ignore"):
            vis_flat = np.where(wsum > 0.0, vwsum / np.where(wsum > 0.0, wsum, 1.0), 0.0).astype(np.complex64)
        wgt_flat = wsum.astype(np.float32)
    else:
        v1 = clean_data[:, :, i0]
        v2 = clean_data[:, :, i1]
        w1 = clean_wgt[:, :, i0]
        w2 = clean_wgt[:, :, i1]
        p1 = w1 > 0.0
        p2 = w2 > 0.0
        both = p1 & p2
        only1 = p1 & (~p2)
        only2 = (~p1) & p2
        vis_flat = np.zeros((clean_data.shape[0], nchan), dtype=np.complex64)
        wgt_flat = np.zeros((clean_data.shape[0], nchan), dtype=np.float32)
        if np.any(both):
            vis_flat[both] = (0.5 * (v1[both] + v2[both])).astype(np.complex64)
            inv_sum = 1.0 / w1[both].astype(np.float64) + 1.0 / w2[both].astype(np.float64)
            wgt_flat[both] = np.where(inv_sum > 0.0, 4.0 / inv_sum, 0.0).astype(np.float32)
            bad_both = both & (wgt_flat <= 0.0)
            if np.any(bad_both):
                vis_flat[bad_both] = 0.0
        if np.any(only1):
            vis_flat[only1] = v1[only1]
            wgt_flat[only1] = w1[only1]
        if np.any(only2):
            vis_flat[only2] = v2[only2]
            wgt_flat[only2] = w2[only2]

    vis_3d[bl_arr, :, t_dst] = vis_flat
    wgt_3d[bl_arr, :, t_dst] = wgt_flat
    uvw_3d[bl_arr, :, t_dst] = uvw_arr
    return vis_3d, wgt_3d, uvw_3d


def stage_ms_rows_gpu(
    bl_idx: Any,
    t_rel: Any,
    uvw_rows: Any,
    data_rows: Any,
    wgt_rows: Any,
    flag_rows: Any,
    *,
    flag_row: Optional[Any] = None,
    nbl: Optional[int] = None,
    nt_stage: Optional[int] = None,
    t_offset: int = 0,
    vis_3d: Optional[Any] = None,
    wgt_3d: Optional[Any] = None,
    uvw_3d: Optional[Any] = None,
    corr_mode: str = "average",
    basis: str = "auto",
    single_pol: str = "XX",
    average_correlations: bool = True,
    corr_index: Optional[int] = None,
    labels: Optional[List[str]] = None,
    lbl2idx: Optional[Dict[str, int]] = None,
    min_uv_distance_m: Optional[float] = None,
    return_host: bool = False,
) -> Tuple[Any, Any, Any]:
    """
    Stage streamed MS row buffers on CUDA via ``stage_ms_rows_kernel``.

    Inputs may reside on host (NumPy) or device (CuPy). Returns staged tensors on
    GPU by default (``return_host=False``) for zero-copy chaining into
    :func:`dedisperse_and_rasterise_chunk_gpu`.
    """
    _require_cuda()
    d_bl = cp.ascontiguousarray(cp.asarray(bl_idx), dtype=cp.int32)
    d_tr = cp.ascontiguousarray(cp.asarray(t_rel), dtype=cp.int32)
    d_uvw = cp.ascontiguousarray(cp.asarray(uvw_rows), dtype=cp.float64)
    d_data = cp.ascontiguousarray(cp.asarray(data_rows), dtype=cp.complex64)
    d_wgt = cp.ascontiguousarray(cp.asarray(wgt_rows), dtype=cp.float32)
    d_flag = cp.ascontiguousarray(cp.asarray(flag_rows), dtype=cp.uint8)

    if d_data.ndim == 2:
        d_data = d_data[:, :, None]
        d_flag = d_flag[:, :, None]
        if d_wgt.ndim == 2:
            d_wgt = d_wgt[:, :, None]
        elif d_wgt.ndim == 1:
            d_wgt = d_wgt[:, None]

    if d_data.ndim != 3:
        raise ValueError(f"data_rows must be 2D or 3D; got shape {d_data.shape}.")

    nrow, nchan, ncorr = (int(d_data.shape[0]), int(d_data.shape[1]), int(d_data.shape[2]))
    if d_wgt.ndim == 2 and d_wgt.shape == (nrow, ncorr):
        wgt_is_spectrum = 0
    elif d_wgt.ndim == 3 and d_wgt.shape == (nrow, nchan, ncorr):
        wgt_is_spectrum = 1
    else:
        raise ValueError(f"wgt_rows shape {d_wgt.shape} incompatible with data_rows {d_data.shape}.")

    if d_flag.shape != (nrow, nchan, ncorr):
        raise ValueError(f"flag_rows shape {d_flag.shape} must match data_rows {d_data.shape}.")
    if d_uvw.shape != (nrow, 3):
        raise ValueError(f"uvw_rows must have shape ({nrow}, 3); got {d_uvw.shape}.")
    if d_bl.shape != (nrow,) or d_tr.shape != (nrow,):
        raise ValueError("bl_idx and t_rel must be 1D arrays of length nrow.")

    d_flag_row_ptr: Any = 0
    if flag_row is not None:
        d_flag_row = cp.ascontiguousarray(cp.asarray(flag_row), dtype=cp.uint8)
        if d_flag_row.shape != (nrow,):
            raise ValueError(f"flag_row must have shape ({nrow},); got {d_flag_row.shape}.")
        d_flag_row_ptr = d_flag_row

    eff_nbl = (
        int(vis_3d.shape[0])
        if vis_3d is not None
        else (int(nbl) if nbl is not None else (int(cp.max(d_bl).item()) + 1 if nrow > 0 else 0))
    )
    eff_nt = (
        int(vis_3d.shape[2])
        if vis_3d is not None
        else (int(nt_stage) if nt_stage is not None else (int(t_offset) + int(cp.max(d_tr).item()) + 1 if nrow > 0 else int(t_offset)))
    )

    if vis_3d is None:
        d_vis_3d = cp.zeros((eff_nbl, nchan, eff_nt), dtype=cp.complex64)
    else:
        if not isinstance(vis_3d, cp.ndarray) or vis_3d.shape != (eff_nbl, nchan, eff_nt) or vis_3d.dtype != cp.complex64 or not vis_3d.flags.c_contiguous:
            raise ValueError(
                f"Pre-allocated vis_3d must be a C-contiguous cupy.complex64 array of shape ({eff_nbl}, {nchan}, {eff_nt})."
            )
        d_vis_3d = vis_3d

    if wgt_3d is None:
        d_wgt_3d = cp.zeros((eff_nbl, nchan, eff_nt), dtype=cp.float32)
    else:
        if not isinstance(wgt_3d, cp.ndarray) or wgt_3d.shape != (eff_nbl, nchan, eff_nt) or wgt_3d.dtype != cp.float32 or not wgt_3d.flags.c_contiguous:
            raise ValueError(
                f"Pre-allocated wgt_3d must be a C-contiguous cupy.float32 array of shape ({eff_nbl}, {nchan}, {eff_nt})."
            )
        d_wgt_3d = wgt_3d

    if uvw_3d is None:
        d_uvw_3d = cp.zeros((eff_nbl, 3, eff_nt), dtype=cp.float64)
    else:
        if not isinstance(uvw_3d, cp.ndarray) or uvw_3d.shape != (eff_nbl, 3, eff_nt) or uvw_3d.dtype != cp.float64 or not uvw_3d.flags.c_contiguous:
            raise ValueError(
                f"Pre-allocated uvw_3d must be a C-contiguous cupy.float64 array of shape ({eff_nbl}, 3, {eff_nt})."
            )
        d_uvw_3d = uvw_3d

    if nrow > 0 and eff_nbl > 0 and nchan > 0 and eff_nt > 0:
        mode_id, i0, i1 = _resolve_corr_spec(
            ncorr=ncorr,
            corr_mode=corr_mode,
            basis=basis,
            single_pol=single_pol,
            average_correlations=average_correlations,
            corr_index=corr_index,
            labels=labels,
            lbl2idx=lbl2idx,
        )
        min_uv = 0.0 if min_uv_distance_m is None else float(min_uv_distance_m)
        mod = _get_module()
        kernel = mod.get_function("stage_ms_rows_kernel")
        gx = _ceil_div(nrow, 8)
        gy = min(_MAX_GRID_Y, _ceil_div(nchan, 32))
        kernel(
            (gx, gy, 1),
            (32, 8, 1),
            (
                d_bl,
                d_tr,
                d_uvw,
                d_data,
                d_wgt,
                d_flag,
                d_flag_row_ptr,
                d_vis_3d,
                d_wgt_3d,
                d_uvw_3d,
                np.int32(nrow),
                np.int32(eff_nbl),
                np.int32(nchan),
                np.int32(ncorr),
                np.int32(eff_nt),
                np.int32(t_offset),
                np.int32(wgt_is_spectrum),
                np.int32(mode_id),
                np.int32(i0),
                np.int32(i1),
                np.float64(min_uv),
            ),
        )

    if return_host:
        return cp.asnumpy(d_vis_3d), cp.asnumpy(d_wgt_3d), cp.asnumpy(d_uvw_3d)
    return d_vis_3d, d_wgt_3d, d_uvw_3d


def stage_ms_rows(
    bl_idx: Any,
    t_rel: Any,
    uvw_rows: Any,
    data_rows: Any,
    wgt_rows: Any,
    flag_rows: Any,
    *,
    device: str = "auto",
    return_host: bool = False,
    **kwargs: Any,
) -> Tuple[Any, Any, Any]:
    """
    Dispatch MS row staging and correlation collapse to CUDA or CPU.

    Parameters
    ----------
    device : {'auto', 'cuda', 'cpu'}
        Target execution backend.
    return_host : bool
        When running on CUDA, copy staged tensors back to host NumPy arrays if True.
    **kwargs :
        Forwarded to :func:`stage_ms_rows_gpu` or :func:`stage_ms_rows_cpu`.
    """
    eff_dev = resolve_device(device)
    if eff_dev == "cuda":
        return stage_ms_rows_gpu(
            bl_idx, t_rel, uvw_rows, data_rows, wgt_rows, flag_rows,
            return_host=return_host, **kwargs
        )
    return stage_ms_rows_cpu(
        bl_idx, t_rel, uvw_rows, data_rows, wgt_rows, flag_rows, **kwargs
    )


# =============================================================================
# CPU Dedispersion + Rasterisation Fallbacks (Numba & Pure NumPy)
# =============================================================================

if HAS_NUMBA:
    @nb.njit(inline="always", fastmath=False)
    def _accum_cell_numba(
        t: int,
        nt_out: int,
        nv: int,
        u_half: int,
        cu: int,
        cv: int,
        r_f32: np.float32,
        i_f32: np.float32,
        w_f32: np.float32,
        uv_to_m: np.ndarray,
        has_uv_to_m: bool,
        sparse_output: bool,
        out_re: np.ndarray,
        out_im: np.ndarray,
        out_w: np.ndarray,
    ) -> None:
        if has_uv_to_m:
            m_idx = int(uv_to_m[cv, cu])
            if m_idx < 0:
                return
            if sparse_output:
                idx = m_idx * nt_out + t
            else:
                idx = (t * nv + cv) * u_half + cu
        else:
            idx = (t * nv + cv) * u_half + cu
        out_re[idx] += r_f32
        out_im[idx] += i_f32
        out_w[idx] += w_f32

    @nb.njit(inline="always", fastmath=False)
    def _splat_sample_numba(
        t: int,
        nt_out: int,
        nv: int,
        u_half: int,
        u_f: float,
        v_f: float,
        r_f32: np.float32,
        i_f32: np.float32,
        w_f32: np.float32,
        mode_id: int,
        uv_to_m: np.ndarray,
        has_uv_to_m: bool,
        sparse_output: bool,
        out_re: np.ndarray,
        out_im: np.ndarray,
        out_w: np.ndarray,
    ) -> None:
        if mode_id == 0:
            u_cell = np.int64(np.rint(u_f))
            v_cell = np.int64(np.rint(v_f))
            if u_cell < 0:
                u_cell = -u_cell
                v_cell = -v_cell
                i_f32 = -i_f32
            if 0 <= u_cell < u_half:
                v_idx = ((v_cell % nv) + nv) % nv
                _accum_cell_numba(
                    t, nt_out, nv, u_half, int(u_cell), int(v_idx),
                    r_f32, i_f32, w_f32, uv_to_m, has_uv_to_m, sparse_output,
                    out_re, out_im, out_w,
                )
            return

        if u_f < 0.0:
            u_f = -u_f
            v_f = -v_f
            i_f32 = -i_f32
        u0 = np.int64(math.floor(u_f))
        v0 = np.int64(math.floor(v_f))
        if 0 <= u0 < u_half:
            au = np.float32(u_f - float(u0))
            av = np.float32(v_f - float(v0))
            c00 = np.float32((np.float32(1.0) - au) * (np.float32(1.0) - av))
            c10 = np.float32(au * (np.float32(1.0) - av))
            c01 = np.float32((np.float32(1.0) - au) * av)
            c11 = np.float32(au * av)
            v0w = ((v0 % nv) + nv) % nv
            v1w = v0w + 1 if v0w + 1 < nv else 0
            for du_off in range(2):
                cu = u0 + du_off
                if cu >= u_half:
                    continue
                for dv_off in range(2):
                    cv = v0w if dv_off == 0 else v1w
                    cw = (c00 if dv_off == 0 else c01) if du_off == 0 else (c10 if dv_off == 0 else c11)
                    if cw <= np.float32(0.0):
                        continue
                    _accum_cell_numba(
                        t, nt_out, nv, u_half, int(cu), int(cv),
                        np.float32(r_f32 * cw), np.float32(i_f32 * cw), np.float32(w_f32 * cw),
                        uv_to_m, has_uv_to_m, sparse_output,
                        out_re, out_im, out_w,
                    )

    @nb.njit(parallel=True, fastmath=False)
    def _rasterise_numba_kernel(
        vis_3d: np.ndarray,
        wgt_3d: np.ndarray,
        uvw_3d: np.ndarray,
        delays: np.ndarray,
        sub_offsets: np.ndarray,
        sub_freqs_hz: np.ndarray,
        sub_delays: np.ndarray,
        uv_to_m: np.ndarray,
        has_uv_to_m: bool,
        sparse_output: bool,
        m_active: int,
        nt_out: int,
        nv: int,
        u_half: int,
        du: float,
        dv: float,
        min_uv_distance_m: float,
        min_valid_channels: int,
        collapse_channels: bool,
        exact_uvw: bool,
        mode_id: int,
        uv_threshold: float,
        out_re: np.ndarray,
        out_im: np.ndarray,
        out_w: np.ndarray,
    ) -> None:
        nbl, nchan, nt_in = vis_3d.shape
        nsub_eff = len(sub_freqs_hz)
        req_chans = min(min_valid_channels, nchan) if min_valid_channels > 0 else 1
        thr_sq = np.float32(uv_threshold * uv_threshold) if uv_threshold > 0.0 else np.float32(0.0)
        is_collapsed = collapse_channels and (nsub_eff == 1)
        mean_freq_over_c = sub_freqs_hz[0] / C_LIGHT
        spatial = nv * u_half

        for t_u in nb.prange(nt_out):
            t = np.int64(t_u)
            # 1. Track raw and dedispersed valid channels for snapshot t
            raw_cnt = 0
            dedisp_cnt = 0
            for c in range(nchan):
                d = int(delays[c])
                t_src = t + d

                raw_ok = False
                for bl in range(nbl):
                    w = wgt_3d[bl, c, t]
                    if w > 0.0 and math.isfinite(w):
                        vr = vis_3d[bl, c, t].real
                        vi = vis_3d[bl, c, t].imag
                        if math.isfinite(vr) and math.isfinite(vi):
                            if min_uv_distance_m <= 0.0:
                                raw_ok = True
                                break
                            um = uvw_3d[bl, 0, t]
                            vm = uvw_3d[bl, 1, t]
                            if math.isfinite(um) and math.isfinite(vm) and math.hypot(um, vm) >= min_uv_distance_m:
                                raw_ok = True
                                break
                if raw_ok:
                    raw_cnt += 1

                dedisp_ok = False
                if 0 <= t_src < nt_in:
                    if d == 0:
                        dedisp_ok = raw_ok
                    else:
                        for bl in range(nbl):
                            w = wgt_3d[bl, c, t_src]
                            if w > 0.0 and math.isfinite(w):
                                vr = vis_3d[bl, c, t_src].real
                                vi = vis_3d[bl, c, t_src].imag
                                if math.isfinite(vr) and math.isfinite(vi):
                                    if min_uv_distance_m <= 0.0:
                                        dedisp_ok = True
                                        break
                                    um = uvw_3d[bl, 0, t_src]
                                    vm = uvw_3d[bl, 1, t_src]
                                    if math.isfinite(um) and math.isfinite(vm) and math.hypot(um, vm) >= min_uv_distance_m:
                                        dedisp_ok = True
                                        break
                if dedisp_ok:
                    dedisp_cnt += 1

            if nchan <= 0:
                continue
            if collapse_channels:
                if raw_cnt < req_chans or dedisp_cnt < 1:
                    continue
            else:
                if dedisp_cnt < req_chans:
                    continue

            # 2. Dedisperse, reduce in subbands, and splat onto half-plane grid
            for bl in range(nbl):
                if is_collapsed:
                    sum_re = 0.0
                    sum_im = 0.0
                    sum_w = 0.0
                    for c in range(nchan):
                        d = int(delays[c])
                        t_src = t + d
                        if t_src < 0 or t_src >= nt_in:
                            continue
                        w = wgt_3d[bl, c, t_src]
                        if w <= 0.0 or not math.isfinite(w):
                            continue
                        vr = vis_3d[bl, c, t_src].real
                        vi = vis_3d[bl, c, t_src].imag
                        if not (math.isfinite(vr) and math.isfinite(vi)):
                            continue
                        if min_uv_distance_m > 0.0:
                            u_s = uvw_3d[bl, 0, t_src]
                            v_s = uvw_3d[bl, 1, t_src]
                            if not (math.isfinite(u_s) and math.isfinite(v_s) and math.hypot(u_s, v_s) >= min_uv_distance_m):
                                continue
                        wd = float(w)
                        sum_re += float(vr) * wd
                        sum_im += float(vi) * wd
                        sum_w += wd

                    if sum_w <= 0.0:
                        continue

                    u_m_t = uvw_3d[bl, 0, t]
                    v_m_t = uvw_3d[bl, 1, t]
                    u_f = (u_m_t * mean_freq_over_c) / du
                    v_f = (v_m_t * mean_freq_over_c) / dv
                    if not (math.isfinite(u_f) and math.isfinite(v_f)) or abs(u_f) > 1.0e15 or abs(v_f) > 1.0e15:
                        continue

                    _splat_sample_numba(
                        t, nt_out, nv, u_half, u_f, v_f,
                        np.float32(sum_re), np.float32(sum_im), np.float32(sum_w),
                        mode_id, uv_to_m, has_uv_to_m, sparse_output,
                        out_re, out_im, out_w,
                    )
                else:
                    for s in range(nsub_eff):
                        c0 = int(sub_offsets[s])
                        c1 = int(sub_offsets[s + 1])
                        sum_re = 0.0
                        sum_im = 0.0
                        sum_w = 0.0
                        for c in range(c0, c1):
                            d = int(delays[c])
                            t_src = t + d
                            if t_src < 0 or t_src >= nt_in:
                                continue
                            w = wgt_3d[bl, c, t_src]
                            if w <= 0.0 or not math.isfinite(w):
                                continue
                            vr = vis_3d[bl, c, t_src].real
                            vi = vis_3d[bl, c, t_src].imag
                            if not (math.isfinite(vr) and math.isfinite(vi)):
                                continue
                            if min_uv_distance_m > 0.0:
                                u_s = uvw_3d[bl, 0, t_src]
                                v_s = uvw_3d[bl, 1, t_src]
                                if not (math.isfinite(u_s) and math.isfinite(v_s) and math.hypot(u_s, v_s) >= min_uv_distance_m):
                                    continue
                            wd = float(w)
                            sum_re += float(vr) * wd
                            sum_im += float(vi) * wd
                            sum_w += wd

                        if sum_w <= 0.0:
                            continue

                        if exact_uvw:
                            ts = t + int(sub_delays[s])
                            t_uvw = 0 if ts < 0 else (nt_in - 1 if ts >= nt_in else ts)
                        else:
                            t_uvw = t

                        u_m = uvw_3d[bl, 0, t_uvw]
                        v_m = uvw_3d[bl, 1, t_uvw]
                        freq_over_c = sub_freqs_hz[s] / C_LIGHT
                        u_f = (u_m * freq_over_c) / du
                        v_f = (v_m * freq_over_c) / dv
                        if not (math.isfinite(u_f) and math.isfinite(v_f)) or abs(u_f) > 1.0e15 or abs(v_f) > 1.0e15:
                            continue

                        _splat_sample_numba(
                            t, nt_out, nv, u_half, u_f, v_f,
                            np.float32(sum_re), np.float32(sum_im), np.float32(sum_w),
                            mode_id, uv_to_m, has_uv_to_m, sparse_output,
                            out_re, out_im, out_w,
                        )

            # 3. Normalise and threshold snapshot t
            if sparse_output:
                for m_idx in range(m_active):
                    idx = m_idx * nt_out + t
                    w = out_w[idx]
                    if w > 0.0 and math.isfinite(w):
                        inv_w = np.float32(1.0) / w
                        rx = np.float32(out_re[idx] * inv_w)
                        ry = np.float32(out_im[idx] * inv_w)
                        if math.isfinite(rx) and math.isfinite(ry) and (thr_sq <= 0.0 or (rx * rx + ry * ry) > thr_sq):
                            out_re[idx] = rx
                            out_im[idx] = ry
                        else:
                            out_re[idx] = np.float32(0.0)
                            out_im[idx] = np.float32(0.0)
                    else:
                        out_re[idx] = np.float32(0.0)
                        out_im[idx] = np.float32(0.0)
            else:
                base_t = t * spatial
                for off in range(spatial):
                    idx = base_t + off
                    w = out_w[idx]
                    if w > 0.0 and math.isfinite(w):
                        inv_w = np.float32(1.0) / w
                        rx = np.float32(out_re[idx] * inv_w)
                        ry = np.float32(out_im[idx] * inv_w)
                        if math.isfinite(rx) and math.isfinite(ry) and (thr_sq <= 0.0 or (rx * rx + ry * ry) > thr_sq):
                            out_re[idx] = rx
                            out_im[idx] = ry
                        else:
                            out_re[idx] = np.float32(0.0)
                            out_im[idx] = np.float32(0.0)
                    else:
                        out_re[idx] = np.float32(0.0)
                        out_im[idx] = np.float32(0.0)


def _dedisperse_and_rasterise_numpy(
    vis_3d: np.ndarray,
    wgt_3d: np.ndarray,
    uvw_3d: np.ndarray,
    delays: np.ndarray,
    sub_offsets: np.ndarray,
    sub_freqs_hz: np.ndarray,
    sub_delays: np.ndarray,
    uv_to_m: Optional[np.ndarray],
    m_active: int,
    sparse_output: bool,
    nt_out: int,
    nv: int,
    u_half: int,
    du: float,
    dv: float,
    min_uv_distance_m: float,
    min_valid_channels: int,
    collapse_channels: bool,
    exact_uvw: bool,
    mode_id: int,
    uv_threshold: float,
) -> np.ndarray:
    """Pure vectorised NumPy CPU implementation used when Numba is unavailable."""
    nbl, nchan, nt_in = vis_3d.shape
    nsub_eff = int(len(sub_freqs_hz))
    out_shape = (m_active, nt_out) if sparse_output else (nt_out, nv, u_half)
    if nt_out == 0 or nbl == 0 or nchan == 0 or (sparse_output and m_active == 0):
        return np.zeros(out_shape, dtype=np.complex64)

    valid_w = (wgt_3d > 0.0) & np.isfinite(wgt_3d) & np.isfinite(vis_3d.real) & np.isfinite(vis_3d.imag)
    if min_uv_distance_m > 0.0:
        uv_ok = (
            np.isfinite(uvw_3d[:, 0, :])
            & np.isfinite(uvw_3d[:, 1, :])
            & (np.hypot(uvw_3d[:, 0, :], uvw_3d[:, 1, :]) >= min_uv_distance_m)
        )
        valid_w &= uv_ok[:, None, :]

    wgt_eff = np.where(valid_w, wgt_3d, np.float32(0.0))
    vis_eff = np.where(valid_w, vis_3d, np.complex64(0.0 + 0.0j))

    raw_valid_chans = np.count_nonzero(np.any(wgt_eff[:, :, :nt_out] > 0.0, axis=0), axis=0)
    req_chans = min(min_valid_channels, nchan) if min_valid_channels > 0 else 1

    # Dedisperse visibility and weight streams onto [0, nt_out)
    t_grid = np.arange(nt_out, dtype=np.int64)
    vis_dedisp = np.zeros((nbl, nchan, nt_out), dtype=np.complex64)
    wgt_dedisp = np.zeros((nbl, nchan, nt_out), dtype=np.float32)
    for c in range(nchan):
        t_src = t_grid + int(delays[c])
        ok_t = (t_src >= 0) & (t_src < nt_in)
        if np.any(ok_t):
            vis_dedisp[:, c, ok_t] = vis_eff[:, c, t_src[ok_t]]
            wgt_dedisp[:, c, ok_t] = wgt_eff[:, c, t_src[ok_t]]

    dedisp_valid_chans = np.count_nonzero(np.any(wgt_dedisp > 0.0, axis=0), axis=0)
    if collapse_channels:
        snap_valid = (raw_valid_chans >= req_chans) & (dedisp_valid_chans >= 1)
    else:
        snap_valid = dedisp_valid_chans >= req_chans

    if not np.any(snap_valid):
        return np.zeros(out_shape, dtype=np.complex64)

    # Subband reduction
    is_collapsed = collapse_channels and (nsub_eff == 1)
    vw_sub = np.empty((nbl, nsub_eff, nt_out), dtype=np.complex64)
    w_sub = np.empty((nbl, nsub_eff, nt_out), dtype=np.float32)
    u_m_sub = np.empty((nbl, nsub_eff, nt_out), dtype=np.float64)
    v_m_sub = np.empty((nbl, nsub_eff, nt_out), dtype=np.float64)

    for s in range(nsub_eff):
        c0 = int(sub_offsets[s])
        c1 = int(sub_offsets[s + 1])
        w_slice = wgt_dedisp[:, c0:c1, :].astype(np.float64)
        v_slice = vis_dedisp[:, c0:c1, :].astype(np.complex128)
        w_sub[:, s, :] = np.sum(w_slice, axis=1).astype(np.float32)
        vw_sub[:, s, :] = np.sum(v_slice * w_slice, axis=1).astype(np.complex64)

        if (not is_collapsed) and exact_uvw:
            ts = np.clip(t_grid + int(sub_delays[s]), 0, nt_in - 1)
            u_m_sub[:, s, :] = uvw_3d[:, 0, ts]
            v_m_sub[:, s, :] = uvw_3d[:, 1, ts]
        else:
            u_m_sub[:, s, :] = uvw_3d[:, 0, :nt_out]
            v_m_sub[:, s, :] = uvw_3d[:, 1, :nt_out]

    freq_over_c = sub_freqs_hz[None, :, None] / C_LIGHT
    u_f = (u_m_sub * freq_over_c) / du
    v_f = (v_m_sub * freq_over_c) / dv

    valid = (
        (w_sub > 0.0)
        & snap_valid[None, None, :]
        & np.isfinite(u_f)
        & np.isfinite(v_f)
        & (np.abs(u_f) <= 1.0e15)
        & (np.abs(v_f) <= 1.0e15)
    )
    if not np.any(valid):
        return np.zeros(out_shape, dtype=np.complex64)

    t_idx_grid = np.broadcast_to(t_grid[None, None, :], (nbl, nsub_eff, nt_out))
    t_sel = t_idx_grid[valid]
    u_sel = u_f[valid]
    v_sel = v_f[valid]
    re_sel = vw_sub.real[valid]
    im_sel = vw_sub.imag[valid]
    w_sel = w_sub[valid]

    n_bins = int(np.prod(out_shape))
    accum_re = np.zeros(n_bins, dtype=np.float32)
    accum_im = np.zeros(n_bins, dtype=np.float32)
    accum_w = np.zeros(n_bins, dtype=np.float32)

    def _accumulate_bins(cu: np.ndarray, cv: np.ndarray, r_arr: np.ndarray, i_arr: np.ndarray, w_arr: np.ndarray, t_arr: np.ndarray) -> None:
        mask = (cu >= 0) & (cu < u_half) & (w_arr > 0.0)
        if not np.any(mask):
            return
        cu = cu[mask]
        cv = cv[mask]
        r_arr = r_arr[mask]
        i_arr = i_arr[mask]
        w_arr = w_arr[mask]
        t_arr = t_arr[mask]

        if uv_to_m is not None:
            m_arr = uv_to_m[cv, cu]
            m_ok = m_arr >= 0
            if not np.any(m_ok):
                return
            cu = cu[m_ok]
            cv = cv[m_ok]
            m_arr = m_arr[m_ok]
            r_arr = r_arr[m_ok]
            i_arr = i_arr[m_ok]
            w_arr = w_arr[m_ok]
            t_arr = t_arr[m_ok]
            if sparse_output:
                flat_bin = m_arr.astype(np.int64) * nt_out + t_arr
            else:
                flat_bin = (t_arr * nv + cv) * u_half + cu
        else:
            flat_bin = (t_arr * nv + cv) * u_half + cu

        accum_re[:] += np.bincount(flat_bin, weights=r_arr, minlength=n_bins).astype(np.float32)
        accum_im[:] += np.bincount(flat_bin, weights=i_arr, minlength=n_bins).astype(np.float32)
        accum_w[:] += np.bincount(flat_bin, weights=w_arr, minlength=n_bins).astype(np.float32)

    if mode_id == 0:
        u_cell = np.round(u_sel).astype(np.int64)
        v_cell = np.round(v_sel).astype(np.int64)
        flip = u_cell < 0
        u_cell = np.where(flip, -u_cell, u_cell)
        v_cell = np.where(flip, -v_cell, v_cell)
        im_use = np.where(flip, -im_sel, im_sel)
        v_idx = np.mod(v_cell, nv)
        _accumulate_bins(u_cell, v_idx, re_sel, im_use, w_sel, t_sel)
    else:
        flip = u_sel < 0.0
        uf = np.where(flip, -u_sel, u_sel)
        vf = np.where(flip, -v_sel, v_sel)
        im_use = np.where(flip, -im_sel, im_sel)

        u0 = np.floor(uf).astype(np.int64)
        v0 = np.floor(vf).astype(np.int64)
        au = (uf - u0.astype(np.float64)).astype(np.float32)
        av = (vf - v0.astype(np.float64)).astype(np.float32)

        v0w = np.mod(v0, nv)
        v1w = np.where(v0w + 1 < nv, v0w + 1, 0)
        corners = (
            (u0, v0w, (np.float32(1.0) - au) * (np.float32(1.0) - av)),
            (u0 + 1, v0w, au * (np.float32(1.0) - av)),
            (u0, v1w, (np.float32(1.0) - au) * av),
            (u0 + 1, v1w, au * av),
        )
        for cu, cv, cw in corners:
            _accumulate_bins(cu, cv, re_sel * cw, im_use * cw, w_sel * cw, t_sel)

    out_re = accum_re.reshape(out_shape)
    out_im = accum_im.reshape(out_shape)
    out_w = accum_w.reshape(out_shape)

    pos_w = (out_w > 0.0) & np.isfinite(out_w)
    inv_w = np.where(pos_w, np.float32(1.0) / np.where(pos_w, out_w, np.float32(1.0)), np.float32(0.0))
    norm_re = out_re * inv_w
    norm_im = out_im * inv_w
    valid_out = pos_w & np.isfinite(norm_re) & np.isfinite(norm_im)
    if uv_threshold > 0.0:
        thr_sq = np.float32(uv_threshold * uv_threshold)
        valid_out &= (norm_re * norm_re + norm_im * norm_im) > thr_sq

    out = np.zeros(out_shape, dtype=np.complex64)
    out.real[valid_out] = norm_re[valid_out]
    out.imag[valid_out] = norm_im[valid_out]
    return out


def dedisperse_and_rasterise_chunk_cpu(
    vis_3d: np.ndarray,
    wgt_3d: np.ndarray,
    uvw_3d: np.ndarray,
    freqs_hz: np.ndarray,
    delays: Optional[np.ndarray] = None,
    *,
    nt_out: Optional[int] = None,
    dm: float = 0.0,
    dt: Optional[float] = None,
    align_to: str = "fmax",
    collapse_channels: bool = False,
    nsubbands: int = 1,
    exact_uvw: bool = True,
    npix_x: int = 384,
    npix_y: int = 384,
    pixsize_arcsec: Optional[float] = None,
    pix_rad: Optional[float] = None,
    grid_nu: Optional[int] = None,
    nv: Optional[int] = None,
    mode: str = "nearest",
    min_valid_channels: int = 24,
    min_uv_distance_m: Optional[float] = None,
    uv_threshold: float = 0.0,
    uv_to_m: Optional[np.ndarray] = None,
    m_active: Optional[int] = None,
    return_sparse: bool = False,
    flat_indices: Optional[np.ndarray] = None,
    use_numba: bool = True,
) -> np.ndarray:
    """
    CPU fallback for fused visibility dedispersion and direct-to-grid Hermitian
    half-plane rasterisation.

    Mathematical operations, rounding (IEEE-754 half-to-even), Hermitian folding,
    subband reduction, valid-channel tracking, and per-cell weight normalisation
    match :func:`dedisperse_and_rasterise_chunk_gpu` to within ``< 1e-6`` relative error.

    Parameters
    ----------
    vis_3d : np.ndarray (complex64), shape (nbl, nchan, nt_in)
        Input visibilities including any trailing overlap buffer.
    wgt_3d : np.ndarray (float32), shape (nbl, nchan, nt_in)
        Input visibility weights.
    uvw_3d : np.ndarray (float64), shape (nbl, 3, nt_in)
        Baseline UVW coordinates in metres.
    freqs_hz : np.ndarray (float64), shape (nchan,)
        Channel centre frequencies in Hz.
    delays : np.ndarray (int32/int64), shape (nchan,), optional
        Non-negative integer dispersion delays in samples (from
        ``fastducc.dedisp.get_dm_delays(..., align_to='fmax')``). If omitted,
        computed from ``dm`` and ``dt`` (or zeros when ``dm == 0.0``).
    nt_out : int, optional
        Number of output snapshots ``<= nt_in`` (defaults to ``nt_in``).
    dm : float
        Dispersion measure in pc cm^-3 (used only when ``delays`` is None).
    dt : float, optional
        Sampling interval in seconds (used only when ``delays`` is None and ``dm != 0.0``).
    align_to : {'fmax', 'fmin'}
        Reference alignment frequency when computing delays.
    collapse_channels : bool
        If True with ``nsubbands == 1``, collapse all channels at snapshot UVW.
    nsubbands : int
        Number of coherent subbands when ``collapse_channels=False`` or ``nsubbands > 1``.
    exact_uvw : bool
        If True in subband/per-channel mode, sample UVW at ``clip(t + d_sub, 0, nt_in - 1)``.
    npix_x, npix_y : int
        Default image-plane dimensions in pixels.
    pixsize_arcsec : float, optional
        Pixel angular size in arcseconds (defaults to 22.0 if ``pix_rad`` is None).
    pix_rad : float, optional
        Pixel angular size in radians/pixel (overrides ``pixsize_arcsec``).
    grid_nu, nv : int, optional
        Optional overrides for raster grid dimensions ``(nv, grid_nu)``.
    mode : {'nearest', 'bilinear'}
        Grid splatting kernel.
    min_valid_channels : int
        Minimum valid channel count threshold per snapshot.
    min_uv_distance_m : float, optional
        Minimum projected baseline length ``hypot(u_m, v_m)`` in metres.
    uv_threshold : float
        Post-normalisation amplitude threshold; cells with ``|V| <= uv_threshold``
        (when ``uv_threshold > 0``) are zeroed.
    uv_to_m : np.ndarray (int32), shape (nv, u_half), optional
        Optional active-cell mapping table ``(v, u) -> m_idx`` (``-1`` for inactive).
    m_active : int, optional
        Active cell count ``M`` when ``return_sparse=True``.
    return_sparse : bool
        If True, return compressed ``(M, nt_out)`` array directly using ``uv_to_m``.
    flat_indices : np.ndarray (int64), optional
        Optional 1D active-cell indices used to build ``uv_to_m`` if not supplied.
    use_numba : bool
        Use Numba parallel JIT kernel when available (default True).

    Returns
    -------
    uv_out : np.ndarray (complex64)
        Normalised rasterised visibilities of shape ``(nt_out, nv, u_half)`` or
        ``(M, nt_out)`` when ``return_sparse=True``.
    """
    vis_arr = np.ascontiguousarray(vis_3d, dtype=np.complex64)
    wgt_arr = np.ascontiguousarray(wgt_3d, dtype=np.float32)
    uvw_arr = np.ascontiguousarray(uvw_3d, dtype=np.float64)

    if vis_arr.ndim != 3:
        raise ValueError(f"vis_3d must be 3D (nbl, nchan, nt_in); got shape {vis_arr.shape}.")
    nbl, nchan, nt_in = vis_arr.shape
    if wgt_arr.shape != vis_arr.shape:
        raise ValueError(f"wgt_3d shape {wgt_arr.shape} must match vis_3d {vis_arr.shape}.")
    if uvw_arr.shape != (nbl, 3, nt_in):
        raise ValueError(f"uvw_3d shape {uvw_arr.shape} must match ({nbl}, 3, {nt_in}).")

    eff_nt_out = nt_in if nt_out is None else int(nt_out)
    if eff_nt_out < 0 or eff_nt_out > nt_in:
        raise ValueError(f"nt_out ({eff_nt_out}) must satisfy 0 <= nt_out <= nt_in ({nt_in}).")

    geom = compute_raster_grid_geometry(
        npix_x=npix_x,
        npix_y=npix_y,
        pix_rad=pix_rad,
        pixsize_arcsec=pixsize_arcsec,
        grid_nu=grid_nu,
        nv=nv,
    )
    eff_nv = int(geom["nv"])
    u_half = int(geom["u_half"])
    du = float(geom["du"])
    dv = float(geom["dv"])

    _, delays_host, sub_offsets, sub_freqs_host, sub_delays_host, _ = _resolve_delays_and_subbands(
        freqs_hz=freqs_hz,
        delays=delays,
        dm=dm,
        dt=dt,
        align_to=align_to,
        collapse_channels=collapse_channels,
        nsubbands=nsubbands,
    )
    mode_id = _normalise_mode(mode)
    min_uv = 0.0 if min_uv_distance_m is None else float(min_uv_distance_m)
    thr = float(uv_threshold)

    uv_to_m_arr, m_val = _resolve_uv_to_m_and_m_active(
        uv_to_m=uv_to_m,
        flat_indices=flat_indices,
        nv=eff_nv,
        u_half=u_half,
        m_active=m_active,
        return_sparse=return_sparse,
        target_device="cpu",
    )

    out_shape = (m_val, eff_nt_out) if return_sparse else (eff_nt_out, eff_nv, u_half)
    if eff_nt_out == 0 or nbl == 0 or nchan == 0 or (return_sparse and m_val == 0):
        return np.zeros(out_shape, dtype=np.complex64)

    if use_numba and HAS_NUMBA:
        n_total = int(np.prod(out_shape))
        out_re_flat = np.zeros(n_total, dtype=np.float32)
        out_im_flat = np.zeros(n_total, dtype=np.float32)
        out_w_flat = np.zeros(n_total, dtype=np.float32)
        has_lookup = uv_to_m_arr is not None
        dummy_lookup = np.empty((1, 1), dtype=np.int32) if not has_lookup else uv_to_m_arr

        _rasterise_numba_kernel(
            vis_arr,
            wgt_arr,
            uvw_arr,
            delays_host,
            sub_offsets,
            sub_freqs_host,
            sub_delays_host,
            dummy_lookup,
            has_lookup,
            bool(return_sparse),
            m_val,
            eff_nt_out,
            eff_nv,
            u_half,
            du,
            dv,
            min_uv,
            int(min_valid_channels),
            bool(collapse_channels),
            bool(exact_uvw),
            mode_id,
            thr,
            out_re_flat,
            out_im_flat,
            out_w_flat,
        )
        out = np.empty(out_shape, dtype=np.complex64)
        out.real[:] = out_re_flat.reshape(out_shape)
        out.imag[:] = out_im_flat.reshape(out_shape)
        return out

    return _dedisperse_and_rasterise_numpy(
        vis_3d=vis_arr,
        wgt_3d=wgt_arr,
        uvw_3d=uvw_arr,
        delays=delays_host,
        sub_offsets=sub_offsets,
        sub_freqs_hz=sub_freqs_host,
        sub_delays=sub_delays_host,
        uv_to_m=uv_to_m_arr,
        m_active=m_val,
        sparse_output=bool(return_sparse),
        nt_out=eff_nt_out,
        nv=eff_nv,
        u_half=u_half,
        du=du,
        dv=dv,
        min_uv_distance_m=min_uv,
        min_valid_channels=int(min_valid_channels),
        collapse_channels=bool(collapse_channels),
        exact_uvw=bool(exact_uvw),
        mode_id=mode_id,
        uv_threshold=thr,
    )


# =============================================================================
# GPU Dedispersion + Rasterisation Engine
# =============================================================================

def dedisperse_and_rasterise_chunk_gpu(
    vis_3d: Any,
    wgt_3d: Any,
    uvw_3d: Any,
    freqs_hz: Any,
    delays: Optional[Any] = None,
    *,
    nt_out: Optional[int] = None,
    dm: float = 0.0,
    dt: Optional[float] = None,
    align_to: str = "fmax",
    collapse_channels: bool = False,
    nsubbands: int = 1,
    exact_uvw: bool = True,
    npix_x: int = 384,
    npix_y: int = 384,
    pixsize_arcsec: Optional[float] = None,
    pix_rad: Optional[float] = None,
    grid_nu: Optional[int] = None,
    nv: Optional[int] = None,
    mode: str = "nearest",
    min_valid_channels: int = 24,
    min_uv_distance_m: Optional[float] = None,
    uv_threshold: float = 0.0,
    uv_to_m: Optional[Any] = None,
    m_active: Optional[int] = None,
    return_sparse: bool = False,
    flat_indices: Optional[Any] = None,
    return_host: bool = True,
) -> Any:
    """
    CUDA-accelerated fused visibility dedispersion and direct-to-grid Hermitian
    half-plane rasterisation.

    Inputs may be NumPy arrays or CuPy device arrays. Output is returned either
    as a dense ``(nt_out, nv, u_half)`` ``complex64`` grid (ready for
    ``fastducc.ragged_gpu.pack_uv_chunk_to_ragged_gpu``) or, when
    ``return_sparse=True``, as a compact ``(M, nt_out)`` ``complex64`` matrix
    (ready for ``fastducc.ragged_gpu.pack_chunk_to_ragged_gpu``).

    Parameters
    ----------
    return_host : bool
        If True (default), copy the normalised output to host as ``np.ndarray``.
        If False, return the device ``cupy.ndarray`` directly for zero-copy
        downstream CUDA CSR packing.
    """
    _require_cuda()
    d_vis = cp.ascontiguousarray(cp.asarray(vis_3d), dtype=cp.complex64)
    d_wgt = cp.ascontiguousarray(cp.asarray(wgt_3d), dtype=cp.float32)
    d_uvw = cp.ascontiguousarray(cp.asarray(uvw_3d), dtype=cp.float64)

    if d_vis.ndim != 3:
        raise ValueError(f"vis_3d must be 3D (nbl, nchan, nt_in); got shape {d_vis.shape}.")
    nbl, nchan, nt_in = (int(d_vis.shape[0]), int(d_vis.shape[1]), int(d_vis.shape[2]))
    if d_wgt.shape != d_vis.shape:
        raise ValueError(f"wgt_3d shape {d_wgt.shape} must match vis_3d {d_vis.shape}.")
    if d_uvw.shape != (nbl, 3, nt_in):
        raise ValueError(f"uvw_3d shape {d_uvw.shape} must match ({nbl}, 3, {nt_in}).")

    eff_nt_out = nt_in if nt_out is None else int(nt_out)
    if eff_nt_out < 0 or eff_nt_out > nt_in:
        raise ValueError(f"nt_out ({eff_nt_out}) must satisfy 0 <= nt_out <= nt_in ({nt_in}).")

    geom = compute_raster_grid_geometry(
        npix_x=npix_x,
        npix_y=npix_y,
        pix_rad=pix_rad,
        pixsize_arcsec=pixsize_arcsec,
        grid_nu=grid_nu,
        nv=nv,
    )
    eff_nv = int(geom["nv"])
    u_half = int(geom["u_half"])
    du = float(geom["du"])
    dv = float(geom["dv"])

    _, delays_host, sub_offsets_host, sub_freqs_host, sub_delays_host, nsub_eff = _resolve_delays_and_subbands(
        freqs_hz=freqs_hz,
        delays=delays,
        dm=dm,
        dt=dt,
        align_to=align_to,
        collapse_channels=collapse_channels,
        nsubbands=nsubbands,
    )
    mode_id = _normalise_mode(mode)
    min_uv = 0.0 if min_uv_distance_m is None else float(min_uv_distance_m)
    thr = float(uv_threshold)

    d_uv_to_m, m_val = _resolve_uv_to_m_and_m_active(
        uv_to_m=uv_to_m,
        flat_indices=flat_indices,
        nv=eff_nv,
        u_half=u_half,
        m_active=m_active,
        return_sparse=return_sparse,
        target_device="cuda",
    )

    out_shape = (m_val, eff_nt_out) if return_sparse else (eff_nt_out, eff_nv, u_half)
    if eff_nt_out == 0 or nbl == 0 or nchan == 0 or (return_sparse and m_val == 0):
        if return_host:
            return np.zeros(out_shape, dtype=np.complex64)
        return cp.zeros(out_shape, dtype=cp.complex64)

    d_delays = cp.asarray(delays_host, dtype=cp.int32)
    d_raw_valid = cp.zeros(eff_nt_out, dtype=cp.int32)
    d_dedisp_valid = cp.zeros(eff_nt_out, dtype=cp.int32)

    d_grid_vis = cp.zeros(out_shape, dtype=cp.complex64)
    d_grid_wgt = cp.zeros(out_shape, dtype=cp.float32)

    mod = _get_module()
    valid_kernel = mod.get_function("compute_valid_channels_kernel")
    gy_valid = min(_MAX_GRID_Y, _ceil_div(nchan, 8))
    valid_kernel(
        (eff_nt_out, gy_valid, 1),
        (32, 8, 1),
        (
            d_vis,
            d_wgt,
            d_uvw,
            d_delays,
            d_raw_valid,
            d_dedisp_valid,
            np.int32(nbl),
            np.int32(nchan),
            np.int32(nt_in),
            np.int32(eff_nt_out),
            np.float64(min_uv),
        ),
    )

    d_uv_to_m_ptr: Any = d_uv_to_m if d_uv_to_m is not None else 0
    sparse_flag = np.int32(1 if return_sparse else 0)
    gx_raster = _ceil_div(eff_nt_out, 32)
    gy_raster = min(_MAX_GRID_Y, _ceil_div(nbl, 8))

    if collapse_channels and nsub_eff == 1:
        collapsed_kernel = mod.get_function("dedisperse_and_rasterise_collapsed_kernel")
        collapsed_kernel(
            (gx_raster, gy_raster, 1),
            (32, 8, 1),
            (
                d_vis,
                d_wgt,
                d_uvw,
                d_delays,
                d_raw_valid,
                d_dedisp_valid,
                d_uv_to_m_ptr,
                d_grid_vis,
                d_grid_wgt,
                np.int32(nbl),
                np.int32(nchan),
                np.int32(nt_in),
                np.int32(eff_nt_out),
                np.int32(eff_nv),
                np.int32(u_half),
                np.float64(du),
                np.float64(dv),
                np.float64(sub_freqs_host[0]),
                np.float64(min_uv),
                np.int32(min_valid_channels),
                np.int32(mode_id),
                sparse_flag,
            ),
        )
    else:
        d_sub_offsets = cp.asarray(sub_offsets_host, dtype=cp.int32)
        d_sub_freqs = cp.asarray(sub_freqs_host, dtype=cp.float64)
        d_sub_delays = cp.asarray(sub_delays_host, dtype=cp.int32)
        subband_kernel = mod.get_function("dedisperse_and_rasterise_subband_kernel")
        subband_kernel(
            (gx_raster, gy_raster, 1),
            (32, 8, 1),
            (
                d_vis,
                d_wgt,
                d_uvw,
                d_delays,
                d_sub_offsets,
                d_sub_freqs,
                d_sub_delays,
                d_raw_valid,
                d_dedisp_valid,
                d_uv_to_m_ptr,
                d_grid_vis,
                d_grid_wgt,
                np.int32(nbl),
                np.int32(nchan),
                np.int32(nsub_eff),
                np.int32(nt_in),
                np.int32(eff_nt_out),
                np.int32(eff_nv),
                np.int32(u_half),
                np.float64(du),
                np.float64(dv),
                np.float64(min_uv),
                np.int32(min_valid_channels),
                np.int32(1 if collapse_channels else 0),
                np.int32(1 if exact_uvw else 0),
                np.int32(mode_id),
                sparse_flag,
            ),
        )

    if return_sparse:
        norm_sparse_kernel = mod.get_function("normalise_sparse_grid_kernel")
        gy_norm = min(_MAX_GRID_Y, _ceil_div(m_val, 8))
        norm_sparse_kernel(
            (gx_raster, gy_norm, 1),
            (32, 8, 1),
            (
                d_grid_vis,
                d_grid_wgt,
                d_raw_valid,
                d_dedisp_valid,
                d_grid_vis,
                np.int32(m_val),
                np.int32(eff_nt_out),
                np.int32(nchan),
                np.int32(1 if collapse_channels else 0),
                np.int32(min_valid_channels),
                np.float32(thr),
            ),
        )
    else:
        norm_kernel = mod.get_function("normalise_raster_grid_kernel")
        spatial = eff_nv * u_half
        gx_norm = _ceil_div(spatial, 32)
        gy_norm = min(_MAX_GRID_Y, _ceil_div(eff_nt_out, 8))
        norm_kernel(
            (gx_norm, gy_norm, 1),
            (32, 8, 1),
            (
                d_grid_vis,
                d_grid_wgt,
                d_raw_valid,
                d_dedisp_valid,
                d_grid_vis,
                np.int32(eff_nt_out),
                np.int32(eff_nv),
                np.int32(u_half),
                np.int32(nchan),
                np.int32(1 if collapse_channels else 0),
                np.int32(min_valid_channels),
                np.float32(thr),
            ),
        )

    if return_host:
        return cp.asnumpy(d_grid_vis)
    return d_grid_vis


def dedisperse_and_rasterise_chunk(
    vis_3d: Any,
    wgt_3d: Any,
    uvw_3d: Any,
    freqs_hz: Any,
    delays: Optional[Any] = None,
    *,
    device: str = "auto",
    mode: str = "nearest",
    return_host: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Unified dispatcher for fused visibility dedispersion and direct-to-grid
    Hermitian half-plane rasterisation on CUDA or CPU.

    Parameters
    ----------
    vis_3d : array_like (complex64), shape (nbl, nchan, nt_in)
    wgt_3d : array_like (float32), shape (nbl, nchan, nt_in)
    uvw_3d : array_like (float64), shape (nbl, 3, nt_in)
    freqs_hz : array_like (float64), shape (nchan,)
    delays : array_like (int32), shape (nchan,), optional
    device : {'auto', 'cuda', 'cpu'}
        Execution backend selector.
    mode : {'nearest', 'bilinear'}
        Splatting mode.
    return_host : bool
        When running on CUDA, return a host NumPy array if True or a CuPy device
        array if False.
    **kwargs :
        Additional keyword arguments forwarded to the backend implementation.
    """
    eff_dev = resolve_device(device)
    if eff_dev == "cuda":
        kwargs.pop("use_numba", None)
        return dedisperse_and_rasterise_chunk_gpu(
            vis_3d,
            wgt_3d,
            uvw_3d,
            freqs_hz,
            delays=delays,
            mode=mode,
            return_host=return_host,
            **kwargs,
        )
    return dedisperse_and_rasterise_chunk_cpu(
        vis_3d,
        wgt_3d,
        uvw_3d,
        freqs_hz,
        delays=delays,
        mode=mode,
        **kwargs,
    )


def dedisperse_and_rasterise_rows(
    bl_idx: Any,
    t_rel: Any,
    uvw_rows: Any,
    data_rows: Any,
    wgt_rows: Any,
    flag_rows: Any,
    freqs_hz: Any,
    delays: Optional[Any] = None,
    *,
    flag_row: Optional[Any] = None,
    nbl: Optional[int] = None,
    nt_in: Optional[int] = None,
    nt_out: Optional[int] = None,
    corr_mode: str = "average",
    basis: str = "auto",
    single_pol: str = "XX",
    average_correlations: bool = True,
    corr_index: Optional[int] = None,
    labels: Optional[List[str]] = None,
    lbl2idx: Optional[Dict[str, int]] = None,
    min_uv_distance_m: Optional[float] = None,
    device: str = "auto",
    mode: str = "nearest",
    return_host: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Direct row-streamed helper: stage raw MS row buffers ``(nrow, nchan[, ncorr])``
    onto device/host 3D tensors with correlation collapse and immediately execute
    dedispersion + direct-to-grid rasterisation.

    Parameters
    ----------
    bl_idx, t_rel, uvw_rows, data_rows, wgt_rows, flag_rows, flag_row :
        Streamed MS row buffers (see :func:`stage_ms_rows_cpu`).
    freqs_hz : array_like (float64), shape (nchan,)
        Channel frequencies in Hz.
    delays : array_like (int32), shape (nchan,), optional
        Non-negative channel dispersion delays in samples.
    nbl : int, optional
        Total number of baselines in the staging tensor.
    nt_in : int, optional
        Total time depth of the staging tensor (defaults to ``max(t_rel) + 1``).
    nt_out : int, optional
        Output snapshot count ``<= nt_in`` (defaults to ``nt_in``).
    device : {'auto', 'cuda', 'cpu'}
        Execution backend selector.
    mode : {'nearest', 'bilinear'}
        Splatting mode.
    return_host : bool
        For CUDA execution, copy final output to host if True.
    **kwargs :
        Forwarded to :func:`dedisperse_and_rasterise_chunk`.
    """
    eff_dev = resolve_device(device)
    vis_3d, wgt_3d, uvw_3d = stage_ms_rows(
        bl_idx=bl_idx,
        t_rel=t_rel,
        uvw_rows=uvw_rows,
        data_rows=data_rows,
        wgt_rows=wgt_rows,
        flag_rows=flag_rows,
        flag_row=flag_row,
        nbl=nbl,
        nt_stage=nt_in,
        t_offset=0,
        corr_mode=corr_mode,
        basis=basis,
        single_pol=single_pol,
        average_correlations=average_correlations,
        corr_index=corr_index,
        labels=labels,
        lbl2idx=lbl2idx,
        min_uv_distance_m=min_uv_distance_m,
        device=eff_dev,
        return_host=False,
    )
    return dedisperse_and_rasterise_chunk(
        vis_3d=vis_3d,
        wgt_3d=wgt_3d,
        uvw_3d=uvw_3d,
        freqs_hz=freqs_hz,
        delays=delays,
        nt_out=nt_out,
        min_uv_distance_m=min_uv_distance_m,
        device=eff_dev,
        mode=mode,
        return_host=return_host,
        **kwargs,
    )


def dedisperse_rasterise_to_ragged(
    vis_3d: Any,
    wgt_3d: Any,
    uvw_3d: Any,
    freqs_hz: Any,
    t_chunk: Any,
    flat_indices: Optional[Any] = None,
    delays: Optional[Any] = None,
    *,
    uv_to_m: Optional[Any] = None,
    uv_threshold: float = 1e-4,
    use_sparse_buffer: bool = True,
    device: str = "auto",
    mode: str = "nearest",
    return_host: bool = True,
    **kwargs: Any,
) -> Tuple[Any, Any, Any, Any]:
    """
    End-to-end fused visibility dedispersion, direct-to-grid rasterisation,
    normalisation, and ragged CSR fragment packing.

    On CUDA, retains intermediate arrays exclusively on GPU so no large dense
    UV chunk traverses PCIe.

    Parameters
    ----------
    vis_3d, wgt_3d, uvw_3d, freqs_hz, delays :
        Visibility chunk inputs (see :func:`dedisperse_and_rasterise_chunk`).
    t_chunk : array_like (float64), shape (nt_out,)
        Output snapshot timestamps relative to observation start.
    flat_indices : array_like (int64), shape (M,)
        Linear spatial indices ``v * u_half + u`` for active UV cells.
    uv_to_m : array_like (int32), shape (nv, u_half), optional
        Pre-built active-cell lookup table; constructed from ``flat_indices`` if omitted.
    uv_threshold : float
        Amplitude cutoff ``|V| > uv_threshold`` for both normalisation and CSR packing.
    use_sparse_buffer : bool
        If True, rasterise directly into a compact ``(M, nt_out)`` buffer instead
        of allocating a dense ``(nt_out, nv, u_half)`` cube.
    device : {'auto', 'cuda', 'cpu'}
        Execution backend selector.
    mode : {'nearest', 'bilinear'}
        Splatting mode.
    return_host : bool
        Copy final CSR fragment arrays back to host NumPy arrays if True.
    **kwargs :
        Additional keyword arguments forwarded to :func:`dedisperse_and_rasterise_chunk`.

    Returns
    -------
    cell_indices : np.ndarray or cupy.ndarray (uint32)
    cell_counts : np.ndarray or cupy.ndarray (uint16)
    vis_data : np.ndarray or cupy.ndarray (complex64)
    vis_times : np.ndarray or cupy.ndarray (float64)
    """
    eff_dev = resolve_device(device)
    nt_out_req = kwargs.get("nt_out")
    if nt_out_req is None:
        nt_out_req = int(vis_3d.shape[2])

    t_arr = np.asarray(cp.asnumpy(t_chunk) if (cp is not None and isinstance(t_chunk, cp.ndarray)) else t_chunk, dtype=np.float64)
    if t_arr.ndim != 1 or int(t_arr.shape[0]) != int(nt_out_req):
        raise ValueError(f"t_chunk must be 1D of length nt_out ({nt_out_req}); got shape {t_arr.shape}.")
    if (not use_sparse_buffer) and flat_indices is None:
        raise ValueError("flat_indices is required when use_sparse_buffer=False.")

    raster_out = dedisperse_and_rasterise_chunk(
        vis_3d=vis_3d,
        wgt_3d=wgt_3d,
        uvw_3d=uvw_3d,
        freqs_hz=freqs_hz,
        delays=delays,
        device=eff_dev,
        mode=mode,
        uv_threshold=float(uv_threshold),
        uv_to_m=uv_to_m,
        flat_indices=flat_indices,
        return_sparse=bool(use_sparse_buffer),
        return_host=False,
        **kwargs,
    )

    if eff_dev == "cuda":
        if use_sparse_buffer:
            from fastducc.ragged_gpu import pack_chunk_to_ragged_gpu

            return pack_chunk_to_ragged_gpu(
                raster_out,
                t_chunk,
                threshold=float(uv_threshold),
                return_host=return_host,
            )
        from fastducc.ragged_gpu import pack_uv_chunk_to_ragged_gpu

        return pack_uv_chunk_to_ragged_gpu(
            raster_out,
            flat_indices,
            t_chunk,
            threshold=float(uv_threshold),
            return_host=return_host,
        )

    from fastducc.ragged import pack_chunk_to_ragged_csr

    if use_sparse_buffer:
        dga_sub = raster_out
    else:
        flat_idx_np = np.asarray(flat_indices, dtype=np.int64)
        dga_sub = np.ascontiguousarray(raster_out.reshape(int(nt_out_req), -1)[:, flat_idx_np].T, dtype=np.complex64)
    return pack_chunk_to_ragged_csr(
        dga_sub,
        t_arr,
        threshold=float(uv_threshold),
        device="cpu",
    )


__all__ = [
    "HAS_NUMBA",
    "is_cuda_available",
    "resolve_device",
    "compute_raster_grid_geometry",
    "compute_subband_slices",
    "build_uv_to_m_lookup",
    "stage_ms_rows_cpu",
    "stage_ms_rows_gpu",
    "stage_ms_rows",
    "dedisperse_and_rasterise_chunk_cpu",
    "dedisperse_and_rasterise_chunk_gpu",
    "dedisperse_and_rasterise_chunk",
    "dedisperse_and_rasterise_rows",
    "dedisperse_rasterise_to_ragged",
]
