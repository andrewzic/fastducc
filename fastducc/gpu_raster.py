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
            mod = cp.RawModule(code=src, options=("--std=c++17",))
            _MODULE_CACHE[dev_id] = mod
        return mod


def compute_raster_grid_geometry(
    npix_x: int = 384,
    npix_y: int = 384,
    pix_rad: Optional[float] = None,
    pixsize_arcsec: Optional[float] = None,
    grid_nu: Optional[int] = None,
    nv: Optional[int] = None,
    mode: str = "nearest",
    epsilon: float = 1e-5,
    do_wgridding: bool = False,
    sigma_min: float = 1.1,
    sigma_max: float = 2.6,
    oversample: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Compute Hermitian half-plane direct rasterisation grid dimensions and cell spacings.

    Mathematical definition:
        - For ``mode="nearest"`` (``"pillbox"``) and ``mode="bilinear"`` (or when
          ``oversample=False``):
            grid_nu = npix_x (unless overridden)
            nv      = npix_y (unless overridden)
        - For ``mode="wgridder"`` when ``grid_nu is None`` and ``nv is None`` and
          ``oversample`` is True (the default for ``mode="wgridder"``):
            grid_nu, nv = 2 * good_size_complex(int(npix * ofactor * 0.5) + 1)
            matching ``ducc0.wgridder.vis2grid``.
        u_half  = grid_nu // 2 + 1
        pix_rad = pixsize_arcsec / 206265.0 (unless passed explicitly)
        du      = 1.0 / (grid_nu * pix_rad)  [wavelengths per u cell]
        dv      = 1.0 / (nv * pix_rad)       [wavelengths per v cell]
    """
    if pix_rad is not None:
        eff_pix_rad = float(pix_rad)
    else:
        arcsec = 22.0 if pixsize_arcsec is None else float(pixsize_arcsec)
        eff_pix_rad = arcsec / 206265.0

    if not math.isfinite(eff_pix_rad) or eff_pix_rad <= 0.0:
        raise ValueError(f"pix_rad must be finite and positive; got {eff_pix_rad}.")

    mode_str = str(mode).strip().lower()
    use_oversample = (mode_str in ("wgridder", "ducc_wgridder")) if oversample is None else bool(oversample)

    if use_oversample and (grid_nu is None or nv is None):
        kpar = select_ducc_kernel_params(
            epsilon=float(epsilon),
            do_wgridding=bool(do_wgridding),
            sigma_min=float(sigma_min),
            sigma_max=float(sigma_max),
        )
        ofactor = float(kpar["ofactor"])
        auto_nu = max(16, 2 * _ducc_good_size_complex(int(int(npix_x) * ofactor * 0.5) + 1))
        auto_nv = max(16, 2 * _ducc_good_size_complex(int(int(npix_y) * ofactor * 0.5) + 1))
        eff_nu = int(auto_nu if grid_nu is None else grid_nu)
        eff_nv = int(auto_nv if nv is None else nv)
    else:
        eff_nu = int(npix_x if grid_nu is None else grid_nu)
        eff_nv = int(npix_y if nv is None else nv)

    if eff_nu <= 0 or eff_nv <= 0:
        raise ValueError(f"Grid dimensions must be positive; got grid_nu={eff_nu}, nv={eff_nv}.")

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
    if m in ("nearest", "pillbox"):
        return 0
    if m == "bilinear":
        return 1
    if m in ("wgridder", "ducc_wgridder"):
        return 2
    raise ValueError(
        f"Unsupported rasterisation mode='{mode}'; expected 'nearest' ('pillbox'), 'bilinear', or 'wgridder'."
    )


# =============================================================================
# DUCC W-Gridder Kernel Table, Piecewise Chebyshev Polynomials & W-Phase Helpers
# =============================================================================

# Single-precision (singleprec=True) entries from DUCC src/ducc0/math/gridding_kernel.cc
# Tuple layout: (W, ofactor, epsilon, beta, e0, ndim)
_DUCC_KERNEL_DB_SINGLEPREC: Tuple[Tuple[int, float, float, float, float, int], ...] = (
    # ndim == 2
    (4, 1.20, 0.027638471, 1.30709338717990464, 0.588199361165364421, 2),
    (4, 1.25, 0.017179163, 1.33188061184353357, 0.594018046061197902, 2),
    (4, 1.30, 0.011473319, 1.36280729505750875, 0.596257019042968683, 2),
    (4, 1.35, 0.0085024842, 1.38681869506835920, 0.598430379231771004, 2),
    (4, 1.40, 0.0066874095, 1.42907129923502607, 0.595376586914062478, 2),
    (4, 1.45, 0.0056239728, 1.48220782809787344, 0.588401285807291785, 2),
    (4, 1.50, 0.0047698072, 1.55200220743815076, 0.577678934733072857, 2),
    (4, 1.55, 0.0040717361, 1.61562355889214393, 0.568545023600260491, 2),
    (4, 1.60, 0.0034281853, 1.66557473076714380, 0.562941996256510202, 2),
    (4, 1.65, 0.0029479114, 1.70882788764105853, 0.558200836181640403, 2),
    (4, 1.70, 0.002512611, 1.75829043918185701, 0.552664438883463616, 2),
    (4, 1.75, 0.0021231746, 1.78472069634331620, 0.551698048909505245, 2),
    (4, 1.80, 0.001819194, 1.81566416422526022, 0.549645741780599106, 2),
    (4, 1.85, 0.0015552516, 1.82287784152560750, 0.551604970296223796, 2),
    (4, 1.90, 0.0013487471, 1.84992726643880223, 0.550009918212890803, 2),
    (4, 1.95, 0.0011799293, 1.88333621554904518, 0.547602589925130112, 2),
    (4, 2.00, 0.001038997, 1.90694291856553799, 0.546800994873046786, 2),
    (4, 2.05, 0.00094331549, 1.92870330810546897, 0.545942433675130068, 2),
    (4, 2.10, 0.00086063741, 1.94683447943793375, 0.545642852783203014, 2),
    (4, 2.15, 0.00079978431, 1.95983327229817705, 0.545700836181640558, 2),
    (4, 2.20, 0.0007352812, 1.97344377305772567, 0.546037038167317745, 2),
    (4, 2.25, 0.00068776638, 1.98338402642144063, 0.546487172444661384, 2),
    (4, 2.30, 0.00064537642, 1.99225379096137201, 0.547050221761067768, 2),
    (4, 2.35, 0.00060763777, 2.00018734402126697, 0.547697194417317679, 2),
    (4, 2.40, 0.00058055746, 2.00592770046658009, 0.548258209228515314, 2),
    (4, 2.45, 0.00055028377, 2.01244015163845535, 0.548992156982421986, 2),
    (4, 2.50, 0.0005287527, 2.01718461778428892, 0.549589284261067679, 2),
    (5, 1.20, 0.0091081213, 1.47271143595377563, 0.547661844889323124, 2),
    (5, 1.25, 0.0051393036, 1.51477381388346322, 0.547609456380208193, 2),
    (5, 1.30, 0.0029950485, 1.56042251586914049, 0.546399434407552009, 2),
    (5, 1.35, 0.0018668854, 1.59968931410047732, 0.545594278971354263, 2),
    (5, 1.40, 0.0011678541, 1.62376225789388040, 0.547126770019531361, 2),
    (5, 1.45, 0.00083949653, 1.67027257283528652, 0.544658406575520782, 2),
    (5, 1.50, 0.00064510114, 1.71065928141276036, 0.543058522542317568, 2),
    (5, 1.55, 0.00051838121, 1.74115142822265612, 0.542447662353515714, 2),
    (5, 1.60, 0.00042394497, 1.76781667073567705, 0.542211151123046964, 2),
    (5, 1.65, 0.00036339906, 1.80731658935546857, 0.539609527587890825, 2),
    (5, 1.70, 0.00030426196, 1.83974456787109331, 0.537987009684244954, 2),
    (5, 1.75, 0.00024801871, 1.87821875678168393, 0.535735321044921831, 2),
    (5, 1.80, 0.00020379638, 1.92916208902994835, 0.531913503011067768, 2),
    (5, 1.85, 0.00017221407, 1.97668134901258674, 0.528661346435546742, 2),
    (5, 1.90, 0.00015103689, 2.01164194742838554, 0.526721445719401160, 2),
    (5, 1.95, 0.00013246567, 2.04016469319661509, 0.525634002685546831, 2),
    (5, 2.00, 0.0001160673, 2.06404741075303821, 0.525147755940755356, 2),
    (5, 2.05, 0.00010362875, 2.08155195448133679, 0.525099436442057366, 2),
    (5, 2.10, 9.2543096e-05, 2.09675140380859348, 0.525319671630859353, 2),
    (5, 2.15, 8.431446e-05, 2.10787777370876706, 0.525670623779296853, 2),
    (5, 2.20, 7.5533655e-05, 2.11950395372178768, 0.526264699300130090, 2),
    (5, 2.25, 6.9114106e-05, 2.12779930962456554, 0.526887766520182321, 2),
    (5, 2.30, 6.3452285e-05, 2.13487870958116410, 0.527610015869140359, 2),
    (5, 2.35, 5.852956e-05, 2.14067128499348991, 0.528427378336588482, 2),
    (5, 2.40, 5.5120029e-05, 2.14426320393880232, 0.529153188069661606, 2),
    (5, 2.45, 5.1575852e-05, 2.14864959716796866, 0.530002593994140625, 2),
    (5, 2.50, 4.9135627e-05, 2.15946265326605946, 0.529833730061848529, 2),
    (6, 1.20, 0.0018267494, 1.48702875773111987, 0.543441772460937278, 2),
    (6, 1.25, 0.0010812568, 1.55846243964301179, 0.538631693522135246, 2),
    (6, 1.30, 0.00063328686, 1.62978795369466178, 0.533901977539062544, 2),
    (6, 1.35, 0.00038126795, 1.67900339762369777, 0.531911468505859464, 2),
    (6, 1.40, 0.00022540405, 1.72446373833550348, 0.530658213297526271, 2),
    (6, 1.45, 0.00015329233, 1.77476925320095469, 0.528379058837890714, 2),
    (6, 1.50, 0.00011089008, 1.81697709825303821, 0.527150217692057388, 2),
    (6, 1.55, 8.3901712e-05, 1.84778255886501697, 0.526968638102213394, 2),
    (6, 1.60, 6.4776566e-05, 1.87210167778862835, 0.527448781331380290, 2),
    (6, 1.65, 5.3697783e-05, 1.89141523573133630, 0.527762095133463505, 2),
    (6, 1.70, 4.4299041e-05, 1.93276146782769054, 0.525681813557942856, 2),
    (6, 1.75, 3.5391822e-05, 1.96848670111762192, 0.524261728922526204, 2),
    (6, 1.80, 2.8212779e-05, 1.99691992865668366, 0.523411814371744888, 2),
    (6, 1.85, 2.1992397e-05, 2.02382388644748223, 0.522790781656901182, 2),
    (6, 1.90, 1.8350135e-05, 2.06700524224175330, 0.520410919189453147, 2),
    (6, 1.95, 1.5972068e-05, 2.08590850830078134, 0.520696258544921631, 2),
    (6, 2.00, 1.3859999e-05, 2.10745629204644125, 0.520621490478515692, 2),
    (6, 2.05, 1.2116191e-05, 2.13673621283637205, 0.519232940673827970, 2),
    (6, 2.10, 1.0516966e-05, 2.16247270372178901, 0.518196360270182232, 2),
    (6, 2.15, 9.3851633e-06, 2.17432640923394116, 0.518426767985025960, 2),
    (6, 2.20, 8.1990803e-06, 2.18673960367838571, 0.518857574462890714, 2),
    (6, 2.25, 7.3319484e-06, 2.19571753607855902, 0.519317881266276093, 2),
    (6, 2.30, 6.5752637e-06, 2.20337880452473955, 0.519864654541015603, 2),
    (6, 2.35, 5.9298948e-06, 2.20968983968098920, 0.520482126871744888, 2),
    (6, 2.40, 5.4904875e-06, 2.21353166368272536, 0.521029408772786695, 2),
    (6, 2.45, 5.0308227e-06, 2.23052520751953054, 0.520584869384765669, 2),
    (6, 2.50, 4.625964e-06, 2.25309499104817679, 0.519467417399088638, 2),
    (7, 1.20, 0.0007571742, 1.57580659654405375, 0.528578694661458459, 2),
    (7, 1.25, 0.0003367945, 1.63487184312608536, 0.526731363932291696, 2),
    (7, 1.30, 0.00017840443, 1.68659379747178795, 0.525407663981120043, 2),
    (7, 1.35, 0.00010456205, 1.73841366238064232, 0.523535919189453192, 2),
    (7, 1.40, 5.9581163e-05, 1.78485124376085036, 0.522390492757161540, 2),
    (7, 1.45, 3.5499217e-05, 1.81957685682508630, 0.522037506103515625, 2),
    (7, 1.50, 2.341159e-05, 1.87820756700303804, 0.519385019938150894, 2),
    (7, 1.55, 1.6709902e-05, 1.91922556559244817, 0.518272145589192657, 2),
    (7, 1.60, 1.2162659e-05, 1.95363786485460023, 0.517866770426432477, 2),
    (7, 1.65, 9.3333251e-06, 1.97871042887369786, 0.517832692464192723, 2),
    (7, 1.70, 7.2180318e-06, 2.00276845296223982, 0.517857615152994666, 2),
    (7, 1.75, 5.6180048e-06, 2.02892269558376759, 0.517637888590494311, 2),
    (7, 1.80, 4.4823027e-06, 2.06023915608723929, 0.516692860921223907, 2),
    (7, 1.85, 3.6615236e-06, 2.09614715576171839, 0.515417734781901116, 2),
    (7, 1.90, 3.1044253e-06, 2.09791141086154465, 0.517407989501953436, 2),
    (7, 1.95, 2.6579785e-06, 2.11293928358289929, 0.518028004964192657, 2),
    (7, 2.00, 2.292283e-06, 2.13445349799262107, 0.517974090576171942, 2),
    (7, 2.05, 2.0048563e-06, 2.20160708957248241, 0.512737274169921653, 2),
    (7, 2.10, 1.7431904e-06, 2.22028656005859348, 0.512581634521484086, 2),
    (7, 2.15, 1.5434754e-06, 2.23297949896918402, 0.512734731038411162, 2),
    (7, 2.20, 1.3478827e-06, 2.24713083902994715, 0.512998199462890270, 2),
    (7, 2.25, 1.2005263e-06, 2.25832807752821108, 0.513249460856119444, 2),
    (7, 2.30, 1.0711768e-06, 2.26957007514105769, 0.513472747802734553, 2),
    (7, 2.35, 9.585121e-07, 2.28155059814453098, 0.513612620035807321, 2),
    (7, 2.40, 8.7454818e-07, 2.29392649332682330, 0.513477325439453236, 2),
    (7, 2.45, 7.9380344e-07, 2.30712670220269001, 0.513562774658203103, 2),
    (7, 2.50, 7.3427447e-07, 2.31122216118706580, 0.514211273193359131, 2),
    (8, 1.20, 0.00073262844, 1.66550945705837661, 0.516498819986978996, 2),
    (8, 1.25, 0.00026113174, 1.71241234673394072, 0.516673533121744799, 2),
    (8, 1.30, 0.0001097567, 1.74426591661241281, 0.518212127685547030, 2),
    (8, 1.35, 5.7327797e-05, 1.78767852783203129, 0.517631785074869954, 2),
    (8, 1.40, 3.1679034e-05, 1.83155364990234371, 0.516968027750650871, 2),
    (8, 1.45, 1.9085793e-05, 1.87205691867404500, 0.516137440999349106, 2),
    (8, 1.50, 1.2356395e-05, 1.92130686442057286, 0.514534759521484197, 2),
    (8, 1.55, 8.3078466e-06, 1.96372358534071156, 0.513400522867838327, 2),
    (8, 1.60, 5.9643915e-06, 2.00066104465060768, 0.512841542561849106, 2),
    (8, 1.65, 4.4710511e-06, 2.02756873236762081, 0.512707773844401027, 2),
    (8, 1.70, 3.4523033e-06, 2.05249209933810794, 0.512742869059244599, 2),
    (8, 1.75, 2.7373352e-06, 2.07740800645616330, 0.512746938069661429, 2),
    (8, 1.80, 2.2109254e-06, 2.10268198649088545, 0.512466684977213371, 2),
    (8, 1.85, 1.8013756e-06, 2.13499806722005214, 0.511604054768880157, 2),
    (8, 1.90, 1.5252763e-06, 2.16502397325303830, 0.510731252034505090, 2),
    (8, 1.95, 1.3043269e-06, 2.19154747856987875, 0.510195668538411362, 2),
    (8, 2.00, 1.1225258e-06, 2.21735483805338607, 0.509709421793619666, 2),
    (8, 2.05, 9.913048e-07, 2.23568742540147580, 0.509645334879556988, 2),
    (8, 2.10, 8.7847996e-07, 2.25366193983290009, 0.509593963623046542, 2),
    (8, 2.15, 7.863896e-07, 2.26725006103515625, 0.509669240315754779, 2),
    (8, 2.20, 7.1103723e-07, 2.28289710150824599, 0.509803517659505046, 2),
    (8, 2.25, 6.487745e-07, 2.29611223008897580, 0.509888966878255023, 2),
    (8, 2.30, 5.937697e-07, 2.31027103000217027, 0.509893544514973596, 2),
    (8, 2.35, 5.4509224e-07, 2.32824554443359366, 0.509586842854817479, 2),
    (8, 2.40, 5.0481753e-07, 2.34016638861762161, 0.509583791097005245, 2),
    (8, 2.45, 4.7251724e-07, 2.35166948106553786, 0.509892018636067701, 2),
    (8, 2.50, 4.4514883e-07, 2.35677575005425410, 0.510489654541015692, 2),
    # ndim == 3
    (4, 1.20, 0.04150764, 1.30709338717990464, 0.588199361165364421, 3),
    (4, 1.25, 0.025790962, 1.33188061184353357, 0.594018046061197902, 3),
    (4, 1.30, 0.017221092, 1.36280729505750875, 0.596257019042968683, 3),
    (4, 1.35, 0.012760142, 1.38681869506835920, 0.598430379231771004, 3),
    (4, 1.40, 0.010034843, 1.42907129923502607, 0.595376586914062478, 3),
    (4, 1.45, 0.0084382415, 1.48220782809787344, 0.588401285807291785, 3),
    (4, 1.50, 0.0071561917, 1.55200220743815076, 0.577678934733072857, 3),
    (4, 1.55, 0.0061085655, 1.61562355889214393, 0.568545023600260491, 3),
    (4, 1.60, 0.0051429525, 1.66557473076714380, 0.562941996256510202, 3),
    (4, 1.65, 0.0044223467, 1.70882788764105853, 0.558200836181640403, 3),
    (4, 1.70, 0.0037692633, 1.75829043918185701, 0.552664438883463616, 3),
    (4, 1.75, 0.0031850242, 1.78472069634331620, 0.551698048909505245, 3),
    (4, 1.80, 0.0027289867, 1.81566416422526022, 0.549645741780599106, 3),
    (4, 1.85, 0.0023330248, 1.82287784152560750, 0.551604970296223796, 3),
    (4, 1.90, 0.0020232305, 1.84992726643880223, 0.550009918212890803, 3),
    (4, 1.95, 0.0017699727, 1.88333621554904518, 0.547602589925130112, 3),
    (4, 2.00, 0.0015585499, 1.90694291856553799, 0.546800994873046786, 3),
    (4, 2.05, 0.00141501, 1.92870330810546897, 0.545942433675130068, 3),
    (4, 2.10, 0.0012909783, 1.94683447943793375, 0.545642852783203014, 3),
    (4, 2.15, 0.0011996866, 1.95983327229817705, 0.545700836181640558, 3),
    (4, 2.20, 0.001102923, 1.97344377305772567, 0.546037038167317745, 3),
    (4, 2.25, 0.0010316431, 1.98338402642144063, 0.546487172444661384, 3),
    (4, 2.30, 0.00096805153, 1.99225379096137201, 0.547050221761067768, 3),
    (4, 2.35, 0.00091143774, 2.00018734402126697, 0.547697194417317679, 3),
    (4, 2.40, 0.00087081226, 2.00592770046658009, 0.548258209228515314, 3),
    (4, 2.45, 0.00082539805, 2.01244015163845535, 0.548992156982421986, 3),
    (4, 2.50, 0.00079309819, 2.01718461778428892, 0.549589284261067679, 3),
    (5, 1.20, 0.013925114, 1.47271143595377563, 0.547661844889323124, 3),
    (5, 1.25, 0.0078059267, 1.51477381388346322, 0.547609456380208193, 3),
    (5, 1.30, 0.0045340779, 1.56053441365559875, 0.546380106608072791, 3),
    (5, 1.35, 0.002821534, 1.59968931410047732, 0.545594278971354263, 3),
    (5, 1.40, 0.0017635314, 1.62376225789388018, 0.547126770019531250, 3),
    (5, 1.45, 0.0012659595, 1.67027257283528652, 0.544658406575520782, 3),
    (5, 1.50, 0.00097191815, 1.71065928141276036, 0.543058522542317568, 3),
    (5, 1.55, 0.00078030659, 1.74115142822265612, 0.542447662353515714, 3),
    (5, 1.60, 0.00063781753, 1.76781667073567705, 0.542211151123046964, 3),
    (5, 1.65, 0.00054641212, 1.80731658935546857, 0.539609527587890825, 3),
    (5, 1.70, 0.0004573431, 1.83974456787109331, 0.537987009684244954, 3),
    (5, 1.75, 0.00037273199, 1.87821875678168393, 0.535735321044921831, 3),
    (5, 1.80, 0.0003062092, 1.92916208902994835, 0.531913503011067768, 3),
    (5, 1.85, 0.00025869826, 1.97668134901258674, 0.528661346435546742, 3),
    (5, 1.90, 0.00022684339, 2.01164194742838554, 0.526721445719401160, 3),
    (5, 1.95, 0.00019891815, 2.04016469319661509, 0.525634002685546831, 3),
    (5, 2.00, 0.00017426706, 2.06404741075303821, 0.525147755940755356, 3),
    (5, 2.05, 0.00015557231, 2.08155195448133679, 0.525099436442057366, 3),
    (5, 2.10, 0.00013891336, 2.09675140380859348, 0.525319671630859353, 3),
    (5, 2.15, 0.00012654554, 2.10787777370876706, 0.525670623779296853, 3),
    (5, 2.20, 0.00011335614, 2.11950395372178768, 0.526264699300130090, 3),
    (5, 2.25, 0.0001037115, 2.12779930962456554, 0.526887766520182321, 3),
    (5, 2.30, 9.5205561e-05, 2.13487870958116410, 0.527610015869140359, 3),
    (5, 2.35, 8.7810064e-05, 2.14067128499348991, 0.528427378336588482, 3),
    (5, 2.40, 8.2686012e-05, 2.14426320393880232, 0.529153188069661606, 3),
    (5, 2.45, 7.7362661e-05, 2.14864959716796866, 0.530002593994140625, 3),
    (5, 2.50, 7.3695816e-05, 2.15946265326605946, 0.529833730061848529, 3),
    (6, 1.20, 0.0045594551, 1.50582012600368942, 0.540683492024739643, 3),
    (6, 1.25, 0.0021050185, 1.56927736070421009, 0.537108357747395737, 3),
    (6, 1.30, 0.0011154959, 1.62996512518988745, 0.533878580729166607, 3),
    (6, 1.35, 0.00064507169, 1.67900339762369777, 0.531911468505859464, 3),
    (6, 1.40, 0.00037315221, 1.72458309597439219, 0.530643971761067812, 3),
    (6, 1.45, 0.00024824541, 1.77476925320095469, 0.528379058837890714, 3),
    (6, 1.50, 0.00017724061, 1.81697709825303821, 0.527150217692057388, 3),
    (6, 1.55, 0.0001324623, 1.84778255886501697, 0.526968638102213394, 3),
    (6, 1.60, 0.00010158561, 1.87210167778862835, 0.527448781331380290, 3),
    (6, 1.65, 8.3583491e-05, 1.89237382676866250, 0.527660369873046875, 3),
    (6, 1.70, 6.8551138e-05, 1.93276146782769054, 0.525681813557942856, 3),
    (6, 1.75, 5.4616423e-05, 1.96848670111762192, 0.524261728922526204, 3),
    (6, 1.80, 4.3449372e-05, 1.99702436659071147, 0.523401641845703258, 3),
    (6, 1.85, 3.3827605e-05, 2.02382388644748223, 0.522790781656901182, 3),
    (6, 1.90, 2.8156339e-05, 2.06700524224175330, 0.520410919189453147, 3),
    (6, 1.95, 2.4445725e-05, 2.08590850830078134, 0.520696258544921631, 3),
    (6, 2.00, 2.1164377e-05, 2.10745629204644125, 0.520621490478515692, 3),
    (6, 2.05, 1.8468148e-05, 2.13673621283637205, 0.519232940673827970, 3),
    (6, 2.10, 1.6005415e-05, 2.16247270372178857, 0.518196360270182121, 3),
    (6, 2.15, 1.4258813e-05, 2.17432640923394116, 0.518426767985025960, 3),
    (6, 2.20, 1.2444461e-05, 2.18673960367838571, 0.518857574462890714, 3),
    (6, 2.25, 1.1114558e-05, 2.19571753607855946, 0.519317881266276093, 3),
    (6, 2.30, 9.9546958e-06, 2.20337880452473955, 0.519864654541015603, 3),
    (6, 2.35, 8.965476e-06, 2.20968983968098920, 0.520482126871744888, 3),
    (6, 2.40, 8.2885097e-06, 2.21353166368272536, 0.521029408772786695, 3),
    (6, 2.45, 7.5852234e-06, 2.23052520751953054, 0.520584869384765669, 3),
    (6, 2.50, 6.9655517e-06, 2.25309499104817679, 0.519467417399088638, 3),
    (7, 1.30, 0.0008254969, 1.74045766194661389, 0.518593088785807343, 3),
    (7, 1.35, 0.00038442438, 1.76697370741102411, 0.520175933837890514, 3),
    (7, 1.40, 0.00018937161, 1.80296749538845447, 0.520410919189453147, 3),
    (7, 1.45, 0.00010361285, 1.82994232177734317, 0.521003468831379934, 3),
    (7, 1.50, 6.235019e-05, 1.87820756700303826, 0.519385019938151005, 3),
    (7, 1.55, 4.0294879e-05, 1.91923302544487862, 0.518271636962890581, 3),
    (7, 1.60, 2.7808874e-05, 1.95363786485460067, 0.517866770426432477, 3),
    (7, 1.65, 2.026018e-05, 1.97871042887369786, 0.517832692464192612, 3),
    (7, 1.70, 1.5136149e-05, 2.00276845296224026, 0.517857615152994666, 3),
    (7, 1.75, 1.151624e-05, 2.02899729410807383, 0.517629750569661207, 3),
    (7, 1.80, 8.9463191e-06, 2.06023915608723929, 0.516692860921223907, 3),
    (7, 1.85, 7.0887638e-06, 2.09950408935546839, 0.515063222249348862, 3),
    (7, 1.90, 5.8925966e-06, 2.11824323866102393, 0.515318044026692790, 3),
    (7, 1.95, 4.9355286e-06, 2.13582611083984419, 0.515685272216796831, 3),
    (7, 2.00, 4.1417575e-06, 2.18169674343532982, 0.512996164957682188, 3),
    (7, 2.05, 3.5555298e-06, 2.20350562201605849, 0.512531789143879868, 3),
    (7, 2.10, 3.0496661e-06, 2.22028656005859348, 0.512581634521484086, 3),
    (7, 2.15, 2.6606782e-06, 2.23297949896918446, 0.512734731038411162, 3),
    (7, 2.20, 2.303615e-06, 2.24713083902994715, 0.512998199462890270, 3),
    (7, 2.25, 2.0304322e-06, 2.25832807752821108, 0.513249460856119444, 3),
    (7, 2.30, 1.7924162e-06, 2.26957007514105769, 0.513472747802734553, 3),
    (7, 2.35, 1.5860954e-06, 2.28171844482421804, 0.513596852620442856, 3),
    (7, 2.40, 1.4287925e-06, 2.29392649332682330, 0.513477325439453236, 3),
    (7, 2.45, 1.2852443e-06, 2.30712670220269001, 0.513562774658203103, 3),
    (7, 2.50, 1.1779738e-06, 2.31585845947265678, 0.513826243082682321, 3),
    (8, 1.65, 1.9452728e-05, 2.05627424452040009, 0.509817759195963283, 3),
    (8, 1.70, 1.3629565e-05, 2.07524464925130303, 0.510539499918619577, 3),
    (8, 1.75, 9.9866112e-06, 2.09811655680338527, 0.510826365152994732, 3),
    (8, 1.80, 7.4765723e-06, 2.12325252956814214, 0.510612742106119621, 3),
    (8, 1.85, 5.6599612e-06, 2.14303978814019125, 0.510897064208984397, 3),
    (8, 1.90, 4.4795853e-06, 2.16549021402994812, 0.510690053304036273, 3),
    (8, 1.95, 3.5909836e-06, 2.19277462429470438, 0.510074615478515847, 3),
    (8, 2.00, 2.9123621e-06, 2.21736975775824741, 0.509707895914713438, 3),
    (8, 2.05, 2.4525224e-06, 2.23633270263671724, 0.509581756591796942, 3),
    (8, 2.10, 2.0782745e-06, 2.25366566975911509, 0.509593454996744688, 3),
    (8, 2.15, 1.7816567e-06, 2.26725006103515581, 0.509669240315755001, 3),
    (8, 2.20, 1.5571862e-06, 2.28289710150824643, 0.509803517659505379, 3),
    (8, 2.25, 1.3738399e-06, 2.29649641248914893, 0.509857432047526093, 3),
    (8, 2.30, 1.2158171e-06, 2.31117740207248312, 0.509815724690755201, 3),
    (8, 2.35, 1.0793857e-06, 2.32826419406467000, 0.509585316975911473, 3),
    (8, 2.40, 9.6672011e-07, 2.34473554823133679, 0.509188588460286473, 3),
    (8, 2.45, 8.8091041e-07, 2.36683909098307277, 0.508585357666015647, 3),
    (8, 2.50, 8.0898294e-07, 2.38400794135199634, 0.508149464925130023, 3),
)

_DUCC_POLY_CACHE: Dict[Tuple[int, float, float], np.ndarray] = {}


def _ducc_good_size_complex(n: int) -> int:
    """Port of ``ducc0::detail_fft::good_size_complex`` (2, 3, 5, 7, 11 smooth)."""
    if n <= 6:
        return max(1, int(n))
    best = 2 * n
    f2 = 1
    while f2 < best:
        f23 = f2
        while f23 < best:
            f235 = f23
            while f235 < best:
                f2357 = f235
                while f2357 < best:
                    f235711 = f2357
                    while f235711 < best:
                        if f235711 >= n:
                            best = f235711
                        f235711 *= 11
                    f2357 *= 7
                f235 *= 5
            f23 *= 3
        f2 *= 2
    return int(best)


def get_ducc_available_kernels(
    epsilon: float = 1e-5,
    ndim: int = 2,
    sigma_min: float = 1.1,
    sigma_max: float = 2.6,
) -> List[Tuple[int, float, float, float, float, int]]:
    """
    Port of ``ducc0::getAvailableKernels<float>`` (``gridding_kernel.h:555-581``).
    """
    eps = float(epsilon)
    smin = float(sigma_min)
    smax = float(sigma_max)
    ofc: Dict[int, float] = {w: smax for w in range(1, 20)}
    best: Dict[int, Optional[Tuple[int, float, float, float, float, int]]] = {w: None for w in range(1, 20)}
    for entry in _DUCC_KERNEL_DB_SINGLEPREC:
        w, ofactor, k_eps, _beta, _e0, k_ndim = entry
        if k_ndim == int(ndim) and w <= 8 and k_eps <= eps and smin <= ofactor <= ofc[w]:
            ofc[w] = ofactor
            best[w] = entry
    res = [best[w] for w in range(1, 20) if best[w] is not None]
    if not res:
        # Relax epsilon to best achievable single-precision kernel for (ndim, sigma_min, sigma_max)
        candidates = [
            e for e in _DUCC_KERNEL_DB_SINGLEPREC
            if e[5] == int(ndim) and e[0] <= 8 and smin <= e[1] <= smax
        ]
        if not candidates:
            raise ValueError(
                f"No appropriate DUCC gridding kernel found for epsilon={epsilon}, "
                f"ndim={ndim}, sigma_min={sigma_min}, sigma_max={sigma_max}."
            )
        min_eps_entry = min(candidates, key=lambda e: e[2])
        return [min_eps_entry]
    return res  # type: ignore[return-value]


def select_ducc_kernel_params(
    epsilon: float = 1e-5,
    do_wgridding: bool = False,
    sigma_min: float = 1.1,
    sigma_max: float = 2.6,
    supp: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Select DUCC ``KernelParams`` matching ``Wgridder::getNuNv`` when ``use_grid_out=True``
    (smallest oversampling factor among available single-precision kernels).
    """
    ndim = 3 if do_wgridding else 2
    avail = get_ducc_available_kernels(
        epsilon=epsilon, ndim=ndim, sigma_min=sigma_min, sigma_max=sigma_max
    )
    if supp is not None:
        supp_i = int(supp)
        avail_supp = [e for e in avail if e[0] == supp_i]
        if avail_supp:
            avail = avail_supp
    chosen = min(avail, key=lambda e: e[1])
    w, ofactor, k_eps, beta, e0, k_ndim = chosen
    deg = w + 3 + (w & 1)
    return {
        "W": int(w),
        "deg": int(deg),
        "ofactor": float(ofactor),
        "epsilon": float(k_eps),
        "beta": float(beta),
        "e0": float(e0),
        "ndim": int(k_ndim),
    }


def build_ducc_polynomial_coeffs(
    W: int,
    beta_param: float,
    e0: float,
) -> np.ndarray:
    """
    Exact port of DUCC ``FunctionApproximator`` (``gridding_kernel.h:101-156``)
    and ``TemplateKernel::transferCoeffs`` / ``prepCoeff`` (``gridding_kernel.h:328-344``,
    ``wgridder_sycl.h:866-880``).

    Returns a float32 array of shape ``((W + 1) // 2, deg + 1)`` where
    ``deg = W + 3 + (W & 1)``.
    """
    key = (int(W), float(beta_param), float(e0))
    cached = _DUCC_POLY_CACHE.get(key)
    if cached is not None:
        return cached

    supp = int(W)
    d_input = supp + 3
    deg = supp + 3 + (supp & 1)
    beta = float(beta_param) * supp
    e0_f = float(e0)

    def _esknew(v: np.ndarray) -> np.ndarray:
        tmp = (1.0 - v) * (1.0 + v)
        mask = tmp >= 0.0
        out = np.zeros_like(v, dtype=np.float64)
        out[mask] = np.exp(beta * (np.power(tmp[mask], e0_f) - 1.0))
        return out

    idx_j = np.arange(d_input + 1, dtype=np.float64)
    chebroot = np.cos((2.0 * idx_j + 1.0) * math.pi / (2.0 * d_input + 2.0))

    locations = np.empty((supp, d_input + 1), dtype=np.float64)
    for i in range(supp):
        l_bound = -1.0 + 2.0 * i / float(supp)
        r_bound = -1.0 + 2.0 * (i + 1) / float(supp)
        locations[i, :] = chebroot * (r_bound - l_bound) * 0.5 + (r_bound + l_bound) * 0.5

    funcval = _esknew(locations)

    cheb_C = np.zeros((d_input + 1, d_input + 1), dtype=np.float64)
    cheb_C[0, 0] = 1.0
    cheb_C[1, 1] = 1.0
    for j in range(2, d_input + 1):
        cheb_C[j, 0] = -cheb_C[j - 2, 0]
        for k in range(1, j + 1):
            cheb_C[j, k] = 2.0 * cheb_C[j - 1, k - 1] - cheb_C[j - 2, k]

    cos_mat = np.cos(
        np.outer(idx_j, 2.0 * idx_j + 1.0) * math.pi / (2.0 * d_input + 2.0)
    )

    raw_coeff = np.empty((supp, d_input + 1), dtype=np.float64)
    for i in range(supp):
        y = funcval[i].copy()
        avg = float(np.mean(y))
        y -= avg
        lcf = (2.0 / (d_input + 1.0)) * (cos_mat @ y)
        lcf[0] *= 0.5
        lcf2 = cheb_C.T @ lcf
        lcf2[0] += avg
        raw_coeff[i, :] = lcf2[::-1]

    lim = (supp + 1) // 2
    ofs = deg - d_input  # 1 when W is odd, 0 when W is even
    prep = np.zeros((lim, deg + 1), dtype=np.float32)
    for i in range(lim):
        for j in range(d_input + 1):
            prep[i, j + ofs] = np.float32(raw_coeff[i, j])

    prep = np.ascontiguousarray(prep, dtype=np.float32)
    _DUCC_POLY_CACHE[key] = prep
    return prep


def build_ducc_corfunc(
    W: int,
    beta_param: float,
    e0: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Port of ``ducc0::detail_gridding_kernel::GLFullCorrection`` (``gridding_kernel.h:253-271``).
    Returns ``(x_scaled, wgtpsi)`` where ``corfunc(v) = 1.0 / sum(wgtpsi * cos(x_scaled * v))``.
    """
    from scipy.special import roots_legendre

    supp = int(W)
    beta = float(beta_param) * supp
    e0_f = float(e0)
    p = int(1.5 * supp) + 2
    roots, weights = roots_legendre(2 * p)
    pos = roots > 0.0
    x_sym = np.ascontiguousarray(roots[pos], dtype=np.float64)
    w_sym = np.ascontiguousarray(weights[pos] * 2.0, dtype=np.float64)
    tmp = (1.0 - x_sym) * (1.0 + x_sym)
    esk = np.where(tmp >= 0.0, np.exp(beta * (np.power(np.maximum(tmp, 0.0), e0_f) - 1.0)), 0.0)
    wgtpsi = np.ascontiguousarray(w_sym * esk * (supp * 0.5), dtype=np.float64)
    x_scaled = np.ascontiguousarray(x_sym * (math.pi * supp), dtype=np.float64)
    return x_scaled, wgtpsi


def eval_ducc_corfunc(
    v: np.ndarray,
    x_scaled: np.ndarray,
    wgtpsi: np.ndarray,
) -> np.ndarray:
    """Evaluate DUCC ``KernelCorrection::corfunc(v)`` on scalar or array ``v``."""
    v_arr = np.asarray(v, dtype=np.float64)
    denom = np.sum(wgtpsi[:, None] * np.cos(x_scaled[:, None] * v_arr.ravel()[None, :]), axis=0)
    return (1.0 / denom).reshape(v_arr.shape)


def compute_ducc_w_params(
    npix_x: int = 384,
    npix_y: int = 384,
    pixsize_x: float = 22.0 / 206265.0,
    pixsize_y: float = 22.0 / 206265.0,
    center_x: float = 0.0,
    center_y: float = 0.0,
    allow_nshift: bool = True,
    apply_w_phase: bool = True,
    do_wgridding: bool = False,
) -> Dict[str, float]:
    """
    Port of ``Wgridder::getNuNv`` field-of-view $(l, m, n - 1)$ extrema and
    $w$-phase shift parameters (``wgridder_impl.h:1560-1587``).
    """
    lshift = float(center_x)
    mshift = float(center_y)
    xmin = lshift - 0.5 * float(npix_x) * float(pixsize_x)
    xmax = xmin + float(npix_x - 1) * float(pixsize_x)
    ymin = mshift - 0.5 * float(npix_y) * float(pixsize_y)
    ymax = ymin + float(npix_y - 1) * float(pixsize_y)

    xext = [xmin, xmax]
    yext = [ymin, ymax]
    if xmin * xmax < 0.0:
        xext.append(0.0)
    if ymin * ymax < 0.0:
        yext.append(0.0)

    nm1min = 1e300
    nm1max = -1e300
    for xc in xext:
        for yc in yext:
            tmp = xc * xc + yc * yc
            nval = (math.sqrt(1.0 - tmp) - 1.0) if tmp <= 1.0 else (-math.sqrt(tmp - 1.0) - 1.0)
            if nval < nm1min:
                nm1min = nval
            if nval > nm1max:
                nm1max = nval

    if not allow_nshift or not apply_w_phase:
        nshift = 0.0
    elif do_wgridding:
        nshift = -0.5 * (nm1max + nm1min)
    elif lshift != 0.0 or mshift != 0.0:
        tmp0 = lshift * lshift + mshift * mshift
        nm1_0 = (math.sqrt(1.0 - tmp0) - 1.0) if tmp0 <= 1.0 else (-math.sqrt(tmp0 - 1.0) - 1.0)
        nshift = -nm1_0
    else:
        nshift = 0.0
    do_shift = (lshift != 0.0) or (mshift != 0.0) or (nshift != 0.0)
    return {
        "lshift": lshift,
        "mshift": mshift,
        "nm1min": float(nm1min),
        "nm1max": float(nm1max),
        "nshift": float(nshift),
        "do_shift": bool(do_shift),
    }


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

    @nb.njit(inline="always", fastmath=False)
    def _splat_wgridder_sample_numba(
        t: int,
        nt_out: int,
        grid_nu: int,
        nv: int,
        u_half: int,
        u_lam: float,
        v_lam: float,
        w_lam: float,
        r_f32: np.float32,
        i_f32: np.float32,
        w_f32: np.float32,
        pixsize_x: float,
        pixsize_y: float,
        supp: int,
        deg: int,
        poly_coeffs: np.ndarray,
        lshift: float,
        mshift: float,
        nshift: float,
        do_shift: bool,
        uv_to_m: np.ndarray,
        has_uv_to_m: bool,
        sparse_output: bool,
        out_re: np.ndarray,
        out_im: np.ndarray,
        out_w: np.ndarray,
        ku: np.ndarray,
        kv: np.ndarray,
    ) -> None:
        imflip = np.float32(1.0)
        if w_lam < 0.0:
            u_lam = -u_lam
            v_lam = -v_lam
            w_lam = -w_lam
            imflip = np.float32(-1.0)
        if do_shift:
            tmp = float(imflip) * (u_lam * lshift + v_lam * mshift + w_lam * nshift)
            ang = np.float32(2.0 * math.pi) * np.float32(tmp - math.floor(tmp))
            c = np.float32(math.cos(float(ang)))
            s = np.float32(math.sin(float(ang)))
            nvr = np.float32(r_f32 * c - i_f32 * s)
            nvi = np.float32(r_f32 * s + i_f32 * c)
            r_f32 = nvr
            i_f32 = nvi
        i_f32 = np.float32(i_f32 * imflip)

        ushift = float(supp) * (-0.5) + 1.0 + float(grid_nu)
        vshift = float(supp) * (-0.5) + 1.0 + float(nv)
        maxiu0 = (grid_nu + supp) - ((supp + 1) // 2)
        maxiv0 = (nv + supp) - ((supp + 1) // 2)

        u = u_lam * pixsize_x
        u = (u - math.floor(u)) * float(grid_nu)
        iu0 = int(u + ushift) - grid_nu
        if iu0 > maxiu0:
            iu0 = maxiu0
        ufrac = np.float32(u - float(iu0))

        v = v_lam * pixsize_y
        v = (v - math.floor(v)) * float(nv)
        iv0 = int(v + vshift) - nv
        if iv0 > maxiv0:
            iv0 = maxiv0
        vfrac = np.float32(v - float(iv0))

        x0 = np.float32(-2.0) * ufrac + np.float32(supp - 1)
        y0 = np.float32(-2.0) * vfrac + np.float32(supp - 1)
        x2 = np.float32(x0 * x0)
        y2 = np.float32(y0 * y0)
        lim = (supp + 1) >> 1

        for i in range(lim):
            tx = poly_coeffs[i, 0]
            tx2 = poly_coeffs[i, 1]
            ty = poly_coeffs[i, 0]
            ty2 = poly_coeffs[i, 1]
            for j in range(2, deg, 2):
                c_even = poly_coeffs[i, j]
                c_odd = poly_coeffs[i, j + 1]
                tx = np.float32(tx * x2 + c_even)
                ty = np.float32(ty * y2 + c_even)
                tx2 = np.float32(tx2 * x2 + c_odd)
                ty2 = np.float32(ty2 * y2 + c_odd)
            px = np.float32(tx * x0)
            py = np.float32(ty * y0)
            ku[i] = np.float32(tx2 + px)
            kv[i] = np.float32(ty2 + py)
            ku[supp - 1 - i] = np.float32(tx2 - px)
            kv[supp - 1 - i] = np.float32(ty2 - py)

        for cu in range(supp):
            iu = iu0 + cu
            if iu < 0:
                iu += grid_nu
            elif iu >= grid_nu:
                iu -= grid_nu
            xu = 0 if iu == 0 else (grid_nu - iu)
            do_pos = iu < u_half
            do_neg = xu < u_half
            if not (do_pos or do_neg):
                continue

            ku_val = np.float32(0.5) * ku[cu]
            vr_u = np.float32(r_f32 * ku_val)
            vi_u = np.float32(i_f32 * ku_val)
            w_u = np.float32(w_f32 * ku_val)

            for cv in range(supp):
                iv = iv0 + cv
                if iv < 0:
                    iv += nv
                elif iv >= nv:
                    iv -= nv
                kv_val = kv[cv]
                kw = np.float32(w_u * kv_val)
                if kw <= np.float32(0.0):
                    continue
                re_tap = np.float32(vr_u * kv_val)
                im_tap = np.float32(vi_u * kv_val)

                if do_pos:
                    _accum_cell_numba(
                        t, nt_out, nv, u_half, iu, iv,
                        re_tap, im_tap, kw,
                        uv_to_m, has_uv_to_m, sparse_output,
                        out_re, out_im, out_w,
                    )
                if do_neg:
                    xv = 0 if iv == 0 else (nv - iv)
                    _accum_cell_numba(
                        t, nt_out, nv, u_half, xu, xv,
                        re_tap, np.float32(-im_tap), kw,
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
        grid_nu: int,
        nv: int,
        u_half: int,
        du: float,
        dv: float,
        pixsize_x: float,
        pixsize_y: float,
        min_uv_distance_m: float,
        min_valid_channels: int,
        collapse_channels: bool,
        exact_uvw: bool,
        mode_id: int,
        supp: int,
        deg: int,
        poly_coeffs: np.ndarray,
        lshift: float,
        mshift: float,
        nshift: float,
        do_shift: bool,
        flip_u: bool,
        flip_v: bool,
        flip_w: bool,
        use_snapshot_norm: bool,
        norm_div_mode: int,
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
        su = -1.0 if flip_u else 1.0
        sv = -1.0 if flip_v else 1.0
        sw = -1.0 if flip_w else 1.0

        for t_u in nb.prange(nt_out):
            t = np.int64(t_u)
            ku_buf = np.empty(8, dtype=np.float32)
            kv_buf = np.empty(8, dtype=np.float32)
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

            snap_wmax = np.float32(0.0)
            snap_nvalid = 0

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

                    wf = np.float32(sum_w)
                    snap_nvalid += 1
                    if wf > snap_wmax:
                        snap_wmax = wf

                    u_m_t = uvw_3d[bl, 0, t]
                    v_m_t = uvw_3d[bl, 1, t]
                    if mode_id == 2:
                        w_m_t = uvw_3d[bl, 2, t]
                        u_lam = (u_m_t * su) * mean_freq_over_c
                        v_lam = (v_m_t * sv) * mean_freq_over_c
                        w_lam = (w_m_t * sw) * mean_freq_over_c
                        if not (math.isfinite(u_lam) and math.isfinite(v_lam) and math.isfinite(w_lam)):
                            continue
                        _splat_wgridder_sample_numba(
                            t, nt_out, grid_nu, nv, u_half,
                            u_lam, v_lam, w_lam,
                            np.float32(sum_re), np.float32(sum_im), wf,
                            pixsize_x, pixsize_y, supp, deg, poly_coeffs,
                            lshift, mshift, nshift, do_shift,
                            uv_to_m, has_uv_to_m, sparse_output,
                            out_re, out_im, out_w, ku_buf, kv_buf,
                        )
                    else:
                        u_f = (u_m_t * mean_freq_over_c) / du
                        v_f = (v_m_t * mean_freq_over_c) / dv
                        if not (math.isfinite(u_f) and math.isfinite(v_f)) or abs(u_f) > 1.0e15 or abs(v_f) > 1.0e15:
                            continue
                        _splat_sample_numba(
                            t, nt_out, nv, u_half, u_f, v_f,
                            np.float32(sum_re), np.float32(sum_im), wf,
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

                        wf = np.float32(sum_w)
                        snap_nvalid += 1
                        if wf > snap_wmax:
                            snap_wmax = wf

                        if exact_uvw:
                            ts = t + int(sub_delays[s])
                            t_uvw = 0 if ts < 0 else (nt_in - 1 if ts >= nt_in else ts)
                        else:
                            t_uvw = t

                        u_m = uvw_3d[bl, 0, t_uvw]
                        v_m = uvw_3d[bl, 1, t_uvw]
                        freq_over_c = sub_freqs_hz[s] / C_LIGHT
                        if mode_id == 2:
                            w_m = uvw_3d[bl, 2, t_uvw]
                            u_lam = (u_m * su) * freq_over_c
                            v_lam = (v_m * sv) * freq_over_c
                            w_lam = (w_m * sw) * freq_over_c
                            if not (math.isfinite(u_lam) and math.isfinite(v_lam) and math.isfinite(w_lam)):
                                continue
                            _splat_wgridder_sample_numba(
                                t, nt_out, grid_nu, nv, u_half,
                                u_lam, v_lam, w_lam,
                                np.float32(sum_re), np.float32(sum_im), wf,
                                pixsize_x, pixsize_y, supp, deg, poly_coeffs,
                                lshift, mshift, nshift, do_shift,
                                uv_to_m, has_uv_to_m, sparse_output,
                                out_re, out_im, out_w, ku_buf, kv_buf,
                            )
                        else:
                            u_f = (u_m * freq_over_c) / du
                            v_f = (v_m * freq_over_c) / dv
                            if not (math.isfinite(u_f) and math.isfinite(v_f)) or abs(u_f) > 1.0e15 or abs(v_f) > 1.0e15:
                                continue
                            _splat_sample_numba(
                                t, nt_out, nv, u_half, u_f, v_f,
                                np.float32(sum_re), np.float32(sum_im), wf,
                                mode_id, uv_to_m, has_uv_to_m, sparse_output,
                                out_re, out_im, out_w,
                            )

            # 3. Normalise and threshold snapshot t
            snap_inv = np.float32(0.0)
            if use_snapshot_norm and snap_wmax > 0.0 and snap_nvalid > 0:
                if norm_div_mode == 1:
                    denom = float(snap_wmax) * float(snap_nvalid)
                elif norm_div_mode == 2:
                    nsub_norm = 1 if is_collapsed else max(1, nsub_eff)
                    nbl_eff = max(1.0, float(snap_nvalid) / float(nsub_norm))
                    denom = float(snap_wmax) * nbl_eff
                else:
                    denom = float(snap_wmax)
                snap_inv = np.float32(1.0 / denom)

            if sparse_output:
                for m_idx in range(m_active):
                    idx = m_idx * nt_out + t
                    w = out_w[idx]
                    if w > 0.0 and math.isfinite(w):
                        inv_w = snap_inv if use_snapshot_norm else (np.float32(1.0) / w)
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
                        inv_w = snap_inv if use_snapshot_norm else (np.float32(1.0) / w)
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
    grid_nu: int,
    nv: int,
    u_half: int,
    du: float,
    dv: float,
    pixsize_x: float,
    pixsize_y: float,
    min_uv_distance_m: float,
    min_valid_channels: int,
    collapse_channels: bool,
    exact_uvw: bool,
    mode_id: int,
    supp: int,
    deg: int,
    poly_coeffs: np.ndarray,
    lshift: float,
    mshift: float,
    nshift: float,
    do_shift: bool,
    flip_u: bool,
    flip_v: bool,
    flip_w: bool,
    use_snapshot_norm: bool,
    norm_div_mode: int,
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
    w_m_sub = np.empty((nbl, nsub_eff, nt_out), dtype=np.float64)

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
            w_m_sub[:, s, :] = uvw_3d[:, 2, ts]
        else:
            u_m_sub[:, s, :] = uvw_3d[:, 0, :nt_out]
            v_m_sub[:, s, :] = uvw_3d[:, 1, :nt_out]
            w_m_sub[:, s, :] = uvw_3d[:, 2, :nt_out]

    snap_inv_arr = np.zeros(nt_out, dtype=np.float32)
    if use_snapshot_norm:
        nsub_norm = 1 if is_collapsed else max(1, nsub_eff)
        for t_i in range(nt_out):
            if not snap_valid[t_i]:
                continue
            w_t = w_sub[:, :, t_i]
            nval = int(np.count_nonzero(w_t > 0.0))
            if nval > 0:
                wmax = float(np.max(w_t))
                if wmax > 0.0:
                    if norm_div_mode == 1:
                        denom = wmax * float(nval)
                    elif norm_div_mode == 2:
                        nbl_eff = max(1.0, float(nval) / float(nsub_norm))
                        denom = wmax * nbl_eff
                    else:
                        denom = wmax
                    snap_inv_arr[t_i] = np.float32(1.0 / denom)

    freq_over_c = sub_freqs_hz[None, :, None] / C_LIGHT
    su = -1.0 if flip_u else 1.0
    sv = -1.0 if flip_v else 1.0
    sw = -1.0 if flip_w else 1.0

    u_lam_all = (u_m_sub * su) * freq_over_c
    v_lam_all = (v_m_sub * sv) * freq_over_c
    w_lam_all = (w_m_sub * sw) * freq_over_c
    u_f = (u_m_sub * freq_over_c) / du
    v_f = (v_m_sub * freq_over_c) / dv

    if mode_id == 2:
        valid = (
            (w_sub > 0.0)
            & snap_valid[None, None, :]
            & np.isfinite(u_lam_all)
            & np.isfinite(v_lam_all)
            & np.isfinite(w_lam_all)
        )
    else:
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
        u_sel = u_f[valid]
        v_sel = v_f[valid]
        u_cell = np.round(u_sel).astype(np.int64)
        v_cell = np.round(v_sel).astype(np.int64)
        flip = u_cell < 0
        u_cell = np.where(flip, -u_cell, u_cell)
        v_cell = np.where(flip, -v_cell, v_cell)
        im_use = np.where(flip, -im_sel, im_sel)
        v_idx = np.mod(v_cell, nv)
        _accumulate_bins(u_cell, v_idx, re_sel, im_use, w_sel, t_sel)
    elif mode_id == 1:
        u_sel = u_f[valid]
        v_sel = v_f[valid]
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
    else:
        u_lam = u_lam_all[valid].copy()
        v_lam = v_lam_all[valid].copy()
        w_lam = w_lam_all[valid].copy()
        imflip = np.where(w_lam < 0.0, np.float32(-1.0), np.float32(1.0))
        neg_w = w_lam < 0.0
        u_lam = np.where(neg_w, -u_lam, u_lam)
        v_lam = np.where(neg_w, -v_lam, v_lam)
        w_lam = np.where(neg_w, -w_lam, w_lam)

        vr_arr = re_sel.astype(np.float32, copy=True)
        vi_arr = im_sel.astype(np.float32, copy=True)
        if do_shift:
            tmp = imflip.astype(np.float64) * (u_lam * lshift + v_lam * mshift + w_lam * nshift)
            ang = (2.0 * math.pi * (tmp - np.floor(tmp))).astype(np.float32)
            c_ang = np.cos(ang).astype(np.float32)
            s_ang = np.sin(ang).astype(np.float32)
            nvr = (vr_arr * c_ang - vi_arr * s_ang).astype(np.float32)
            nvi = (vr_arr * s_ang + vi_arr * c_ang).astype(np.float32)
            vr_arr = nvr
            vi_arr = nvi
        vi_arr = (vi_arr * imflip).astype(np.float32)

        ushift = float(supp) * (-0.5) + 1.0 + float(grid_nu)
        vshift = float(supp) * (-0.5) + 1.0 + float(nv)
        maxiu0 = (grid_nu + supp) - ((supp + 1) // 2)
        maxiv0 = (nv + supp) - ((supp + 1) // 2)

        u_sc = u_lam * pixsize_x
        u_sc = (u_sc - np.floor(u_sc)) * float(grid_nu)
        iu0 = np.minimum((u_sc + ushift).astype(np.int64) - grid_nu, maxiu0)
        ufrac = (u_sc - iu0.astype(np.float64)).astype(np.float32)

        v_sc = v_lam * pixsize_y
        v_sc = (v_sc - np.floor(v_sc)) * float(nv)
        iv0 = np.minimum((v_sc + vshift).astype(np.int64) - nv, maxiv0)
        vfrac = (v_sc - iv0.astype(np.float64)).astype(np.float32)

        x0 = np.float32(-2.0) * ufrac + np.float32(supp - 1)
        y0 = np.float32(-2.0) * vfrac + np.float32(supp - 1)
        x2 = (x0 * x0).astype(np.float32)
        y2 = (y0 * y0).astype(np.float32)
        lim = (supp + 1) >> 1

        n_pts = len(ufrac)
        ku_all = np.empty((supp, n_pts), dtype=np.float32)
        kv_all = np.empty((supp, n_pts), dtype=np.float32)
        for i in range(lim):
            tx = np.full(n_pts, poly_coeffs[i, 0], dtype=np.float32)
            tx2 = np.full(n_pts, poly_coeffs[i, 1], dtype=np.float32)
            ty = np.full(n_pts, poly_coeffs[i, 0], dtype=np.float32)
            ty2 = np.full(n_pts, poly_coeffs[i, 1], dtype=np.float32)
            for j in range(2, deg, 2):
                c_even = poly_coeffs[i, j]
                c_odd = poly_coeffs[i, j + 1]
                tx = (tx * x2 + c_even).astype(np.float32)
                ty = (ty * y2 + c_even).astype(np.float32)
                tx2 = (tx2 * x2 + c_odd).astype(np.float32)
                ty2 = (ty2 * y2 + c_odd).astype(np.float32)
            px = (tx * x0).astype(np.float32)
            py = (ty * y0).astype(np.float32)
            ku_all[i] = tx2 + px
            kv_all[i] = ty2 + py
            ku_all[supp - 1 - i] = tx2 - px
            kv_all[supp - 1 - i] = ty2 - py

        for cu_off in range(supp):
            iu = np.mod(iu0 + cu_off, grid_nu)
            xu = np.where(iu == 0, 0, grid_nu - iu)
            ku_val = np.float32(0.5) * ku_all[cu_off]
            vr_u = (vr_arr * ku_val).astype(np.float32)
            vi_u = (vi_arr * ku_val).astype(np.float32)
            w_u = (w_sel * ku_val).astype(np.float32)
            for cv_off in range(supp):
                iv = np.mod(iv0 + cv_off, nv)
                xv = np.where(iv == 0, 0, nv - iv)
                kv_val = kv_all[cv_off]
                kw = (w_u * kv_val).astype(np.float32)
                re_tap = (vr_u * kv_val).astype(np.float32)
                im_tap = (vi_u * kv_val).astype(np.float32)
                _accumulate_bins(iu, iv, re_tap, im_tap, kw, t_sel)
                _accumulate_bins(xu, xv, re_tap, -im_tap, kw, t_sel)

    out_re = accum_re.reshape(out_shape)
    out_im = accum_im.reshape(out_shape)
    out_w = accum_w.reshape(out_shape)

    pos_w = (out_w > 0.0) & np.isfinite(out_w)
    if use_snapshot_norm:
        if sparse_output:
            inv_w = np.where(pos_w, snap_inv_arr[None, :], np.float32(0.0))
        else:
            inv_w = np.where(pos_w, snap_inv_arr[:, None, None], np.float32(0.0))
    else:
        inv_w = np.where(pos_w, np.float32(1.0) / np.where(pos_w, out_w, np.float32(1.0)), np.float32(0.0))
    norm_re = out_re * inv_w
    norm_im = out_im * inv_w
    valid_out = pos_w & (inv_w > 0.0) & np.isfinite(norm_re) & np.isfinite(norm_im)
    if uv_threshold > 0.0:
        thr_sq = np.float32(uv_threshold * uv_threshold)
        valid_out &= (norm_re * norm_re + norm_im * norm_im) > thr_sq

    out = np.zeros(out_shape, dtype=np.complex64)
    out.real[valid_out] = norm_re[valid_out]
    out.imag[valid_out] = norm_im[valid_out]
    return out


def _dedisperse_and_wgridder_3d_cpu(
    vis_arr: np.ndarray,
    wgt_arr: np.ndarray,
    uvw_arr: np.ndarray,
    delays_host: np.ndarray,
    sub_offsets: np.ndarray,
    sub_freqs_host: np.ndarray,
    sub_delays_host: np.ndarray,
    uv_to_m_arr: Optional[np.ndarray],
    m_val: int,
    return_sparse: bool,
    eff_nt_out: int,
    eff_grid_nu: int,
    eff_nv: int,
    u_half: int,
    npix_x: int,
    npix_y: int,
    pixsize_x: float,
    pixsize_y: float,
    min_uv: float,
    min_valid_channels: int,
    collapse_channels: bool,
    exact_uvw: bool,
    supp: int,
    deg: int,
    ofactor: float,
    beta_param: float,
    e0: float,
    poly_coeffs: np.ndarray,
    lshift: float,
    mshift: float,
    nm1min: float,
    nm1max: float,
    nshift: float,
    do_shift: bool,
    flip_u: bool,
    flip_v: bool,
    flip_w: bool,
    divide_by_n: bool,
    use_snapshot_norm: bool,
    uv_threshold: float,
) -> np.ndarray:
    """
    Full 3D DUCC w-stacking CPU implementation matching ``ducc0.wgridder.vis2grid(..., do_wgridding=True)``
    to ``< 1e-6`` relative error.
    """
    nbl, nchan, nt_in = vis_arr.shape
    nsub_eff = int(len(sub_freqs_host))
    out_shape = (m_val, eff_nt_out) if return_sparse else (eff_nt_out, eff_nv, u_half)
    out = np.zeros(out_shape, dtype=np.complex64)
    if eff_nt_out == 0 or nbl == 0 or nchan == 0 or (return_sparse and m_val == 0):
        return out

    valid_w = (wgt_arr > 0.0) & np.isfinite(wgt_arr) & np.isfinite(vis_arr.real) & np.isfinite(vis_arr.imag)
    if min_uv > 0.0:
        uv_ok = (
            np.isfinite(uvw_arr[:, 0, :])
            & np.isfinite(uvw_arr[:, 1, :])
            & (np.hypot(uvw_arr[:, 0, :], uvw_arr[:, 1, :]) >= min_uv)
        )
        valid_w &= uv_ok[:, None, :]

    wgt_eff = np.where(valid_w, wgt_arr, np.float32(0.0))
    vis_eff = np.where(valid_w, vis_arr, np.complex64(0.0 + 0.0j))

    raw_valid_chans = np.count_nonzero(np.any(wgt_eff[:, :, :eff_nt_out] > 0.0, axis=0), axis=0)
    req_chans = min(min_valid_channels, nchan) if min_valid_channels > 0 else 1

    t_grid = np.arange(eff_nt_out, dtype=np.int64)
    vis_dedisp = np.zeros((nbl, nchan, eff_nt_out), dtype=np.complex64)
    wgt_dedisp = np.zeros((nbl, nchan, eff_nt_out), dtype=np.float32)
    for c in range(nchan):
        t_src = t_grid + int(delays_host[c])
        ok_t = (t_src >= 0) & (t_src < nt_in)
        if np.any(ok_t):
            vis_dedisp[:, c, ok_t] = vis_eff[:, c, t_src[ok_t]]
            wgt_dedisp[:, c, ok_t] = wgt_eff[:, c, t_src[ok_t]]

    dedisp_valid_chans = np.count_nonzero(np.any(wgt_dedisp > 0.0, axis=0), axis=0)
    if collapse_channels:
        snap_valid = (raw_valid_chans >= req_chans) & (dedisp_valid_chans >= 1)
    else:
        snap_valid = dedisp_valid_chans >= req_chans

    is_collapsed = collapse_channels and (nsub_eff == 1)
    su = -1.0 if flip_u else 1.0
    sv = -1.0 if flip_v else 1.0
    sw = -1.0 if flip_w else 1.0
    dw = 0.5 / float(ofactor) / max(abs(nm1max + nshift), abs(nm1min + nshift))
    xdw = 1.0 / dw

    cor_x, cor_wgtpsi = build_ducc_corfunc(supp, beta_param, e0)
    hnx = int(npix_x) // 2
    hny = int(npix_y) // 2
    ix_arr = np.arange(eff_grid_nu, dtype=np.int64)
    jx_arr = np.arange(eff_nv, dtype=np.int64)
    kx_arr = np.clip(np.where(ix_arr <= eff_grid_nu // 2, ix_arr, ix_arr - eff_grid_nu), -hnx, hnx)
    ky_arr = np.clip(np.where(jx_arr <= eff_nv // 2, jx_arr, jx_arr - eff_nv), -hny, hny)
    x_coord = lshift + kx_arr.astype(np.float64) * pixsize_x
    y_coord = mshift + ky_arr.astype(np.float64) * pixsize_y
    xsq = x_coord[:, None] ** 2
    ysq = y_coord[None, :] ** 2
    tmp_nm1 = 1.0 - xsq - ysq
    nm1_grid = np.where(
        tmp_nm1 >= 0.0,
        (-xsq - ysq) / (np.sqrt(np.maximum(tmp_nm1, 0.0)) + 1.0),
        -np.sqrt(np.maximum(-tmp_nm1, 0.0)) - 1.0,
    )
    cor_vals = eval_ducc_corfunc((nm1_grid + nshift) * dw, cor_x, cor_wgtpsi)
    if divide_by_n:
        cor_vals = np.where(tmp_nm1 >= 0.0, cor_vals / (nm1_grid + 1.0), 0.0)
    tfct_grid = (cor_vals / (float(eff_grid_nu) * float(eff_nv))).astype(np.float32)

    ushift = float(supp) * (-0.5) + 1.0 + float(eff_grid_nu)
    vshift = float(supp) * (-0.5) + 1.0 + float(eff_nv)
    maxiu0 = (eff_grid_nu + supp) - ((supp + 1) // 2)
    maxiv0 = (eff_nv + supp) - ((supp + 1) // 2)
    lim = (supp + 1) >> 1
    half_supp = supp >> 1
    hsupp = (supp + 1) // 2
    lmax = abs(lshift) + 0.5 * float(npix_x) * pixsize_x
    mmax = abs(mshift) + 0.5 * float(npix_y) * pixsize_y
    thr_sq = np.float32(uv_threshold * uv_threshold) if uv_threshold > 0.0 else np.float32(0.0)
    freq_over_c = sub_freqs_host / C_LIGHT

    for t in range(eff_nt_out):
        if not snap_valid[t]:
            continue

        vw_sub = np.empty((nbl, nsub_eff), dtype=np.complex64)
        w_sub = np.empty((nbl, nsub_eff), dtype=np.float32)
        u_m_sub = np.empty((nbl, nsub_eff), dtype=np.float64)
        v_m_sub = np.empty((nbl, nsub_eff), dtype=np.float64)
        w_m_sub = np.empty((nbl, nsub_eff), dtype=np.float64)

        for s in range(nsub_eff):
            c0 = int(sub_offsets[s])
            c1 = int(sub_offsets[s + 1])
            w_slice = wgt_dedisp[:, c0:c1, t].astype(np.float64)
            v_slice = vis_dedisp[:, c0:c1, t].astype(np.complex128)
            w_sub[:, s] = np.sum(w_slice, axis=1).astype(np.float32)
            vw_sub[:, s] = np.sum(v_slice * w_slice, axis=1).astype(np.complex64)
            if (not is_collapsed) and exact_uvw:
                ts = min(max(0, t + int(sub_delays_host[s])), nt_in - 1)
            else:
                ts = t
            u_m_sub[:, s] = uvw_arr[:, 0, ts]
            v_m_sub[:, s] = uvw_arr[:, 1, ts]
            w_m_sub[:, s] = uvw_arr[:, 2, ts]

        wmax = float(np.max(w_sub)) if w_sub.size > 0 else 0.0
        if wmax <= 0.0:
            continue
        wgt_norm = (w_sub / wmax).astype(np.float32)
        n_valid = int(np.count_nonzero(wgt_norm > 0.0))
        if n_valid == 0:
            continue

        vis_scaled = (vw_sub / np.float32(wmax)).astype(np.complex64)
        mask = (wgt_norm > 0.0) & ((vis_scaled.real != 0.0) | (vis_scaled.imag != 0.0))
        if not np.any(mask):
            continue

        u_lam = ((u_m_sub * su) * freq_over_c[None, :])[mask]
        v_lam = ((v_m_sub * sv) * freq_over_c[None, :])[mask]
        w_lam = ((w_m_sub * sw) * freq_over_c[None, :])[mask]
        v_sel = vis_scaled[mask]

        imflip = np.where(w_lam < 0.0, np.float32(-1.0), np.float32(1.0))
        neg_w = w_lam < 0.0
        u_lam = np.where(neg_w, -u_lam, u_lam)
        v_lam = np.where(neg_w, -v_lam, v_lam)
        w_lam = np.where(neg_w, -w_lam, w_lam)

        wmin_d = float(np.min(w_lam))
        wmax_d = float(np.max(w_lam))
        nplanes = int((wmax_d - wmin_d) / dw + supp)
        wmin = (wmin_d + wmax_d) * 0.5 - 0.5 * (nplanes - 1) * dw
        wshift = dw - (0.5 * supp * dw) - wmin

        vr = v_sel.real.astype(np.float32, copy=True)
        vi = v_sel.imag.astype(np.float32, copy=True)
        if do_shift:
            tmp = imflip.astype(np.float64) * (u_lam * lshift + v_lam * mshift + w_lam * nshift)
            ang = (2.0 * math.pi * (tmp - np.floor(tmp))).astype(np.float32)
            c_ang = np.cos(ang).astype(np.float32)
            s_ang = np.sin(ang).astype(np.float32)
            nvr = (vr * c_ang - vi * s_ang).astype(np.float32)
            nvi = (vr * s_ang + vi * c_ang).astype(np.float32)
            vr = nvr
            vi = nvi
        vi = (vi * imflip).astype(np.float32)

        u_sc = u_lam * pixsize_x
        u_sc = (u_sc - np.floor(u_sc)) * float(eff_grid_nu)
        iu0 = np.minimum((u_sc + ushift).astype(np.int64) - eff_grid_nu, maxiu0)
        ufrac = (u_sc - iu0.astype(np.float64)).astype(np.float32)

        v_sc = v_lam * pixsize_y
        v_sc = (v_sc - np.floor(v_sc)) * float(eff_nv)
        iv0 = np.minimum((v_sc + vshift).astype(np.int64) - eff_nv, maxiv0)
        vfrac = (v_sc - iv0.astype(np.float64)).astype(np.float32)

        minplane = np.maximum(0, ((w_lam + wshift) * xdw).astype(np.int64))

        x0 = np.float32(-2.0) * ufrac + np.float32(supp - 1)
        y0 = np.float32(-2.0) * vfrac + np.float32(supp - 1)
        x2 = (x0 * x0).astype(np.float32)
        y2 = (y0 * y0).astype(np.float32)
        n_pts = len(ufrac)
        ku_base = np.empty((supp, n_pts), dtype=np.float32)
        kv_base = np.empty((supp, n_pts), dtype=np.float32)
        for i in range(lim):
            tx = np.full(n_pts, poly_coeffs[i, 0], dtype=np.float32)
            tx2 = np.full(n_pts, poly_coeffs[i, 1], dtype=np.float32)
            ty = np.full(n_pts, poly_coeffs[i, 0], dtype=np.float32)
            ty2 = np.full(n_pts, poly_coeffs[i, 1], dtype=np.float32)
            for j in range(2, deg, 2):
                c_even = poly_coeffs[i, j]
                c_odd = poly_coeffs[i, j + 1]
                tx = (tx * x2 + c_even).astype(np.float32)
                ty = (ty * y2 + c_even).astype(np.float32)
                tx2 = (tx2 * x2 + c_odd).astype(np.float32)
                ty2 = (ty2 * y2 + c_odd).astype(np.float32)
            px = (tx * x0).astype(np.float32)
            py = (ty * y0).astype(np.float32)
            ku_base[i] = tx2 + px
            kv_base[i] = ty2 + py
            ku_base[supp - 1 - i] = tx2 - px
            kv_base[supp - 1 - i] = ty2 - py

        rgrid = np.zeros((eff_grid_nu, eff_nv), dtype=np.float32)
        for p0 in range(nplanes):
            w0 = wmin + p0 * dw
            hit = (minplane <= p0) & ((minplane + supp) > p0)
            if not np.any(hit):
                continue
            nth = (p0 - minplane[hit]).astype(np.int64)
            z_in = (xdw * (w0 - w_lam[hit])).astype(np.float32)
            z0 = (z_in - nth.astype(np.float32)) * np.float32(2.0) + np.float32(supp - 1)
            flip_z = nth >= lim
            nth_eff = np.where(flip_z, supp - 1 - nth, nth)
            z0 = np.where(flip_z, -z0, z0)
            z2 = (z0 * z0).astype(np.float32)
            tz = poly_coeffs[nth_eff, 0].copy()
            tz2 = poly_coeffs[nth_eff, 1].copy()
            for j in range(2, deg, 2):
                tz = (tz * z2 + poly_coeffs[nth_eff, j]).astype(np.float32)
                tz2 = (tz2 * z2 + poly_coeffs[nth_eff, j + 1]).astype(np.float32)
            kw = (tz * z0 + tz2).astype(np.float32)

            plane_grid = np.zeros((eff_grid_nu, eff_nv), dtype=np.complex64)
            iu0_h = iu0[hit]
            iv0_h = iv0[hit]
            vr_h = (vr[hit] * kw).astype(np.float32)
            vi_h = (vi[hit] * kw).astype(np.float32)
            ku_h = ku_base[:, hit]
            kv_h = kv_base[:, hit]

            for cu_off in range(supp):
                iu = np.mod(iu0_h + cu_off, eff_grid_nu)
                vr_u = (vr_h * ku_h[cu_off]).astype(np.float32)
                vi_u = (vi_h * ku_h[cu_off]).astype(np.float32)
                for cv_off in range(supp):
                    iv = np.mod(iv0_h + cv_off, eff_nv)
                    kv_c = kv_h[cv_off]
                    np.add.at(plane_grid.real, (iu, iv), (vr_u * kv_c).astype(np.float32))
                    np.add.at(plane_grid.imag, (iu, iv), (vi_u * kv_c).astype(np.float32))

            img_c = np.fft.ifft2(plane_grid) * (eff_grid_nu * eff_nv)
            img_c = img_c.astype(np.complex64)
            phs = -w0 * (nm1_grid + nshift)
            ang = (2.0 * math.pi * (phs - np.floor(phs))).astype(np.float32)
            ang = np.where(tmp_nm1 > 0.0, ang, np.float32(0.0))
            c_scr = np.cos(ang).astype(np.float32)
            s_scr = np.sin(ang).astype(np.float32)
            rgrid += img_c.real * c_scr - img_c.imag * s_scr

        rgrid *= tfct_grid
        F = np.fft.fft2(rgrid).astype(np.complex64)
        fht = (F.real - F.imag).astype(np.float32)

        seed = np.zeros((eff_grid_nu, eff_nv), dtype=np.uint8)
        uc = np.mod(iu0 + half_supp, eff_grid_nu)
        vc = np.mod(iv0 + half_supp, eff_nv)
        xuc = np.where(uc == 0, 0, eff_grid_nu - uc)
        xvc = np.where(vc == 0, 0, eff_nv - vc)
        seed[uc, vc] = 1
        seed[xuc, xvc] = 1

        du_f = min(eff_grid_nu // 2, int(math.ceil(wmax_d * lmax * eff_grid_nu * pixsize_x)) + hsupp + 3)
        dv_f = min(eff_nv // 2, int(math.ceil(wmax_d * mmax * eff_nv * pixsize_y)) + hsupp + 3)
        tmp_mask = np.zeros((eff_grid_nu, eff_nv), dtype=np.uint8)
        for dv_i in range(-dv_f, dv_f + 1):
            tmp_mask |= np.roll(seed, dv_i, axis=1)
        dilated = np.zeros((eff_grid_nu, eff_nv), dtype=np.uint8)
        for du_i in range(-du_f, du_f + 1):
            dilated |= np.roll(tmp_mask, du_i, axis=0)
        fht[dilated == 0] = 0.0

        u_idx_hp = np.arange(u_half, dtype=np.int64)
        xu_hp = np.where(u_idx_hp == 0, 0, eff_grid_nu - u_idx_hp)
        v_idx_hp = np.arange(eff_nv, dtype=np.int64)
        xv_hp = np.where(v_idx_hp == 0, 0, eff_nv - v_idx_hp)

        a = fht[u_idx_hp[:, None], v_idx_hp[None, :]]
        b = fht[xu_hp[:, None], xv_hp[None, :]]
        snap_scale = np.float32(1.0 / float(n_valid)) if (use_snapshot_norm and divide_by_n) else np.float32(1.0)
        hp_u_v = np.empty((u_half, eff_nv), dtype=np.complex64)
        hp_u_v.real = 0.5 * (a + b) * snap_scale
        hp_u_v.imag = 0.5 * (b - a) * snap_scale
        hp_v_u = hp_u_v.T  # (nv, u_half)

        if thr_sq > 0.0:
            keep = (hp_v_u.real * hp_v_u.real + hp_v_u.imag * hp_v_u.imag) > thr_sq
            hp_v_u = np.where(keep, hp_v_u, np.complex64(0.0 + 0.0j))

        if return_sparse and uv_to_m_arr is not None:
            active_mask = uv_to_m_arr >= 0
            m_ids = uv_to_m_arr[active_mask]
            out[m_ids, t] = hp_v_u[active_mask]
        elif uv_to_m_arr is not None:
            out[t] = np.where(uv_to_m_arr >= 0, hp_v_u, np.complex64(0.0 + 0.0j))
        else:
            out[t] = hp_v_u

    return out


def _dedisperse_and_wgridder_3d_gpu(
    d_vis: Any,
    d_wgt: Any,
    d_uvw: Any,
    d_delays: Any,
    d_raw_valid: Any,
    d_dedisp_valid: Any,
    sub_offsets_host: np.ndarray,
    sub_freqs_host: np.ndarray,
    sub_delays_host: np.ndarray,
    d_uv_to_m: Optional[Any],
    m_val: int,
    return_sparse: bool,
    eff_nt_out: int,
    eff_grid_nu: int,
    eff_nv: int,
    u_half: int,
    npix_x: int,
    npix_y: int,
    pixsize_x: float,
    pixsize_y: float,
    min_uv: float,
    min_valid_channels: int,
    collapse_channels: bool,
    exact_uvw: bool,
    supp: int,
    deg: int,
    ofactor: float,
    beta_param: float,
    e0: float,
    d_poly_coeffs: Any,
    lshift: float,
    mshift: float,
    nm1min: float,
    nm1max: float,
    nshift: float,
    do_shift: bool,
    flip_u: bool,
    flip_v: bool,
    flip_w: bool,
    divide_by_n: bool,
    use_snapshot_norm: bool,
    uv_threshold: float,
) -> Any:
    """
    Full 3D DUCC w-stacking CUDA implementation matching ``ducc0.wgridder.vis2grid(..., do_wgridding=True)``
    to ``< 1e-6`` relative error.
    """
    nbl, nchan, nt_in = (int(d_vis.shape[0]), int(d_vis.shape[1]), int(d_vis.shape[2]))
    nsub_eff = int(len(sub_freqs_host))
    out_shape = (m_val, eff_nt_out) if return_sparse else (eff_nt_out, eff_nv, u_half)
    d_out = cp.zeros(out_shape, dtype=cp.complex64)
    if eff_nt_out == 0 or nbl == 0 or nchan == 0 or (return_sparse and m_val == 0):
        return d_out

    mod = _get_module()
    splat_3d_kernel = mod.get_function("ducc_wgridder_3d_plane_splat_kernel")
    wscreen_kernel = mod.get_function("ducc_wscreen_accumulate_kernel")
    wcorr_kernel = mod.get_function("ducc_apply_w_corrections_rgrid_kernel")
    fft_to_fht_kernel = mod.get_function("ducc_fft_to_fht_kernel")
    dilate_v_kernel = mod.get_function("ducc_fresnel_dilate_v_kernel")
    dilate_u_kernel = mod.get_function("ducc_fresnel_dilate_u_and_mask_kernel")
    fht_to_hp_kernel = mod.get_function("ducc_fht_to_halfplane_kernel")

    cor_x_host, cor_wgtpsi_host = build_ducc_corfunc(supp, beta_param, e0)
    d_cor_x = cp.asarray(cor_x_host, dtype=cp.float64)
    d_cor_wgtpsi = cp.asarray(cor_wgtpsi_host, dtype=cp.float64)
    n_cor = int(len(cor_x_host))

    dw = 0.5 / float(ofactor) / max(abs(nm1max + nshift), abs(nm1min + nshift))
    xdw = 1.0 / dw
    hsupp = (supp + 1) // 2
    lmax = abs(lshift) + 0.5 * float(npix_x) * pixsize_x
    mmax = abs(mshift) + 0.5 * float(npix_y) * pixsize_y

    d_sub_freqs = cp.asarray(sub_freqs_host, dtype=cp.float64)
    d_freq_over_c = d_sub_freqs / C_LIGHT
    sw = -1.0 if flip_w else 1.0

    raw_valid_host = cp.asnumpy(d_raw_valid)
    dedisp_valid_host = cp.asnumpy(d_dedisp_valid)
    req_chans = min(min_valid_channels, nchan) if min_valid_channels > 0 else 1
    is_collapsed = collapse_channels and (nsub_eff == 1)

    d_plane_grid = cp.zeros((eff_grid_nu, eff_nv), dtype=cp.complex64)
    d_rgrid = cp.zeros((eff_grid_nu, eff_nv), dtype=cp.float32)
    d_fht = cp.zeros((eff_grid_nu, eff_nv), dtype=cp.float32)
    d_seed = cp.zeros((eff_grid_nu, eff_nv), dtype=cp.uint8)
    d_tmp_mask = cp.zeros((eff_grid_nu, eff_nv), dtype=cp.uint8)

    gx_splat = _ceil_div(nsub_eff, 32)
    gy_splat = min(_MAX_GRID_Y, _ceil_div(nbl, 8))
    gx_grid = _ceil_div(eff_nv, 32)
    gy_grid = _ceil_div(eff_grid_nu, 8)
    gx_hp = _ceil_div(u_half, 32)
    gy_hp = _ceil_div(eff_nv, 8)
    d_uv_to_m_ptr: Any = d_uv_to_m if d_uv_to_m is not None else 0
    sparse_flag = np.int32(1 if return_sparse else 0)

    delays_cpu = cp.asnumpy(d_delays)
    d_vis_sub = cp.zeros((nbl, nsub_eff), dtype=cp.complex64)
    d_wgt_sub = cp.zeros((nbl, nsub_eff), dtype=cp.float32)
    d_uvw_sub = cp.zeros((nbl, nsub_eff, 3), dtype=cp.float64)

    if min_uv > 0.0:
        d_uv_ok = (
            cp.isfinite(d_uvw[:, 0, :])
            & cp.isfinite(d_uvw[:, 1, :])
            & (cp.hypot(d_uvw[:, 0, :], d_uvw[:, 1, :]) >= min_uv)
        )
    else:
        d_uv_ok = None

    for t in range(eff_nt_out):
        if collapse_channels:
            if raw_valid_host[t] < req_chans or dedisp_valid_host[t] < 1:
                continue
        else:
            if dedisp_valid_host[t] < req_chans:
                continue

        d_vis_sub.fill(0)
        d_wgt_sub.fill(0)
        for s in range(nsub_eff):
            c0 = int(sub_offsets_host[s])
            c1 = int(sub_offsets_host[s + 1])
            sum_vw = cp.zeros(nbl, dtype=cp.complex128)
            sum_w = cp.zeros(nbl, dtype=cp.float64)
            for c in range(c0, c1):
                t_src = t + int(delays_cpu[c])
                if 0 <= t_src < nt_in:
                    vc = d_vis[:, c, t_src]
                    wc = d_wgt[:, c, t_src]
                    ok = (wc > 0.0) & cp.isfinite(wc) & cp.isfinite(vc.real) & cp.isfinite(vc.imag)
                    if d_uv_ok is not None:
                        ok = ok & d_uv_ok[:, t_src]
                    wc_eff = cp.where(ok, wc, cp.float32(0.0)).astype(cp.float64)
                    vc_eff = cp.where(ok, vc, cp.complex64(0.0 + 0.0j)).astype(cp.complex128)
                    sum_w += wc_eff
                    sum_vw += vc_eff * wc_eff
            pos = sum_w > 0.0
            d_wgt_sub[:, s] = sum_w.astype(cp.float32)
            d_vis_sub[:, s] = cp.where(pos, sum_vw / cp.where(pos, sum_w, 1.0), 0.0).astype(cp.complex64)
            if (not is_collapsed) and exact_uvw:
                ts = min(max(0, t + int(sub_delays_host[s])), nt_in - 1)
            else:
                ts = t
            d_uvw_sub[:, s, :] = d_uvw[:, :, ts]

        wmax = float(cp.max(d_wgt_sub).item())
        if wmax <= 0.0:
            continue
        d_wgt_norm = (d_wgt_sub / np.float32(wmax)).astype(cp.float32)
        n_valid = int(cp.count_nonzero(d_wgt_norm > 0.0).item())
        if n_valid == 0:
            continue

        d_active_mask = (d_wgt_norm > 0.0) & ((d_vis_sub.real != 0.0) | (d_vis_sub.imag != 0.0))
        if not bool(cp.any(d_active_mask).item()):
            continue

        d_w_lam_abs = cp.abs((d_uvw_sub[:, :, 2] * sw) * d_freq_over_c[None, :])[d_active_mask]
        wmin_d = float(cp.min(d_w_lam_abs).item())
        wmax_d = float(cp.max(d_w_lam_abs).item())
        nplanes = int((wmax_d - wmin_d) / dw + supp)
        wmin = (wmin_d + wmax_d) * 0.5 - 0.5 * (nplanes - 1) * dw
        wshift = dw - (0.5 * supp * dw) - wmin

        d_rgrid.fill(0.0)
        d_seed.fill(0)

        for p0 in range(nplanes):
            w0 = wmin + p0 * dw
            d_plane_grid.fill(0.0)
            splat_3d_kernel(
                (gx_splat, gy_splat, 1),
                (32, 8, 1),
                (
                    d_vis_sub,
                    d_wgt_norm,
                    d_uvw_sub,
                    d_sub_freqs,
                    d_poly_coeffs,
                    d_plane_grid,
                    d_seed,
                    np.int32(nbl),
                    np.int32(nsub_eff),
                    np.int32(eff_grid_nu),
                    np.int32(eff_nv),
                    np.float64(pixsize_x),
                    np.float64(pixsize_y),
                    np.int32(supp),
                    np.int32(deg),
                    np.float64(lshift),
                    np.float64(mshift),
                    np.float64(nshift),
                    np.int32(1 if do_shift else 0),
                    np.int32(1 if flip_u else 0),
                    np.int32(1 if flip_v else 0),
                    np.int32(1 if flip_w else 0),
                    np.int32(p0),
                    np.float64(w0),
                    np.float64(xdw),
                    np.float64(wshift),
                    np.int32(1 if p0 == 0 else 0),
                ),
            )
            d_img_c = cp.fft.ifft2(d_plane_grid) * np.float32(eff_grid_nu * eff_nv)
            d_img_c = cp.ascontiguousarray(d_img_c, dtype=cp.complex64)
            wscreen_kernel(
                (gx_grid, gy_grid, 1),
                (32, 8, 1),
                (
                    d_img_c,
                    d_rgrid,
                    np.int32(eff_grid_nu),
                    np.int32(eff_nv),
                    np.int32(npix_x),
                    np.int32(npix_y),
                    np.float64(pixsize_x),
                    np.float64(pixsize_y),
                    np.float64(lshift),
                    np.float64(mshift),
                    np.float64(nshift),
                    np.float64(w0),
                ),
            )

        wcorr_kernel(
            (gx_grid, gy_grid, 1),
            (32, 8, 1),
            (
                d_rgrid,
                d_cor_x,
                d_cor_wgtpsi,
                np.int32(n_cor),
                np.int32(eff_grid_nu),
                np.int32(eff_nv),
                np.int32(npix_x),
                np.int32(npix_y),
                np.float64(pixsize_x),
                np.float64(pixsize_y),
                np.float64(lshift),
                np.float64(mshift),
                np.float64(nshift),
                np.float64(dw),
                np.int32(1 if divide_by_n else 0),
            ),
        )

        d_fft_r = cp.ascontiguousarray(cp.fft.fft2(d_rgrid), dtype=cp.complex64)
        fft_to_fht_kernel(
            (gx_grid, gy_grid, 1),
            (32, 8, 1),
            (
                d_fft_r,
                d_fht,
                np.int32(eff_grid_nu),
                np.int32(eff_nv),
            ),
        )

        du_f = min(eff_grid_nu // 2, int(math.ceil(wmax_d * lmax * eff_grid_nu * pixsize_x)) + hsupp + 3)
        dv_f = min(eff_nv // 2, int(math.ceil(wmax_d * mmax * eff_nv * pixsize_y)) + hsupp + 3)
        dilate_v_kernel(
            (gx_grid, gy_grid, 1),
            (32, 8, 1),
            (
                d_seed,
                d_tmp_mask,
                np.int32(eff_grid_nu),
                np.int32(eff_nv),
                np.int32(dv_f),
            ),
        )
        dilate_u_kernel(
            (gx_grid, gy_grid, 1),
            (32, 8, 1),
            (
                d_tmp_mask,
                d_fht,
                np.int32(eff_grid_nu),
                np.int32(eff_nv),
                np.int32(du_f),
            ),
        )

        snap_scale = np.float32(1.0 / float(n_valid)) if (use_snapshot_norm and divide_by_n) else np.float32(1.0)
        fht_to_hp_kernel(
            (gx_hp, gy_hp, 1),
            (32, 8, 1),
            (
                d_fht,
                d_uv_to_m_ptr,
                d_out,
                np.int32(t),
                np.int32(eff_nt_out),
                np.int32(eff_grid_nu),
                np.int32(eff_nv),
                np.int32(u_half),
                snap_scale,
                np.float32(uv_threshold),
                sparse_flag,
            ),
        )

    return d_out


def _resolve_normalise_mode(mode_id: int, normalise_mode: Optional[str]) -> Tuple[bool, int]:
    r"""
    Resolve weight normalisation mode into ``(use_snapshot_norm, norm_div_mode)``.
    - For ``nearest`` (``mode_id=0``) and ``bilinear`` (``mode_id=1``), ``"auto"`` uses
      per-cell weight normalisation (``use_snapshot_norm=False``).
    - For ``wgridder`` (``mode_id=2``), ``"auto"`` uses per-baseline snapshot natural
      weighting (``use_snapshot_norm=True, norm_div_mode=2``, ``denom = wgt_max * nsub_eff``)
      which preserves the exact DUCC polynomial kernel shape $k_u(\delta u) k_v(\delta v)$
      and natural baseline weighting while keeping per-baseline UV cell amplitudes at
      $\mathcal{O}(|V|)$ so ``uv_threshold`` prunes only the outer kernel wings.
    - Pass ``normalise_mode="vis2grid"`` (or ``"snapshot"``) for DUCC ``vis2grid`` per-snapshot
      natural weighting across all ``(baseline, subband)`` pairs (``denom = wgt_max * n_valid``).
    """
    if normalise_mode is None or str(normalise_mode).strip().lower() == "auto":
        if mode_id == 2:
            return True, 2
        return False, 0
    nm = str(normalise_mode).strip().lower()
    if nm in ("cell", "uniform"):
        return False, 0
    if nm in ("baseline", "subband"):
        return True, 2
    if nm in ("vis2grid", "snapshot", "natural"):
        return True, 1
    if nm in ("wmax", "unnormalised", "none"):
        return True, 0
    raise ValueError(
        f"Unsupported normalise_mode='{normalise_mode}'; expected 'auto', 'baseline', 'cell', 'vis2grid', or 'wmax'."
    )


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
    uv_threshold: float = 0.0,
    uv_to_m: Optional[np.ndarray] = None,
    m_active: Optional[int] = None,
    return_sparse: bool = False,
    flat_indices: Optional[np.ndarray] = None,
    use_numba: bool = True,
) -> np.ndarray:
    """
    CPU fallback for fused visibility dedispersion and direct-to-grid Hermitian
    half-plane rasterisation (supporting ``mode="nearest"``/``"pillbox"``,
    ``mode="bilinear"``, and ``mode="wgridder"``).

    Mathematical operations, rounding (IEEE-754 half-to-even), Hermitian folding,
    DUCC piecewise Chebyshev polynomial evaluation, $w$-phase preconditioning,
    subband reduction, valid-channel tracking, and weight normalisation match
    :func:`dedisperse_and_rasterise_chunk_gpu` to within ``< 1e-6`` relative error.
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

    mode_id = _normalise_mode(mode)
    geom = compute_raster_grid_geometry(
        npix_x=npix_x,
        npix_y=npix_y,
        pix_rad=pix_rad,
        pixsize_arcsec=pixsize_arcsec,
        grid_nu=grid_nu,
        nv=nv,
        mode=mode,
        epsilon=epsilon,
        do_wgridding=do_wgridding,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        oversample=oversample,
    )
    eff_grid_nu = int(geom["grid_nu"])
    eff_nv = int(geom["nv"])
    u_half = int(geom["u_half"])
    eff_pix_rad = float(geom["pix_rad"])
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

    use_snapshot_norm, norm_div_mode = _resolve_normalise_mode(mode_id, normalise_mode)
    eff_norm_div_mode = int(norm_div_mode) if bool(divide_by_n) else 0
    eff_divide_by_n = bool(divide_by_n) and (eff_norm_div_mode != 0)

    if mode_id == 2:
        kpar = select_ducc_kernel_params(
            epsilon=float(epsilon),
            do_wgridding=bool(do_wgridding),
            sigma_min=float(sigma_min),
            sigma_max=float(sigma_max),
            supp=supp,
        )
        eff_supp = int(kpar["W"])
        eff_deg = int(kpar["deg"])
        eff_ofactor = float(kpar["ofactor"])
        eff_beta = float(kpar["beta"])
        eff_e0 = float(kpar["e0"])
        poly_coeffs = build_ducc_polynomial_coeffs(eff_supp, eff_beta, eff_e0)
        wpar = compute_ducc_w_params(
            npix_x=int(npix_x),
            npix_y=int(npix_y),
            pixsize_x=eff_pix_rad,
            pixsize_y=eff_pix_rad,
            center_x=float(center_x),
            center_y=float(center_y),
            allow_nshift=bool(allow_nshift),
            apply_w_phase=bool(apply_w_phase),
            do_wgridding=bool(do_wgridding),
        )
        lshift = float(wpar["lshift"])
        mshift = float(wpar["mshift"])
        nm1min = float(wpar["nm1min"])
        nm1max = float(wpar["nm1max"])
        nshift = float(wpar["nshift"])
        do_shift = bool(wpar["do_shift"])

        if do_wgridding:
            return _dedisperse_and_wgridder_3d_cpu(
                vis_arr=vis_arr,
                wgt_arr=wgt_arr,
                uvw_arr=uvw_arr,
                delays_host=delays_host,
                sub_offsets=sub_offsets,
                sub_freqs_host=sub_freqs_host,
                sub_delays_host=sub_delays_host,
                uv_to_m_arr=uv_to_m_arr,
                m_val=m_val,
                return_sparse=bool(return_sparse),
                eff_nt_out=eff_nt_out,
                eff_grid_nu=eff_grid_nu,
                eff_nv=eff_nv,
                u_half=u_half,
                npix_x=int(npix_x),
                npix_y=int(npix_y),
                pixsize_x=eff_pix_rad,
                pixsize_y=eff_pix_rad,
                min_uv=min_uv,
                min_valid_channels=int(min_valid_channels),
                collapse_channels=bool(collapse_channels),
                exact_uvw=bool(exact_uvw),
                supp=eff_supp,
                deg=eff_deg,
                ofactor=eff_ofactor,
                beta_param=eff_beta,
                e0=eff_e0,
                poly_coeffs=poly_coeffs,
                lshift=lshift,
                mshift=mshift,
                nm1min=nm1min,
                nm1max=nm1max,
                nshift=nshift,
                do_shift=do_shift,
                flip_u=bool(flip_u),
                flip_v=bool(flip_v),
                flip_w=bool(flip_w),
                divide_by_n=eff_divide_by_n,
                use_snapshot_norm=use_snapshot_norm,
                uv_threshold=thr,
            )
    else:
        eff_supp = 4
        eff_deg = 7
        poly_coeffs = np.zeros((2, 8), dtype=np.float32)
        lshift = 0.0
        mshift = 0.0
        nshift = 0.0
        do_shift = False

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
            eff_grid_nu,
            eff_nv,
            u_half,
            du,
            dv,
            eff_pix_rad,
            eff_pix_rad,
            min_uv,
            int(min_valid_channels),
            bool(collapse_channels),
            bool(exact_uvw),
            mode_id,
            eff_supp,
            eff_deg,
            poly_coeffs,
            lshift,
            mshift,
            nshift,
            do_shift,
            bool(flip_u),
            bool(flip_v),
            bool(flip_w),
            use_snapshot_norm,
            eff_norm_div_mode,
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
        grid_nu=eff_grid_nu,
        nv=eff_nv,
        u_half=u_half,
        du=du,
        dv=dv,
        pixsize_x=eff_pix_rad,
        pixsize_y=eff_pix_rad,
        min_uv_distance_m=min_uv,
        min_valid_channels=int(min_valid_channels),
        collapse_channels=bool(collapse_channels),
        exact_uvw=bool(exact_uvw),
        mode_id=mode_id,
        supp=eff_supp,
        deg=eff_deg,
        poly_coeffs=poly_coeffs,
        lshift=lshift,
        mshift=mshift,
        nshift=nshift,
        do_shift=do_shift,
        flip_u=bool(flip_u),
        flip_v=bool(flip_v),
        flip_w=bool(flip_w),
        use_snapshot_norm=use_snapshot_norm,
        norm_div_mode=eff_norm_div_mode,
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
    uv_threshold: float = 0.0,
    uv_to_m: Optional[Any] = None,
    m_active: Optional[int] = None,
    return_sparse: bool = False,
    flat_indices: Optional[Any] = None,
    return_host: bool = True,
) -> Any:
    """
    CUDA-accelerated fused visibility dedispersion and direct-to-grid Hermitian
    half-plane rasterisation (supporting ``mode="nearest"``/``"pillbox"``,
    ``mode="bilinear"``, and ``mode="wgridder"``).

    Inputs may be NumPy arrays or CuPy device arrays. Output is returned either
    as a dense ``(nt_out, nv, u_half)`` ``complex64`` grid (ready for
    ``fastducc.ragged_gpu.pack_uv_chunk_to_ragged_gpu``) or, when
    ``return_sparse=True``, as a compact ``(M, nt_out)`` ``complex64`` matrix
    (ready for ``fastducc.ragged_gpu.pack_chunk_to_ragged_gpu``).
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

    mode_id = _normalise_mode(mode)
    geom = compute_raster_grid_geometry(
        npix_x=npix_x,
        npix_y=npix_y,
        pix_rad=pix_rad,
        pixsize_arcsec=pixsize_arcsec,
        grid_nu=grid_nu,
        nv=nv,
        mode=mode,
        epsilon=epsilon,
        do_wgridding=do_wgridding,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        oversample=oversample,
    )
    eff_grid_nu = int(geom["grid_nu"])
    eff_nv = int(geom["nv"])
    u_half = int(geom["u_half"])
    eff_pix_rad = float(geom["pix_rad"])
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

    use_snapshot_norm, norm_div_mode = _resolve_normalise_mode(mode_id, normalise_mode)
    eff_norm_div_mode = int(norm_div_mode) if bool(divide_by_n) else 0
    eff_divide_by_n = bool(divide_by_n) and (eff_norm_div_mode != 0)

    if mode_id == 2:
        kpar = select_ducc_kernel_params(
            epsilon=float(epsilon),
            do_wgridding=bool(do_wgridding),
            sigma_min=float(sigma_min),
            sigma_max=float(sigma_max),
            supp=supp,
        )
        eff_supp = int(kpar["W"])
        eff_deg = int(kpar["deg"])
        eff_ofactor = float(kpar["ofactor"])
        eff_beta = float(kpar["beta"])
        eff_e0 = float(kpar["e0"])
        poly_coeffs_host = build_ducc_polynomial_coeffs(eff_supp, eff_beta, eff_e0)
        d_poly_coeffs = cp.asarray(poly_coeffs_host, dtype=cp.float32)

        wpar = compute_ducc_w_params(
            npix_x=int(npix_x),
            npix_y=int(npix_y),
            pixsize_x=eff_pix_rad,
            pixsize_y=eff_pix_rad,
            center_x=float(center_x),
            center_y=float(center_y),
            allow_nshift=bool(allow_nshift),
            apply_w_phase=bool(apply_w_phase),
            do_wgridding=bool(do_wgridding),
        )
        lshift = float(wpar["lshift"])
        mshift = float(wpar["mshift"])
        nm1min = float(wpar["nm1min"])
        nm1max = float(wpar["nm1max"])
        nshift = float(wpar["nshift"])
        do_shift = bool(wpar["do_shift"])

        if do_wgridding:
            d_out = _dedisperse_and_wgridder_3d_gpu(
                d_vis=d_vis,
                d_wgt=d_wgt,
                d_uvw=d_uvw,
                d_delays=d_delays,
                d_raw_valid=d_raw_valid,
                d_dedisp_valid=d_dedisp_valid,
                sub_offsets_host=sub_offsets_host,
                sub_freqs_host=sub_freqs_host,
                sub_delays_host=sub_delays_host,
                d_uv_to_m=d_uv_to_m,
                m_val=m_val,
                return_sparse=bool(return_sparse),
                eff_nt_out=eff_nt_out,
                eff_grid_nu=eff_grid_nu,
                eff_nv=eff_nv,
                u_half=u_half,
                npix_x=int(npix_x),
                npix_y=int(npix_y),
                pixsize_x=eff_pix_rad,
                pixsize_y=eff_pix_rad,
                min_uv=min_uv,
                min_valid_channels=int(min_valid_channels),
                collapse_channels=bool(collapse_channels),
                exact_uvw=bool(exact_uvw),
                supp=eff_supp,
                deg=eff_deg,
                ofactor=eff_ofactor,
                beta_param=eff_beta,
                e0=eff_e0,
                d_poly_coeffs=d_poly_coeffs,
                lshift=lshift,
                mshift=mshift,
                nm1min=nm1min,
                nm1max=nm1max,
                nshift=nshift,
                do_shift=do_shift,
                flip_u=bool(flip_u),
                flip_v=bool(flip_v),
                flip_w=bool(flip_w),
                divide_by_n=eff_divide_by_n,
                use_snapshot_norm=use_snapshot_norm,
                uv_threshold=thr,
            )
            if return_host:
                return cp.asnumpy(d_out)
            return d_out

    d_grid_vis = cp.zeros(out_shape, dtype=cp.complex64)
    d_grid_wgt = cp.zeros(out_shape, dtype=cp.float32)

    d_uv_to_m_ptr: Any = d_uv_to_m if d_uv_to_m is not None else 0
    sparse_flag = np.int32(1 if return_sparse else 0)
    gx_raster = _ceil_div(eff_nt_out, 32)
    gy_raster = min(_MAX_GRID_Y, _ceil_div(nbl, 8))
    d_sub_offsets = cp.asarray(sub_offsets_host, dtype=cp.int32)

    if mode_id == 2:
        if collapse_channels and nsub_eff == 1:
            wg_collapsed_kernel = mod.get_function("dedisperse_and_wgridder_collapsed_kernel")
            wg_collapsed_kernel(
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
                    d_poly_coeffs,
                    d_grid_vis,
                    d_grid_wgt,
                    np.int32(nbl),
                    np.int32(nchan),
                    np.int32(nt_in),
                    np.int32(eff_nt_out),
                    np.int32(eff_grid_nu),
                    np.int32(eff_nv),
                    np.int32(u_half),
                    np.float64(eff_pix_rad),
                    np.float64(eff_pix_rad),
                    np.float64(sub_freqs_host[0]),
                    np.float64(min_uv),
                    np.int32(min_valid_channels),
                    np.int32(eff_supp),
                    np.int32(eff_deg),
                    np.float64(lshift),
                    np.float64(mshift),
                    np.float64(nshift),
                    np.int32(1 if do_shift else 0),
                    np.int32(1 if flip_u else 0),
                    np.int32(1 if flip_v else 0),
                    np.int32(1 if flip_w else 0),
                    sparse_flag,
                ),
            )
        else:
            d_sub_freqs = cp.asarray(sub_freqs_host, dtype=cp.float64)
            d_sub_delays = cp.asarray(sub_delays_host, dtype=cp.int32)
            wg_subband_kernel = mod.get_function("dedisperse_and_wgridder_subband_kernel")
            wg_subband_kernel(
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
                    d_poly_coeffs,
                    d_grid_vis,
                    d_grid_wgt,
                    np.int32(nbl),
                    np.int32(nchan),
                    np.int32(nsub_eff),
                    np.int32(nt_in),
                    np.int32(eff_nt_out),
                    np.int32(eff_grid_nu),
                    np.int32(eff_nv),
                    np.int32(u_half),
                    np.float64(eff_pix_rad),
                    np.float64(eff_pix_rad),
                    np.float64(min_uv),
                    np.int32(min_valid_channels),
                    np.int32(1 if collapse_channels else 0),
                    np.int32(1 if exact_uvw else 0),
                    np.int32(eff_supp),
                    np.int32(eff_deg),
                    np.float64(lshift),
                    np.float64(mshift),
                    np.float64(nshift),
                    np.int32(1 if do_shift else 0),
                    np.int32(1 if flip_u else 0),
                    np.int32(1 if flip_v else 0),
                    np.int32(1 if flip_w else 0),
                    sparse_flag,
                ),
            )
    else:
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

    if use_snapshot_norm:
        d_snap_inv = cp.zeros(eff_nt_out, dtype=cp.float32)
        snap_norm_kernel = mod.get_function("compute_snapshot_vis2grid_norm_kernel")
        snap_norm_kernel(
            (eff_nt_out, 1, 1),
            (32, 8, 1),
            (
                d_vis,
                d_wgt,
                d_uvw,
                d_delays,
                d_sub_offsets,
                d_raw_valid,
                d_dedisp_valid,
                d_snap_inv,
                np.int32(nbl),
                np.int32(nchan),
                np.int32(nsub_eff),
                np.int32(nt_in),
                np.int32(eff_nt_out),
                np.float64(min_uv),
                np.int32(min_valid_channels),
                np.int32(1 if collapse_channels else 0),
                np.int32(eff_norm_div_mode),
            ),
        )
        if return_sparse:
            norm_sparse_snap = mod.get_function("normalise_sparse_grid_snapshot_kernel")
            gy_norm = min(_MAX_GRID_Y, _ceil_div(m_val, 8))
            norm_sparse_snap(
                (gx_raster, gy_norm, 1),
                (32, 8, 1),
                (
                    d_grid_vis,
                    d_grid_wgt,
                    d_snap_inv,
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
            norm_snap_kernel = mod.get_function("normalise_raster_grid_snapshot_kernel")
            spatial = eff_nv * u_half
            gx_norm = _ceil_div(spatial, 32)
            gy_norm = min(_MAX_GRID_Y, _ceil_div(eff_nt_out, 8))
            norm_snap_kernel(
                (gx_norm, gy_norm, 1),
                (32, 8, 1),
                (
                    d_grid_vis,
                    d_grid_wgt,
                    d_snap_inv,
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
    else:
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

    del d_grid_wgt
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
    "get_ducc_available_kernels",
    "select_ducc_kernel_params",
    "build_ducc_polynomial_coeffs",
    "build_ducc_corfunc",
    "eval_ducc_corfunc",
    "compute_ducc_w_params",
    "stage_ms_rows_cpu",
    "stage_ms_rows_gpu",
    "stage_ms_rows",
    "dedisperse_and_rasterise_chunk_cpu",
    "dedisperse_and_rasterise_chunk_gpu",
    "dedisperse_and_rasterise_chunk",
    "dedisperse_and_rasterise_rows",
    "dedisperse_rasterise_to_ragged",
]
