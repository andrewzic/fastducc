# fastducc/wgridded_nufft.py
"""
High-Performance W-Stacked 3D NUFFT Engine for Interferometric Periodicity Searches.

Transforms non-coplanar radio visibilities (u, v, w, t) -> (l, m, f) while correcting
wide-field non-coplanar w-phase errors:
    phi_w = 2*pi * w * (sqrt(1 - l^2 - m^2) - 1).

Key Features:
- W-Stacking decomposition: partitions the non-coplanar baseline height w into Nw discrete
  planes, applies a 1D Kaiser-Bessel convolution kernel along w, executes ducc0.nufft.nu2u
  per plane, and rotates by the 2D spatial w-screen:
      S_k(l, m) = exp(2*pi*i * w_k * (sqrt(1 - l^2 - m^2) - 1))
- Numba JIT acceleration:
  1. prepare_visibilities_and_coords: fast multi-channel coordinate extraction & DM time shifts.
  2. filter_and_weight_w_plane: parallel Kaiser-Bessel window evaluation & plane selection.
  3. accumulate_w_screen_inplace: zero-allocation in-place 2D w-screen phase rotation and 3D accumulation.
- Seamless integration with fastducc.nufft_periodicity.
"""

from __future__ import annotations

import math
import logging
from typing import Optional, Tuple
import numpy as np

try:
    import numba
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    prange = range

try:
    import ducc0
    from ducc0 import nufft as ducc_nufft
    from ducc0 import wgridder as ducc_wgridder
    HAS_DUCC = True
except ImportError:
    ducc0 = None
    ducc_nufft = None
    ducc_wgridder = None
    HAS_DUCC = False

from fastducc import constants

C_LIGHT = constants.c
logger = logging.getLogger(__name__)


# =============================================================================
# NUMBA ACCELERATED KERNELS
# =============================================================================

@njit(fastmath=True, inline="always")
def _wrap_pi_numba(x: float) -> float:
    """Wrap angle in radians to [-pi, pi)."""
    two_pi = 2.0 * math.pi
    return (x + math.pi) % two_pi - math.pi


@njit(parallel=True, fastmath=True)
def prepare_visibilities_and_coords(
    uvw_m: np.ndarray,      # (nrows, 3) float64
    times: np.ndarray,      # (nrows,) float64
    vis: np.ndarray,        # (nrows, nchan) complex64
    flags: np.ndarray,      # (nrows, nchan) bool
    freqs_hz: np.ndarray,   # (nchan,) float64
    tau_dm: np.ndarray,     # (nchan,) float64
    dl: float,             # float64
    df: float,             # float64
    c_light: float,        # float64
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-accelerated multi-channel coordinate generation & DM time shift.
    Prepares contiguous buffers for 3D NUFFT without Python object overhead.
    """
    nrows, nchan = vis.shape
    counts = np.zeros(nchan, dtype=np.int64)
    for ic in range(nchan):
        c = 0
        for ir in range(nrows):
            if not flags[ir, ic]:
                c += 1
        counts[ic] = c

    total_valid = np.sum(counts)
    offsets = np.zeros(nchan, dtype=np.int64)
    for ic in range(1, nchan):
        offsets[ic] = offsets[ic - 1] + counts[ic - 1]

    out_pts = np.empty(total_valid, dtype=np.complex64)
    out_coords = np.empty((total_valid, 3), dtype=np.float64)
    out_w_lam = np.empty(total_valid, dtype=np.float64)

    two_pi = 2.0 * math.pi

    for ic in prange(nchan):
        nu = freqs_hz[ic]
        lam = c_light / nu
        inv_lam = 1.0 / lam
        t_shift = tau_dm[ic]

        idx = offsets[ic]
        for ir in range(nrows):
            if not flags[ir, ic]:
                u_m = uvw_m[ir, 0]
                v_m = uvw_m[ir, 1]
                w_m = uvw_m[ir, 2]
                t_val = times[ir] - t_shift

                u_lam_val = u_m * inv_lam
                v_lam_val = v_m * inv_lam
                w_lam_val = w_m * inv_lam

                om_u = _wrap_pi_numba(two_pi * u_lam_val * dl)
                om_v = _wrap_pi_numba(two_pi * v_lam_val * dl)
                om_f = _wrap_pi_numba(-two_pi * t_val * df)

                out_pts[idx] = vis[ir, ic]
                out_coords[idx, 0] = om_u
                out_coords[idx, 1] = om_v
                out_coords[idx, 2] = om_f
                out_w_lam[idx] = w_lam_val
                idx += 1

    return out_pts, out_coords, out_w_lam


@njit(parallel=True, fastmath=True)
def filter_and_weight_w_plane(
    all_pts: np.ndarray,        # (nvis,) complex64
    all_coords: np.ndarray,     # (nvis, 3) float64
    all_w_lam: np.ndarray,      # (nvis,) float64
    w_curr: float,              # float64
    dw: float,                  # float64
    supp_w: float,              # float64
    beta: float,                # float64
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba-accelerated parallel Kaiser-Bessel weighting and w-plane filtering.
    """
    nvis = len(all_w_lam)
    half_supp = 0.5 * supp_w
    inv_dw = 1.0 / dw
    norm_fac = 2.0 / supp_w

    mask = np.empty(nvis, dtype=np.uint8)
    for i in prange(nvis):
        w_dist = (all_w_lam[i] - w_curr) * inv_dw
        if abs(w_dist) <= half_supp:
            mask[i] = 1
        else:
            mask[i] = 0

    count = 0
    for i in range(nvis):
        if mask[i]:
            count += 1

    plane_pts = np.empty(count, dtype=np.complex64)
    plane_coords = np.empty((count, 3), dtype=np.float64)

    out_idx = 0
    for i in range(nvis):
        if mask[i]:
            w_dist = (all_w_lam[i] - w_curr) * inv_dw
            x_norm = w_dist * norm_fac
            val = math.exp(beta * (math.sqrt(max(0.0, 1.0 - x_norm * x_norm)) - 1.0))

            plane_pts[out_idx] = complex(all_pts[i].real * val, all_pts[i].imag * val)
            plane_coords[out_idx, 0] = all_coords[i, 0]
            plane_coords[out_idx, 1] = all_coords[i, 1]
            plane_coords[out_idx, 2] = all_coords[i, 2]
            out_idx += 1

    return plane_pts, plane_coords


@njit(parallel=True, fastmath=True)
def accumulate_w_screen_inplace(
    accum_cube: np.ndarray,     # (npix, npix, nkeep) complex64
    plane_buf: np.ndarray,      # (npix, npix, Nf) complex64
    nm1_screen: np.ndarray,     # (npix, npix) float64
    w_curr: float,              # float64
    kmin: int,                  # int
    nkeep: int,                 # int
) -> None:
    """
    In-place Numba SIMD phase screen rotation and accumulation.
    Eliminates all temporary 3D array allocations.
    """
    npix_x, npix_y, _ = accum_cube.shape
    two_pi_w = 2.0 * math.pi * w_curr

    for ix in prange(npix_x):
        for iy in range(npix_y):
            phase = two_pi_w * nm1_screen[ix, iy]
            cos_p = math.cos(phase)
            sin_p = math.sin(phase)

            for ik in range(nkeep):
                src_val = plane_buf[ix, iy, kmin + ik]
                re_part = src_val.real * cos_p - src_val.imag * sin_p
                im_part = src_val.real * sin_p + src_val.imag * cos_p

                accum_cube[ix, iy, ik] = complex(
                    accum_cube[ix, iy, ik].real + re_part,
                    accum_cube[ix, iy, ik].imag + im_part
                )


@njit(parallel=True, fastmath=True)
def compute_nm1_screen(npix: int, pixsize_rad: float) -> np.ndarray:
    """
    Computes nm1(x, y) = sqrt(1 - x^2 - y^2) - 1 arranged in DUCC fft_order:
    Mode x < npix//2 corresponds to index x; mode x >= npix//2 corresponds to x - npix.
    """
    screen = np.empty((npix, npix), dtype=np.float64)
    half = npix // 2
    for ix in prange(npix):
        mx = (ix if ix < half else ix - npix) * pixsize_rad
        x2 = mx * mx
        for iy in range(npix):
            my = (iy if iy < half else iy - npix) * pixsize_rad
            y2 = my * my
            r2 = x2 + y2
            # Stably evaluate sqrt(1 - r2) - 1
            if r2 < 1.0:
                screen[ix, iy] = -r2 / (math.sqrt(1.0 - r2) + 1.0)
            else:
                screen[ix, iy] = -1.0
    return screen


# =============================================================================
# GEOMETRY & ENGINE INTERFACES
# =============================================================================

def compute_w_plane_geometry(
    npix: int,
    pixsize_rad: float,
    w_min: float,
    w_max: float,
    supp_w: int = 4,
    ofactor: float = 1.5,
) -> Tuple[float, int, float]:
    """
    Computes (dw, nplanes, w0) for w-stacking based on field of view.
    """
    r_max = (npix / 2.0) * math.sqrt(2.0) * pixsize_rad
    nm1_max = 1.0 - math.sqrt(max(0.0, 1.0 - r_max**2))
    nm1_max = max(nm1_max, 1e-12)
    dw = 1.0 / (2.0 * ofactor * nm1_max)
    w_span = max(0.0, w_max - w_min)
    nplanes = int(math.ceil(w_span / dw)) + supp_w
    w0 = (w_min + w_max) / 2.0 - 0.5 * (nplanes - 1) * dw
    return dw, nplanes, w0


def kb_kernel_eval(x_dist: np.ndarray, supp: int) -> np.ndarray:
    """Kaiser-Bessel / Exponential Sempione window kernel."""
    beta = 2.3 * supp
    x_norm = 2.0 * x_dist / float(supp)
    val = np.zeros_like(x_norm, dtype=np.float32)
    mask = np.abs(x_norm) <= 1.0
    val[mask] = np.exp(beta * (np.sqrt(1.0 - x_norm[mask] ** 2) - 1.0))
    return val


def wstacked_3d_nufft_block(
    uvw_m: np.ndarray,          # (nrows, 3)
    times: np.ndarray,          # (nrows,)
    vis: np.ndarray,            # (nrows, nchan) complex64
    flags: np.ndarray,          # (nrows, nchan) bool
    freqs_hz: np.ndarray,       # (nchan,)
    tau_dm: np.ndarray,         # (nchan,)
    npix: int,                  # e.g. 384
    pixsize_rad: float,         # rad/pix
    Nf: int,                    # spin FFT length
    dt: float,                  # sampling interval
    df: float,                  # frequency bin width
    kmin: int,                  # index on positive side where f >= fmin
    nkeep: int,                 # number of spin-frequency bins to keep
    supp_w: int = 4,
    ofactor: float = 1.5,
    epsilon: float = 1e-5,
    nthreads: int = 0,
) -> np.ndarray:
    """
    Full Numba-accelerated W-Stacked 3D NUFFT for a visibility block.
    Grids all channels and baselines simultaneously into a complex cube
    of shape (npix, npix, nkeep).
    """
    if not HAS_DUCC:
        raise ImportError("ducc0 is required for wstacked_3d_nufft_block")

    # 1. Numba-accelerated data extraction & coordinate preprocessing
    all_pts, all_coords, all_w_lam = prepare_visibilities_and_coords(
        uvw_m=uvw_m,
        times=times,
        vis=vis,
        flags=flags,
        freqs_hz=freqs_hz,
        tau_dm=tau_dm,
        dl=pixsize_rad,
        df=df,
        c_light=C_LIGHT,
    )

    if len(all_pts) == 0:
        return np.zeros((npix, npix, nkeep), dtype=np.complex64)

    # 2. W-plane geometry
    w_min = float(np.min(all_w_lam))
    w_max = float(np.max(all_w_lam))
    dw, nplanes, w0 = compute_w_plane_geometry(
        npix=npix,
        pixsize_rad=pixsize_rad,
        w_min=w_min,
        w_max=w_max,
        supp_w=supp_w,
        ofactor=ofactor,
    )
    beta = 2.3 * supp_w

    # 3. Precompute 2D w-screen coordinate table (nm1_screen)
    nm1_screen = compute_nm1_screen(npix, pixsize_rad)

    # 4. Allocate streaming buffers
    accum_cube = np.zeros((npix, npix, nkeep), dtype=np.complex64)
    plane_buf = np.zeros((npix, npix, Nf), dtype=np.complex64)

    # 5. Stream through W-planes
    for pl in range(nplanes):
        w_curr = w0 + pl * dw
        plane_pts, plane_coords = filter_and_weight_w_plane(
            all_pts=all_pts,
            all_coords=all_coords,
            all_w_lam=all_w_lam,
            w_curr=w_curr,
            dw=dw,
            supp_w=float(supp_w),
            beta=beta,
        )

        if len(plane_pts) == 0:
            continue

        plane_buf.fill(0.0)
        ducc_nufft.nu2u(
            points=plane_pts,
            coord=plane_coords,
            forward=False,
            epsilon=max(float(epsilon), 1e-5),
            nthreads=nthreads,
            out=plane_buf,
            fft_order=True,
        )

        # In-place Numba phase-screen multiplication & accumulation into accum_cube
        accumulate_w_screen_inplace(
            accum_cube=accum_cube,
            plane_buf=plane_buf,
            nm1_screen=nm1_screen,
            w_curr=w_curr,
            kmin=kmin,
            nkeep=nkeep,
        )

    return accum_cube


def wstacked_3d_nufft(
    u_lam: np.ndarray,
    v_lam: np.ndarray,
    w_lam: np.ndarray,
    t_sec: np.ndarray,
    vis: np.ndarray,
    weights: Optional[np.ndarray] = None,
    npix: int = 384,
    pixsize_rad: float = 1.0666e-4,
    Nf: int = 1024,
    dt: float = 0.11,
    epsilon: float = 1e-5,
    supp_w: int = 4,
    nthreads: int = 0,
) -> np.ndarray:
    """
    Convenience function for flat un-gridded arrays: (u, v, w, t) -> (npix, npix, Nf // 2).
    """
    if not HAS_DUCC:
        raise ImportError("ducc0 is required for wstacked_3d_nufft")

    df = 1.0 / (Nf * dt)
    w_min = float(np.min(w_lam))
    w_max = float(np.max(w_lam))
    dw, nplanes, w0 = compute_w_plane_geometry(npix, pixsize_rad, w_min, w_max, supp_w=supp_w)
    beta = 2.3 * supp_w

    def wrap_pi(x):
        return (x + math.pi) % (2.0 * math.pi) - math.pi

    omega_u = wrap_pi(2.0 * math.pi * u_lam * pixsize_rad)
    omega_v = wrap_pi(2.0 * math.pi * v_lam * pixsize_rad)
    omega_f = wrap_pi(-2.0 * math.pi * t_sec * df)
    coords_3d = np.stack([omega_u, omega_v, omega_f], axis=1).astype(np.float64)

    nm1_screen = compute_nm1_screen(npix, pixsize_rad)

    if weights is not None:
        weighted_vis = (vis * weights).astype(np.complex64)
    else:
        weighted_vis = vis.astype(np.complex64)

    accum_cube = np.zeros((npix, npix, Nf // 2), dtype=np.complex64)
    plane_buf = np.zeros((npix, npix, Nf), dtype=np.complex64)

    for pl in range(nplanes):
        w_curr = w0 + pl * dw
        plane_pts, plane_coords = filter_and_weight_w_plane(
            all_pts=weighted_vis,
            all_coords=coords_3d,
            all_w_lam=w_lam,
            w_curr=w_curr,
            dw=dw,
            supp_w=float(supp_w),
            beta=beta,
        )

        if len(plane_pts) == 0:
            continue

        plane_buf.fill(0.0)
        ducc_nufft.nu2u(
            points=plane_pts,
            coord=plane_coords,
            forward=False,
            epsilon=max(float(epsilon), 1e-5),
            nthreads=nthreads,
            out=plane_buf,
            fft_order=True,
        )

        accumulate_w_screen_inplace(
            accum_cube=accum_cube,
            plane_buf=plane_buf,
            nm1_screen=nm1_screen,
            w_curr=w_curr,
            kmin=0,
            nkeep=Nf // 2,
        )

    return accum_cube


def snapshot_wgridder_fft(
    uvw_track: np.ndarray,      # (nbl, 3, nt) in meters
    vis_track: np.ndarray,      # (nbl, nt) complex
    wgt_track: Optional[np.ndarray] = None, # (nbl, nt) float
    freq_hz: float = 1.4e9,
    npix: int = 384,
    pixsize_rad: float = 1.0666e-4,
    dt: float = 0.11,
    epsilon: float = 1e-5,
    nthreads: int = 0,
) -> np.ndarray:
    """
    Concept B: Snapshot 2D W-Gridder + 1D Temporal FFT.
    Transforms visibility tracks into a spin-frequency cube.
    """
    if not HAS_DUCC:
        raise ImportError("ducc0 is required for snapshot_wgridder_fft")

    nbl, nt = vis_track.shape
    freq_arr = np.array([freq_hz], dtype=np.float64)
    img_cube = np.zeros((nt, npix, npix), dtype=np.float32)

    for it in range(nt):
        uvw_it = np.ascontiguousarray(uvw_track[:, :, it], dtype=np.float64)
        vis_it = np.ascontiguousarray(vis_track[:, it : it + 1], dtype=np.complex64)
        wgt_it = (
            np.ascontiguousarray(wgt_track[:, it : it + 1], dtype=np.float32)
            if wgt_track is not None
            else None
        )

        dirty = ducc_wgridder.vis2dirty(
            uvw=uvw_it,
            freq=freq_arr,
            vis=vis_it,
            wgt=wgt_it,
            npix_x=npix,
            npix_y=npix,
            pixsize_x=pixsize_rad,
            pixsize_y=pixsize_rad,
            epsilon=epsilon,
            do_wgridding=True,
            nthreads=nthreads,
        )
        img_cube[it] = dirty

    fft_cube = np.fft.rfft(img_cube, axis=0)
    return fft_cube
