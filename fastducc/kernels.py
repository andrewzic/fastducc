import argparse
import glob
import os
import shutil
import sys
from typing import Iterable, Tuple, List, Dict, Any, Optional
from dataclasses import dataclass

from numba import njit, prange
import numpy as np
import math
from tqdm import tqdm

import astropy.constants as const
import astropy.units as u
from astropy.visualization.wcsaxes import WCSAxes
from astropy.wcs import WCS
from astropy.io import fits
from astropy.table import Table, vstack

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from scipy.ndimage import maximum_filter

from casacore.tables import table
try:
    import ducc0
except Exception as e:
    raise RuntimeError('ducc0 is required') from e

# ============================
# NUMBA-ACCELERATED KERNELS
# ============================

@njit
def _moving_sum_from_csum(csum: np.ndarray, w: int) -> np.ndarray:
    """
    csum: (T+1, Ny, Nx) cumulative sum along time.
    returns S: (T-w+1, Ny, Nx) windowed sum for width w.
    """
    T_plus, Ny, Nx = csum.shape
    T = T_plus - 1
    T_eff = T - w + 1
    out = np.empty((T_eff, Ny, Nx), dtype=csum.dtype)
    #out = csum[w:T_eff+w, :, :] - csum[:T_eff, :, :]
    for t0 in range(T_eff):
         out[t0, :, :] = csum[t0 + w, :, :] - csum[t0, :, :]
    return out


@njit
def _median_1d(arr: np.ndarray, n: int) -> float:
    """
    Median over arr[:n] (n valid values). arr is modified (sorted).
    Assumes n >= 1.
    """
    tmp = arr[:n].copy()
    tmp.sort()
    mid = n // 2
    if (n % 2) == 1:
        return tmp[mid]
    else:
        return 0.5 * (tmp[mid - 1] + tmp[mid])


@njit
def _mad_std_2d_per_time(S: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Robust spatial std per time using 1.4826 * MAD across (y, x).
    S: (T_eff, Ny, Nx)
    mask: (Ny, Nx) boolean; True means valid pixel
    Returns: sigma_w: (T_eff,)
    """
    T_eff, Ny, Nx = S.shape
    out = np.empty(T_eff, dtype=S.dtype)

    # Precompute linear indices of masked pixels
    idxs = np.empty(Ny * Nx, np.int64)
    nmask = 0
    for y in range(Ny):
        for x in range(Nx):
            if mask[y, x]:
                idxs[nmask] = y * Nx + x
                nmask += 1

    # Scratch buffers sized to masked pixels
    vals = np.empty(nmask, dtype=S.dtype)
    devs = np.empty(nmask, dtype=S.dtype)

    for t in range(T_eff):
        # Collect finite masked values for this time slice
        n = 0
        for k in range(nmask):
            lin = idxs[k]
            y = lin // Nx
            x = lin % Nx
            v = S[t, y, x]
            if np.isfinite(v):
                vals[n] = v
                n += 1

        if n == 0:
            out[t] = 1.0  # or np.nan
            continue

        # Median of values (_median_1d must operate on vals[:n])
        med = _median_1d(vals, n)

        # Absolute deviations; no need for isfinite check
        for i in range(n):
            devs[i] = abs(vals[i] - med)

        # MAD and robust std
        mad = _median_1d(devs, n)
        std = 1.4826 * mad

        out[t] = std if std > 0.0 and np.isfinite(std) else 1.0  # or np.nan

    return out



@njit
def _mad_sigma_2d(std_map: np.ndarray, valid_mask: np.ndarray) -> float:
    """
    Robust spatial sigma from MAD across the 2-D std_map on valid pixels.
    Returns 1.4826 * median(|x - median(x)|). Fallback to 1.0 if no valid data or MAD=0.
    """
    Ny, Nx = std_map.shape
    # Collect valid finite values into a buffer
    vals = np.empty(Ny * Nx, dtype=np.float64)
    n = 0
    for y in range(Ny):
        for x in range(Nx):
            if valid_mask[y, x]:
                v = std_map[y, x]
                if np.isfinite(v):
                    vals[n] = v
                    n += 1
    if n == 0:
        return 1.0

    med = _median_1d(vals, n)

    # Compute absolute deviations and take their median
    dev = np.empty(n, dtype=np.float64)
    m = 0
    for i in range(n):
        d = abs(vals[i] - med)
        if np.isfinite(d):
            dev[m] = d
            m += 1
    if m == 0:
        return 1.0

    mad = _median_1d(dev, m)
    sig = 1.4826 * mad
    if (not np.isfinite(sig)) or (sig <= 0.0):
        return 1.0
    return sig


@njit
def _clipped_rms_sigma_2d(std_map: np.ndarray,
                          valid_mask: np.ndarray,
                          clip_sigma: float = 3.0,
                          max_iter: int = 5) -> float:
    """
    Iterative sigma-clipped RMS across the 2-D std_map on valid pixels.
    Initialization uses median/MAD; iteratively clips values within clip_sigma*std.
    Fallback to 1.0 if no valid data or std <= 0.
    """
    Ny, Nx = std_map.shape
    # Collect valid finite values
    vals = np.empty(Ny * Nx, dtype=np.float64)
    n = 0
    for y in range(Ny):
        for x in range(Nx):
            if valid_mask[y, x]:
                v = std_map[y, x]
                if np.isfinite(v):
                    vals[n] = v
                    n += 1
    if n == 0:
        return 1.0

    # Robust init via median/MAD
    med = _median_1d(vals, n)

    dev = np.empty(n, dtype=np.float64)
    m = 0
    for i in range(n):
        d = abs(vals[i] - med)
        if np.isfinite(d):
            dev[m] = d
            m += 1
    mad = _median_1d(dev, m) if m > 0 else 0.0
    std = 1.4826 * mad if mad > 0.0 else 0.0
    mu = med

    # Iterative clipping: keep values within clip_sigma * std
    for _ in range(max_iter):
        if (not np.isfinite(std)) or (std <= 0.0):
            break
        keep = np.empty(n, dtype=np.float64)
        k = 0
        thr = clip_sigma * std
        for i in range(n):
            if abs(vals[i] - mu) <= thr:
                keep[k] = vals[i]
                k += 1
        if k == 0:
            break

        # mean
        s = 0.0
        for i in range(k):
            s += keep[i]
        mu = s / k

        # std (ddof=1)
        ss = 0.0
        for i in range(k):
            d = keep[i] - mu
            ss += d * d
        std = np.sqrt(ss / max(1, k - 1))

    if (not np.isfinite(std)) or (std <= 0.0):
        return 1.0
    return std

@njit
def _clipped_rms_2d_per_time(S: np.ndarray, mask: np.ndarray, sigma: float, max_iter: int = 5) -> np.ndarray:
    """
    Iterative sigma-clipped RMS per time across (y, x).

    Parameters
    ----------
    S : np.ndarray
        Array of shape (T_eff, Ny, Nx) containing image cube.
    mask : np.ndarray
        Boolean array of shape (Ny, Nx) selecting valid spatial pixels.
    sigma : float
        Clipping threshold multiplier (e.g., 3.0 for 3-sigma).
    max_iter : int
        Maximum number of clipping iterations.

    Returns
    -------
    np.ndarray
        Per-time-slice RMS estimates (shape: (T_eff,)).
    """
    T_eff, Ny, Nx = S.shape
    out = np.empty(T_eff, dtype=S.dtype)

    # Precompute linear indices of masked pixels once
    idxs = np.empty(Ny * Nx, np.int64)
    nmask = 0
    for y in range(Ny):
        for x in range(Nx):
            if mask[y, x]:
                idxs[nmask] = y * Nx + x
                nmask += 1
    # Scratch buffers reused for every time slice
    vals = np.empty(nmask, dtype=S.dtype)   # original pixel values
    devs = np.empty(nmask, dtype=S.dtype)   # absolute deviations from median
    for t in range(T_eff):
        # Gather finite values for this time slice
        n = 0
        for k in range(nmask):
            lin = idxs[k]
            y = lin // Nx
            x = lin % Nx
            v = S[t, y, x]
            if np.isfinite(v):
                vals[n] = v
                n += 1
        if n == 0:
            out[t] = 1.0  # or np.nan
            continue
        # robust init via median / MAD (requires _median_1d helper)
        med = _median_1d(vals, n)
        # compute absolute deviations without destroying original vals
        for i in range(n):
            devs[i] = abs(vals[i] - med)

        mad = _median_1d(devs, n)  # all devs are finite if vals are finite
        std = 1.4826 * mad
        mu = med

        # If initial robust std failed, bail out
        if (not np.isfinite(std)) or std <= 0.0:
            out[t] = 1.0  # or np.nan
            continue

        # Iterative sigma clipping
        m = n
        for _ in range(max_iter):
            thresh = sigma * std
            keep = 0
            # in-place filter on vals
            for i in range(m):
                v = vals[i]
                if abs(v - mu) <= thresh:
                    vals[keep] = v
                    keep += 1

            if keep == 0 or keep == m:
                # converged (no change), or nothing left
                break

            # recompute mean and sample std (ddof=1) on kept values
            s = 0.0
            for i in range(keep):
                s += vals[i]
            mu = s / keep

            ss = 0.0
            for i in range(keep):
                d = vals[i] - mu
                ss += d * d
            std = np.sqrt(ss / max(1, keep - 1))
            m = keep
            
        out[t] = std if (np.isfinite(std) and std > 0.0) else 1.0  # or np.nan
    return out

@njit(parallel=True)
def _temporal_std_snr(data: np.ndarray, csum: np.ndarray, csum2: np.ndarray, w: int) -> np.ndarray:
    """
    Per-pixel temporal variance over window width w, SNR computed as s / (std * sqrt(w)).
    data: (T, Ny, Nx) float64
    csum: (T+1, Ny, Nx)
    csum2: (T+1, Ny, Nx)
    returns snr: (T-w+1, Ny, Nx)
    """
    T, Ny, Nx = data.shape
    T_eff = T - w + 1
    snr = np.zeros((T_eff, Ny, Nx), dtype=np.float64)
    for t0 in prange(T_eff):
        # windowed sums
        s = csum[t0 + w, :, :] - csum[t0, :, :]
        s2 = csum2[t0 + w, :, :] - csum2[t0, :, :]
        # mean & var
        mean = s / w
        var = s2 / w - mean * mean
        # clip negatives
        for y in range(Ny):
            for x in range(Nx):
                v = var[y, x]
                if v < 0.0:
                    v = 0.0
                std = np.sqrt(v)
                denom = std * np.sqrt(w)
                snr[t0, y, x] = (s[y, x] / denom) if denom > 0 else 0.0
    return snr


@njit
def _max_filter_2d(src: np.ndarray, radius: int) -> np.ndarray:
    """
    Simple 2D max filter with square window (size 2*radius+1).
    Pads by nearest behavior (clamped indices).
    """
    Ny, Nx = src.shape
    k = 2 * radius + 1
    out = np.empty((Ny, Nx), dtype=src.dtype)
    for y in range(Ny):
        for x in range(Nx):
            y0 = max(0, y - radius)
            y1 = min(Ny - 1, y + radius)
            x0 = max(0, x - radius)
            x1 = min(Nx - 1, x + radius)
            m = -np.inf
            for yy in range(y0, y1 + 1):
                for xx in range(x0, x1 + 1):
                    v = src[yy, xx]
                    if v > m:
                        m = v
            out[y, x] = m
    return out




def _compute_boxcar_1d(lc_full: np.ndarray, w: int) -> Tuple[np.ndarray, int]:
    """
    Fast boxcar smoothing (mean) of a 1D array using cumulative sums.

    Parameters
    ----------
    lc_full : (T,) ndarray
        Full-resolution light curve (float64 recommended).
    w : int
        Window width (samples), must be >=1.

    Returns
    -------
    lc_sm : (T_eff,) ndarray
        Boxcar-smoothed light curve (mean over each window).
    T_eff : int
        Number of smoothed frames (T - w + 1).
    """
    T = lc_full.shape[0]
    w = max(1, int(w))
    if w > T:
        return np.empty((0,), dtype=np.float64), 0
    csum = np.zeros(T + 1, dtype=np.float64)
    csum[1:] = np.cumsum(lc_full)
    lc_sm = (csum[w:] - csum[:-w]) / float(w)  # length T - w + 1
    return lc_sm, (T - w + 1)

###
# welford std algorithm
@njit(inline='always')
def _welford_update(c, m, M2, x):
    """
    One online update step: given aggregate (c, m, M2) and new sample x,
    return updated (c, m, M2).
    """
    c += 1
    delta = x - m
    m += delta / c
    M2 += delta * (x - m)
    return c, m, M2


@njit(parallel=True)
def welford_update_cube(count, mean, M2, cube, ignore_nan=True):
    """
    Update running per-pixel aggregates using all samples in `cube` (T, Ny, Nx).
    In-place updates on `count`, `mean`, `M2`.
    """
    T, Ny, Nx = cube.shape
    for y in prange(Ny):
        for x in range(Nx):
            c = count[y, x]
            m = mean[y, x]
            m2 = M2[y, x]
            for t in range(T):
                val = np.float64(cube[t, y, x])
                if ignore_nan and not np.isfinite(val):
                    continue
                c, m, m2 = _welford_update(c, m, m2, val)
            count[y, x] = c
            mean[y, x]  = m
            M2[y, x]    = m2


@njit(parallel=True)
def welford_finalise_std(count, M2, ddof=1):
    """
    Compute per-pixel std from aggregates: std = sqrt(M2 / (count - ddof)).
    Returns 2-D array (Ny, Nx). If count <= ddof, returns NaN.
    """
    Ny, Nx = count.shape
    out = np.empty((Ny, Nx), dtype=np.float64)
    for y in prange(Ny):
        for x in range(Nx):
            denom = count[y, x] - ddof
            out[y, x] = np.sqrt(M2[y, x] / denom) if denom > 0 else np.nan
    return out
