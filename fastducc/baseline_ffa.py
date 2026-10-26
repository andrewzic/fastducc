# -*- coding: utf-8 -*-
"""
fastducc.baseline_ffa: Baseline-Domain Periodicity Search (FFA) & DUCC0 wgridder Imaging.

Integrates riptide's C++ Fast Folding Algorithm (FFA) engine with fastducc's
baseline-based data model and ducc0.wgridder imaging pipeline.

Key Capabilities:
1. Baseline-Domain FFA:
   Folds individual baseline time series (or frequency-resolved pseudo-baselines)
   into (period, phase) profiles without prior UV gridding, preserving phase coherence.
2. Full-Observation & Segmented Two-Pass FFA Imaging:
   Supports both direct full-observation baseline-folded imaging and time-segmented FFA
   where Earth-rotation UV tracks are approximately stationary per segment, Fourier
   phase-aligned to reference epoch t=0, and synthesized in wgridder.
3. Memory-Efficient Streaming Architecture:
   Streams period trials and phase bins directly to wgridder and spatial SNR evaluation,
   evaluating candidates on-the-fly without allocating massive 4D image cubes.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import ducc0
except ImportError:
    ducc0 = None

try:
    import riptide.libcpp as libcpp
    from riptide.ffautils import generate_width_trials, plan_ffa
except ImportError:
    libcpp = None
    generate_width_trials = None
    plan_ffa = None

from fastducc.constants import c as c_light
from fastducc import ms_utils, imaging, dedisp
from fastducc import wcs as ducc_wcs

logger = logging.getLogger(__name__)


@dataclass
class BaselineFFACandidate:
    """Detection candidate from baseline FFA search."""
    period: float
    snr: float
    x: int
    y: int
    width_bins: int
    phase_bin: int
    ra_deg: Optional[float] = None
    dec_deg: Optional[float] = None
    dm: float = 0.0
    cutout: Optional[np.ndarray] = None


@dataclass
class BaselineFFAResult:
    """Container for baseline FFA periodicity search results."""
    periods: np.ndarray
    snr_periodogram: np.ndarray
    best_snr_plane: np.ndarray
    candidates: List[BaselineFFACandidate] = field(default_factory=list)
    best_candidate: Optional[BaselineFFACandidate] = None
    best_dirty_cube: Optional[np.ndarray] = None  # (nbins, ny, nx) if requested
    wcs: Optional[Any] = None
    plan: Optional[List[Dict[str, Any]]] = None


def prepare_baseline_series(
    vis_3d: np.ndarray,
    wgt_3d: np.ndarray,
    uvw_3d: np.ndarray,
    chan_freq: np.ndarray,
    dt: float,
    *,
    dm: float = 0.0,
    collapse_channels: bool = False,
    nsubbands: int = 1,
    align_to: str = "fmax",
    exact_uvw: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Format raw/chunk visibility arrays into baseline time series ready for FFA folding.

    Parameters
    ----------
    vis_3d : np.ndarray
        Shape (nbl, nchan, nt), complex64.
    wgt_3d : np.ndarray
        Shape (nbl, nchan, nt), float32.
    uvw_3d : np.ndarray
        Shape (nbl, 3, nt), float64 (meters).
    chan_freq : np.ndarray
        Channel frequencies in Hz.
    dt : float
        Sampling interval in seconds.
    dm : float
        Dispersion measure trial in pc/cm^3.
    collapse_channels : bool
        If True, collapse frequency channels into an effective single-channel visibility series.
        If False (default), flatten (nbl, nchan) into pseudo-baselines with exact per-channel coordinates.
    nsubbands : int
        Number of subbands when collapsing channels (default 1).
    align_to : str
        Reference frequency alignment ('fmax' or 'fmin').
    exact_uvw : bool
        If True (default) and collapse_channels is False, compute exact dimensionless wavelength coordinates
        accounting for dispersion delay baseline migration.

    Returns
    -------
    vis_2d : np.ndarray
        Shape (N_series, nt), complex64.
    wgt_2d : np.ndarray
        Shape (N_series, nt), float32.
    uvw_track : np.ndarray
        Shape (N_series, 3, nt), float64.
    eff_freq : np.ndarray
        1D array of frequency values for wgridder.
    dt : float
        Sampling interval in seconds.
    """
    nbl, nchan, nt = vis_3d.shape

    # 1. Dedispersion if DM != 0
    engine = None
    if dm != 0.0 or collapse_channels:
        plan = dedisp.DedispersionPlan(chan_freq, dt, dm_min=dm, dm_max=dm)
        engine = dedisp.BruteForceDedisperser(plan, dm, align_to=align_to)
        vis_dedisp, wgt_dedisp, eff_freq = engine.dedisperse_chunk(
            vis_3d, wgt_3d, iblock=0, collapse_channels=collapse_channels,
            nsubbands=nsubbands, use_history=False
        )
    else:
        vis_dedisp = vis_3d
        wgt_dedisp = wgt_3d
        eff_freq = chan_freq

    # 2. Format baselines vs pseudo-baselines
    if collapse_channels and nsubbands == 1:
        # Full collapse to 1 channel
        vis_2d = np.ascontiguousarray(vis_dedisp[:, 0, :], dtype=np.complex64)
        wgt_2d = np.ascontiguousarray(wgt_dedisp[:, 0, :], dtype=np.float32)
        uvw_track = uvw_3d  # (nbl, 3, nt) in meters
        freq_out = np.asarray(eff_freq, dtype=np.float64)
    else:
        # Pseudo-baseline expansion:
        # Either N_sub subbands (collapse_channels=True, nsubbands > 1)
        # or N_chan channels (collapse_channels=False)
        n_freqs = vis_dedisp.shape[1]
        freqs = np.asarray(eff_freq, dtype=np.float64)  # (n_freqs,)
        freq_scale = freqs[None, :, None, None] / c_light  # (1, n_freqs, 1, 1)

        if exact_uvw and (dm != 0.0):
            delays = dedisp.get_dm_delays(freqs, dm, dt, align_to=align_to)
            t_phys = np.clip(np.arange(nt)[None, :] + delays[:, None], 0, nt - 1)  # (n_freqs, nt)
            uvw_lambda = uvw_3d[:, :, t_phys].transpose(0, 2, 1, 3) * freq_scale
        else:
            uvw_lambda = uvw_3d[:, None, :, :] * freq_scale

        uvw_track = uvw_lambda.reshape(nbl * n_freqs, 3, nt)
        vis_2d = np.ascontiguousarray(vis_dedisp.reshape(nbl * n_freqs, nt), dtype=np.complex64)
        wgt_2d = np.ascontiguousarray(wgt_dedisp.reshape(nbl * n_freqs, nt), dtype=np.float32)
        freq_out = np.array([c_light], dtype=np.float64)

    return vis_2d, wgt_2d, uvw_track, freq_out, dt


def shift_profile_phase_fourier(profile: np.ndarray, delta_phase: np.ndarray) -> np.ndarray:
    """
    Apply fractional circular phase shifts to folded profiles using Fourier phase rotation.

    Parameters
    ----------
    profile : np.ndarray
        Shape (N_periods, nbins) or (M, N_periods, nbins), complex64.
    delta_phase : np.ndarray
        Shape (N_periods,), fractional phase shift in range [0, 1).

    Returns
    -------
    shifted_profile : np.ndarray
        Same shape as profile, complex64.
    """
    nbins = profile.shape[-1]
    spec = np.fft.fft(profile, axis=-1)
    k = np.fft.fftfreq(nbins)  # integer / nbins frequencies

    if profile.ndim == 2:
        phase_rot = np.exp(-2j * np.pi * delta_phase[:, None] * (k * nbins)[None, :])
    elif profile.ndim == 3:
        phase_rot = np.exp(-2j * np.pi * delta_phase[None, :, None] * (k * nbins)[None, None, :])
    else:
        raise ValueError(f"Expected 2D or 3D profile array, got shape {profile.shape}")

    shifted_spec = spec * phase_rot.astype(np.complex64)
    return np.fft.ifft(shifted_spec, axis=-1).astype(np.complex64)


def fold_baselines_full(
    vis_2d: np.ndarray,
    wgt_2d: np.ndarray,
    uvw_track: np.ndarray,
    base_period_bins: int,
    tsamp: float,
    rows_eval: int = 0,
    downsample_factor: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full-Observation Baseline-Domain Fast Folding Algorithm (FFA).

    Folds baseline or pseudo-baseline visibility time series across the entire observation,
    associating folded visibilities with their time-averaged UVW baseline coordinates.

    Parameters
    ----------
    vis_2d : np.ndarray
        Shape (N_series, nt), complex64 baseline time series.
    wgt_2d : np.ndarray
        Shape (N_series, nt), float32 visibility weights.
    uvw_track : np.ndarray
        Shape (N_series, 3, nt), float64 baseline coordinates (meters or wavelengths).
    base_period_bins : int
        Number of phase bins at base period.
    tsamp : float
        Sampling interval in seconds.
    rows_eval : int
        Number of FFA transform rows to evaluate (0 for full).
    downsample_factor : float
        Time downsampling factor for this base period block.

    Returns
    -------
    uvw_eff : np.ndarray
        Shape (N_series, 3), effective time-averaged coordinates.
    wgt_eff : np.ndarray
        Shape (N_series, 1), effective time-averaged weights.
    folded_vis : np.ndarray
        Shape (N_series, N_periods, base_period_bins), complex64 folded visibilities.
    periods : np.ndarray
        1D array of evaluated trial periods in seconds.
    """
    if libcpp is None:
        raise ImportError("riptide.libcpp is required for baseline FFA folding.")

    n_series, nt = vis_2d.shape

    # 1. Compute time-averaged UVW coordinates weighted by visibility weights
    wsum = np.sum(wgt_2d, axis=1, keepdims=True)  # (N_series, 1)
    has_w = (wsum > 0.0)

    # Weighted mean UVW: (N_series, 3)
    weighted_uvw = np.sum(uvw_track * wgt_2d[:, None, :], axis=2)  # (N_series, 3)
    uvw_eff = np.where(has_w, weighted_uvw / np.where(has_w, wsum, 1.0), np.mean(uvw_track, axis=2))
    wgt_eff = np.where(has_w, wsum / nt, 0.0).astype(np.float32)

    # 2. Run C++ FFA transform on all baseline time series
    effective_tau = float(tsamp) * float(downsample_factor)
    folded_vis, periods = libcpp.vis_ffa_transform_basep(
        vis_2d,
        int(base_period_bins),
        effective_tau,
        int(rows_eval),
        float(downsample_factor),
    )

    return uvw_eff, wgt_eff, folded_vis, periods


# Alias for backward compatibility
fold_baselines_option_b = fold_baselines_full


def fold_baselines_segmented(
    vis_2d: np.ndarray,
    wgt_2d: np.ndarray,
    uvw_track: np.ndarray,
    base_period_bins: int,
    tsamp: float,
    segment_duration_s: float,
    rows_eval: int = 0,
    downsample_factor: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Segmented Baseline-Domain Fast Folding Algorithm (FFA) with Fourier Phase Alignment.

    Partitions the observation into short time segments of length segment_duration_s where
    Earth-rotation UV tracks are approximately stationary. Folds each segment independently,
    circularly shifts folded profiles to align to reference epoch t=0 via Fourier rotation,
    and concatenates effective segment baselines for coherent spatial synthesis in wgridder.

    Parameters
    ----------
    vis_2d : np.ndarray
        Shape (N_series, nt), complex64.
    wgt_2d : np.ndarray
        Shape (N_series, nt), float32.
    uvw_track : np.ndarray
        Shape (N_series, 3, nt), float64.
    base_period_bins : int
        Number of phase bins.
    tsamp : float
        Sampling interval in seconds.
    segment_duration_s : float
        Stationary segment duration in seconds.
    rows_eval : int
        FFA rows to evaluate.
    downsample_factor : float
        Time downsampling factor.

    Returns
    -------
    uvw_eff : np.ndarray
        Shape (S * N_series, 3), stacked segment UVW coordinates.
    wgt_eff : np.ndarray
        Shape (S * N_series, 1), stacked weights.
    folded_vis : np.ndarray
        Shape (S * N_series, N_periods, base_period_bins), phase-aligned folded visibilities.
    periods : np.ndarray
        1D array of trial periods in seconds.
    """
    if libcpp is None:
        raise ImportError("riptide.libcpp is required for baseline FFA folding.")

    n_series, nt = vis_2d.shape
    effective_tau = tsamp * downsample_factor
    base_period_s = effective_tau * base_period_bins

    # Ensure each segment contains at least 2 rotations for valid FFA folding (rows >= 2)
    effective_seg_duration = max(segment_duration_s, 2.05 * base_period_s)
    seg_samples_native = int(round(effective_seg_duration / tsamp))
    if seg_samples_native <= 0:
        seg_samples_native = nt

    n_segments = max(1, nt // seg_samples_native)
    if n_segments <= 1:
        # Fall back to full-observation folding if observation fits in <= 1 segment
        return fold_baselines_full(
            vis_2d, wgt_2d, uvw_track, base_period_bins, tsamp,
            rows_eval=rows_eval, downsample_factor=downsample_factor
        )

    seg_len = nt // n_segments

    all_uvw = []
    all_wgt = []
    all_folded = []
    periods_out = None

    for s in range(n_segments):
        t_start = s * seg_len
        t_end = (s + 1) * seg_len
        t_ref_s = t_start * tsamp

        vis_s = np.ascontiguousarray(vis_2d[:, t_start:t_end], dtype=np.complex64)
        wgt_s = np.ascontiguousarray(wgt_2d[:, t_start:t_end], dtype=np.float32)
        uvw_s = np.ascontiguousarray(uvw_track[:, :, t_start:t_end], dtype=np.float64)

        # Segment UVW centroid
        wsum_s = np.sum(wgt_s, axis=1, keepdims=True)
        has_w_s = (wsum_s > 0.0)
        weighted_uvw_s = np.sum(uvw_s * wgt_s[:, None, :], axis=2)
        uvw_eff_s = np.where(has_w_s, weighted_uvw_s / np.where(has_w_s, wsum_s, 1.0), np.mean(uvw_s, axis=2))
        wgt_eff_s = np.where(has_w_s, wsum_s / seg_len, 0.0).astype(np.float32)

        # Fold segment
        folded_s, periods = libcpp.vis_ffa_transform_basep(
            vis_s,
            int(base_period_bins),
            effective_tau,
            int(rows_eval),
            float(downsample_factor),
        )
        if periods_out is None:
            periods_out = periods

        # Phase alignment relative to t=0 epoch:
        # Phase shift: delta_phi = (t_ref_s / Period) mod 1
        delta_phase = (t_ref_s / periods) % 1.0  # (N_periods,)

        # Rotate phase axis across all baselines vectorized:
        # folded_s has shape (N_series, N_periods, base_period_bins)
        aligned_s = shift_profile_phase_fourier(folded_s, delta_phase)

        all_uvw.append(uvw_eff_s)
        all_wgt.append(wgt_eff_s)
        all_folded.append(aligned_s)

    # Concatenate across segments
    uvw_eff = np.concatenate(all_uvw, axis=0)       # (S * N_series, 3)
    wgt_eff = np.concatenate(all_wgt, axis=0)       # (S * N_series, 1)
    folded_vis = np.concatenate(all_folded, axis=0) # (S * N_series, N_periods, base_period_bins)

    return uvw_eff, wgt_eff, folded_vis, periods_out


# Alias for backward compatibility
fold_baselines_option_a = fold_baselines_segmented


def image_phase_bins_wgridder(
    uvw: np.ndarray,
    freq: np.ndarray,
    vis_period: np.ndarray,
    wgt: Optional[np.ndarray] = None,
    npix_x: int = 384,
    npix_y: int = 384,
    pixsize_x: float = 22.0 / 206265.0,
    pixsize_y: float = 22.0 / 206265.0,
    epsilon: float = 1e-6,
    do_wgridding: bool = True,
    nthreads: int = 0,
    divide_by_n: bool = True,
    center_x: float = 0.0,
    center_y: float = 0.0,
    *,
    wgt_norm: Optional[np.ndarray] = None,
    n_valid: Optional[int] = None,
    dirty_buf: Optional[np.ndarray] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Synthesize dirty images for all phase bins at a single trial period using ducc0.wgridder.

    Parameters
    ----------
    uvw : np.ndarray
        Shape (M, 3), baseline coordinates in meters or wavelengths.
    freq : np.ndarray
        Frequency array (Hz or [c_light]).
    vis_period : np.ndarray
        Shape (M, nbins), complex64 folded visibilities for this period across M baselines.
    wgt : np.ndarray, optional
        Shape (M, 1) or (M,) visibility weights.
    npix_x, npix_y : int
        Image dimensions.
    pixsize_x, pixsize_y : float
        Angular pixel sizes in radians.
    epsilon : float
        Gridding accuracy tolerance.
    do_wgridding : bool
        Whether to perform w-gridding.
    nthreads : int
        Number of threads for wgridder.
    divide_by_n : bool
        Whether to divide dirty image pixels by direction cosine n.
    center_x, center_y : float
        Image center shift in radians.
    wgt_norm : np.ndarray, optional
        Pre-normalized weights shape (M, 1) to amortize normalization overhead across periods.
    n_valid : int, optional
        Precomputed number of valid non-zero weight baselines.
    dirty_buf : np.ndarray, optional
        Pre-allocated work buffer of shape (npix_x, npix_y), float64, passed to vis2dirty.
    out : np.ndarray, optional
        Pre-allocated output array of shape (nbins, npix_y, npix_x), float64.

    Returns
    -------
    phase_images : np.ndarray
        Shape (nbins, npix_y, npix_x), float64 dirty image planes.
    """
    if ducc0 is None:
        raise ImportError("ducc0 is required for wgridder imaging.")

    M, nbins = vis_period.shape
    if wgt_norm is None:
        if wgt is None:
            wgt_in = np.ones((M, 1), dtype=np.float32)
        elif wgt.ndim == 1:
            wgt_in = wgt[:, None].astype(np.float32, copy=False)
        else:
            wgt_in = wgt.astype(np.float32, copy=False)

        wgt_max = float(np.max(wgt_in)) if wgt_in.size > 0 else 1.0
        if wgt_max > 0:
            wgt_norm = (wgt_in / wgt_max).astype(np.float32, copy=False)
        else:
            wgt_norm = wgt_in
        n_valid = int(np.count_nonzero(wgt_norm))
    elif n_valid is None:
        n_valid = int(np.count_nonzero(wgt_norm))

    dirty_dtype = np.float32 if vis_period.dtype == np.complex64 else np.float64
    if dirty_buf is None or dirty_buf.dtype != dirty_dtype or dirty_buf.shape != (npix_x, npix_y):
        dirty_buf = np.empty((npix_x, npix_y), dtype=dirty_dtype)

    if out is None:
        phase_images = np.empty((nbins, npix_y, npix_x), dtype=np.float64)
    else:
        phase_images = out

    for k in range(nbins):
        vis_bin = np.ascontiguousarray(vis_period[:, k : k + 1], dtype=np.complex64)

        ducc0.wgridder.vis2dirty(
            uvw=uvw,
            freq=freq,
            vis=vis_bin,
            wgt=wgt_norm,
            npix_x=npix_x,
            npix_y=npix_y,
            pixsize_x=pixsize_x,
            pixsize_y=pixsize_y,
            epsilon=epsilon,
            do_wgridding=do_wgridding,
            nthreads=nthreads,
            divide_by_n=divide_by_n,
            center_x=center_x,
            center_y=center_y,
            dirty=dirty_buf,
        )
        # Transpose DUCC (x, y) -> (y, x) convention into phase_images[k]
        if divide_by_n and n_valid > 0:
            phase_images[k] = dirty_buf.T / n_valid
        else:
            phase_images[k] = dirty_buf.T

    return phase_images


def evaluate_period_snr(
    phase_images: np.ndarray,
    trial_period: float,
    widths: Optional[np.ndarray] = None,
    ducy_max: float = 0.5,
    wtsp: float = 1.5,
    snr_thresh: float = 8.0,
    cutout_size: int = 16,
) -> Tuple[np.ndarray, float, Optional[BaselineFFACandidate]]:
    """
    Evaluate boxcar-matched spatial S/N plane across phase bins for a trial period.

    Matches the statistical model from riptide.image_snr:
      - Computes phase-mean continuum dirty image.
      - Applies circular boxcar matched filters of various widths.
      - Normalizes by the spatial image RMS stddev(x, y).
      - Returns maximum SNR plane and peak candidate if above threshold.

    Parameters
    ----------
    phase_images : np.ndarray
        Shape (nbins, npix_y, npix_x).
    trial_period : float
        Period in seconds.
    widths : np.ndarray, optional
        Array of trial boxcar widths in bins.
    snr_thresh : float
        Detection threshold.

    Returns
    -------
    snr_plane : np.ndarray
        Shape (npix_y, npix_x), maximum SNR across all widths and starting phases.
    best_snr : float
        Peak SNR value in the image plane.
    candidate : Optional[BaselineFFACandidate]
        Candidate object if best_snr >= snr_thresh, else None.
    """
    nbins, ny, nx = phase_images.shape
    if widths is None:
        if generate_width_trials is not None:
            widths = generate_width_trials(nbins, ducy_max=ducy_max, wtsp=wtsp)
        else:
            w_list = [1]
            w = 2
            while w <= max(1, int(ducy_max * nbins)):
                w_list.append(w)
                w = int(max(w + 1, round(wtsp * w)))
            widths = np.asarray(w_list, dtype=int)

    # 1. Continuum phase-median subtraction in float32:
    # Using the phase median (rather than phase mean) removes steady continuum emission
    # without leaking the pulsed source's dirty-beam sidelobes into off-pulse phase bins.
    phase_f32 = phase_images.astype(np.float32, copy=False)
    med_img = np.median(phase_f32, axis=0)  # (ny, nx)
    diff = phase_f32 - med_img[None, :, :]  # (nbins, ny, nx)

    # Off-pulse single-bin spatial noise floor via Median Absolute Deviation (MAD)
    med_xy = np.median(diff, axis=(1, 2), keepdims=True)
    sigma_per_bin = 1.4826 * np.median(np.abs(diff - med_xy), axis=(1, 2))
    sigma_base = float(np.median(sigma_per_bin))

    # 2. Circular prefix sum along phase axis for fast boxcar evaluation
    max_w = int(np.max(widths)) if len(widths) > 0 else 1
    diff_wrap = np.concatenate([diff, diff[:max_w]], axis=0)
    cumsum = np.concatenate([np.zeros((1, ny, nx), dtype=np.float32), np.cumsum(diff_wrap, axis=0, dtype=np.float32)], axis=0)

    snr_plane = np.full((ny, nx), -1e9, dtype=np.float32)
    box_buf = np.empty((nbins, ny, nx), dtype=np.float32)
    max_snr_w = np.empty((ny, nx), dtype=np.float32)

    best_snr = -1e9
    best_w = 1
    best_k = 0
    best_y = 0
    best_x = 0

    for w in widths:
        w_int = int(w)
        if w_int >= nbins or w_int <= 0:
            continue

        # Boxcar sum for all nbins starting phases in-place: (nbins, ny, nx)
        np.subtract(cumsum[w_int : w_int + nbins], cumsum[:nbins], out=box_buf)

        sigma_w = sigma_base * math.sqrt(float(w_int))
        inv_sigma = np.float32(1.0 / sigma_w if sigma_w > 1e-12 else 0.0)
        box_buf *= inv_sigma  # snr_w in-place: (nbins, ny, nx)

        # Max over starting phase bins for this width in-place:
        np.max(box_buf, axis=0, out=max_snr_w)
        np.maximum(snr_plane, max_snr_w, out=snr_plane)

        peak_w_val = float(np.max(max_snr_w))
        if peak_w_val > best_snr:
            best_snr = peak_w_val
            best_w = w_int
            best_k, best_y, best_x = np.unravel_index(np.argmax(box_buf), box_buf.shape)

    cand = None
    if best_snr >= snr_thresh:
        # Extract cutout from best phase image
        half = cutout_size // 2
        y0 = max(0, best_y - half)
        y1 = min(ny, best_y + half)
        x0 = max(0, best_x - half)
        x1 = min(nx, best_x + half)
        cutout = phase_images[best_k, y0:y1, x0:x1].copy()

        cand = BaselineFFACandidate(
            period=trial_period,
            snr=best_snr,
            x=int(best_x),
            y=int(best_y),
            width_bins=best_w,
            phase_bin=int(best_k),
            cutout=cutout
        )

    return snr_plane, float(best_snr), cand


def search_baseline_ffa(
    msname_or_data: Union[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]],
    period_min: float,
    period_max: float,
    bins_min: int = 16,
    bins_max: int = 64,
    dm: float = 0.0,
    collapse_channels: bool = False,
    nsubbands: int = 1,
    exact_uvw: bool = True,
    segment_duration_s: Optional[float] = None,
    npix_x: int = 384,
    npix_y: int = 384,
    pixsize_x: float = 22.0 / 206265.0,
    pixsize_y: float = 22.0 / 206265.0,
    snr_thresh: float = 8.0,
    start_time_idx: Optional[int] = None,
    end_time_idx: Optional[int] = None,
    epsilon: float = 1e-6,
    do_wgridding: bool = True,
    nthreads: int = 0,
    save_best_dirty_cube: bool = False,
) -> BaselineFFAResult:
    """
    End-to-End Baseline FFA Periodicity Search & DUCC0 Imaging Pipeline.

    Parameters
    ----------
    msname_or_data : str or tuple
        Either path to Measurement Set or a tuple of:
        (vis_3d, wgt_3d, uvw_3d, chan_freq, dt).
    period_min, period_max : float
        Trial period range in seconds.
    bins_min, bins_max : int
        Minimum and maximum phase folding bins (e.g. 16, 64).
    dm : float
        Dispersion measure in pc/cm^3.
    collapse_channels : bool
        If True, sum across channels before folding (standard mode).
        If False (default), fold pseudo-baselines with exact wavelength coordinates.
    nsubbands : int
        Number of subbands when collapse_channels is True (default 1 for full collapse).
    exact_uvw : bool
        If True (default) and collapse_channels is False, compute exact per-channel UVW delay tracks.
    segment_duration_s : float, optional
        Stationary segment duration in seconds. If provided, partitions time series into
        segments where UV tracks are approximately stationary, then synthesizes across segments.
        If None, runs full-observation baseline-folded imaging.
    npix_x, npix_y : int
        Image grid dimensions.
    pixsize_x, pixsize_y : float
        Pixel angular size in radians.
    snr_thresh : float
        Candidate detection SNR threshold.
    start_time_idx, end_time_idx : int, optional
        Time index range to process from MS.

    Returns
    -------
    result : BaselineFFAResult
    """
    # 1. Load data
    wcs_obj = None
    if isinstance(msname_or_data, str):
        logger.info(f"Loading visibility data from MS: {msname_or_data}")
        u_times, chan_freq, dt, vis_3d, wgt_3d, uvw_3d = imaging.load_chunk_data(
            msname_or_data,
            start_time_idx=start_time_idx,
            end_time_idx=end_time_idx,
            corr_mode='average',
            average_correlations=True
        )
        try:
            wcs_obj = ducc_wcs.get_ms_wcs(
                msname_or_data, npix_x=npix_x, npix_y=npix_y,
                pixsize_x=pixsize_x, pixsize_y=pixsize_y
            )
        except Exception as e:
            logger.warning(f"Could not construct WCS from MS: {e}")
    else:
        vis_3d, wgt_3d, uvw_3d, chan_freq, dt = msname_or_data

    nbl, nchan, nt = vis_3d.shape
    total_duration_s = nt * dt
    logger.info(
        f"Input: {nbl} baselines, {nchan} channels, {nt} time samples ({total_duration_s:.2f}s). "
        f"Search: P in [{period_min:.4f}, {period_max:.4f}] s, DM={dm:.2f}"
    )

    # 2. Prepare baseline time series and coordinates
    vis_2d, wgt_2d, uvw_track, eff_freq, dt = prepare_baseline_series(
        vis_3d, wgt_3d, uvw_3d, chan_freq, dt,
        dm=dm, collapse_channels=collapse_channels, nsubbands=nsubbands, exact_uvw=exact_uvw
    )

    # 3. Generate FFA Plan
    if plan_ffa is None:
        raise ImportError("riptide.ffautils.plan_ffa is required.")

    plan = plan_ffa(nt, dt, period_min, period_max, bins_min, bins_max)
    logger.info(f"Generated FFA plan with {len(plan)} base periods.")

    all_periods = []
    all_best_snrs = []
    candidates = []
    best_candidate = None
    global_best_snr = -1e9
    global_best_snr_plane = np.zeros((npix_y, npix_x), dtype=np.float32)
    best_dirty_cube = None

    # 4. Stream over FFA base period blocks
    for i_bp, bp_info in enumerate(plan):
        b_bins = int(bp_info["bins"])
        ds_factor = float(bp_info["downsample_factor"])
        rows_eval = int(bp_info["rows_eval"])

        # Fold baselines
        if segment_duration_s is not None and segment_duration_s > 0:
            uvw_eff, wgt_eff, folded_vis, trial_periods = fold_baselines_segmented(
                vis_2d, wgt_2d, uvw_track, b_bins, dt,
                segment_duration_s=segment_duration_s,
                rows_eval=rows_eval, downsample_factor=ds_factor
            )
        else:
            uvw_eff, wgt_eff, folded_vis, trial_periods = fold_baselines_full(
                vis_2d, wgt_2d, uvw_track, b_bins, dt,
                rows_eval=rows_eval, downsample_factor=ds_factor
            )

        n_periods_bp = len(trial_periods)
        logger.debug(
            f"Base period {i_bp+1}/{len(plan)}: bins={b_bins}, ds={ds_factor:.2f}, "
            f"evaluating {n_periods_bp} periods [{trial_periods[0]:.4f} - {trial_periods[-1]:.4f}] s"
        )

        # Precompute normalized weights and valid baseline count once per base period block
        m_eff = uvw_eff.shape[0]
        if wgt_eff is None:
            wgt_in = np.ones((m_eff, 1), dtype=np.float32)
        elif wgt_eff.ndim == 1:
            wgt_in = wgt_eff[:, None].astype(np.float32, copy=False)
        else:
            wgt_in = wgt_eff.astype(np.float32, copy=False)

        wgt_max = float(np.max(wgt_in)) if wgt_in.size > 0 else 1.0
        wgt_norm = (wgt_in / wgt_max).astype(np.float32, copy=False) if wgt_max > 0 else wgt_in
        n_valid = int(np.count_nonzero(wgt_norm))

        # Pre-allocate reusable imaging work buffer and phase images output buffer
        dirty_dtype = np.float32 if folded_vis.dtype == np.complex64 else np.float64
        dirty_work_buf = np.empty((npix_x, npix_y), dtype=dirty_dtype)
        phase_imgs_buf = np.empty((b_bins, npix_y, npix_x), dtype=np.float64)

        # 5. Stream through trial periods and image phase bins
        for r, period in enumerate(trial_periods):
            # folded_vis: (M, N_periods, b_bins) -> extract period r: (M, b_bins)
            vis_period = folded_vis[:, r, :]

            # Image phase bins with zero in-loop buffer allocations
            phase_imgs = image_phase_bins_wgridder(
                uvw=uvw_eff,
                freq=eff_freq,
                vis_period=vis_period,
                wgt_norm=wgt_norm,
                n_valid=n_valid,
                dirty_buf=dirty_work_buf,
                out=phase_imgs_buf,
                npix_x=npix_x,
                npix_y=npix_y,
                pixsize_x=pixsize_x,
                pixsize_y=pixsize_y,
                epsilon=epsilon,
                do_wgridding=do_wgridding,
                nthreads=nthreads,
            )

            # Evaluate spatial SNR across phase bins
            snr_plane, best_snr, cand = evaluate_period_snr(
                phase_imgs, period, snr_thresh=snr_thresh
            )

            all_periods.append(period)
            all_best_snrs.append(best_snr)

            if cand is not None:
                if wcs_obj is not None:
                    try:
                        sky = wcs_obj.pixel_to_world(cand.x, cand.y)
                        cand.ra_deg = float(sky.ra.deg)
                        cand.dec_deg = float(sky.dec.deg)
                    except Exception:
                        pass
                cand.dm = dm
                candidates.append(cand)

            if best_snr > global_best_snr:
                global_best_snr = best_snr
                global_best_snr_plane = snr_plane.copy()
                best_candidate = cand
                if save_best_dirty_cube:
                    best_dirty_cube = phase_imgs.copy()

    periods_arr = np.array(all_periods, dtype=np.float64)
    snr_arr = np.array(all_best_snrs, dtype=np.float32)

    # Sort strictly monotonically by period
    sort_order = np.argsort(periods_arr)
    periods_arr = periods_arr[sort_order]
    snr_arr = snr_arr[sort_order]

    # Sort candidates by SNR descending
    candidates.sort(key=lambda c: c.snr, reverse=True)

    result = BaselineFFAResult(
        periods=periods_arr,
        snr_periodogram=snr_arr,
        best_snr_plane=global_best_snr_plane,
        candidates=candidates,
        best_candidate=best_candidate,
        best_dirty_cube=best_dirty_cube,
        wcs=wcs_obj,
        plan=plan
    )

    logger.info(
        f"Search completed: {len(periods_arr)} periods evaluated. "
        f"Global peak SNR: {global_best_snr:.2f}, {len(candidates)} candidates >= {snr_thresh} sigma."
    )
    return result
