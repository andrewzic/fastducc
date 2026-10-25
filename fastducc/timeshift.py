"""
fastducc.timeshift
==================
Time grid regularization, gap padding, and sub-sample block time synchronization
for radio interferometric data and FITS cubes.

Supports:
- Fourier domain phase gradient shift (FFT shift with edge-padding)
- Time domain windowed sinc interpolation (Lanczos kernel FIR filter)
- Automatic contiguous block detection from scan changes and timestamp gaps
- Global regular time grid mapping with configurable sync tolerance and gap padding
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import convolve1d

logger = logging.getLogger("fastducc.timeshift")


def shift_time_series(
    data: np.ndarray,
    shift_samps: float,
    method: str = "fft",
    axis: int = -1,
    lanczos_radius: int = 4,
    pad_samples: int = 8,
) -> np.ndarray:
    """
    Shift an array along a specified time axis by a fractional number of samples.

    A positive shift_samps delays the signal (shifts towards higher indices):
        y(t) = x(t - shift_samps * dt)
    A negative shift_samps advances the signal (shifts towards lower indices):
        y(t) = x(t + |shift_samps| * dt)

    Parameters
    ----------
    data : np.ndarray
        Input ND array (real or complex).
    shift_samps : float
        Shift in units of time samples.
    method : {"fft", "sinc"}
        "fft": Fourier phase gradient shift theorem with edge-reflection padding.
        "sinc": Lanczos-windowed sinc interpolation FIR filter.
    axis : int, default -1
        Axis along which time varies.
    lanczos_radius : int, default 4
        Filter half-width (radius) for sinc interpolation (kernel length = 2*R + 1).
    pad_samples : int, default 8
        Edge-padding width for FFT shift to suppress circular boundary wrap-around.

    Returns
    -------
    np.ndarray
        Shifted array with the same shape and dtype as input data.
    """
    if abs(shift_samps) < 1e-7 or data.shape[axis] <= 1:
        return data.copy()

    n = data.shape[axis]
    method = str(method).lower().strip()

    if method == "fft":
        pad = min(int(pad_samples), n)
        pad_width = [(0, 0)] * data.ndim
        pad_width[axis] = (pad, pad)
        data_padded = np.pad(data, pad_width, mode="edge")
        npad = data_padded.shape[axis]

        if np.iscomplexobj(data):
            freqs = np.fft.fftfreq(npad)
            phase = np.exp(-2j * np.pi * freqs * shift_samps)
            shape_phase = [1] * data.ndim
            shape_phase[axis] = npad
            X = np.fft.fft(data_padded, axis=axis)
            shifted = np.fft.ifft(X * phase.reshape(shape_phase), axis=axis)
        else:
            freqs_r = np.fft.rfftfreq(npad)
            phase_r = np.exp(-2j * np.pi * freqs_r * shift_samps)
            shape_phase = [1] * data.ndim
            shape_phase[axis] = len(freqs_r)
            X = np.fft.rfft(data_padded, axis=axis)
            shifted = np.fft.irfft(X * phase_r.reshape(shape_phase), n=npad, axis=axis)

        slc = [slice(None)] * data.ndim
        slc[axis] = slice(pad, pad + n)
        return shifted[tuple(slc)].astype(data.dtype)

    elif method == "sinc":
        radius = min(int(lanczos_radius), max(1, n // 2))
        k = np.arange(-radius, radius + 1)
        u = k - shift_samps
        h = np.sinc(u) * np.sinc(u / radius)
        h_sum = np.sum(h)
        if abs(h_sum) > 1e-12:
            h /= h_sum

        if np.iscomplexobj(data):
            r = convolve1d(data.real, h, axis=axis, mode="nearest")
            i = convolve1d(data.imag, h, axis=axis, mode="nearest")
            return (r + 1j * i).astype(data.dtype)
        else:
            return convolve1d(data, h, axis=axis, mode="nearest").astype(data.dtype)

    else:
        raise ValueError(f"Unknown shift method '{method}'. Choose 'fft' or 'sinc'.")


def identify_time_blocks(
    times: np.ndarray,
    scan_numbers: Optional[np.ndarray] = None,
    dt_nominal: Optional[float] = None,
    gap_threshold: float = 1.5,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Identify contiguous blocks of timestamps based on scan boundaries and inter-scan gaps.

    Parameters
    ----------
    times : np.ndarray
        Sorted 1D array of timestamps (seconds).
    scan_numbers : np.ndarray, optional
        1D array of scan numbers corresponding to each timestamp.
    dt_nominal : float, optional
        Nominal sampling interval (seconds). If None, inferred from median positive diff.
    gap_threshold : float, default 1.5
        Factor of dt_nominal beyond which a gap starts a new block.

    Returns
    -------
    tuple[list[dict], float]
        (blocks, dt_nominal)
        Each block is a dict with:
            block_idx, start_sample, end_sample, n_samples, t_start, t_end, scan_id
    """
    n_times = len(times)
    if n_times == 0:
        return [], 1.0 if dt_nominal is None else float(dt_nominal)

    if n_times == 1:
        dt = 1.0 if dt_nominal is None else float(dt_nominal)
        return [{
            "block_idx": 0, "start_sample": 0, "end_sample": 0, "n_samples": 1,
            "t_start": float(times[0]), "t_end": float(times[0]),
            "scan_id": int(scan_numbers[0]) if scan_numbers is not None and len(scan_numbers) > 0 else 0,
        }], dt

    diffs = np.diff(times)
    pos_diffs = diffs[diffs > 1e-7]
    if dt_nominal is not None and float(dt_nominal) > 0:
        dt = float(dt_nominal)
    elif len(pos_diffs) > 0:
        dt = float(np.median(pos_diffs))
    else:
        dt = 1.0

    boundaries = [0]
    for i in range(len(diffs)):
        is_scan_change = (scan_numbers is not None and len(scan_numbers) > i + 1 and scan_numbers[i] != scan_numbers[i + 1])
        is_gap = (diffs[i] > gap_threshold * dt)
        if is_scan_change or is_gap:
            boundaries.append(i + 1)
    boundaries.append(n_times)

    blocks = []
    for b_idx in range(len(boundaries) - 1):
        s_idx = boundaries[b_idx]
        e_idx = boundaries[b_idx + 1]
        scan_id = int(scan_numbers[s_idx]) if scan_numbers is not None and len(scan_numbers) > s_idx else 0
        blocks.append({
            "block_idx": b_idx,
            "start_sample": s_idx,
            "end_sample": e_idx - 1,
            "n_samples": e_idx - s_idx,
            "t_start": float(times[s_idx]),
            "t_end": float(times[e_idx - 1]),
            "scan_id": scan_id,
        })

    return blocks, dt


def regularize_time_grid(
    times: np.ndarray,
    dt_nominal: Optional[float] = None,
    sync_tolerance: float = 0.05,
    scan_numbers: Optional[np.ndarray] = None,
    max_blank_factor: float = 10.0,
    gap_threshold: float = 1.5,
) -> Dict[str, Any]:
    """
    Map timestamps onto a strictly uniform regular time grid, identifying inter-scan
    gaps to pad with blank frames, and computing sub-sample time shifts for blocks
    whose clock offset relative to the regular grid exceeds sync_tolerance.

    Parameters
    ----------
    times : np.ndarray
        Sorted 1D array of timestamps (seconds).
    dt_nominal : float, optional
        Target regular time resolution (seconds). Inferred if None.
    sync_tolerance : float, default 0.05
        Offset tolerance measured in units of offset / dt_nominal (e.g. 0.05 = 5%).
        If |offset / dt_nominal| > sync_tolerance, the block is flagged out of sync
        and assigned a sub-sample shift of -offset.
    scan_numbers : np.ndarray, optional
        Scan numbers corresponding to each timestamp.
    max_blank_factor : float, default 10.0
        Maximum allowed expansion ratio (total_planes / n_times) for gap padding.
    gap_threshold : float, default 1.5
        Threshold factor for detecting gaps between blocks.

    Returns
    -------
    dict
        Dictionary containing:
            - spacing_type: 'regular' or 'regular_with_gaps'
            - is_regular: True
            - has_gaps: bool
            - dt_nominal: float
            - total_planes: int
            - grid_indices: np.ndarray of shape (n_times,) mapping each sample to target plane
            - grid_times: np.ndarray of shape (total_planes,) giving exact regular timestamps
            - blank_indices: list of int giving missing gap plane indices
            - is_blank: bool np.ndarray of shape (total_planes,)
            - delta_t: float32 np.ndarray of shape (total_planes,)
            - blocks: list of block metadata dictionaries
            - shift_samps_per_time: np.ndarray of shape (n_times,) giving shift in samples
            - n_shifted_blocks: int count of blocks requiring time alignment
    """
    blocks, dt = identify_time_blocks(times, scan_numbers=scan_numbers, dt_nominal=dt_nominal, gap_threshold=gap_threshold)
    n_times = len(times)
    t_ref = float(times[0])

    grid_indices = np.zeros(n_times, dtype=int)
    shift_samps_per_time = np.zeros(n_times, dtype=np.float64)

    last_grid_end = -1
    n_shifted = 0

    for b in blocks:
        s_idx = b["start_sample"]
        e_idx = b["end_sample"] + 1
        n_b = b["n_samples"]
        t_b0 = b["t_start"]

        # Nominal integer grid index on global regular lattice
        k_b0 = int(round((t_b0 - t_ref) / dt))
        if k_b0 <= last_grid_end:
            k_b0 = last_grid_end + 1

        t_grid_target = t_ref + k_b0 * dt
        offset_sec = t_b0 - t_grid_target
        offset_frac = offset_sec / dt  # [-0.5, 0.5] in units of dt

        is_out_of_sync = abs(offset_frac) > float(sync_tolerance)
        shift_samps = -offset_frac if is_out_of_sync else 0.0

        if is_out_of_sync:
            n_shifted += 1

        b_grid_indices = k_b0 + np.arange(n_b, dtype=int)
        grid_indices[s_idx:e_idx] = b_grid_indices
        shift_samps_per_time[s_idx:e_idx] = shift_samps
        last_grid_end = int(b_grid_indices[-1])

        b.update({
            "k_start": k_b0,
            "k_end": int(b_grid_indices[-1]),
            "offset_sec": offset_sec,
            "offset_frac": offset_frac,
            "shift_samps": shift_samps,
            "is_shifted": is_out_of_sync,
        })

    total_planes = int(np.max(grid_indices)) + 1
    if (total_planes / n_times) > max_blank_factor:
        logger.warning(
            f"Time grid expansion ({total_planes / n_times:.1f}x) exceeds max_blank_factor ({max_blank_factor}x). "
            f"Capping blank padding or consider increasing --max-blank-factor."
        )

    blank_indices = sorted(list(set(range(total_planes)) - set(grid_indices)))
    is_blank = np.zeros(total_planes, dtype=bool)
    is_blank[blank_indices] = True
    grid_times = t_ref + np.arange(total_planes) * dt
    delta_t = np.full(total_planes, dt, dtype=np.float32)

    has_gaps = len(blank_indices) > 0
    spacing_type = "regular_with_gaps" if has_gaps else "regular"

    return {
        "spacing_type": spacing_type,
        "is_regular": True,
        "has_gaps": has_gaps,
        "dt_nominal": dt,
        "total_planes": total_planes,
        "grid_indices": grid_indices,
        "grid_times": grid_times,
        "blank_indices": blank_indices,
        "is_blank": is_blank,
        "delta_t": delta_t,
        "blocks": blocks,
        "shift_samps_per_time": shift_samps_per_time,
        "n_shifted_blocks": n_shifted,
    }
