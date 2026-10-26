from __future__ import annotations

import logging
import sys
from typing import Any, Tuple

import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

try:
    from casacore.tables import table
except ImportError:
    table = None

try:
    import ducc0
except ImportError:
    ducc0 = None

from fastducc import ms_utils
from fastducc.constants import c as c_light

logger = logging.getLogger(__name__)

def _collapse_correlations(
    data: np.ndarray,
    wgt: np.ndarray,
    corr_mode: str,
    basis: str,
    single_pol: str,
    average_correlations: bool,
    corr_index: int | None,
    labels: list[str],
    lbl2idx: dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collapse correlation axis of visibility data and weights into 2D (nrow, nchan) arrays.
    """
    if corr_mode == 'average':
        if average_correlations:
            wsum = wgt.sum(axis=2)
            with np.errstate(invalid='ignore', divide='ignore'):
                vis = (data * wgt).sum(axis=2) / np.where(wsum > 0.0, wsum, np.nan)
            vis = np.nan_to_num(vis, nan=0.0)
            wgt_2d = wsum
        else:
            ci = 0 if corr_index is None else corr_index
            vis = data[:, :, ci]
            wgt_2d = wgt[:, :, ci]
    elif corr_mode == 'single':
        if single_pol not in lbl2idx:
            raise ValueError(f"Requested single_pol='{single_pol}' not present in MS correlations: {labels}")
        ci = lbl2idx[single_pol]
        vis = data[:, :, ci]
        wgt_2d = wgt[:, :, ci]
    elif corr_mode == 'stokesI':
        have_linear = ('XX' in lbl2idx) and ('YY' in lbl2idx)
        have_circular = ('RR' in lbl2idx) and ('LL' in lbl2idx)
        use_linear = False
        if basis == 'linear':
            if not have_linear:
                raise ValueError("basis='linear' requested but XX/YY not found in MS correlations")
            use_linear = True
        elif basis == 'circular':
            if not have_circular:
                raise ValueError("basis='circular' requested but RR/LL not found in MS correlations")
        else:
            if have_linear:
                use_linear = True
            elif not have_circular:
                raise ValueError("Cannot form Stokes I: XX/YY or RR/LL not present in MS correlations")
        i1, i2 = (lbl2idx['XX'], lbl2idx['YY']) if use_linear else (lbl2idx['RR'], lbl2idx['LL'])
        v1, w1 = data[:, :, i1], wgt[:, :, i1]
        v2, w2 = data[:, :, i2], wgt[:, :, i2]
        present1 = (w1 > 0.0)
        present2 = (w2 > 0.0)
        n_valid = present1.astype(np.int32) + present2.astype(np.int32)
        sum_vis = np.where(present1, v1, 0.0) + np.where(present2, v2, 0.0)
        with np.errstate(invalid='ignore', divide='ignore'):
            vis = sum_vis / np.where(n_valid > 0, n_valid, np.nan)
        vis = np.nan_to_num(vis, nan=0.0)
        with np.errstate(invalid='ignore', divide='ignore'):
            w_two = 4.0 / (np.where(present1, 1.0 / w1, 0.0) + np.where(present2, 1.0 / w2, 0.0))
        w_one = np.where(present1 & (~present2), w1, 0.0) + np.where((~present1) & present2, w2, 0.0)
        wgt_2d = np.where(n_valid == 2, np.nan_to_num(w_two, nan=0.0, posinf=0.0, neginf=0.0), w_one)
    else:
        raise ValueError(f"Unknown corr_mode='{corr_mode}'. Use 'average', 'stokesI', or 'single'.")
    return vis, wgt_2d


def continuum_image(
    msname: str,
    *,
    t_main: Any | None = None,
    data_column: str = 'DATA',
    corr_mode: str = 'average',  # 'average' | 'stokesI' | 'single'
    basis: str = 'auto',         # for stokesI: 'auto' | 'linear' | 'circular'
    single_pol: str = 'XX',      # used when corr_mode='single'
    average_correlations: bool = True,
    corr_index: int | None = None,
    use_weight_spectrum: bool = True,
    npix_x: int = 384,
    npix_y: int = 384,
    pixsize_x: float = 22.0/206265.0,
    pixsize_y: float = 22.0/206265.0,
    epsilon: float = 1e-6,
    do_wgridding: bool = True,
    nthreads: int = 0,
    verbosity: int = 0,
    flip_u: bool = False,
    flip_v: bool = False,
    flip_w: bool = False,
    divide_by_n: bool = True,
    sigma_min: float = 1.1,
    sigma_max: float = 2.6,
    center_x: float = 0.0,
    center_y: float = 0.0,
    allow_nshift: bool = True,
    double_precision_accumulation: bool = False,
) -> np.ndarray:
    """
    Synthesize a full-observation continuum dirty image from a Measurement Set
    using ducc0.wgridder.vis2dirty.
    """
    opened_locally = False
    if t_main is None:
        logger.debug(f"reading table from msname {msname}")
        t_main = table(msname, readonly=True)
        opened_locally = True

    try:
        with table(f"{msname}/SPECTRAL_WINDOW", readonly=True) as t_spw:
            n_spw = t_spw.nrows()
            if n_spw != 1:
                raise ValueError(f"continuum_image() currently supports a single SPW; found {n_spw}")
            chan_freq = t_spw.getcell('CHAN_FREQ', 0)  # [nchan] Hz

        labels, lbl2idx = ms_utils.get_corr_label_indices(msname)
        colnames = set(t_main.colnames())

        uvw = t_main.getcol('UVW')
        data = t_main.getcol(data_column)   # [nrow, nchan, ncorr]
        flags = t_main.getcol('FLAG')        # [nrow, nchan, ncorr]
        flag_row = t_main.getcol('FLAG_ROW') if 'FLAG_ROW' in colnames else None

        if use_weight_spectrum and 'WEIGHT_SPECTRUM' in colnames:
            wgt = t_main.getcol('WEIGHT_SPECTRUM')
        else:
            wgt_row = t_main.getcol('WEIGHT')
            wgt = np.broadcast_to(wgt_row[:, None, :], data.shape)
    finally:
        if opened_locally and t_main is not None:
            t_main.close()

    logger.debug("applying flags")
    good = ~flags
    if flag_row is not None:
        good &= (~flag_row[:, None, None])
    wgt = np.where(good, wgt, 0.0)

    vis, wgt_2d = _collapse_correlations(
        data, wgt, corr_mode, basis, single_pol, average_correlations, corr_index, labels, lbl2idx
    )

    if uvw.shape[0] != vis.shape[0]:
        raise ValueError('Row count mismatch between UVW and VIS')
    if chan_freq.shape[0] != vis.shape[1]:
        raise ValueError('Channel count mismatch between CHAN_FREQ and VIS')

    wgt_max = float(np.max(wgt_2d)) if wgt_2d.size > 0 else 0.0
    if wgt_max > 0.0:
        wgt_2d = wgt_2d / wgt_max

    dirty = ducc0.wgridder.vis2dirty(
        uvw=uvw,
        freq=chan_freq,
        vis=vis,
        wgt=wgt_2d,
        npix_x=npix_x,
        npix_y=npix_y,
        pixsize_x=pixsize_x,
        pixsize_y=pixsize_y,
        epsilon=epsilon,
        do_wgridding=do_wgridding,
        nthreads=nthreads,
        verbosity=verbosity,
        flip_u=flip_u,
        flip_v=flip_v,
        flip_w=flip_w,
        divide_by_n=divide_by_n,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        center_x=center_x,
        center_y=center_y,
        allow_nshift=allow_nshift,
        double_precision_accumulation=double_precision_accumulation,
    )
    dirty = dirty.T

    n_valid = int(np.count_nonzero(wgt_2d))
    if n_valid > 0:
        dirty = dirty / n_valid

    return dirty
    
def load_chunk_data(
    msname: str,
    *,
    t_main: Any | None = None,
    start_time_idx: int | None = None,
    end_time_idx: int | None = None,
    chunk_times: np.ndarray | None = None,
    data_column: str = 'DATA',
    corr_mode: str = 'average',  # 'average' | 'stokesI' | 'single'
    basis: str = 'auto',         # for stokesI: 'auto' | 'linear' | 'circular'
    single_pol: str = 'XX',      # used when corr_mode='single'
    average_correlations: bool = True,
    corr_index: int | None = None,
    use_weight_spectrum: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read visibilities, weights, and UVW coordinates from Measurement Set for a single
    time chunk into memory arrays.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]
        (times, chan_freq, dt, vis_3d, wgt_3d, uvw_3d)
        vis_3d: (nbl, nchan, nt) complex64
        wgt_3d: (nbl, nchan, nt) float32
        uvw_3d: (nbl, 3, nt) float64
    """
    opened_locally = False
    if t_main is None:
        t_main = table(msname, readonly=True)
        opened_locally = True

    t_chunk_window = None
    try:
        colnames = set(t_main.colnames())
        time_col = 'TIME_CENTROID' if 'TIME_CENTROID' in colnames else 'TIME'

        # Frequencies
        with table(f"{msname}/SPECTRAL_WINDOW", readonly=True) as t_spw:
            n_spw = t_spw.nrows()
            if n_spw != 1:
                raise ValueError(f"fastducc currently supports a single SPW; found {n_spw}")
            chan_freq = t_spw.getcell('CHAN_FREQ', 0)  # [nchan] Hz

        if chunk_times is not None:
            times = chunk_times
            query_str = f"{time_col} >= {times[0]:.17g} AND {time_col} <= {times[-1]:.17g}"
            t_chunk_window = t_main.query(query_str)
        else:
            nsub, vis_time, all_times = ms_utils.get_time(t_main)
            start_idx = 0 if start_time_idx is None else int(start_time_idx)
            end_idx   = (nsub - 1) if end_time_idx is None else int(end_time_idx)
            if start_idx < 0 or end_idx < start_idx or end_idx >= nsub:
                raise ValueError("Invalid start/end time indices")
            times = all_times[start_idx:end_idx+1]
            query_str = f"{time_col} >= {times[0]:.17g} AND {time_col} <= {times[-1]:.17g}"
            t_chunk_window = t_main.query(query_str)

        dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
        labels, lbl2idx = ms_utils.get_corr_label_indices(msname)

        time_vals = t_chunk_window.getcol(time_col)
        a1 = t_chunk_window.getcol('ANTENNA1')
        a2 = t_chunk_window.getcol('ANTENNA2')
        uvw_all = t_chunk_window.getcol('UVW')
        data_all = t_chunk_window.getcol(data_column)
        flags_all = t_chunk_window.getcol('FLAG')
        flag_row = t_chunk_window.getcol('FLAG_ROW') if 'FLAG_ROW' in set(t_chunk_window.colnames()) else None

        if use_weight_spectrum and 'WEIGHT_SPECTRUM' in set(t_chunk_window.colnames()):
            wgt_all = t_chunk_window.getcol('WEIGHT_SPECTRUM')
        else:
            wgt_row = t_chunk_window.getcol('WEIGHT')
            wgt_all = np.broadcast_to(wgt_row[:, None, :], data_all.shape)

        good = ~flags_all
        if flag_row is not None:
            good &= (~flag_row[:, None, None])
        wgt_all = np.where(good, wgt_all, 0.0)

        vis_flat, wgt_flat = _collapse_correlations(
            data_all, wgt_all, corr_mode, basis, single_pol, average_correlations, corr_index, labels, lbl2idx
        )

        bl_keys = (a1.astype(np.int64) << 32) | a2.astype(np.int64)
        u_bls, bl_inv = np.unique(bl_keys, return_inverse=True)
        nbl = len(u_bls)

        u_times, time_inv = np.unique(time_vals, return_inverse=True)
        nt = len(u_times)
        nchan = len(chan_freq)

        vis_3d = np.zeros((nbl, nchan, nt), dtype=np.complex64)
        wgt_3d = np.zeros((nbl, nchan, nt), dtype=np.float32)
        uvw_3d = np.zeros((nbl, 3, nt), dtype=np.float64)

        vis_3d[bl_inv, :, time_inv] = vis_flat
        wgt_3d[bl_inv, :, time_inv] = wgt_flat
        uvw_3d[bl_inv, :, time_inv] = uvw_all
    finally:
        if t_chunk_window is not None:
            t_chunk_window.close()
        if opened_locally and t_main is not None:
            t_main.close()

    return u_times, chan_freq, dt, vis_3d, wgt_3d, uvw_3d


def grid_chunk_cube(
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
    npix_x: int = 384,
    npix_y: int = 384,
    pixsize_x: float = 22.0 / 206265.0,
    pixsize_y: float = 22.0 / 206265.0,
    epsilon: float = 1e-6,
    do_wgridding: bool = True,
    nthreads: int = 0,
    verbosity: int = 0,
    flip_u: bool = False,
    flip_v: bool = False,
    flip_w: bool = False,
    divide_by_n: bool = True,
    sigma_min: float = 1.1,
    sigma_max: float = 2.6,
    center_x: float = 0.0,
    center_y: float = 0.0,
    allow_nshift: bool = True,
    double_precision_accumulation: bool = False,
    min_valid_channels: int = 24,
) -> np.ndarray:
    """
    Grid an in-memory visibility chunk into a dirty image time cube for a specified trial DM.
    Supports in-memory dedispersion and exact per-channel UVW baseline migration gridding.
    """
    nbl, nchan, nt = vis_3d.shape
    engine = None
    if dm != 0.0 or collapse_channels:
        from fastducc.dedisp import DedispersionPlan, BruteForceDedisperser
        plan = DedispersionPlan(chan_freq, dt, dm_min=dm, dm_max=dm)
        engine = BruteForceDedisperser(plan, dm, align_to=align_to)
        vis_dedisp, wgt_dedisp, eff_freq = engine.dedisperse_chunk(
            vis_3d, wgt_3d, iblock=0, collapse_channels=collapse_channels, nsubbands=nsubbands, use_history=False
        )
    else:
        vis_dedisp = vis_3d
        wgt_dedisp = wgt_3d
        eff_freq = chan_freq

    use_exact = bool(exact_uvw and (dm != 0.0) and (not collapse_channels) and (engine is not None))
    freq_1chan = np.array([c_light], dtype=np.float64)
    freq_scale = (eff_freq[None, :, None] / c_light) if use_exact else None

    # Vectorized precomputation of valid channel count per time snapshot
    has_valid_raw_wgts = np.any(wgt_3d > 0.0, axis=0)  # (nchan, nt)
    raw_valid_chans_per_t = np.count_nonzero(has_valid_raw_wgts, axis=0)  # (nt,)
    req_chans = min(min_valid_channels, nchan) if min_valid_channels > 0 else 1

    has_valid_dedisp_wgts = np.any(wgt_dedisp > 0.0, axis=0)
    dedisp_valid_chans_per_t = np.count_nonzero(has_valid_dedisp_wgts, axis=0)

    cube = np.empty((nt, npix_y, npix_x), dtype=np.float64)
    dirty_dtype = np.float32 if (vis_dedisp.dtype == np.complex64) else np.float64
    dirty_buf = np.empty((npix_x, npix_y), dtype=dirty_dtype)

    for t_idx in range(nt):
        if collapse_channels:
            if (raw_valid_chans_per_t[t_idx] < req_chans) or (dedisp_valid_chans_per_t[t_idx] < 1):
                cube[t_idx, :, :] = 0.0
                continue
        else:
            if dedisp_valid_chans_per_t[t_idx] < req_chans:
                cube[t_idx, :, :] = 0.0
                continue

        vis_snap = vis_dedisp[:, :, t_idx]
        wgt_snap = wgt_dedisp[:, :, t_idx]

        if use_exact:
            # Physical Dispersion Delay & Earth-Rotation Baseline Migration:
            # Pulse arrivals across the frequency band are delayed by tau_DM(nu).
            # Because baselines rotate with the Earth during this dispersion sweep,
            # each channel c samples visibility at physical timestamp t_phys = t_idx + delays[c].
            t_phys = np.clip(t_idx + engine.delays, 0, nt - 1)
            uvw_snap_m = uvw_3d[:, :, t_phys].transpose(0, 2, 1)  # (nbl, nchan, 3) in meters

            # Dimensionless Wavelength Transformation:
            uvw_lambda = uvw_snap_m * freq_scale  # cycles

            # Pseudo-Row Representation: (nbl * nchan) independent 1-channel samples
            uvw_input = uvw_lambda.reshape(-1, 3)
            vis_input = vis_snap.reshape(-1, 1)
            wgt_input = wgt_snap.reshape(-1, 1)
            freq_input = freq_1chan
        else:
            # Standard path: fixed baseline UVW across all channels at snapshot reference time
            uvw_input = uvw_3d[:, :, t_idx]
            freq_input = eff_freq
            vis_input = vis_snap
            wgt_input = wgt_snap

        wgt_max = float(np.max(wgt_input)) if wgt_input.size > 0 else 0.0
        if wgt_max <= 0.0:
            cube[t_idx, :, :] = 0.0
            continue

        wgt_norm = (wgt_input / wgt_max).astype(np.float32, copy=False)
        n_valid = int(np.count_nonzero(wgt_norm))
        if n_valid == 0:
            cube[t_idx, :, :] = 0.0
            continue

        ducc0.wgridder.vis2dirty(
            uvw=uvw_input,
            freq=freq_input,
            vis=vis_input,
            wgt=wgt_norm,
            npix_x=npix_x,
            npix_y=npix_y,
            pixsize_x=pixsize_x,
            pixsize_y=pixsize_y,
            epsilon=epsilon,
            do_wgridding=do_wgridding,
            nthreads=nthreads,
            verbosity=verbosity,
            flip_u=flip_u,
            flip_v=flip_v,
            flip_w=flip_w,
            divide_by_n=divide_by_n,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            center_x=center_x,
            center_y=center_y,
            allow_nshift=allow_nshift,
            double_precision_accumulation=double_precision_accumulation,
            dirty=dirty_buf,
        )
        if divide_by_n and n_valid > 0:
            cube[t_idx, :, :] = dirty_buf.T / n_valid
        else:
            cube[t_idx, :, :] = dirty_buf.T

    if vis_dedisp is not vis_3d:
        del vis_dedisp, wgt_dedisp

    return cube
