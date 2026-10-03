import sys
from typing import Any, Tuple

import numpy as np
from tqdm import tqdm

from casacore.tables import table

try:
    import ducc0
except Exception as e:
    raise RuntimeError('ducc0 is required') from e

from fastducc import ms_utils

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
):
    """
    Iterate over time samples of a Measurement Set and grid visibilities
    into dirty images using ducc0.wgridder.vis2dirty.

    Parameters
    ----------
    start_time_idx : int | None
        0-based index of the first time chunk to process (inclusive). If None, start at 0.
    end_time_idx : int | None
        0-based index of the last time chunk to process (inclusive). If None, process until the end.

    Returns
    -------
    list[tuple[float, np.ndarray]]
        A list of (time_value, dirty_image) for each processed time sample.
    """
    #print("hi")

    if t_main is None:
        print("reading table from msname {msname}")
        t_main = table(msname, readonly=True)
        
    # Frequencies
    t_spw = table(f"{msname}/SPECTRAL_WINDOW", readonly=True)
    n_spw = t_spw.nrows()
    if n_spw != 1:
        raise ValueError(f"image_time_samples() currently supports a single SPW; found {n_spw}")
    chan_freq = t_spw.getcell('CHAN_FREQ', 0)  # [nchan] Hz
    t_spw.close()

    #set up iterator over table, grouped by common timestamps
    
    #declare results list as list of tuple of float (for timestamp) and ndarray (for image)
    #results: list[tuple[float, np.ndarray]] = []

    
    labels, lbl2idx = ms_utils.get_corr_label_indices(msname)
    
    uvw   = t_main.getcol('UVW')
    data  = t_main.getcol(data_column)   # [nrow, nchan, ncorr]
    flags = t_main.getcol('FLAG')        # [nrow, nchan, ncorr]
    flag_row = t_main.getcol('FLAG_ROW') if 'FLAG_ROW' in set(t_main.colnames()) else None
    
    # Weights
    if use_weight_spectrum and 'WEIGHT_SPECTRUM' in set(t_main.colnames()):
        wgt = t_main.getcol('WEIGHT_SPECTRUM')
    else:
        wgt_row = t_main.getcol('WEIGHT')
        wgt = np.broadcast_to(wgt_row[:, None, :], data.shape)

    print("applying flags")
        
    # Apply flags -> zero weights
    good = ~flags
    if flag_row is not None:
        good &= (~flag_row[:, None, None])
    wgt = np.where(good, wgt, 0.0)

    # Correlation collapse
    if corr_mode == 'average':
        if average_correlations:
            wsum = wgt.sum(axis=2)
            with np.errstate(invalid='ignore', divide='ignore'):
                vis = (data * wgt).sum(axis=2) / np.where(wsum > 0.0, wsum, np.nan)
            vis = np.nan_to_num(vis, nan=0.0)
            wgt_2d = wsum
        else:
            if corr_index is None:
                corr_index = 0
            vis    = data[:, :, corr_index]
            wgt_2d = wgt[:, :, corr_index]
    elif corr_mode == 'single':
        if single_pol not in lbl2idx:
            raise ValueError(f"Requested single_pol='{single_pol}' not present in MS correlations: {labels}")
        ci = lbl2idx[single_pol]
        vis    = data[:, :, ci]
        wgt_2d = wgt[:, :, ci]
    elif corr_mode == 'stokesI':
        have_linear   = ('XX' in lbl2idx) and ('YY' in lbl2idx)
        have_circular = ('RR' in lbl2idx) and ('LL' in lbl2idx)
        use_linear = False
        use_circ   = False
        if basis == 'linear':
            use_linear = have_linear
            if not use_linear:
                raise ValueError("basis='linear' requested but XX/YY not found in MS correlations")
        elif basis == 'circular':
            use_circ = have_circular
            if not use_circ:
                raise ValueError("basis='circular' requested but RR/LL not found in MS correlations")
        else:
            if have_linear:
                use_linear = True
            elif have_circular:
                use_circ = True
            else:
                raise ValueError("Cannot form Stokes I: XX/YY or RR/LL not present in MS correlations")
        if use_linear:
            i1, i2 = lbl2idx['XX'], lbl2idx['YY']
        else:
            i1, i2 = lbl2idx['RR'], lbl2idx['LL']
        v1, w1 = data[:, :, i1], wgt[:, :, i1]
        v2, w2 = data[:, :, i2], wgt[:, :, i2]
        present1 = (w1 > 0.0)
        present2 = (w2 > 0.0)
        n_valid  = present1.astype(np.int32) + present2.astype(np.int32)
        sum_vis = np.zeros_like(v1)
        sum_vis += np.where(present1, v1, 0.0)
        sum_vis += np.where(present2, v2, 0.0)
        with np.errstate(invalid='ignore', divide='ignore'):
            vis = sum_vis / np.where(n_valid > 0, n_valid, np.nan)
        vis = np.nan_to_num(vis, nan=0.0)
        with np.errstate(invalid='ignore', divide='ignore'):
            w_two = 4.0 / (np.where(present1, 1.0 / w1, 0.0) + np.where(present2, 1.0 / w2, 0.0))
        w_one = np.where(present1 & (~present2), w1, 0.0) + np.where((~present1) & present2, w2, 0.0)
        wgt_2d = np.where(n_valid == 2, np.nan_to_num(w_two, nan=0.0, posinf=0.0, neginf=0.0), w_one)
    else:
            raise ValueError(f"Unknown corr_mode='{corr_mode}'. Use 'average', 'stokesI', or 'single'.")

    if uvw.shape[0] != vis.shape[0]:
            raise ValueError('Row count mismatch between UVW and VIS')
    if chan_freq.shape[0] != vis.shape[1]:
            raise ValueError('Channel count mismatch between CHAN_FREQ and VIS')

    #print(np.max(wgt_2d), np.median(wgt_2d), np.min(wgt_2d))
    wgt_2d /= np.max(wgt_2d)
        
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
    #immediately transpose the data
    dirty = dirty.T
    
    n_valid = int(np.count_nonzero(wgt_2d)) #np.sum(wgt_2d/np.max(wgt_2d)) #
    if n_valid > 0:
        #I think divide_by_n should be dealing with this already but whatever
        dirty = dirty / n_valid

    return dirty
    
def image_time_samples(
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
    do_plot: bool = False,
    dm: float = 0.0,
    collapse_channels: bool = False,
    nsubbands: int = 1,
    align_to: str = "fmax",
    exact_uvw: bool = True,
    min_valid_channels: int = 24,
):
    """
    Iterate over time samples of a Measurement Set and grid visibilities
    into dirty images using ducc0.wgridder.vis2dirty.
    Supports brute-force dedispersion, channel collapsing, and exact per-channel UVW gridding.

    Parameters
    ----------
    start_time_idx : int | None
        0-based index of the first time chunk to process (inclusive). If None, start at 0.
    end_time_idx : int | None
        0-based index of the last time chunk to process (inclusive). If None, process until the end.
    dm : float
        Dispersion Measure in pc/cm^3. If != 0, frequency channels are shifted in time.
    collapse_channels : bool
        If True, frequency channels are averaged/collapsed before gridding into snapshot images.
    nsubbands : int
        Number of subbands when collapsing frequency channels.
    align_to : {'fmax', 'fmin'}
        Reference frequency alignment for dispersion delays.
    exact_uvw : bool
        If True, calculate exact per-channel UVW coordinates accounting for baseline migration.
    min_valid_channels : int
        Minimum number of non-zero-weight channels required across the band before gridding.
        Timesteps with fewer valid channels are zeroed out (default: 24).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (times, cube) where cube has shape (nt, npix_y, npix_x).
    """
    opened_locally = False
    if t_main is None:
        t_main = table(msname, readonly=True)
        opened_locally = True

    colnames = set(t_main.colnames())
    time_col = 'TIME_CENTROID' if 'TIME_CENTROID' in colnames else 'TIME'

    # Frequencies
    t_spw = table(f"{msname}/SPECTRAL_WINDOW", readonly=True)
    n_spw = t_spw.nrows()
    if n_spw != 1:
        raise ValueError(f"image_time_samples() currently supports a single SPW; found {n_spw}")
    chan_freq = t_spw.getcell('CHAN_FREQ', 0)  # [nchan] Hz
    t_spw.close()

    if chunk_times is not None:
        times = chunk_times
        nt_window = len(times)
        # Fast query selecting only rows in the time window
        query_str = f"{time_col} >= {times[0]:.17g} AND {time_col} <= {times[-1]:.17g}"
        t_chunk_window = t_main.query(query_str)
    else:
        nsub, vis_time, all_times = ms_utils.get_time(t_main)
        start_idx = 0 if start_time_idx is None else int(start_time_idx)
        end_idx   = (nsub - 1) if end_time_idx is None else int(end_time_idx)
        if start_idx < 0 or end_idx < start_idx or end_idx >= nsub:
            raise ValueError("Invalid start/end time indices")
        times = all_times[start_idx:end_idx+1]
        nt_window = end_idx - start_idx + 1
        query_str = f"{time_col} >= {times[0]:.17g} AND {time_col} <= {times[-1]:.17g}"
        t_chunk_window = t_main.query(query_str)

    labels, lbl2idx = ms_utils.get_corr_label_indices(msname)

    # Dedispersed / channel-collapsed imaging path
    if dm != 0.0 or collapse_channels:
        from fastducc.dedisp import DedispersionPlan, BruteForceDedisperser
        dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0

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

        # Correlation collapse
        if corr_mode == 'average':
            if average_correlations:
                wsum = wgt_all.sum(axis=2)
                with np.errstate(invalid='ignore', divide='ignore'):
                    vis_flat = (data_all * wgt_all).sum(axis=2) / np.where(wsum > 0.0, wsum, np.nan)
                vis_flat = np.nan_to_num(vis_flat, nan=0.0)
                wgt_flat = wsum
            else:
                ci = 0 if corr_index is None else corr_index
                vis_flat = data_all[:, :, ci]
                wgt_flat = wgt_all[:, :, ci]
        elif corr_mode == 'single':
            if single_pol not in lbl2idx:
                raise ValueError(f"Requested single_pol='{single_pol}' not present in MS correlations: {labels}")
            ci = lbl2idx[single_pol]
            vis_flat = data_all[:, :, ci]
            wgt_flat = wgt_all[:, :, ci]
        elif corr_mode == 'stokesI':
            have_linear   = ('XX' in lbl2idx) and ('YY' in lbl2idx)
            have_circular = ('RR' in lbl2idx) and ('LL' in lbl2idx)
            use_linear = False
            use_circ   = False
            if basis == 'linear':
                use_linear = have_linear
                if not use_linear:
                    raise ValueError("basis='linear' requested but XX/YY not found in MS correlations")
            elif basis == 'circular':
                use_circ = have_circular
                if not use_circ:
                    raise ValueError("basis='circular' requested but RR/LL not found in MS correlations")
            else:
                if have_linear:
                    use_linear = True
                elif have_circular:
                    use_circ = True
                else:
                    raise ValueError("Cannot form Stokes I: XX/YY or RR/LL not present in MS correlations")
            if use_linear:
                i1, i2 = lbl2idx['XX'], lbl2idx['YY']
            else:
                i1, i2 = lbl2idx['RR'], lbl2idx['LL']
            v1, w1 = data_all[:, :, i1], wgt_all[:, :, i1]
            v2, w2 = data_all[:, :, i2], wgt_all[:, :, i2]
            present1 = (w1 > 0.0)
            present2 = (w2 > 0.0)
            n_valid  = present1.astype(np.int32) + present2.astype(np.int32)
            sum_vis = np.zeros_like(v1)
            sum_vis += np.where(present1, v1, 0.0)
            sum_vis += np.where(present2, v2, 0.0)
            with np.errstate(invalid='ignore', divide='ignore'):
                vis_flat = sum_vis / np.where(n_valid > 0, n_valid, np.nan)
            vis_flat = np.nan_to_num(vis_flat, nan=0.0)
            with np.errstate(invalid='ignore', divide='ignore'):
                w_two = 4.0 / (np.where(present1, 1.0 / w1, 0.0) + np.where(present2, 1.0 / w2, 0.0))
            w_one = np.where(present1 & (~present2), w1, 0.0) + np.where((~present1) & present2, w2, 0.0)
            wgt_flat = np.where(n_valid == 2, np.nan_to_num(w_two, nan=0.0, posinf=0.0, neginf=0.0), w_one)
        else:
            raise ValueError(f"Unknown corr_mode='{corr_mode}'. Use 'average', 'stokesI', or 'single'.")

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

        plan = DedispersionPlan(chan_freq, dt, dm_min=dm, dm_max=dm)
        engine = BruteForceDedisperser(plan, dm, align_to=align_to)
        vis_dedisp, wgt_dedisp, eff_freq = engine.dedisperse_chunk(
            vis_3d, wgt_3d, iblock=0, collapse_channels=collapse_channels, nsubbands=nsubbands, use_history=False
        )

        c_light = 299792458.0
        use_exact = bool(exact_uvw and (dm != 0.0) and (not collapse_channels))
        freq_1chan = np.array([c_light], dtype=np.float64)

        cube = np.empty((nt, npix_y, npix_x), dtype=np.float64)
        for t_idx in range(nt):
            v_snap = vis_dedisp[:, :, t_idx]
            w_snap = wgt_dedisp[:, :, t_idx]

            n_chans_valid = int(np.count_nonzero(np.any(w_snap > 0.0, axis=0)))
            if n_chans_valid < min_valid_channels:
                cube[t_idx, :, :] = 0.0
                continue

            if use_exact:
                # --------------------------------------------------------------------------
                # OPTION A: Exact Per-Channel UVW Tracking & Dimensionless Wavelength Gridding
                # --------------------------------------------------------------------------
                # 1. Physical Dispersion Delay & Baseline Migration:
                #    A dispersed astronomical pulse arrives at different times across the
                #    frequency band. For a dedispersed snapshot at reference time `t_idx`,
                #    the visibility sample in channel `c` was actually acquired at physical
                #    time `t_phys = t_idx + delays[c]`.
                #    Because the Earth rotates during this dispersion sweep, baseline `b`
                #    migrates through (u, v, w) space across channels. Therefore, coordinates
                #    are 3D: shape (nbl, nchan, 3) in meters.
                t_phys = np.clip(t_idx + engine.delays, 0, nt - 1)
                # uvw_3d has shape (nbl, 3, nt); indexing on axis 2 gives (nbl, 3, nchan)
                uvw_snap_m = uvw_3d[:, :, t_phys].transpose(0, 2, 1)  # (nbl, nchan, 3) in meters

                # 2. Dimensionless Wavelength Transformation:
                #    DUCC's internal gridding algorithm operates in units of wavelengths:
                #        u_lambda = u_meters * (freq_c / c)
                #    Normally DUCC calculates this internally per channel from a single
                #    per-row coordinate in meters. Because each channel here has its own
                #    migrated physical coordinate in meters, we pre-convert the coordinates
                #    directly into dimensionless wavelengths (cycles):
                #        u_lambda = u_meters * (freq_c / c)
                uvw_lambda = uvw_snap_m * (eff_freq[None, :, None] / c_light)

                # 3. 1D Pseudo-Row Representation:
                #    We flatten the spatial baselines and frequency channels into N_vis = (nbl * nchan)
                #    independent 1-channel visibility samples. Each sample has:
                #        - its exact pre-scaled (u, v, w) coordinate in wavelengths, shape (N_vis, 3)
                #        - its visibility and weight values, shaped as 1-channel arrays: (N_vis, 1)
                #        - a reference frequency array [c_light], ensuring DUCC's internal scaling
                #          factor (f / c_light) equals exactly 1.0, preserving our wavelength coordinates.
                #
                #    DUCC's C++ AVX/SIMD vectorization operates across the Kaiser-Bessel convolution
                #    support footprint in the image plane (not across the channel dimension), and its
                #    multithreading operates across spatial (u, v) tiles and w-planes. Thus, flattening
                #    into 1-channel pseudo-rows maintains full C++ compiled execution speed (~10 ms
                #    per snapshot) while guaranteeing exact per-channel geometric accuracy.
                uvw_input = uvw_lambda.reshape(-1, 3)
                vis_input = v_snap.reshape(-1, 1)
                wgt_input = w_snap.reshape(-1, 1).copy()
                freq_input = freq_1chan
            else:
                # Standard path: fixed baseline UVW across all channels at snapshot reference time
                uvw_input = uvw_3d[:, :, t_idx]
                freq_input = eff_freq
                vis_input = v_snap
                wgt_input = w_snap.copy()

            w_max = np.max(wgt_input)
            if w_max > 0:
                wgt_input = wgt_input / w_max

            dirty = ducc0.wgridder.vis2dirty(
                uvw=uvw_input,
                freq=freq_input,
                vis=vis_input,
                wgt=wgt_input,
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
            n_valid = int(np.count_nonzero(wgt_input))
            if n_valid > 0:
                dirty = dirty / n_valid
            cube[t_idx, :, :] = dirty

        t_chunk_window.close()
        if opened_locally:
            t_main.close()

        return u_times, cube

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

    colnames = set(t_main.colnames())
    time_col = 'TIME_CENTROID' if 'TIME_CENTROID' in colnames else 'TIME'

    # Frequencies
    t_spw = table(f"{msname}/SPECTRAL_WINDOW", readonly=True)
    n_spw = t_spw.nrows()
    if n_spw != 1:
        raise ValueError(f"fastducc currently supports a single SPW; found {n_spw}")
    chan_freq = t_spw.getcell('CHAN_FREQ', 0)  # [nchan] Hz
    t_spw.close()

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

    # Correlation collapse
    if corr_mode == 'average':
        if average_correlations:
            wsum = wgt_all.sum(axis=2)
            with np.errstate(invalid='ignore', divide='ignore'):
                vis_flat = (data_all * wgt_all).sum(axis=2) / np.where(wsum > 0.0, wsum, np.nan)
            vis_flat = np.nan_to_num(vis_flat, nan=0.0)
            wgt_flat = wsum
        else:
            ci = 0 if corr_index is None else corr_index
            vis_flat = data_all[:, :, ci]
            wgt_flat = wgt_all[:, :, ci]
    elif corr_mode == 'single':
        if single_pol not in lbl2idx:
            raise ValueError(f"Requested single_pol='{single_pol}' not present in MS correlations: {labels}")
        ci = lbl2idx[single_pol]
        vis_flat = data_all[:, :, ci]
        wgt_flat = wgt_all[:, :, ci]
    elif corr_mode == 'stokesI':
        have_linear   = ('XX' in lbl2idx) and ('YY' in lbl2idx)
        have_circular = ('RR' in lbl2idx) and ('LL' in lbl2idx)
        use_linear = False
        use_circ   = False
        if basis == 'linear':
            use_linear = have_linear
            if not use_linear:
                raise ValueError("basis='linear' requested but XX/YY not found in MS correlations")
        elif basis == 'circular':
            use_circ = have_circular
            if not use_circ:
                raise ValueError("basis='circular' requested but RR/LL not found in MS correlations")
        else:
            if have_linear:
                use_linear = True
            elif have_circular:
                use_circ = True
            else:
                raise ValueError("Cannot form Stokes I: XX/YY or RR/LL not present in MS correlations")
        if use_linear:
            i1, i2 = lbl2idx['XX'], lbl2idx['YY']
        else:
            i1, i2 = lbl2idx['RR'], lbl2idx['LL']
        v1, w1 = data_all[:, :, i1], wgt_all[:, :, i1]
        v2, w2 = data_all[:, :, i2], wgt_all[:, :, i2]
        present1 = (w1 > 0.0)
        present2 = (w2 > 0.0)
        n_valid  = present1.astype(np.int32) + present2.astype(np.int32)
        sum_vis = np.zeros_like(v1)
        sum_vis += np.where(present1, v1, 0.0)
        sum_vis += np.where(present2, v2, 0.0)
        with np.errstate(invalid='ignore', divide='ignore'):
            vis_flat = sum_vis / np.where(n_valid > 0, n_valid, np.nan)
        vis_flat = np.nan_to_num(vis_flat, nan=0.0)
        with np.errstate(invalid='ignore', divide='ignore'):
            w_two = 4.0 / (np.where(present1, 1.0 / w1, 0.0) + np.where(present2, 1.0 / w2, 0.0))
        w_one = np.where(present1 & (~present2), w1, 0.0) + np.where((~present1) & present2, w2, 0.0)
        wgt_flat = np.where(n_valid == 2, np.nan_to_num(w_two, nan=0.0, posinf=0.0, neginf=0.0), w_one)
    else:
        raise ValueError(f"Unknown corr_mode='{corr_mode}'. Use 'average', 'stokesI', or 'single'.")

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

    t_chunk_window.close()
    if opened_locally:
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
    Supports in-memory dedispersion and Option A exact per-channel UVW gridding.
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

    c_light = 299792458.0
    use_exact = bool(exact_uvw and (dm != 0.0) and (not collapse_channels) and (engine is not None))
    freq_1chan = np.array([c_light], dtype=np.float64)

    cube = np.empty((nt, npix_y, npix_x), dtype=np.float64)
    for t_idx in range(nt):
        v_snap = vis_dedisp[:, :, t_idx]
        w_snap = wgt_dedisp[:, :, t_idx]

        n_chans_valid = int(np.count_nonzero(np.any(w_snap > 0.0, axis=0)))
        if n_chans_valid < min_valid_channels:
            cube[t_idx, :, :] = 0.0
            continue

        if use_exact:
            # --------------------------------------------------------------------------
            # Exact Per-Channel UVW Tracking & Dimensionless Wavelength Gridding
            # --------------------------------------------------------------------------
            # 1. Physical Dispersion Delay & Baseline Migration:
            #    A dispersed astronomical pulse arrives at different times across the
            #    frequency band. For a dedispersed snapshot at reference time `t_idx`,
            #    the visibility sample in channel `c` was actually acquired at physical
            #    time `t_phys = t_idx + delays[c]`.
            #    Because the Earth rotates during this dispersion sweep, baseline `b`
            #    migrates through (u, v, w) space across channels. Therefore, coordinates
            #    are 3D: shape (nbl, nchan, 3) in meters.
            t_phys = np.clip(t_idx + engine.delays, 0, nt - 1)
            # uvw_3d has shape (nbl, 3, nt); indexing on axis 2 gives (nbl, 3, nchan)
            uvw_snap_m = uvw_3d[:, :, t_phys].transpose(0, 2, 1)  # (nbl, nchan, 3) in meters

            # 2. Dimensionless Wavelength Transformation:
            #    DUCC's internal gridding algorithm operates in units of wavelengths:
            #        u_lambda = u_meters * (freq_c / c)
            #    Normally DUCC calculates this internally per channel from a single
            #    per-row coordinate in meters. Because each channel here has its own
            #    migrated physical coordinate in meters, we pre-convert the coordinates
            #    directly into dimensionless wavelengths (cycles):
            #        u_lambda = u_meters * (freq_c / c)
            uvw_lambda = uvw_snap_m * (eff_freq[None, :, None] / c_light)

            # 3. 1D Pseudo-Row Representation:
            #    We flatten the spatial baselines and frequency channels into N_vis = (nbl * nchan)
            #    independent 1-channel visibility samples. Each sample has:
            #        - its exact pre-scaled (u, v, w) coordinate in wavelengths, shape (N_vis, 3)
            #        - its visibility and weight values, shaped as 1-channel arrays: (N_vis, 1)
            #        - a reference frequency array [c_light], ensuring DUCC's internal scaling
            #          factor (f / c_light) equals exactly 1.0, preserving our wavelength coordinates.
            #
            #    DUCC's C++ AVX/SIMD vectorization operates across the Kaiser-Bessel convolution
            #    support footprint in the image plane (not across the channel dimension), and its
            #    multithreading operates across spatial (u, v) tiles and w-planes. Thus, flattening
            #    into 1-channel pseudo-rows maintains full C++ compiled execution speed (~10 ms
            #    per snapshot) while guaranteeing exact per-channel geometric accuracy.
            uvw_input = uvw_lambda.reshape(-1, 3)
            vis_input = v_snap.reshape(-1, 1)
            wgt_input = w_snap.reshape(-1, 1).copy()
            freq_input = freq_1chan
        else:
            # Standard path: fixed baseline UVW across all channels at snapshot reference time
            uvw_input = uvw_3d[:, :, t_idx]
            freq_input = eff_freq
            vis_input = v_snap
            wgt_input = w_snap.copy()

        w_max = np.max(wgt_input)
        if w_max > 0:
            wgt_input = wgt_input / w_max

        dirty = ducc0.wgridder.vis2dirty(
            uvw=uvw_input,
            freq=freq_input,
            vis=vis_input,
            wgt=wgt_input,
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
        n_valid = int(np.count_nonzero(wgt_input))
        if n_valid > 0:
            dirty = dirty / n_valid
        cube[t_idx, :, :] = dirty

    return cube
