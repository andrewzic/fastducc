import argparse
import glob
import os
import shutil
import sys
import astropy.constants as const
import astropy.units as u
from casacore.tables import table

from typing import Iterable, Tuple, List, Dict, Any


# CASA POLARIZATION CORR_TYPE integer codes to labels
_CORR_CODE_TO_NAME = {
    4: 'RR', 5: 'RL', 6: 'LR', 7: 'LL',
    8: 'XX', 9: 'XY', 10: 'YX', 11: 'YY',
}

def _get_corr_label_indices(msname: str):
    """Return (labels_list, label->index dict) from POLARIZATION/CORR_TYPE."""
    t_pol = table(f"{msname}/POLARIZATION", readonly=True)
    corr_types = t_pol.getcell('CORR_TYPE', 0)
    t_pol.close()
    labels = [ _CORR_CODE_TO_NAME.get(int(c), str(c)) for c in corr_types ]
    lbl2idx = {lab: i for i, lab in enumerate(labels)}
    return labels, lbl2idx

def parse_args():
    parser = argparse.ArgumentParser(description="Run CASA applycal on MS files for specified beams (SBID-aware).")
    parser.add_argument("--msname", required=True, help="measurement set name")
    parser.add_argument("--cube-length", default=1000, help="maximum length of output cube chunk")
    parser.add_argument("--img-interval", default=1, help="imaging interval in units of samples")    
    parser.add_argument("--channels-out", default=1, help="number of channels to write out")
    parser.add_argument("--do-uv", default=1, help="also save uv grid cube")
    return parser.parse_args()

def get_channel_lambdas(ms):
    tf = table(f"{ms}/SPECTRAL_WINDOW")
    # get all channel frequencies and convert that to wavelengths
    channel_freqs = tf[0]["CHAN_FREQ"]
    nchan = len(channel_freqs)
    channel_lambdas = const.c.to(u.m/u.s).value / channel_freqs
    tf.close()
    return nchan, channel_freqs, channel_lambdas


def get_time(t):
    vis_time = t.getcol('TIME_CENTROID')
    unique_times = np.unique(vis_time)
    assert np.all(np.abs(np.diff(unique_times) - np.diff(unique_times)[0]) < 1e-2)
    nsub = unique_times.shape[0]
    return nsub, vis_time, unique_times

def get_uvwave_indices(t, channel_lambdas, uvcell_size, N_pixels):    
    #  get the uv positions (in lambdas) for each visibility sample
    uvws = t.getcol('UVW') # in units of metres
    #populate uvw_l grid
    uvws_l = np.ones((uvws.shape[0], uvws.shape[1], len(channel_lambdas))) * uvws[:, :, None]
    # duplicate each uv sample into a size such that we can multiply with the channel lambdas    
    #uvs_l = uvws_l[:, :2, :]        # gets rid of w axis. DONT FORGET: need to do w-projection! can't just ignore the w axis like im doing here
    
    chan_tiled = np.ones_like(uvws_l)*channel_lambdas[None, None, :] # arrange and repeat the channel lambdas in an array shape such that we can multiply it onto the uv samples
    
    uvws_l = uvws_l / chan_tiled    # get the uvw positions in lambdas


def image_time_samples(
    msname: str,
    *,
    start_time_idx: int | None = None,
    end_time_idx: int | None = None,
    data_column: str = 'CORRECTED_DATA',
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

    try:
        import ducc0
    except Exception as e:
        raise RuntimeError('ducc0 is required for image_time_samples()') from e

    t_main = table(msname, readonly=True)
    colnames = set(t_main.colnames())
    time_col = 'TIME_CENTROID' if 'TIME_CENTROID' in colnames else 'TIME'

    # Frequencies
    t_spw = table(f"{msname}/SPECTRAL_WINDOW", readonly=True)
    n_spw = t_spw.nrows()
    if n_spw != 1:
        raise ValueError(f"image_time_samples() currently supports a single SPW; found {n_spw}")
    chan_freq = t_spw.getcell('CHAN_FREQ', 0)  # [nchan] Hz
    t_spw.close()

    #set up iterator over table, grouped by common timestamps
    it = t_main.iter([time_col], sort=True)
    #declare results list as list of tuple of float (for timestamp) and ndarray (for image)
    #results: list[tuple[float, np.ndarray]] = []

    
    nsub, vis_time, times = get_time(t_main)
    #pre initialise img cube in mem and fill as we go
    # the number of time samples should be the min of (N - start_idx, end_idx - start_idx, N) where N is no. of time samples
    cube = np.empty((min([len(times) - start_time_idx, end_time_idx - start_time_idx, len(times)]), npix_y, npix_x))#, dtype=dtype_img)
    
    cube_idx = 0
    for t_chunk_idx, t_chunk in enumerate(it):
        # Apply start/end time-chunk windowing
        if start_time_idx is not None and t_chunk_idx < start_time_idx:
            continue
        if end_time_idx is not None and t_chunk_idx > end_time_idx:
            break

        times = t_chunk.getcol(time_col)
        time_val = float(times[0])

        uvw   = t_chunk.getcol('UVW')
        data  = t_chunk.getcol(data_column)   # [nrow, nchan, ncorr]
        flags = t_chunk.getcol('FLAG')        # [nrow, nchan, ncorr]
        flag_row = t_chunk.getcol('FLAG_ROW') if 'FLAG_ROW' in set(t_chunk.colnames()) else None

        # Weights
        if use_weight_spectrum and 'WEIGHT_SPECTRUM' in set(t_chunk.colnames()):
            wgt = t_chunk.getcol('WEIGHT_SPECTRUM')
        else:
            wgt_row = t_chunk.getcol('WEIGHT')
            wgt = np.broadcast_to(wgt_row[:, None, :], data.shape)

        # Apply flags -> zero weights
        good = ~flags
        if flag_row is not None:
            good &= (~flag_row[:, None, None])
        wgt = np.where(good, wgt, 0.0)

        # Correlation collapse
        labels, lbl2idx = _get_corr_label_indices(msname)
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
        cube[cube_idx, :, :] = dirty
        cube_idx += 1

        #results.append((time_val, dirty))
        
    it.close()
    t_main.close()
    
    results = (times, cube)    
    return results


def boxcar_search_time(
    times: np.ndarray,
    cube: np.ndarray,                      # shape (T, Ny, Nx)
    widths: Iterable[float],               # list of widths; samples (int) or seconds (float)
    *,
    widths_in_seconds: bool = False,
    threshold_sigma: float = 5.0,
    return_snr_cubes: bool = False,
    keep_top_k: int | None = None,         # per width, cap detections to top-K by SNR
    valid_mask: np.ndarray | None = None,  # optional (Ny, Nx) mask of pixels to search
    subtract_mean_per_pixel: bool = False, # subtract static mean over time before search
) -> Tuple[List[Dict[str, Any]], Dict[int, np.ndarray] | None]:
    """
    Boxcar (matched-filter) search along the time axis of an image cube.

    Parameters
    ----------
    times : (T,) float array
        Time stamps (e.g., seconds/MJD). Assumed monotonic and ~uniform.
    cube : (T, Ny, Nx) array
        Image cube stacked along time (e.g., dirty images per time chunk).
    widths : iterable of float
        Boxcar widths. If `widths_in_seconds=False`, treat as *integer samples*.
        If `widths_in_seconds=True`, convert seconds to nearest integer samples.
    widths_in_seconds : bool
        Interpret `widths` as seconds (converted using median dt) if True.
    threshold_sigma : float
        Report detections with SNR >= threshold_sigma.
    return_snr_cubes : bool
        If True, returns a dict: {width_samples: SNR_cube} with shape (T_eff, Ny, Nx)
        where T_eff = T - width_samples + 1.
    keep_top_k : int | None
        If set, keep only the top-K detections (by SNR) per width.
    valid_mask : (Ny, Nx) bool array | None
        If provided, only search pixels where valid_mask==True.
    subtract_mean_per_pixel : bool
        If True, subtract each pixel's time-mean before filtering (high-pass).

    Returns
    -------
    detections : list of dict
        Each detection has keys:
          - 'time_center': float
          - 'y': int, 'x': int
          - 'width_samples': int
          - 'snr': float
          - 'value_sum': float   (sum in the boxcar window)
    snr_cubes : dict | None
        If return_snr_cubes=True, maps width -> SNR cube (time-convolved),
        otherwise None.

    Notes
    -----
    - SNR computed as: SNR = sum / (sqrt(w) * sigma), where
      sigma is the moving std estimated from the same window via
      cumulative sums of x and x^2 (per pixel).
    - Time center reported as the midpoint of the window indices.
    - Assumes relatively uniform time sampling; for irregular times
      with big gaps, consider widths_in_seconds=True and validate dt.
    """
    # Basic shape checks
    if cube.ndim != 3:
        raise ValueError("cube must be (T, Ny, Nx)")
    T, Ny, Nx = cube.shape
    if times.shape[0] != T:
        raise ValueError("times length must match cube time axis")

    # Optional pixel mask
    if valid_mask is None:
        valid_mask = np.ones((Ny, Nx), dtype=bool)
    else:
        if valid_mask.shape != (Ny, Nx):
            raise ValueError("valid_mask must be shape (Ny, Nx)")

    # Optionally subtract mean per pixel to suppress static baselines
    data = cube.astype(np.float64, copy=False)  # use float64 for precision
    if subtract_mean_per_pixel:
        mean_map = data.mean(axis=0, keepdims=True)  # (1, Ny, Nx)
        data = data - mean_map

    # Convert widths to integer samples if they are in seconds
    widths_samples: List[int] = []
    if widths_in_seconds:
        # Use median dt for conversion
        dt = np.median(np.diff(times))
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Non-positive or invalid dt; cannot convert seconds to samples.")
        for w_sec in widths:
            w_samp = int(round(w_sec / dt))
            if w_samp < 1:
                w_samp = 1
            widths_samples.append(w_samp)
    else:
        for w in widths:
            w_samp = int(w)
            if w_samp < 1:
                w_samp = 1
            widths_samples.append(w_samp)

    detections: List[Dict[str, Any]] = []
    snr_cubes: Dict[int, np.ndarray] | None = {} if return_snr_cubes else None

    # Precompute cumulative sums along time for sum and sum of squares
    # Shapes: (T+1, Ny, Nx) for easy window [t0, t0+w)
    csum   = np.zeros((T + 1, Ny, Nx), dtype=np.float64)
    csum2  = np.zeros((T + 1, Ny, Nx), dtype=np.float64)
    csum[1:]  = np.cumsum(data, axis=0)
    csum2[1:] = np.cumsum(data * data, axis=0)

    # Helper to extract moving sum and std for a given width w
    def moving_sum_and_std(w: int):
        # sum over [t0, t0+w) for t0 = 0..T-w
        s   = csum[w:]  - csum[:-w]     # (T-w+1, Ny, Nx)
        s2  = csum2[w:] - csum2[:-w]
        # variance in window: E[x^2] - E[x]^2
        mean = s / w
        var  = s2 / w - mean * mean
        # numerical stability: clamp small negatives to zero
        var = np.clip(var, a_min=0.0, a_max=None)
        std = np.sqrt(var)
        return s, std

    for w in widths_samples:
        if w > T:
            # window longer than data; skip
            continue

        s, std = moving_sum_and_std(w)               # (T_eff, Ny, Nx)
        T_eff = s.shape[0]
        # SNR = sum / (sqrt(w)*std); avoid division by zero
        denom = std * np.sqrt(w)
        with np.errstate(divide='ignore', invalid='ignore'):
            snr = np.where(denom > 0.0, s / denom, 0.0)

        # Apply pixel mask
        if valid_mask is not None:
            snr = np.where(valid_mask[None, :, :], snr, 0.0)

        # Optionally return the SNR cube
        if return_snr_cubes:
            snr_cubes[w] = snr.astype(np.float32, copy=False)

        # Threshold and gather detections
        hits = np.where(snr >= threshold_sigma)
        # hits is a tuple (t0_idx, y, x)
        t0_idx, ys, xs = hits
        if t0_idx.size == 0:
            continue

        # Time-center for each window [t0, t0+w)
        # Approximate center as times[t0 + w//2]
        centers = times[t0_idx + (w // 2)]

        # If keep_top_k is set, sort by SNR and keep top-K
        if keep_top_k is not None and t0_idx.size > keep_top_k:
            order = np.argsort(snr[t0_idx, ys, xs])[::-1]  # descending
            sel   = order[:keep_top_k]
            t0_idx, ys, xs, centers = t0_idx[sel], ys[sel], xs[sel], centers[sel]

        # Append detections
        for i in range(t0_idx.size):
            detections.append({
                "time_center": float(centers[i]),
                "y": int(ys[i]),
                "x": int(xs[i]),
                "width_samples": int(w),
                "snr": float(snr[t0_idx[i], ys[i], xs[i]]),
                "value_sum": float(s[t0_idx[i], ys[i], xs[i]]),
            })

    return detections, snr_cubes




def main():
    parser = argparse.ArgumentParser(description='Image MS in time chunks using ducc0 wgridder')
    parser.add_argument('--msname', required=True, help='Path to Measurement Set')
    parser.add_argument('--chunk-size', type=int, default=1000,
                        help='Number of time samples per chunk (default: 1000)')
    parser.add_argument('--corr-mode', choices=['average','stokesI','single'], default='single',
                        help='Correlation handling mode (default: single)')
    parser.add_argument('--basis', choices=['auto','linear','circular'], default='linear',
                        help='Basis for stokesI (default: auto)')
    parser.add_argument('--single-pol', default='XX',
                        help='Single pol to image when corr-mode=single (default: XX)')
    parser.add_argument('--npix-x', type=int, default=384)
    parser.add_argument('--npix-y', type=int, default=384)
    parser.add_argument('--pixsize-arcsec', type=float, default=22.0,
                        help='Pixel size (arcsec); applied to both axes (default: 22.0)')
    parser.add_argument('--epsilon', type=float, default=1e-6)
    parser.add_argument('--do-wgridding', action='store_true', default=True)
    parser.add_argument('--nthreads', type=int, default=0)
    parser.add_argument('--verbosity', type=int, default=0)
    args = parser.parse_args()

    # Convert arcsec to radians
    pix_rad = args.pixsize_arcsec / 206265.0

    # Discover total number of time chunks via iter
    t_main = table(args.msname, readonly=True)
    time_col = 'TIME_CENTROID' if 'TIME_CENTROID' in set(t_main.colnames()) else 'TIME'
    it = t_main.iter([time_col], sort=True)
    total_chunks = sum(1 for _ in it)
    it.close()
    t_main.close()

    print(f"Found {total_chunks} time chunks in MS: {args.msname}")

    start = 0
    chunk_size = max(1, args.chunk_size)
    chunk_id = 0

    while start < total_chunks:
        end = min(start + chunk_size - 1, total_chunks - 1)
        print(f"
[Chunk {chunk_id}] imaging time_idx {start}..{end}")
        times, cube = image_time_samples(msname=args.msname,
                                         start_time_idx=start,
                                         end_time_idx=end,
                                         corr_mode=args.corr_mode,
                                         basis=args.basis,
                                         single_pol=args.single_pol,
                                         npix_x=args.npix_x,
                                         npix_y=args.npix_y,
                                         pixsize_x=pix_rad,
                                         pixsize_y=pix_rad,
                                         epsilon=args.epsilon,
                                         do_wgridding=args.do_wgridding,
                                         nthreads=args.nthreads,
                                         verbosity=args.verbosity,
                                         )

        
        detections, snr_cubes = boxcar_search_time(times, cube,
                                                   widths=[1, 2, 4, 8, 16, 32, 64, 128],
                                                   widths_in_seconds=False,      # set True if widths are in seconds
                                                   threshold_sigma=6.0,
                                                   return_snr_cubes=True,
                                                   keep_top_k=50,                # keep top 50 per width
                                                   subtract_mean_per_pixel=True  # high-pass in time per pixel
                                                   )

        print(f"chunk start:{start} end:{end}: Found {len(detections)} candidates")

        
        start = end + 1
        chunk_id += 1
        
if __name__ == '__main__':
    main()
