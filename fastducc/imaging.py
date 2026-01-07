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
        
    colnames = set(t_main.colnames())

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
        
    colnames = set(t_main.colnames())
    time_col = 'TIME_CENTROID' if 'TIME_CENTROID' in colnames else 'TIME'
    #import ipdb; ipdb.set_trace()
    it = t_main.iter([time_col], sort=True)
    
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

    
    nsub, vis_time, all_times = ms_utils.get_time(t_main)
    #pre initialise img cube in mem and fill as we go
    # the number of time samples should be the min of (N - start_idx, end_idx - start_idx, N) where N is no. of time samples

    start_idx = 0 if start_time_idx is None else int(start_time_idx)
    end_idx   = (nsub - 1) if end_time_idx is None else int(end_time_idx)
    if start_idx < 0 or end_idx < start_idx or end_idx >= nsub:
        raise ValueError("Invalid start/end time indices")

    times = all_times[start_idx:end_idx+1]
    nt_window = end_idx - start_idx + 1
    
    cube = np.empty((nt_window, npix_y, npix_x))#, dtype=dtype_img)
    
    cube_idx = 0
    labels, lbl2idx = ms_utils.get_corr_label_indices(msname)
    
    for t_chunk_idx, t_chunk in enumerate(tqdm(it)):
        # Apply start/end time-chunk windowing
        if start_time_idx is not None and t_chunk_idx < start_time_idx:
            continue
        if end_time_idx is not None and t_chunk_idx > end_time_idx:
            break

        times_ = t_chunk.getcol(time_col)
        time_val = float(times_[0])

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

        
        # if t_chunk_idx == 10:
        #     print("adding fake transient")
        #     dirty[198:202,198:202] = 100.0
        #     import matplotlib.pyplot as plt
        #     plt.imshow(dirty, aspect='auto', interpolation='none', origin='lower', cmap='gray')
        #     plt.colorbar()
        #     plt.savefig(f'img_test_{cube_idx:04d}.png', dpi=300)
        # #     plt.close()
            
        cube[cube_idx, :, :] = dirty
        cube_idx += 1

        if do_plot:
            import matplotlib.pyplot as plt
            plt.imshow(dirty, aspect='auto', interpolation='none', origin='lower', cmap='gray')
            plt.colorbar()            
            plt.savefig(f"img_{cube_idx:04d}.png", dpi=300)
            sys.exit()
            #plt.show()
        #results.append((time_val, dirty))
        
    #it.close()
    #t_main.close()
    
    results = (times, cube)    
    return results
