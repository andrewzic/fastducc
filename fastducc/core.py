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


from fastducc import wcs as ducc_wcs
from fastducc.types import Config, WelfordState
from fastducc import filters, kernels, candidates, ms_utils, detection, imaging



def init_welford(cfg: Config) -> WelfordState:
    Ny, Nx = cfg.npix_y, cfg.npix_x
    return WelfordState(
        count=np.zeros((Ny, Nx), dtype=np.int64),
        mean=np.zeros((Ny, Nx), dtype=np.float64),
        M2=np.zeros((Ny, Nx), dtype=np.float64)
    )

def process_variance_cube_chunk(cfg: Config, times, cube, wf: WelfordState, start_idx: int):
    # Accumulate Welford
    kernels.welford_update_cube(wf.count, wf.mean, wf.M2, cube, ignore_nan=True)
    if not cfg.enable_var:
        return []  # no candidates in this chunk

    if not cfg.enable_var_chunk:
        return []

    # Partial std-map (optional, for visuals or chunk-level variance search)
    std_map_partial = kernels.welford_finalise_std(wf.count, wf.M2, ddof=1)

    # Run Welford-based variance search on partial map
    var_dets, snr_img = detection.variance_search_welford(
        std_map_partial,
        threshold_sigma=cfg.var_threshold,
        return_snr_image=True,
        keep_top_k=cfg.var_keep_k,
        valid_mask=None,
        spatial_estimator="clipped_rms",
        clip_sigma=cfg.rms_clip_sigma,
        subtract_mean_of_std_map=True
    )

    # Spatial NMS + annotation
    var_nms = filters.nms_snr_map_2d(
        snr_2d=snr_img, base_detections=var_dets,
        threshold_sigma=cfg.var_threshold,
        spatial_radius=cfg.nms_radius,
        valid_mask=None,
        times=times, cube=cube, time_tag_policy="peak_absdev"
    )
    annotated_var = ducc_wcs.annotate_candidates_with_sky_coords(
        msname=cfg.msname, final_detections=var_nms,
        npix_x=cfg.npix_x, npix_y=cfg.npix_y,
        pixsize_x=cfg.pix_rad, pixsize_y=cfg.pix_rad,
        flip_u=True, flip_v=True, field_name=None
    )

    # Save per-chunk VAR catalogue
    var_root = cfg.chunk_prefix_root(start_idx) + "_var"
    t_var = candidates.candidates_to_astropy_table(annotated_var)
    candidates.save_candidates_table(t_var,
        csv_path=f"{var_root}_candidates.csv",
        vot_path=f"{var_root}_candidates.vot"
    )


   # 7) Lightcurves + snippet products for each VAR candidate (using std map)
    for i, cand in enumerate(annotated_var):
        srcname = cand["srcname"]
        # Lightcurves figure (top panels use std-map images)
        if cfg.save_var_lightcurves:
            _ = candidates.save_candidate_lightcurves(
                times=times, cube=cube, candidate=cand,
                out_prefix=f"{var_root}_cand_{srcname}_{i:03d}_lc",
                spatial_size=50, save_format="npz",
                center_policy="right", cmap="gray", dpi=180,
                # WCS / scale
                npix_x=cfg.npix_x, npix_y=cfg.npix_y,
                ra0_rad=cfg.ra0_rad, dec0_rad=cfg.dec0_rad,
                pix_rad=cfg.pix_rad,
                ra_sign=-1, dec_sign=-1, radesys="ICRS", equinox=None,
                # Draw std-map images on the top panels:
                std_map=std_map_partial, use_std_images=True
            )

        # Snippet products from std-map (time length = 1)
        if cfg.save_var_snippets:
            std_snip = candidates.make_stdmap_snippet(std_map_partial, cand, spatial_size=50)
            _ = candidates.save_candidate_snippet_products(
                snippet_rec=std_snip,
                out_prefix=f"{var_root}_cand_{srcname}_{i:03d}_snip",
                pixscale_rad=cfg.pix_rad,
                ra_rad=float(cand["ra_rad"]), dec_rad=float(cand["dec_rad"]),
                ra_sign=-1, dec_sign=-1, cmap="gray", gif_fps=1, dpi=180
            )
    return annotated_var

def process_boxcar_chunk(cfg: Config, times, cube, start_idx: int):
    if not cfg.enable_boxcar:
        return []
    dets, snr_cubes = detection.boxcar_search_time(
        times, cube,
        widths=cfg.boxcar_widths,
        widths_in_seconds=False,
        threshold_sigma=cfg.boxcar_threshold,
        return_snr_cubes=True,
        keep_top_k=50,
        std_mode="spatial_per_window",
        subtract_mean_per_pixel=True
    )
    dets_by_w = filters.nms_snr_maps_per_width(
        snr_cubes, times,
        threshold_sigma=cfg.boxcar_threshold,
        spatial_radius=cfg.nms_radius,
        time_radius=2,
        valid_mask=None
    )
    final_dets = filters.group_filter_across_widths(
        dets_by_w, times,
        spatial_radius=cfg.nms_radius,
        time_radius=0,
        policy="max_snr",
        max_per_time_group=1,
        ny_nx=(cube.shape[1], cube.shape[2])
    )
    if len(final_dets) == 0:
        return []

    annotated = ducc_wcs.annotate_candidates_with_sky_coords(
        msname=cfg.msname, final_detections=final_dets,
        npix_x=cfg.npix_x, npix_y=cfg.npix_y,
        pixsize_x=cfg.pix_rad, pixsize_y=cfg.pix_rad,
        flip_u=True, flip_v=True, field_name=None
    )

    # Save per-chunk BOXCAR catalogue
    box_root = cfg.chunk_prefix_root(start_idx) + "_boxcar"
    t_box = candidates.candidates_to_astropy_table(annotated)
    candidates.save_candidates_table(t_box,
        csv_path=f"{box_root}_candidates.csv",
        vot_path=f"{box_root}_candidates.vot"
    )


    # Lightcurves + snippet products (boxcar)
    for i, cand in enumerate(annotated):
        srcname = cand["srcname"]
        w = max(1, int(cand.get("width_samples", 1)))
        # Lightcurves figure (top panels show the full-res frame)
        if cfg.save_box_lightcurves:
            _ = candidates.save_candidate_lightcurves(
                times=times, cube=cube, candidate=cand,
                out_prefix=f"{box_root}_cand_{srcname}_w{w}_{i:03d}_lc",
                spatial_size=50, save_format="npz",
                center_policy="right", cmap="gray", dpi=180,
                # WCS / scale:
                npix_x=cfg.npix_x, npix_y=cfg.npix_y,
                ra0_rad=cfg.ra0_rad, dec0_rad=cfg.dec0_rad,
                pix_rad=cfg.pix_rad,
                ra_sign=-1, dec_sign=-1, radesys="ICRS", equinox=None,
                # Use full-res images on top panels (std_map=None)
                std_map=None, use_std_images=False
            )

        # Extract smoothed snippet (your updated function that uses boxcar smoothing per width)
        if cfg.save_box_snippets:
            snippets = candidates.extract_candidate_snippets(
                times, cube, [cand],   # pass single cand to get one snippet
                spatial_size=50,
                time_factor=50,
                pad_mode="constant", pad_value=0.0,
                return_indices=True,
                center_policy="right"  # or "left"
            )
            # There will be exactly one snippet for this cand
            snip = snippets[0]
            _ = candidates.save_candidate_snippet_products(
                snippet_rec=snip,
                out_prefix=f"{box_root}_cand_{srcname}_w{w}_{i:03d}_snip",
                pixscale_rad=cfg.pix_rad,
                ra_rad=float(cand["ra_rad"]), dec_rad=float(cand["dec_rad"]),
                ra_sign=-1, dec_sign=-1, cmap="gray", gif_fps=6, dpi=180
            )
    return annotated


def welford_combine_aggregates(count_a, mean_a, M2_a, count_b, mean_b, M2_b):
    """
    Combine two Welford aggregates (array-wise) into one.
    All inputs are (Ny, Nx) arrays: count=int64, mean=float64, M2=float64.
    """
    total = count_a + count_b
    # Where no samples, keep zeros
    mask = (count_a > 0) & (count_b > 0)

    # Prepare outputs
    mean_out = mean_a.copy()
    M2_out = M2_a.copy()
    count_out = total

    if np.any(mask):
        # delta = mean_b - mean_a
        delta = np.zeros_like(mean_a)
        delta[mask] = mean_b[mask] - mean_a[mask]
        # mean = mean_a + delta * (count_b / total)
        mean_out[mask] = mean_a[mask] + delta[mask] * (count_b[mask] / total[mask])
        # M2 = M2_a + M2_b + delta^2 * count_a * count_b / total
        M2_out[mask] = (M2_a[mask] + M2_b[mask] +
                        (delta[mask] * delta[mask]) * (count_a[mask] * count_b[mask] / total[mask]))
    # Where only one side has samples, just take that side
    only_a = (count_a > 0) & (count_b == 0)
    mean_out[only_a] = mean_a[only_a]
    M2_out[only_a]   = M2_a[only_a]
    only_b = (count_b > 0) & (count_a == 0)
    mean_out[only_b] = mean_b[only_b]
    M2_out[only_b]   = M2_b[only_b]

    return count_out, mean_out, M2_out


def finalise_welford(cfg: Config, wf: WelfordState, times, cube):
    # Write full std-map (global)
    std_map_full = kernels.welford_finalise_std(wf.count, wf.M2, ddof=1)
    wcs_full = ducc_wcs._build_fullframe_wcs(
        npix_x=cfg.npix_x, npix_y=cfg.npix_y,
        ra0_rad=cfg.ra0_rad, dec0_rad=cfg.dec0_rad,
        pixscale_rad=cfg.pix_rad, ra_sign=-1, dec_sign=-1, radesys="ICRS", equinox=None
    )
    hdr = wcs_full.to_header()
    hdr["BUNIT"] = "std"; hdr["CDELT1"] = wcs_full.wcs.cdelt[0]; hdr["CDELT2"] = wcs_full.wcs.cdelt[1]
    hdr["PC1_1"] = 1.0; hdr["PC1_2"] = 0.0; hdr["PC2_1"] = 0.0; hdr["PC2_2"] = 1.0
    full_std_fits = os.path.join(cfg.candidates_dir, f"{cfg.ms_base}_std_map_full.fits")
    fits.writeto(full_std_fits, data=std_map_full.astype(np.float32), header=hdr, overwrite=True)

    # Optional: run final variance detection on full std-map and save catalogue
    if cfg.enable_var:
        var_final, snr_img = detection.variance_search_welford(
            std_map_full,
            threshold_sigma=cfg.var_threshold,
            return_snr_image=True,
            keep_top_k=cfg.var_keep_k,
            valid_mask=None,
            spatial_estimator="clipped_rms",
            clip_sigma=cfg.rms_clip_sigma,
            subtract_mean_of_std_map=True
        )
        if len(var_final) > 0:
            var_nms = filters.nms_snr_map_2d(
                snr_2d=snr_img, base_detections=var_final,
                threshold_sigma=cfg.var_threshold,
                spatial_radius=cfg.nms_radius,
                valid_mask=None,
                times=times, cube=cube, time_tag_policy="peak_absdev"
            )
            annotated_var = ducc_wcs.annotate_candidates_with_sky_coords(
                msname=cfg.msname, final_detections=var_nms,
                npix_x=cfg.npix_x, npix_y=cfg.npix_y,
                pixsize_x=cfg.pix_rad, pixsize_y=cfg.pix_rad,
                flip_u=True, flip_v=True, field_name=None
            )
            var_root = cfg.all_prefix_root + "_var"
            t_var = candidates.candidates_to_astropy_table(annotated_var)
            candidates.save_candidates_table(t_var,
                csv_path=f"{var_root}_candidates.csv",
                vot_path=f"{var_root}_candidates.vot"
            )


def finalise_welford_parallel(
        cfg: Config,
        agg_list: List,               # List[Tuple[np.ndarray, np.ndarray, np.ndarray]] as (count, mean, M2) per chunk
        *,
        run_final_variance: bool = True
):
    """
    Reduce per-chunk Welford aggregates into a full-observation std-map,
    write FITS with WCS, optionally run final variance search, and
    write the consolidated variance catalogue.

    Parameters
    ----------
    cfg : Config
        Pipeline configuration (paths, WCS info, toggles, thresholds).
    agg_list : list of (times, cube, count, mean, M2)
        Per-chunk aggregates returned by the parallel chunk tasks.
        Each array is shape (Ny, Nx): count=int64, mean=float64, M2=float64.
    run_final_variance : bool
        If True and cfg.enable_var, run variance_search_welford on the full std-map.

    Returns
    -------
    std_map_full : np.ndarray
        Final global standard deviation map, shape (Ny, Nx), float64.
    """
    # --- 1) Reduce all per-chunk aggregates into one global aggregate ---
    Ny, Nx = cfg.npix_y, cfg.npix_x
    count_acc = np.zeros((Ny, Nx), dtype=np.int64)
    mean_acc  = np.zeros((Ny, Nx), dtype=np.float64)
    M2_acc    = np.zeros((Ny, Nx), dtype=np.float64)

    
    for (times, cube, c, m, M2) in agg_list:
        # combine per-chunk aggregate into the accumulator
        count_acc, mean_acc, M2_acc = welford_combine_aggregates(count_acc, mean_acc, M2_acc, c, m, M2)
        
    all_times = np.concatenate([a[0] for a in agg_list])
    order = np.argsort(all_times)
    all_times = all_times[order]
    if cfg.save_full_var_lightcurves:
        all_cube = np.empty((int(np.sum(a[1].shape[0] for a in agg_list)), cfg.npix_y, cfg.npix_x), dtype=agg_list[0][1].dtype)
        start_ = 0
        for a in agg_list:
            nsub = a[1].shape[0]
            end_ = start_ + nsub
            all_cube[start_:end_] = a[1]
            start_ += nsub

        all_cube = all_cube[order]
    else:
        all_cube = None

    # --- 2) Finalise to full std-map ---
    std_map_full = kernels.welford_finalise_std(count_acc, M2_acc, ddof=1)

    # --- 3) Write full std-map FITS with full-frame WCS ---
    wcs_full = ducc_wcs._build_fullframe_wcs(
        npix_x=cfg.npix_x, npix_y=cfg.npix_y,
        ra0_rad=cfg.ra0_rad, dec0_rad=cfg.dec0_rad,
        pixscale_rad=cfg.pix_rad,
        ra_sign=-1, dec_sign=-1, radesys="ICRS", equinox=None
    )
    hdr = wcs_full.to_header()
    hdr["BUNIT"]  = "std"
    hdr["CDELT1"] = wcs_full.wcs.cdelt[0]
    hdr["CDELT2"] = wcs_full.wcs.cdelt[1]
    hdr["PC1_1"]  = 1.0; hdr["PC1_2"] = 0.0
    hdr["PC2_1"]  = 0.0; hdr["PC2_2"] = 1.0

    full_std_fits = os.path.join(cfg.candidates_dir, f"{cfg.ms_base}_std_map_full.fits")
    fits.writeto(full_std_fits, data=std_map_full.astype(np.float32), header=hdr, overwrite=True)
    print(f"[Final] wrote full std-map -> {full_std_fits}")

    # --- 4) Optional final variance search on the full std-map ---
    if run_final_variance and cfg.enable_var:
        var_final, snr_img = detection.variance_search_welford(
            std_map_full,
            threshold_sigma=cfg.var_threshold,
            return_snr_image=True,
            keep_top_k=cfg.var_keep_k,
            valid_mask=None,
            spatial_estimator="clipped_rms",
            clip_sigma=cfg.rms_clip_sigma,
            subtract_mean_of_std_map=True
        )
        if len(var_final) > 0:

            var_nms = filters.nms_snr_map_2d(
                snr_2d=snr_img, base_detections=var_final,
                threshold_sigma=cfg.var_threshold,
                spatial_radius=cfg.nms_radius,
                valid_mask=None,
                times=all_times, time_tag_policy="none"
            )
            annotated_var = ducc_wcs.annotate_candidates_with_sky_coords(
                msname=cfg.msname, final_detections=var_nms,
                npix_x=cfg.npix_x, npix_y=cfg.npix_y,
                pixsize_x=cfg.pix_rad, pixsize_y=cfg.pix_rad,
                flip_u=True, flip_v=True, field_name=None
            )
            var_root = cfg.all_prefix_root + "_var"
            t_var = candidates.candidates_to_astropy_table(annotated_var)
            candidates.save_candidates_table(t_var,
                                             csv_path=f"{var_root}_candidates.csv",
                                             vot_path=f"{var_root}_candidates.vot"
                                             )
            for i, cand in enumerate(annotated_var):
                srcname = cand["srcname"]
                if cfg.save_full_var_lightcurves:
                    candidates.save_candidate_lightcurves(
                        all_times, all_cube, cand,
                        out_prefix=f"{var_root}_cand_{srcname}_{i:03d}_lc",
                        spatial_size=50, save_format="npz",
                        center_policy="right", cmap="gray", dpi=180,
                        npix_x=cfg.npix_x, npix_y=cfg.npix_y,
                        ra0_rad=cfg.ra0_rad, dec0_rad=cfg.dec0_rad,
                        pix_rad=cfg.pix_rad, ra_sign=-1, dec_sign=-1, radesys="ICRS", equinox=None,
                        std_map=std_map_full, use_std_images=True
                    )
                if cfg.save_var_snippets:
                    std_snip = candidates.make_stdmap_snippet(std_map_full, cand, spatial_size=50)
                    candidates.save_candidate_snippet_products(
                        snippet_rec=std_snip,
                        out_prefix=f"{var_root}_cand_{srcname}_{i:03d}_snip",
                        pixscale_rad=cfg.pix_rad,
                        ra_rad=float(cand["ra_rad"]), dec_rad=float(cand["dec_rad"]),
                        ra_sign=-1, dec_sign=-1, cmap="gray", gif_fps=1, dpi=180
                    )

    return std_map_full

def finalise_welford_serial(cfg: Config, wf_state: WelfordState, *, run_final_variance=True):
    """
    Finalise from a live WelfordState (count, mean, M2) as in serial runs,
    using the same parallel finalise routine.
    """
    agg_list = [(wf_state.count, wf_state.mean, wf_state.M2)]
    return finalise_welford_parallel(cfg, agg_list, run_final_variance=run_final_variance)


def consolidate_catalogues(cfg: Config):
    var_pattern = os.path.join(cfg.candidates_dir, f"{cfg.ms_base}*_var_candidates.csv")
    box_pattern = os.path.join(cfg.candidates_dir, f"{cfg.ms_base}_chunk_*_boxcar_candidates.csv")
    candidates.consolidate_chunk_catalogues(
        ms_base=cfg.ms_base,
        out_dir=cfg.candidates_dir,
        var_csv_pattern=var_pattern,
        box_csv_pattern=box_pattern,
        remove_chunk_catalogues=True
    )



def process_chunk_task(cfg: Config, ms_base: str, candidates_dir: str, start: int, end: int):
    """
    Worker task: image the chunk, produce per-chunk outputs, and return per-chunk Welford aggregates.
    """
    # Imaging: open MS inside worker (t_main=None)
    times, cube = imaging.image_time_samples(
        msname=cfg.msname, t_main=None,
        start_time_idx=start, end_time_idx=end,
        corr_mode=cfg.corr_mode, basis=cfg.basis, single_pol=cfg.single_pol,
        data_column=cfg.data_column,
        npix_x=cfg.npix_x, npix_y=cfg.npix_y,
        pixsize_x=cfg.pix_rad, pixsize_y=cfg.pix_rad,
        epsilon=cfg.epsilon, do_wgridding=cfg.do_wgridding,
        nthreads=cfg.nthreads, verbosity=cfg.verbosity, do_plot=cfg.do_plot
    )

    # Per-chunk Welford aggregates
    Ny, Nx = cfg.npix_y, cfg.npix_x
    c = np.zeros((Ny, Nx), dtype=np.int64) #count for welford
    m = np.zeros((Ny, Nx), dtype=np.float64) #
    M2 = np.zeros((Ny, Nx), dtype=np.float64)
    kernels.welford_update_cube(c, m, M2, cube, ignore_nan=True)

    chunk_root = os.path.join(candidates_dir, f"{ms_base}_chunk_{start:06d}")

    # Optional per-chunk variance search
    if cfg.enable_var and cfg.enable_var_chunk:
        std_map_partial = kernels.welford_finalise_std(c, M2, ddof=1)
        var_dets, snr_img = detection.variance_search_welford(
            std_map_partial,
            threshold_sigma=cfg.var_threshold,
            return_snr_image=True,
            keep_top_k=cfg.var_keep_k,
            valid_mask=None,
            spatial_estimator="clipped_rms",
            clip_sigma=cfg.rms_clip_sigma,
            subtract_mean_of_std_map=True
        )
        if len(var_dets) > 0:
            var_nms = filters.nms_snr_map_2d(
                snr_2d=snr_img, base_detections=var_dets,
                threshold_sigma=cfg.var_threshold,
                spatial_radius=cfg.nms_radius,
                valid_mask=None,
                times=times, cube=cube, time_tag_policy="peak_absdev"
            )
            annotated_var = ducc_wcs.annotate_candidates_with_sky_coords(
                msname=cfg.msname, final_detections=var_nms,
                npix_x=cfg.npix_x, npix_y=cfg.npix_y,
                pixsize_x=cfg.pix_rad, pixsize_y=cfg.pix_rad,
                flip_u=True, flip_v=True, field_name=None
            )
            var_root = chunk_root + "_var"
            t_var = candidates.candidates_to_astropy_table(annotated_var)
            candidates.save_candidates_table(t_var,
                                             csv_path=f"{var_root}_candidates.csv",
                                             vot_path=f"{var_root}_candidates.vot"
                                             )
            for i, cand in enumerate(annotated_var):
                srcname = cand["srcname"]
                if cfg.save_var_lightcurves:
                    candidates.save_candidate_lightcurves(
                        times, cube, cand,
                        out_prefix=f"{var_root}_cand_{srcname}_{i:03d}_lc",
                        spatial_size=50, save_format="npz",
                        center_policy="right", cmap="gray", dpi=180,
                        npix_x=cfg.npix_x, npix_y=cfg.npix_y,
                        ra0_rad=cfg.ra0_rad, dec0_rad=cfg.dec0_rad,
                        pix_rad=cfg.pix_rad, ra_sign=-1, dec_sign=-1, radesys="ICRS", equinox=None,
                        std_map=std_map_partial, use_std_images=True
                    )
                if cfg.save_var_snippets:
                    std_snip = candidates.make_stdmap_snippet(std_map_partial, cand, spatial_size=50)
                    candidates.save_candidate_snippet_products(
                        snippet_rec=std_snip,
                        out_prefix=f"{var_root}_cand_{srcname}_{i:03d}_snip",
                        pixscale_rad=cfg.pix_rad,
                        ra_rad=float(cand["ra_rad"]), dec_rad=float(cand["dec_rad"]),
                        ra_sign=-1, dec_sign=-1, cmap="gray", gif_fps=1, dpi=180
                    )

    # Optional boxcar search
    if cfg.enable_boxcar:
        dets, snr_cubes = detection.boxcar_search_time(
            times, cube,
            widths=cfg.boxcar_widths,
            widths_in_seconds=False,
            threshold_sigma=cfg.boxcar_threshold,
            return_snr_cubes=True,
            keep_top_k=50,
            std_mode="spatial_per_window",
            subtract_mean_per_pixel=True
        )
        dets_by_w = filters.nms_snr_maps_per_width(
            snr_cubes, times,
            threshold_sigma=cfg.boxcar_threshold,
            spatial_radius=cfg.nms_radius, time_radius=2, valid_mask=None
        )
        final_dets = filters.group_filter_across_widths(
            dets_by_w, times,
            spatial_radius=cfg.nms_radius, time_radius=0,
            policy="max_snr", max_per_time_group=1,
            ny_nx=(cube.shape[1], cube.shape[2])
        )
        if len(final_dets) > 0:
            annotated = ducc_wcs.annotate_candidates_with_sky_coords(
                msname=cfg.msname, final_detections=final_dets,
                npix_x=cfg.npix_x, npix_y=cfg.npix_y,
                pixsize_x=cfg.pix_rad, pixsize_y=cfg.pix_rad,
                flip_u=True, flip_v=True, field_name=None
            )
            box_root = chunk_root + "_boxcar"
            t_box = candidates.candidates_to_astropy_table(annotated)
            candidates.save_candidates_table(t_box,
                csv_path=f"{box_root}_candidates.csv",
                vot_path=f"{box_root}_candidates.vot"
            )
            for i, cand in enumerate(annotated):
                w = max(1, int(cand.get("width_samples", 1)))
                if cfg.save_box_lightcurves:
                    srcname = cand["srcname"]
                    candidates.save_candidate_lightcurves(
                        times, cube, cand,
                        out_prefix=f"{box_root}_cand_{srcname}_w{w}_{i:03d}_lc",
                        spatial_size=50, save_format="npz",
                        center_policy="right", cmap="gray", dpi=180,
                        npix_x=cfg.npix_x, npix_y=cfg.npix_y,
                        ra0_rad=cfg.ra0_rad, dec0_rad=cfg.dec0_rad,
                        pix_rad=cfg.pix_rad, ra_sign=-1, dec_sign=-1, radesys="ICRS", equinox=None,
                        std_map=None, use_std_images=False
                    )
                if cfg.save_box_snippets:
                    snippets = candidates.extract_candidate_snippets(
                        times, cube, [cand],
                        spatial_size=50, time_factor=50,
                        pad_mode="constant", pad_value=0.0,
                        return_indices=True, center_policy="right"
                    )
                    snip = snippets[0]
                    candidates.save_candidate_snippet_products(
                        snippet_rec=snip,
                        out_prefix=f"{box_root}_cand_{srcname}_w{w}_{i:03d}_snip",
                        pixscale_rad=cfg.pix_rad,
                        ra_rad=float(cand["ra_rad"]), dec_rad=float(cand["dec_rad"]),
                        ra_sign=-1, dec_sign=-1, cmap="gray", gif_fps=6, dpi=180
                    )

    if cfg.save_full_var_lightcurves:
        return times, cube, c, m, M2
    else:
        return times, None, c, m, M2
