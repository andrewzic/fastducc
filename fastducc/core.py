from __future__ import annotations
import logging
import os
from typing import List

import numpy as np
from astropy.io import fits

from fastducc import wcs as ducc_wcs
from fastducc.fd_types import Config
from fastducc import filters, kernels, candidates, detection, imaging, catalogues
from astropy.coordinates import SkyCoord
import astropy.units as u

logger = logging.getLogger(__name__)


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


def finalise_welford_parallel(
        cfg: Config,
        agg_list: List,
        dm: float | None = None,
):
    """
    Reduce per-chunk Welford aggregates into a full-observation std-map,
    write FITS with WCS, optionally run final variance search, and
    write the consolidated variance catalogue.

    Parameters
    ----------
    cfg : Config
        Pipeline configuration (paths, WCS info, toggles, thresholds).
    agg_list : list
        Per-chunk results returned by chunk tasks. Each element can be:
        (times, cube, welford_by_dm_dict, sid) or legacy (times, cube, count, mean, M2, [sid]).
    dm : float, optional
        Trial DM to finalise. Defaults to cfg.current_dm or 0.0.

    Returns
    -------
    std_map_full : np.ndarray or None
        Final global standard deviation map, shape (Ny, Nx), float64.
    """
    dm_val = float(dm) if dm is not None else float(cfg.current_dm)
    is_search = cfg.is_dm_search or (dm_val != 0.0)
    dm_suffix = f"_dm{dm_val:06.2f}" if is_search else ""

    # --- 1) Reduce per-chunk aggregates for this dm into one global aggregate ---
    Ny, Nx = cfg.npix_y, cfg.npix_x
    count_acc = np.zeros((Ny, Nx), dtype=np.int64)
    mean_acc  = np.zeros((Ny, Nx), dtype=np.float64)
    M2_acc    = np.zeros((Ny, Nx), dtype=np.float64)

    valid_items = []
    for item in agg_list:
        times = item[0]
        cube = item[1]
        if len(item) == 4:
            # (times, cube, welford_data, sid)
            w_data = item[2]
            sid = item[3]
            if isinstance(w_data, dict):
                if dm_val in w_data:
                    c, m, M2 = w_data[dm_val]
                elif len(w_data) == 1 and dm is None:
                    c, m, M2 = next(iter(w_data.values()))
                else:
                    continue
            elif isinstance(w_data, (tuple, list)):
                c, m, M2 = w_data[0], w_data[1], w_data[2]
            else:
                continue
        elif len(item) >= 5:
            # Legacy: (times, cube, c, m, M2, [sid])
            c, m, M2 = item[2], item[3], item[4]
            sid = item[5] if len(item) > 5 else ""
        else:
            continue

        count_acc, mean_acc, M2_acc = welford_combine_aggregates(count_acc, mean_acc, M2_acc, c, m, M2)
        valid_items.append((times, cube, c, m, M2, sid))

    if not valid_items or np.sum(count_acc) == 0:
        return None

    all_times = np.concatenate([item[0] for item in valid_items if item[0] is not None and len(item[0]) > 0]) if any(item[0] is not None and len(item[0]) > 0 for item in valid_items) else np.array([])
    order = np.argsort(all_times) if len(all_times) > 0 else np.array([])
    if len(order) > 0:
        all_times = all_times[order]

    if cfg.save_full_var_lightcurves and any(item[1] is not None for item in valid_items):
        cubes_to_stack = [item[1] for item in valid_items if item[1] is not None]
        total_slices = sum(c.shape[0] for c in cubes_to_stack)
        all_cube = np.empty((total_slices, cfg.npix_y, cfg.npix_x), dtype=cubes_to_stack[0].dtype)
        start_ = 0
        for i in range(len(valid_items)):
            c = valid_items[i][1]
            if c is not None:
                nsub = c.shape[0]
                all_cube[start_:start_ + nsub] = c
                start_ += nsub
                # Release chunk cube reference early to reclaim memory
                valid_items[i] = (valid_items[i][0], None, *valid_items[i][2:])
        del cubes_to_stack

        if len(order) == all_cube.shape[0] and not np.array_equal(order, np.arange(len(order))):
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
    hdr["PC1_1"]  = 1.0 
    hdr["PC1_2"] = 0.0
    hdr["PC2_1"]  = 0.0 
    hdr["PC2_2"] = 1.0

    scan_tag = f"_scan_{cfg.scan_id}" if cfg.scan_id else ""
    full_std_fits = os.path.join(cfg.candidates_dir, f"{cfg.ms_base}{scan_tag}{dm_suffix}_std_map_full.fits")
    hdr["DM"] = dm_val
    fits.writeto(full_std_fits, data=std_map_full.astype(np.float32), header=hdr, overwrite=True)
    logger.info(f"[Final] Wrote full std-map (DM={dm_val:.2f}) -> {full_std_fits}")

    # --- 3b) Optional per-scan variance search on aggregated Welford maps within each scan ---
    if cfg.enable_var and cfg.enable_var_scan:
        scan_groups = {}
        for item in valid_items:
            sid = item[5] if len(item) > 5 else ""
            if sid:
                if sid not in scan_groups:
                    scan_groups[sid] = []
                scan_groups[sid].append(item)

        for sid, s_items in scan_groups.items():
            c_scan = np.zeros((Ny, Nx), dtype=np.int64)
            m_scan = np.zeros((Ny, Nx), dtype=np.float64)
            M2_scan = np.zeros((Ny, Nx), dtype=np.float64)
            s_times = []
            for item in s_items:
                times, cube, c, m, M2 = item[0], item[1], item[2], item[3], item[4]
                c_scan, m_scan, M2_scan = welford_combine_aggregates(c_scan, m_scan, M2_scan, c, m, M2)
                if times is not None:
                    s_times.append(times)
            scan_times = np.concatenate(s_times) if s_times else np.array([])
            
            std_map_scan = kernels.welford_finalise_std(c_scan, M2_scan, ddof=1)
            var_scan_dets, snr_img = detection.variance_search_welford(
                std_map_scan,
                threshold_sigma=cfg.var_threshold,
                return_snr_image=True,
                keep_top_k=cfg.var_keep_k,
                valid_mask=None,
                spatial_estimator="clipped_rms",
                clip_sigma=cfg.rms_clip_sigma,
                subtract_mean_of_std_map=True,
                use_local_threshold=cfg.use_local_threshold,
                local_window_size=cfg.local_window_size,
                dm=dm_val,
            )
            if len(var_scan_dets) > 0:
                var_nms = filters.nms_snr_map_2d(
                    snr_2d=snr_img, base_detections=var_scan_dets,
                    threshold_sigma=cfg.var_threshold,
                    spatial_radius=cfg.nms_radius,
                    valid_mask=None,
                    times=scan_times, time_tag_policy="none"
                )
                annotated_var = ducc_wcs.annotate_candidates_with_sky_coords(
                    msname=cfg.msname, final_detections=var_nms,
                    npix_x=cfg.npix_x, npix_y=cfg.npix_y,
                    pixsize_x=cfg.pix_rad, pixsize_y=cfg.pix_rad,
                    flip_u=True, flip_v=True, field_name=None
                )
                for cand in annotated_var:
                    cand["scan_id"] = sid
                    cand["dm"] = dm_val
                
                var_root = os.path.join(cfg.candidates_dir, f"{cfg.ms_base}_scan_{sid}{dm_suffix}_var")
                t_var = candidates.candidates_to_astropy_table(annotated_var)
                candidates.save_candidates_table(
                    t_var,
                    csv_path=f"{var_root}_candidates.csv",
                    vot_path=f"{var_root}_candidates.vot"
                )
                logger.info(f"[Final] Wrote per-scan variance candidates for scan {sid} (DM={dm_val:.2f}) -> {var_root}_candidates.csv")

    # --- 4) Optional final variance search on the full std-map ---
    if cfg.enable_var_obs and cfg.enable_var:
        var_final, snr_img = detection.variance_search_welford(
            std_map_full,
            threshold_sigma=cfg.var_threshold,
            return_snr_image=True,
            keep_top_k=cfg.var_keep_k,
            valid_mask=None,
            spatial_estimator="clipped_rms",
            clip_sigma=cfg.rms_clip_sigma,
            subtract_mean_of_std_map=True,
            use_local_threshold=cfg.use_local_threshold,
            local_window_size=cfg.local_window_size,
            dm=dm_val,
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
            for cand in annotated_var:
                cand["dm"] = dm_val

            var_root = f"{cfg.all_prefix_root}{scan_tag}{dm_suffix}_var"
            t_var = candidates.candidates_to_astropy_table(annotated_var)
            candidates.save_candidates_table(t_var,
                                             csv_path=f"{var_root}_candidates.csv",
                                             vot_path=f"{var_root}_candidates.vot"
                                             )
            for i, cand in enumerate(annotated_var):
                srcname = cand["srcname"]
                if cfg.save_full_var_lightcurves and all_cube is not None:
                    rms_full = candidates.compute_candidate_local_rms(
                        all_cube, cand, window_size=cfg.local_window_size,
                        clip_sigma=cfg.rms_clip_sigma,
                    )
                    _ = candidates.save_candidate_lightcurves(
                        times=all_times, cube=all_cube, candidate=cand,
                        out_prefix=f"{var_root}_cand_{srcname}_lc",
                        save_format="npz",
                        rms_full=rms_full,
                    )
                    candidates.save_candidate_summary(
                        all_times, all_cube, cand,
                        out_prefix=f"{var_root}_cand_{srcname}",
                        spatial_size=50,
                        center_policy="right", cmap="viridis", dpi=180,
                        npix_x=cfg.npix_x, npix_y=cfg.npix_y,
                        ra0_rad=cfg.ra0_rad, dec0_rad=cfg.dec0_rad,
                        pix_rad=cfg.pix_rad, ra_sign=-1, dec_sign=-1, radesys="ICRS", equinox=None,
                        std_map=std_map_full, use_std_images=True,
                        continuum_dir=cfg.continuum_dir,
                        method="variance",
                        var_highpass_cutoff_sec=cfg.var_highpass_cutoff_sec,
                        rms_full=rms_full,
                    )
                if cfg.save_var_snippets:
                    std_snip = candidates.make_stdmap_snippet(std_map_full, cand, spatial_size=50)
                    candidates.save_candidate_snippet_products(
                        snippet_rec=std_snip,
                        out_prefix=f"{var_root}_cand_{srcname}_{i:03d}_snip",
                        pixscale_rad=cfg.pix_rad,
                        ra_rad=float(cand["ra_rad"]), dec_rad=float(cand["dec_rad"]),
                        ra_sign=-1, dec_sign=-1, cmap="viridis", gif_fps=1, dpi=180
                    )

            if all_cube is not None:
                del all_cube
                all_cube = None

    # Ensure all obs-level variance candidates have cutouts, fullframes, plots, and lightcurves
    try:
        candidates.generate_obs_variance_candidate_products(
            ms_base=cfg.ms_base,
            candidates_dir=cfg.candidates_dir,
            continuum_dir=cfg.continuum_dir,
            spatial_size=50
        )
    except Exception as e:
        logger.warning(f"[FinaliseWelford] Note: Obs variance products generation: {e}")

    return std_map_full


def consolidate_catalogues(cfg: Config):
    dm_val = float(cfg.current_dm)
    scan_tag = f"_scan_{cfg.scan_id}" if cfg.scan_id else ""
    is_search = cfg.is_dm_search or (dm_val != 0.0)
    if is_search:
        dm_tag = f"_dm{dm_val:06.2f}"
        ms_base_tag = f"{cfg.ms_base}{scan_tag}{dm_tag}"
        var_pattern = os.path.join(cfg.candidates_dir, f"{cfg.ms_base}*{scan_tag}*{dm_tag}*_var_candidates.csv")
        box_pattern = os.path.join(cfg.candidates_dir, f"{cfg.ms_base}*{scan_tag}*{dm_tag}*_chunk_*_boxcar_candidates.csv")
    else:
        ms_base_tag = f"{cfg.ms_base}{scan_tag}"
        var_pattern = os.path.join(cfg.candidates_dir, f"{cfg.ms_base}*{scan_tag}*_var_candidates.csv")
        box_pattern = os.path.join(cfg.candidates_dir, f"{cfg.ms_base}*{scan_tag}*_chunk_*_boxcar_candidates.csv")
    candidates.consolidate_chunk_catalogues(
        ms_base=ms_base_tag,
        out_dir=cfg.candidates_dir,
        var_csv_pattern=var_pattern,
        box_csv_pattern=box_pattern,
        remove_chunk_catalogues=True
    )



def _load_plot_only_candidates(
    candidates_dir: str,
    root_path: str,
    ms_base: str,
    scan_suffix: str,
    dm_suffix: str,
    search_type: str,
) -> list[dict]:
    """Load candidates for plot-only mode from VOTables."""
    vot_path = f"{root_path}_candidates.vot"
    if os.path.exists(vot_path):
        return candidates.astropy_table_to_candidates(vot_path)

    vot_path = os.path.join(candidates_dir, f"{ms_base}{scan_suffix}{dm_suffix}_{search_type}_all.vot")
    if not os.path.exists(vot_path):
        vot_path = os.path.join(candidates_dir, f"{ms_base}_{search_type}_all.vot")

    if os.path.exists(vot_path):
        return candidates.astropy_table_to_candidates(vot_path)

    logger.warning(f"No candidates found for plotting: no vot files at {root_path}_candidates.vot or {vot_path}")
    return []


def _export_variance_chunk_products(
    cfg: Config,
    times: np.ndarray,
    cube: np.ndarray,
    std_map_partial: np.ndarray,
    annotated_var: list[dict],
    var_root: str,
):
    """Generate lightcurves, summary plots, and snippets for chunk variance candidates."""
    for i, cand in enumerate(annotated_var):
        srcname = cand["srcname"]
        if cfg.save_var_lightcurves:
            if times[0] <= cand["time_center"] <= times[-1]:
                rms_full = candidates.compute_candidate_local_rms(
                    cube, cand, window_size=cfg.local_window_size,
                    clip_sigma=cfg.rms_clip_sigma,
                )
                candidates.save_candidate_lightcurves(
                    times, cube, cand,
                    out_prefix=f"{var_root}_cand_{srcname}_lc",
                    save_format="npz",
                    rms_full=rms_full,
                )
                _ = candidates.save_candidate_summary(
                    times=times, cube=cube, candidate=cand,
                    out_prefix=f"{var_root}_cand_{srcname}",
                    spatial_size=50,
                    center_policy="right", cmap="viridis", dpi=300,
                    npix_x=cfg.npix_x, npix_y=cfg.npix_y,
                    ra0_rad=cfg.ra0_rad, dec0_rad=cfg.dec0_rad,
                    pix_rad=cfg.pix_rad,
                    ra_sign=-1, dec_sign=-1, radesys="ICRS", equinox=None,
                    std_map=std_map_partial, use_std_images=True,
                    continuum_dir=cfg.continuum_dir,
                    method="variance",
                    var_highpass_cutoff_sec=cfg.var_highpass_cutoff_sec,
                    rms_full=rms_full,
                )
            else:
                logger.warning(f"Candidate {srcname} has time_center={cand['time_center']} outside of chunk times [{times[0]}, {times[-1]}], skipping lightcurve and snippet products.")
        if cfg.save_var_snippets:
            std_snip = candidates.make_stdmap_snippet(std_map_partial, cand, spatial_size=50)
            candidates.save_candidate_snippet_products(
                snippet_rec=std_snip,
                out_prefix=f"{var_root}_cand_{srcname}_{i:03d}_snip",
                pixscale_rad=cfg.pix_rad,
                ra_rad=float(cand["ra_rad"]), dec_rad=float(cand["dec_rad"]),
                ra_sign=-1, dec_sign=-1, cmap="viridis", gif_fps=1, dpi=180
            )


def _export_boxcar_chunk_products(
    cfg: Config,
    times: np.ndarray,
    cube: np.ndarray,
    annotated: list[dict],
    box_root: str,
):
    """Generate lightcurves, summary plots, and snippets for chunk boxcar candidates."""
    for i, cand in enumerate(annotated):
        srcname = cand["srcname"]
        w = max(1, int(cand.get("width_samples", 1)))
        if times[0] <= cand["time_center"] <= times[-1]:
            if cfg.save_box_lightcurves:
                rms_full = candidates.compute_candidate_local_rms(
                    cube, cand, window_size=cfg.local_window_size,
                    clip_sigma=cfg.rms_clip_sigma,
                )
                candidates.save_candidate_lightcurves(
                    times, cube, cand,
                    out_prefix=f"{box_root}_cand_{srcname}_w{w}_lc",
                    save_format="npz",
                    rms_full=rms_full,
                )
                _ = candidates.save_candidate_summary(
                    times=times, cube=cube, candidate=cand,
                    out_prefix=f"{box_root}_cand_{srcname}_w{w}",
                    spatial_size=50,
                    center_policy="right", cmap="viridis", dpi=180,
                    npix_x=cfg.npix_x, npix_y=cfg.npix_y,
                    ra0_rad=cfg.ra0_rad, dec0_rad=cfg.dec0_rad,
                    pix_rad=cfg.pix_rad,
                    ra_sign=-1, dec_sign=-1, radesys="ICRS", equinox=None,
                    std_map=None, use_std_images=False,
                    continuum_dir=cfg.continuum_dir,
                    method="boxcar",
                    rms_full=rms_full,
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
                    ra_sign=-1, dec_sign=-1, cmap="viridis", gif_fps=6, dpi=180
                )
        else:
            logger.warning(f"Candidate {srcname} has time_center={cand['time_center']} outside of chunk times [{times[0]}, {times[-1]}], skipping lightcurve and snippet products.")


def _process_variance_chunk_dm(
    cfg: Config,
    times: np.ndarray,
    cube: np.ndarray,
    dm_val: float,
    chunk_root: str,
    scan_id_str: str,
    candidates_dir: str,
    ms_base: str,
    scan_suffix: str,
    dm_suffix: str,
    racs_coords,
    psrcat_coords,
    dm0_var_cands: list[dict],
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], list[dict]]:
    """Process variance detection and Welford accumulation for a single trial DM on a chunk."""
    Ny, Nx = cfg.npix_y, cfg.npix_x
    do_highpass = cfg.var_highpass_cutoff_sec > 0
    alphas = np.zeros(len(times), dtype=np.float64)
    if do_highpass:
        ema_mean = np.nanmean(cube, axis=0, dtype=np.float64)
        last_t = np.nan
        dt_median = np.median(np.diff(times)) if len(times) > 1 else 0.0
        for i, t in enumerate(times):
            if np.isnan(last_t):
                alphas[i] = 1.0 - np.exp(-dt_median / cfg.var_highpass_cutoff_sec)
            else:
                dt_step = max(0.0, t - last_t)
                alphas[i] = 1.0 - np.exp(-dt_step / cfg.var_highpass_cutoff_sec)
            last_t = t
    else:
        ema_mean = np.full((Ny, Nx), np.nan, dtype=np.float64)

    c_dm = np.zeros((Ny, Nx), dtype=np.int64)
    m_dm = np.zeros((Ny, Nx), dtype=np.float64)
    M2_dm = np.zeros((Ny, Nx), dtype=np.float64)
    kernels.welford_update_cube(c_dm, m_dm, M2_dm, ema_mean, cube, alphas, do_highpass=do_highpass, ignore_nan=True)

    annotated_var: list[dict] = []
    if cfg.enable_var_chunk:
        var_root = chunk_root + "_var"
        std_map_partial = kernels.welford_finalise_std(c_dm, M2_dm, ddof=1)
        if cfg.do_var_search:
            var_dets, snr_img = detection.variance_search_welford(
                std_map_partial,
                threshold_sigma=cfg.var_threshold,
                return_snr_image=True,
                keep_top_k=cfg.var_keep_k,
                valid_mask=None,
                spatial_estimator="clipped_rms",
                clip_sigma=cfg.rms_clip_sigma,
                subtract_mean_of_std_map=True,
                use_local_threshold=cfg.use_local_threshold,
                local_window_size=cfg.local_window_size,
                dm=dm_val,
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
                for cand in annotated_var:
                    cand["scan_id"] = scan_id_str
                    cand["dm"] = dm_val
                    if "ra_deg" not in cand and "ra_rad" in cand:
                        cand["ra_deg"] = float(np.degrees(cand["ra_rad"]))
                    if "dec_deg" not in cand and "dec_rad" in cand:
                        cand["dec_deg"] = float(np.degrees(cand["dec_rad"]))

                # (RACS - PSRCAT) IPS AGN candidate clustering & suppression at DM > 0
                if dm_val == 0.0:
                    dm0_var_cands = [dict(c) for c in annotated_var]
                elif cfg.filter_ips_agn and len(dm0_var_cands) > 0 and (racs_coords is not None):
                    annotated_var = filters.filter_ips_agn_candidates(
                        annotated_var, dm0_var_cands, racs_coords, psrcat_coords,
                        ips_thresh=cfg.ips_snr_ratio_threshold,
                        ips_radius_arcsec=cfg.ips_match_radius_arcsec,
                    )

                t_var = candidates.candidates_to_astropy_table(annotated_var)
                candidates.save_candidates_table(
                    t_var,
                    csv_path=f"{var_root}_candidates.csv",
                    vot_path=f"{var_root}_candidates.vot"
                )
        elif cfg.plot_cands_only:
            annotated_var = _load_plot_only_candidates(
                candidates_dir=candidates_dir,
                root_path=var_root,
                ms_base=ms_base,
                scan_suffix=scan_suffix,
                dm_suffix=dm_suffix,
                search_type="variance",
            )

        _export_variance_chunk_products(
            cfg=cfg,
            times=times,
            cube=cube,
            std_map_partial=std_map_partial,
            annotated_var=annotated_var,
            var_root=var_root,
        )

    return (c_dm, m_dm, M2_dm), annotated_var


def _process_boxcar_chunk_dm(
    cfg: Config,
    times: np.ndarray,
    cube: np.ndarray,
    dm_val: float,
    chunk_root: str,
    scan_id_str: str,
    candidates_dir: str,
    ms_base: str,
    scan_suffix: str,
    dm_suffix: str,
    racs_coords,
    psrcat_coords,
    dm0_box_cands: list[dict],
) -> list[dict]:
    """Process boxcar detection for a single trial DM on a chunk."""
    box_root = chunk_root + "_boxcar"
    annotated: list[dict] = []
    if cfg.do_boxcar_search:
        dets, snr_cubes = detection.boxcar_search_time(
            times, cube,
            widths=cfg.boxcar_widths,
            widths_in_seconds=False,
            threshold_sigma=cfg.boxcar_threshold,
            return_snr_cubes=True,
            keep_top_k=50,
            std_mode="spatial_per_window",
            subtract_mean_per_pixel=True,
            use_local_threshold=cfg.use_local_threshold,
            local_window_size=cfg.local_window_size,
            dm=dm_val,
        )
        dets_by_w = filters.nms_snr_maps_per_width(
            snr_cubes, times,
            threshold_sigma=cfg.boxcar_threshold,
            spatial_radius=cfg.nms_radius, time_radius=2, valid_mask=None,
            cube=cube
        )
        final_dets = filters.group_filter_across_widths(
            dets_by_w, times,
            spatial_radius=cfg.nms_radius,
            time_radius=8,
            policy="max_snr",
            max_per_time_group=1,
            ny_nx=(cube.shape[1], cube.shape[2]),
            cube=cube
        )
        if len(final_dets) > 0:
            annotated = ducc_wcs.annotate_candidates_with_sky_coords(
                msname=cfg.msname, final_detections=final_dets,
                npix_x=cfg.npix_x, npix_y=cfg.npix_y,
                pixsize_x=cfg.pix_rad, pixsize_y=cfg.pix_rad,
                flip_u=True, flip_v=True, field_name=None
            )
            for cand in annotated:
                cand["scan_id"] = scan_id_str
                cand["dm"] = dm_val
                if "ra_deg" not in cand and "ra_rad" in cand:
                    cand["ra_deg"] = float(np.degrees(cand["ra_rad"]))
                if "dec_deg" not in cand and "dec_rad" in cand:
                    cand["dec_deg"] = float(np.degrees(cand["dec_rad"]))

            # (RACS - PSRCAT) IPS AGN candidate clustering & suppression at DM > 0
            if dm_val == 0.0:
                dm0_box_cands = [dict(c) for c in annotated]
            elif cfg.filter_ips_agn and len(dm0_box_cands) > 0 and (racs_coords is not None):
                annotated = filters.filter_ips_agn_candidates(
                    annotated, dm0_box_cands, racs_coords, psrcat_coords,
                    ips_thresh=cfg.ips_snr_ratio_threshold,
                    ips_radius_arcsec=cfg.ips_match_radius_arcsec,
                )

            t_box = candidates.candidates_to_astropy_table(annotated)
            candidates.save_candidates_table(
                t_box,
                csv_path=f"{box_root}_candidates.csv",
                vot_path=f"{box_root}_candidates.vot"
            )
    elif cfg.plot_cands_only:
        annotated = _load_plot_only_candidates(
            candidates_dir=candidates_dir,
            root_path=box_root,
            ms_base=ms_base,
            scan_suffix=scan_suffix,
            dm_suffix=dm_suffix,
            search_type="boxcar",
        )

    _export_boxcar_chunk_products(
        cfg=cfg,
        times=times,
        cube=cube,
        annotated=annotated,
        box_root=box_root,
    )

    return annotated


def process_chunk_task(cfg: Config, ms_base: str, candidates_dir: str, start: int, end: int, scan_id_str: str = "", chunk_times: np.ndarray | None = None):
    """
    Worker task: load the chunk data from MS once, loop over trial DMs in memory,
    run variance search on each trial DM (if enabled), run boxcar search on all trial DMs,
    and return per-chunk Welford aggregates.
    """
    dm_trials = cfg.dm_trials
    if not dm_trials:
        dm_trials = [float(cfg.current_dm)]

    # Load chunk visibility and coordinate data once from MS into RAM
    times, chan_freq, dt, vis_3d, wgt_3d, uvw_3d = imaging.load_chunk_data(
        msname=cfg.msname, t_main=None,
        start_time_idx=start, end_time_idx=end,
        chunk_times=chunk_times,
        corr_mode=cfg.corr_mode, basis=cfg.basis, single_pol=cfg.single_pol,
        data_column=cfg.data_column,
    )

    welford_by_dm = {}
    cube_dm0 = None
    scan_suffix = f"_scan_{scan_id_str}" if scan_id_str else ""

    # Tracking candidates at DM=0 for (RACS - PSRCAT) IPS suppression
    dm0_var_cands = []
    dm0_box_cands = []
    racs_coords = None
    psrcat_coords = None
    if cfg.filter_ips_agn:
        center_c = SkyCoord(cfg.ra0_rad, cfg.dec0_rad, unit="rad", frame="icrs")
        psrcat_coords = catalogues.get_cached_psrcat_coords()
        racs_coords = catalogues.get_cached_racs_coords(center_coord=center_c, cone_radius_deg=3.0)

    # Ensure DM 0.0 is processed first if present, so Welford state is accumulated early
    sorted_dm_trials = list(dm_trials)
    if 0.0 in sorted_dm_trials:
        sorted_dm_trials.remove(0.0)
        sorted_dm_trials.insert(0, 0.0)

    for dm_val in sorted_dm_trials:
        # In-memory dedispersion + exact UVW wavelength scaling + DUCC gridding
        cube = imaging.grid_chunk_cube(
            vis_3d=vis_3d,
            wgt_3d=wgt_3d,
            uvw_3d=uvw_3d,
            chan_freq=chan_freq,
            dt=dt,
            npix_x=cfg.npix_x,
            npix_y=cfg.npix_y,
            pixsize_x=cfg.pix_rad,
            pixsize_y=cfg.pix_rad,
            epsilon=cfg.epsilon,
            do_wgridding=cfg.do_wgridding,
            nthreads=cfg.nthreads,
            verbosity=cfg.verbosity,
            dm=dm_val,
            collapse_channels=cfg.collapse_channels,
            nsubbands=cfg.nsubbands,
            exact_uvw=cfg.exact_uvw,
            min_valid_channels=cfg.min_valid_channels,
        )

        is_search = cfg.is_dm_search or (dm_val != 0.0)
        dm_suffix = f"_dm{dm_val:06.2f}" if is_search else ""
        chunk_root = os.path.join(candidates_dir, f"{ms_base}{scan_suffix}{dm_suffix}_chunk_{start:06d}")

        # ---------------------------------------------------------------------
        # 1. Variance search: on each trial DM (if cfg.enable_var)
        # ---------------------------------------------------------------------
        if cfg.enable_var:
            welford_agg, var_cands = _process_variance_chunk_dm(
                cfg=cfg,
                times=times,
                cube=cube,
                dm_val=dm_val,
                chunk_root=chunk_root,
                scan_id_str=scan_id_str,
                candidates_dir=candidates_dir,
                ms_base=ms_base,
                scan_suffix=scan_suffix,
                dm_suffix=dm_suffix,
                racs_coords=racs_coords,
                psrcat_coords=psrcat_coords,
                dm0_var_cands=dm0_var_cands,
            )
            welford_by_dm[dm_val] = welford_agg
            if dm_val == 0.0:
                dm0_var_cands = var_cands

            if (dm_val == 0.0 or cube_dm0 is None) and cfg.save_full_var_lightcurves:
                cube_dm0 = cube.copy()

        # ---------------------------------------------------------------------
        # 2. Boxcar search: applied to ALL DMs (if cfg.enable_boxcar)
        # ---------------------------------------------------------------------
        if cfg.enable_boxcar:
            box_cands = _process_boxcar_chunk_dm(
                cfg=cfg,
                times=times,
                cube=cube,
                dm_val=dm_val,
                chunk_root=chunk_root,
                scan_id_str=scan_id_str,
                candidates_dir=candidates_dir,
                ms_base=ms_base,
                scan_suffix=scan_suffix,
                dm_suffix=dm_suffix,
                racs_coords=racs_coords,
                psrcat_coords=psrcat_coords,
                dm0_box_cands=dm0_box_cands,
            )
            if dm_val == 0.0:
                dm0_box_cands = box_cands

        # Immediately free cube memory before next DM trial
        del cube

    # Free large chunk arrays
    del vis_3d, wgt_3d, uvw_3d

    if cfg.save_full_var_lightcurves:
        return times, cube_dm0, welford_by_dm, scan_id_str
    else:
        return times, None, welford_by_dm, scan_id_str
