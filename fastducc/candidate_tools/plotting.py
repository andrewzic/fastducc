# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import glob
import logging
import subprocess
import socket
import getpass
import datetime
import textwrap
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

import numpy as np

import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.offsetbox import AnchoredText
from matplotlib.animation import FuncAnimation, PillowWriter

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
from astropy.time import Time
from astropy.nddata import Cutout2D

try:
    from ligo.skymap.plot.marker import reticle
except Exception:
    def reticle(which='rt'):
        return 'o'

from fastducc import wcs as ducc_wcs
from fastducc.filters import is_zero_flux_candidate
from fastducc import kernels
from fastducc.candidate_tools.utils import parse_candidate_filename
from fastducc.candidate_tools.snippets import compute_candidate_local_rms


def wrap_candidate_metadata_text(text: str, max_width: int = 33) -> str:
    """Wrap candidate summary metadata text cleanly so that no line exceeds max_width."""
    lines = []
    for raw_line in text.split("\n"):
        raw_line_s = raw_line.rstrip()
        if not raw_line_s:
            lines.append("")
            continue
        if len(raw_line_s) <= max_width:
            lines.append(raw_line_s)
            continue
        # Special handling for CASA timerange line (with ~)
        if "~" in raw_line_s:
            t0, t1 = raw_line_s.split("~", 1)
            lines.append(f"  {t0.strip()} ~")
            lines.append(f"  {t1.strip()}")
            continue
        # Check if line has a key: value structure
        if ":" in raw_line_s and not raw_line_s.startswith(" "):
            parts = raw_line_s.split(":", 1)
            k = parts[0].strip() + ":"
            v = parts[1].strip()
            if len(k) + 1 + len(v) <= max_width:
                lines.append(f"{k} {v}")
                continue
            lines.append(k)
            # Special handling for coordinates
            if " " in v and any(c in v for c in ("h", "d", "m", "s", ":")):
                v_parts = v.split()
                if len(v_parts) == 2:
                    lines.append(f"  {v_parts[0]}")
                    lines.append(f"  {v_parts[1]}")
                    continue
            v_spaced = v.replace(".", ". ").replace("_", "_ ")
            wrapped = textwrap.wrap(v_spaced, width=max_width - 2, break_long_words=True)
            for w in wrapped:
                lines.append("  " + w.replace(". ", ".").replace("_ ", "_"))
        else:
            indent = "  " if raw_line_s.startswith(" ") else ""
            content = raw_line_s.strip()
            content_spaced = content.replace(".", ". ").replace("_", "_ ")
            wrapped = textwrap.wrap(content_spaced, width=max_width - len(indent), break_long_words=True)
            for w in wrapped:
                lines.append(indent + w.replace(". ", ".").replace("_ ", "_"))
    return "\n".join(lines)


def save_candidate_summary(
    times: Optional[np.ndarray] = None,
    cube: Optional[np.ndarray] = None,
    candidate: Optional[Dict[str, Any]] = None,
    out_prefix: str = "",
    *,
    continuum_dir: Optional[str] = None,
    std_map: Optional[np.ndarray] = None,
    use_std_images: bool = False,
    spatial_size: int = 50,
    center_policy: str = "right",
    cmap: str = "viridis",
    dpi: int = 200,
    npix_x: Optional[int] = None, npix_y: Optional[int] = None,
    ra0_rad: Optional[float] = None, dec0_rad: Optional[float] = None,
    pix_rad: Optional[float] = None,
    ra_sign: int = -1, dec_sign: int = 1,
    radesys: str = "ICRS", equinox: float = None,
    method: str = "boxcar",
    var_highpass_cutoff_sec: float = 0.0,
    local_window_size: int = 64,
    rms_full: Optional[np.ndarray] = None,
    mean_cube: Optional[np.ndarray] = None,
    defer_summary_plots: bool = False,
    **kwargs: Any,
) -> Dict[str, str]:
    """
    Build a prepfold-like single candidate summary PDF page and save cutout/fullframe FITS.
    Automatically dispatches to save_periodicity_candidate_summary when processing periodicity candidates.
    """
    if candidate is None:
        raise ValueError("candidate dict must be provided.")

    if times is None or cube is None or ("period_s" in candidate or "f_hz" in candidate or candidate.get("algo") in ("uv_ffa", "periodicity", "nufft")):
        return save_periodicity_candidate_summary(
            candidate=candidate,
            out_prefix=out_prefix,
            cutout=candidate.get("cutout", kwargs.get("cutout")),
            fullframe=kwargs.get("fullframe"),
            profile=candidate.get("profile", kwargs.get("profile")),
            continuum_dir=continuum_dir,
            spatial_size=spatial_size,
            cmap=cmap,
            dpi=dpi,
            npix_x=npix_x,
            npix_y=npix_y,
            ra0_rad=ra0_rad,
            dec0_rad=dec0_rad,
            pix_rad=pix_rad,
            ra_sign=ra_sign,
            dec_sign=dec_sign,
            radesys=radesys,
            equinox=equinox,
            method=candidate.get("algo", method),
            out_dir_plots=kwargs.get("out_dir_plots"),
            out_dir_cutouts=kwargs.get("out_dir_cutouts"),
            out_dir_profiles=kwargs.get("out_dir_profiles"),
            out_dir_fullframes=kwargs.get("out_dir_fullframes"),
        )

    if cube.ndim != 3:
        raise ValueError("cube must be (T, Ny, Nx)")
    T, Ny, Nx = cube.shape
    if times.shape[0] != T:
        raise ValueError("times length must match cube time axis")
    if any(v is None for v in (npix_x, npix_y, ra0_rad, dec0_rad, pix_rad)):
        raise ValueError("Provide npix_x/npix_y and ra0_rad/dec0_rad/pix_rad for WCS.")

    if is_zero_flux_candidate(candidate, cube):
        logger.warning(f"Skipping summary plot for zero-flux candidate {candidate.get('srcname', '')}")
        return {}

    y = int(candidate["y"])
    x = int(candidate["x"])
    w = max(1, int(candidate.get("width_samples", 1)))

    if "t0_idx" in candidate:
        t0_idx = int(candidate["t0_idx"])
    else:
        cidx = int(candidate.get("center_idx", 0))
        t0_idx = cidx - (w // 2) if (center_policy == "right") else cidx - ((w - 1) // 2)
        t0_idx = max(0, min(t0_idx, T - w))
    offset = (w // 2) if (center_policy == "right") else ((w - 1) // 2)
    t_full_center = max(0, min(t0_idx + offset, T - 1))

    lc_full = cube[:, y, x].astype(np.float64, copy=False)
    lc_sm, T_eff = kernels._compute_boxcar_1d(lc_full, w)
    times_sm = times[offset:offset + T_eff] if T_eff > 0 else np.empty((0,), dtype=times.dtype)

    if rms_full is None:
        rms_full = compute_candidate_local_rms(cube, candidate, window_size=local_window_size)
    else:
        rms_full = np.asarray(rms_full, dtype=np.float64).copy()
    bad_rms = (~np.isfinite(rms_full)) | (rms_full <= 0.0) | (lc_full == 0.0) | (~np.isfinite(lc_full))
    if np.any(bad_rms):
        rms_full[bad_rms] = 0.0

    imjd = int(times[0] / 86400.0)
    mjd = times / 86400.0
    rel_times_start = times - times[0]
    rel_times_start_sm = times_sm - times[0]
    det_time = candidate.get("time_center", np.nan)
    det_time_mjd = det_time / 86400.0 if np.isfinite(det_time) else np.nan
    det_time_str = f"{det_time_mjd:.7f} MJD" if np.isfinite(det_time_mjd) else "—"

    if use_std_images and (std_map is not None):
        frame_for_display = std_map
        frame_title_suffix = "std map"
    elif w > 1:
        t0_avg = max(0, t0_idx)
        t1_avg = min(T, t0_idx + w)
        frame_for_display = np.mean(cube[t0_avg:t1_avg], axis=0)
        frame_title_suffix = f"avg w={w}"
    else:
        frame_for_display = cube[t_full_center]
        frame_title_suffix = ""
    peak_flux = frame_for_display[y, x]
    vmin = np.nanpercentile(frame_for_display, 5.0)
    if np.isfinite(peak_flux) and peak_flux > vmin:
        vmax = peak_flux
    else:
        vmax = np.nanpercentile(frame_for_display, 99.5)

    wcs_full = ducc_wcs._build_fullframe_wcs(
        npix_x=npix_x, npix_y=npix_y,
        ra0_rad=ra0_rad, dec0_rad=dec0_rad,
        pixscale_rad=pix_rad, ra_sign=ra_sign, dec_sign=dec_sign,
        radesys=radesys, equinox=equinox
    )

    if "ra_rad" in candidate and "dec_rad" in candidate:
        ra_c = float(candidate["ra_rad"])
        dec_c = float(candidate["dec_rad"])
    else:
        ra_c = ra0_rad
        dec_c = dec0_rad

    half_sp = spatial_size // 2
    y0_des = y - half_sp
    y1_des = y0_des + spatial_size
    x0_des = x - half_sp
    x1_des = x0_des + spatial_size

    y0_clip = max(0, y0_des)
    y1_clip = min(Ny, y1_des)
    x0_clip = max(0, x0_des)
    x1_clip = min(Nx, x1_des)

    cut = frame_for_display[y0_clip:y1_clip, x0_clip:x1_clip]

    pad_y_top = y0_clip - y0_des
    pad_y_bot = y1_des - y1_clip
    pad_x_left = x0_clip - x0_des
    pad_x_right = x1_des - x1_clip

    cutout_det = np.pad(
        cut,
        ((pad_y_top, pad_y_bot), (pad_x_left, pad_x_right)),
        mode="constant",
        constant_values=np.nan,
    )

    wcs_cut = ducc_wcs._build_tan_wcs_for_snippet(
        spatial_size=spatial_size,
        ra_rad=ra_c, dec_rad=dec_c,
        pixscale_rad=pix_rad, ra_sign=ra_sign, dec_sign=dec_sign,
        radesys=radesys, equinox=equinox
    )

    cont_im, cont_wcs, cont_vmin, cont_vmax = None, None, None, None
    if continuum_dir:
        meta_guess = parse_candidate_filename(out_prefix)
        beam_lbl = meta_guess.get("beam", "") if isinstance(meta_guess, dict) else ""
        logger.debug(f"Looking for continuum FITS in '{continuum_dir}' with beam label '{beam_lbl}'...")
        fits_cands = sorted(glob.glob(os.path.join(continuum_dir, f"*{beam_lbl}*.fits")))
        if len(fits_cands) == 0:
            fits_cands = sorted(glob.glob(os.path.join(continuum_dir, "*.fits")))
        if len(fits_cands) > 0:
            with fits.open(fits_cands[0], memmap=False) as hdul:
                data = hdul[0].data
                hdr  = hdul[0].header
            if data.ndim > 2:
                data = np.squeeze(data)
            cont_w = WCS(hdr)
            cont_w_2d = cont_w.celestial
            size_rad = spatial_size * float(pix_rad)
            pos = SkyCoord(ra=np.degrees(ra_c) * u.deg, dec=np.degrees(dec_c) * u.deg, frame="icrs")
            c2d = Cutout2D(
                data=data,
                position=pos,
                size=(size_rad * u.rad, size_rad * u.rad),
                wcs=cont_w_2d,
                mode="partial",
                fill_value=np.nan,
            )
            cont_im = c2d.data
            cont_wcs = c2d.wcs
            cont_vmin = np.nanpercentile(cont_im, 5.0)
            cont_vmax = np.nanpercentile(cont_im, 99.5)

    if np.isfinite(det_time_mjd):
        mjd_tag = f"_MJD{det_time_mjd:.7f}"
    else:
        mjd_tag = ""
    out_fig = f"{out_prefix}{mjd_tag}_summary.pdf"

    # Dynamic spectrum integration: check candidate dict, kwargs, or disk
    dynspec_obj = candidate.get("dynspec") or kwargs.get("dynspec")
    ds_path_val = candidate.get("ds_path") or kwargs.get("ds_path")
    if dynspec_obj is None and ds_path_val is None:
        src_name = candidate.get("srcname", "")
        patterns = [
            f"{out_prefix}.ds",
            f"{out_prefix}_cand_{src_name}.ds",
            f"{out_prefix}_dynspec.ds",
        ]
        parent_dir = os.path.dirname(out_prefix)
        if parent_dir and src_name:
            patterns.extend([
                os.path.join(parent_dir, f"*{src_name}*.ds"),
                os.path.join(parent_dir, "dynamic_spectra", f"*{src_name}*.ds"),
                os.path.join(parent_dir, "..", "dynamic_spectra", f"*{src_name}*.ds"),
                os.path.join(parent_dir, "..", "candidates", f"*{src_name}*.ds"),
            ])
        for p in patterns:
            matches = glob.glob(p)
            if matches:
                ds_path_val = matches[0]
                break

    has_dynspec = False
    ds_slice = None
    ds_t_slice = None
    ds_freqs = None
    if dynspec_obj is not None or ds_path_val is not None:
        try:
            from fastducc.dynspec import load_dynamic_spectrum_hdf5
            if dynspec_obj is not None and isinstance(dynspec_obj, dict):
                ds_dict = dynspec_obj
            elif ds_path_val is not None and os.path.isfile(str(ds_path_val)):
                ds_dict = load_dynamic_spectrum_hdf5(ds_path_val)
            else:
                ds_dict = None

            if ds_dict is not None and "data" in ds_dict and "freq" in ds_dict:
                ds_raw_data = np.asarray(ds_dict["data"], dtype=np.float32)
                ds_freqs = np.asarray(ds_dict["freq"], dtype=np.float64)
                ds_raw_times = ds_dict.get("time_raw")

                if ds_raw_times is not None and len(ds_raw_times) == ds_raw_data.shape[0] and len(times) > 0:
                    t_start = times[0]
                    t_end = times[-1]
                    idx0 = max(0, int(np.searchsorted(ds_raw_times, t_start)) - 1)
                    idx1 = min(len(ds_raw_times), int(np.searchsorted(ds_raw_times, t_end)) + 2)
                    if idx1 > idx0:
                        ds_slice = ds_raw_data[idx0:idx1, :]
                        ds_t_slice = ds_raw_times[idx0:idx1] - t_start
                    else:
                        ds_slice = ds_raw_data
                        ds_t_slice = ds_dict.get("time", np.arange(len(ds_slice), dtype=float))
                else:
                    ds_slice = ds_raw_data
                    ds_t_slice = ds_dict.get("time", np.arange(len(ds_slice), dtype=float))

                if ds_slice is not None and ds_slice.size > 0:
                    has_dynspec = True
        except Exception as e:
            logger.debug(f"Could not load dynamic spectrum for summary plot: {e}")

    if has_dynspec:
        fig = plt.figure(figsize=(18, 12.5), dpi=dpi)
        gs = GridSpec(
            nrows=3, ncols=3, figure=fig,
            height_ratios=[1.0, 0.96, 0.80], width_ratios=[1.0, 1.0, 0.95],
            hspace=0.28, wspace=0.20,
        )
    else:
        fig = plt.figure(figsize=(18, 12), dpi=dpi)
        gs = GridSpec(
            nrows=2, ncols=3, figure=fig,
            height_ratios=[2.0, 1.0], width_ratios=[1.0, 1.0, 1.0],
        )

    if cont_wcs is not None:
        ax_cont = fig.add_subplot(gs[0, 0], projection=cont_wcs)
        im_cont = ax_cont.imshow(cont_im, origin="lower", cmap=cmap, vmin=cont_vmin, vmax=cont_vmax)
        ax_cont.coords.grid(True, color="white", alpha=0.35, ls=":")
        ax_cont.set_xlabel("RA (J2000)")
        ax_cont.set_ylabel("Dec (J2000)")
        ax_cont.set_aspect('equal', adjustable='box')
        ax_cont.plot([np.degrees(ra_c)], [np.degrees(dec_c)],
                     marker=reticle(which='rt'), ms=39, mec="black", mew=3.0, mfc="none",
                     transform=ax_cont.get_transform('world'))
        ax_cont.plot([np.degrees(ra_c)], [np.degrees(dec_c)],
                     marker=reticle(which='rt'), ms=36, mec="white", mew=1.5, mfc="none",
                     transform=ax_cont.get_transform('world'))
        cb3 = fig.colorbar(
            im_cont, ax=ax_cont,
            orientation="horizontal", fraction=0.046, pad=0.10
        )
        cb3.set_label("Flux density (mJy/beam)")
        ax_cont.set_title("Cont.")
    else:
        ax_cont = fig.add_subplot(gs[0, 0], projection=wcs_full)
        if mean_cube is None:
            mean_cube = np.mean(cube, axis=0)
        mean_vmin = np.nanpercentile(mean_cube, 5.0)
        mean_vmax = np.nanpercentile(mean_cube, 99.5)
        im_cont = ax_cont.imshow(mean_cube, origin="lower", cmap=cmap, vmin=mean_vmin, vmax=mean_vmax)
        ax_cont.coords.grid(True, color="white", alpha=0.35, ls=":")
        ax_cont.set_xlabel("RA (J2000)")
        ax_cont.set_ylabel("Dec (J2000)")
        ax_cont.set_aspect('equal', adjustable='box')
        ax_cont.plot([x], [y], marker=reticle(which='rt'), ms=39, mec="black", mew=3.0, mfc="none")
        ax_cont.plot([x], [y], marker=reticle(which='rt'), ms=36, mec="white", mew=1.5, mfc="none")

        cb3 = fig.colorbar(
            im_cont, ax=ax_cont,
            orientation="horizontal", fraction=0.046, pad=0.10
        )
        cb3.set_label("Flux density (mJy/beam)")
        ax_cont.set_title("Mean cube (no cont.)")

    ax_full = fig.add_subplot(gs[0, 1], projection=wcs_full)
    im_full = ax_full.imshow(frame_for_display, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax_full.set_xlabel("RA (J2000)")
    ax_full.set_ylabel("Dec (J2000)")
    ax_full.set_aspect('equal', adjustable='box')
    ax_full.plot([x], [y], marker=reticle(which='rt'), ms=39, mec="black", mew=3.0, mfc="none")
    ax_full.plot([x], [y], marker=reticle(which='rt'), ms=36, mec="white", mew=1.5, mfc="none")

    cb1 = fig.colorbar(
        im_full, ax=ax_full,
        orientation="horizontal", fraction=0.046, pad=0.10
    )
    cb1.set_label("Flux density (mJy/beam)")
    if frame_title_suffix:
        ax_full.set_title(f"Field @ {det_time_str} ({frame_title_suffix})")
    else:
        ax_full.set_title(f"Field @ {det_time_str}")

    ax_cut = fig.add_subplot(gs[0, 2], projection=wcs_cut)
    im_cut = ax_cut.imshow(cutout_det, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax_cut.set_xlabel("RA (J2000)")
    ax_cut.set_ylabel("Dec (J2000)")
    ax_cut.set_aspect('equal', adjustable='box')
    ax_cut.plot([half_sp], [half_sp], marker=reticle(which='rt'), ms=39, mec="black", mew=3.0, mfc="none")
    ax_cut.plot([half_sp], [half_sp], marker=reticle(which='rt'), ms=36, mec="white", mew=1.5, mfc="none")

    cb2 = fig.colorbar(
        im_cut, ax=ax_cut,
        orientation="horizontal", fraction=0.046, pad=0.10
    )
    cb2.set_label("Flux density (mJy/beam)")

    if frame_title_suffix:
        ax_cut.set_title(f"Cutout @ {det_time_str} ({frame_title_suffix})")
    else:
        ax_cut.set_title(f"Cutout @ {det_time_str}")

    if has_dynspec:
        ax_txt = fig.add_subplot(gs[1:3, 2])
    else:
        ax_txt = fig.add_subplot(gs[1, 2])
    ax_txt.set_axis_off()

    ra_hms = candidate.get("ra_hms", None)
    dec_dms = candidate.get("dec_dms", None)
    if (not ra_hms) or (not dec_dms):
        ra_hms, dec_dms = ducc_wcs.rad_to_hmsdms(ra_c, dec_c, dp=1)

    meta = parse_candidate_filename(out_prefix)
    field = meta.get("field", "") if isinstance(meta, dict) else ""
    sbid  = meta.get("sbid", "") if isinstance(meta, dict) else ""
    scan  = meta.get("scan_id", "") if isinstance(meta, dict) else ""
    beam  = meta.get("beam", "") if isinstance(meta, dict) else ""

    cand_scan = str(candidate.get("scan_id", "") or "")
    if cand_scan and not scan:
        scan = cand_scan
    if not scan or (scan.isdigit() and len(scan) != 14):
        t_ref = candidate.get("time_start") or candidate.get("time_center") or times[0]
        if t_ref is not None and np.isfinite(t_ref):
            try:
                scan = Time(float(t_ref) / 86400.0, format="mjd", scale="utc").strftime("%Y%m%d%H%M%S")
            except Exception:
                pass

    snr_val = candidate.get("snr", np.nan)
    snr_str = f"{float(snr_val):.2f}" if np.isfinite(snr_val) else "—"
    cand_dm = float(candidate.get("dm", 0.0) if candidate.get("dm") is not None and not np.isnan(float(candidate.get("dm", 0.0))) else 0.0)
    dm_str = f"{cand_dm:.2f} pc/cm³"
    width_str = f"{w:d}" if (method.lower() == "boxcar") else "—"

    cand_t_start = candidate.get("time_start", np.nan)
    cand_t_end = candidate.get("time_end", np.nan)
    cand_t0_idx = candidate.get("t0_idx", None)
    cand_t1_idx = candidate.get("t1_idx_excl", None)

    if np.isfinite(cand_t_start) and np.isfinite(cand_t_end):
        t_start_obj = Time(cand_t_start / 86400.0, format='mjd', scale='utc')
        t_end_obj = Time(cand_t_end / 86400.0, format='mjd', scale='utc')
        casa_start = t_start_obj.iso.replace('-', '/').replace(' ', '/')
        casa_end = t_end_obj.iso.replace('-', '/').replace(' ', '/')
        casa_timerange = f"{casa_start}~{casa_end}"
    else:
        casa_timerange = "—"

    if cand_t0_idx is not None and cand_t1_idx is not None:
        wsclean_interval = f"-interval {int(cand_t0_idx)} {int(cand_t1_idx)}"
    else:
        wsclean_interval = "—"

    srcname = candidate.get("srcname", "—")

    # Evaluate transient candidate morphology and lightcurve flux integrity
    morph_res = None
    if cutout_det is not None and cutout_det.size > 0:
        try:
            from fastducc.filters import evaluate_candidate_morphology_and_flux
            on_bins_t = None
            if "t0_idx" in candidate:
                on_bins_t = list(range(int(candidate["t0_idx"]), int(candidate.get("t1_idx_excl", int(candidate["t0_idx"]) + w))))
            morph_res = evaluate_candidate_morphology_and_flux(
                cutout=cutout_det,
                profile=lc_full,
                on_bins=on_bins_t,
            )
        except Exception as e:
            logger.debug(f"Could not compute transient morphology metrics: {e}")

    morph_str = ""
    if morph_res is not None:
        status_pt = "Pass" if morph_res["is_point_like"] else "FAIL (Extended)"
        status_flx = "Pass" if morph_res["is_valid_flux"] else "FAIL (Neg Trough)"
        morph_str = (
            f"\nMorphology: {status_pt}\n"
            f"  HP Lag-1: {morph_res['hp_lag1']:.3f} | Sharp: {morph_res['sharpness']:.2f}\n"
            f"Flux Quality: {status_flx}\n"
            f"  On-Share: {morph_res['on_flux_share']:.2f} | On-SNR: {morph_res['on_snr']:.1f}"
        )

    txt = (
        f"Name: {srcname}\n"
        f"Field: {field}\n"
        f"S/N: {snr_str}\n"
        f"DM: {dm_str}\n"
        f"SBID: {sbid}\n"
        f"Beam: {beam}\n"        
        f"Scan: {scan}\n"
        f"Coord (J2000): {ra_hms} {dec_dms}\n"
        f"Coord (x,y): ({x:d}, {y:d})\n"
        f"iMJD: {imjd}\n"
        f"Obs time (start): {mjd[0]:.7f} MJD\n"
        f"Det. time: {det_time_str}\n"
        f"Method: {method}\n"
        f"Width: {width_str}\n"
        f"wsclean: {wsclean_interval}\n"
        f"CASA timerange:\n"
        f"{casa_timerange}"
        f"{morph_str}"
    )

    if has_dynspec:
        f_min_ds = float(ds_freqs[0]) if (ds_freqs is not None and len(ds_freqs) > 0) else 0.0
        f_max_ds = float(ds_freqs[-1]) if (ds_freqs is not None and len(ds_freqs) > 0) else 0.0
        if ds_path_val:
            ds_name = Path(ds_path_val).name
            txt += f"\nDynamic Spectrum:\n  {ds_name}\n  ({len(ds_freqs)} ch | {f_min_ds:.1f}-{f_max_ds:.1f} MHz)"
        wrapped_txt = wrap_candidate_metadata_text(txt, max_width=33)
        ax_txt.text(
            0.03, 0.98, wrapped_txt,
            transform=ax_txt.transAxes,
            va="top", ha="left",
            fontsize=9.5,
            linespacing=1.26,
            bbox=dict(boxstyle="square,pad=0.8", facecolor="#fcfcfc", edgecolor="#bbbbbb", lw=1.2),
        )
        ax_txt.set_title("Candidate Summary", pad=8, fontsize=11)
    else:
        box = AnchoredText(txt, loc="upper left", frameon=True, pad=1.0, prop={"size": 10})
        ax_txt.add_artist(box)
        ax_txt.set_title("Summary", pad=12)

    if has_dynspec:
        fig.canvas.draw()
        fig_w, fig_h = fig.get_size_inches()
        bbox_ds_area = gs[1, 0:2].get_position(fig)
        total_w_px = bbox_ds_area.width * fig_w * dpi
        total_h_px = bbox_ds_area.height * fig_h * dpi
        h_beam_px = (0.20 / 1.20) * total_h_px
        w_cbar_px = 24.0
        pad_cbar_px = 14.0
        w_ds_spec_total = max(100.0, total_w_px - w_cbar_px - pad_cbar_px)
        w_spec_px = h_beam_px
        w_ds_px = max(100.0, w_ds_spec_total - w_spec_px)

        gs_outer = GridSpecFromSubplotSpec(
            nrows=1, ncols=2,
            subplot_spec=gs[1, 0:2],
            width_ratios=[w_ds_spec_total, w_cbar_px],
            wspace=pad_cbar_px / w_ds_spec_total,
        )

        gs_inner = GridSpecFromSubplotSpec(
            nrows=2, ncols=2,
            subplot_spec=gs_outer[0],
            height_ratios=[1.0, 0.20],
            width_ratios=[w_ds_px, w_spec_px],
            hspace=0.0, wspace=0.0,
        )

        ax_ds = fig.add_subplot(gs_inner[0, 0])
        ax_spec = fig.add_subplot(gs_inner[0, 1], sharey=ax_ds)
        ax_beam_lc = fig.add_subplot(gs_inner[1, 0], sharex=ax_ds)
        cax = fig.add_subplot(gs_outer[1])

        # Row 2: Main Image-plane Lightcurve matching height and aligned with ax_ds
        gs_lc_outer = GridSpecFromSubplotSpec(
            nrows=1, ncols=2,
            subplot_spec=gs[2, 0:2],
            width_ratios=[w_ds_spec_total, w_cbar_px],
            wspace=pad_cbar_px / w_ds_spec_total,
        )
        gs_lc_inner = GridSpecFromSubplotSpec(
            nrows=1, ncols=2,
            subplot_spec=gs_lc_outer[0],
            width_ratios=[w_ds_px, w_spec_px],
            wspace=0.0,
        )
        ax_lc = fig.add_subplot(gs_lc_inner[0, 0], sharex=ax_ds)

        # Downsample dynamic spectrum for display according to candidate detection width W
        ds_disp = ds_slice
        t_bin = max(1, w // 4)
        f_bin = t_bin
        if (t_bin > 1 or f_bin > 1) and ds_slice.ndim == 2:
            Nt, Nf = ds_slice.shape
            Nt_trim = (Nt // t_bin) * t_bin
            Nf_trim = (Nf // f_bin) * f_bin
            if Nt_trim > 0 and Nf_trim > 0:
                sub = ds_slice[:Nt_trim, :Nf_trim]
                ds_disp = np.nanmean(
                    sub.reshape(Nt_trim // t_bin, t_bin, Nf_trim // f_bin, f_bin),
                    axis=(1, 3)
                )

        vmin_ds, vmax_ds = np.nanpercentile(ds_disp, [5.0, 99.5])
        t_min_ds = float(ds_t_slice[0])
        t_max_ds = float(ds_t_slice[-1])
        f_min_ds = float(ds_freqs[0])
        f_max_ds = float(ds_freqs[-1])
        im_ds = ax_ds.imshow(
            ds_disp.T,
            origin="lower",
            aspect="auto",
            extent=[t_min_ds, t_max_ds, f_min_ds, f_max_ds],
            cmap=cmap,
            vmin=vmin_ds,
            vmax=vmax_ds,
        )
        ax_ds.set_ylabel("Frequency (MHz)", fontsize=10)
        ax_ds.set_title("Dynamic spectrum", fontsize=11, fontweight="normal")
        if np.isfinite(det_time):
            t_det_rel = det_time - times[0]
            if t_min_ds <= t_det_rel <= t_max_ds:
                ax_ds.axvline(t_det_rel, color="crimson", ls="--", lw=1.2, alpha=0.85, label="Det Time")
                if cand_dm > 0:
                    nu_top = max(f_min_ds, f_max_ds)
                    f_eval = np.linspace(min(f_min_ds, f_max_ds), nu_top, 100)
                    dt_sweep = 4.1488e3 * cand_dm * (f_eval**(-2) - nu_top**(-2))
                    ax_ds.plot(t_det_rel + dt_sweep, f_eval, color="gold", ls=":", lw=1.5, alpha=0.9, label=f"DM={cand_dm:.1f} track")
        ax_ds.tick_params(labelbottom=False, bottom=False)

        cb_ds = fig.colorbar(im_ds, cax=cax, orientation="vertical")
        cb_ds.set_label("Flux density (Jy)", fontsize=9)

        # Time-averaged spectrum to the right of imshow (wspace=0)
        beam_spec = np.nanmean(ds_slice, axis=0)
        ax_spec.plot(beam_spec, ds_freqs, color="indigo", lw=1.2)
        ax_spec.tick_params(labelleft=False, left=False)
        ax_spec.set_xlabel("Mean (Jy)", fontsize=8)
        ax_spec.grid(True, alpha=0.3)

        # Frequency-averaged lightcurve below imshow (hspace=0)
        beam_lc = np.nanmean(ds_slice, axis=1) * 1000.0  # mJy
        ax_beam_lc.plot(ds_t_slice, beam_lc, color="teal", lw=1.2)
        if np.isfinite(det_time):
            t_det_rel = det_time - times[0]
            if t_min_ds <= t_det_rel <= t_max_ds:
                ax_beam_lc.axvline(t_det_rel, color="crimson", ls="--", lw=1.2, alpha=0.85)
        ax_beam_lc.set_ylabel("Beamformed\n(mJy)", fontsize=8)
        ax_beam_lc.tick_params(labelbottom=False)
        ax_beam_lc.grid(True, alpha=0.3)
    else:
        ax_lc = fig.add_subplot(gs[1, 0:2])
    if method.lower() == "variance" and var_highpass_cutoff_sec > 0:
        lc_filtered = np.empty_like(lc_full)
        ema = np.nanmean(lc_full)
        for ii in range(len(lc_full)):
            dt = (times[ii] - times[max(0, ii - 1)]) if ii > 0 else 0.0
            alpha = 1.0 - np.exp(-dt / var_highpass_cutoff_sec)
            ema = ema * (1.0 - alpha) + lc_full[ii] * alpha
            lc_filtered[ii] = lc_full[ii] - ema
        ax_lc.errorbar(rel_times_start, lc_full, yerr=rms_full, fmt="none",
                       ecolor="grey", elinewidth=0.8, capsize=0, alpha=0.6, zorder=1)
        ax_lc.plot(rel_times_start, lc_full, color="lightsteelblue", lw=1.0, alpha=0.7, label="Raw", zorder=2)
        ax_lc.plot(rel_times_start, lc_filtered, color="tab:green", lw=1.4,
                   label=f"EMA highpass (τ={var_highpass_cutoff_sec:.0f}s)", zorder=3)
    else:
        ax_lc.errorbar(rel_times_start, lc_full, yerr=rms_full, fmt="none",
                       ecolor="grey", elinewidth=0.8, capsize=0, alpha=0.6, zorder=1)
        ax_lc.plot(rel_times_start, lc_full, color="tab:blue", lw=1.4, label="Raw", zorder=2)
        if T_eff > 0:
            ax_lc.plot(rel_times_start_sm, lc_sm, color="tab:green", lw=1.6, label=f"Boxcar (w={w})", zorder=3)
            k_center = max(0, min(t0_idx, T_eff - 1))
            ax_lc.axvline(rel_times_start_sm[k_center], color="crimson", ls="--", lw=1.0, zorder=4)
        else:
            ax_lc.text(0.02, 0.92, f"w={w} > T; smoothed LC N/A",
                       transform=ax_lc.transAxes, ha="left", va="top", color="crimson")
    ax_lc.set_xlabel(f"Time from MJD {mjd[0]:.7f} (s)")
    ax_lc.set_ylabel("Flux density (mJy/beam)")
    ax_lc.grid(True, alpha=0.3)
    ax_lc.legend(loc="best")

    gen_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    host = socket.gethostname()
    username = getpass.getuser()
    git_hash = "—"
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        pass
    
    meta_footer = f"Generated: {gen_time} | Host: {host} | User: {username} | Git: {git_hash}"
    fig.text(0.5, 0.01, meta_footer, ha="center", va="bottom", fontsize=8, color="gray", alpha=0.7)

    if not defer_summary_plots:
        fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)

    out_fits = f"{out_prefix}{mjd_tag}_cutout.fits"
    hdr = wcs_cut.to_header()
    hdr["BUNIT"] = "JY/BEAM"
    hdr["COMMENT"] = "Detection frame from transient snippet"
    if ra_hms and dec_dms:
        hdr["OBJRA"] = ra_hms
        hdr["OBJDEC"] = dec_dms
        
    hdr["CDELT1"] = wcs_cut.wcs.cdelt[0]
    hdr["CDELT2"] = wcs_cut.wcs.cdelt[1]
    hdr["PC1_1"] = 1.0
    hdr["PC1_2"] = 0.0
    hdr["PC2_1"] = 0.0
    hdr["PC2_2"] = 1.0
    
    if getattr(wcs_cut.wcs, "radesys", None):
        hdr["RADESYS"] = wcs_cut.wcs.radesys
        
    hdu = fits.PrimaryHDU(data=cutout_det.astype(np.float32), header=hdr)
    hdu.writeto(out_fits, overwrite=True)

    out_fits_full = f"{out_prefix}{mjd_tag}_fullframe.fits"
    hdr_full = wcs_full.to_header()
    hdr_full["BUNIT"] = "JY/BEAM"
    hdr_full["COMMENT"] = "Detection frame full field from transient imaging"
    if ra_hms and dec_dms:
        hdr_full["OBJRA"] = ra_hms
        hdr_full["OBJDEC"] = dec_dms

    hdr_full["CDELT1"] = wcs_full.wcs.cdelt[0]
    hdr_full["CDELT2"] = wcs_full.wcs.cdelt[1]
    hdr_full["PC1_1"] = 1.0
    hdr_full["PC1_2"] = 0.0
    hdr_full["PC2_1"] = 0.0
    hdr_full["PC2_2"] = 1.0

    if getattr(wcs_full.wcs, "radesys", None):
        hdr_full["RADESYS"] = wcs_full.wcs.radesys

    hdu_full = fits.PrimaryHDU(data=frame_for_display.astype(np.float32), header=hdr_full)
    hdu_full.writeto(out_fits_full, overwrite=True)

    fig_ret = None if defer_summary_plots else out_fig
    return {"figure": fig_ret, "fits_cutout": out_fits, "fits_fullframe": out_fits_full}


def save_candidate_snippet_products(
    snippet_rec: dict,
    out_prefix: str,
    *,
    pixscale_rad: float,
    ra_rad: float,
    dec_rad: float,
    ra_sign: int = -1,
    dec_sign: int = 1,
    cmap: str = "viridis",
    gif_fps: int = 5,
    dpi: int = 300
) -> dict:
    """Save animated GIF across time with RA/Dec labels (WCSAxes) for a candidate snippet."""
    cube = snippet_rec["snippet_cube"]
    times = snippet_rec["snippet_times"]
    T, N, M = cube.shape
    assert N == M, "snippet must be square"

    t_center = T // 2
    y_center = N // 2
    x_center = M // 2

    wcs2d = ducc_wcs._build_tan_wcs_for_snippet(
        spatial_size=N,
        ra_rad=ra_rad,
        dec_rad=dec_rad,
        pixscale_rad=pixscale_rad,
        ra_sign=ra_sign,
        dec_sign=dec_sign,
        radesys="ICRS",
        equinox=None
    )
    
    det_frame = cube[t_center]
    peak_flux = det_frame[y_center, x_center]
    vmin = np.nanpercentile(det_frame, 5.0)
    if np.isfinite(peak_flux) and peak_flux > vmin:
        vmax = np.abs(peak_flux)
    else:
        vmax = np.nanpercentile(det_frame, 99.5)

    out_gif  = f"{out_prefix}_snippet.gif"

    fig = plt.figure(figsize=(10, 10), dpi=dpi)
    ax = fig.add_subplot(111, projection=wcs2d)
    im = ax.imshow(cube[0], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.coords.grid(True, color="white", alpha=0.3, ls=":")
    ax.set_xlabel("Right Ascension (J2000)")
    ax.set_ylabel("Declination (J2000)")
    ax.set_aspect('equal', adjustable='box')
    ax.plot([x_center], [y_center], marker=reticle(which='rt'), ms=35, mec="black", mew=3.0, mfc="none")
    ax.plot([x_center], [y_center], marker=reticle(which='rt'), ms=32, mec="white", mew=1.5, mfc="none")

    def _update(k: int):
        im.set_data(cube[k])
        t = times[k]
        if np.isfinite(t):
            ax.set_title(f"t = {t:.2f} s")
        else:
            ax.set_title("t = (pad)")
        return (im,)

    anim = FuncAnimation(fig, _update, frames=T, interval=1000//gif_fps, blit=False)
    anim.save(out_gif, writer=PillowWriter(fps=gif_fps))
    plt.close(fig)

    return {"gif": out_gif}


def _crop_or_pad_cutout(img: np.ndarray, target_size: int) -> np.ndarray:
    """Crop or pad 2D array to (target_size, target_size) keeping center aligned."""
    ny, nx = img.shape
    if ny == target_size and nx == target_size:
        return img.astype(np.float32)
    out = np.full((target_size, target_size), np.nan, dtype=np.float32)
    sy = min(ny, target_size)
    sx = min(nx, target_size)
    y0_src = (ny - sy) // 2
    x0_src = (nx - sx) // 2
    y0_dst = (target_size - sy) // 2
    x0_dst = (target_size - sx) // 2
    out[y0_dst:y0_dst + sy, x0_dst:x0_dst + sx] = img[y0_src:y0_src + sy, x0_src:x0_src + sx]
    return out


def save_periodicity_candidate_summary(
    candidate: Dict[str, Any],
    out_prefix: str,
    *,
    cutout: Optional[np.ndarray] = None,
    cutout_off: Optional[np.ndarray] = None,
    fullframe: Optional[np.ndarray] = None,
    profile: Optional[np.ndarray] = None,
    continuum_dir: Optional[str] = None,
    spatial_size: int = 32,
    cmap: str = "viridis",
    dpi: int = 200,
    npix_x: Optional[int] = None,
    npix_y: Optional[int] = None,
    ra0_rad: Optional[float] = None,
    dec0_rad: Optional[float] = None,
    pix_rad: Optional[float] = None,
    ra_sign: int = -1,
    dec_sign: int = 1,
    radesys: str = "ICRS",
    equinox: Optional[float] = None,
    wcs_full: Optional[WCS] = None,
    method: str = "UV-FFA (c2r)",
    out_dir_plots: Optional[str] = None,
    out_dir_cutouts: Optional[str] = None,
    out_dir_profiles: Optional[str] = None,
    out_dir_fullframes: Optional[str] = None,
) -> Dict[str, str]:
    """
    Build a publication-quality FastDUCC candidate summary PDF and candidate artefacts
    (FITS cutouts, FITS fullframes, folded profile arrays) for periodicity search candidates.
    Matches the 2x3 GridSpec layout, information density, and style of FastDUCC transient summaries.
    """
    x = int(candidate.get("x", 0))
    y = int(candidate.get("y", 0))
    snr_val = float(candidate.get("snr", 0.0))
    period_s = float(candidate.get("period_s", candidate.get("period", 0.0)))
    f_cand = candidate.get("f_hz")
    if f_cand is not None and np.isfinite(float(f_cand)) and float(f_cand) > 0:
        f_hz = float(f_cand)
    else:
        f_hz = 1.0 / period_s if period_s > 0 else 0.0
    dm_val = float(candidate.get("dm", 0.0)) if candidate.get("dm") is not None else 0.0
    width_bins = max(1, int(candidate.get("width_bins", candidate.get("width", 1))))
    duty_cycle = float(candidate.get("duty_cycle", 0.0))

    if pix_rad is None:
        cand_pix = candidate.get("pixsize_rad", candidate.get("pixscale_rad"))
        pix_rad = float(cand_pix) if cand_pix is not None else ducc_wcs.arcsec_to_rad(20.0)

    ra_c = candidate.get("ra_rad")
    dec_c = candidate.get("dec_rad")
    if ra_c is None or dec_c is None or not np.isfinite(float(ra_c)):
        if wcs_full is not None:
            try:
                sky = wcs_full.pixel_to_world(x, y)
                ra_c = float(sky.ra.rad)
                dec_c = float(sky.dec.rad)
            except Exception:
                ra_c = float(ra0_rad or 0.0)
                dec_c = float(dec0_rad or 0.0)
        else:
            ra_c = float(ra0_rad or 0.0)
            dec_c = float(dec0_rad or 0.0)
    else:
        ra_c = float(ra_c)
        dec_c = float(dec_c)

    ra_hms = candidate.get("ra_hms", "")
    dec_dms = candidate.get("dec_dms", "")
    if not ra_hms or not dec_dms:
        ra_hms, dec_dms = ducc_wcs.rad_to_hmsdms(ra_c, dec_c, dp=1)

    srcname = candidate.get("srcname", "")
    if not srcname or srcname == "—":
        srcname = ducc_wcs.rad_to_srcname(ra_c, dec_c)

    # Cutout WCS
    wcs_cut = ducc_wcs._build_tan_wcs_for_snippet(
        spatial_size=spatial_size,
        ra_rad=ra_c,
        dec_rad=dec_c,
        pixscale_rad=pix_rad,
        ra_sign=ra_sign,
        dec_sign=dec_sign,
        radesys=radesys,
        equinox=equinox,
    )

    # Fullframe WCS
    if wcs_full is None and npix_x is not None and npix_y is not None:
        wcs_full = ducc_wcs._build_fullframe_wcs(
            npix_x=npix_x,
            npix_y=npix_y,
            ra0_rad=ra0_rad if ra0_rad is not None else ra_c,
            dec0_rad=dec0_rad if dec0_rad is not None else dec_c,
            pixscale_rad=pix_rad,
            ra_sign=ra_sign,
            dec_sign=dec_sign,
            radesys=radesys,
            equinox=equinox,
        )

    # Prepare on-pulse and off-pulse cutout arrays
    has_cutout = False
    cutout_det = None
    if cutout is not None and isinstance(cutout, np.ndarray) and cutout.size > 0:
        cutout_det = _crop_or_pad_cutout(cutout, spatial_size)
        has_cutout = True
    elif "cutout" in candidate and candidate["cutout"] is not None:
        cand_cut = np.asarray(candidate["cutout"])
        if cand_cut.size > 0:
            cutout_det = _crop_or_pad_cutout(cand_cut, spatial_size)
            has_cutout = True
    elif candidate.get("cutout_fits") and os.path.exists(str(candidate["cutout_fits"])):
        try:
            with fits.open(str(candidate["cutout_fits"]), memmap=False) as hdul:
                loaded_cut = np.squeeze(hdul[0].data)
            if loaded_cut.size > 0:
                cutout_det = _crop_or_pad_cutout(loaded_cut, spatial_size)
                has_cutout = True
        except Exception as e:
            logger.debug(f"Could not load cutout_fits '{candidate.get('cutout_fits')}': {e}")
    elif fullframe is not None:
        half_sp = spatial_size // 2
        y0_des = y - half_sp
        y1_des = y0_des + spatial_size
        x0_des = x - half_sp
        x1_des = x0_des + spatial_size
        y0_clip = max(0, y0_des)
        y1_clip = min(fullframe.shape[0], y1_des)
        x0_clip = max(0, x0_des)
        x1_clip = min(fullframe.shape[1], x1_des)
        cut = fullframe[y0_clip:y1_clip, x0_clip:x1_clip]
        cutout_det = np.pad(
            cut,
            ((y0_clip - y0_des, y1_des - y1_clip), (x0_clip - x0_des, x1_des - x1_clip)),
            mode="constant",
            constant_values=np.nan,
        ).astype(np.float32)
        has_cutout = True

    cutout_off_det = None
    if cutout_off is not None and isinstance(cutout_off, np.ndarray) and cutout_off.size > 0:
        cutout_off_det = _crop_or_pad_cutout(cutout_off, spatial_size)
    elif "cutout_off" in candidate and candidate["cutout_off"] is not None:
        cand_off = np.asarray(candidate["cutout_off"])
        if cand_off.size > 0:
            cutout_off_det = _crop_or_pad_cutout(cand_off, spatial_size)
    elif candidate.get("cutout_off_fits") and os.path.exists(str(candidate["cutout_off_fits"])):
        try:
            with fits.open(str(candidate["cutout_off_fits"]), memmap=False) as hdul:
                loaded_off = np.squeeze(hdul[0].data)
            if loaded_off.size > 0:
                cutout_off_det = _crop_or_pad_cutout(loaded_off, spatial_size)
        except Exception as e:
            logger.debug(f"Could not load cutout_off_fits: {e}")

    # Prepare 1D rotational pulse profile
    has_profile = False
    p_raw = None
    if profile is not None and len(profile) > 0:
        p_raw = np.asarray(profile, dtype=np.float64)
        has_profile = True
    elif "profile" in candidate and candidate["profile"] is not None and len(candidate["profile"]) > 0:
        p_raw = np.asarray(candidate["profile"], dtype=np.float64)
        has_profile = True
    elif candidate.get("profile_npz") and os.path.exists(str(candidate["profile_npz"])):
        try:
            with np.load(str(candidate["profile_npz"])) as nz:
                for k in ("profile", "p_raw", "folded_profile", "flux", "arr_0"):
                    if k in nz:
                        p_raw = np.asarray(nz[k], dtype=np.float64)
                        has_profile = True
                        break
        except Exception as e:
            logger.debug(f"Could not load profile_npz: {e}")
    elif candidate.get("lightcurve_npz") and os.path.exists(str(candidate["lightcurve_npz"])):
        try:
            with np.load(str(candidate["lightcurve_npz"])) as nz:
                for k in ("profile", "p_raw", "folded_profile", "flux", "arr_0"):
                    if k in nz:
                        p_raw = np.asarray(nz[k], dtype=np.float64)
                        has_profile = True
                        break
        except Exception as e:
            logger.debug(f"Could not load lightcurve_npz: {e}")

    if has_profile and p_raw is not None and len(p_raw) > 0:
        p_std = float(np.std(p_raw))
        p_med = float(np.median(p_raw))
        p_norm = (p_raw - p_med) / (p_std if p_std > 1e-12 else 1.0)
        nbins = len(p_norm)

        # Boxcar matched-filter response along phase
        box_filt = np.zeros(nbins, dtype=np.float64)
        for b in range(nbins):
            for k in range(width_bins):
                box_filt[b] += p_norm[(b + k) % nbins]
        box_filt /= np.sqrt(max(1, width_bins))

        p2_raw = np.concatenate([p_norm, p_norm])
        p2_box = np.concatenate([box_filt, box_filt])
        phase2 = np.linspace(0.0, 2.0, len(p2_raw), endpoint=False)
    else:
        p_norm = None
        p2_raw = None
        p2_box = None
        phase2 = None
        nbins = 0

    # Continuum Cutout
    cont_im, cont_wcs, cont_vmin, cont_vmax = None, None, None, None
    if continuum_dir and os.path.isdir(continuum_dir):
        meta_guess = parse_candidate_filename(out_prefix)
        beam_lbl = meta_guess.get("beam", "") if isinstance(meta_guess, dict) else candidate.get("beam", "")
        if not beam_lbl:
            beam_lbl = candidate.get("beam", "")
        fits_cands = sorted(glob.glob(os.path.join(continuum_dir, f"*{beam_lbl}*.fits"))) if beam_lbl else []
        if not fits_cands:
            fits_cands = sorted(glob.glob(os.path.join(continuum_dir, "*.fits")))
        if fits_cands:
            try:
                # Use latest selfcal / best calibrated image (sorted last)
                with fits.open(fits_cands[-1], memmap=False) as hdul:
                    cdata = np.squeeze(hdul[0].data)
                    chdr = hdul[0].header
                cw = WCS(chdr).celestial
                size_rad = spatial_size * float(pix_rad)
                pos = SkyCoord(ra=np.degrees(ra_c) * u.deg, dec=np.degrees(dec_c) * u.deg, frame="icrs")
                c2d = Cutout2D(data=cdata, position=pos, size=(size_rad * u.rad, size_rad * u.rad), wcs=cw, mode="partial", fill_value=np.nan)
                cont_im = c2d.data
                bunit = str(chdr.get("BUNIT", "")).upper()
                if "JY" in bunit and "MJY" not in bunit and "UJY" not in bunit:
                    cont_im = cont_im * 1000.0  # Convert Jy/beam to mJy/beam
                cont_wcs = c2d.wcs
                cont_vmin = float(np.nanpercentile(cont_im, 1.0))
                cont_vmax = float(np.nanpercentile(cont_im, 99.5))
            except Exception as e:
                logger.debug(f"Could not load continuum cutout: {e}")

    # Evaluate candidate morphology and pulse flux integrity metrics
    morph_res = None
    if cutout_det is not None:
        try:
            from fastducc.filters import evaluate_candidate_morphology_and_flux
            on_bins_list = None
            if has_profile and p_raw is not None:
                peak_b = int(candidate.get("best_b", np.argmax(p_raw)))
                on_bins_list = [(peak_b + offset) % len(p_raw) for offset in range(width_bins)]
            morph_res = evaluate_candidate_morphology_and_flux(
                cutout=cutout_det,
                cutout_off=cutout_off_det,
                profile=p_raw if has_profile else None,
                on_bins=on_bins_list,
            )
        except Exception as e:
            logger.debug(f"Could not compute morphology and flux metrics: {e}")

    # Build GridSpec figure: 4 columns if off-pulse cutout is available, else 3
    base_stem = os.path.basename(out_prefix)
    has_off = cutout_off_det is not None
    out_fig = ""
    if out_dir_plots:
        os.makedirs(out_dir_plots, exist_ok=True)
        out_fig = os.path.join(out_dir_plots, f"{base_stem}_summary.pdf")
        ncols = 4 if has_off else 3
        figsize = (22, 11) if has_off else (18, 12)
    
        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = GridSpec(
            nrows=2, ncols=ncols, figure=fig,
            height_ratios=[1.8, 1.0], width_ratios=[1.0] * ncols,
        )
    
        # Panel (0, 0): Continuum or Reference Field Image
        if cont_wcs is not None:
            ax_cont = fig.add_subplot(gs[0, 0], projection=cont_wcs)
            im_cont = ax_cont.imshow(cont_im, origin="lower", cmap=cmap, vmin=cont_vmin, vmax=cont_vmax)
            ax_cont.coords.grid(True, color="white", alpha=0.35, ls=":")
            ax_cont.set_xlabel("RA (J2000)")
            ax_cont.set_ylabel("Dec (J2000)")
            ax_cont.set_aspect("equal", adjustable="box")
            ax_cont.plot([np.degrees(ra_c)], [np.degrees(dec_c)],
                         marker=reticle(which="rt"), ms=39, mec="black", mew=3.0, mfc="none",
                         transform=ax_cont.get_transform("world"))
            ax_cont.plot([np.degrees(ra_c)], [np.degrees(dec_c)],
                         marker=reticle(which="rt"), ms=36, mec="white", mew=1.5, mfc="none",
                         transform=ax_cont.get_transform("world"))
            cb0 = fig.colorbar(im_cont, ax=ax_cont, orientation="horizontal", fraction=0.046, pad=0.10)
            cb0.set_label("Flux density (mJy/beam)")
            ax_cont.set_title("Continuum")
        elif fullframe is not None and wcs_full is not None:
            ax_cont = fig.add_subplot(gs[0, 0], projection=wcs_full)
            vmin_ff = float(np.nanpercentile(fullframe, 1.0))
            vmax_ff = float(np.nanpercentile(fullframe, 99.8))
            im_cont = ax_cont.imshow(fullframe, origin="lower", cmap=cmap, vmin=vmin_ff, vmax=vmax_ff)
            ax_cont.coords.grid(True, color="white", alpha=0.35, ls=":")
            ax_cont.set_xlabel("RA (J2000)")
            ax_cont.set_ylabel("Dec (J2000)")
            ax_cont.set_aspect("equal", adjustable="box")
            ax_cont.plot([x], [y], marker=reticle(which="rt"), ms=39, mec="black", mew=3.0, mfc="none")
            ax_cont.plot([x], [y], marker=reticle(which="rt"), ms=36, mec="white", mew=1.5, mfc="none")
            cb0 = fig.colorbar(im_cont, ax=ax_cont, orientation="horizontal", fraction=0.046, pad=0.10)
            cb0.set_label("Folded Dirty Intensity")
            ax_cont.set_title("Full Field (no cont.)")
        else:
            ax_cont = fig.add_subplot(gs[0, 0])
            ax_cont.set_facecolor("#222222")
            ax_cont.text(
                0.5, 0.5, "NO CONTINUUM REFERENCE\nAVAILABLE",
                ha="center", va="center", color="silver", weight="bold", fontsize=11,
                transform=ax_cont.transAxes,
            )
            ax_cont.set_title("Continuum (Unavailable)")
            ax_cont.set_xticks([])
            ax_cont.set_yticks([])
    
        # Panel (0, 1): Full Field of View Detection Frame @ Trial Period
        if fullframe is not None and wcs_full is not None:
            ax_full = fig.add_subplot(gs[0, 1], projection=wcs_full)
            vmin_ff = float(np.nanpercentile(fullframe, 1.0))
            vmax_ff = float(np.nanpercentile(fullframe, 99.8))
            im_full = ax_full.imshow(fullframe, origin="lower", cmap=cmap, vmin=vmin_ff, vmax=vmax_ff)
            ax_full.coords.grid(True, color="white", alpha=0.35, ls=":")
            ax_full.set_xlabel("RA (J2000)")
            ax_full.set_ylabel("Dec (J2000)")
            ax_full.set_aspect("equal", adjustable="box")
            ax_full.plot([x], [y], marker=reticle(which="rt"), ms=39, mec="black", mew=3.0, mfc="none")
            ax_full.plot([x], [y], marker=reticle(which="rt"), ms=36, mec="white", mew=1.5, mfc="none")
            cb1 = fig.colorbar(im_full, ax=ax_full, orientation="horizontal", fraction=0.046, pad=0.10)
            cb1.set_label("Folded Dirty Intensity")
            ax_full.set_title(f"Field @ P={period_s*1e3:.2f} ms ({f_hz:.4f} Hz, DM={dm_val:.2f})")
        else:
            ax_full = fig.add_subplot(gs[0, 1])
            ax_full.set_facecolor("#222222")
            ax_full.text(
                0.5, 0.5, "NO FULL-FRAME IMAGE ARCHIVED\n(save_fullframes=False or omitted)",
                ha="center", va="center", color="silver", weight="bold", fontsize=11,
                transform=ax_full.transAxes,
            )
            ax_full.set_title(f"Detection Map @ P={period_s*1e3:.2f} ms")
            ax_full.set_xticks([])
            ax_full.set_yticks([])
    
        # Panel (0, 2): Zoomed On-Pulse Cutout
        if has_cutout and cutout_det is not None:
            ax_cut = fig.add_subplot(gs[0, 2], projection=wcs_cut)
            vmin_c = float(np.nanpercentile(cutout_det, 1.0))
            vmax_c = float(np.nanpercentile(cutout_det, 99.8))
            im_cut = ax_cut.imshow(cutout_det, origin="lower", cmap=cmap, vmin=vmin_c, vmax=vmax_c)
            ax_cut.coords.grid(True, color="white", alpha=0.35, ls=":")
            ax_cut.set_xlabel("RA (J2000)")
            ax_cut.set_ylabel("Dec (J2000)")
            ax_cut.set_aspect("equal", adjustable="box")
            c_x = cutout_det.shape[1] // 2
            c_y = cutout_det.shape[0] // 2
            ax_cut.plot([c_x], [c_y], marker=reticle(which="rt"), ms=39, mec="black", mew=3.0, mfc="none")
            ax_cut.plot([c_x], [c_y], marker=reticle(which="rt"), ms=36, mec="white", mew=1.5, mfc="none")
            cb2 = fig.colorbar(im_cut, ax=ax_cut, orientation="horizontal", fraction=0.046, pad=0.10)
            cb2.set_label("Folded Dirty Intensity")
            ax_cut.set_title(f"On-Pulse Cutout (S/N={snr_val:.2f})")
        else:
            ax_cut = fig.add_subplot(gs[0, 2])
            ax_cut.set_facecolor("#222222")
            ax_cut.text(
                0.5, 0.5, "NO CUTOUT DATA AVAILABLE",
                ha="center", va="center", color="silver", weight="bold", fontsize=12,
                transform=ax_cut.transAxes,
            )
            ax_cut.set_title(f"On-Pulse Cutout (S/N={snr_val:.2f})")
            ax_cut.set_xticks([])
            ax_cut.set_yticks([])
    
        # Panel (0, 3): Zoomed Off-Pulse Cutout (if available)
        if has_off and cutout_off_det is not None:
            ax_cut_off = fig.add_subplot(gs[0, 3], projection=wcs_cut)
            vmin_off = float(np.nanpercentile(cutout_off_det, 1.0)) if not has_cutout else vmin_c
            vmax_off = float(np.nanpercentile(cutout_off_det, 99.8)) if not has_cutout else vmax_c
            im_cut_off = ax_cut_off.imshow(cutout_off_det, origin="lower", cmap=cmap, vmin=vmin_off, vmax=vmax_off)
            ax_cut_off.coords.grid(True, color="white", alpha=0.35, ls=":")
            ax_cut_off.set_xlabel("RA (J2000)")
            ax_cut_off.set_ylabel("Dec (J2000)")
            ax_cut_off.set_aspect("equal", adjustable="box")
            c_x = cutout_off_det.shape[1] // 2
            c_y = cutout_off_det.shape[0] // 2
            ax_cut_off.plot([c_x], [c_y], marker=reticle(which="rt"), ms=39, mec="black", mew=3.0, mfc="none")
            ax_cut_off.plot([c_x], [c_y], marker=reticle(which="rt"), ms=36, mec="white", mew=1.5, mfc="none")
            cb3 = fig.colorbar(im_cut_off, ax=ax_cut_off, orientation="horizontal", fraction=0.046, pad=0.10)
            cb3.set_label("Folded Dirty Intensity")
            ax_cut_off.set_title("Off-Pulse Cutout")
    
        # Bottom Row Left: Folded Pulse Profile (2 Full Cycles)
        ax_lc = fig.add_subplot(gs[1, 0:ncols - 1])
        if has_profile and p2_raw is not None and p2_box is not None and phase2 is not None:
            ax_lc.step(phase2, p2_raw, where="mid", color="tab:blue", lw=1.5, label="Raw Folded Profile", zorder=2)
            ax_lc.plot(phase2, p2_box, color="tab:green", lw=1.8, label=f"Boxcar Matched Filter (w={width_bins} bins, duty={duty_cycle*100.0:.1f}%)", zorder=3)
            peak_b = int(candidate.get("best_b", np.argmax(p_norm) if p_norm is not None else 0))
            peak_phase1 = peak_b / float(max(1, nbins))
            peak_phase2 = peak_phase1 + 1.0
            ax_lc.axvline(peak_phase1, color="crimson", ls="--", lw=1.2, label=f"Peak Phase ({peak_phase1:.3f})", zorder=4)
            ax_lc.axvline(peak_phase2, color="crimson", ls="--", lw=1.2, zorder=4)
            # Highlight on-pulse phase intervals
            w_frac = width_bins / float(max(1, nbins))
            ax_lc.axvspan(peak_phase1, peak_phase1 + w_frac, color="tab:green", alpha=0.15, label="On-Pulse Window")
            ax_lc.axvspan(peak_phase2, peak_phase2 + w_frac, color="tab:green", alpha=0.15)
            ax_lc.axvline(1.0, color="gray", ls=":", lw=1.2, alpha=0.7)
            ax_lc.set_xlabel("Rotational Phase [0..2]")
            ax_lc.set_ylabel("Normalized Flux / S/N")
            ax_lc.set_title(f"Folded Pulse Profile | P = {period_s*1000.0:.3f} ms (f = {f_hz:.4f} Hz) | S/N = {snr_val:.2f}")
            ax_lc.grid(True, alpha=0.3)
            ax_lc.legend(loc="upper right", framealpha=0.9)
        else:
            ax_lc.set_facecolor("#222222")
            ax_lc.text(
                0.5, 0.5, "NO PROFILE DATA AVAILABLE",
                ha="center", va="center", color="silver", weight="bold", fontsize=14,
                transform=ax_lc.transAxes,
            )
            ax_lc.set_title(f"Folded Pulse Profile | P = {period_s*1000.0:.3f} ms (f = {f_hz:.4f} Hz) | S/N = {snr_val:.2f}")
            ax_lc.set_xticks([])
            ax_lc.set_yticks([])
        ax_lc.axvline(1.0, color="gray", ls=":", lw=1.2, alpha=0.7)
        ax_lc.set_xlabel("Rotational Phase [0..2]")
        ax_lc.set_ylabel("Normalized Flux / S/N")
        ax_lc.set_title(f"Folded Pulse Profile | P = {period_s*1000.0:.3f} ms (f = {f_hz:.4f} Hz) | S/N = {snr_val:.2f}")
        ax_lc.grid(True, alpha=0.3)
        ax_lc.legend(loc="upper right", framealpha=0.9)
    
        # Bottom Row Right: FastDUCC-style AnchoredText Summary Box
        ax_txt = fig.add_subplot(gs[1, ncols - 1])
        ax_txt.set_axis_off()
    
        meta = parse_candidate_filename(out_prefix)
        field = candidate.get("phase_center_field", meta.get("field", "") if isinstance(meta, dict) else "")
        sbid = candidate.get("sbid", meta.get("sbid", "") if isinstance(meta, dict) else "")
        beam = candidate.get("beam", meta.get("beam", "") if isinstance(meta, dict) else "")
        scan = candidate.get("scan", candidate.get("scan_id", meta.get("scan_id", "") if isinstance(meta, dict) else ""))
    
        psrcat_match = candidate.get("psrcat_match_name", candidate.get("psrcat_name", ""))
        psrcat_sep = candidate.get("psrcat_sep_arcsec")
        psrcat_str = f"{psrcat_match} ({float(psrcat_sep):.1f}\")" if psrcat_match and psrcat_sep is not None else (psrcat_match or "None")
    
        racs_match = candidate.get("racs_match_name", candidate.get("racs_name", ""))
        racs_sep = candidate.get("racs_sep_arcsec")
        racs_str = f"{racs_match} ({float(racs_sep):.1f}\")" if racs_match and racs_sep is not None else (racs_match or "None")

        # Morphology & Flux Quality Status
        if morph_res is not None:
            status_pt = "Pass" if morph_res["is_point_like"] else "FAIL (Extended)"
            status_flx = "Pass" if morph_res["is_valid_flux"] else "FAIL (Neg Trough)"
            morph_str = (
                f"\nMorphology: {status_pt}\n"
                f"  HP Lag-1: {morph_res['hp_lag1']:.3f} | Sharp: {morph_res['sharpness']:.2f}\n"
                f"Flux Quality: {status_flx}\n"
                f"  On-Share: {morph_res['on_flux_share']:.2f} | On-SNR: {morph_res['on_snr']:.1f}"
            )
        else:
            morph_str = ""

        txt = (
            f"Name: {srcname}\n"
            f"Field: {field}\n"
            f"S/N: {snr_val:.2f}\n"
            f"DM: {dm_val:.2f} pc/cm³\n"
            f"Period: {period_s:.6f} s ({period_s * 1000.0:.3f} ms)\n"
            f"Spin Freq: {f_hz:.6f} Hz\n"
            f"FFA Width: {width_bins} bins\n"
            f"Duty Cycle: {duty_cycle * 100.0:.1f}%\n"
            f"SBID: {sbid}\n"
            f"Beam: {beam}\n"
            f"Scan: {scan}\n"
            f"Coord (J2000): {ra_hms} {dec_dms}\n"
            f"Coord (x,y): ({x:d}, {y:d})\n"
            f"Method: {method}\n"
            f"PSRCAT: {psrcat_str}\n"
            f"RACS: {racs_str}"
            f"{morph_str}"
        )
    
        box = AnchoredText(txt, loc="upper left", frameon=True, pad=1.0, prop={"size": 10})
        ax_txt.add_artist(box)
        ax_txt.set_title("Summary", pad=12)
    
        # Footer
        gen_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        host = socket.gethostname()
        username = getpass.getuser()
        git_hash = "—"
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stderr=subprocess.DEVNULL,
            ).decode("utf-8").strip()
        except Exception:
            pass
        meta_footer = f"Generated: {gen_time} | Host: {host} | User: {username} | Git: {git_hash}"
        fig.text(0.5, 0.01, meta_footer, ha="center", va="bottom", fontsize=8, color="gray", alpha=0.7)
    
        fig.savefig(out_fig, bbox_inches="tight")
        plt.close(fig)

    # Save Cutout FITS with WCS & metadata if available
    out_fits = ""
    if has_cutout and cutout_det is not None and wcs_cut is not None:
        if out_dir_cutouts:
            os.makedirs(out_dir_cutouts, exist_ok=True)
            out_fits = os.path.join(out_dir_cutouts, f"{base_stem}_cutout.fits")
        else:
            out_fits = f"{out_prefix}_cutout.fits"
        hdr = wcs_cut.to_header()
        hdr["BUNIT"] = "JY/BEAM"
        hdr["COMMENT"] = "Detection cutout frame from periodicity search"
        if ra_hms and dec_dms:
            hdr["OBJRA"] = str(ra_hms)
            hdr["OBJDEC"] = str(dec_dms)
        hdr["SRCNAME"] = str(srcname)
        hdr["SNR"] = float(snr_val)
        hdr["PERIOD"] = float(period_s)
        hdr["FREQ_HZ"] = float(f_hz)
        hdr["DM"] = float(dm_val)
        hdr["WIDTH"] = int(width_bins)
        hdr["CDELT1"] = float(wcs_cut.wcs.cdelt[0])
        hdr["CDELT2"] = float(wcs_cut.wcs.cdelt[1])
        hdr["PC1_1"] = 1.0
        hdr["PC1_2"] = 0.0
        hdr["PC2_1"] = 0.0
        hdr["PC2_2"] = 1.0
        if getattr(wcs_cut.wcs, "radesys", None):
            hdr["RADESYS"] = str(wcs_cut.wcs.radesys)
        fits.writeto(out_fits, cutout_det.astype(np.float32), header=hdr, overwrite=True)

    out_fits_off = ""
    if has_cutout and cutout_off_det is not None and wcs_cut is not None:
        if out_dir_cutouts:
            os.makedirs(out_dir_cutouts, exist_ok=True)
            out_fits_off = os.path.join(out_dir_cutouts, f"{base_stem}_cutout_off.fits")
        else:
            out_fits_off = f"{out_prefix}_cutout_off.fits"
        fits.writeto(out_fits_off, cutout_off_det.astype(np.float32), header=hdr, overwrite=True)

    # Save Fullframe FITS if available
    out_fits_full = ""
    if fullframe is not None and wcs_full is not None:
        if out_dir_fullframes:
            os.makedirs(out_dir_fullframes, exist_ok=True)
            out_fits_full = os.path.join(out_dir_fullframes, f"{base_stem}_fullframe.fits")
        else:
            out_fits_full = f"{out_prefix}_fullframe.fits"
        hdr_full = wcs_full.to_header()
        hdr_full["BUNIT"] = "JY/BEAM"
        hdr_full["COMMENT"] = "Full detection frame from periodicity imaging"
        if ra_hms and dec_dms:
            hdr_full["OBJRA"] = str(ra_hms)
            hdr_full["OBJDEC"] = str(dec_dms)
        hdr_full["SRCNAME"] = str(srcname)
        hdr_full["SNR"] = float(snr_val)
        hdr_full["PERIOD"] = float(period_s)
        hdr_full["FREQ_HZ"] = float(f_hz)
        hdr_full["DM"] = float(dm_val)
        fits.writeto(out_fits_full, fullframe.astype(np.float32), header=hdr_full, overwrite=True)

    # Save 1D Folded Pulse Profile NPZ if available
    out_npz = ""
    if has_profile and p_norm is not None:
        if out_dir_profiles:
            os.makedirs(out_dir_profiles, exist_ok=True)
            out_npz = os.path.join(out_dir_profiles, f"{base_stem}_profile.npz")
        else:
            out_npz = f"{out_prefix}_profile.npz"
        np.savez(
            out_npz,
            profile=p_norm.astype(np.float32),
            period_s=float(period_s),
            f_hz=float(f_hz),
            dm=float(dm_val),
            snr=float(snr_val),
            srcname=str(srcname),
            width_bins=int(width_bins),
            duty_cycle=float(duty_cycle),
        )

    return {
        "figure": out_fig,
        "summary_plot": out_fig,
        "fits_cutout": out_fits,
        "cutout_fits": out_fits,
        "cutout_off_fits": out_fits_off,
        "fits_fullframe": out_fits_full,
        "fullframe_fits": out_fits_full,
        "profile_npz": out_npz,
        "lightcurve_npz": out_npz,
    }


def update_candidate_summary_with_dynspec(
    summary_plot_path: Union[str, Path],
    ds_path: Union[str, Path],
    out_path: Optional[Union[str, Path]] = None,
    candidate_info: Optional[Dict[str, Any]] = None,
    dpi: int = 200,
    cmap: str = "viridis",
) -> Path:
    """
    Update an existing transient candidate summary plot to incorporate a dynamic spectrum waterfall.

    Locates the candidate's existing cutout FITS, fullframe FITS, lightcurve NPZ, and uses
    the provided .ds file to re-render the comprehensive 3-row diagnostic summary plot
    (Continuum | Full Frame | Cutout // Dynamic Spectrum Waterfall // Lightcurve // Metadata).

    Parameters
    ----------
    summary_plot_path : str or Path
        Path to the existing summary PDF (e.g. *cand_*_summary.pdf).
    ds_path : str or Path
        Path to the .ds dynamic spectrum file.
    out_path : str or Path, optional
        Destination path for the updated plot. If omitted, overwrites summary_plot_path.
    candidate_info : dict, optional
        Dictionary of candidate metadata (e.g. snr, dm, sbid, beam, scan, field, srcname).
    dpi : int, default 200
        Figure DPI resolution.
    cmap : str, default "viridis"
        Colormap name.

    Returns
    -------
    Path
        Path to the updated summary PDF file.
    """
    from fastducc.dynspec import load_dynamic_spectrum_hdf5

    summary_plot_path = Path(summary_plot_path).resolve()
    ds_path = Path(ds_path).resolve()
    if not ds_path.is_file():
        raise FileNotFoundError(f"Dynamic spectrum file not found: {ds_path}")

    if out_path is None:
        out_path = summary_plot_path
    else:
        out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load dynamic spectrum
    ds_dict = load_dynamic_spectrum_hdf5(ds_path)
    ds_raw_data = np.asarray(ds_dict["data"], dtype=np.float32)
    ds_freqs = np.asarray(ds_dict["freq"], dtype=np.float64)
    ds_raw_times = ds_dict.get("time_raw")

    # 2. Discover candidate artefacts (cutout, fullframe, lightcurve)
    pdir = summary_plot_path.parent
    base_stem = summary_plot_path.name.replace("_summary.pdf", "").replace("_summary.png", "")

    # Parse metadata from filename using parse_candidate_filename and regex
    meta = parse_candidate_filename(summary_plot_path.name)
    field = meta.get("field", "") if isinstance(meta, dict) else ""
    sbid = meta.get("sbid", "") if isinstance(meta, dict) else ""
    beam = meta.get("beam", "") if isinstance(meta, dict) else ""
    scan = meta.get("scan_id", "") if isinstance(meta, dict) else ""
    meta_dm = float(meta.get("dm", 0.0)) if isinstance(meta, dict) and meta.get("dm") else 0.0

    # Parse candidate identifiers from filename
    cand_name_match = re.search(r'(cand_[^._]+)', base_stem)
    cand_sub = cand_name_match.group(1) if cand_name_match else base_stem
    cand_id_clean = cand_sub.replace("cand_", "")
    chunk_match = re.search(r'(chunk_\d+)', base_stem)
    chunk_tag = chunk_match.group(1) if chunk_match else ""
    scan_match = re.search(r'(scan_[^._]+)', base_stem)
    if scan_match and not scan:
        scan = scan_match.group(1).replace("scan_", "")
    width_match = re.search(r'_w(\d+)', base_stem)
    width_str = width_match.group(1) if width_match else "—"

    # Method
    if "_var_" in base_stem:
        method = "variance"
    elif "_boxcar_" in base_stem:
        method = "boxcar"
    elif "_periodicity_" in base_stem:
        method = "periodicity"
    else:
        method = meta.get("kind", "transient") if isinstance(meta, dict) else "transient"

    # MJD / Detection time
    mjd_match = re.search(r'MJD(\d+\.\d+)', base_stem)
    if mjd_match:
        mjd_val = float(mjd_match.group(1))
        det_time = mjd_val * 86400.0
        det_time_str = f"{mjd_val:.7f} MJD"
        imjd = str(int(mjd_val))
    else:
        det_time = np.nan
        det_time_str = "—"
        imjd = "—"

    # Targeted lookups for auxiliary files without slow broad directory scans
    cutout_path = None
    for cdir in [pdir / ".." / "cutouts", pdir / "cutouts", pdir]:
        if cdir.is_dir():
            cands = list(cdir.glob(f"*{base_stem}*cutout.fits")) or list(cdir.glob(f"*{cand_sub}*cutout.fits"))
            if cands:
                cutout_path = cands[0]
                break

    fullframe_path = None
    for fdir in [pdir / ".." / "fullframes", pdir / "fullframes", pdir]:
        if fdir.is_dir():
            cands = list(fdir.glob(f"*{base_stem}*fullframe.fits")) or list(fdir.glob(f"*{cand_sub}*fullframe.fits"))
            if cands:
                fullframe_path = cands[0]
                break

    # Look for lightcurve files (both raw/full and boxcar smoothed)
    lc_path = None
    lc_sm_path = None
    w_int = int(width_str) if (width_str and width_str.isdigit()) else 1

    lc_patterns = []
    if scan and chunk_tag and width_match:
        lc_patterns.append(f"*{scan}*{chunk_tag}*{cand_sub}*{width_match.group(0)}*lc*.npz")
    if chunk_tag:
        lc_patterns.append(f"*{chunk_tag}*{cand_sub}*lc*.npz")
    lc_patterns.extend([
        f"*{base_stem}*lc_full.npz",
        f"*{base_stem}*lc*.npz",
        f"*{cand_sub}*lc_full.npz",
    ])
    for ldir in [pdir / ".." / "lightcurves", pdir / "lightcurves", pdir]:
        if ldir.is_dir():
            for pat in lc_patterns:
                cands = list(ldir.glob(pat))
                if cands:
                    # Prefer lc_full for raw if present
                    cands_full = [c for c in cands if "_lc_full.npz" in c.name]
                    lc_path = cands_full[0] if cands_full else cands[0]
                    break
            if lc_path:
                break

    # Look for precomputed boxcar smoothed lightcurve matching width W
    if w_int > 1:
        sm_patterns = [
            f"*{chunk_tag}*{cand_sub}*w{w_int}*boxcar*w{w_int}.npz",
            f"*{cand_sub}*w{w_int}*boxcar*w{w_int}.npz",
            f"*{chunk_tag}*{cand_sub}*boxcar*w{w_int}.npz",
            f"*w{w_int}*boxcar*w{w_int}.npz",
        ]
        for ldir in [pdir / ".." / "lightcurves", pdir / "lightcurves", pdir]:
            if ldir.is_dir():
                for pat in sm_patterns:
                    cands = list(ldir.glob(pat))
                    if cands:
                        lc_sm_path = cands[0]
                        break
                if lc_sm_path:
                    break

    # Look for continuum image in continuum_images directory (under SBID or parent tree)
    cont_path = None
    beam_tag = beam if (beam and "beam" in beam) else f"beam{beam}" if beam else ""
    cont_search_dirs = [
        pdir / ".." / ".." / ".." / "continuum_images",
        pdir / ".." / ".." / ".." / "cont_combined",
        pdir / ".." / ".." / "continuum_images",
        pdir / ".." / "continuum_images",
    ]
    if sbid:
        cont_search_dirs.extend([
            Path("/fred/oz451/azic/data") / sbid / "continuum_images",
            Path("/fred/oz451/azic/data") / sbid / "cont_combined",
            Path("/fred/oz451/azic/data") / f"SB{sbid}" / "continuum_images",
        ])
    for cdir in cont_search_dirs:
        try:
            cdir_res = cdir.resolve()
            if cdir_res.is_dir():
                matches = []
                if beam_tag:
                    matches = sorted(list(cdir_res.glob(f"*{beam_tag}*img-MFS-image.fits"))) or sorted(list(cdir_res.glob(f"*{beam_tag}*.fits")))
                if not matches:
                    matches = sorted(list(cdir_res.glob("*img-MFS-image.fits")))
                if matches:
                    cont_path = matches[0]
                    break
        except Exception:
            pass

    # Look for candidate table to extract image pixel coordinates (x, y) if present
    cand_px_x = None
    cand_px_y = None
    cand_table_dirs = [pdir / ".." / "catalogs", pdir / "..", pdir, pdir / "catalogs"]
    for tdir in cand_table_dirs:
        try:
            if tdir.resolve().is_dir():
                csvs = list(tdir.resolve().glob(f"*{cand_sub}*candidate_list.csv")) or list(tdir.resolve().glob("*candidate_list.csv"))
                if csvs:
                    import pandas as pd
                    df_c = pd.read_csv(csvs[0])
                    if "x" in df_c.columns and "y" in df_c.columns:
                        for _, row_c in df_c.iterrows():
                            val_x = row_c["x"]
                            val_y = row_c["y"]
                            if np.isfinite(val_x) and np.isfinite(val_y) and val_x < 900000 and val_y < 900000:
                                cand_px_x = float(val_x)
                                cand_px_y = float(val_y)
                                break
                    if cand_px_x is not None:
                        break
        except Exception:
            pass

    # Read cutout FITS header if available for metadata
    cut_hdr = {}
    if cutout_path and cutout_path.is_file():
        try:
            with fits.open(cutout_path) as hdul:
                cut_hdr = dict(hdul[0].header)
        except Exception:
            pass

    # Ingest candidate_info overrides if provided
    cand_info = candidate_info or {}
    srcname = cand_info.get("srcname") or cand_id_clean
    if cand_info.get("field"):
        field = cand_info["field"]
    if cand_info.get("sbid"):
        sbid = cand_info["sbid"]
    if cand_info.get("beam"):
        beam = cand_info["beam"]
    if cand_info.get("scan"):
        scan = cand_info["scan"]
    if cand_info.get("method"):
        method = cand_info["method"]
    if cand_info.get("width"):
        width_str = str(cand_info["width"])

    cand_dm = float(cand_info.get("dm", 0.0) or ds_dict.get("attrs", {}).get("dm", 0.0) or meta_dm)
    dm_str = f"{cand_dm:.2f} pc/cm³"

    # SNR resolution: check candidate_info, then cutout header
    snr_val = cand_info.get("snr", np.nan)
    if not np.isfinite(snr_val):
        for k in ("SNR", "MAX_SNR", "DET_SNR"):
            if k in cut_hdr:
                try:
                    snr_val = float(cut_hdr[k])
                    break
                except Exception:
                    pass
    snr_str = f"{float(snr_val):.2f}" if np.isfinite(snr_val) else "—"

    # Coordinates
    ra_str = cand_info.get("ra_hms") or cut_hdr.get("OBJRA") or ds_dict.get("ra_str", "")
    dec_str = cand_info.get("dec_dms") or cut_hdr.get("OBJDEC") or ds_dict.get("dec_str", "")
    telescope = ds_dict.get("telescope", "ASKAP")

    # Load lightcurve for the detection chunk if available
    lc_time, lc_flux = None, None
    if lc_path is not None and lc_path.is_file():
        try:
            with np.load(lc_path) as z:
                lc_time = np.array(z["time"], dtype=np.float64)
                lc_flux = np.array(z["flux"], dtype=np.float64)
        except Exception as e:
            logger.debug(f"Could not load lightcurve NPZ {lc_path}: {e}")

    # Load precomputed smoothed lightcurve or compute boxcar smoothing
    lc_sm_time, lc_sm_flux = None, None
    if lc_sm_path is not None and lc_sm_path.is_file():
        try:
            with np.load(lc_sm_path) as z:
                lc_sm_time = np.array(z["time"], dtype=np.float64)
                lc_sm_flux = np.array(z["flux"], dtype=np.float64)
        except Exception as e:
            logger.debug(f"Could not load smoothed lightcurve NPZ {lc_sm_path}: {e}")

    # Fallback: slice a detection chunk (~512 timestamps) from the full-obs dynamic spectrum
    if lc_time is None or lc_flux is None:
        if np.isfinite(det_time) and ds_raw_times is not None:
            det_idx = int(np.searchsorted(ds_raw_times, det_time))
            half_chunk = 256
            idx0 = max(0, det_idx - half_chunk)
            idx1 = min(len(ds_raw_times), det_idx + half_chunk)
            lc_time = ds_raw_times[idx0:idx1]
            lc_flux = np.nanmean(ds_raw_data[idx0:idx1, :], axis=1) * 1000.0  # Jy -> mJy
        elif ds_raw_times is not None and len(ds_raw_times) > 512:
            lc_time = ds_raw_times[:512]
            lc_flux = np.nanmean(ds_raw_data[:512, :], axis=1) * 1000.0
        else:
            lc_time = np.array(ds_dict["time"], dtype=np.float64)
            lc_flux = np.nanmean(ds_raw_data, axis=1) * 1000.0

    # If smoothed lightcurve wasn't loaded from disk but w_int > 1, compute boxcar smoothing
    if (lc_sm_flux is None or len(lc_sm_flux) == 0) and w_int > 1 and len(lc_flux) >= w_int:
        try:
            from fastducc import kernels
            offset = w_int // 2
            sm_arr, T_eff = kernels._compute_boxcar_1d(lc_flux, w_int)
            if T_eff > 0:
                lc_sm_flux = sm_arr
                lc_sm_time = lc_time[offset:offset + T_eff]
        except Exception as e:
            logger.debug(f"Could not compute boxcar smoothing on-the-fly: {e}")

    # 3. Pull detection chunk dynamic spectrum slice out of the full-obs dynamic spectrum
    t_start = lc_time[0] if len(lc_time) > 0 else 0.0
    t_end = lc_time[-1] if len(lc_time) > 0 else 1.0
    rel_times_start = lc_time - t_start

    # CASA timerange
    if len(lc_time) > 0:
        try:
            t_start_obj = Time(t_start / 86400.0, format='mjd', scale='utc')
            t_end_obj = Time(t_end / 86400.0, format='mjd', scale='utc')
            casa_start = t_start_obj.iso.replace('-', '/').replace(' ', '/')
            casa_end = t_end_obj.iso.replace('-', '/').replace(' ', '/')
            casa_timerange = f"{casa_start}~{casa_end}"
        except Exception:
            casa_timerange = "—"
    else:
        casa_timerange = "—"

    if ds_raw_times is not None and len(ds_raw_times) == ds_raw_data.shape[0] and len(lc_time) > 0:
        idx0 = max(0, int(np.searchsorted(ds_raw_times, t_start)) - 1)
        idx1 = min(len(ds_raw_times), int(np.searchsorted(ds_raw_times, t_end)) + 2)
        if idx1 > idx0:
            ds_slice = ds_raw_data[idx0:idx1, :]
            ds_t_slice = ds_raw_times[idx0:idx1] - t_start
        else:
            ds_slice = ds_raw_data
            ds_t_slice = ds_dict.get("time", np.arange(len(ds_slice), dtype=float))
    else:
        ds_slice = ds_raw_data
        ds_t_slice = ds_dict.get("time", np.arange(len(ds_slice), dtype=float))

    # Build 3-tier publication figure with dynamic spectrum & main lightcurve shorter by 20%
    fig = plt.figure(figsize=(18, 12.5), dpi=dpi)
    gs_main = GridSpec(
        nrows=3, ncols=3, figure=fig,
        height_ratios=[1.0, 0.96, 0.80], width_ratios=[1.0, 1.0, 0.95],
        hspace=0.28, wspace=0.20,
    )

    # Resolve sky position SkyCoord for reticles
    cand_pos = None
    if ra_str and dec_str and ra_str != "—" and dec_str != "—":
        try:
            cand_pos = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg), frame="icrs")
        except Exception:
            try:
                cand_pos = SkyCoord(float(ra_str) * u.deg, float(dec_str) * u.deg, frame="icrs")
            except Exception:
                pass

    # Panel (0, 0): Continuum / Reference Field
    cont_plotted = False
    if cont_path and cont_path.is_file():
        try:
            with fits.open(cont_path) as hdul:
                cont_raw = np.squeeze(hdul[0].data)
                cont_w = WCS(hdul[0].header).celestial
            if cand_pos is not None:
                from astropy.nddata import Cutout2D
                c2d = Cutout2D(
                    data=cont_raw,
                    position=cand_pos,
                    size=(50, 50),
                    wcs=cont_w,
                    mode="partial",
                    fill_value=np.nan,
                )
                ax_cont = fig.add_subplot(gs_main[0, 0], projection=c2d.wcs)
                c_v0, c_v1 = np.nanpercentile(c2d.data, [1.0, 99.8])
                im_cont = ax_cont.imshow(c2d.data, origin="lower", cmap=cmap, vmin=c_v0, vmax=c_v1)
                ax_cont.coords.grid(True, color="white", alpha=0.35, ls=":")
                ax_cont.set_xlabel("RA (J2000)")
                ax_cont.set_ylabel("Dec (J2000)")
                ax_cont.set_aspect("equal", adjustable="box")
                ax_cont.set_title("Continuum")

                # Reticle on continuum
                c_rx, c_ry = c2d.wcs.world_to_pixel(cand_pos)
                ax_cont.plot([c_rx], [c_ry], marker=reticle(which="rt"), ms=39, mec="black", mew=3.0, mfc="none")
                ax_cont.plot([c_rx], [c_ry], marker=reticle(which="rt"), ms=36, mec="white", mew=1.5, mfc="none")

                cb_c = fig.colorbar(im_cont, ax=ax_cont, orientation="horizontal", fraction=0.046, pad=0.10)
                cb_c.set_label("Flux density (Jy/beam)")
                cont_plotted = True
        except Exception as e:
            logger.debug(f"Could not load continuum cutout: {e}")

    if not cont_plotted:
        ax_cont = fig.add_subplot(gs_main[0, 0])
        ax_cont.set_facecolor("#1e1e1e")
        ax_cont.text(
            0.5, 0.5, "CONTINUUM\nN/A",
            ha="center", va="center", color="silver", weight="bold", fontsize=11,
            transform=ax_cont.transAxes,
        )
        ax_cont.set_title("Continuum / Reference")
        ax_cont.set_xticks([])
        ax_cont.set_yticks([])

    # Panel (0, 1): Full-Frame Detection Frame
    if fullframe_path and fullframe_path.is_file():
        try:
            with fits.open(fullframe_path) as hdul:
                ff_data = np.squeeze(hdul[0].data)
                ff_wcs = WCS(hdul[0].header).celestial
            ax_full = fig.add_subplot(gs_main[0, 1], projection=ff_wcs)
            v0, v1 = np.nanpercentile(ff_data, [1.0, 99.8])
            im_ff = ax_full.imshow(ff_data, origin="lower", cmap=cmap, vmin=v0, vmax=v1)
            ax_full.coords.grid(True, color="white", alpha=0.35, ls=":")
            ax_full.set_xlabel("RA (J2000)")
            ax_full.set_ylabel("Dec (J2000)")
            ax_full.set_aspect("equal", adjustable="box")
            ax_full.set_title("Detection Field of View")

            # Reticle on fullframe: candidate pixel coordinate (x, y)
            ff_rx, ff_ry = None, None
            if cand_px_x is not None and cand_px_y is not None:
                ff_rx, ff_ry = cand_px_x, cand_px_y
            elif cand_pos is not None:
                wx, wy = ff_wcs.world_to_pixel(cand_pos)
                # Check if fullframe WCS Dec is inverted relative to standard imaging convention
                ff_rx, ff_ry = wx, (ff_data.shape[0] - 1 - wy)

            if ff_rx is not None and ff_ry is not None:
                ax_full.plot([ff_rx], [ff_ry], marker=reticle(which="rt"), ms=39, mec="black", mew=3.0, mfc="none")
                ax_full.plot([ff_rx], [ff_ry], marker=reticle(which="rt"), ms=36, mec="white", mew=1.5, mfc="none")

            cb_ff = fig.colorbar(im_ff, ax=ax_full, orientation="horizontal", fraction=0.046, pad=0.10)
            cb_ff.set_label("Flux density (Jy/beam)")
        except Exception:
            ax_full = fig.add_subplot(gs_main[0, 1])
            ax_full.set_facecolor("#1e1e1e")
            ax_full.text(0.5, 0.5, "FULL FIELD N/A", ha="center", va="center", color="silver", transform=ax_full.transAxes)
            ax_full.set_xticks([]); ax_full.set_yticks([])
    else:
        ax_full = fig.add_subplot(gs_main[0, 1])
        ax_full.set_facecolor("#1e1e1e")
        ax_full.text(0.5, 0.5, "FULL FIELD N/A", ha="center", va="center", color="silver", transform=ax_full.transAxes)
        ax_full.set_xticks([]); ax_full.set_yticks([])

    # Panel (0, 2): Zoomed Cutout
    if cutout_path and cutout_path.is_file():
        try:
            with fits.open(cutout_path) as hdul:
                cut_data = np.squeeze(hdul[0].data)
                cut_wcs = WCS(hdul[0].header).celestial
            ax_cut = fig.add_subplot(gs_main[0, 2], projection=cut_wcs)
            v0, v1 = np.nanpercentile(cut_data, [1.0, 99.8])
            im_cut = ax_cut.imshow(cut_data, origin="lower", cmap=cmap, vmin=v0, vmax=v1)
            ax_cut.coords.grid(True, color="white", alpha=0.35, ls=":")
            ax_cut.set_xlabel("RA (J2000)")
            ax_cut.set_ylabel("Dec (J2000)")
            ax_cut.set_aspect("equal", adjustable="box")
            ax_cut.set_title("Candidate Cutout")

            # Reticle on cutout
            if cand_pos is not None:
                cut_rx, cut_ry = cut_wcs.world_to_pixel(cand_pos)
            else:
                cut_rx, cut_ry = cut_data.shape[1] // 2, cut_data.shape[0] // 2
            ax_cut.plot([cut_rx], [cut_ry], marker=reticle(which="rt"), ms=39, mec="black", mew=3.0, mfc="none")
            ax_cut.plot([cut_rx], [cut_ry], marker=reticle(which="rt"), ms=36, mec="white", mew=1.5, mfc="none")

            cb_cut = fig.colorbar(im_cut, ax=ax_cut, orientation="horizontal", fraction=0.046, pad=0.10)
            cb_cut.set_label("Flux density (Jy/beam)")
        except Exception:
            ax_cut = fig.add_subplot(gs_main[0, 2])
            ax_cut.set_facecolor("#1e1e1e")
            ax_cut.text(0.5, 0.5, "CUTOUT N/A", ha="center", va="center", color="silver", transform=ax_cut.transAxes)
            ax_cut.set_xticks([]); ax_cut.set_yticks([])
    else:
        ax_cut = fig.add_subplot(gs_main[0, 2])
        ax_cut.set_facecolor("#1e1e1e")
        ax_cut.text(0.5, 0.5, "CUTOUT N/A", ha="center", va="center", color="silver", transform=ax_cut.transAxes)
        ax_cut.set_xticks([]); ax_cut.set_yticks([])

    # Panel (1:3, 2): Tall Full Summary Metadata Box matching top cutout column width
    f_min_ds = float(ds_freqs[0]) if (ds_freqs is not None and len(ds_freqs) > 0) else 0.0
    f_max_ds = float(ds_freqs[-1]) if (ds_freqs is not None and len(ds_freqs) > 0) else 0.0
    ax_txt = fig.add_subplot(gs_main[1:3, 2])
    ax_txt.set_axis_off()
    raw_txt = (
        f"Name: {srcname}\n"
        f"Field: {field or '—'}\n"
        f"S/N: {snr_str}\n"
        f"DM: {dm_str}\n"
        f"SBID: {sbid or '—'}\n"
        f"Beam: {beam or '—'}\n"
        f"Scan: {scan or '—'}\n"
        f"Coord (J2000): {ra_str} {dec_str}\n"
        f"iMJD: {imjd}\n"
        f"Det. time: {det_time_str}\n"
        f"Method: {method}\n"
        f"Width: {width_str}\n"
        f"CASA timerange:\n"
        f"{casa_timerange}\n"
        f"Dynamic Spectrum:\n"
        f"  {ds_path.name}\n"
        f"  ({len(ds_freqs)} ch | {f_min_ds:.1f}-{f_max_ds:.1f} MHz | {t_end - t_start:.1f} s)"
    )
    wrapped_txt = wrap_candidate_metadata_text(raw_txt, max_width=33)
    ax_txt.text(
        0.03, 0.98, wrapped_txt,
        transform=ax_txt.transAxes,
        va="top", ha="left",
        fontsize=9.5,
        linespacing=1.26,
        bbox=dict(boxstyle="square,pad=0.8", facecolor="#fcfcfc", edgecolor="#bbbbbb", lw=1.2),
    )
    ax_txt.set_title("Candidate Summary", pad=8, fontsize=11)

    # Dynamic spectrum block & lightcurves within gs_main
    fig.canvas.draw()
    fig_w, fig_h = fig.get_size_inches()
    bbox_ds_area = gs_main[1, 0:2].get_position(fig)
    total_w_px = bbox_ds_area.width * fig_w * dpi
    total_h_px = bbox_ds_area.height * fig_h * dpi
    h_beam_px = (0.20 / 1.20) * total_h_px
    w_cbar_px = 24.0
    pad_cbar_px = 14.0
    w_ds_spec_total = max(100.0, total_w_px - w_cbar_px - pad_cbar_px)
    w_spec_px = h_beam_px
    w_ds_px = max(100.0, w_ds_spec_total - w_spec_px)

    gs_outer = GridSpecFromSubplotSpec(
        nrows=1, ncols=2,
        subplot_spec=gs_main[1, 0:2],
        width_ratios=[w_ds_spec_total, w_cbar_px],
        wspace=pad_cbar_px / w_ds_spec_total,
    )

    gs_inner = GridSpecFromSubplotSpec(
        nrows=2, ncols=2,
        subplot_spec=gs_outer[0],
        height_ratios=[1.0, 0.20],
        width_ratios=[w_ds_px, w_spec_px],
        hspace=0.0, wspace=0.0,
    )

    ax_ds = fig.add_subplot(gs_inner[0, 0])
    ax_spec = fig.add_subplot(gs_inner[0, 1], sharey=ax_ds)
    ax_beam_lc = fig.add_subplot(gs_inner[1, 0], sharex=ax_ds)
    cax = fig.add_subplot(gs_outer[1])

    # Row 2: Main Image-plane Lightcurve matching height and aligned with ax_ds
    gs_lc_outer = GridSpecFromSubplotSpec(
        nrows=1, ncols=2,
        subplot_spec=gs_main[2, 0:2],
        width_ratios=[w_ds_spec_total, w_cbar_px],
        wspace=pad_cbar_px / w_ds_spec_total,
    )
    gs_lc_inner = GridSpecFromSubplotSpec(
        nrows=1, ncols=2,
        subplot_spec=gs_lc_outer[0],
        width_ratios=[w_ds_px, w_spec_px],
        wspace=0.0,
    )
    ax_lc = fig.add_subplot(gs_lc_inner[0, 0], sharex=ax_ds)

    # Dynamic Spectrum Waterfall (averaged in time by W/4 and frequency by ~W/4)
    ds_disp = ds_slice
    t_bin = max(1, w_int // 4)
    f_bin = t_bin
    if (t_bin > 1 or f_bin > 1) and ds_slice.ndim == 2:
        Nt, Nf = ds_slice.shape
        Nt_trim = (Nt // t_bin) * t_bin
        Nf_trim = (Nf // f_bin) * f_bin
        if Nt_trim > 0 and Nf_trim > 0:
            sub = ds_slice[:Nt_trim, :Nf_trim]
            ds_disp = np.nanmean(
                sub.reshape(Nt_trim // t_bin, t_bin, Nf_trim // f_bin, f_bin),
                axis=(1, 3)
            )

    vmin_ds, vmax_ds = np.nanpercentile(ds_disp, [5.0, 99.5])
    t_min_ds = float(ds_t_slice[0])
    t_max_ds = float(ds_t_slice[-1])
    f_min_ds = float(ds_freqs[0])
    f_max_ds = float(ds_freqs[-1])
    im_ds = ax_ds.imshow(
        ds_disp.T,
        origin="lower",
        aspect="auto",
        extent=[t_min_ds, t_max_ds, f_min_ds, f_max_ds],
        cmap=cmap,
        vmin=vmin_ds,
        vmax=vmax_ds,
    )
    ax_ds.set_ylabel("Frequency (MHz)", fontsize=10)
    ax_ds.set_title("Dynamic spectrum", fontsize=11, fontweight="normal")
    if np.isfinite(det_time):
        t_det_rel = det_time - t_start
        if t_min_ds <= t_det_rel <= t_max_ds:
            ax_ds.axvline(t_det_rel, color="crimson", ls="--", lw=1.2, alpha=0.85, label="Det Time")
            if cand_dm > 0:
                nu_top = max(f_min_ds, f_max_ds)
                f_eval = np.linspace(min(f_min_ds, f_max_ds), nu_top, 100)
                dt_sweep = 4.1488e3 * cand_dm * (f_eval**(-2) - nu_top**(-2))
                ax_ds.plot(t_det_rel + dt_sweep, f_eval, color="gold", ls=":", lw=1.5, alpha=0.9, label=f"DM={cand_dm:.1f} track")
    ax_ds.tick_params(labelbottom=False, bottom=False)

    # Vertical Colorbar to the right of the dynamic spectrum block
    cb_ds = fig.colorbar(im_ds, cax=cax, orientation="vertical")
    cb_ds.set_label("Flux density (Jy)", fontsize=9)

    # Time-averaged spectrum directly adjoining to the right along frequency axis (wspace=0)
    beam_spec = np.nanmean(ds_slice, axis=0)
    ax_spec.plot(beam_spec, ds_freqs, color="indigo", lw=1.2)
    ax_spec.tick_params(labelleft=False, left=False)
    ax_spec.set_xlabel("Mean (Jy)", fontsize=8)
    ax_spec.grid(True, alpha=0.3)

    # Frequency-averaged lightcurve directly adjoining below along time axis (hspace=0)
    beam_lc = np.nanmean(ds_slice, axis=1) * 1000.0  # mJy
    ax_beam_lc.plot(ds_t_slice, beam_lc, color="teal", lw=1.2)
    if np.isfinite(det_time):
        t_det_rel = det_time - t_start
        if t_min_ds <= t_det_rel <= t_max_ds:
            ax_beam_lc.axvline(t_det_rel, color="crimson", ls="--", lw=1.2, alpha=0.85)
    ax_beam_lc.set_ylabel("Beamformed\n(mJy)", fontsize=8)
    ax_beam_lc.tick_params(labelbottom=False)
    ax_beam_lc.grid(True, alpha=0.3)

    # Main Image-plane Lightcurve below (matching height of image cutouts)
    ax_lc.plot(rel_times_start, lc_flux, color="tab:blue", lw=1.2, label="Raw", alpha=0.75, zorder=2)
    if lc_sm_flux is not None and len(lc_sm_flux) > 0 and lc_sm_time is not None and len(lc_sm_time) > 0:
        rel_sm_start = lc_sm_time - t_start
        ax_lc.plot(rel_sm_start, lc_sm_flux, color="tab:green", lw=1.6, label=f"Boxcar (w={w_int})", zorder=3)
    if np.isfinite(det_time):
        t_det_rel = det_time - t_start
        if t_min_ds <= t_det_rel <= t_max_ds:
            ax_lc.axvline(t_det_rel, color="crimson", ls="--", lw=1.2, alpha=0.85, zorder=4)
    ax_lc.set_xlabel("Time from Start (s)", fontsize=10)
    ax_lc.set_ylabel("Image Flux\n(mJy/beam)", fontsize=9)
    ax_lc.grid(True, alpha=0.3)
    ax_lc.legend(loc="upper right", fontsize=8)

    meta_footer = f"Generated with fastducc.dynspec | File: {ds_path.name}"
    fig.text(0.5, 0.01, meta_footer, ha="center", va="bottom", fontsize=8, color="gray", alpha=0.7)

    fig.savefig(out_path, bbox_inches="tight")
    # Also save PNG alongside PDF
    png_path = out_path.with_suffix(".png")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Updated candidate summary with dynamic spectrum: {out_path} and {png_path}")
    return out_path
