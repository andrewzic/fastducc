# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import glob
import logging
import subprocess
import socket
import getpass
import datetime
from typing import Dict, Any, Optional

import numpy as np

import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
        mode="edge"
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
                mode="partial"
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
    )

    box = AnchoredText(txt, loc="upper left", frameon=True, pad=1.0, prop={"size": 10})
    ax_txt.add_artist(box)
    ax_txt.set_title("Summary", pad=12)

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

    return {"figure": out_fig, "fits_cutout": out_fits, "fits_fullframe": out_fits_full}


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
    out = np.zeros((target_size, target_size), dtype=np.float32)
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
    f_hz = float(candidate.get("f_hz", 1.0 / period_s if period_s > 0 else 0.0))
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
    if cutout is not None and isinstance(cutout, np.ndarray) and cutout.size > 0:
        cutout_det = _crop_or_pad_cutout(cutout, spatial_size)
    elif "cutout" in candidate and candidate["cutout"] is not None:
        cutout_det = _crop_or_pad_cutout(np.asarray(candidate["cutout"]), spatial_size)
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
            mode="edge",
        ).astype(np.float32)
    else:
        cutout_det = np.zeros((spatial_size, spatial_size), dtype=np.float32)
        cutout_det[spatial_size // 2, spatial_size // 2] = snr_val

    cutout_off_det = None
    if cutout_off is not None and isinstance(cutout_off, np.ndarray) and cutout_off.size > 0:
        cutout_off_det = _crop_or_pad_cutout(cutout_off, spatial_size)
    elif "cutout_off" in candidate and candidate["cutout_off"] is not None:
        cutout_off_det = _crop_or_pad_cutout(np.asarray(candidate["cutout_off"]), spatial_size)

    # Prepare 1D rotational pulse profile
    if profile is not None and len(profile) > 0:
        p_raw = np.asarray(profile, dtype=np.float64)
    elif "profile" in candidate and candidate["profile"] is not None and len(candidate["profile"]) > 0:
        p_raw = np.asarray(candidate["profile"], dtype=np.float64)
    else:
        p_raw = np.mean(cutout_det, axis=1).astype(np.float64)

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
                c2d = Cutout2D(data=cdata, position=pos, size=(size_rad * u.rad, size_rad * u.rad), wcs=cw, mode="partial")
                cont_im = c2d.data
                bunit = str(chdr.get("BUNIT", "")).upper()
                if "JY" in bunit and "MJY" not in bunit and "UJY" not in bunit:
                    cont_im = cont_im * 1000.0  # Convert Jy/beam to mJy/beam
                cont_wcs = c2d.wcs
                cont_vmin = float(np.nanpercentile(cont_im, 1.0))
                cont_vmax = float(np.nanpercentile(cont_im, 99.5))
            except Exception as e:
                logger.debug(f"Could not load continuum cutout: {e}")

    # Build GridSpec figure: 4 columns if off-pulse cutout is available, else 3
    base_stem = os.path.basename(out_prefix)
    if out_dir_plots:
        os.makedirs(out_dir_plots, exist_ok=True)
        out_fig = os.path.join(out_dir_plots, f"{base_stem}_summary.pdf")
    else:
        out_dir = os.path.dirname(os.path.abspath(out_prefix))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        out_fig = f"{out_prefix}_summary.pdf"

    has_off = cutout_off_det is not None
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
        ax_cont = fig.add_subplot(gs[0, 0], projection=wcs_cut)
        vmin_c = float(np.nanpercentile(cutout_det, 5.0))
        vmax_c = float(np.nanpercentile(cutout_det, 99.5))
        im_cont = ax_cont.imshow(cutout_det, origin="lower", cmap=cmap, vmin=vmin_c, vmax=vmax_c)
        ax_cont.coords.grid(True, color="white", alpha=0.35, ls=":")
        ax_cont.set_xlabel("RA (J2000)")
        ax_cont.set_ylabel("Dec (J2000)")
        ax_cont.set_aspect("equal", adjustable="box")
        c_x = cutout_det.shape[1] // 2
        c_y = cutout_det.shape[0] // 2
        ax_cont.plot([c_x], [c_y], marker=reticle(which="rt"), ms=39, mec="black", mew=3.0, mfc="none")
        ax_cont.plot([c_x], [c_y], marker=reticle(which="rt"), ms=36, mec="white", mew=1.5, mfc="none")
        cb0 = fig.colorbar(im_cont, ax=ax_cont, orientation="horizontal", fraction=0.046, pad=0.10)
        cb0.set_label("Folded Dirty Intensity")
        ax_cont.set_title("Field (Reference)")

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
        ax_full = fig.add_subplot(gs[0, 1], projection=wcs_cut)
        vmin_c = float(np.nanpercentile(cutout_det, 5.0))
        vmax_c = float(np.nanpercentile(cutout_det, 99.5))
        im_full = ax_full.imshow(cutout_det, origin="lower", cmap=cmap, vmin=vmin_c, vmax=vmax_c)
        ax_full.coords.grid(True, color="white", alpha=0.35, ls=":")
        ax_full.set_xlabel("RA (J2000)")
        ax_full.set_ylabel("Dec (J2000)")
        ax_full.set_aspect("equal", adjustable="box")
        c_x = cutout_det.shape[1] // 2
        c_y = cutout_det.shape[0] // 2
        ax_full.plot([c_x], [c_y], marker=reticle(which="rt"), ms=39, mec="black", mew=3.0, mfc="none")
        ax_full.plot([c_x], [c_y], marker=reticle(which="rt"), ms=36, mec="white", mew=1.5, mfc="none")
        cb1 = fig.colorbar(im_full, ax=ax_full, orientation="horizontal", fraction=0.046, pad=0.10)
        cb1.set_label("Folded Dirty Intensity")
        ax_full.set_title(f"Detection Map @ P={period_s*1e3:.2f} ms ({f_hz:.4f} Hz)")

    # Panel (0, 2): Zoomed On-Pulse Cutout
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

    # Panel (0, 3): Zoomed Off-Pulse Cutout (if available)
    if has_off:
        ax_cut_off = fig.add_subplot(gs[0, 3], projection=wcs_cut)
        # Use shared vmin/vmax with on-pulse cutout for direct contrast comparison
        im_cut_off = ax_cut_off.imshow(cutout_off_det, origin="lower", cmap=cmap, vmin=vmin_c, vmax=vmax_c)
        ax_cut_off.coords.grid(True, color="white", alpha=0.35, ls=":")
        ax_cut_off.set_xlabel("RA (J2000)")
        ax_cut_off.set_ylabel("Dec (J2000)")
        ax_cut_off.set_aspect("equal", adjustable="box")
        ax_cut_off.plot([c_x], [c_y], marker=reticle(which="rt"), ms=39, mec="black", mew=3.0, mfc="none")
        ax_cut_off.plot([c_x], [c_y], marker=reticle(which="rt"), ms=36, mec="white", mew=1.5, mfc="none")
        cb3 = fig.colorbar(im_cut_off, ax=ax_cut_off, orientation="horizontal", fraction=0.046, pad=0.10)
        cb3.set_label("Folded Dirty Intensity")
        ax_cut_off.set_title("Off-Pulse Cutout")

    # Bottom Row Left: Folded Pulse Profile (2 Full Cycles)
    ax_lc = fig.add_subplot(gs[1, 0:ncols - 1])
    ax_lc.step(phase2, p2_raw, where="mid", color="tab:blue", lw=1.5, label="Raw Folded Profile", zorder=2)
    ax_lc.plot(phase2, p2_box, color="tab:green", lw=1.8, label=f"Boxcar Matched Filter (w={width_bins} bins, duty={duty_cycle*100.0:.1f}%)", zorder=3)
    peak_b = int(candidate.get("best_b", np.argmax(p_norm)))
    peak_phase1 = peak_b / float(nbins)
    peak_phase2 = peak_phase1 + 1.0
    ax_lc.axvline(peak_phase1, color="crimson", ls="--", lw=1.2, label=f"Peak Phase ({peak_phase1:.3f})", zorder=4)
    ax_lc.axvline(peak_phase2, color="crimson", ls="--", lw=1.2, zorder=4)
    # Highlight on-pulse phase intervals
    w_frac = width_bins / float(nbins)
    ax_lc.axvspan(peak_phase1, peak_phase1 + w_frac, color="tab:green", alpha=0.15, label="On-Pulse Window")
    ax_lc.axvspan(peak_phase2, peak_phase2 + w_frac, color="tab:green", alpha=0.15)
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

    # Save Cutout FITS with WCS & metadata
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
    if cutout_off_det is not None:
        if out_dir_cutouts:
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

    # Save 1D Folded Pulse Profile NPZ
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
