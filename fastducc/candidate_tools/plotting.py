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
    times: np.ndarray,
    cube: np.ndarray,
    candidate: Dict[str, Any],
    out_prefix: str,
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
) -> Dict[str, str]:
    """
    Build a prepfold-like single candidate summary PDF page and save cutout/fullframe FITS.
    """
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
        pixscale_rad=pix_rad, ra_sign=ra_sign, radesys=radesys, equinox=equinox
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
        pixscale_rad=pix_rad, ra_sign=ra_sign, radesys=radesys, equinox=equinox
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
