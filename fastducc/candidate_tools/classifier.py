# -*- coding: utf-8 -*-
"""
Candidate diagnostic scoring, harmonic vs non-harmonic structure evaluation,
local continuum flux sampling, and classification for periodicity candidate aggregation.
"""
from __future__ import annotations

import glob
import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("fastducc.candidate_tools.classifier")


def evaluate_cluster_harmonic_structure(
    periods: Sequence[float] | np.ndarray,
    snrs: Sequence[float] | np.ndarray,
    tol_frac: float = 0.02,
    harmonic_factors: Sequence[int] = (1, 2, 3, 4, 5, 6, 8, 16),
) -> Dict[str, Any]:
    """
    Analyse the distribution of periods {P_i} within a spatial candidate cluster.

    Step 1 Requirement:
    - Identify the highest-S/N detection as candidate fundamental period P0.
    - Test whether other periods are consistent with integer harmonics (P ≈ k * P0)
      or subharmonics (P ≈ P0 / k) within fractional tolerance tol_frac (default 2%).
    - Compute:
        n_periods: Total number of distinct period trials in this spatial cluster.
        n_non_harmonic: Number of periods that cannot be explained as P0 * (k or 1/k).
        frac_harmonic: Fraction of detections consistent with harmonic structure.
        period_dispersion_ratio: Ratio of non-harmonic periods to total periods.
    """
    p_arr = np.asarray(periods, dtype=np.float64)
    s_arr = np.asarray(snrs, dtype=np.float64)

    valid_mask = np.isfinite(p_arr) & (p_arr > 0)
    if not np.any(valid_mask):
        return {
            "fundamental_p0": float("nan"),
            "n_periods": 0,
            "n_non_harmonic": 0,
            "frac_harmonic": 1.0,
            "period_dispersion_ratio": 0.0,
        }

    v_periods = p_arr[valid_mask]
    v_snrs = s_arr[valid_mask] if len(s_arr) == len(p_arr) else np.ones_like(v_periods)

    # 1. Fundamental period P0 from highest-SNR detection
    best_idx = int(np.argmax(v_snrs))
    p0 = float(v_periods[best_idx])

    # 2. Group into distinct period trials within tol_frac
    sorted_p = np.unique(v_periods)
    distinct_p: List[float] = [float(sorted_p[0])]
    last_p = float(sorted_p[0])
    for p in sorted_p[1:]:
        p_val = float(p)
        if abs(p_val - last_p) > tol_frac * max(1e-6, last_p):
            distinct_p.append(p_val)
            last_p = p_val

    n_periods = len(distinct_p)

    if p0 > 0 and np.isfinite(p0) and len(harmonic_factors) > 0:
        factors = np.asarray(harmonic_factors, dtype=np.float64)
        targets = np.concatenate([p0 * factors, p0 / factors])
        tols = tol_frac * targets

        dp_arr = np.asarray(distinct_p, dtype=np.float64)
        dp_is_harm = np.any(np.abs(dp_arr[:, None] - targets[None, :]) <= tols[None, :], axis=1)
        n_non_harmonic = int(np.sum(~dp_is_harm))

        vp_is_harm = np.any(np.abs(v_periods[:, None] - targets[None, :]) <= tols[None, :], axis=1)
        n_harm_detections = int(np.sum(vp_is_harm))
    else:
        n_non_harmonic = n_periods
        n_harm_detections = 0

    frac_harmonic = float(n_harm_detections / len(v_periods)) if len(v_periods) > 0 else 1.0
    period_dispersion_ratio = float(n_non_harmonic / max(1, n_periods))

    return {
        "fundamental_p0": p0,
        "n_periods": n_periods,
        "n_non_harmonic": n_non_harmonic,
        "frac_harmonic": frac_harmonic,
        "period_dispersion_ratio": period_dispersion_ratio,
    }


def find_beam_continuum_image(continuum_dir: Optional[str], beam_str: Any) -> Optional[str]:
    """
    Locate the matching beam's continuum image FITS cube/image in continuum_dir.
    Prefers higher selfcal iteration (e.g. selfcal_7 over selfcal_6) and restored images.
    """
    if not continuum_dir or not os.path.isdir(continuum_dir):
        return None

    beam_str_clean = str(beam_str).strip()
    m = re.search(r"(\d+)", beam_str_clean)
    if not m:
        return None

    beam_num = int(m.group(1))
    patterns = [
        f"*beam{beam_num:02d}*selfcal*image.fits",
        f"*beam{beam_num:02d}*image.fits",
        f"*beam{beam_num:02d}*.fits",
        f"*beam{beam_num}*.fits",
    ]

    for pat in patterns:
        matches = sorted(glob.glob(os.path.join(continuum_dir, pat)))
        if matches:
            return matches[-1]

    return None


def sample_local_continuum_flux(
    ra_deg: float,
    dec_deg: float,
    continuum_fits_path: Optional[str],
    box_radius_pix: int = 30,
    inner_annulus_pix: int = 6,
    peak_box_pix: int = 3,
    fits_cache: Optional[Dict[str, Tuple[np.ndarray, Any]]] = None,
) -> Dict[str, float]:
    """
    Extract the local continuum pixel value I_cont, local background RMS sigma_cont,
    and local continuum significance SNR_cont = I_cont / sigma_cont.

    Parameters
    ----------
    ra_deg : float
        Candidate right ascension in degrees.
    dec_deg : float
        Candidate declination in degrees.
    continuum_fits_path : Optional[str]
        Path to beam continuum FITS image.
    box_radius_pix : int
        Outer radius for local background annulus (default 30 pixels).
    inner_annulus_pix : int
        Inner radius for local background annulus (default 6 pixels).
    peak_box_pix : int
        Size of central box to extract peak continuum flux (default 3 pixels).
    fits_cache : Optional[dict]
        Cache dictionary mapping path -> (2D data, celestial WCS).

    Returns
    -------
    dict with keys 'cont_flux_jy', 'cont_rms_jy', 'cont_snr'.
    """
    default_res = {"cont_flux_jy": float("nan"), "cont_rms_jy": float("nan"), "cont_snr": 0.0}
    if not continuum_fits_path or not os.path.isfile(continuum_fits_path):
        return default_res

    try:
        from astropy.io import fits
        from astropy.wcs import WCS
        import warnings
        from astropy.wcs import FITSFixedWarning

        if fits_cache is not None and continuum_fits_path in fits_cache:
            data_2d, wcs_cel = fits_cache[continuum_fits_path]
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FITSFixedWarning)
                with fits.open(continuum_fits_path) as hdul:
                    raw_data = hdul[0].data
                    hdr = hdul[0].header
                    data_2d = np.squeeze(raw_data).astype(np.float64)
                    wcs_cel = WCS(hdr).celestial
            if fits_cache is not None:
                fits_cache[continuum_fits_path] = (data_2d, wcs_cel)

        # Coordinate transformation to pixel values
        px_val, py_val = wcs_cel.world_to_pixel_values(ra_deg, dec_deg)
        px, py = float(px_val), float(py_val)
        if not (np.isfinite(px) and np.isfinite(py)):
            return default_res

        ix, iy = int(round(px)), int(round(py))
        ny, nx = data_2d.shape
        if not (0 <= ix < nx and 0 <= iy < ny):
            return default_res

        # Peak flux within central peak_box_pix x peak_box_pix
        hp = peak_box_pix // 2
        y0, y1 = max(0, iy - hp), min(ny, iy + hp + 1)
        x0, x1 = max(0, ix - hp), min(nx, ix + hp + 1)
        peak_flux = float(np.max(data_2d[y0:y1, x0:x1]))

        # Local background in annulus: inner_annulus_pix to box_radius_pix
        by0, by1 = max(0, iy - box_radius_pix), min(ny, iy + box_radius_pix + 1)
        bx0, bx1 = max(0, ix - box_radius_pix), min(nx, ix + box_radius_pix + 1)

        yy, xx = np.ogrid[by0:by1, bx0:bx1]
        dist = np.hypot(xx - ix, yy - iy)
        ann_mask = (dist >= inner_annulus_pix) & (dist <= box_radius_pix)

        ann_vals = data_2d[by0:by1, bx0:bx1][ann_mask]
        ann_vals = ann_vals[np.isfinite(ann_vals)]

        if len(ann_vals) > 10:
            med_bg = float(np.median(ann_vals))
            mad = float(np.median(np.abs(ann_vals - med_bg)))
            rms_bg = 1.4826 * mad
            if rms_bg <= 0 or not np.isfinite(rms_bg):
                rms_bg = float(np.std(ann_vals))
        else:
            med_bg = 0.0
            rms_bg = float("nan")

        if np.isfinite(rms_bg) and rms_bg > 0:
            cont_snr = float((peak_flux - med_bg) / rms_bg)
        else:
            cont_snr = 0.0

        return {
            "cont_flux_jy": peak_flux,
            "cont_rms_jy": rms_bg,
            "cont_snr": cont_snr,
        }
    except Exception as exc:
        logger.debug(f"Error sampling continuum flux at ({ra_deg:.4f}, {dec_deg:.4f}): {exc}")
        return default_res


def classify_candidate(
    is_psrcat: bool,
    cont_snr: float,
    cont_flux_jy: float,
    racs_match_name: str,
    racs_flux_jy: float,
    n_periods: int,
    n_non_harmonic: int,
    frac_harmonic: float,
    n_detections_total: int,
    dm_vals: Optional[Sequence[float]] = None,
    snr_vals: Optional[Sequence[float]] = None,
) -> Tuple[str, str]:
    """
    Step 3 Diagnostic Scoring & Classification Flagging.

    Returns
    -------
    continuum_association : str ("NONE", "FAINT", "BRIGHT")
        NONE   : SNR_cont < 4sigma and no RACS match
        FAINT  : 4sigma <= SNR_cont < 10sigma or faint RACS source (< 5 mJy)
        BRIGHT : SNR_cont >= 10sigma or strong RACS source (> 20 mJy)

    cand_classification   : str ("KNOWN_PULSAR", "REAL_CANDIDATE",
                                 "CONTINUUM_RESIDUAL_ARTIFACT",
                                 "RFI_OR_LEAKAGE", "SINGLETON")
    """
    has_racs = bool(
        racs_match_name
        and str(racs_match_name).strip()
        and str(racs_match_name).lower() not in ("none", "unknown", "")
    )
    r_flux = float(racs_flux_jy) if (has_racs and np.isfinite(racs_flux_jy)) else 0.0
    snr_c = float(cont_snr) if np.isfinite(cont_snr) else 0.0

    # 1. Continuum association
    if snr_c >= 10.0 or r_flux >= 0.020:
        cont_assoc = "BRIGHT"
    elif snr_c >= 4.0 or (has_racs and r_flux > 0.0):
        cont_assoc = "FAINT"
    else:
        cont_assoc = "NONE"

    # Significant continuum threshold for artifact classification:
    # SNR_cont > 5sigma OR strong RACS match (>= 5 mJy = 0.005 Jy)
    has_sig_continuum = (snr_c > 5.0) or (has_racs and r_flux >= 0.005)

    # Multi-period explosion indicator:
    # n_non_harmonic >= 3 OR frac_harmonic < 0.5 (when multi-period triggers exist)
    is_multi_period_explosion = (n_non_harmonic >= 3) or (n_periods > 1 and frac_harmonic < 0.5)

    # 2. Classification
    if is_psrcat:
        cand_class = "KNOWN_PULSAR"
    elif is_multi_period_explosion and has_sig_continuum:
        cand_class = "CONTINUUM_RESIDUAL_ARTIFACT"
    elif is_multi_period_explosion:
        cand_class = "RFI_OR_LEAKAGE"
    elif n_detections_total <= 1 or n_periods <= 1:
        if n_detections_total <= 1:
            cand_class = "SINGLETON"
        else:
            cand_class = "REAL_CANDIDATE"
    else:
        # Check for flat S/N vs DM response across multiple trials
        is_flat_dm = False
        if dm_vals is not None and snr_vals is not None and len(dm_vals) >= 8:
            unique_dms = np.unique(np.round(dm_vals, 1))
            if len(unique_dms) >= 8:
                s_max = np.max(snr_vals)
                s_min = np.min(snr_vals)
                if s_min > 0 and (s_max / s_min) < 1.25:
                    is_flat_dm = True

        if is_flat_dm:
            cand_class = "RFI_OR_LEAKAGE"
        else:
            cand_class = "REAL_CANDIDATE"

    return cont_assoc, cand_class
