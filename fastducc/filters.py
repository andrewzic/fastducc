from typing import Tuple, List, Dict, Any, Optional, Sequence
import numpy as np

from scipy.ndimage import maximum_filter


def extract_beam_pix(
    candidate: Optional[Dict[str, Any]] = None,
    header: Optional[Any] = None,
    pixscale_rad: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract synthesized beam major and minor FWHM in pixels from candidate metadata,
    FITS header, or explicit pixel scale.

    Parameters
    ----------
    candidate : dict, optional
        Candidate metadata dictionary.
    header : Header-like or dict, optional
        FITS header containing BMAJ/BMIN/CDELT.
    pixscale_rad : float, optional
        Pixel scale in radians.

    Returns
    -------
    (bmaj_pix, bmin_pix) : Tuple of Optional[float]
    """
    cand = candidate or {}
    hdr = header or cand.get("header") or {}

    bmaj_pix = cand.get("bmaj_pix")
    bmin_pix = cand.get("bmin_pix")
    if bmaj_pix is not None and bmin_pix is not None:
        try:
            return float(bmaj_pix), float(bmin_pix)
        except (ValueError, TypeError):
            pass

    # Determine pixel scale
    p_rad = None
    if pixscale_rad is not None:
        p_rad = float(pixscale_rad)
    else:
        for k in ("pixsize_rad", "pixscale_rad", "pix_rad"):
            if cand.get(k) is not None:
                try:
                    p_rad = float(cand[k])
                    break
                except (ValueError, TypeError):
                    pass

    p_deg = None
    p_arcsec = None
    if p_rad is not None and p_rad > 0:
        p_deg = float(np.degrees(p_rad))
        p_arcsec = p_deg * 3600.0
    elif hasattr(hdr, "get"):
        cdelt2 = hdr.get("CDELT2")
        if cdelt2 is not None:
            try:
                p_deg = abs(float(cdelt2))
                p_arcsec = p_deg * 3600.0
            except (ValueError, TypeError):
                pass

    # Extract BMAJ / BMIN
    bmaj_val = None
    bmin_val = None
    if hasattr(hdr, "get"):
        bmaj_val = hdr.get("BMAJ")
        bmin_val = hdr.get("BMIN")

    if bmaj_val is None:
        bmaj_val = cand.get("bmaj", cand.get("bmaj_deg", cand.get("bmaj_arcsec")))
    if bmin_val is None:
        bmin_val = cand.get("bmin", cand.get("bmin_deg", cand.get("bmin_arcsec")))

    out_bmaj_pix = None
    out_bmin_pix = None

    if bmaj_val is not None:
        try:
            bv = float(bmaj_val)
            if bv < 1.0 and p_deg is not None and p_deg > 0:
                out_bmaj_pix = abs(bv / p_deg)
            elif bv >= 1.0 and p_arcsec is not None and p_arcsec > 0:
                out_bmaj_pix = abs(bv / p_arcsec)
        except (ValueError, TypeError):
            pass

    if bmin_val is not None:
        try:
            bv = float(bmin_val)
            if bv < 1.0 and p_deg is not None and p_deg > 0:
                out_bmin_pix = abs(bv / p_deg)
            elif bv >= 1.0 and p_arcsec is not None and p_arcsec > 0:
                out_bmin_pix = abs(bv / p_arcsec)
        except (ValueError, TypeError):
            pass

    # Check beam_fwhm_arcsec
    if out_bmaj_pix is None or out_bmin_pix is None:
        fwhm_sec = cand.get("beam_fwhm_arcsec", cand.get("bmaj_arcsec"))
        if fwhm_sec is not None and p_arcsec is not None and p_arcsec > 0:
            try:
                sec_pix = float(fwhm_sec) / p_arcsec
                if out_bmaj_pix is None:
                    out_bmaj_pix = sec_pix
                if out_bmin_pix is None:
                    out_bmin_pix = sec_pix
            except (ValueError, TypeError):
                pass

    return out_bmaj_pix, out_bmin_pix


def evaluate_candidate_morphology_and_flux(
    cutout: np.ndarray,
    cutout_off: Optional[np.ndarray] = None,
    profile: Optional[np.ndarray] = None,
    on_bins: Optional[Sequence[int]] = None,
    off_bins: Optional[Sequence[int]] = None,
    *,
    is_synthesis_image: bool = True,
    bmaj_pix: Optional[float] = None,
    bmin_pix: Optional[float] = None,
    box_size: Optional[int] = None,
    max_hp_lag1: Optional[float] = None,
    min_sharpness: Optional[float] = None,
    max_fwhm_factor: float = 2.0,
    max_axial_ratio: float = 2.5,
    min_on_flux_share: float = 0.40,
    min_on_snr: float = 2.5,
) -> Dict[str, Any]:
    """
    Evaluates candidate morphology and pulse flux integrity to reject non-point source
    structures (diffuse blobs, sidelobe ridges, annular rings) and negative off-pulse
    trough artifacts.

    1. Morphology Filter (Beam-Aware Unsharp Mask + 2D ACF + Gaussian Fit / Sharpness):
       - If is_synthesis_image=True, accommodates synthesized beam PSF oversampling
         (expected lag-1 autocorrelation r1 > 0.50 in dirty maps).
       - Evaluates normalized 2D Autocorrelation Function (ACF) of high-pass residual.
       - Evaluates peak sharpness (relative drop to 4-connected neighbors).
       - Fits a 2D Gaussian / second moments to confirm the source is unresolved
         (FWHM <= max_fwhm_factor * W_beam) and non-elongated (axial_ratio <= max_axial_ratio).
       - Rejects annular rings / central depressions and diffuse extended structures.

    2. Flux Integrity Filter (On-to-Detection Flux Share & On-Pulse Significance):
       - Detects false positives driven purely by deep negative off-pulse baseline troughs
         where on-pulse flux is noise-like.
       - Evaluates share of detection flux coming from actual positive on-pulse emission:
         rho_on = I_on / (I_on - I_off) >= min_on_flux_share
       - Evaluates empirical on-pulse significance: S_on = I_on / std(I_off) >= min_on_snr

    Parameters
    ----------
    cutout : np.ndarray
        2D candidate detection cutout (or on-pulse cutout), typically 32x32.
    cutout_off : np.ndarray, optional
        2D off-pulse cutout for direct off-pulse flux verification.
    profile : np.ndarray, optional
        1D folded rotational pulse profile.
    on_bins : sequence of int, optional
        Indices of phase bins defining the on-pulse window.
    off_bins : sequence of int, optional
        Indices of phase bins defining the off-pulse window.
    is_synthesis_image : bool, optional
        Whether the cutout is an interferometric synthesis image / dirty map convolved
        with a synthesized beam PSF (default: True).
    bmaj_pix : float, optional
        Synthesized beam major axis FWHM in pixels.
    bmin_pix : float, optional
        Synthesized beam minor axis FWHM in pixels.
    box_size : int, optional
        Size of smoothing kernel for unsharp mask. If None, automatically derived
        (9+ for synthesis images, 5 for non-synthesis).
    max_hp_lag1 : float, optional
        Maximum allowed ACF Lag-1 correlation on high-pass residual. Defaults to
        0.75 for synthesis images and 0.25 for non-synthesis images.
    min_sharpness : float, optional
        Minimum allowed peak sharpness relative drop. Defaults to adaptive threshold
        based on synthesized beam PSF (~0.05-0.08) for synthesis images, 0.25 for non-synthesis.
    max_fwhm_factor : float, optional
        Maximum allowed effective FWHM in units of synthesized beam FWHM (default: 2.0).
    max_axial_ratio : float, optional
        Maximum allowed major-to-minor axial ratio to reject elongated ridges (default: 2.5).
    min_on_flux_share : float, optional
        Minimum fraction of detection flux coming from on-pulse (default: 0.40).
    min_on_snr : float, optional
        Minimum on-pulse significance in units of off-pulse std (default: 2.5).

    Returns
    -------
    dict:
        {
            "is_point_like": bool,
            "is_valid_flux": bool,
            "passed_all": bool,
            "morphology_verdict": str,
            "hp_lag1": float,
            "hp_lag2": float,
            "sharpness": float,
            "fwhm_eff_pix": float,
            "axial_ratio": float,
            "on_flux_share": float,
            "on_snr": float,
            "rejection_reasons": List[str],
        }
    """
    from scipy.ndimage import uniform_filter
    from scipy.optimize import curve_fit

    reasons: List[str] = []
    cut = np.asarray(cutout, dtype=np.float64)

    # Defaults
    hp_lag1 = 0.0
    hp_lag2 = 0.0
    sharpness = 1.0
    fwhm_eff = 0.0
    axial_ratio = 1.0
    is_point_like = True

    if cut.ndim == 2 and cut.size >= 16:
        ny, nx = cut.shape
        cy, cx = ny // 2, nx // 2

        # 1. Determine synthesized beam reference size (in pixels)
        if bmaj_pix is not None and bmin_pix is not None:
            w_beam = float(np.sqrt(float(bmaj_pix) * float(bmin_pix)))
        elif bmaj_pix is not None:
            w_beam = float(bmaj_pix)
        elif bmin_pix is not None:
            w_beam = float(bmin_pix)
        else:
            w_beam = 4.0 if is_synthesis_image else 1.5

        # 2. Calibrate thresholds based on synthesis mode
        if is_synthesis_image:
            expected_sharp = float(1.0 - np.exp(-4.0 * np.log(2.0) / (w_beam**2)))
            eff_min_sharp = min_sharpness if min_sharpness is not None else max(0.05, 0.45 * expected_sharp)
            eff_max_lag1 = max_hp_lag1 if max_hp_lag1 is not None else 0.75
            eff_box = box_size if box_size is not None else max(9, int(round(2.0 * w_beam)) | 1)
        else:
            eff_min_sharp = min_sharpness if min_sharpness is not None else 0.25
            eff_max_lag1 = max_hp_lag1 if max_hp_lag1 is not None else 0.25
            eff_box = box_size if box_size is not None else 5

        # Border background estimate
        border = np.concatenate([cut[0, :], cut[-1, :], cut[:, 0], cut[:, -1]])
        bg = float(np.median(border))

        # Peak search near center (within central search window)
        search_r = max(2, int(round(w_beam * 0.75)))
        y_lo, y_hi = max(0, cy - search_r), min(ny, cy + search_r + 1)
        x_lo, x_hi = max(0, cx - search_r), min(nx, cx + search_r + 1)
        central_sub = cut[y_lo:y_hi, x_lo:x_hi]
        py_rel, px_rel = np.unravel_index(np.argmax(central_sub), central_sub.shape)
        py = y_lo + py_rel
        px = x_lo + px_rel
        peak_val = float(cut[py, px])
        peak_exc = peak_val - bg

        # 4-connected sharpness around candidate peak
        nb = [
            cut[max(0, py - 1), px],
            cut[min(ny - 1, py + 1), px],
            cut[py, max(0, px - 1)],
            cut[py, min(nx - 1, px + 1)],
        ]
        if peak_exc > 0:
            sharpness = float((peak_val - np.mean(nb)) / peak_exc)
        elif peak_val > 0:
            sharpness = float((peak_val - np.mean(nb)) / peak_val)
        else:
            sharpness = 0.0

        # Unsharp Mask & 2D ACF
        bg_unsharp = uniform_filter(cut, size=eff_box, mode="nearest")
        hp = cut - bg_unsharp
        hp_sub = hp - np.mean(hp)
        var = np.sum(hp_sub**2)
        if var > 0:
            f = np.fft.fft2(hp_sub)
            acf = np.fft.ifft2(np.abs(f)**2).real
            acf = np.fft.fftshift(acf) / var
            mid_y, mid_x = acf.shape[0] // 2, acf.shape[1] // 2
            l1_vals = [acf[mid_y + 1, mid_x], acf[mid_y - 1, mid_x], acf[mid_y, mid_x + 1], acf[mid_y, mid_x - 1]]
            hp_lag1 = float(np.mean(l1_vals))
            if acf.shape[0] >= 5 and acf.shape[1] >= 5:
                l2_vals = [acf[mid_y + 2, mid_x], acf[mid_y - 2, mid_x], acf[mid_y, mid_x + 2], acf[mid_y, mid_x - 2]]
                hp_lag2 = float(np.mean(l2_vals))
            else:
                hp_lag2 = 0.0
        else:
            hp_lag1 = 0.0
            hp_lag2 = 0.0

        # Gaussian Fit for 2D size and axial ratio
        fit_ok = False
        if is_synthesis_image and peak_exc > 0:
            def gauss_model(coords, A, x_c, y_c, sig_x, sig_y, bg_val):
                y, x = coords
                return bg_val + A * np.exp(-0.5 * (((x - x_c) / sig_x)**2 + ((y - y_c) / sig_y)**2))

            Y, X = np.indices((ny, nx))
            sig0 = max(1.0, w_beam / 2.355)
            try:
                popt, _ = curve_fit(
                    gauss_model,
                    (Y.ravel(), X.ravel()),
                    cut.ravel(),
                    p0=[peak_exc, px, py, sig0, sig0, bg],
                    bounds=(
                        [0.0, max(0, px - 4), max(0, py - 4), 0.5, 0.5, -np.inf],
                        [np.inf, min(nx - 1, px + 4), min(ny - 1, py + 4), 30.0, 30.0, np.inf],
                    ),
                    maxfev=200,
                )
                sig_x, sig_y = abs(popt[3]), abs(popt[4])
                fwhm_eff = float(2.355 * np.sqrt(sig_x * sig_y))
                axial_ratio = float(max(sig_x, sig_y) / max(0.1, min(sig_x, sig_y)))
                fit_ok = True
            except Exception:
                fit_ok = False

        if not fit_ok and is_synthesis_image:
            # Fallback aperture second moments
            r_aper = max(6, int(round(1.5 * w_beam)))
            Y, X = np.ogrid[:ny, :nx]
            R = np.sqrt((X - px)**2 + (Y - py)**2)
            aper = R <= r_aper
            I_pos = np.maximum(cut - bg, 0.0) * aper
            tot_flux = np.sum(I_pos)
            if tot_flux > 0:
                mu_x = np.sum(X * I_pos) / tot_flux
                mu_y = np.sum(Y * I_pos) / tot_flux
                mxx = np.sum((X - mu_x)**2 * I_pos) / tot_flux
                myy = np.sum((Y - mu_y)**2 * I_pos) / tot_flux
                mxy = np.sum((X - mu_x) * (Y - mu_y) * I_pos) / tot_flux
                t1 = (mxx + myy) / 2.0
                t2 = np.sqrt(((mxx - myy) / 2.0)**2 + mxy**2)
                lam1, lam2 = t1 + t2, max(0.0, t1 - t2)
                sig_maj, sig_min = np.sqrt(lam1), np.sqrt(lam2)
                axial_ratio = float(sig_maj / max(sig_min, 0.3))
                fwhm_eff = float(2.355 * np.sqrt(sig_maj * sig_min))

        # Morphological decisions
        if hp_lag1 > eff_max_lag1:
            is_point_like = False
            reasons.append(f"Extended ACF (HP Lag-1 = {hp_lag1:.3f} > {eff_max_lag1:.2f})")
        if sharpness < eff_min_sharp:
            is_point_like = False
            reasons.append(f"Flat peak morphology (Sharpness = {sharpness:.3f} < {eff_min_sharp:.2f})")

        if is_synthesis_image:
            max_allowed_fwhm = max_fwhm_factor * w_beam
            if fwhm_eff > max_allowed_fwhm:
                is_point_like = False
                reasons.append(f"Resolved source size (FWHM = {fwhm_eff:.1f} > {max_allowed_fwhm:.1f} pix)")
            if axial_ratio > max_axial_ratio:
                is_point_like = False
                reasons.append(f"Elongated ridge artifact (Axial ratio = {axial_ratio:.2f} > {max_axial_ratio:.2f})")
            global_max = float(np.max(cut))
            center_val = float(cut[cy, cx])
            if center_val < 0.4 * global_max and np.hypot(py - cy, px - cx) > (w_beam * 0.65):
                is_point_like = False
                reasons.append(f"Annular ring artifact (Center={center_val:.2g} < 0.4*Max={global_max:.2g})")

    # 4. Flux Integrity Check (Pulse Profile or Cutouts)
    is_valid_flux = True
    on_flux_share = 1.0
    on_snr = 10.0

    if profile is not None and len(profile) >= 4:
        prof = np.asarray(profile, dtype=np.float64)
        nbins = len(prof)
        if on_bins is None:
            b_peak = int(np.argmax(prof))
            on_bins = [b_peak]
        if off_bins is None:
            off_bins = [b for b in range(nbins) if b not in on_bins]

        on_flux = float(np.mean(prof[on_bins]))
        off_vals = prof[off_bins]
        off_flux = float(np.mean(off_vals))
        off_std = float(np.std(off_vals)) if len(off_vals) > 1 else 1.0
        if off_std <= 0:
            off_std = 1.0

        det_flux = on_flux - off_flux
        if det_flux > 0:
            on_flux_share = float(on_flux / det_flux)
        else:
            on_flux_share = 0.0
        on_snr = float(on_flux / off_std)

        if on_flux_share < min_on_flux_share:
            is_valid_flux = False
            reasons.append(f"Negative off-pulse trough artifact (On-share = {on_flux_share:.2f} < {min_on_flux_share:.2f})")
        if on_snr < min_on_snr:
            is_valid_flux = False
            reasons.append(f"Insignificant on-pulse flux (On-SNR = {on_snr:.2f} < {min_on_snr:.2f})")

    elif cutout_off is not None and cutout_off.shape == cut.shape:
        # Evaluate via on-pulse and off-pulse cutouts if profile is not provided
        c_off = np.asarray(cutout_off, dtype=np.float64)
        cy, cx = cut.shape[0] // 2, cut.shape[1] // 2
        on_flux = float(cut[cy, cx])
        off_flux = float(c_off[cy, cx])
        # Off-pulse std from image annulus/outer pixels
        off_std = float(np.std(c_off)) if c_off.size > 1 else 1.0
        if off_std <= 0:
            off_std = 1.0

        det_flux = on_flux - off_flux
        if det_flux > 0:
            on_flux_share = float(on_flux / det_flux)
        else:
            on_flux_share = 0.0
        on_snr = float(on_flux / off_std)

        if on_flux_share < min_on_flux_share:
            is_valid_flux = False
            reasons.append(f"Negative off-pulse trough artifact (On-share = {on_flux_share:.2f} < {min_on_flux_share:.2f})")
        if on_snr < min_on_snr:
            is_valid_flux = False
            reasons.append(f"Insignificant on-pulse flux (On-SNR = {on_snr:.2f} < {min_on_snr:.2f})")

    passed_all = is_point_like and is_valid_flux
    if is_point_like and is_valid_flux:
        morphology_verdict = "POINT-LIKE"
    elif not is_point_like:
        morphology_verdict = "RESOLVED / EXTENDED"
    else:
        morphology_verdict = "DUBIOUS (Flux)"

    return {
        "is_point_like": is_point_like,
        "is_valid_flux": is_valid_flux,
        "passed_all": passed_all,
        "morphology_verdict": morphology_verdict,
        "hp_lag1": hp_lag1,
        "hp_lag2": hp_lag2,
        "sharpness": sharpness,
        "fwhm_eff_pix": fwhm_eff,
        "axial_ratio": axial_ratio,
        "on_flux_share": on_flux_share,
        "on_snr": on_snr,
        "rejection_reasons": reasons,
    }


def is_zero_flux_candidate(
    det: Dict[str, Any],
    cube: Optional[np.ndarray],
    zero_frames: Optional[np.ndarray] = None,
) -> bool:
    """
    Check if candidate has identically 0.0 or NaN flux at its central detection frame,
    or over its boxcar detection window in the given image cube, or if the frame is zero-padded.
    """
    if cube is None or cube.ndim != 3:
        return False

    T, Ny, Nx = cube.shape
    y = int(det.get("y", -1))
    x = int(det.get("x", -1))
    if y < 0 or y >= Ny or x < 0 or x >= Nx:
        return True

    w = max(1, int(det.get("width_samples", 1)))
    c_idx = int(det.get("center_idx", 0))
    t0 = int(det.get("t0_idx", 0))
    t1 = int(det.get("t1_idx_excl", t0 + w))

    c_idx = max(0, min(c_idx, T - 1))
    t0 = max(0, min(t0, T - 1))
    t1 = max(t0 + 1, min(t1, T))

    flux_center = float(cube[c_idx, y, x])
    flux_boxcar = float(np.mean(cube[t0:t1, y, x]))

    # If central frame pixel or boxcar average is 0.0 or NaN
    if np.isnan(flux_center) or flux_center == 0.0 or np.isnan(flux_boxcar) or flux_boxcar == 0.0:
        return True

    # If the central detection frame is entirely zero-padded (all pixels in frame == 0.0)
    if zero_frames is not None:
        if zero_frames[c_idx]:
            return True
    elif np.all(cube[c_idx] == 0.0):
        return True

    # If any pixel values in the detection window at (y,x) are 0.0 or NaN
    if np.any(cube[t0:t1, y, x] == 0.0) or np.any(np.isnan(cube[t0:t1, y, x])):
        return True

    # If any frame in the detection window is entirely zero-padded
    if zero_frames is not None:
        if np.any(zero_frames[t0:t1]):
            return True
    elif np.any(np.all(cube[t0:t1] == 0.0, axis=(1, 2))):
        return True

    return False


def nms_snr_map_2d(
    snr_2d: np.ndarray,                      # (Ny, Nx) SNR map
    base_detections: List[Dict[str, Any]],   # result from variance_search()
    *,
    threshold_sigma: float = 5.0,
    spatial_radius: int = 3,
    valid_mask: Optional[np.ndarray] = None,
    # Optional: provide a time axis + cube to derive a representative time
    times: Optional[np.ndarray] = None,      # (T,)
    cube: Optional[np.ndarray] = None,       # (T, Ny, Nx)
    time_tag_policy: str = "none"            # "none" | "peak_absdev" | "peak_flux"
) -> List[Dict[str, Any]]:
    """
    Spatial NMS on a single 2-D SNR map, merging metadata from base detections
    and (optionally) inferring a representative time index/time_center.
    """
    Ny, Nx = snr_2d.shape
    if valid_mask is not None:
        if valid_mask.shape != (Ny, Nx):
            raise ValueError("valid_mask must match shape of snr_2d")
        work = np.where(valid_mask, snr_2d, -np.inf)
    else:
        work = snr_2d.copy()

    # threshold then local maxima
    work = np.where(work >= threshold_sigma, work, -np.inf)
    local_max = maximum_filter(work, size=spatial_radius, mode='nearest')
    peaks_mask = (work >= local_max) & np.isfinite(work)

    ys, xs = np.where(peaks_mask)
    if ys.size == 0:
        return []

    # sort by SNR descending
    snr_vals = work[ys, xs]
    order = np.argsort(snr_vals)[::-1]
    ys, xs, snr_vals = ys[order], xs[order], snr_vals[order]

    # Build an index for quick lookup from (y,x) to base detection
    # (Use the highest-SNR base detection at that pixel if duplicates exist)
    base_by_pixel: Dict[Tuple[int,int], Dict[str,Any]] = {}
    for d in base_detections:
        key = (int(d["y"]), int(d["x"]))
        # keep the one with larger snr if multiple
        if (key not in base_by_pixel) or (float(d.get("snr", -np.inf)) > float(base_by_pixel[key].get("snr", -np.inf))):
            base_by_pixel[key] = d

    # NMS suppression and merge metadata
    detections: List[Dict[str, Any]] = []
    occ = np.ones((Ny, Nx), dtype=bool)

    # Precompute zero and valid frame masks once if cube is provided
    zero_frames = None
    valid_frames = None
    if cube is not None:
        zero_frames = np.all(cube == 0.0, axis=(1, 2))
        valid_frames = ~zero_frames

    for y, x, s in zip(ys, xs, snr_vals):
        if not occ[y, x]:
            continue
        # base metadata (if present)
        d0 = base_by_pixel.get((int(y), int(x)), {})
        det = dict(d0)  # copy
        det["y"] = int(y)
        det["x"] = int(x)
        det["snr"] = float(s)
        det.setdefault("std", float(snr_2d[y, x]))  # include std if not already there
        det.setdefault("width_samples", 1)
        det.setdefault("time_start", 0.0)
        det.setdefault("time_end", 0.0)
        det.setdefault("duration", 0.0)
        det.setdefault("t0_idx", 0)
        det.setdefault("t1_idx_excl", 0)
        det.setdefault("center_idx", 0)
        det.setdefault("time_center", 0.0)

        # Optional: derive a representative time from the full cube
        if (time_tag_policy != "none") and (times is not None) and (cube is not None):
            lc = cube[:, y, x].astype(np.float64, copy=False)
            valid_t = (lc != 0.0) & np.isfinite(lc) & valid_frames
            if np.any(valid_t):
                valid_indices = np.where(valid_t)[0]
                if time_tag_policy == "peak_absdev":
                    mu = np.nanmean(lc[valid_t])
                    best_sub_idx = int(np.nanargmax(np.abs(lc[valid_t] - mu)))
                    k = valid_indices[best_sub_idx]
                elif time_tag_policy == "peak_flux":
                    best_sub_idx = int(np.nanargmax(lc[valid_t]))
                    k = valid_indices[best_sub_idx]
                else:
                    k = None
            else:
                k = None

            if k is not None:
                dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
                det["t0_idx"] = k
                det["t1_idx_excl"] = k + 1
                det["time_start"] = float(times[k]) - 0.5 * dt
                det["time_end"] = float(times[k]) + 0.5 * dt
                det["duration"] = dt
                det["center_idx"] = k
                det["time_center"] = float(times[k])

        if cube is not None and is_zero_flux_candidate(det, cube, zero_frames=zero_frames):
            continue

        detections.append(det)

        # suppress neighborhood
        y0 = max(0, y - spatial_radius)
        y1 = min(Ny, y + spatial_radius + 1)
        x0 = max(0, x - spatial_radius)
        x1 = min(Nx, x + spatial_radius + 1)
        occ[y0:y1, x0:x1] = False

    return detections

def nms_snr_maps_per_width(
    snr_cubes: Dict[int, np.ndarray],  # width_samples -> SNR cube (T_eff, Ny, Nx)
    times: np.ndarray,
    *,
    threshold_sigma: float = 5.0,
    spatial_radius: int = 3,
    time_radius: int = 0,
    valid_mask: Optional[np.ndarray] = None,
    cube: Optional[np.ndarray] = None,   # (T, Ny, Nx)
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Spatial (and optional temporal) NMS using compiled 2D max-filter for peaks.
    """
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    detections_by_width: Dict[int, List[Dict[str, Any]]] = {}

    zero_frames = np.all(cube == 0.0, axis=(1, 2)) if cube is not None else None

    for w, snr_w in snr_cubes.items():
        T_eff, Ny, Nx = snr_w.shape
        if valid_mask is not None and valid_mask.shape != (Ny, Nx):
            raise ValueError("valid_mask must match (Ny, Nx) of SNR cubes")

        detections: List[Dict[str, Any]] = []

        # temporal occupancy (optional)
        if time_radius > 0:
            occ_time = [np.ones((Ny, Nx), dtype=bool) for _ in range(T_eff)]
        else:
            occ_time = None

        for t0 in range(T_eff):
            # copy slice
            snr_slice = snr_w[t0].copy()

            # apply mask & threshold first
            if valid_mask is not None:
                snr_slice = np.where(valid_mask, snr_slice, -np.inf)
            snr_slice = np.where(snr_slice >= threshold_sigma, snr_slice, -np.inf)

            # temporal occupancy gate
            if occ_time is not None:
                gate = np.ones((Ny, Nx), dtype=bool)
                for dt_step in range(-time_radius, time_radius + 1):
                    t2 = t0 + dt_step
                    if 0 <= t2 < T_eff:
                        gate &= occ_time[t2] #bitwise and assignment
                snr_slice = np.where(gate, snr_slice, -np.inf)

            # local maxima
            local_max = maximum_filter(snr_slice, size=spatial_radius, mode='nearest')
            #local_max = kernels._max_filter_2d(snr_slice, spatial_radius)
            peaks_mask = (snr_slice >= local_max) & np.isfinite(snr_slice)
            ys, xs = np.where(peaks_mask)
            if ys.size == 0:
                continue

            # sort by SNR descending
            snr_vals = snr_slice[ys, xs]
            order = np.argsort(snr_vals)[::-1]
            ys, xs, snr_vals = ys[order], xs[order], snr_vals[order]

            # accept peaks, suppress neighbors
            for y, x, s in zip(ys, xs, snr_vals):
                y0 = max(0, y - spatial_radius)
                y1 = min(Ny, y + spatial_radius + 1)
                x0 = max(0, x - spatial_radius)
                x1 = min(Nx, x + spatial_radius + 1)

                t1 = t0 + w  # exclusive
                
                time_start = float(times[t0]) - 0.5 * dt
                time_end = float(times[t1 - 1]) + 0.5 * dt
                duration = time_end - time_start
                
                center_idx  = t0 + (w // 2)
                time_center = 0.5 * (time_start + time_end)

                det = {
                    "y": int(y),
                    "x": int(x),
                    "snr": float(s),
                    "width_samples": int(w),
                    "t0_idx": int(t0),
                    "t1_idx_excl": int(t1),
                    "center_idx": int(center_idx),
                    "time_start": time_start,
                    "time_end": time_end,
                    "time_center": float(time_center),
                    "duration": float(duration),
                }

                if cube is not None and is_zero_flux_candidate(det, cube, zero_frames=zero_frames):
                    snr_w[t0, y0:y1, x0:x1] = -np.inf
                    continue

                detections.append(det)

                # suppress neighborhood in current time
                snr_w[t0, y0:y1, x0:x1] = -np.inf
                if occ_time is not None:
                    for dt_step in range(-time_radius, time_radius + 1):
                        t2 = t0 + dt_step
                        if 0 <= t2 < T_eff:
                            snr_w[t2, y0:y1, x0:x1] = -np.inf
                            occ_time[t2][y0:y1, x0:x1] = False

        detections_by_width[w] = detections

    return detections_by_width


def group_filter_across_widths(
    detections_by_width: Dict[int, List[Dict[str, Any]]],
    times: np.ndarray,
    *,
    spatial_radius: int = 3,
    time_radius: int = 8,                # in samples of center_idx
    policy: str = "max_snr",             # "max_snr" | "prefer_short" | "prefer_long"
    max_per_time_group: Optional[int] = None,  # kept for API parity; rarely needed here
    ny_nx: Optional[Tuple[int, int]] = None,
    cube: Optional[np.ndarray] = None,   # (T, Ny, Nx)
) -> List[Dict[str, Any]]:

    # 1) Flatten all detections across widths
    zero_frames = np.all(cube == 0.0, axis=(1, 2)) if cube is not None else None
    all_dets: List[Dict[str, Any]] = []
    for dets in detections_by_width.values():
        if cube is not None:
            dets = [d for d in dets if not is_zero_flux_candidate(d, cube, zero_frames=zero_frames)]
        all_dets.extend(dets)
    if not all_dets:
        return []

    # 2) Determine Ny, Nx
    if ny_nx is None:
        max_y = max(int(d["y"]) for d in all_dets)
        max_x = max(int(d["x"]) for d in all_dets)
        Ny, Nx = max_y + 1, max_x + 1
    else:
        Ny, Nx = ny_nx

    # 3) Sort globally by policy (SNR-first)
    def sort_key(d):
        if policy == "max_snr":
            return (-float(d["snr"]),)
        elif policy == "prefer_short":
            return (int(d["width_samples"]), -float(d["snr"]))
        elif policy == "prefer_long":
            return (-int(d["width_samples"]), -float(d["snr"]))
        else:
            raise ValueError(f"Unknown policy='{policy}'")
    all_dets.sort(key=sort_key)

    # 4) Occupancy across time centers (lazy-allocated per center_idx)
    occ_by_center: Dict[int, np.ndarray] = {}
    final: List[Dict[str, Any]] = []

    # (Optional) enforce at most K per exact center_idx
    kept_per_center: Dict[int, int] = {}

    for d in all_dets:
        ci = int(d["center_idx"])
        y  = int(d["y"]); x = int(d["x"])

        # Check availability across +/-time_radius
        available = True
        for dt in range(-time_radius, time_radius + 1):
            ci2 = ci + dt
            occ = occ_by_center.get(ci2)
            if occ is None:
                # not yet touched => all True
                continue
            if not occ[y, x]:
                available = False
                break
        if not available:
            continue

        # Respect optional per-time limit (usually None here)
        if (max_per_time_group is not None) and (kept_per_center.get(ci, 0) >= max_per_time_group):
            continue

        # Accept
        final.append(d)
        kept_per_center[ci] = kept_per_center.get(ci, 0) + 1

        # Suppress spatial neighborhood across +/-time_radius
        y0 = max(0, y - spatial_radius); y1 = min(Ny, y + spatial_radius + 1)
        x0 = max(0, x - spatial_radius); x1 = min(Nx, x + spatial_radius + 1)
        for dt_step in range(-time_radius, time_radius + 1):
            ci2 = ci + dt_step
            occ = occ_by_center.get(ci2)
            if occ is None:
                occ = np.ones((Ny, Nx), dtype=bool)
                occ_by_center[ci2] = occ
            occ[y0:y1, x0:x1] = False

    return final


def filter_ips_agn_candidates(
    candidates_list: List[Dict[str, Any]],
    dm0_candidates: List[Dict[str, Any]],
    racs_coords: Any,
    psrcat_coords: Any,
    *,
    ips_thresh: float = 0.8,
    ips_radius_arcsec: float = 30.0,
) -> List[Dict[str, Any]]:
    """
    Vectorized candidate filter suppressing DM > 0 candidates that match DM=0 candidates
    coincident with (RACS - PSRCAT) persistent continuum sources.
    """
    if not candidates_list or not dm0_candidates or racs_coords is None:
        return candidates_list

    from astropy.coordinates import SkyCoord
    from fastducc import catalogues

    # 1. Pre-filter and pre-compute coordinates for DM=0 candidates that match (RACS - PSRCAT)
    c0_ra_list = []
    c0_dec_list = []
    c0_snr_list = []
    for c0 in dm0_candidates:
        c0_ra = float(c0.get("ra_deg", np.degrees(c0.get("ra_rad", 0.0))))
        c0_dec = float(c0.get("dec_deg", np.degrees(c0.get("dec_rad", 0.0))))
        if catalogues.is_racs_minus_psrcat(c0_ra, c0_dec, racs_coords, psrcat_coords, match_radius_arcsec=ips_radius_arcsec):
            c0_ra_list.append(c0_ra)
            c0_dec_list.append(c0_dec)
            c0_snr_list.append(float(c0.get("snr", 0.0)))

    if not c0_ra_list:
        return candidates_list

    c0_sky = SkyCoord(c0_ra_list, c0_dec_list, unit="deg", frame="icrs")
    c0_snr_arr = np.array(c0_snr_list, dtype=np.float64)

    filtered = []
    for cand in candidates_list:
        c_ra = float(cand.get("ra_deg", np.degrees(cand.get("ra_rad", 0.0))))
        c_dec = float(cand.get("dec_deg", np.degrees(cand.get("dec_rad", 0.0))))
        c_coord = SkyCoord(c_ra, c_dec, unit="deg", frame="icrs")

        seps_arcsec = c_coord.separation(c0_sky).arcsec
        match_mask = seps_arcsec <= ips_radius_arcsec
        if np.any(match_mask):
            cand_snr = float(cand.get("snr", 0.0))
            if np.any(cand_snr <= ips_thresh * c0_snr_arr[match_mask]):
                continue  # Suppress IPS candidate
        filtered.append(cand)

    return filtered
