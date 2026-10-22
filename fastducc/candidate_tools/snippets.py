# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Dict, Any, Optional

import numpy as np

from fastducc.filters import is_zero_flux_candidate
from fastducc import kernels


def make_stdmap_snippet(
    std_map: np.ndarray,                # (Ny, Nx)
    candidate: Dict[str, Any],
    spatial_size: int = 50
) -> Dict[str, Any]:
    """
    Create a snippet record from a 2-D variance/std map.
    Time length is 1 (single frame), with NaN time stamp for compatibility.
    """
    Ny, Nx = std_map.shape
    y = int(candidate["y"]); x = int(candidate["x"])
    half_sp = spatial_size // 2

    y0 = max(0, y - half_sp)
    y1 = min(Ny, y0 + spatial_size)
    x0 = max(0, x - half_sp)
    x1 = min(Nx, x0 + spatial_size)

    cut = std_map[y0:y1, x0:x1]
    pad_y_top = max(0, (y - half_sp) - y0)
    pad_y_bot = max(0, (y0 + spatial_size) - y1)
    pad_x_left = max(0, (x - half_sp) - x0)
    pad_x_right = max(0, (x0 + spatial_size) - x1)

    snippet_2d = np.pad(cut,
                        ((pad_y_top, pad_y_bot), (pad_x_left, pad_x_right)),
                        mode="edge")
    snippet_cube = snippet_2d[None, :, :]  # (1, spatial_size, spatial_size)
    snippet_times = np.array([np.nan], dtype=float)

    rec = {
        "candidate": candidate,
        "snippet_cube": snippet_cube.astype(np.float32),
        "snippet_times": snippet_times,
        "meta": {
            "time_indices": {"desired": (0, 1), "clipped": (0, 1), "pad": (0, 0)},
            "y_indices":    {"desired": (y - half_sp, y - half_sp + spatial_size),
                             "clipped": (y0, y1), "pad": (pad_y_top, pad_y_bot)},
            "x_indices":    {"desired": (x - half_sp, x - half_sp + spatial_size),
                             "clipped": (x0, x1), "pad": (pad_x_left, pad_x_right)},
            "center_idx": 0,
            "snippet_shape": (1, spatial_size, spatial_size),
        }
    }
    return rec


def extract_candidate_snippets(
    times: np.ndarray,
    cube: np.ndarray,
    detections: List[Dict[str, Any]],
    *,
    spatial_size: int = 50,
    time_factor: int = 25,            # in units of *smoothed* frames
    pad_mode: str = "constant",
    pad_value: float = 0.0,
    return_indices: bool = True,
    center_policy: str = "right",    # 'right' uses w//2, 'left' uses (w-1)//2 for center time
) -> List[Dict[str, Any]]:
    """
    Extract candidate snippets from boxcar-smoothed cube,
    using the candidate width as the temporal averaging window.
    """
    if cube.ndim != 3:
        raise ValueError("cube must be (T, Ny, Nx)")
    T, Ny, Nx = cube.shape
    if times.shape[0] != T:
        raise ValueError("times length must match cube time axis")
    if spatial_size < 1 or time_factor < 1:
        raise ValueError("spatial_size and time_factor must be >= 1")

    detections = [d for d in detections if not is_zero_flux_candidate(d, cube)]
    if not detections:
        return []

    half_sp = spatial_size // 2
    t_len_sm = max(1, time_factor)
    half_t = t_len_sm // 2

    # Precompute smoothed times per unique width (avoids full-cube 3D smoothing)
    unique_w = sorted(set(max(1, int(d.get("width_samples", 1))) for d in detections))
    times_by_w: Dict[int, np.ndarray] = {}

    for w in unique_w:
        if w > T:
            sm_times = np.empty((0,), dtype=times.dtype)
        else:
            T_eff = T - w + 1
            offset = (w // 2) if (center_policy == "right") else ((w - 1) // 2)
            sm_times = times[offset : offset + T_eff]
        times_by_w[w] = sm_times

    out: List[Dict[str, Any]] = []

    def _resolve_w(det: Dict[str, Any]) -> int:
        return max(1, int(det.get("width_samples", 1)))

    def _resolve_sm_center_idx(det: Dict[str, Any], w: int) -> int:
        if "t0_idx" in det:
            return int(det["t0_idx"])
        c = int(det.get("center_idx", 0))
        if center_policy == "right":
            return max(0, c - (w // 2))
        else:
            return max(0, c - ((w - 1) // 2))

    for det in detections:
        y = int(det["y"]); x = int(det["x"])
        w = _resolve_w(det)

        sm_times = times_by_w[w]
        T_eff = sm_times.shape[0]

        if T_eff == 0:
            snippet_cube = np.full((t_len_sm, spatial_size, spatial_size), np.nan, dtype=np.float32)
            snippet_times = np.full((t_len_sm,), np.nan, dtype=times.dtype)
            rec = {"candidate": det, "snippet_cube": snippet_cube, "snippet_times": snippet_times}
            if return_indices:
                rec["meta"] = {
                    "time_indices": {"desired": (0, t_len_sm), "clipped": (0, 0), "pad": (t_len_sm, 0)},
                    "y_indices": {"desired": (y - half_sp, y - half_sp + spatial_size),
                                  "clipped": (0, 0), "pad": (spatial_size, 0)},
                    "x_indices": {"desired": (x - half_sp, x - half_sp + spatial_size),
                                  "clipped": (0, 0), "pad": (spatial_size, 0)},
                    "center_idx_smoothed": 0,
                    "snippet_shape": (t_len_sm, spatial_size, spatial_size),
                }
            out.append(rec)
            continue

        k_center = _resolve_sm_center_idx(det, w)

        k0 = k_center - half_t
        k1 = k0 + t_len_sm

        k0_clip = max(0, k0)
        k1_clip = min(T_eff, k1)

        y0 = y - half_sp
        y1 = y0 + spatial_size
        x0 = x - half_sp
        x1 = x0 + spatial_size

        y0_clip = max(0, y0)
        y1_clip = min(Ny, y1)
        x0_clip = max(0, x0)
        x1_clip = min(Nx, x1)

        pad_t_front = k0_clip - k0
        pad_t_back  = k1 - k1_clip
        pad_y_top   = y0_clip - y0
        pad_y_bot   = y1 - y1_clip
        pad_x_left  = x0_clip - x0
        pad_x_right = x1 - x1_clip

        n_sm_frames = k1_clip - k0_clip
        if n_sm_frames > 0:
            raw_t0 = k0_clip
            raw_t1 = k1_clip + w - 1
            raw_cutout = cube[raw_t0:raw_t1, y0_clip:y1_clip, x0_clip:x1_clip].astype(np.float64, copy=False)
            csum_cut = np.zeros((raw_cutout.shape[0] + 1, raw_cutout.shape[1], raw_cutout.shape[2]), dtype=np.float64)
            csum_cut[1:] = np.cumsum(raw_cutout, axis=0)
            sub_cube = (csum_cut[w : n_sm_frames + w] - csum_cut[:n_sm_frames]) / float(w)
        else:
            sub_cube = np.empty((0, y1_clip - y0_clip, x1_clip - x0_clip), dtype=np.float64)

        sub_times = sm_times[k0_clip:k1_clip]

        pad_widths = (
            (pad_t_front, pad_t_back),
            (pad_y_top,   pad_y_bot),
            (pad_x_left,  pad_x_right),
        )

        if pad_mode == "constant":
            snippet_cube = np.pad(sub_cube, pad_widths, mode="constant", constant_values=pad_value)
            snippet_times = np.pad(sub_times, (pad_t_front, pad_t_back),
                                   mode="constant", constant_values=np.nan)
        elif pad_mode == "edge":
            snippet_cube = np.pad(sub_cube, pad_widths, mode="edge")
            if sub_times.size == 0:
                snippet_times = np.full((t_len_sm,), np.nan, dtype=times.dtype)
            else:
                front_vals = np.full((pad_t_front,), sub_times[0], dtype=sub_times.dtype)
                back_vals  = np.full((pad_t_back,),  sub_times[-1], dtype=sub_times.dtype)
                snippet_times = np.concatenate([front_vals, sub_times, back_vals], axis=0)
        else:
            raise ValueError(f"Unknown pad_mode='{pad_mode}'")

        if snippet_cube.shape != (t_len_sm, spatial_size, spatial_size):
            raise RuntimeError(f"Snippet cube has unexpected shape {snippet_cube.shape}")
        if snippet_times.shape != (t_len_sm,):
            raise RuntimeError(f"Snippet times has unexpected shape {snippet_times.shape}")

        lc = snippet_cube[:, half_sp, half_sp]
        good = np.isfinite(snippet_times) & np.isfinite(lc)
        if np.any(good):
            rel_peak_all = int(np.nanargmax(lc))
            det_time_peak = snippet_times[rel_peak_all]
            det["time_center_peak_smoothed"] = float(det_time_peak) if np.isfinite(det_time_peak) else float(det.get("time_center", np.nan))
            det["center_idx_peak_smoothed"]  = int(k_center + (rel_peak_all - half_t))

        det["time_center_smoothed"] = float(sm_times[k_center]) if (0 <= k_center < sm_times.shape[0]) else float(det.get("time_center", np.nan))
        det["center_idx_smoothed"]  = int(k_center)

        rec: Dict[str, Any] = {
            "candidate": det,
            "snippet_cube": snippet_cube.astype(np.float32, copy=False),
            "snippet_times": snippet_times,
        }

        if return_indices:
            rec["meta"] = {
                "time_indices": {"desired": (k0, k1), "clipped": (k0_clip, k1_clip), "pad": (pad_t_front, pad_t_back)},
                "y_indices":    {"desired": (y0, y1), "clipped": (y0_clip, y1_clip), "pad": (pad_y_top, pad_y_bot)},
                "x_indices":    {"desired": (x0, x1), "clipped": (x0_clip, x1_clip), "pad": (pad_x_left, pad_x_right)},
                "center_idx_smoothed": int(k_center),
                "snippet_shape": (t_len_sm, spatial_size, spatial_size),
                "width_samples": int(w),
            }

        out.append(rec)

    return out


def compute_candidate_local_rms(
    cube: np.ndarray,
    candidate: Dict[str, Any],
    *,
    window_size: int = 64,
    inner_radius: float = 3.0,
    clip_sigma: float = 3.0,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute per-time-slice local off-source RMS noise around a candidate."""
    T, Ny, Nx = cube.shape
    y = int(candidate["y"])
    x = int(candidate["x"])

    hw = max(4, int(window_size)) // 2
    y0 = max(0, y - hw)
    y1 = min(Ny, y + hw)
    x0 = max(0, x - hw)
    x1 = min(Nx, x + hw)
    H = y1 - y0
    W = x1 - x0

    subcube = np.ascontiguousarray(cube[:, y0:y1, x0:x1], dtype=np.float64)

    cy = y - y0
    cx = x - x0
    loc_mask = np.ones((H, W), dtype=bool)
    if valid_mask is not None:
        loc_mask &= valid_mask[y0:y1, x0:x1]

    if inner_radius > 0:
        yy, xx = np.ogrid[:H, :W]
        dist_sq = (yy - cy) ** 2 + (xx - cx) ** 2
        off_source = dist_sq > (inner_radius ** 2)
        if np.count_nonzero(loc_mask & off_source) >= 8:
            loc_mask &= off_source
        elif np.count_nonzero(loc_mask) > 1:
            loc_mask[cy, cx] = False

    if np.count_nonzero(loc_mask) == 0:
        loc_mask = np.ones((H, W), dtype=bool)

    rms = kernels._clipped_rms_2d_per_time(subcube, loc_mask, float(clip_sigma))
    cand_lc = cube[:, y, x]
    zero_or_edge = (cand_lc == 0.0) | (~np.isfinite(cand_lc))
    bad = (~np.isfinite(rms)) | (rms <= 0.0) | zero_or_edge
    if np.any(bad):
        rms[bad] = 0.0

    return rms


def save_candidate_lightcurves(
    times: np.ndarray,
    cube: np.ndarray,
    candidate: Dict[str, Any],
    out_prefix: str,
    *,
    save_format: str = "npz",
    center_policy: str = "right",
    local_window_size: int = 64,
    rms_full: Optional[np.ndarray] = None,
) -> Dict[str, str]:
    """
    Extract and save full-resolution and boxcar-smoothed light curves for a candidate,
    including per-frame local off-source RMS errorbars.
    Returns dict of file paths: {"lc_full": out_full, "lc_boxcar": out_box}.
    """
    if cube.ndim != 3:
        raise ValueError("cube must be (T, Ny, Nx)")
    T, Ny, Nx = cube.shape
    if times.shape[0] != T:
        raise ValueError("times length must match cube time axis")

    if is_zero_flux_candidate(candidate, cube):
        print(f"[Warning] Skipping lightcurves for zero-flux candidate {candidate.get('srcname', '')}")
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

    lc_full = cube[:, y, x].astype(np.float64, copy=False)

    if rms_full is None:
        rms_full = compute_candidate_local_rms(cube, candidate, window_size=local_window_size)
    else:
        rms_full = np.asarray(rms_full, dtype=np.float64).copy()
    bad_rms = (~np.isfinite(rms_full)) | (rms_full <= 0.0) | (lc_full == 0.0) | (~np.isfinite(lc_full))
    if np.any(bad_rms):
        rms_full[bad_rms] = 0.0

    lc_sm, T_eff = kernels._compute_boxcar_1d(lc_full, w)
    if T_eff > 0:
        times_sm = times[offset:offset + T_eff]
        csum_var = np.zeros(T + 1, dtype=np.float64)
        csum_var[1:] = np.cumsum(rms_full ** 2)
        rms_sm = np.sqrt(np.maximum(0.0, csum_var[w:] - csum_var[:-w])) / float(w)
    else:
        times_sm = np.empty((0,), dtype=times.dtype)
        rms_sm = np.empty((0,), dtype=np.float64)

    out_full = f"{out_prefix}_lc_full.{ 'npz' if save_format=='npz' else 'txt' }"
    out_box  = f"{out_prefix}_lc_boxcar_w{w}.{ 'npz' if save_format=='npz' else 'txt' }"

    if save_format == "npz":
        np.savez(out_full, time=times, flux=lc_full, flux_err=rms_full, error=rms_full)
        np.savez(out_box, time=times_sm, flux=lc_sm, flux_err=rms_sm, error=rms_sm, width=w)
    else:
        with open(out_full, "w") as f:
            f.write("# time  flux(full_res)  flux_err\n")
            for t, v, e in zip(times, lc_full, rms_full):
                f.write(f"{t:.9f} {v:.9e} {e:.9e}\n")
        with open(out_box, "w") as f:
            f.write(f"# time  flux(boxcar_mean_w={w})  flux_err\n")
            for t, v, e in zip(times_sm, lc_sm, rms_sm):
                f.write(f"{t:.9f} {v:.9e} {e:.9e}\n")

    return {"lc_full": out_full, "lc_boxcar": out_box}
