from typing import Tuple, List, Dict, Any, Optional
import numpy as np

from scipy.ndimage import maximum_filter

try:
    import ducc0
except Exception as e:
    raise RuntimeError('ducc0 is required') from e

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
        det.setdefault("t0_idx", 0)
        det.setdefault("t1_idx_excl", 0)
        det.setdefault("center_idx", 0)
        det.setdefault("time_center", 0.0)

        # Optional: derive a representative time from the full cube
        if (time_tag_policy != "none") and (times is not None) and (cube is not None):
            lc = cube[:, y, x].astype(np.float64, copy=False)
            if time_tag_policy == "peak_absdev":
                mu = np.nanmean(lc)
                k = int(np.nanargmax(np.abs(lc - mu)))
            elif time_tag_policy == "peak_flux":
                k = int(np.nanargmax(lc))
            else:
                k = None
            if k is not None:
                det["center_idx"] = k
                det["time_center"] = float(times[k])

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
    valid_mask: Optional[np.ndarray] = None
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Spatial (and optional temporal) NMS using compiled 2D max-filter for peaks.
    """
    detections_by_width: Dict[int, List[Dict[str, Any]]] = {}

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
                for dt in range(-time_radius, time_radius + 1):
                    t2 = t0 + dt
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
                
                time_start = float(times[t0])
                time_end = float(times[t1 - 1])
                
                center_idx  = t0 + (w // 2)
                time_center = float(times[center_idx])

                # time_center = 0.5 * (time_start + time_end)

                # k = np.searchsorted(times[t0:t1], time_center)
                # if k == 0:
                #     center_idx = t0
                # elif k >= (t1 - t0):
                #     center_idx = t1 - 1
                # else:
                #     left = t0 + (k - 1)
                #     right = t0 + k
                #     center_idx = left if abs(times[left] - time_center) <= abs(times[right] - time_center) else right

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
                }
                detections.append(det)

                # suppress neighborhood in current time
                snr_w[t0, y0:y1, x0:x1] = -np.inf
                if occ_time is not None:
                    for dt in range(-time_radius, time_radius + 1):
                        t2 = t0 + dt
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
) -> List[Dict[str, Any]]:

    # 1) Flatten all detections across widths
    all_dets: List[Dict[str, Any]] = []
    for dets in detections_by_width.values():
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
        for dt in range(-time_radius, time_radius + 1):
            ci2 = ci + dt
            occ = occ_by_center.get(ci2)
            if occ is None:
                occ = np.ones((Ny, Nx), dtype=bool)
                occ_by_center[ci2] = occ
            occ[y0:y1, x0:x1] = False

    return final
