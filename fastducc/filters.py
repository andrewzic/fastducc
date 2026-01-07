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

from fastducc import kernels


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
        time_radius: int = 0,
        policy: str = "max_snr",
        max_per_time_group: Optional[int] = None,
        ny_nx: Optional[Tuple[int, int]] = None,
) -> List[Dict[str, Any]]:



        # Flatten by time center
    by_time: Dict[int, List[Dict[str, Any]]] = {}
    for w, dets in detections_by_width.items():
        for d in dets:
            ci = int(d["center_idx"])
            by_time.setdefault(ci, []).append(d)

    # If temporal NMS is requested, prepare a global occupancy structure keyed by center_idx
    occ_by_center: Dict[int, np.ndarray] = {}

    # If Ny,Nx not provided, infer from detections
    if ny_nx is None:
        max_y = -1
        max_x = -1
        for dets in by_time.values():
            for d in dets:
                if d["y"] > max_y: max_y = d["y"]
                if d["x"] > max_x: max_x = d["x"]
        Ny = max_y + 1 if max_y >= 0 else 1
        Nx = max_x + 1 if max_x >= 0 else 1
    else:
        Ny, Nx = ny_nx

    final: List[Dict[str, Any]] = []

    # Helper to get sorted order per policy
    def sort_dets(dets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if policy == "max_snr":
            return sorted(dets, key=lambda d: d["snr"], reverse=True)
        elif policy == "prefer_short":
            return sorted(dets, key=lambda d: (d["width_samples"], -d["snr"]))
        elif policy == "prefer_long":
            return sorted(dets, key=lambda d: (-d["width_samples"], -d["snr"]))
        else:
            raise ValueError(f"Unknown policy='{policy}'")

    # Process each time-center group independently
    for center_idx, dets in by_time.items():
        if len(dets) == 0:
            continue

        # Initialize occupancy for this time center if needed
        if center_idx not in occ_by_center:
            occ_by_center[center_idx] = np.ones((Ny, Nx), dtype=bool)

        # Sort by policy
        dets_sorted = sort_dets(dets)

        kept = 0
        for d in dets_sorted:
            y = int(d["y"]); x = int(d["x"])
            # If spatial spot already suppressed for this center, skip
            if not occ_by_center[center_idx][y, x]:
                continue

            # Accept detection
            final.append(d)

            # Suppress neighborhood for this center
            y0 = max(0, y - spatial_radius)
            y1 = min(Ny, y + spatial_radius + 1)
            x0 = max(0, x - spatial_radius)
            x1 = min(Nx, x + spatial_radius + 1)
            occ_by_center[center_idx][y0:y1, x0:x1] = False

            # Optional temporal NMS: suppress the same spatial neighborhood in nearby center_idx groups
            if time_radius > 0:
                for dt in range(-time_radius, time_radius + 1):
                    ci2 = center_idx + dt
                    if ci2 == center_idx:
                        continue
                    if ci2 in by_time:
                        if ci2 not in occ_by_center:
                            occ_by_center[ci2] = np.ones((Ny, Nx), dtype=bool)
                        occ_by_center[ci2][y0:y1, x0:x1] = False

            kept += 1
            if (max_per_time_group is not None) and (kept >= max_per_time_group):
                break

    return final
