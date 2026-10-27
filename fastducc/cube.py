# -*- coding: utf-8 -*-
"""
fastducc.cube: Distributed and asynchronous utility to image MS time samples
using ducc0.wgridder, imbue each plane with WCS and temporal metadata, and
stream them directly into an output FITS cube without intermediate disk frames.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import inspect
import logging
import math
import os
import shutil
import subprocess
import tempfile
from typing import List, Tuple, Any, Optional, Dict, Union
import warnings

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from astropy.time import Time

logger = logging.getLogger(__name__)
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, total=None, desc=None, unit=None, **kwargs):
        class SimplePbar:
            def __init__(self, total): self.total = total
            def update(self, n=1): pass
            def close(self): pass
            def __iter__(self): return iter(iterable) if iterable else iter([])
        return iterable if iterable is not None else SimplePbar(total)

try:
    from casacore.tables import table
except ImportError:
    table = None

try:
    import ducc0
except ImportError:
    ducc0 = None

from fastducc.fd_types import CubeConfig
from fastducc import ms_utils, imaging, dedisp, timeshift
from fastducc import wcs as ducc_wcs


# =========================================================================
# Coordinate & Geometry Parsing
# =========================================================================

def parse_coordinate(
    coord: Optional[Any] = None,
    ra: Optional[Any] = None,
    dec: Optional[Any] = None,
) -> Optional[SkyCoord]:
    """Parse flexible celestial coordinates into an astropy SkyCoord."""
    if coord is None and ra is None and dec is None:
        return None
    if isinstance(coord, SkyCoord):
        return coord

    if coord is not None:
        if isinstance(coord, (tuple, list)) and len(coord) == 2:
            ra, dec = coord[0], coord[1]
        else:
            s = str(coord).strip()
            if "," in s:
                parts = [p.strip() for p in s.split(",", 1)]
                ra, dec = parts[0], parts[1]
            else:
                parts = s.split()
                if len(parts) == 6:
                    ra, dec = " ".join(parts[:3]), " ".join(parts[3:])
                elif len(parts) == 2:
                    ra, dec = parts[0], parts[1]
                else:
                    ra, dec = s, None

    if dec is None and ra is not None:
        for unit_opt in (None, (u.hourangle, u.deg), u.deg):
            try:
                return SkyCoord(ra, unit=unit_opt) if unit_opt else SkyCoord(ra)
            except Exception:
                continue
        raise ValueError(f"Could not parse single coordinate string '{ra}'")

    if isinstance(ra, (int, float, np.number)) and isinstance(dec, (int, float, np.number)):
        return SkyCoord(ra=float(ra) * u.deg, dec=float(dec) * u.deg)

    ra_s, dec_s = str(ra).strip(), str(dec).strip()
    if ":" in ra_s or "h" in ra_s.lower():
        return SkyCoord(ra_s, dec_s, unit=(u.hourangle, u.deg))
    elif "deg" in ra_s.lower() or "d" in dec_s.lower():
        return SkyCoord(ra_s, dec_s)

    try:
        return SkyCoord(ra=float(ra_s) * u.deg, dec=float(dec_s) * u.deg)
    except ValueError:
        return SkyCoord(ra_s, dec_s, unit=(u.hourangle, u.deg))


def parse_radius(
    radius: Optional[Any] = None,
    radius_arcsec: Optional[float] = None,
    radius_pix: Optional[float] = None,
    box_size_pix: Optional[int] = None,
    box_size_arcsec: Optional[float] = None,
    pixsize_arcsec: float = 22.0,
    default_unit: str = "arcsec",
) -> Tuple[Optional[float], Optional[int], Optional[int]]:
    """Parse radius or box size into (radius_arcsec, radius_pix, box_size_pix)."""
    if box_size_pix is not None:
        b = int(box_size_pix)
        r = max(1, int(round(b / 2.0)))
        return r * pixsize_arcsec, r, b

    if box_size_arcsec is not None:
        b_sec = float(box_size_arcsec)
        r_sec = b_sec / 2.0
        return r_sec, max(1, int(round(r_sec / pixsize_arcsec))), max(1, int(round(b_sec / pixsize_arcsec)))

    if radius_pix is not None:
        r = max(1, int(round(radius_pix)))
        return r * pixsize_arcsec, r, 2 * r

    if radius_arcsec is not None:
        r_sec = float(radius_arcsec)
        return r_sec, max(1, int(round(r_sec / pixsize_arcsec))), 2 * max(1, int(round(r_sec / pixsize_arcsec)))

    if radius is not None:
        s = str(radius).strip().lower()
        if "pix" in s or "px" in s:
            val = float(s.replace("pix", "").replace("px", "").strip())
            r = max(1, int(round(val)))
            return r * pixsize_arcsec, r, 2 * r

        unit_map = [
            ("arcmin", 60.0), ("amin", 60.0), ("'", 60.0),
            ("deg", 3600.0), ("d", 3600.0),
            ("arcsec", 1.0), ("asec", 1.0), ("as", 1.0), ('"', 1.0),
        ]
        for suffix, scale in unit_map:
            if s.endswith(suffix):
                val = float(s[:-len(suffix)].strip()) * scale
                r = max(1, int(round(val / pixsize_arcsec)))
                return val, r, 2 * r

        val = float(s)
        if default_unit == "pix":
            r = max(1, int(round(val)))
            return r * pixsize_arcsec, r, 2 * r
        r = max(1, int(round(val / pixsize_arcsec)))
        return val, r, 2 * r

    return None, None, None


def compute_cubelet_geometry(
    target_coord: SkyCoord,
    radius_arcsec: float,
    radius_pix: int,
    box_size_pix: Optional[int],
    ra0_rad: float,
    dec0_rad: float,
    pixsize_arcsec: float,
    default_npix_x: int,
    default_npix_y: int,
    ra_sign: int = -1,
    dec_sign: int = -1,
) -> Dict[str, Any]:
    """Calculate grid dimensions, full WCS, bounding box, and cutout bounds for cubelet imaging."""
    pix_rad = pixsize_arcsec / 206265.0
    pixscale_deg = pixsize_arcsec / 3600.0
    ra0_deg, dec0_deg = math.degrees(ra0_rad), math.degrees(dec0_rad)

    phase_coord = SkyCoord(ra=ra0_deg * u.deg, dec=dec0_deg * u.deg)
    sep_deg = float(target_coord.separation(phase_coord).deg)
    if sep_deg > 15.0:
        warnings.warn(f"Target coordinate is {sep_deg:.2f} deg from phase center (potential gnomonic distortion).")

    dx_deg = (target_coord.ra.deg - ra0_deg) * math.cos(dec0_rad)
    dy_deg = target_coord.dec.deg - dec0_deg
    delta_x = (-dx_deg / pixscale_deg) if ra_sign < 0 else (dx_deg / pixscale_deg)
    delta_y = (dy_deg / pixscale_deg) if dec_sign > 0 else (-dy_deg / pixscale_deg)

    nx_cut = int(box_size_pix) if box_size_pix and box_size_pix > 0 else 2 * int(radius_pix)
    ny_cut = nx_cut
    half_x, half_y = nx_cut / 2.0, ny_cut / 2.0

    req_x = max(default_npix_x, int(math.ceil(2 * (abs(delta_x) + half_x) + 32)))
    req_y = max(default_npix_y, int(math.ceil(2 * (abs(delta_y) + half_y) + 32)))
    grid_npix_x = ((req_x + 31) // 32) * 32
    grid_npix_y = ((req_y + 31) // 32) * 32

    wcs_full = ducc_wcs._build_fullframe_wcs(
        npix_x=grid_npix_x, npix_y=grid_npix_y, ra0_rad=ra0_rad, dec0_rad=dec0_rad,
        pixscale_rad=pix_rad, ra_sign=ra_sign, dec_sign=dec_sign, radesys="ICRS",
    )
    x_target, y_target = wcs_full.world_to_pixel_values(target_coord.ra.deg, target_coord.dec.deg)
    xmin = max(0, min(grid_npix_x - nx_cut, int(round(x_target - half_x))))
    ymin = max(0, min(grid_npix_y - ny_cut, int(round(y_target - half_y))))
    xmax, ymax = xmin + nx_cut, ymin + ny_cut

    return {
        "target_coord": target_coord, "radius_arcsec": radius_arcsec, "radius_pix": radius_pix,
        "nx_cut": nx_cut, "ny_cut": ny_cut, "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax,
        "bbox": (ymin, ymax, xmin, xmax), "grid_npix_x": grid_npix_x, "grid_npix_y": grid_npix_y,
        "wcs_full": wcs_full, "x_target": float(x_target), "y_target": float(y_target),
    }


# =========================================================================
# Time Axis Analysis & FITS Tables
# =========================================================================

def analyze_time_axis(
    times: np.ndarray,
    create_blanks: bool = False,
    dt_nominal: Optional[float] = None,
    tolerance: float = 1e-3,
    max_blank_factor: float = 10.0,
) -> Dict[str, Any]:
    """Analyze observation timestamps for regular/irregular spacing and missing gaps."""
    n_times = len(times)
    if n_times == 0:
        raise ValueError("No timestamps provided to analyze_time_axis.")

    if n_times == 1:
        dt = 1.0 if dt_nominal is None else float(dt_nominal)
        return {
            "spacing_type": "single", "is_regular": True, "has_gaps": False, "dt_nominal": dt,
            "total_planes": 1, "grid_indices": np.array([0], dtype=int), "grid_times": times.copy(),
            "blank_indices": [], "is_blank": np.array([False]), "delta_t": np.array([dt], dtype=np.float32),
        }

    diffs = np.diff(times)
    pos_diffs = diffs[diffs > 1e-7]
    if len(pos_diffs) == 0:
        dt = 1.0 if dt_nominal is None else float(dt_nominal)
    elif dt_nominal is not None:
        dt = float(dt_nominal)
    else:
        min_dt = float(np.min(pos_diffs))
        mult_min = pos_diffs / min_dt
        dt = min_dt if np.allclose(mult_min, np.round(mult_min), rtol=tolerance, atol=tolerance * min_dt) else float(np.median(pos_diffs))

    is_strictly_regular = np.allclose(diffs, dt, rtol=tolerance, atol=tolerance * dt)
    multiples = diffs / dt
    is_lattice = np.allclose(multiples, np.round(multiples), rtol=tolerance, atol=tolerance) and np.all(np.round(multiples) >= 1)

    spacing_type = "regular" if is_strictly_regular else ("regular_with_gaps" if is_lattice else "irregular")
    is_regular = is_strictly_regular or is_lattice
    has_gaps = is_lattice and not is_strictly_regular

    t_start, t_end = float(times[0]), float(times[-1])
    can_create_blanks = create_blanks and is_lattice
    if can_create_blanks:
        grid_n = int(np.round((t_end - t_start) / dt)) + 1
        if (grid_n / n_times) > max_blank_factor:
            warnings.warn(f"Time grid expansion ({grid_n/n_times:.1f}x) exceeds --max-blank-factor. Disabling --create-blanks.")
            can_create_blanks = False

    if can_create_blanks:
        grid_indices = np.round((times - t_start) / dt).astype(int)
        grid_n = max(grid_n, int(np.max(grid_indices)) + 1)
        blank_indices = sorted(list(set(range(grid_n)) - set(grid_indices)))
        is_blank = np.zeros(grid_n, dtype=bool)
        is_blank[blank_indices] = True
        grid_times = t_start + np.arange(grid_n) * dt
        total_planes = grid_n
        delta_t = np.full(grid_n, dt, dtype=np.float32)
    else:
        grid_indices = np.arange(n_times, dtype=int)
        blank_indices = []
        is_blank = np.zeros(n_times, dtype=bool)
        grid_times = times.copy()
        total_planes = n_times
        delta_t = np.empty(n_times, dtype=np.float32)
        delta_t[0] = dt
        delta_t[1:] = diffs.astype(np.float32)

    return {
        "spacing_type": spacing_type, "is_regular": is_regular, "has_gaps": has_gaps,
        "dt_nominal": dt, "total_planes": total_planes, "grid_indices": grid_indices,
        "grid_times": grid_times, "blank_indices": blank_indices, "is_blank": is_blank,
        "delta_t": delta_t,
    }


def create_timestamps_table_hdu(
    times: np.ndarray,
    is_blank: Optional[np.ndarray] = None,
    delta_t: Optional[np.ndarray] = None,
    dt_nominal: Optional[float] = None,
) -> fits.BinTableHDU:
    """Create a standard TIMESTAMPS FITS binary table HDU."""
    n_rows = len(times)
    sec_arr = (times * 86400.0 if n_rows > 0 and times[0] < 1e6 else times).astype(np.float64)
    mjd_arr = sec_arr / 86400.0

    date_obs_list = []
    for m in mjd_arr:
        try: date_obs_list.append(Time(float(m), format="mjd", scale="utc").isot)
        except Exception: date_obs_list.append("")

    delta_t = np.full(n_rows, dt_nominal or 1.0, dtype=np.float32) if delta_t is None else delta_t
    is_blank = np.zeros(n_rows, dtype=bool) if is_blank is None else is_blank

    cols = [
        fits.Column(name="SLICE", format="1J", array=np.arange(n_rows, dtype=np.int32)),
        fits.Column(name="TIME", format="1D", unit="s", array=sec_arr),
        fits.Column(name="MJD", format="1D", unit="d", array=mjd_arr),
        fits.Column(name="DATE_OBS", format="32A", unit="ISO-8601", array=np.array(date_obs_list, dtype="U32")),
        fits.Column(name="DELTA_T", format="1E", unit="s", array=delta_t.astype(np.float32)),
        fits.Column(name="IS_BLANK", format="1L", array=is_blank.astype(bool)),
    ]
    table_hdu = fits.BinTableHDU.from_columns(cols, name="TIMESTAMPS")
    table_hdu.header["TIMESYS"] = "UTC"
    table_hdu.header["TIMEUNIT"] = "s"
    return table_hdu


def append_timestamps_extension(out_cube: str, table_hdu: fits.BinTableHDU) -> None:
    """Append or update the TIMESTAMPS binary table HDU in a FITS cube."""
    with fits.open(out_cube, mode="update") as hdul:
        if "TIMESTAMPS" in hdul:
            hdul["TIMESTAMPS"] = table_hdu
        else:
            hdul.append(table_hdu)


# =========================================================================
# FITS Pre-Allocation & UV Headers
# =========================================================================

def init_empty_fits_cube(
    out_cube: str,
    header: fits.Header,
    shape: Tuple[int, ...],
    dtype: np.dtype = np.float32,
    overwrite: bool = True,
    blank_indices: Optional[List[int]] = None,
) -> int:
    """Pre-allocate a FITS cube directly on disk with strict 2880-byte padding."""
    if os.path.exists(out_cube) and not overwrite:
        raise FileExistsError(f"Output cube '{out_cube}' already exists and overwrite is False.")

    parent_dir = os.path.dirname(os.path.abspath(out_cube))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    naxis = len(shape)
    if naxis == 3:
        nt, ny, nx = shape
        npol = None
    elif naxis == 4:
        npol, nt, ny, nx = shape
    else:
        raise ValueError(f"Unsupported shape {shape} (expected 3D or 4D).")

    bitpix = -32 if np.dtype(dtype) == np.float32 else -64
    bytes_per_pix = 4 if bitpix == -32 else 8

    clean_hdr = fits.Header()
    clean_hdr.set("SIMPLE", True)
    clean_hdr.set("BITPIX", bitpix)
    clean_hdr.set("NAXIS", naxis)
    clean_hdr.set("NAXIS1", nx)
    clean_hdr.set("NAXIS2", ny)
    clean_hdr.set("NAXIS3", nt)
    if naxis == 4:
        clean_hdr.set("NAXIS4", npol)
    clean_hdr.set("EXTEND", True)

    standard_keys = {"SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "NAXIS4", "EXTEND"}
    for card in header.cards:
        if card.keyword not in standard_keys:
            clean_hdr.append(card)

    clean_hdr.tofile(out_cube, overwrite=True)
    header_size = os.path.getsize(out_cube)
    data_bytes = int(np.prod(shape)) * bytes_per_pix
    pad_d = (2880 - (data_bytes % 2880)) % 2880
    total_size = header_size + data_bytes + pad_d

    with open(out_cube, "r+b") as f:
        f.seek(total_size - 1)
        f.write(b"\0")
        if blank_indices:
            plane_bytes = ny * nx * bytes_per_pix
            dtype_code = ">f4" if bytes_per_pix == 4 else ">f8"
            nan_plane = np.full((ny, nx), np.nan, dtype=dtype_code).tobytes()
            for b_idx in blank_indices:
                f.seek(header_size + int(b_idx) * plane_bytes)
                f.write(nan_plane)

    return header_size


def derive_uv_cube_names(
    out_cube: Optional[str],
    msname: str,
    out_uv_real: Optional[str] = None,
    out_uv_imag: Optional[str] = None,
    start_idx: int = 0,
    end_idx: int = 0,
) -> Tuple[str, str]:
    """Derive output file paths for real and imaginary UV cubes."""
    if out_uv_real and out_uv_imag:
        return out_uv_real, out_uv_imag
    if out_cube:
        base = out_cube[:-10] if out_cube.endswith(".cube.fits") else (out_cube[:-5] if out_cube.endswith(".fits") else out_cube)
    else:
        ms_base = os.path.basename(msname.rstrip("/")).replace(".ms", "")
        base = f"{ms_base}_t{start_idx:05d}_{end_idx:05d}"
    return out_uv_real or f"{base}-uv-real.cube.fits", out_uv_imag or f"{base}-uv-imag.cube.fits"


def build_uv_headers(
    header_base: fits.Header,
    npix_x: int,
    npix_y: int,
    pix_rad: float,
    total_planes: int,
    time_info: Dict[str, Any],
    as_4d: bool = False,
    hermitian_half: bool = False,
) -> Tuple[fits.Header, fits.Header]:
    """Construct standard FITS headers for gridded UV real/imag cubes directly from image header."""
    hdr_real = header_base.copy()
    hdr_real["NAXIS"] = 4 if as_4d else 3
    hdr_real["NAXIS1"] = npix_x
    hdr_real["NAXIS2"] = npix_y
    hdr_real["NAXIS3"] = total_planes
    if as_4d:
        hdr_real["NAXIS4"] = 1
        hdr_real["CTYPE4"] = "STOKES"
        hdr_real["CRPIX4"] = 1.0
        hdr_real["CRVAL4"] = 1.0
        hdr_real["CDELT4"] = 1.0

    hdr_real["BUNIT"] = "JY"
    hdr_real["BTYPE"] = "Intensity"
    hdr_real["CTYPE1"] = "U---WAV"
    hdr_real["CRPIX1"] = 1.0 if hermitian_half else float(npix_x // 2 + 1)
    hdr_real["CRVAL1"] = 0.0
    hdr_real["CDELT1"] = float(1.0 / (npix_x * pix_rad))
    hdr_real["CUNIT1"] = "lambda"
    if hermitian_half:
        hdr_real["HERMHALF"] = True

    hdr_real["CTYPE2"] = "V---WAV"
    hdr_real["CRPIX2"] = float(npix_y // 2 + 1)
    hdr_real["CRVAL2"] = 0.0
    hdr_real["CDELT2"] = float(1.0 / (npix_y * pix_rad))
    hdr_real["CUNIT2"] = "lambda"

    hdr_real["CTYPE3"] = "TIME"
    hdr_real["CRPIX3"] = 1.0
    hdr_real["CRVAL3"] = float(time_info["grid_times"][0])
    hdr_real["CDELT3"] = float(time_info["dt_nominal"])
    hdr_real["CUNIT3"] = "s"

    hdr_real["WSCIMGTY"] = "UV"
    hdr_real["WSCVTYPE"] = "REAL"

    hdr_imag = hdr_real.copy()
    hdr_imag["WSCVTYPE"] = "IMAG"
    return hdr_real, hdr_imag


# =========================================================================
# Worker Chunk Execution & Streaming
# =========================================================================

def _write_slices(
    file_path: str,
    header_size: int,
    data_chunk: np.ndarray,
    grid_indices: Optional[np.ndarray],
    offset_start: int,
    bytes_per_plane: int,
) -> None:
    """Write a contiguous or scattered chunk of planes into a pre-allocated FITS file."""
    with open(file_path, "r+b") as f:
        if grid_indices is None or (len(grid_indices) > 0 and np.all(np.diff(grid_indices) == 1)):
            s = int(grid_indices[0]) if grid_indices is not None else offset_start
            f.seek(header_size + s * bytes_per_plane)
            f.write(data_chunk.tobytes())
        else:
            for j, g_idx in enumerate(grid_indices):
                f.seek(header_size + int(g_idx) * bytes_per_plane)
                f.write(data_chunk[j].tobytes())


def process_chunk_task(
    msname: str,
    out_cube: Optional[str],
    chunk_times: np.ndarray,
    offset_start: int,
    offset_end: int,
    header_size: int = 0,
    npix_x: int = 384,
    npix_y: int = 384,
    pix_rad: float = 1e-4,
    grid_indices: Optional[np.ndarray] = None,
    save_uv: bool = False,
    uv_mode: str = "fft",
    out_cube_real: Optional[str] = None,
    out_cube_imag: Optional[str] = None,
    header_size_uv_real: int = 0,
    header_size_uv_imag: int = 0,
    write_dirty: bool = True,
    cutout_bbox: Optional[Tuple[int, int, int, int]] = None,
    n_target_samples: Optional[int] = None,
    direct_write: bool = True,
    dtype_str: str = ">f4",
    data_column: str = "DATA",
    corr_mode: str = "single",
    basis: str = "linear",
    single_pol: str = "XX",
    epsilon: float = 1e-6,
    do_wgridding: bool = True,
    nthreads: int = 0,
    verbosity: int = 0,
    flip_u: bool = False,
    flip_v: bool = False,
    flip_w: bool = False,
    divide_by_n: bool = True,
    sigma_min: float = 1.1,
    sigma_max: float = 2.6,
    min_valid_channels: int = 24,
    dm: float = 0.0,
    collapse_channels: bool = False,
    nsubbands: int = 1,
    exact_uvw: bool = True,
    align_to: str = "fmax",
    shift_samps: float = 0.0,
    shift_method: str = "fft",
    shift_domain: str = "native",
    uv_hermitian_half: bool = False,
    uv_zarr_path: Optional[str] = None,
    **kwargs,
) -> Tuple[int, int, int, Any]:
    """Worker task: load visibilities, apply dedispersion, grid snapshots, compute UV FFT, and stream to cubes."""
    times_chunk, chunk_freq, dt, vis_3d, wgt_3d, uvw_3d = imaging.load_chunk_data(
        msname=msname, chunk_times=chunk_times, data_column=data_column,
        corr_mode=corr_mode, basis=basis, single_pol=single_pol,
    )

    if abs(shift_samps) > 1e-7 and shift_domain == "native":
        vis_3d = timeshift.shift_time_series(vis_3d, shift_samps, method=shift_method, axis=-1)
        wgt_3d = timeshift.shift_time_series(wgt_3d, shift_samps, method=shift_method, axis=-1)
        wgt_3d = np.maximum(wgt_3d, 0.0)
        uvw_3d = timeshift.shift_time_series(uvw_3d, shift_samps, method=shift_method, axis=-1)

    real_chunk, imag_chunk = None, None
    cube_chunk = None

    if save_uv and uv_mode == "vis2grid":
        uv_cube_chunk = imaging.grid_chunk_uv_cube(
            vis_3d=vis_3d, wgt_3d=wgt_3d, uvw_3d=uvw_3d, chan_freq=chunk_freq, dt=dt,
            dm=dm, collapse_channels=collapse_channels, nsubbands=nsubbands,
            align_to=align_to, exact_uvw=exact_uvw, npix_x=npix_x, npix_y=npix_y,
            pixsize_x=pix_rad, pixsize_y=pix_rad, epsilon=epsilon, do_wgridding=do_wgridding,
            nthreads=nthreads, verbosity=verbosity, flip_u=flip_u, flip_v=flip_v, flip_w=flip_w,
            divide_by_n=divide_by_n, sigma_min=sigma_min, sigma_max=sigma_max,
            min_valid_channels=min_valid_channels,
        )
        if abs(shift_samps) > 1e-7 and shift_domain == "gridded":
            uv_cube_chunk = timeshift.shift_time_series(uv_cube_chunk, shift_samps, method=shift_method, axis=0)

        if n_target_samples is not None and len(uv_cube_chunk) > n_target_samples:
            uv_cube_chunk = uv_cube_chunk[:n_target_samples]

        if uv_hermitian_half and uv_cube_chunk is not None:
            u_start = uv_cube_chunk.shape[-1] // 2
            uv_cube_chunk = uv_cube_chunk[..., u_start:]

        real_chunk = uv_cube_chunk.real.astype(dtype_str)
        imag_chunk = uv_cube_chunk.imag.astype(dtype_str)

        if write_dirty:
            cube_chunk = imaging.grid_chunk_cube(
                vis_3d=vis_3d, wgt_3d=wgt_3d, uvw_3d=uvw_3d, chan_freq=chunk_freq, dt=dt,
                dm=dm, collapse_channels=collapse_channels, nsubbands=nsubbands,
                align_to=align_to, exact_uvw=exact_uvw, npix_x=npix_x, npix_y=npix_y,
                pixsize_x=pix_rad, pixsize_y=pix_rad, epsilon=epsilon, do_wgridding=do_wgridding,
                nthreads=nthreads, verbosity=verbosity, flip_u=flip_u, flip_v=flip_v, flip_w=flip_w,
                divide_by_n=divide_by_n, sigma_min=sigma_min, sigma_max=sigma_max,
                min_valid_channels=min_valid_channels,
            )
            if abs(shift_samps) > 1e-7 and shift_domain == "gridded":
                cube_chunk = timeshift.shift_time_series(cube_chunk, shift_samps, method=shift_method, axis=0)
            if n_target_samples is not None and len(cube_chunk) > n_target_samples:
                cube_chunk = cube_chunk[:n_target_samples]
            if cutout_bbox is not None:
                ymin, ymax, xmin, xmax = cutout_bbox
                cube_chunk = cube_chunk[:, ymin:ymax, xmin:xmax]
    else:
        cube_chunk = imaging.grid_chunk_cube(
            vis_3d=vis_3d, wgt_3d=wgt_3d, uvw_3d=uvw_3d, chan_freq=chunk_freq, dt=dt,
            dm=dm, collapse_channels=collapse_channels, nsubbands=nsubbands,
            align_to=align_to, exact_uvw=exact_uvw, npix_x=npix_x, npix_y=npix_y,
            pixsize_x=pix_rad, pixsize_y=pix_rad, epsilon=epsilon, do_wgridding=do_wgridding,
            nthreads=nthreads, verbosity=verbosity, flip_u=flip_u, flip_v=flip_v, flip_w=flip_w,
            divide_by_n=divide_by_n, sigma_min=sigma_min, sigma_max=sigma_max,
            min_valid_channels=min_valid_channels,
        )

        if abs(shift_samps) > 1e-7 and shift_domain == "gridded":
            cube_chunk = timeshift.shift_time_series(cube_chunk, shift_samps, method=shift_method, axis=0)

        if n_target_samples is not None and len(cube_chunk) > n_target_samples:
            cube_chunk = cube_chunk[:n_target_samples]

        if cutout_bbox is not None:
            ymin, ymax, xmin, xmax = cutout_bbox
            cube_chunk = cube_chunk[:, ymin:ymax, xmin:xmax]

        if save_uv:
            uv_chunk = np.fft.fftshift(
                np.fft.fft2(np.fft.ifftshift(cube_chunk, axes=(-2, -1)), axes=(-2, -1)),
                axes=(-2, -1),
            )
            if uv_hermitian_half and uv_chunk is not None:
                u_start = uv_chunk.shape[-1] // 2
                uv_chunk = uv_chunk[..., u_start:]
            real_chunk = uv_chunk.real.astype(dtype_str)
            imag_chunk = uv_chunk.imag.astype(dtype_str)

    n_samples = len(times_chunk)
    bytes_per_plane = (cube_chunk.shape[-2] * cube_chunk.shape[-1] * (4 if "4" in dtype_str else 8)) if cube_chunk is not None else 0
    bytes_per_plane_uv = (real_chunk.shape[-2] * real_chunk.shape[-1] * (4 if "4" in dtype_str else 8)) if real_chunk is not None else 0

    if direct_write:
        try:
            if write_dirty and out_cube and os.path.exists(out_cube) and cube_chunk is not None:
                _write_slices(out_cube, header_size, cube_chunk.astype(dtype_str), grid_indices, offset_start, bytes_per_plane)
            if save_uv and out_cube_real and os.path.exists(out_cube_real) and real_chunk is not None:
                _write_slices(out_cube_real, header_size_uv_real, real_chunk, grid_indices, offset_start, bytes_per_plane_uv)
            if save_uv and out_cube_imag and os.path.exists(out_cube_imag) and imag_chunk is not None:
                _write_slices(out_cube_imag, header_size_uv_imag, imag_chunk, grid_indices, offset_start, bytes_per_plane_uv)
            return (offset_start, offset_end, n_samples, None)
        except OSError:
            pass

    ret_dirty = cube_chunk.astype(dtype_str) if (write_dirty and cube_chunk is not None) else None
    return (offset_start, offset_end, n_samples, (ret_dirty, real_chunk, imag_chunk))
    return (offset_start, offset_end, n_samples, (ret_dirty, real_chunk, imag_chunk))


# =========================================================================
# Fitscube External Combiner
# =========================================================================

def combine_with_fitscube(
    image_files: List[str],
    out_cube: str,
    time_domain: bool = True,
    overwrite: bool = True,
    create_blanks: bool = False,
    no_blank_zero_beams: bool = True,
    floating: Optional[int] = 32,
    max_workers: Optional[int] = None,
    verbosity: int = 0,
    fallback_astropy: bool = False,
) -> None:
    """Use Alec Thomson's fitscube library (or fallback) to combine 2D images into a FITS cube."""
    if not image_files:
        raise ValueError("No image files provided to combine into a FITS cube.")

    for pkg in ("fitscube", "fitscube_rs"):
        try:
            mod = __import__(pkg)
            if hasattr(mod, "combine_fits"):
                sig = inspect.signature(mod.combine_fits)
                kw = {}
                arg_name = "file_list" if "file_list" in sig.parameters else "files"
                kw[arg_name] = [str(f) for f in image_files]
                kw["out_cube" if "out_cube" in sig.parameters else "output"] = str(out_cube)
                if "overwrite" in sig.parameters: kw["overwrite"] = bool(overwrite)
                if "create_blanks" in sig.parameters: kw["create_blanks"] = bool(create_blanks)
                mod.combine_fits(**kw)
                if os.path.exists(out_cube): return
        except Exception:
            continue

    for bin_name in ("fitscube", "fitscubers"):
        p = shutil.which(bin_name)
        if p:
            cmd = [p, "combine", "-o" if overwrite else "", "--create-blanks" if create_blanks else ""]
            cmd = [c for c in cmd if c] + [str(f) for f in image_files] + [str(out_cube)]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0 and os.path.exists(out_cube): return

    if fallback_astropy:
        first_hdr = fits.getheader(image_files[0])
        ny, nx = fits.getdata(image_files[0]).shape[-2:]
        times_arr = np.array([float(fits.getheader(f).get("TIME", fits.getheader(f).get("MJD-OBS", 0.0) * 86400.0)) for f in image_files])
        time_info = analyze_time_axis(times_arr, create_blanks=create_blanks)
        total_planes = time_info["total_planes"]

        cube_data = np.full((total_planes, ny, nx), np.nan if create_blanks else 0.0, dtype=np.float32)
        for i, f in enumerate(image_files):
            cube_data[int(time_info["grid_indices"][i]), :, :] = fits.getdata(f).astype(np.float32)

        hdr3d = first_hdr.copy()
        hdr3d.update({"NAXIS": 3, "NAXIS1": nx, "NAXIS2": ny, "NAXIS3": total_planes, "CTYPE3": "TIME",
                      "CRVAL3": float(time_info["grid_times"][0]), "CRPIX3": 1.0, "CDELT3": float(time_info["dt_nominal"]),
                      "CUNIT3": "s", "IS_EVEN": bool(time_info["is_regular"] and not (time_info["has_gaps"] and not create_blanks))})
        if time_info["has_gaps"] and create_blanks:
            hdr3d["BLANKS"] = True
        elif not time_info["is_regular"] or (time_info["has_gaps"] and not create_blanks):
            hdr3d["IRREGTIM"] = True

        table_hdu = create_timestamps_table_hdu(time_info["grid_times"], time_info["is_blank"], time_info["delta_t"], time_info["dt_nominal"])
        fits.HDUList([fits.PrimaryHDU(data=cube_data, header=hdr3d), table_hdu]).writeto(out_cube, overwrite=overwrite)
        return

    raise RuntimeError("fitscube library not available. Install fitscube or enable direct streaming.")


def _normalize_time(t: float, all_times: np.ndarray) -> float:
    return t * 86400.0 if t < 1e6 and len(all_times) > 0 and all_times[0] >= 1e6 else t


def resolve_time_bounds(
    all_times: np.ndarray,
    start_time_idx: Optional[int] = None,
    end_time_idx: Optional[int] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> Tuple[int, int]:
    """Resolve start and end indices in sorted observation timestamps."""
    n_times = len(all_times)
    if n_times == 0:
        raise ValueError("Cannot resolve bounds for empty timestamp array.")

    start_idx = 0 if start_time_idx is None else max(0, min(n_times - 1, int(start_time_idx)))
    end_idx = n_times - 1 if end_time_idx is None else max(0, min(n_times - 1, int(end_time_idx)))

    if start_time is not None:
        t_val = _normalize_time(start_time, all_times)
        start_idx = max(start_idx, int(np.searchsorted(all_times, t_val, side="left")))
    if end_time is not None:
        t_val = _normalize_time(end_time, all_times)
        end_idx = min(end_idx, int(np.searchsorted(all_times, t_val, side="right") - 1))

    if start_idx > end_idx:
        raise ValueError(f"Start index ({start_idx}) cannot exceed end index ({end_idx}).")

    return start_idx, end_idx


# =========================================================================
# Main Orchestration Engine
# =========================================================================

def make_dirty_cube(
    msname: str = "",
    out_cube: Optional[str] = None,
    *,
    cfg: Optional[CubeConfig] = None,
    config: Optional[CubeConfig] = None,
    coord: Optional[Any] = None,
    ra: Optional[Any] = None,
    dec: Optional[Any] = None,
    radius: Optional[Any] = None,
    radius_arcsec: Optional[float] = None,
    radius_pix: Optional[float] = None,
    box_size_pix: Optional[int] = None,
    box_size_arcsec: Optional[float] = None,
    radius_unit: str = "arcsec",
    start_time_idx: Optional[int] = None,
    end_time_idx: Optional[int] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    chunk_size: int = 512,
    time_step: Optional[float] = None,
    time_tolerance: float = 1e-3,
    max_blank_factor: float = 10.0,
    write_timestamps_table: bool = True,
    save_uv: bool = False,
    uv_only: bool = False,
    uv_mode: str = "auto",
    uv_format: str = "fits",
    uv_hermitian_half: bool = False,
    uv_zarr_path: Optional[str] = None,
    uv_threshold: float = 1e-4,
    out_uv_real: Optional[str] = None,
    out_uv_imag: Optional[str] = None,
    uv_4d: bool = False,
    npix_x: int = 384,
    npix_y: int = 384,
    pixsize_arcsec: float = 22.0,
    corr_mode: str = "single",
    single_pol: str = "XX",
    basis: str = "linear",
    data_column: str = "DATA",
    field_name: Optional[str] = None,
    ra_sign: int = -1,
    dec_sign: int = -1,
    epsilon: float = 1e-6,
    do_wgridding: bool = True,
    nthreads: int = 0,
    verbosity: int = 0,
    flip_u: bool = False,
    flip_v: bool = False,
    flip_w: bool = False,
    divide_by_n: bool = True,
    sigma_min: float = 1.1,
    sigma_max: float = 2.6,
    min_valid_channels: int = 24,
    dm: float = 0.0,
    collapse_channels: bool = False,
    nsubbands: int = 1,
    exact_uvw: bool = True,
    align_to: str = "fmax",
    overwrite: bool = True,
    parallel_mode: Optional[str] = None,
    max_workers: Optional[int] = None,
    threads_per_worker: int = 1,
    dask_scheduler: str = "processes",
    slurm_partition: Optional[str] = None,
    slurm_account: Optional[str] = None,
    slurm_cores_per_worker: int = 1,
    slurm_mem: str = "32GB",
    slurm_walltime: str = "02:30:00",
    slurm_job_extra: Optional[List[str]] = None,
    slurm_interface: Optional[str] = None,
    use_fitscube: bool = False,
    create_blanks: bool = False,
    floating: int = 32,
    regularize_time: bool = False,
    sync_tolerance: float = 0.05,
    shift_method: str = "fft",
    shift_domain: str = "native",
    **kwargs,
) -> str:
    """Image MS snapshots with ducc0.wgridder and asynchronously stream directly into FITS cubes."""
    if cfg is None:
        cfg = config

    if cfg is None:
        if not msname:
            raise ValueError("msname or cfg/config must be provided to make_dirty_cube.")
        cfg = CubeConfig(
            msname=msname, out_cube=out_cube, coord=coord, ra=ra, dec=dec,
            radius=radius, radius_arcsec=radius_arcsec, radius_pix=radius_pix,
            box_size_pix=box_size_pix, box_size_arcsec=box_size_arcsec,
            radius_unit=radius_unit, start_time_idx=start_time_idx,
            end_time_idx=end_time_idx, start_time=start_time, end_time=end_time,
            chunk_size=chunk_size, time_step=time_step, time_tolerance=time_tolerance,
            max_blank_factor=max_blank_factor, write_timestamps_table=write_timestamps_table,
            save_uv=save_uv, uv_only=uv_only, uv_mode=uv_mode, out_uv_real=out_uv_real, out_uv_imag=out_uv_imag,
            uv_format=uv_format, uv_hermitian_half=uv_hermitian_half,
            uv_zarr_path=uv_zarr_path, uv_threshold=uv_threshold,
            uv_4d=uv_4d, npix_x=npix_x, npix_y=npix_y, pixsize_arcsec=pixsize_arcsec,
            corr_mode=corr_mode, single_pol=single_pol, basis=basis, data_column=data_column,
            field_name=field_name, ra_sign=ra_sign, dec_sign=dec_sign, epsilon=epsilon,
            do_wgridding=do_wgridding, nthreads=nthreads, verbosity=verbosity,
            flip_u=flip_u, flip_v=flip_v, flip_w=flip_w, divide_by_n=divide_by_n,
            sigma_min=sigma_min, sigma_max=sigma_max, min_valid_channels=min_valid_channels,
            dm=dm, collapse_channels=collapse_channels, nsubbands=nsubbands,
            exact_uvw=exact_uvw, align_to=align_to, overwrite=overwrite,
            parallel_mode=parallel_mode, max_workers=max_workers,
            threads_per_worker=threads_per_worker, dask_scheduler=dask_scheduler,
            slurm_partition=slurm_partition, slurm_account=slurm_account,
            slurm_cores_per_worker=slurm_cores_per_worker, slurm_mem=slurm_mem,
            slurm_walltime=slurm_walltime, slurm_job_extra=slurm_job_extra,
            slurm_interface=slurm_interface, use_fitscube=use_fitscube,
            create_blanks=create_blanks, floating=floating,
            regularize_time=regularize_time, sync_tolerance=sync_tolerance,
            shift_method=shift_method, shift_domain=shift_domain,
        )
    else:
        if msname:
            cfg.msname = msname
        if out_cube is not None:
            cfg.out_cube = out_cube
        for k, v in kwargs.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    msname = cfg.msname
    out_cube = cfg.out_cube
    coord = cfg.coord
    ra = cfg.ra
    dec = cfg.dec
    radius = cfg.radius
    radius_arcsec = cfg.radius_arcsec
    radius_pix = cfg.radius_pix
    box_size_pix = cfg.box_size_pix
    box_size_arcsec = cfg.box_size_arcsec
    radius_unit = cfg.radius_unit
    start_time_idx = cfg.start_time_idx
    end_time_idx = cfg.end_time_idx
    start_time = cfg.start_time
    end_time = cfg.end_time
    chunk_size = cfg.chunk_size
    time_step = cfg.time_step
    time_tolerance = cfg.time_tolerance
    max_blank_factor = cfg.max_blank_factor
    write_timestamps_table = cfg.write_timestamps_table
    save_uv = cfg.save_uv
    uv_only = cfg.uv_only
    uv_mode = getattr(cfg, "uv_mode", uv_mode)
    out_uv_real = cfg.out_uv_real
    out_uv_imag = cfg.out_uv_imag
    uv_format = getattr(cfg, "uv_format", "fits")
    uv_hermitian_half = getattr(cfg, "uv_hermitian_half", False) or (uv_format == "fits-half")
    uv_zarr_path = getattr(cfg, "uv_zarr_path", None)
    uv_threshold = getattr(cfg, "uv_threshold", 1e-4)
    uv_4d = cfg.uv_4d
    npix_x = cfg.npix_x
    npix_y = cfg.npix_y
    pixsize_arcsec = cfg.pixsize_arcsec
    corr_mode = cfg.corr_mode
    single_pol = cfg.single_pol
    basis = cfg.basis
    data_column = cfg.data_column
    field_name = cfg.field_name
    ra_sign = cfg.ra_sign
    dec_sign = cfg.dec_sign
    epsilon = cfg.epsilon
    do_wgridding = cfg.do_wgridding
    nthreads = cfg.nthreads
    verbosity = cfg.verbosity
    flip_u = cfg.flip_u
    flip_v = cfg.flip_v
    flip_w = cfg.flip_w
    divide_by_n = cfg.divide_by_n
    sigma_min = cfg.sigma_min
    sigma_max = cfg.sigma_max
    min_valid_channels = cfg.min_valid_channels
    dm = cfg.dm
    collapse_channels = cfg.collapse_channels
    nsubbands = cfg.nsubbands
    exact_uvw = cfg.exact_uvw
    align_to = cfg.align_to
    overwrite = cfg.overwrite
    parallel_mode = cfg.parallel_mode
    max_workers = cfg.max_workers
    threads_per_worker = cfg.threads_per_worker
    dask_scheduler = cfg.dask_scheduler
    slurm_partition = cfg.slurm_partition
    slurm_account = cfg.slurm_account
    slurm_cores_per_worker = cfg.slurm_cores_per_worker
    slurm_mem = cfg.slurm_mem
    slurm_walltime = cfg.slurm_walltime
    slurm_job_extra = cfg.slurm_job_extra
    slurm_interface = cfg.slurm_interface
    use_fitscube = cfg.use_fitscube
    create_blanks = cfg.create_blanks
    floating = cfg.floating
    regularize_time = cfg.regularize_time
    sync_tolerance = cfg.sync_tolerance
    shift_method = cfg.shift_method
    shift_domain = cfg.shift_domain

    if not os.path.exists(msname):
        raise FileNotFoundError(f"Measurement Set '{msname}' not found.")

    ms_base = os.path.basename(msname.rstrip("/")).replace(".ms", "")
    t_main = table(msname, readonly=True)
    time_col = "TIME_CENTROID" if "TIME_CENTROID" in t_main.colnames() else "TIME"
    has_scan = "SCAN_NUMBER" in t_main.colnames()
    time_vals = t_main.getcol(time_col)
    scan_vals = t_main.getcol("SCAN_NUMBER") if has_scan else None
    t_main.close()

    unique_times, u_idx = np.unique(time_vals, return_index=True)
    scan_per_time = scan_vals[u_idx] if scan_vals is not None else np.zeros(len(unique_times), dtype=int)

    start_idx, end_idx = resolve_time_bounds(unique_times, start_time_idx, end_time_idx, start_time, end_time)
    sel_times = unique_times[start_idx : end_idx + 1]
    sel_scans = scan_per_time[start_idx : end_idx + 1]
    total_frames = len(sel_times)

    if regularize_time:
        create_blanks = True
        time_info = timeshift.regularize_time_grid(
            sel_times,
            dt_nominal=time_step,
            sync_tolerance=sync_tolerance,
            scan_numbers=sel_scans,
            max_blank_factor=max_blank_factor,
        )
    else:
        time_info = analyze_time_axis(sel_times, create_blanks=create_blanks, dt_nominal=time_step, tolerance=time_tolerance, max_blank_factor=max_blank_factor)
    total_planes = time_info["total_planes"]

    save_uv = save_uv or uv_only
    write_dirty = not uv_only
    if uv_mode is None or uv_mode == "auto":
        uv_mode = "vis2grid" if uv_only else "fft"
    ra0_rad, dec0_rad, used_field = ms_utils.get_phase_center(msname, field_name)
    pix_rad = pixsize_arcsec / 206265.0

    target_coord = parse_coordinate(coord=coord, ra=ra, dec=dec)
    r_arcsec, r_pix, b_pix = parse_radius(radius, radius_arcsec, radius_pix, box_size_pix, box_size_arcsec, pixsize_arcsec, radius_unit)
    is_cubelet = bool((target_coord is not None) or (r_arcsec is not None))

    if is_cubelet:
        target_coord = target_coord or SkyCoord(ra=math.degrees(ra0_rad) * u.deg, dec=math.degrees(dec0_rad) * u.deg)
        r_arcsec = r_arcsec or 60.0
        r_pix = max(1, int(round(r_arcsec / pixsize_arcsec)))
        geom = compute_cubelet_geometry(target_coord, r_arcsec, r_pix, b_pix, ra0_rad, dec0_rad, pixsize_arcsec, npix_x, npix_y, ra_sign, dec_sign)
        grid_npix_x, grid_npix_y = geom["grid_npix_x"], geom["grid_npix_y"]
        nx_out, ny_out = geom["nx_cut"], geom["ny_cut"]
        cutout_bbox, wcs_full = geom["bbox"], geom["wcs_full"]
        xmin, ymin = geom["xmin"], geom["ymin"]
    else:
        grid_npix_x, grid_npix_y = npix_x, npix_y
        nx_out, ny_out = npix_x, npix_y
        cutout_bbox, xmin, ymin = None, 0, 0
        wcs_full = ducc_wcs._build_fullframe_wcs(npix_x=npix_x, npix_y=npix_y, ra0_rad=ra0_rad, dec0_rad=dec0_rad, pixscale_rad=pix_rad, ra_sign=ra_sign, dec_sign=dec_sign, radesys="ICRS")

    dm_tag = f"_dm{dm:06.1f}" if dm != 0.0 else ""
    out_cube = out_cube or f"{ms_base}_t{start_idx:05d}_{end_idx:05d}{dm_tag}_{'cubelet' if is_cubelet else 'dirty_cube'}.fits"
    out_cube_real, out_cube_imag = derive_uv_cube_names(out_cube, msname, out_uv_real, out_uv_imag, start_idx, end_idx) if save_uv else (None, None)

    for path, flag in [(out_cube, write_dirty), (out_cube_real, save_uv), (out_cube_imag, save_uv)]:
        if flag and path and os.path.exists(path) and not overwrite:
            raise FileExistsError(f"Output cube '{path}' exists and overwrite=False.")

    hdr3d = wcs_full.to_header()
    hdr3d.update({
        "NAXIS": 3, "NAXIS1": nx_out, "NAXIS2": ny_out, "NAXIS3": total_planes, "BUNIT": "Jy/beam",
        "CDELT1": wcs_full.wcs.cdelt[0], "CDELT2": wcs_full.wcs.cdelt[1],
        "CRPIX1": float(wcs_full.wcs.crpix[0] - xmin), "CRPIX2": float(wcs_full.wcs.crpix[1] - ymin),
        "PC1_1": 1.0, "PC1_2": 0.0, "PC2_1": 0.0, "PC2_2": 1.0, "CTYPE3": "TIME",
        "CRVAL3": float(time_info["grid_times"][0]), "CRPIX3": 1.0, "CDELT3": float(time_info["dt_nominal"]),
        "CUNIT3": "s", "IS_EVEN": bool(time_info["is_regular"] and not (time_info["has_gaps"] and not create_blanks)),
        "TIME": float(unique_times[start_idx]), "MJD-OBS": float(unique_times[start_idx]) / 86400.0,
        "MJD-END": float(unique_times[end_idx]) / 86400.0, "MJD": float(unique_times[start_idx]) / 86400.0,
        "TIMESYS": "UTC", "CREATOR": "fastducc", "INSTRUME": "ducc0.wgridder", "DM": float(dm),
    })
    if is_cubelet:
        hdr3d.update({"CUBELET": True, "TARG_RA": float(target_coord.ra.deg), "TARG_DEC": float(target_coord.dec.deg), "CUB_RAD": float(r_arcsec)})
    if dm != 0.0:
        hdr3d.update({"DEDISP": True, "COLLCHAN": bool(collapse_channels), "NSUBBAND": int(nsubbands), "EXACTUVW": bool(exact_uvw), "ALIGNDM": str(align_to)})
    if regularize_time:
        hdr3d.update({
            "TIMEREG": True,
            "SYNCTOL": float(sync_tolerance),
            "SHIFTMET": str(shift_method),
            "SHIFTDOM": str(shift_domain),
            "NBLOCKS": int(len(time_info.get("blocks", []))),
            "NSHIFTED": int(time_info.get("n_shifted_blocks", 0)),
        })
    if time_info["has_gaps"] and create_blanks:
        hdr3d["BLANKS"] = True
        hdr3d["NBLANKS"] = len(time_info["blank_indices"])
    elif not time_info["is_regular"] or (time_info["has_gaps"] and not create_blanks):
        hdr3d["IRREGTIM"] = True

    try:
        hdr3d["DATE-OBS"] = Time(hdr3d["MJD-OBS"], format="mjd", scale="utc").isot
        hdr3d["DATE-END"] = Time(hdr3d["MJD-END"], format="mjd", scale="utc").isot
    except Exception: pass
    if used_field: hdr3d["OBJECT"] = str(used_field)

    chan_freq = None
    try:
        with table(f"{msname}/SPECTRAL_WINDOW", readonly=True, ack=False) as t_spw:
            chan_freq = t_spw.getcell("CHAN_FREQ", 0)
            if chan_freq is not None and len(chan_freq) > 0:
                hdr3d["RESTFRQ"] = float(np.mean(chan_freq))
                hdr3d["FREQ"] = float(np.mean(chan_freq))
    except Exception: pass

    max_delay_samps = 0
    if dm != 0.0 and chan_freq is not None and len(chan_freq) > 0 and time_info["dt_nominal"] > 0:
        max_delay_samps = dedisp.get_dm_samps(chan_freq, abs(dm), time_info["dt_nominal"])

    chunk_tasks = []
    if regularize_time and "blocks" in time_info:
        for b in time_info["blocks"]:
            b_start = start_idx + b["start_sample"]
            b_end = start_idx + b["end_sample"]
            b_shift = float(b["shift_samps"])
            for c_start in range(b_start, b_end + 1, chunk_size):
                c_end = min(c_start + chunk_size - 1, b_end)
                n_target = c_end - c_start + 1
                c_end_pad = min(b_end, c_end + max_delay_samps) if dm != 0.0 and max_delay_samps > 0 else c_end
                c_times = unique_times[c_start : c_end_pad + 1]
                c_g_idx = time_info["grid_indices"][c_start - start_idx : c_end - start_idx + 1]
                chunk_tasks.append((c_start, c_end, c_times, c_g_idx, int(c_g_idx[0]), int(c_g_idx[-1]) + 1, n_target, b_shift))
    else:
        for c_start in range(start_idx, end_idx + 1, chunk_size):
            c_end = min(c_start + chunk_size - 1, end_idx)
            n_target = c_end - c_start + 1
            c_end_pad = min(len(unique_times) - 1, c_end + max_delay_samps) if dm != 0.0 and max_delay_samps > 0 else c_end
            c_times = unique_times[c_start : c_end_pad + 1]
            c_g_idx = time_info["grid_indices"][c_start - start_idx : c_end - start_idx + 1]
            chunk_tasks.append((c_start, c_end, c_times, c_g_idx, int(c_g_idx[0]), int(c_g_idx[-1]) + 1, n_target, 0.0))

    parallel_mode = parallel_mode or ("processes" if max_workers and max_workers > 1 else "serial")
    n_workers = max_workers if max_workers and max_workers > 0 else (os.cpu_count() or 4 if parallel_mode in ("processes", "threads", "dask-local") else 1)
    np_dtype = np.float32 if floating == 32 else np.float64
    dtype_str = ">f4" if floating == 32 else ">f8"

    gap_str = f"gaps={len(time_info['blank_indices'])} blanks" if time_info["has_gaps"] else "no gaps"
    dm_str = f" | DM: {dm:.2f} pc/cm^3 (delay_sweep={max_delay_samps} samps, exact_uvw={exact_uvw})" if dm != 0.0 else ""
    reg_str = f" | Regularized time (blocks={len(time_info.get('blocks', []))}, shifted={time_info.get('n_shifted_blocks', 0)}, method={shift_method}, domain={shift_domain})" if regularize_time else ""
    logger.info(f"MS: {msname} | Observed Frames: {total_frames} (indices {start_idx}..{end_idx}) | Output Planes: {total_planes} | Spacing: {time_info['spacing_type']} ({gap_str}, dt={time_info['dt_nominal']:.4f}s){reg_str} | Targets: {'save-uv (uv-only)' if uv_only else ('save-uv (dirty+uv)' if save_uv else 'dirty-only')}{dm_str} | Chunks: {len(chunk_tasks)} | Mode: {parallel_mode} (workers={n_workers})")

    # In-memory streaming initialization
    header_size = init_empty_fits_cube(out_cube, hdr3d, (total_planes, ny_out, nx_out), dtype=np_dtype, overwrite=overwrite, blank_indices=time_info["blank_indices"] if create_blanks else None) if write_dirty else 0
    header_size_uv_r, header_size_uv_i = 0, 0
    if save_uv:
        if uv_mode == "vis2grid":
            nu_grid, nv_grid = imaging.get_uv_grid_dims(
                npix_x=grid_npix_x, npix_y=grid_npix_y, pixsize_x=pix_rad, pixsize_y=pix_rad,
                epsilon=epsilon, do_wgridding=do_wgridding,
                sigma_min=sigma_min, sigma_max=sigma_max
            )
            nu_stored = (nu_grid // 2 + 1) if uv_hermitian_half else nu_grid
            hdr_uv_r, hdr_uv_i = build_uv_headers(hdr3d, nu_stored, nv_grid, pix_rad, total_planes, time_info, as_4d=uv_4d, hermitian_half=uv_hermitian_half)
            uv_shape = (1, total_planes, nv_grid, nu_stored) if uv_4d else (total_planes, nv_grid, nu_stored)
        else:
            nu_stored = (nx_out // 2 + 1) if uv_hermitian_half else nx_out
            hdr_uv_r, hdr_uv_i = build_uv_headers(hdr3d, nu_stored, ny_out, pix_rad, total_planes, time_info, as_4d=uv_4d, hermitian_half=uv_hermitian_half)
            uv_shape = (1, total_planes, ny_out, nu_stored) if uv_4d else (total_planes, ny_out, nu_stored)
        header_size_uv_r = init_empty_fits_cube(out_cube_real, hdr_uv_r, uv_shape, dtype=np_dtype, overwrite=overwrite, blank_indices=time_info["blank_indices"] if create_blanks else None)
        header_size_uv_i = init_empty_fits_cube(out_cube_imag, hdr_uv_i, uv_shape, dtype=np_dtype, overwrite=overwrite, blank_indices=time_info["blank_indices"] if create_blanks else None)

    task_kwargs = dict(
        msname=msname, out_cube=out_cube if write_dirty else None, npix_x=grid_npix_x, npix_y=grid_npix_y,
        pix_rad=pix_rad, header_size=header_size, data_column=data_column, corr_mode=corr_mode,
        basis=basis, single_pol=single_pol, epsilon=epsilon, do_wgridding=do_wgridding, nthreads=nthreads,
        verbosity=verbosity, flip_u=flip_u, flip_v=flip_v, flip_w=flip_w, divide_by_n=divide_by_n,
        sigma_min=sigma_min, sigma_max=sigma_max, min_valid_channels=min_valid_channels, dm=dm,
        collapse_channels=collapse_channels, nsubbands=nsubbands, exact_uvw=exact_uvw, align_to=align_to,
        dtype_str=dtype_str, save_uv=save_uv, uv_mode=uv_mode, out_cube_real=out_cube_real, out_cube_imag=out_cube_imag,
        header_size_uv_real=header_size_uv_r, header_size_uv_imag=header_size_uv_i, write_dirty=write_dirty,
        cutout_bbox=cutout_bbox, direct_write=True,
        shift_method=shift_method, shift_domain=shift_domain,
        uv_hermitian_half=uv_hermitian_half,
    )

    def _driver_write_chunk(path, arr, c_g_idx, off_s):
        with fits.open(path, mode="update", memmap=True) as hdul:
            target = hdul[0].data[0] if hdul[0].data.ndim == 4 else hdul[0].data
            if c_g_idx is None or (len(c_g_idx) > 0 and np.all(np.diff(c_g_idx) == 1)):
                s = int(c_g_idx[0]) if c_g_idx is not None else off_s
                target[s : s + len(arr)] = arr
            else:
                for j, g in enumerate(c_g_idx): target[int(g)] = arr[j]

    def _handle_worker_result(res, c_g_idx, off_s):
        if not res: return
        d_arr, r_arr, i_arr = res if isinstance(res, tuple) and len(res) == 3 else (res, None, None)
        if d_arr is not None and write_dirty: _driver_write_chunk(out_cube, d_arr, c_g_idx, off_s)
        if r_arr is not None and save_uv: _driver_write_chunk(out_cube_real, r_arr, c_g_idx, off_s)
        if i_arr is not None and save_uv: _driver_write_chunk(out_cube_imag, i_arr, c_g_idx, off_s)

    pbar = tqdm(total=total_frames, desc=f"Streaming snapshots to cube(s) ({parallel_mode})", unit="frame")
    if parallel_mode == "serial" or len(chunk_tasks) <= 1 or n_workers <= 1:
        for _, _, c_times, c_g_idx, off_s, off_e, n_target, b_shift in chunk_tasks:
            _, _, n_done, chunk_arr = process_chunk_task(chunk_times=c_times, grid_indices=c_g_idx, offset_start=off_s, offset_end=off_e, n_target_samples=n_target, shift_samps=b_shift, **task_kwargs)
            _handle_worker_result(chunk_arr, c_g_idx, off_s)
            pbar.update(n_done)
    elif parallel_mode in ("threads", "processes"):
        ExecutorClass = ThreadPoolExecutor if parallel_mode == "threads" else ProcessPoolExecutor
        with ExecutorClass(max_workers=n_workers) as executor:
            future_to_task = {
                executor.submit(process_chunk_task, chunk_times=c_times, grid_indices=c_g_idx, offset_start=off_s, offset_end=off_e, n_target_samples=n_target, shift_samps=b_shift, **task_kwargs): (c_g_idx, off_s)
                for (_, _, c_times, c_g_idx, off_s, off_e, n_target, b_shift) in chunk_tasks
            }
            for fut in as_completed(future_to_task):
                c_g_idx, off_s = future_to_task[fut]
                _, _, n_done, chunk_arr = fut.result()
                _handle_worker_result(chunk_arr, c_g_idx, off_s)
                pbar.update(n_done)
    elif parallel_mode in ("dask-local", "dask-slurm"):
        from dask.distributed import Client, as_completed as dask_as_completed
        if parallel_mode == "dask-local":
            from dask.distributed import LocalCluster
            cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker, processes=(dask_scheduler == "processes"))
            client_ctx = Client(cluster)
        else:
            from fastducc.dask_utils import setup_slurm_cluster, derive_dask_job_name, resolve_dask_log_directory
            job_name = derive_dask_job_name(stage="cube", ms_path=msname)
            log_dir = resolve_dask_log_directory()
            cluster = setup_slurm_cluster(
                queue=slurm_partition,
                account=slurm_account,
                cores_per_worker=slurm_cores_per_worker,
                mem=slurm_mem,
                walltime=slurm_walltime,
                job_name=job_name,
                log_directory=log_dir,
                job_extra_directives=slurm_job_extra or [],
                interface=slurm_interface,
                min_workers=1,
                max_workers=n_workers,
                adaptive=False,
            )
            client_ctx = Client(cluster)
        with client_ctx as client:
            future_to_task = {
                client.submit(process_chunk_task, chunk_times=c_times, grid_indices=c_g_idx, offset_start=off_s, offset_end=off_e, n_target_samples=n_target, shift_samps=b_shift, **task_kwargs): (c_g_idx, off_s)
                for (_, _, c_times, c_g_idx, off_s, off_e, n_target, b_shift) in chunk_tasks
            }
            for fut in dask_as_completed(list(future_to_task.keys())):
                c_g_idx, off_s = future_to_task[fut]
                _, _, n_done, chunk_arr = fut.result()
                _handle_worker_result(chunk_arr, c_g_idx, off_s)
                pbar.update(n_done)
    pbar.close()

    if write_timestamps_table:
        table_hdu = create_timestamps_table_hdu(time_info["grid_times"], time_info["is_blank"], time_info["delta_t"], time_info["dt_nominal"])
        for p, flag in [(out_cube, write_dirty), (out_cube_real, save_uv), (out_cube_imag, save_uv)]:
            if flag and p and os.path.exists(p):
                append_timestamps_extension(p, table_hdu)

    if write_dirty: logger.info(f"Successfully written dirty cube -> {out_cube}")
    if save_uv:
        logger.info(f"Successfully written UV real cube -> {out_cube_real}")
        logger.info(f"Successfully written UV imag cube -> {out_cube_imag}")

    return out_cube_real if uv_only else out_cube


# =========================================================================
# CLI Parser & Entry Point
# =========================================================================

def build_cli_cube(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Build command-line parser for 'fastducc cube'."""
    parser = argparse.ArgumentParser(prog="fastducc cube", description="Image MS time samples into FITS cube(s)")
    parser.add_argument("--msname", "-m", required=True, help="Input Measurement Set")
    parser.add_argument("--out-cube", "-o", default=None, help="Output dirty FITS cube")
    parser.add_argument("--coord", "--target", dest="coord", default=None, help="Target celestial position for cubelet mode")
    parser.add_argument("--ra", dest="ra", default=None, help="Target RA for cubelet mode")
    parser.add_argument("--dec", dest="dec", default=None, help="Target Dec for cubelet mode")
    parser.add_argument("--radius", dest="radius", default=None, help="Cubelet cutout radius (e.g. '45arcsec', '32pix')")
    parser.add_argument("--radius-arcsec", type=float, default=None)
    parser.add_argument("--radius-pix", type=float, default=None)
    parser.add_argument("--box-size", "--box-size-pix", dest="box_size_pix", type=int, default=None)
    parser.add_argument("--box-size-arcsec", type=float, default=None)
    parser.add_argument("--radius-unit", choices=["arcsec", "pix"], default="arcsec")
    parser.add_argument("--start-time-idx", "--start-idx", type=int, default=None)
    parser.add_argument("--end-time-idx", "--end-idx", type=int, default=None)
    parser.add_argument("--start-time", type=float, default=None)
    parser.add_argument("--end-time", type=float, default=None)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--npix-x", type=int, default=384)
    parser.add_argument("--npix-y", type=int, default=384)
    parser.add_argument("--pixsize-arcsec", type=float, default=22.0)
    parser.add_argument("--data-column", default="DATA")
    parser.add_argument("--corr-mode", choices=["average", "stokesI", "single"], default="single")
    parser.add_argument("--single-pol", default="XX")
    parser.add_argument("--basis", choices=["auto", "linear", "circular"], default="linear")
    parser.add_argument("--field-name", default=None)
    parser.add_argument("--dm", type=float, default=0.0, help="Trial DM (pc cm^-3)")
    parser.add_argument("--dm-list", type=float, nargs="+", default=None, help="List of trial DMs")
    parser.add_argument("--collapse-channels", dest="collapse_channels", action="store_true", default=False)
    parser.add_argument("--no-collapse-channels", dest="collapse_channels", action="store_false")
    parser.add_argument("--nsubbands", type=int, default=1)
    parser.add_argument("--exact-uvw", dest="exact_uvw", action="store_true", default=True)
    parser.add_argument("--no-exact-uvw", dest="exact_uvw", action="store_false")
    parser.add_argument("--align-to", choices=["fmax", "fmin"], default="fmax")
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--do-wgridding", dest="do_wgridding", action="store_true", default=True)
    parser.add_argument("--no-wgridding", dest="do_wgridding", action="store_false")
    parser.add_argument("--nthreads", type=int, default=0)
    parser.add_argument("--verbosity", type=int, default=0)
    parser.add_argument("--min-valid-channels", type=int, default=24)
    parser.add_argument("--parallel-mode", choices=["serial", "threads", "processes", "dask-local", "dask-slurm"], default="processes")
    parser.add_argument("-j", "--max-workers", "--workers", dest="max_workers", type=int, default=0)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--dask-scheduler", choices=["processes", "threads"], default="processes")
    parser.add_argument("--slurm-partition", default=None)
    parser.add_argument("--slurm-account", default=None)
    parser.add_argument("--slurm-cores-per-worker", type=int, default=1)
    parser.add_argument("--slurm-mem", default="32GB")
    parser.add_argument("--slurm-walltime", default="02:30:00")
    parser.add_argument("--slurm-job-extra", nargs="*", default=[])
    parser.add_argument("--slurm-interface", default=None)
    parser.add_argument("--use-fitscube", action="store_true", default=False)
    parser.add_argument("--overwrite", "-w", dest="overwrite", action="store_true", default=True)
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    parser.add_argument("--scratch-dir", default=None)
    parser.add_argument("--keep-planes", action="store_true", default=False)
    parser.add_argument("--create-blanks", action="store_true", default=False)
    parser.add_argument("--regularize-time", "--grid-time", dest="regularize_time", action="store_true", default=False, help="Grid data onto a strictly regular time grid, padding inter-scan gaps and shifting out-of-sync blocks")
    parser.add_argument("--sync-tolerance", type=float, default=0.05, help="Offset tolerance in units of offset/time_resolution (default: 0.05 = 5%%). Blocks with offset > tolerance are shifted")
    parser.add_argument("--shift-method", choices=["fft", "sinc"], default="fft", help="Method to shift out-of-sync blocks: Fourier domain phase gradient shift ('fft') or windowed sinc interpolation ('sinc')")
    parser.add_argument("--shift-domain", choices=["native", "gridded"], default="native", help="Domain to apply shifting: native visibilities ('native', faster) or gridded UV/dirty cubes ('gridded')")
    parser.add_argument("--time-step", "--dt", dest="time_step", type=float, default=None)
    parser.add_argument("--time-tolerance", type=float, default=1e-3)
    parser.add_argument("--max-blank-factor", type=float, default=10.0)
    parser.add_argument("--timestamps-table", dest="write_timestamps_table", action="store_true", default=True)
    parser.add_argument("--no-timestamps-table", dest="write_timestamps_table", action="store_false")
    parser.add_argument("--save-uv", "-save-uv", dest="save_uv", action="store_true", default=False)
    parser.add_argument("--uv-only", dest="uv_only", action="store_true", default=False)
    parser.add_argument("--uv-mode", dest="uv_mode", choices=["auto", "fft", "vis2grid"], default="auto", help="Gridded UV generation method: 'auto' ('vis2grid' for --uv-only, 'fft' for imaging+save-uv), 'fft' (2D FFT of dirty image), or 'vis2grid' (direct ducc0.wgridder.vis2grid with w-gridding)")
    parser.add_argument("--uv-format", choices=["fits", "fits-half", "dga_zarr", "zarr"], default="fits", help="Storage format for gridded UV data: 'fits', 'fits-half' (Hermitian half-plane), or 'dga_zarr'/'zarr'")
    parser.add_argument("--uv-hermitian-half", dest="uv_hermitian_half", action="store_true", default=False, help="Exploit Hermitian symmetry V(-u,-v)=V*(u,v) to store only the half-plane")
    parser.add_argument("--uv-zarr-path", default=None, help="Explicit path for output UV Zarr directory")
    parser.add_argument("--uv-threshold", type=float, default=1e-4, help="Amplitude threshold for sparse UV cell retention in DGA/Zarr mode")
    parser.add_argument("--out-uv-real", default=None)
    parser.add_argument("--out-uv-imag", default=None)
    parser.add_argument("--uv-4d", dest="uv_4d", action="store_true", default=False)
    parser.add_argument("--no-blank-zero-beams", dest="no_blank_zero_beams", action="store_true", default=True)
    parser.add_argument("--floating", type=int, choices=[16, 32, 64], default=32)
    parser.add_argument("--fallback-astropy", action="store_true", default=False)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for 'fastducc cube'."""
    args = build_cli_cube(argv)
    dm_trials = args.dm_list if (args.dm_list is not None and len(args.dm_list) > 0) else [args.dm]
    for trial_dm in dm_trials:
        out_cube = args.out_cube if len(dm_trials) == 1 else None
        cfg = CubeConfig.from_namespace(args, dm=float(trial_dm), out_cube=out_cube)
        make_dirty_cube(cfg=cfg, dm=float(trial_dm))
    return 0
