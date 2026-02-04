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

from fastducc import ms_utils

# ============================
# WCS HELPER FUNCTIONS
# ============================


def lm_from_xy(x: int, y: int,
               npix_x: int, npix_y: int,
               pixsize_x: float, pixsize_y: float,
               flip_u: bool = False, flip_v: bool = False) -> Tuple[float, float]:
    """
    Map image pixel (x,y) to direction cosines (l,m).
    Pixel scales must be in radians/pixel (as used in ducc0.wgridder).
    """
    xc = (npix_x - 1) / 2.0
    yc = (npix_y - 1) / 2.0
    l = (x - xc) * pixsize_x
    m = (y - yc) * pixsize_y
    if flip_u: l = -l
    if flip_v: m = -m
    return float(l), float(m)

def radec_from_lm(l: float, m: float, ra0_rad: float, dec0_rad: float) -> Tuple[float, float]:
    """
    Invert gnomonic (TAN) projection: (l,m) -> (RA,Dec) in radians.
    """
    # Guard against numerical issues at the edge of the unit disk
    lm2 = l*l + m*m
    if lm2 >= 1.0:
        raise ValueError(f"|l|^2+|m|^2 >= 1 ({lm2:.3f}); outside the valid tangent plane.")
    n = math.sqrt(max(0.0, 1.0 - lm2))
    ra  = ra0_rad  + math.atan2(l, n*math.cos(dec0_rad) - m*math.sin(dec0_rad))
    dec = math.asin(m*math.cos(dec0_rad) + n*math.sin(dec0_rad))
    return ra, dec



def rad_to_hmsdms(ra_rad: float, dec_rad: float, dp: int = 1) -> Tuple[str, str]:

    """
    Format RA,Dec (radians) as sexagesimal strings 'hh:mm:ss.s', '+dd:mm:ss.s',
    with proper rounding and carry to prevent '60.0' seconds/minutes.
    """
    def to_hms(r, dp = 1):
        
        r = r % (2 * np.pi)
        total_hours = r * 12.0 / np.pi

        h = int(total_hours)  # 0..23
        rem_hours = total_hours - h
        total_minutes = rem_hours * 60.0

        m = int(total_minutes)  # 0..59
        rem_minutes = total_minutes - m
        s = rem_minutes * 60.0

        # Round to one decimal
        s = round(s, dp)

        # Carry if seconds hit 60.0
        if s >= 60.0:
            s = 0.0
            m += 1

        # Carry if minutes hit 60
        if m >= 60:
            m = 0
            h += 1

        # Wrap hour 24 -> 0
        if h >= 24:
            h = 0

        return f"{h:02d}:{m:02d}:{s:0{3+dp}.{dp}f}"  # width 4 covers '0.0'..'59.9'

    def to_dms(r, dp = 1):
        deg = math.degrees(r)
        sign = '+' if deg >= 0 else '-'
        deg = abs(deg)

        d = int(deg)  # degrees 0..90 

        rem_deg = deg - d
        total_arcmin = rem_deg * 60.0

        m = int(total_arcmin)  # 0..59
        rem_arcmin = total_arcmin - m
        s = rem_arcmin * 60.0

        # Round to one decimal
        s = round(s, dp)

        # Carry if seconds hit 60.0
        if s >= 60.0:
            s = 0.0
            m += 1

        # Carry if minutes hit 60
        if m >= 60:
            m = 0
            d += 1

        # Cap to 90 degrees if rounding/carry pushes it over (rare edge case)
        if d > 90:
            d = 90
            m = 0
            s = 0.0

        return f"{sign}{d:02d}:{m:02d}:{s:0{3+dp}.{dp}f}"

    return to_hms(ra_rad, dp=dp), to_dms(dec_rad, dp=dp)

def hmsdms_to_srcname(ra_hms, dec_dms):
    """
    Format RA, Dec (hh:mm:ss.s, dd:mm:ss.s) to sourcename str:
    Jhhmmss.s+ddmmss.s
    """
    hhmmss  = ra_hms.split(':')
    src_hhmmss = "".join(hhmmss)
    
    if "-" in dec_dms:
        sign="-"
    else:
        sign="+"
    dec_dms = dec_dms.replace("-", "").replace("+", "")
    ddmmss = dec_dms.split(':')
    src_ddmmss = "".join(ddmmss)
    srcname = f"J{src_hhmmss}{sign}{src_ddmmss}"
    return srcname

def annotate_candidates_with_sky_coords(
        msname: str,
        final_detections: List[Dict[str, Any]],
        npix_x: int, npix_y: int,
        pixsize_x: float, pixsize_y: float,
        flip_u: bool = False, flip_v: bool = False,
        field_name: str = None) -> List[Dict[str, Any]]:
    """
    For each candidate dict (with 'x','y'), add l,m and RA/Dec.
    Returns a new list of candidate dicts augmented with keys:
    'l','m','ra_rad','dec_rad','ra_hms','dec_dms'.
    """
    ra0_rad, dec0_rad, used_field = ms_utils.get_phase_center(msname, field_name) 
    out = []
    for det in final_detections:
        x = int(det["x"]); y = int(det["y"])
        l, m = lm_from_xy(x, y, npix_x, npix_y, pixsize_x, pixsize_y, flip_u, flip_v)
        ra, dec = radec_from_lm(l, m, ra0_rad, dec0_rad)
        ra_hms, dec_dms = rad_to_hmsdms(ra, dec)
        srcname = hmsdms_to_srcname(ra_hms, dec_dms)
        det2 = dict(det)
        det2.update({
            "l": l, "m": m,
            "ra_rad": ra, "dec_rad": dec,
            "ra_hms": ra_hms, "dec_dms": dec_dms,
            "srcname": srcname,
            "phase_center_field": used_field
        })
        out.append(det2)
    return out



def _build_tan_wcs_for_snippet(spatial_size: int,
                               ra_rad: float,
                               dec_rad: float,
                               pixscale_rad: float,
                               *,
                               ra_sign: int = -1,
                               dec_sign: int = 1,
                               radesys: str = "ICRS",
                               equinox: float = None) -> WCS:
    """
    Build a 2D TAN WCS centered at the snippet center pixel with full, explicit
    scale and orientation parameters.

    Parameters
    ----------
    spatial_size : int
        N for an N x N snippet (reference pixel at the center).
    ra_rad, dec_rad : float
        World coordinates of the detection (radians) at the snippet center.
    pixscale_rad : float
        Pixel scale in radians/pixel (from CLI pixsize-arcsec).
    ra_sign : int
        +1  => RA increases to the right (CDELT1 > 0)
        -1  => RA increases to the left (astronomical convention; CDELT1 < 0)
    radesys : str
        Coordinate frame string to include in header (e.g. "ICRS", "FK5").
    equinox : float or None
        Equinox (e.g. 2000.0) if relevant (usually for FK5). Leave None for ICRS.

    Returns
    -------
    w : astropy.wcs.WCS
        A WCS object with CDELT/PC/CTYPE/CRPIX/CRVAL/CUNIT set.
    """
    
    w = WCS(naxis=2)

    crpix_x = (spatial_size + 1) / 2.0
    crpix_y = (spatial_size + 1) / 2.0
    w.wcs.crpix = [crpix_x, crpix_y]

    # Reference world coordinates in degrees
    w.wcs.crval = [math.degrees(ra_rad), math.degrees(dec_rad)]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cunit = ["deg", "deg"]
    
    sc_deg = math.degrees(pixscale_rad)
    w.wcs.cdelt = np.array([ra_sign * sc_deg, dec_sign * sc_deg], dtype=float)

    
    w.wcs.pc = np.array([[1.0, 0.0],
                         [0.0, 1.0]], dtype=float)

    if radesys:
        w.wcs.radesys = radesys
    if equinox is not None:
        w.wcs.equinox = float(equinox)

    return w


def _build_fullframe_wcs(npix_x: int, npix_y: int,
                         ra0_rad: float, dec0_rad: float,
                         pixscale_rad: float,
                         *,
                         ra_sign: int = -1,
                         dec_sign: int = 1,
                         radesys: str = "ICRS",
                         equinox: float = None):


    wcs_full = _build_tan_wcs_for_snippet(
        spatial_size=npix_y,     # WCSAxes expects naxis=2; we only need consistent CRPIX/CRVAL/CDELT
        ra_rad=ra0_rad,
        dec_rad=dec0_rad,
        pixscale_rad=pixscale_rad,
        ra_sign=ra_sign,
        dec_sign=dec_sign,
        radesys=radesys,
        equinox=equinox
    )
    # Overwrite CRPIX to be the full-frame center (FITS is 1-based)
    wcs_full.wcs.crpix = [(npix_x + 1) / 2.0, (npix_y + 1) / 2.0]
    return wcs_full


