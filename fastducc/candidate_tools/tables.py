# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Dict, Any, Optional, Union, Iterable, IO
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.table import Table

from fastducc.candidate_tools.utils import _row_value_to_python


def candidates_to_astropy_table(annotated: List[Dict[str, Any]]) -> Table:
    """
    Convert the list of annotated candidate dicts into an Astropy Table with units.

    Expected keys per candidate dict (from annotate_candidates_with_sky_coords):
      'x','y','l','m','ra_rad','dec_rad','ra_hms','dec_dms','snr','width_samples',
      'time_start','time_end','time_center','duration','t0_idx','t1_idx_excl','center_idx','scan_id',
      'phase_center_field'

    Returns
    -------
    t : astropy.table.Table
        Table with appropriate columns and units.
    """
    if len(annotated) == 0:
        return Table(names=[
            "srcname","x","y","l","m","ra_rad","dec_rad","ra_deg","dec_deg",
            "ra_hms","dec_dms","dm","snr","width_samples",
            "time_start","time_end","time_center","duration",
            "t0_idx","t1_idx_excl","center_idx","scan_id","phase_center_field"
        ], dtype=[
            str, int,int,float,float,float,float,float,float,
            "U20","U20",float,float,int,
            float,float,float,float,
            int,int,int,int,"U64"
        ])

    def col(key, default=np.nan):
        return [c.get(key, default) for c in annotated]

    srcname = np.array(col("srcname", default=""), dtype=str)
    xs  = np.array(col("x", default=np.int64(-1)), dtype=np.int64)
    ys  = np.array(col("y", default=np.int64(-1)), dtype=np.int64)
    ls  = np.array(col("l"), dtype=float)
    ms  = np.array(col("m"), dtype=float)
    ra_rad = np.array(col("ra_rad"), dtype=float)
    dec_rad = np.array(col("dec_rad"), dtype=float)

    ra_deg  = np.degrees(ra_rad)
    dec_deg = np.degrees(dec_rad)

    ra_hms  = np.array(col("ra_hms", default=""), dtype=str)
    dec_dms = np.array(col("dec_dms", default=""), dtype=str)

    dms          = np.array(col("dm", default=0.0), dtype=float)
    dms          = np.nan_to_num(dms, nan=0.0)
    snr          = np.array(col("snr"), dtype=float)
    width_samples= np.array(col("width_samples", default=np.int64(-1)), dtype=np.int64)

    time_start   = np.array(col("time_start"), dtype=float)
    time_end     = np.array(col("time_end"), dtype=float)
    time_center  = np.array(col("time_center"), dtype=float)
    duration     = np.array(col("duration"), dtype=float)

    t0_idx       = np.array(col("t0_idx", default=np.int64(-1)), dtype=np.int64)
    t1_idx_excl  = np.array(col("t1_idx_excl", default=np.int64(-1)), dtype=np.int64)
    scan_id      = np.array([str(s) if s not in (None, -1, "-1", np.nan) else "" for s in col("scan_id", default="")], dtype=str)
    center_idx   = np.array(col("center_idx", default=np.int64(-1)), dtype=np.int64)

    field_name   = np.array(col("phase_center_field", default=""), dtype=str)

    t = Table()
    t["srcname"] = srcname
    t["x"] = xs
    t["y"] = ys
    t["l"] = ls * u.dimensionless_unscaled
    t["m"] = ms * u.dimensionless_unscaled
    t["ra_rad"]  = ra_rad * u.rad
    t["dec_rad"] = dec_rad * u.rad
    t["ra_deg"]  = ra_deg * u.deg
    t["dec_deg"] = dec_deg * u.deg
    t["ra_hms"]  = ra_hms
    t["dec_dms"] = dec_dms
    t["dm"]      = dms * (u.pc / (u.cm**3))
    t["snr"]     = snr
    t["width_samples"] = width_samples
    t["time_start"]  = time_start * u.s
    t["time_end"]    = time_end * u.s
    t["time_center"] = time_center * u.s
    t["duration"]    = duration * u.s
    t["t0_idx"]      = t0_idx
    t["t1_idx_excl"] = t1_idx_excl
    t["center_idx"]  = center_idx
    t["scan_id"]     = scan_id
    t["phase_center_field"] = field_name

    return t


def candidates_to_astropy_table_periodicity(annotated: List[Dict[str, Any]]) -> Table:
    """
    Convert annotated periodicity candidates into an Astropy Table.

    Expected keys:
      x,y,l,m,ra_rad,dec_rad,ra_hms,dec_dms,srcname,
      snr,f_hz,period_s,nharm_used,algo,phase_center_field
    """
    if len(annotated) == 0:
        return Table(names=[
            "srcname","x","y","l","m",
            "ra_rad","dec_rad","ra_deg","dec_deg","ra_hms","dec_dms",
            "dm","snr","f_hz","period_s","nharm_used","algo","phase_center_field"
        ])

    def col(key, default=np.nan):
        return [c.get(key, default) for c in annotated]

    xs = np.array(col("x", -1), dtype=np.int64)
    ys = np.array(col("y", -1), dtype=np.int64)
    ls = np.array(col("l"), dtype=float)
    ms = np.array(col("m"), dtype=float)
    ra_rad = np.array(col("ra_rad"), dtype=float)
    dec_rad = np.array(col("dec_rad"), dtype=float)
    ra_deg = np.degrees(ra_rad)
    dec_deg = np.degrees(dec_rad)
    ra_hms = np.array(col("ra_hms", ""), dtype=str)
    dec_dms = np.array(col("dec_dms", ""), dtype=str)
    srcname = np.array(col("srcname", ""), dtype=str)
    field_name = np.array(col("phase_center_field", ""), dtype=str)

    dms = np.array(col("dm", 0.0), dtype=float)
    dms = np.nan_to_num(dms, nan=0.0)
    snr = np.array(col("snr"), dtype=float)
    f_hz = np.array(col("f_hz"), dtype=float)
    period_s = np.array(col("period_s"), dtype=float)
    nharm_used = np.array(col("nharm_used", 1), dtype=np.int64)
    algo = np.array(col("algo", "periodicity"), dtype=str)

    t = Table()
    t["srcname"] = srcname
    t["x"] = xs
    t["y"] = ys
    t["l"] = ls * u.dimensionless_unscaled
    t["m"] = ms * u.dimensionless_unscaled
    t["ra_rad"] = ra_rad * u.rad
    t["dec_rad"] = dec_rad * u.rad
    t["ra_deg"] = ra_deg * u.deg
    t["dec_deg"] = dec_deg * u.deg
    t["ra_hms"] = ra_hms
    t["dec_dms"] = dec_dms
    t["dm"] = dms * (u.pc / (u.cm**3))
    t["snr"] = snr
    t["f_hz"] = f_hz * u.Hz
    t["period_s"] = period_s * u.s
    t["nharm_used"] = nharm_used
    t["algo"] = algo
    t["phase_center_field"] = field_name
    return t


def astropy_table_to_candidates(
    source: Union[str, Path, IO[bytes], IO[str], Table], 
    *, 
    drop_derived: bool = False, 
    subset_keys: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Convert an Astropy Table back into a list of candidate dicts.
    This is the inverse of candidates_to_astropy_table, but may be lossy if the table was modified.
    """
    if isinstance(source, Table):
        t = source
    else:
        t = Table.read(source, format="votable")

    if len(t) == 0:
        return []

    colnames = list(t.colnames)

    if drop_derived:
        for k in ("ra_deg", "dec_deg"):
            if k in colnames:
                colnames.remove(k)

    if subset_keys is not None:
        subset = set(subset_keys)
        colnames = [c for c in colnames if c in subset]

    candidates: List[Dict[str, Any]] = []
    for row in t:
        d: Dict[str, Any] = {}
        for c in colnames:
            d[c] = _row_value_to_python(row[c])
        candidates.append(d)
    return candidates


def save_candidates_table(table: Table, csv_path: str, vot_path: str) -> None:
    """
    Save the Astropy Table to CSV and VOTable.
    - CSV: plain text; units appear in metadata but not in numeric values.
    - VOTable: includes unit information in FIELDs where available.
    """
    table.write(csv_path, format="csv", overwrite=True)
    table.write(vot_path, format="votable", overwrite=True)


def save_periodicity_candidates_table(table: Table, csv_path: str, vot_path: str) -> None:
    """Save periodicity candidates table to CSV and VOTable."""
    save_candidates_table(table, csv_path, vot_path)
