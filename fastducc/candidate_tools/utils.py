# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import math
from typing import List, Dict, Any, Tuple

import numpy as np
from astropy.table import Table, MaskedColumn, vstack
from fastducc import wcs as ducc_wcs


def _unmask_table(tab: Table) -> Table:
    """Ensure table has no MaskedColumn objects, converting them to standard Columns."""
    if tab is None:
        return tab
    if any(isinstance(tab[c], MaskedColumn) for c in tab.colnames):
        return tab.filled()
    return tab


def _sanitize_row_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Convert numpy / masked scalars in a row dict to native Python types."""
    clean = {}
    for k, v in d.items():
        if np.ma.is_masked(v):
            clean[k] = None
        elif hasattr(v, "item"):
            try:
                clean[k] = v.item()
            except Exception:
                clean[k] = v
        else:
            clean[k] = v
    return clean


def _row_value_to_python(val: Any) -> Any:
    """
    Convert an Astropy Table row cell to a plain Python type.
    - Quantities -> numeric values in their current units
    - numpy scalars -> Python scalars
    - masked -> None
    """
    try:
        import numpy.ma as ma
        if isinstance(val, ma.MaskedConstant):
            return None
        if hasattr(val, "mask") and getattr(val, "mask", False) is True:
            return None
    except Exception:
        pass

    if hasattr(val, "unit") and hasattr(val, "value"):
        scalar = val.value
        if isinstance(scalar, np.generic):
            return scalar.item()
        return scalar

    if isinstance(val, np.generic):
        return val.item()

    if isinstance(val, (bytes, bytearray)):
        return val.decode("utf-8", errors="replace")

    return val


def _sexagesimal_from_deg(ra_deg: float, dec_deg: float) -> Tuple[str, str, str]:
    """(ra_hms, dec_dms, srcname) from degrees using ducc_wcs helpers."""
    ra_rad = math.radians(float(ra_deg))
    dec_rad = math.radians(float(dec_deg))
    ra_hms, dec_dms = ducc_wcs.rad_to_hmsdms(ra_rad, dec_rad, dp=1)
    srcname = ducc_wcs.hmsdms_to_srcname(ra_hms, dec_dms)
    return ra_hms, dec_dms, srcname


def _ensure_sexagesimal_on_rows(
    rows: List[Dict[str, Any]],
    ra_key: str = "ra_deg",
    dec_key: str = "dec_deg",
    ra_hms_key: str = "ra_hms",
    dec_dms_key: str = "dec_dms",
    src_key: str = "srcname"
) -> None:
    """Ensure each dict row has ra_hms, dec_dms, and srcname. Fill if missing/blank."""
    for r in rows:
        have_ra = r.get(ra_key, None) is not None
        have_dec = r.get(dec_key, None) is not None
        if not have_ra or not have_dec:
            continue
        ra_hms = str(r.get(ra_hms_key, "") or "")
        dec_dms = str(r.get(dec_dms_key, "") or "")
        srcname = str(r.get(src_key, "") or "")
        if ra_hms and dec_dms and srcname:
            continue
        ra_hms_new, dec_dms_new, srcname_new = _sexagesimal_from_deg(float(r[ra_key]), float(r[dec_key]))
        if not ra_hms:
            r[ra_hms_key] = ra_hms_new
        if not dec_dms:
            r[dec_dms_key] = dec_dms_new
        if not srcname:
            r[src_key] = srcname_new


def _split_list_str(s: Any) -> List[str]:
    """Parse comma-separated string -> list[str]."""
    if s is None:
        return []
    s = str(s).strip().strip('"')
    if not s:
        return []
    return [t.strip() for t in s.split(",") if t.strip()]


def _split_list_int(s: Any) -> List[int]:
    """Parse comma-separated string -> list[int] (skip non-ints)."""
    out = []
    for tok in _split_list_str(s):
        try:
            out.append(int(tok))
        except Exception:
            pass
    return out


def parse_candidate_filename(path: str, *, require: str | None = None) -> dict:
    """
    Parse pipeline candidate/super-summary filenames.

    Supports boxcar, variance, and periodicity candidate filenames.
    """
    fname = os.path.basename(path)
    info = {
        "field": "",
        "sbid": "",
        "beam": "",
        "scan_id": "",
        "kind": "",
        "dm": 0.0,
        "is_all": False,
        "is_super": False,
    }

    m_field_prefix = re.match(r'^(?P<field>[^.]+)\.', fname)
    if m_field_prefix:
        info["field"] = m_field_prefix.group("field")

    m_sbid = re.search(r'(SB\d{5,})', fname)
    if m_sbid:
        info["sbid"] = m_sbid.group(1)

    m_beam = re.search(r'(beam\d+)', fname)
    if m_beam:
        info["beam"] = m_beam.group(1)

    m_dm = re.search(r'_dm([\d.]+)', fname)
    if m_dm:
        try:
            info["dm"] = float(m_dm.group(1))
        except Exception:
            pass

    m_scan = re.search(r'(?<!\d)(\d{14})(?!\d)', fname)
    if not m_scan:
        m_scan = re.search(r'(?<!\d)(\d{14})(?!\d)', os.path.abspath(path))
    if m_scan:
        info["scan_id"] = m_scan.group(1)
    else:
        m_combined = re.search(r'(\w+_combined)', fname) or re.search(r'(\w+_combined)', os.path.abspath(path))
        if m_combined:
            info["scan_id"] = m_combined.group(1)

    m_kind_super = re.search(r'_(boxcar|variance|periodicity)_super_summary\.(?:vot|xml|csv)$', fname, re.IGNORECASE)
    m_kind_all = re.search(r'_(boxcar|variance|periodicity)_all\.(?:vot|xml|csv)$', fname, re.IGNORECASE)

    if m_kind_super:
        info["kind"] = m_kind_super.group(1).lower()
        info["is_super"] = True
    elif m_kind_all:
        info["kind"] = m_kind_all.group(1).lower()
        info["is_all"] = True

    if require == "super" and not info["is_super"]:
        return {k: (False if isinstance(v, bool) else (0.0 if isinstance(v, (int, float)) else "")) for k, v in info.items()}
    if require == "all" and not info["is_all"]:
        return {k: (False if isinstance(v, bool) else (0.0 if isinstance(v, (int, float)) else "")) for k, v in info.items()}
    return info


def _read_csv_tables(paths: List[str]) -> List[Table]:
    """Read a list of CSV files into Astropy Tables (skip missing)."""
    tables = []
    for p in paths:
        try:
            t = Table.read(p, format="csv")
            tables.append(t)
        except Exception as e:
            print(f"[Consolidation] Skipping '{p}' (read error: {e})")
    return tables


def _stack_tables_or_empty(tables: List[Table]) -> Table:
    """Vstack a list of tables (outer join), return a filled table or empty schema."""
    valid_tables = [t for t in tables if len(t) > 0]
    if len(valid_tables) == 0:
        return Table(names=["x","y","l","m","ra_rad","dec_rad","ra_deg","dec_deg",
                            "ra_hms","dec_dms","dm","snr","std","width_samples",
                            "time_start","time_end","time_center","duration","t0_idx","t1_idx_excl",
                            "center_idx","phase_center_field","chunk_id","algo"],
                     dtype=[int,int,float,float,float,float,float,float,
                            "U20","U20",float,float,float,int,
                            float,float,float,float,int,int,
                            int,"U64",int,"U16"])

    for t in valid_tables:
        if "srcname" in t.colnames:
            t["srcname"] = np.array(t["srcname"], dtype="U64")

    T = vstack(valid_tables, join_type="outer", metadata_conflicts="silent")
    T.fill_value = np.nan
    T = T.filled()
    if "srcname" in T.colnames:
        T["srcname"] = np.array(T["srcname"], dtype="U64")

    if "dm" in T.colnames:
        T["dm"] = np.nan_to_num(np.array(T["dm"], dtype=float), nan=0.0)
    else:
        T["dm"] = np.zeros(len(T), dtype=float)
    return T
