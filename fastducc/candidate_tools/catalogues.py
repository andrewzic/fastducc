# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table


def _load_psrcat_csv(psrcat_csv_path: str) -> Table:
    """
    Read ATNF PSRCAT exported CSV that is semicolon-separated and contains
    sexagesimal 'RAJ' (hms) and 'DECJ' (dms). Returns a Table with:
      ['src_name','ra_deg','dec_deg']
    Skips rows with missing RA/Dec.
    """
    t = Table.read(psrcat_csv_path, format="ascii.csv", delimiter=";", guess=False, fast_reader=False)
    col_name = "PSRJ" if "PSRJ" in t.colnames else ("NAME" if "NAME" in t.colnames else None)
    raj = "RAJ" if "RAJ" in t.colnames else None
    decj = "DECJ" if "DECJ" in t.colnames else None
    if not (col_name and raj and decj):
        raise ValueError("PSRCAT CSV must contain PSRJ, RAJ, DECJ columns")

    names, ra_deg, dec_deg = [], [], []
    for r in t:
        ra_s = str(r[raj]).strip()
        dec_s = str(r[decj]).strip()
        if (not ra_s) or (ra_s in ("*", "nan")) or ((":" not in ra_s) and (" " not in ra_s)):
            continue
        if (not dec_s) or (dec_s in ("*", "nan")):
            continue
        try:
            c = SkyCoord(ra_s, dec_s, unit=(u.hourangle, u.deg), frame="icrs")
            names.append(str(r[col_name]).strip())
            ra_deg.append(float(c.ra.deg))
            dec_deg.append(float(c.dec.deg))
        except Exception:
            continue

    return Table(
        {
            "src_name": names,
            "ra_deg": ra_deg,
            "dec_deg": dec_deg,
        }
    )


def _load_racs_votable(racs_vot_path: str) -> Table:
    """
    Read RACS components VOTable and normalise to ['src_name','ra_deg','dec_deg'].
    Uses 'Name' if present, else 'Source_ID' / 'Gaussian_ID'. Coordinates from RA/Dec (deg).
    """
    t = Table.read(racs_vot_path, format="votable")
    id_col = "Name"
    if id_col not in t.colnames:
        for c in ("Source_ID", "Gaussian_ID", "ID", "name"):
            if c in t.colnames:
                id_col = c
                break
        else:
            raise ValueError("Could not find an identifier column in RACS VOTable (e.g. 'Name', 'Source_ID' or 'Gaussian_ID').")

    if not (("RA" in t.colnames) and ("Dec" in t.colnames)):
        raise ValueError("RACS VOTable must contain 'RA' and 'Dec' columns (degrees).")

    return Table(
        {
            "src_name": [str(x) for x in t[id_col]],
            "ra_deg":   [float(x) for x in t["RA"]],
            "dec_deg":  [float(x) for x in t["Dec"]],
        }
    )


def _nearest_within_radius(target: SkyCoord, cat: SkyCoord, radius: u.Quantity) -> tuple:
    """
    Return (index, separation) of the nearest neighbour within radius.
    If none within radius: return (-1, None).
    """
    if len(cat) == 0:
        return -1, None
    sep = target.separation(cat)
    j = int(sep.argmin())
    if sep[j] <= radius:
        return j, sep[j]
    return -1, None


def annotate_observation_with_catalogues(
    out_tab: Table,
    *,
    psrcat_csv_path: Optional[str] = None,
    racs_vot_path: Optional[str] = None,
    match_radius_arcsec: float = 40.0,
    simbad_enable: bool = True,
) -> Table:
    """
    Enrich an observation-level candidate table with three new columns:
      - 'source_name'           (str)    : matched name or 'unknown'
      - 'source_offset_arcsec'  (float)  : match separation in arcsec (NaN if unknown)
      - 'source_catalog'        (str)    : 'PSRCAT' | 'RACS' | 'SIMBAD' | ''
    Priority: PSRCAT -> RACS -> SIMBAD (if still unmatched).
    """
    if not (("ra_deg" in out_tab.colnames) and ("dec_deg" in out_tab.colnames)):
        raise ValueError("Expected 'ra_deg' and 'dec_deg' columns in observation table")

    source_name = ["unknown"] * len(out_tab)
    source_offset = [float("nan")] * len(out_tab)
    source_cat = [""] * len(out_tab)

    cand_coord = SkyCoord(out_tab["ra_deg"], out_tab["dec_deg"], unit="deg", frame="icrs")
    radius = match_radius_arcsec * u.arcsec

    # 1) PSRCAT (optional)
    psrcat_tbl = None
    idx_psr = None
    sep_psr = None
    if psrcat_csv_path:
        psrcat_tbl = _load_psrcat_csv(psrcat_csv_path)
        psrcat_coord = SkyCoord(psrcat_tbl["ra_deg"], psrcat_tbl["dec_deg"], unit="deg", frame="icrs")
        if len(psrcat_coord) > 0 and len(cand_coord) > 0:
            idx_psr, sep_psr, _ = cand_coord.match_to_catalog_sky(psrcat_coord)

    # 2) RACS (optional)
    racs_tbl = None
    idx_racs = None
    sep_racs = None
    if racs_vot_path:
        racs_tbl = _load_racs_votable(racs_vot_path)
        racs_coord = SkyCoord(racs_tbl["ra_deg"], racs_tbl["dec_deg"], unit="deg", frame="icrs")
        if len(racs_coord) > 0 and len(cand_coord) > 0:
            idx_racs, sep_racs, _ = cand_coord.match_to_catalog_sky(racs_coord)

    # 3) Match candidates
    for i in range(len(out_tab)):
        tcoord = cand_coord[i]
        # PSRCAT first
        if idx_psr is not None and sep_psr[i] <= radius:
            j = int(idx_psr[i])
            source_name[i] = str(psrcat_tbl["src_name"][j])
            source_offset[i] = float(sep_psr[i].to(u.arcsec).value)
            source_cat[i] = "PSRCAT"
            continue

        # Then RACS
        if idx_racs is not None and sep_racs[i] <= radius:
            j = int(idx_racs[i])
            source_name[i] = str(racs_tbl["src_name"][j])
            source_offset[i] = float(sep_racs[i].to(u.arcsec).value)
            source_cat[i] = "RACS"
            continue

        # Finally SIMBAD (network; optional)
        if simbad_enable:
            try:
                from astroquery.simbad import Simbad
                result = Simbad.query_region(tcoord, radius=radius)
                if result is not None and len(result) > 0:
                    sc_list = []
                    for r in result:
                        try:
                            sc = SkyCoord(str(r["RA"]), str(r["DEC"]),
                                          unit=(u.hourangle, u.deg), frame="icrs")
                            sc_list.append(sc)
                        except Exception:
                            continue
                    if sc_list:
                        cat = SkyCoord([c.ra.deg for c in sc_list],
                                       [c.dec.deg for c in sc_list], unit="deg", frame="icrs")
                        j, sep = _nearest_within_radius(tcoord, cat, radius)
                        if j >= 0:
                            main_id = str(result["MAIN_ID"][j]) if "MAIN_ID" in result.colnames else str(result[0][j])
                            source_name[i] = main_id
                            source_offset[i] = float(sep.to(u.arcsec).value)
                            source_cat[i] = "SIMBAD"
                            continue
            except Exception:
                pass

    insert_at = len(out_tab.colnames)
    if "srcname" in out_tab.colnames:
        insert_at = out_tab.colnames.index("srcname") + 1

    for col_name, col_data in [
        ("source_name", source_name),
        ("source_offset_arcsec", source_offset),
        ("source_catalog", source_cat),
    ]:
        if col_name in out_tab.colnames:
            out_tab[col_name] = col_data
        else:
            out_tab.add_column(Table.Column(name=col_name, data=col_data), index=insert_at)
            insert_at += 1

    return out_tab


# Alias for compatibility
annotate_observation_with_catalogs = annotate_observation_with_catalogues
