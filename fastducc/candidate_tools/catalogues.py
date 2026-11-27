# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from typing import Optional

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

logger = logging.getLogger("fastducc.candidate_tools.catalogues")


def _load_psrcat_csv(psrcat_csv_path: str) -> Table:
    """
    Read ATNF PSRCAT exported CSV that is semicolon-separated and contains
    sexagesimal 'RAJ' (hms) and 'DECJ' (dms). Returns a Table with:
      ['src_name','ra_deg','dec_deg','p0','dm']
    Skips rows with missing RA/Dec.
    """
    import numpy as np

    t = Table.read(psrcat_csv_path, format="ascii.csv", delimiter=";", guess=False, fast_reader=False)
    col_name = "PSRJ" if "PSRJ" in t.colnames else ("NAME" if "NAME" in t.colnames else None)
    raj = "RAJ" if "RAJ" in t.colnames else None
    decj = "DECJ" if "DECJ" in t.colnames else None
    if not (col_name and raj and decj):
        raise ValueError("PSRCAT CSV must contain PSRJ, RAJ, DECJ columns")

    names, ra_deg, dec_deg, p0_s, dm_pc = [], [], [], [], []
    has_f0 = "F0" in t.colnames
    has_dm = "DM" in t.colnames
    for r in t:
        ra_s = str(r[raj]).strip()
        dec_s = str(r[decj]).strip()
        if (not ra_s) or (ra_s in ("*", "nan")) or ((":" not in ra_s) and (" " not in ra_s)):
            continue
        if (not dec_s) or (dec_s in ("*", "nan")):
            continue
        try:
            c = SkyCoord(ra_s, dec_s, unit=(u.hourangle, u.deg), frame="icrs")
            r_ra = float(c.ra.deg)
            r_dec = float(c.dec.deg)
        except Exception as exc:
            logger.warning("Failed to parse coordinates (RAJ='%s', DECJ='%s') in PSRCAT catalogue: %s", ra_s, dec_s, exc)
            continue

        p0_val = float("nan")
        if has_f0:
            f0_s = str(r["F0"]).strip()
            if f0_s and f0_s != "*" and f0_s != "nan":
                try:
                    f0_val = float(f0_s)
                    if f0_val > 0:
                        p0_val = 1.0 / f0_val
                except Exception as exc:
                    logger.warning("Failed to parse F0 value '%s' for source '%s' in PSRCAT catalogue: %s", f0_s, r[col_name], exc)

        dm_val = float("nan")
        if has_dm:
            dm_s = str(r["DM"]).strip()
            if dm_s and dm_s != "*" and dm_s != "nan":
                try:
                    dm_val = float(dm_s)
                except Exception as exc:
                    logger.warning("Failed to parse DM value '%s' for source '%s' in PSRCAT catalogue: %s", dm_s, r[col_name], exc)

        names.append(str(r[col_name]).strip())
        ra_deg.append(r_ra)
        dec_deg.append(r_dec)
        p0_s.append(p0_val)
        dm_pc.append(dm_val)

    return Table(
        {
            "src_name": names,
            "ra_deg": ra_deg,
            "dec_deg": dec_deg,
            "p0": p0_s,
            "dm": dm_pc,
        }
    )


def _load_racs_votable(racs_vot_path: str) -> Table:
    """
    Read RACS components VOTable and normalise to ['src_name','ra_deg','dec_deg','flux_mjy','flux_jy'].
    Uses 'Name' if present, else 'Source_ID' / 'Gaussian_ID'. Coordinates from RA/Dec (deg).
    """
    import numpy as np

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

    flux_col = None
    for fc in ("Total_flux", "Peak_flux", "flux", "Flux"):
        if fc in t.colnames:
            flux_col = fc
            break

    flux_mjy = []
    if flux_col is not None:
        for val in t[flux_col]:
            try:
                flux_mjy.append(float(val))
            except Exception as exc:
                logger.warning("Failed to parse flux value '%s' in RACS VOTable: %s", val, exc)
                flux_mjy.append(float("nan"))
    else:
        flux_mjy = [float("nan")] * len(t)

    flux_jy = [f * 1e-3 if np.isfinite(f) else float("nan") for f in flux_mjy]

    return Table(
        {
            "src_name": [str(x) for x in t[id_col]],
            "ra_deg":   [float(x) for x in t["RA"]],
            "dec_deg":  [float(x) for x in t["Dec"]],
            "flux_mjy": flux_mjy,
            "flux_jy":  flux_jy,
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


_PSRCAT_HARMONIC_RATIOS = (1.0, 0.5, 1.0 / 3.0, 0.25, 0.2, 2.0, 3.0, 4.0, 5.0)


def _is_harmonic_period_match(
    cand_period_s: float,
    cat_period_s: float,
    period_tol_frac: float = 0.02,
) -> bool:
    """Return True if cand_period_s matches cat_period_s at fundamental or harmonic ratio."""
    import math

    if not (math.isfinite(cand_period_s) and cand_period_s > 0.0 and math.isfinite(cat_period_s) and cat_period_s > 0.0):
        return False
    ratio = cand_period_s / cat_period_s
    for target in _PSRCAT_HARMONIC_RATIOS:
        if abs(ratio - target) / target <= period_tol_frac:
            return True
    return False


def annotate_observation_with_catalogues(
    out_tab: Table,
    *,
    psrcat_csv_path: Optional[str] = None,
    racs_vot_path: Optional[str] = None,
    match_radius_arcsec: float = 40.0,
    psrcat_match_radius_arcsec: Optional[float] = None,
    psrcat_period_tol_frac: float = 0.02,
    simbad_enable: bool = True,
) -> Table:
    """
    Enrich an observation-level candidate table with catalogue cross-matches:
      - 'source_name'           (str)    : matched name or 'unknown'
      - 'source_offset_arcsec'  (float)  : match separation in arcsec (NaN if unknown)
      - 'source_catalog'        (str)    : 'PSRCAT' | 'RACS' | 'SIMBAD' | ''
    Priority: PSRCAT -> RACS -> SIMBAD (if still unmatched).

    RACS and SIMBAD use the tight ``match_radius_arcsec`` (default 40").
    PSRCAT uses ``psrcat_match_radius_arcsec`` (falling back to ``match_radius_arcsec``
    if omitted); when candidate periods are present in ``out_tab`` (``period_s`` or
    ``f_hz``), PSRCAT matches beyond ``match_radius_arcsec`` require a fundamental or
    harmonic period match within ``psrcat_period_tol_frac`` (default 2%).
    """
    import numpy as np

    if not (("ra_deg" in out_tab.colnames) and ("dec_deg" in out_tab.colnames)):
        raise ValueError("Expected 'ra_deg' and 'dec_deg' columns in observation table")

    source_name = ["unknown"] * len(out_tab)
    source_offset = [float("nan")] * len(out_tab)
    source_cat = [""] * len(out_tab)

    is_psrcat = [False] * len(out_tab)
    psr_name = [""] * len(out_tab)
    psr_sep = [float("nan")] * len(out_tab)
    psr_p0 = [float("nan")] * len(out_tab)
    psr_dm = [float("nan")] * len(out_tab)

    racs_m = [""] * len(out_tab)
    racs_sep = [float("nan")] * len(out_tab)
    racs_flx_jy = [float("nan")] * len(out_tab)
    racs_flx_mjy = [float("nan")] * len(out_tab)

    cand_coord = SkyCoord(out_tab["ra_deg"], out_tab["dec_deg"], unit="deg", frame="icrs")
    radius = float(match_radius_arcsec) * u.arcsec
    eff_psrcat_radius_arcsec = (
        float(psrcat_match_radius_arcsec)
        if (psrcat_match_radius_arcsec is not None and float(psrcat_match_radius_arcsec) > 0.0)
        else float(match_radius_arcsec)
    )
    psrcat_radius = eff_psrcat_radius_arcsec * u.arcsec

    # Extract candidate periods if available for harmonic PSRCAT matching
    cand_periods = np.full(len(out_tab), np.nan, dtype=np.float64)
    p_col = next((c for c in ("period_s", "p_refined", "period", "p0") if c in out_tab.colnames), None)
    f_col = next((c for c in ("f_hz", "f0", "freq_hz") if c in out_tab.colnames), None)
    if p_col is not None:
        for i in range(len(out_tab)):
            try:
                val = float(out_tab[p_col][i])
                if np.isfinite(val) and val > 0.0:
                    cand_periods[i] = val
            except (TypeError, ValueError):
                pass
    elif f_col is not None:
        for i in range(len(out_tab)):
            try:
                val = float(out_tab[f_col][i])
                if np.isfinite(val) and val > 0.0:
                    cand_periods[i] = 1.0 / val
            except (TypeError, ValueError):
                pass

    # 1) PSRCAT (optional): search all pulsars within psrcat_radius for harmonic/positional matching
    psrcat_tbl = None
    psr_candidates_per_row: dict[int, list[tuple[float, int]]] = {}
    if psrcat_csv_path:
        psrcat_tbl = _load_psrcat_csv(psrcat_csv_path)
        psrcat_coord = SkyCoord(psrcat_tbl["ra_deg"], psrcat_tbl["dec_deg"], unit="deg", frame="icrs")
        if len(psrcat_coord) > 0 and len(cand_coord) > 0:
            idx_c, idx_p, sep_2d, _ = psrcat_coord.search_around_sky(cand_coord, psrcat_radius)
            for ci, pi, sep_val in zip(idx_c, idx_p, sep_2d.to(u.arcsec).value):
                psr_candidates_per_row.setdefault(int(ci), []).append((float(sep_val), int(pi)))
            for ci in psr_candidates_per_row:
                psr_candidates_per_row[ci].sort(key=lambda item: item[0])

    # 2) RACS (optional): always uses tight match_radius_arcsec
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

        # Check PSRCAT match (prefer nearest harmonic period match within psrcat_radius;
        # fall back to positional-only match strictly within match_radius_arcsec when no period is available)
        if psrcat_tbl is not None and i in psr_candidates_per_row:
            cand_p = float(cand_periods[i])
            has_cand_p = np.isfinite(cand_p) and cand_p > 0.0
            chosen_match: Optional[tuple[float, int]] = None

            if has_cand_p:
                for sep_a, j in psr_candidates_per_row[i]:
                    cat_p0 = float(psrcat_tbl["p0"][j]) if "p0" in psrcat_tbl.colnames else float("nan")
                    if _is_harmonic_period_match(cand_p, cat_p0, period_tol_frac=psrcat_period_tol_frac):
                        chosen_match = (sep_a, j)
                        break
                if chosen_match is None:
                    # Positional fallback inside tight radius only if PSRCAT lacks P0
                    for sep_a, j in psr_candidates_per_row[i]:
                        if sep_a <= float(match_radius_arcsec):
                            cat_p0 = float(psrcat_tbl["p0"][j]) if "p0" in psrcat_tbl.colnames else float("nan")
                            if not np.isfinite(cat_p0):
                                chosen_match = (sep_a, j)
                                break
            else:
                for sep_a, j in psr_candidates_per_row[i]:
                    if sep_a <= float(match_radius_arcsec):
                        chosen_match = (sep_a, j)
                        break

            if chosen_match is not None:
                m_sep, j = chosen_match
                m_name = str(psrcat_tbl["src_name"][j])
                is_psrcat[i] = True
                psr_name[i] = m_name
                psr_sep[i] = m_sep
                if "p0" in psrcat_tbl.colnames:
                    psr_p0[i] = float(psrcat_tbl["p0"][j])
                if "dm" in psrcat_tbl.colnames:
                    psr_dm[i] = float(psrcat_tbl["dm"][j])

                source_name[i] = m_name
                source_offset[i] = m_sep
                source_cat[i] = "PSRCAT"

        # Check RACS match
        if idx_racs is not None and sep_racs[i] <= radius:
            j_r = int(idx_racs[i])
            r_name = str(racs_tbl["src_name"][j_r])
            r_sep = float(sep_racs[i].to(u.arcsec).value)
            racs_m[i] = r_name
            racs_sep[i] = r_sep
            if "flux_jy" in racs_tbl.colnames:
                racs_flx_jy[i] = float(racs_tbl["flux_jy"][j_r])
            if "flux_mjy" in racs_tbl.colnames:
                racs_flx_mjy[i] = float(racs_tbl["flux_mjy"][j_r])

            if source_cat[i] == "":
                source_name[i] = r_name
                source_offset[i] = r_sep
                source_cat[i] = "RACS"

        # If still unmatched, try SIMBAD (network; optional)
        if source_cat[i] == "" and simbad_enable:
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
                        except Exception as exc:
                            logger.warning(
                                "Failed to parse SIMBAD coordinate record (RA='%s', DEC='%s'): %s",
                                r.get("RA"),
                                r.get("DEC"),
                                exc,
                            )
                            continue
                    if sc_list:
                        cat = SkyCoord([c.ra.deg for c in sc_list],
                                       [c.dec.deg for c in sc_list], unit="deg", frame="icrs")
                        j_s, sep_s = _nearest_within_radius(tcoord, cat, radius)
                        if j_s >= 0:
                            main_id = str(result["MAIN_ID"][j_s]) if "MAIN_ID" in result.colnames else str(result[0][j_s])
                            source_name[i] = main_id
                            source_offset[i] = float(sep_s.to(u.arcsec).value)
                            source_cat[i] = "SIMBAD"
            except Exception as exc:
                logger.warning("SIMBAD cross-matching query failed: %s", exc)

    insert_at = len(out_tab.colnames)
    if "srcname" in out_tab.colnames:
        insert_at = out_tab.colnames.index("srcname") + 1

    extra_cols = [
        ("source_name", source_name),
        ("source_offset_arcsec", source_offset),
        ("source_catalog", source_cat),
        ("is_psrcat_match", is_psrcat),
        ("psrcat_match_name", psr_name),
        ("psrcat_name", psr_name),
        ("psrcat_sep_arcsec", psr_sep),
        ("psrcat_period_s", psr_p0),
        ("psrcat_p0", psr_p0),
        ("psrcat_dm", psr_dm),
        ("racs_match", racs_m),
        ("racs_match_name", racs_m),
        ("racs_source_name", racs_m),
        ("racs_sep_arcsec", racs_sep),
        ("racs_flux_jy", racs_flx_jy),
        ("racs_flux_mjy", racs_flx_mjy),
    ]
    for col_name, col_data in extra_cols:
        if col_name in out_tab.colnames:
            out_tab[col_name] = col_data
        else:
            out_tab.add_column(Table.Column(name=col_name, data=col_data), index=insert_at)
            insert_at += 1

    return out_tab


# Re-export classifier utilities for backwards compatibility
from fastducc.candidate_tools.classifier import (
    evaluate_cluster_harmonic_structure,
    sample_local_continuum_flux,
    find_beam_continuum_image,
    classify_candidate,
)

# Alias for compatibility
annotate_observation_with_catalogs = annotate_observation_with_catalogues

__all__ = [
    "_load_psrcat_csv",
    "_load_racs_votable",
    "_nearest_within_radius",
    "annotate_observation_with_catalogues",
    "annotate_observation_with_catalogs",
    "evaluate_cluster_harmonic_structure",
    "sample_local_continuum_flux",
    "find_beam_continuum_image",
    "classify_candidate",
]
