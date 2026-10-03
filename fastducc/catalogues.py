import os
from importlib import resources

ENV_PSRCAT = "FASTDUCC_PSRCAT_CSV"
ENV_RACS   = "FASTDUCC_RACS_VOT"

def _res_from_root(*parts: str) -> str:
    # resolve relative to the 'fastducc' package root
    return (resources.files("fastducc").joinpath(*parts)).as_posix()

def get_psrcat_csv_path() -> str:
    """Return PSRCAT CSV path: env override or packaged file."""
    path = os.environ.get(ENV_PSRCAT)
    if path and os.path.exists(path):
        return path
    # fastducc/catalogues/psrcat/psrcat_south.csv
    return _res_from_root("catalogues", "psrcat", "psrcat_south.csv")

def get_racs_vot_path() -> str:
    """Return RACS VOTable path: env override or packaged file."""
    path = os.environ.get(ENV_RACS)
    if path and os.path.exists(path):
        return path
    # Adjust filename if you ship a default; otherwise rely on env var
    return _res_from_root("catalogues", "racs", "RACS-mid1_sources_gp_point.xml")

def get_catalog_bundle_version(default: str = "0") -> str:
    """Return text from fastducc/catalogues/VERSION if present."""
    try:
        with open(_res_from_root("catalogues", "VERSION"), "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return default

# -----------------------------------------------------------------------------
# Cached local catalogue loaders and (RACS minus PSRCAT) spatial crossmatching
# -----------------------------------------------------------------------------
_CACHED_PSRCAT_COORDS = None
_CACHED_RACS_COORDS = None

def get_cached_psrcat_coords(psrcat_csv_path: str | None = None):
    """
    Load and cache SkyCoord array of known pulsars from local PSRCAT CSV.
    Uses fastducc/catalogues/psrcat/psrcat_south.csv by default.
    """
    global _CACHED_PSRCAT_COORDS
    if _CACHED_PSRCAT_COORDS is not None:
        return _CACHED_PSRCAT_COORDS
    path = psrcat_csv_path or get_psrcat_csv_path()
    if path and os.path.exists(path):
        try:
            from astropy.table import Table
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            t = Table.read(path, format="ascii.csv", delimiter=";", guess=False, fast_reader=False)
            col_name = "PSRJ" if "PSRJ" in t.colnames else ("NAME" if "NAME" in t.colnames else None)
            raj = "RAJ" if "RAJ" in t.colnames else None
            decj = "DECJ" if "DECJ" in t.colnames else None
            if col_name and raj and decj:
                ra_deg, dec_deg = [], []
                for r in t:
                    ra_s = str(r[raj]).strip()
                    dec_s = str(r[decj]).strip()
                    if (not ra_s) or (ra_s in ("*", "nan")) or ((":" not in ra_s) and (" " not in ra_s)):
                        continue
                    if (not dec_s) or (dec_s in ("*", "nan")):
                        continue
                    try:
                        c = SkyCoord(ra_s, dec_s, unit=(u.hourangle, u.deg), frame="icrs")
                        ra_deg.append(float(c.ra.deg))
                        dec_deg.append(float(c.dec.deg))
                    except Exception:
                        continue
                if len(ra_deg) > 0:
                    _CACHED_PSRCAT_COORDS = SkyCoord(ra_deg, dec_deg, unit="deg", frame="icrs")
        except Exception as e:
            print(f"[Catalogues] Warning: Failed to load local PSRCAT from '{path}': {e}")
    return _CACHED_PSRCAT_COORDS

def get_cached_racs_coords(
    racs_vot_path: str | None = None,
    center_coord = None,
    cone_radius_deg: float = 3.0,
):
    """
    Load and cache SkyCoord array of persistent continuum sources from local RACS VOTable.
    Uses fastducc/catalogues/racs/RACS-mid1_sources_gp_point.xml by default.
    If center_coord is given, returns the subset within cone_radius_deg of that position.
    """
    global _CACHED_RACS_COORDS
    if _CACHED_RACS_COORDS is None:
        path = racs_vot_path or get_racs_vot_path()
        if path and os.path.exists(path):
            try:
                from astropy.table import Table
                from astropy.coordinates import SkyCoord
                t = Table.read(path, format="votable")
                if "RA" in t.colnames and "Dec" in t.colnames:
                    _CACHED_RACS_COORDS = SkyCoord(
                        [float(x) for x in t["RA"]],
                        [float(x) for x in t["Dec"]],
                        unit="deg", frame="icrs"
                    )
            except Exception as e:
                print(f"[Catalogues] Warning: Failed to load local RACS from '{path}': {e}")

    if _CACHED_RACS_COORDS is not None and center_coord is not None:
        import astropy.units as u
        sep = center_coord.separation(_CACHED_RACS_COORDS)
        return _CACHED_RACS_COORDS[sep <= (cone_radius_deg * u.deg)]
    return _CACHED_RACS_COORDS

def is_racs_minus_psrcat(
    ra_deg: float,
    dec_deg: float,
    racs_coords,
    psrcat_coords,
    match_radius_arcsec: float = 30.0,
) -> bool:
    """
    Check if a candidate position matches a known RACS persistent continuum source,
    excluding any source that matches a known pulsar in PSRCAT.
    """
    if racs_coords is None or len(racs_coords) == 0:
        return False

    from astropy.coordinates import SkyCoord
    import astropy.units as u
    import numpy as np

    cand = SkyCoord(ra_deg, dec_deg, unit="deg", frame="icrs")
    radius = match_radius_arcsec * u.arcsec

    # 1. If matches PSRCAT, it's a known pulsar -> NOT in (RACS - PSRCAT)
    if psrcat_coords is not None and len(psrcat_coords) > 0:
        sep_p = cand.separation(psrcat_coords)
        if np.any(sep_p <= radius):
            return False

    # 2. Check if matches RACS
    sep_r = cand.separation(racs_coords)
    return bool(np.any(sep_r <= radius))

