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
