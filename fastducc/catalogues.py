import os
from importlib import resources

ENV_PSRCAT = "FASTDUCC_PSRCAT_CSV"
ENV_RACS   = "FASTDUCC_RACS_VOT"

def _res(package: str, name: str) -> str:
    return resources.files(package).joinpath(name).as_posix()

def get_psrcat_csv_path() -> str:
    path = os.environ.get(ENV_PSRCAT)
    if path and os.path.exists(path):
        return path
    return _res("fastducc.catalogues.psrcat", "psrcat_south.csv")

def get_racs_vot_path() -> str:
    path = os.environ.get(ENV_RACS)
    if path and os.path.exists(path):
        return path
    return _res("fastducc.catalogues.racs", "RACS-mid1_sources_gp_point.xml")

def get_catalog_bundle_version(default: str = "0") -> str:
    try:
        with open(_res("fastducc.catalogues", "VERSION"), "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return default
