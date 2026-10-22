# -*- coding: utf-8 -*-
"""
Compatibility alias for candidate_tools.catalogues.
"""
from fastducc.candidate_tools.catalogues import (
    _load_psrcat_csv,
    _load_racs_votable,
    _nearest_within_radius,
    annotate_observation_with_catalogues,
    annotate_observation_with_catalogs,
)

__all__ = [
    "_load_psrcat_csv",
    "_load_racs_votable",
    "_nearest_within_radius",
    "annotate_observation_with_catalogues",
    "annotate_observation_with_catalogs",
]
