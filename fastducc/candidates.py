# -*- coding: utf-8 -*-
"""
fastducc.candidates
===================
Candidate extraction, annotation, table serialization, summary plotting,
and observation aggregation.

This module provides the public API for candidate processing in fastducc.
The underlying implementations are decomposed into cohesive submodules
in `fastducc.candidate_tools`.
"""
from __future__ import annotations

# Re-export all candidate tools for backward compatibility
from fastducc.candidate_tools.utils import (
    _unmask_table,
    _sanitize_row_dict,
    _row_value_to_python,
    _sexagesimal_from_deg,
    _ensure_sexagesimal_on_rows,
    _split_list_str,
    _split_list_int,
    parse_candidate_filename,
    _read_csv_tables,
    _stack_tables_or_empty,
)
from fastducc.candidate_tools.tables import (
    candidates_to_astropy_table,
    candidates_to_astropy_table_periodicity,
    astropy_table_to_candidates,
    save_candidates_table,
    save_periodicity_candidates_table,
)
from fastducc.candidate_tools.snippets import (
    make_stdmap_snippet,
    extract_candidate_snippets,
    compute_candidate_local_rms,
    save_candidate_lightcurves,
)
from fastducc.candidate_tools.plotting import (
    save_candidate_summary,
    save_candidate_snippet_products,
)
from fastducc.candidate_tools.catalogs import (
    _load_psrcat_csv,
    _load_racs_votable,
    _nearest_within_radius,
    annotate_observation_with_catalogs,
)
from fastducc.candidate_tools.clustering import (
    _connected_components_from_pairs,
    _time_groups,
    _cluster_indices_by_sky,
)
from fastducc.candidate_tools.regions import (
    write_obs_region_files,
)
from fastducc.candidate_tools.aggregation import (
    consolidate_chunk_catalogues,
    generate_obs_variance_candidate_products,
    aggregate_beam_candidate_tables,
    aggregate_observation,
    aggregate_observation_from_super_summaries,
)
from fastducc.candidate_tools.organization import (
    organize_candidate_outputs,
)

__all__ = [
    # Table serialization & conversions
    "candidates_to_astropy_table",
    "candidates_to_astropy_table_periodicity",
    "astropy_table_to_candidates",
    "save_candidates_table",
    "save_periodicity_candidates_table",
    # Snippet & lightcurve extraction
    "make_stdmap_snippet",
    "extract_candidate_snippets",
    "compute_candidate_local_rms",
    "save_candidate_lightcurves",
    # Summary plotting & visualization
    "save_candidate_summary",
    "save_candidate_snippet_products",
    # Catalog annotation & crossmatch
    "annotate_observation_with_catalogs",
    # Region files (DS9 & Karma)
    "write_obs_region_files",
    # Aggregation & consolidation
    "consolidate_chunk_catalogues",
    "generate_obs_variance_candidate_products",
    "aggregate_beam_candidate_tables",
    "aggregate_observation",
    "aggregate_observation_from_super_summaries",
    # Output organization
    "organize_candidate_outputs",
    # Filename parsing & helpers
    "parse_candidate_filename",
    "_unmask_table",
    "_sanitize_row_dict",
    "_row_value_to_python",
    "_sexagesimal_from_deg",
    "_ensure_sexagesimal_on_rows",
    "_split_list_str",
    "_split_list_int",
    "_read_csv_tables",
    "_stack_tables_or_empty",
    "_load_psrcat_csv",
    "_load_racs_votable",
    "_nearest_within_radius",
    "_connected_components_from_pairs",
    "_time_groups",
    "_cluster_indices_by_sky",
]
