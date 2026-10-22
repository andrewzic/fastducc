# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import os
import glob
import shutil
import csv
import collections
from typing import List, Dict, Any, Optional

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from fastducc.candidate_tools.utils import (
    _sanitize_row_dict,
    _row_value_to_python,
    _unmask_table,
    parse_candidate_filename,
)

logger = logging.getLogger(__name__)


def organize_candidate_outputs(
    obs_root: str,
    kind: str,
    out_tab: Table,
    out_dir: str,
    *,
    link_mode: str = "symlink",
    structure: str = "categorized",
    clean_existing: bool = True,
    obs_rows_sub: Optional[List[List[Dict[str, Any]]]] = None,
    max_n_candidates_per_source: Optional[int] = 100,
) -> None:
    """
    Organize per-candidate artefacts (summary plots, lightcurves, FITS cutouts,
    fullframe FITS images, dynamic spectra, catalog entries) under <out_dir>/<kind>/<srcname>/...
    """
    kind = kind.lower().strip()
    if out_tab is None or len(out_tab) == 0:
        logger.info(f"[ObsOrganize] No candidates in output table for kind={kind}; skipping organization.")
        return

    kind_dir = os.path.join(out_dir, kind)
    os.makedirs(kind_dir, exist_ok=True)

    abs_out_dir = os.path.abspath(out_dir)

    cand_dir_set = set()
    for pat in [
        os.path.join(obs_root, "candidates"),
        os.path.join(obs_root, "*", "candidates"),
    ]:
        for d in glob.glob(pat):
            if os.path.isdir(d):
                abs_d = os.path.abspath(d)
                if abs_d == abs_out_dir or abs_d.startswith(abs_out_dir + os.sep):
                    continue
                cand_dir_set.add(d)

    if not cand_dir_set:
        for d in glob.glob(os.path.join(obs_root, "*", "*", "candidates")):
            if os.path.isdir(d):
                abs_d = os.path.abspath(d)
                if abs_d == abs_out_dir or abs_d.startswith(abs_out_dir + os.sep):
                    continue
                cand_dir_set.add(d)

    all_artefacts: List[str] = []
    all_comp_rows: List[Dict[str, Any]] = []

    for cdir in sorted(cand_dir_set):
        try:
            other_family = ["_var_", "_variance_", "_periodicity_"] if kind == "boxcar" else ["_boxcar_", "_box_", "_periodicity_"]
            cdir_items = [f for f in os.listdir(cdir) if not any(t in f.lower() for t in other_family)]

            for item in cdir_items:
                if any(k in item for k in ["_cand_", "_summary", "_lc_", "_cutout", "_fullframe", ".ds"]):
                    all_artefacts.append(os.path.join(cdir, item))

            cand_tables = [f for f in cdir_items if f.endswith(("_summary.csv", "_summary.vot")) and "_super_" not in f]
            if not cand_tables:
                cand_tables = [f for f in cdir_items if f.endswith(("_all.csv", "_all.vot", "_super_summary.csv", "_super_summary.vot", "_candidates.csv", "_candidates.vot"))]
            cand_tables = [f for f in cand_tables if not any(t in f for t in ["_obs_", "_full_candidate_list"])]

            for item in cand_tables:
                item_path = os.path.join(cdir, item)
                meta_comp = parse_candidate_filename(item_path)
                try:
                    if item.lower().endswith(".csv"):
                        with open(item_path, "r", encoding="utf-8", errors="replace") as f:
                            rows = [_sanitize_row_dict(r) for r in csv.DictReader(f)]
                    else:
                        fmt = "votable" if item.lower().endswith((".vot", ".xml")) else "csv"
                        t = _unmask_table(Table.read(item_path, format=fmt))
                        rows = [_sanitize_row_dict({c: _row_value_to_python(r[c]) for c in t.colnames}) for r in t]

                    for d in rows:
                        d["scan_id"] = d.get("scan_id") or meta_comp.get("scan_id")
                        d["beam"] = d.get("beam") or meta_comp.get("beam")
                        if (d.get("dm") is None or str(d.get("dm")).strip() in ("", "nan", "None")) and meta_comp.get("dm") is not None:
                            d["dm"] = meta_comp["dm"]
                        all_comp_rows.append(d)
                except Exception:
                    pass
        except Exception:
            continue

    logger.info(f"[ObsOrganize] Indexed {len(all_artefacts)} artefact paths and {len(all_comp_rows)} component candidate entries under {obs_root}")

    cand_name_to_indices: Dict[str, List[int]] = collections.defaultdict(list)
    for idx, row in enumerate(out_tab):
        s = str(row["srcname"]).strip()
        if s:
            cand_name_to_indices[s].append(idx)
    if obs_rows_sub:
        for idx, sub_list in enumerate(obs_rows_sub):
            for sub_r in sub_list:
                s = str(sub_r.get("srcname", "")).strip()
                if s:
                    cand_name_to_indices[s].append(idx)

    artefacts_by_cand_idx: Dict[int, List[str]] = collections.defaultdict(list)
    for art_path in all_artefacts:
        art_name = os.path.basename(art_path)
        art_name_lower = art_name.lower()

        if kind == "boxcar":
            if any(tag in art_name_lower for tag in ["_var_", "_variance_", "_periodicity_", "_period_"]):
                continue
        elif kind == "variance":
            if any(tag in art_name_lower for tag in ["_boxcar_", "_box_", "_periodicity_", "_period_"]):
                continue
        elif kind == "periodicity":
            if any(tag in art_name_lower for tag in ["_boxcar_", "_box_", "_var_", "_variance_"]):
                continue

        matched_cand_indices = set()
        if "cand_" in art_name:
            after_cand = art_name.split("cand_", 1)[1]
            base_cand = os.path.splitext(after_cand)[0]
            tokens = base_cand.split("_")
            accum = []
            for tok in tokens:
                accum.append(tok)
                prefix = "_".join(accum)
                if prefix in cand_name_to_indices:
                    matched_cand_indices.update(cand_name_to_indices[prefix])
        if not matched_cand_indices:
            base_name = os.path.splitext(art_name)[0]
            tokens = base_name.split("_")
            for tok in tokens:
                if tok in cand_name_to_indices:
                    matched_cand_indices.update(cand_name_to_indices[tok])

        for c_idx in matched_cand_indices:
            artefacts_by_cand_idx[c_idx].append(art_path)

    comp_matches_by_cand: Dict[int, List[Dict[str, Any]]] = collections.defaultdict(list)
    for comp_r in all_comp_rows:
        comp_src = str(comp_r.get("srcname", "")).strip()
        if comp_src in cand_name_to_indices:
            for c_idx in cand_name_to_indices[comp_src]:
                comp_matches_by_cand[c_idx].append(comp_r)

    if len(all_comp_rows) > 0 and len(out_tab) > 0:
        try:
            cand_ra_arr = np.array([float(r["ra_deg"]) for r in out_tab], dtype=float)
            cand_dec_arr = np.array([float(r["dec_deg"]) for r in out_tab], dtype=float)
            cand_coords = SkyCoord(cand_ra_arr * u.deg, cand_dec_arr * u.deg, frame="icrs")

            comp_ra_arr = np.array([float(r.get("ra_deg", np.nan)) for r in all_comp_rows], dtype=float)
            comp_dec_arr = np.array([float(r.get("dec_deg", np.nan)) for r in all_comp_rows], dtype=float)
            valid_comp_mask = ~np.isnan(comp_ra_arr) & ~np.isnan(comp_dec_arr)

            if np.any(valid_comp_mask):
                valid_comp_indices = np.where(valid_comp_mask)[0]
                comp_coords = SkyCoord(comp_ra_arr[valid_comp_mask] * u.deg, comp_dec_arr[valid_comp_mask] * u.deg, frame="icrs")
                idx_cand, idx_comp_valid, _, _ = cand_coords.search_around_sky(comp_coords, 35.0 * u.arcsec)
                for c_i, comp_v_i in zip(idx_cand, idx_comp_valid):
                    comp_matches_by_cand[int(c_i)].append(all_comp_rows[valid_comp_indices[comp_v_i]])
        except Exception as e:
            logger.warning(f"[ObsOrganize] Spatial component pre-matching failed: {e}")

    created_dirs = set()
    def _ensure_dir(dpath: str) -> None:
        if dpath not in created_dirs:
            os.makedirs(dpath, exist_ok=True)
            created_dirs.add(dpath)

    organized_count = 0
    linked_files_count = 0

    for idx, row in enumerate(out_tab):
        srcname = str(row["srcname"]).strip()
        if not srcname:
            continue

        safe_srcname = srcname.replace("/", "_")
        cand_dir = os.path.join(kind_dir, safe_srcname)

        if structure == "categorized":
            plots_dir = os.path.join(cand_dir, "plots")
            lc_dir = os.path.join(cand_dir, "lightcurves")
            cutouts_dir = os.path.join(cand_dir, "cutouts")
            fullframes_dir = os.path.join(cand_dir, "fullframes")
            ds_dir = os.path.join(cand_dir, "dynamic_spectra")
            cat_dir = os.path.join(cand_dir, "catalogs")
        else:
            plots_dir = lc_dir = cutouts_dir = fullframes_dir = ds_dir = cat_dir = cand_dir

        matched_artefacts = artefacts_by_cand_idx.get(idx, [])
        for src_path in sorted(set(matched_artefacts)):
            art_name = os.path.basename(src_path)
            art_name_lower = art_name.lower()

            if art_name_lower.endswith((".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg")):
                dest_folder = plots_dir
            elif art_name_lower.endswith((".fits", ".fit")) and ("fullframe" in art_name_lower or "full_frame" in art_name_lower):
                dest_folder = fullframes_dir
            elif art_name_lower.endswith((".fits", ".fit")) and ("cutout" in art_name_lower or "cand" in art_name_lower or "snip" in art_name_lower):
                dest_folder = cutouts_dir
            elif art_name_lower.endswith(".npz") and ("_lc" in art_name_lower or "lightcurve" in art_name_lower):
                dest_folder = lc_dir
            elif art_name_lower.endswith(".ds") or "dynamic" in art_name_lower or ("ds" in art_name_lower and art_name_lower.endswith((".npz", ".fits"))):
                dest_folder = ds_dir
            elif art_name_lower.endswith((".csv", ".vot", ".xml", ".reg", ".ann")):
                dest_folder = cat_dir
            elif art_name_lower.endswith((".fits", ".fit")):
                dest_folder = cutouts_dir
            else:
                dest_folder = cand_dir

            _ensure_dir(dest_folder)
            dest_path = os.path.join(dest_folder, art_name)

            try:
                if link_mode in ("symlink", "absolute_symlink", "abs_symlink"):
                    abs_src = os.path.abspath(src_path)
                    if os.path.lexists(dest_path):
                        os.remove(dest_path)
                    os.symlink(abs_src, dest_path)
                elif link_mode in ("relative_symlink", "rel_symlink"):
                    rel_src = os.path.relpath(src_path, dest_folder)
                    if os.path.lexists(dest_path):
                        os.remove(dest_path)
                    os.symlink(rel_src, dest_path)
                else:
                    if os.path.isdir(src_path):
                        if os.path.exists(dest_path):
                            shutil.rmtree(dest_path)
                        shutil.copytree(src_path, dest_path)
                    else:
                        shutil.copy2(src_path, dest_path)
                linked_files_count += 1
            except Exception as e:
                logger.warning(f"[ObsOrganize] Failed to {link_mode} '{src_path}' -> '{dest_path}': {e}")

        # Write observation-level summary table (1 row for this candidate) directly as CSV
        try:
            _ensure_dir(cat_dir)
            obs_row_dict = _sanitize_row_dict({col: row[col] for col in out_tab.colnames})
            obs_csv_cat = os.path.join(cat_dir, f"{safe_srcname}_obs_summary.csv")
            with open(obs_csv_cat, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=out_tab.colnames)
                writer.writeheader()
                writer.writerow(obs_row_dict)

            if structure == "categorized":
                _ensure_dir(cand_dir)
                obs_csv_root = os.path.join(cand_dir, f"{safe_srcname}_obs_summary.csv")
                with open(obs_csv_root, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=out_tab.colnames)
                    writer.writeheader()
                    writer.writerow(obs_row_dict)
        except Exception as e:
            logger.warning(f"[ObsOrganize] Could not write obs summary table for {srcname}: {e}")

        # Agglomerate & write all component candidate entries matching this source
        try:
            matched_raw = comp_matches_by_cand.get(idx, [])
            if not matched_raw and obs_rows_sub and idx < len(obs_rows_sub) and obs_rows_sub[idx]:
                matched_raw = obs_rows_sub[idx]

            if matched_raw:
                seen_row_ids = set()
                matched_comp_rows = []
                for r in matched_raw:
                    rid = id(r)
                    if rid not in seen_row_ids:
                        seen_row_ids.add(rid)
                        matched_comp_rows.append(r)

                def _get_comp_snr(d: Dict[str, Any]) -> float:
                    val = d.get("max_snr", d.get("snr", -np.inf))
                    try:
                        f = float(val)
                        return f if not np.isnan(f) else -np.inf
                    except Exception:
                        return -np.inf

                matched_comp_rows.sort(key=_get_comp_snr, reverse=True)
                if max_n_candidates_per_source is not None and max_n_candidates_per_source > 0:
                    matched_comp_rows = matched_comp_rows[:max_n_candidates_per_source]

                clean_matched = [_sanitize_row_dict(r) for r in matched_comp_rows]
                fieldnames = []
                seen_fields = set()
                for r_m in clean_matched:
                    for k in r_m.keys():
                        if k not in seen_fields:
                            seen_fields.add(k)
                            fieldnames.append(k)

                _ensure_dir(cat_dir)
                list_csv_cat = os.path.join(cat_dir, f"{safe_srcname}_full_candidate_list.csv")
                with open(list_csv_cat, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(clean_matched)

                if structure == "categorized":
                    _ensure_dir(cand_dir)
                    list_csv_root = os.path.join(cand_dir, f"{safe_srcname}_full_candidate_list.csv")
                    with open(list_csv_root, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(clean_matched)
        except Exception as e:
            logger.warning(f"[ObsOrganize] Could not write full candidate list table for {srcname}: {e}")

        organized_count += 1
        if (idx + 1) % 500 == 0 or (idx + 1) == len(out_tab):
            logger.info(f"[ObsOrganize] Processed {idx + 1}/{len(out_tab)} candidates ({linked_files_count} links/files created)...")

    logger.info(f"[ObsOrganize] Organized {organized_count} candidates ({linked_files_count} files/links created) under {kind_dir}")
