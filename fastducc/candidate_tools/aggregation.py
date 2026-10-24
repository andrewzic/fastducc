# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import os
import glob
import re
import itertools
from typing import List, Dict, Any, Optional

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.nddata import Cutout2D

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnchoredText

try:
    from ligo.skymap.plot.marker import reticle
except Exception:
    def reticle(which='rt'):
        return 'o'

logger = logging.getLogger(__name__)

from fastducc.candidate_tools.utils import (
    parse_candidate_filename,
    _read_csv_tables,
    _stack_tables_or_empty,
    _unmask_table,
    _row_value_to_python,
    _sexagesimal_from_deg,
    _ensure_sexagesimal_on_rows,
    _split_list_str,
    _split_list_int,
)
from fastducc.candidate_tools.clustering import _cluster_indices_by_sky
from fastducc.candidate_tools.catalogs import annotate_observation_with_catalogs
from fastducc.candidate_tools.regions import write_obs_region_files
from fastducc.candidate_tools.organization import organize_candidate_outputs


def consolidate_chunk_catalogues(
    *,
    ms_base: str,
    out_dir: str,
    var_csv_pattern: str,
    box_csv_pattern: str,
    remove_chunk_catalogues: bool = True
) -> None:
    """
    Consolidate per-chunk catalogues from disk into MS-local `candidates/`,
    write consolidated CSV+VOT.
    """
    os.makedirs(out_dir, exist_ok=True)

    var_csv_files = sorted(glob.glob(var_csv_pattern))
    box_csv_files = sorted(glob.glob(box_csv_pattern))
    if "_dm" not in ms_base:
        var_csv_files = [f for f in var_csv_files if "_dm" not in os.path.basename(f)]
        box_csv_files = [f for f in box_csv_files if "_dm" not in os.path.basename(f)]

    logger.info(f"[Consolidation] Found {len(var_csv_files)} variance CSVs and {len(box_csv_files)} boxcar CSVs.")

    var_tables = _read_csv_tables(var_csv_files)
    box_tables = _read_csv_tables(box_csv_files)

    t_var_all = _stack_tables_or_empty(var_tables)
    t_box_all = _stack_tables_or_empty(box_tables)

    var_out_csv = os.path.join(out_dir, f"{ms_base}_variance_all.csv")
    var_out_vot = os.path.join(out_dir, f"{ms_base}_variance_all.vot")
    t_var_all.write(var_out_csv, format="csv", overwrite=True)
    t_var_all.write(var_out_vot, format="votable", overwrite=True)

    box_out_csv = os.path.join(out_dir, f"{ms_base}_boxcar_all.csv")
    box_out_vot = os.path.join(out_dir, f"{ms_base}_boxcar_all.vot")
    t_box_all.write(box_out_csv, format="csv", overwrite=True)
    t_box_all.write(box_out_vot, format="votable", overwrite=True)

    logger.info(f"[Consolidation] Wrote {len(t_var_all)} variance candidates -> {var_out_csv}, {var_out_vot}")
    logger.info(f"[Consolidation] Wrote {len(t_box_all)} boxcar candidates   -> {box_out_csv}, {box_out_vot}")

    try:
        generate_obs_variance_candidate_products(ms_base=ms_base, candidates_dir=out_dir)
    except Exception as e:
        logger.warning(f"[Consolidation] Note: Obs variance products generation: {e}")

    if remove_chunk_catalogues:
        var_vot_files = sorted(glob.glob(var_csv_pattern.replace(".csv", ".vot")))
        box_vot_files = sorted(glob.glob(box_csv_pattern.replace(".csv", ".vot")))
        if "_dm" not in ms_base:
            var_vot_files = [f for f in var_vot_files if "_dm" not in os.path.basename(f)]
            box_vot_files = [f for f in box_vot_files if "_dm" not in os.path.basename(f)]
        removed = 0
        for p in (var_csv_files + var_vot_files + box_csv_files + box_vot_files):
            try:
                os.remove(p)
                removed += 1
            except Exception as e:
                logger.warning(f"[Consolidation] Could not remove '{p}': {e}")
        logger.info(f"[Consolidation] Removed {removed} per-chunk catalogue files.")


def generate_obs_variance_candidate_products(
    *,
    ms_base: str,
    candidates_dir: str,
    continuum_dir: Optional[str] = None,
    spatial_size: int = 50,
    dpi: int = 180,
    cmap: str = "viridis",
    dm: Optional[float] = None,
    scan_id: Optional[str] = None,
    std_fits_path: Optional[str] = None,
) -> List[str]:
    """
    Ensure all scan-level / observation-level variance candidates (time_center >= 9000.0)
    have their cutout FITS, fullframe FITS, summary plot, and lightcurve products generated
    directly from std_map_full.fits in candidates_dir.
    """
    dm_suffix = f"_dm{dm:06.2f}" if (dm is not None and dm != 0.0) else ("_dm000.00" if dm == 0.0 else "")
    scan_tag = f"_scan_{scan_id}" if scan_id else ""

    # Locate full observation std map FITS
    if std_fits_path and os.path.exists(std_fits_path):
        std_fits = std_fits_path
    else:
        candidate_std_paths = [
            os.path.join(candidates_dir, f"{ms_base}{scan_tag}{dm_suffix}_std_map_full.fits"),
            os.path.join(candidates_dir, f"{ms_base}{dm_suffix}_std_map_full.fits"),
            os.path.join(candidates_dir, f"{ms_base}_std_map_full.fits"),
        ]
        std_fits = None
        for p in candidate_std_paths:
            if os.path.exists(p):
                std_fits = p
                break
        if std_fits is None:
            cands_matches = sorted(glob.glob(os.path.join(candidates_dir, f"*{ms_base}*{dm_suffix}*std_map_full.fits")))
            if not cands_matches:
                cands_matches = sorted(glob.glob(os.path.join(candidates_dir, f"*{ms_base}*std_map_full.fits")))
            if not cands_matches:
                cands_matches = sorted(glob.glob(os.path.join(candidates_dir, "*std_map_full.fits")))
            if cands_matches:
                std_fits = cands_matches[0]
            else:
                return []

    cand_tab = None
    tab_names = [
        f"{ms_base}{scan_tag}{dm_suffix}_var_candidates.vot",
        f"{ms_base}{scan_tag}{dm_suffix}_var_candidates.csv",
        f"{ms_base}{dm_suffix}_var_candidates.vot",
        f"{ms_base}{dm_suffix}_var_candidates.csv",
        f"{ms_base}_variance_all.vot",
        f"{ms_base}_variance_all.csv",
        f"{ms_base}_var_candidates.vot",
        f"{ms_base}_var_candidates.csv",
    ]
    for tab_name in tab_names:
        tab_path = os.path.join(candidates_dir, tab_name)
        if os.path.exists(tab_path):
            try:
                fmt = "votable" if tab_name.endswith(".vot") else "csv"
                cand_tab = Table.read(tab_path, format=fmt)
                cand_tab = _unmask_table(cand_tab)
                if len(cand_tab) > 0:
                    break
            except Exception:
                continue

    if cand_tab is None or len(cand_tab) == 0:
        glob_tabs = sorted(glob.glob(os.path.join(candidates_dir, f"*{ms_base}*{dm_suffix}*var*.vot"))) + \
                    sorted(glob.glob(os.path.join(candidates_dir, f"*{ms_base}*{dm_suffix}*var*.csv")))
        for tab_path in glob_tabs:
            try:
                fmt = "votable" if tab_path.endswith(".vot") else "csv"
                cand_tab = Table.read(tab_path, format=fmt)
                cand_tab = _unmask_table(cand_tab)
                if len(cand_tab) > 0:
                    break
            except Exception:
                continue

    if cand_tab is None or len(cand_tab) == 0:
        return []

    try:
        with fits.open(std_fits, memmap=False) as hdul:
            std_data = hdul[0].data.squeeze().astype(np.float32)
            std_hdr = hdul[0].header.copy()
            std_wcs = WCS(std_hdr).celestial
    except Exception as e:
        logger.warning(f"[ObsVarProducts] Error reading std_map {std_fits}: {e}")
        return []

    valid_std = std_data[np.isfinite(std_data) & (std_data > 0)]
    std_vmin = float(np.nanpercentile(valid_std, 5.0)) if valid_std.size > 0 else 0.0
    std_vmax = float(np.nanpercentile(valid_std, 99.5)) if valid_std.size > 0 else 1.0

    generated = []
    meta = parse_candidate_filename(ms_base)
    beam_str = meta.get("beam", "")
    sbid_str = meta.get("sbid", "")
    field_str = meta.get("field", "")

    cont_im, cont_wcs, cont_vmin, cont_vmax = None, None, None, None
    if continuum_dir and os.path.isdir(continuum_dir):
        fits_cands = sorted(glob.glob(os.path.join(continuum_dir, f"*{beam_str}*.fits"))) if beam_str else []
        if not fits_cands:
            fits_cands = sorted(glob.glob(os.path.join(continuum_dir, "*.fits")))
        if fits_cands:
            try:
                with fits.open(fits_cands[0], memmap=False) as hdul:
                    cdata = np.squeeze(hdul[0].data)
                    chdr = hdul[0].header
                    cont_wcs = WCS(chdr).celestial
                    cont_im = cdata
                    cont_vmin = np.nanpercentile(cdata, 5.0)
                    cont_vmax = np.nanpercentile(cdata, 99.5)
            except Exception:
                cont_im, cont_wcs = None, None

    for r in cand_tab:
        t_center = float(r.get("time_center", 0.0))
        c_idx = int(r.get("center_idx", 0))
        if t_center < 9000.0 and c_idx < 9000:
            continue

        srcname = str(r["srcname"]).strip()
        if not srcname:
            continue

        x = int(r["x"])
        y = int(r["y"])
        snr_val = float(r.get("snr", 0.0))
        ra_deg = float(r.get("ra_deg", 0.0))
        dec_deg = float(r.get("dec_deg", 0.0))
        ra_hms = str(r.get("ra_hms", ""))
        dec_dms = str(r.get("dec_dms", ""))

        cand_dm = float(r.get("dm", meta.get("dm", 0.0)) if r.get("dm", None) is not None and not np.isnan(float(r.get("dm", 0.0))) else meta.get("dm", 0.0))
        dm_str = f"{cand_dm:.2f} pc/cm³"

        out_prefix = os.path.join(candidates_dir, f"{ms_base}_var_cand_{srcname}")
        out_cut = f"{out_prefix}_cutout.fits"
        out_ff = f"{out_prefix}_fullframe.fits"
        out_fig = f"{out_prefix}_summary.pdf"
        out_lc = f"{out_prefix}_lc_full.npz"

        # 1. Fullframe FITS
        if not os.path.exists(out_ff):
            hdr_ff = std_hdr.copy()
            hdr_ff["OBJRA"] = ra_hms
            hdr_ff["OBJDEC"] = dec_dms
            hdr_ff["DM"] = cand_dm
            hdr_ff["COMMENT"] = "Fullframe detection image from std_map_full"
            fits.writeto(out_ff, std_data, hdr_ff, overwrite=True)
            generated.append(out_ff)

        # 2. Cutout FITS
        cutout_obj = None
        if not os.path.exists(out_cut) or not os.path.exists(out_fig):
            cutout_obj = Cutout2D(std_data, (x, y), (spatial_size, spatial_size), wcs=std_wcs, mode="partial", fill_value=np.nan)
            if not os.path.exists(out_cut):
                hdr_cut = cutout_obj.wcs.to_header()
                hdr_cut["BUNIT"] = std_hdr.get("BUNIT", "JY/BEAM")
                hdr_cut["OBJRA"] = ra_hms
                hdr_cut["OBJDEC"] = dec_dms
                hdr_cut["DM"] = cand_dm
                hdr_cut["COMMENT"] = "Cutout detection image from std_map_full"
                fits.writeto(out_cut, cutout_obj.data.astype(np.float32), hdr_cut, overwrite=True)
                generated.append(out_cut)

        # 3. Summary PDF
        if not os.path.exists(out_fig):
            if cutout_obj is None:
                cutout_obj = Cutout2D(std_data, (x, y), (spatial_size, spatial_size), wcs=std_wcs, mode="partial", fill_value=np.nan)
            fig = plt.figure(figsize=(18, 12), dpi=dpi)
            gs = GridSpec(nrows=2, ncols=3, figure=fig, height_ratios=[2.0, 1.0], width_ratios=[1.0, 1.0, 1.0])

            c_valid = cutout_obj.data[np.isfinite(cutout_obj.data) & (cutout_obj.data > 0)]
            val_at_pix = float(std_data[y, x]) if (0 <= y < std_data.shape[0] and 0 <= x < std_data.shape[1]) else 0.0
            cut_vmin = float(np.nanpercentile(c_valid, 5.0)) if c_valid.size > 0 else std_vmin
            cut_vmax = max(val_at_pix, float(np.nanpercentile(c_valid, 99.5))) if c_valid.size > 0 else std_vmax

            if cont_wcs is not None and cont_im is not None:
                try:
                    cpos = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
                    c_cut = Cutout2D(cont_im, position=cpos, size=(spatial_size * 22.0 * u.arcsec, spatial_size * 22.0 * u.arcsec), wcs=cont_wcs, mode="partial")
                    ax_c = fig.add_subplot(gs[0, 0], projection=c_cut.wcs)
                    im_c = ax_c.imshow(c_cut.data, origin="lower", cmap=cmap, vmin=cont_vmin, vmax=cont_vmax)
                    ax_c.set_title("Continuum")
                    ax_c.set_xlabel("RA (J2000)")
                    ax_c.set_ylabel("Dec (J2000)")
                    ax_c.set_aspect('equal', adjustable='box')
                    ax_c.plot([ra_deg], [dec_deg], marker=reticle(which='rt'), ms=36, mec="white", mew=1.5, mfc="none", transform=ax_c.get_transform('world'))
                    fig.colorbar(im_c, ax=ax_c, orientation="horizontal", fraction=0.046, pad=0.10, label="Flux density")
                except Exception:
                    ax_c = fig.add_subplot(gs[0, 0])
                    ax_c.axis("off")
            else:
                ax_c = fig.add_subplot(gs[0, 0], projection=std_wcs)
                im_c = ax_c.imshow(std_data, origin="lower", cmap=cmap, vmin=std_vmin, vmax=std_vmax)
                ax_c.set_title("Full Obs Std Map")
                ax_c.set_xlabel("RA (J2000)")
                ax_c.set_ylabel("Dec (J2000)")
                ax_c.set_aspect('equal', adjustable='box')
                ax_c.plot([x], [y], marker=reticle(which='rt'), ms=36, mec="white", mew=1.5, mfc="none")
                fig.colorbar(im_c, ax=ax_c, orientation="horizontal", fraction=0.046, pad=0.10, label="Std (Jy/beam)")

            ax_mid = fig.add_subplot(gs[0, 1], projection=std_wcs)
            im_mid = ax_mid.imshow(std_data, origin="lower", cmap=cmap, vmin=std_vmin, vmax=std_vmax)
            ax_mid.set_title("Std Map (Full Field)")
            ax_mid.set_xlabel("RA (J2000)")
            ax_mid.set_ylabel("Dec (J2000)")
            ax_mid.set_aspect('equal', adjustable='box')
            ax_mid.plot([x], [y], marker=reticle(which='rt'), ms=36, mec="red", mew=1.5, mfc="none")
            fig.colorbar(im_mid, ax=ax_mid, orientation="horizontal", fraction=0.046, pad=0.10, label="Std (Jy/beam)")

            ax_cut = fig.add_subplot(gs[0, 2], projection=cutout_obj.wcs)
            im_cut = ax_cut.imshow(cutout_obj.data, origin="lower", cmap=cmap, vmin=cut_vmin, vmax=cut_vmax)
            ax_cut.set_title(f"Detection Cutout: {srcname}")
            ax_cut.set_xlabel("RA (J2000)")
            ax_cut.set_ylabel("Dec (J2000)")
            ax_cut.set_aspect('equal', adjustable='box')
            cx, cy = spatial_size // 2, spatial_size // 2
            ax_cut.plot([cx], [cy], marker=reticle(which='rt'), ms=36, mec="red", mew=1.5, mfc="none")
            fig.colorbar(im_cut, ax=ax_cut, orientation="horizontal", fraction=0.046, pad=0.10, label="Std (Jy/beam)")

            ax_lc = fig.add_subplot(gs[1, 0:2])
            val_at_pix = float(std_data[y, x])
            t_plot = np.linspace(0, 5400, 100)

            hw = spatial_size // 2
            y0, y1 = max(0, y - hw), min(std_data.shape[0], y + hw)
            x0, x1 = max(0, x - hw), min(std_data.shape[1], x + hw)
            loc_std = std_data[y0:y1, x0:x1]
            cy, cx = y - y0, x - x0
            mask_off = np.ones_like(loc_std, dtype=bool)
            mask_off[cy, cx] = False
            valid_loc = loc_std[mask_off & np.isfinite(loc_std)]
            err_val = float(np.median(np.abs(valid_loc - np.median(valid_loc))) * 1.4826) if valid_loc.size > 0 else 0.0
            err_arr = np.full_like(t_plot, err_val)

            ax_lc.errorbar(t_plot, np.full_like(t_plot, val_at_pix), yerr=err_arr, fmt="none",
                           ecolor="grey", elinewidth=0.8, capsize=0, alpha=0.6, zorder=1)
            ax_lc.plot(t_plot, np.full_like(t_plot, val_at_pix), color="tab:blue", lw=1.5, label="Obs-level Std (Integrated)", zorder=2)
            ax_lc.set_xlabel("Time (s)")
            ax_lc.set_ylabel("Std (Jy/beam)")
            ax_lc.set_title(f"Variance Detection (time_center={t_center:.1f}, S/N={snr_val:.2f})")
            ax_lc.grid(True, alpha=0.3)
            ax_lc.legend(loc="upper right")

            ax_txt = fig.add_subplot(gs[1, 2])
            ax_txt.axis("off")
            txt = (
                f"Name: {srcname}\n"
                f"Field: {field_str or '—'}\n"
                f"S/N: {snr_val:.2f}\n"
                f"DM: {dm_str}\n"
                f"SBID: {sbid_str or '—'}\n"
                f"Beam: {beam_str or '—'}\n"
                f"Coord (J2000): {ra_hms} {dec_dms}\n"
                f"Coord (x,y): ({x}, {y})\n"
                f"Method: variance (obs-level)\n"
                f"Detection: Full Obs Std Map\n"
            )
            ax_txt.add_artist(AnchoredText(txt, loc="center", frameon=True, pad=1.0, prop={"size": 11}))
            fig.savefig(out_fig, bbox_inches="tight")
            plt.close(fig)
            generated.append(out_fig)

        # 4. Lightcurve NPZ
        if not os.path.exists(out_lc):
            time_arr = np.linspace(0, 5400, 100)
            flux_arr = np.full_like(time_arr, float(std_data[y, x]))
            np.savez(out_lc, time=time_arr, flux=flux_arr, flux_err=err_arr, error=err_arr)
            generated.append(out_lc)

    if generated:
        logger.info(f"[ObsVarProducts] Generated {len(generated)} products for obs-level variance candidates in {ms_base}.")
    return generated


def aggregate_beam_candidate_tables(
    vot_files: List[str],
    *,
    kind: str,
    time_tol_s: float = 0.3,
    sky_tol_arcsec: float = 35.0,
    out_dir: Optional[str] = None,
    max_n_candidates_per_source: Optional[int] = 100,
) -> Table:
    """
    Crossmatch candidates across beams within each scan and build a field-level summary,
    then produce a per-scan super summary by clustering events by sky (time-marginalised).
    """
    if not vot_files:
        raise ValueError("No input files provided")

    obs_root = os.path.abspath(os.path.dirname(vot_files[0]))
    if out_dir is None:
        out_dir = obs_root
    os.makedirs(out_dir, exist_ok=True)

    kind = kind.lower().strip()
    if kind not in ("boxcar", "variance"):
        raise ValueError("kind must be 'boxcar' or 'variance'")

    out_csv = None
    out_vot = None
    for p in vot_files:
        meta = parse_candidate_filename(p, require="all")
        field_name = meta["field"]
        sbid = meta["sbid"]
        scan_id = meta.get("scan_id", "")
        if not scan_id or (str(scan_id).isdigit() and len(str(scan_id)) != 14):
            try:
                t_tmp = Table.read(p, format="votable")
                if "scan_id" in t_tmp.colnames and len(t_tmp) > 0:
                    val = str(t_tmp["scan_id"][0]).strip()
                    if val and val not in ("-1", "0", ""):
                        if len(val) == 14 and val.isdigit():
                            scan_id = val
                        elif "time_start" in t_tmp.colnames or "time_center" in t_tmp.colnames:
                            t_col = "time_start" if "time_start" in t_tmp.colnames else "time_center"
                            t_val = float(t_tmp[t_col][0])
                            scan_id = Time(t_val / 86400.0, format="mjd", scale="utc").strftime("%Y%m%d%H%M%S")
            except Exception:
                pass
        if not scan_id:
            scan_id = "1"
        if field_name and sbid and scan_id:
            out_csv = os.path.join(out_dir, f"{field_name}.{sbid}_{scan_id}_{kind}_summary.csv")
            out_vot = os.path.join(out_dir, f"{field_name}.{sbid}_{scan_id}_{kind}_summary.vot")
            break

    rows: List[Dict[str, Any]] = []
    for p in vot_files:
        try:
            t = Table.read(p, format="votable")
        except Exception as e:
            logger.warning(f"[Aggregate] Skipping '{p}' (read error: {e})")
            continue
        meta = parse_candidate_filename(p, require="all")
        field_name = meta["field"]
        sbid = meta["sbid"]
        beam_id = meta["beam"]
        scan_id = meta.get("scan_id", "")
        file_dm = meta.get("dm", 0.0)

        for col in ("time_center", "ra_deg", "dec_deg", "snr"):
            if col not in t.colnames:
                raise ValueError(f"Missing required column '{col}' in {p}")
        for r in t:
            d = {name: _row_value_to_python(r[name]) for name in t.colnames}
            d["beam_id"] = beam_id
            d["field"]   = field_name
            d["sbid"]    = sbid
            row_dm = d.get("dm", None)
            if row_dm is not None and not (isinstance(row_dm, float) and np.isnan(row_dm)):
                try:
                    d["dm"] = float(getattr(row_dm, "value", row_dm))
                except Exception:
                    d["dm"] = float(file_dm if (file_dm not in (None, "", "nan")) else 0.0)
            else:
                d["dm"] = float(file_dm if (file_dm not in (None, "", "nan")) else 0.0)
            row_scan = str(r.get("scan_id", "")).strip()
            if scan_id and len(str(scan_id)) == 14:
                d["scan_id"] = str(scan_id)
            elif row_scan and len(row_scan) == 14:
                d["scan_id"] = row_scan
            elif scan_id:
                d["scan_id"] = str(scan_id)
            elif row_scan and row_scan not in ("-1", "0", ""):
                t_cand = r.get("time_start", r.get("time_center", np.nan))
                if np.isfinite(t_cand):
                    try:
                        d["scan_id"] = Time(float(t_cand) / 86400.0, format="mjd", scale="utc").strftime("%Y%m%d%H%M%S")
                    except Exception:
                        d["scan_id"] = row_scan
                else:
                    d["scan_id"] = row_scan
            else:
                d["scan_id"] = "1"
            rows.append(d)

    if len(rows) == 0:
        names = [
            'event_id', 'time_center', 'ra_deg', 'dec_deg', 'dm', 'max_snr',
            'max_snr_time_center', 'max_snr_beam'
        ]
        if kind == 'boxcar':
            names.append('max_snr_width')
            names.append('width_samples')
            names.append('width_all')
        names += ['beam_ids', 'beam_all', 'snr_all', 'field', 'sbid', 'scan_ids', 'n_beams', 'n_detections']
        out = Table(names=names)
        if out_csv: 
            out.write(out_csv, format='csv', overwrite=True)
        if out_vot: 
            out.write(out_vot, format='votable', overwrite=True)
        return out

    times = np.array([float(r['time_center']) for r in rows], dtype=np.float64)
    order = np.argsort(times)
    rows = [rows[i] for i in order]
    times = times[order]

    groups = []
    i0 = 0
    N = len(rows)
    while i0 < N:
        t0 = times[i0]
        i1 = i0 + 1
        while i1 < N and (times[i1] - t0) <= time_tol_s:
            i1 += 1
        groups.append((i0, i1))
        i0 = i1

    events: List[Dict[str, Any]] = []
    for (a0, a1) in groups:
        k = a1 - a0
        if k <= 0:
            continue
        if k == 1:
            det = rows[a0]        
            beams = sorted({det.get('beam_id', '')})
            scans = sorted({det.get('scan_id', '')})
            ra_hms, dec_dms, srcname = _sexagesimal_from_deg(float(det["ra_deg"]), float(det["dec_deg"]))
            w = det.get('width_samples', None)
            w_str = str(int(w)) if w is not None else ''
            det_dm = float(det.get('dm', 0.0) if det.get('dm') is not None and not np.isnan(float(det.get('dm', 0.0))) else 0.0)
            ev = {
                'srcname': srcname,
                'time_center': float(det['time_center']),
                'ra_deg': float(det['ra_deg']),
                'dec_deg': float(det['dec_deg']),
                'ra_hms': str(ra_hms),
                'dec_dms': str(dec_dms),
                'dm': det_dm,
                'max_snr': float(det['snr']),
                'beam_ids': ','.join([b for b in beams if b]),
                'scan_ids': ','.join([s for s in scans if s]),
                'fields': det.get('field', ''),
                'sbids' : det.get('sbid', ''),
                'n_beams': len([b for b in beams if b]),
                'n_detections': 1,
                'max_snr_beam': str(det.get('beam_id', '')),
                'max_snr_time_center': float(det['time_center']),
                'beam_all': str(det.get('beam_id', '')),
                'snr_all': f"{float(det['snr']):.2f}",
                'dm_all': f"{det_dm:.2f}",
            }
            if kind == 'boxcar':
                ev['width_samples'] = w_str
                ev['max_snr_width'] = int(w) if w is not None else -1
                ev['width_all'] = w_str
            events.append(ev)
            continue

        ra = np.array([float(rows[a0+i]['ra_deg']) for i in range(k)], dtype=float)
        dec = np.array([float(rows[a0+i]['dec_deg']) for i in range(k)], dtype=float)
        comps = _cluster_indices_by_sky(ra, dec, seplimit_arcsec=sky_tol_arcsec)
        for comp in comps:
            inds = [a0 + int(ii) for ii in comp]
            sub = [rows[ii] for ii in inds]
            sub_sorted = sorted(sub, key=lambda r: float(r.get('snr', -np.inf)), reverse=True)
            best = sub_sorted[0]
            beams_all_lst = [str(r.get('beam_id', '')) for r in sub_sorted]
            beams_unique = sorted({b for b in beams_all_lst if b})
            scans = sorted({r.get('scan_id', '') for r in sub_sorted})
            ra_hms, dec_dms, srcname = _sexagesimal_from_deg(float(best["ra_deg"]), float(best["dec_deg"]))
            best_dm = float(best.get('dm', 0.0) if best.get('dm') is not None and not np.isnan(float(best.get('dm', 0.0))) else 0.0)
            ev = {
                'srcname': srcname,
                'time_center': float(best['time_center']),
                'ra_deg': float(best['ra_deg']),
                'dec_deg': float(best['dec_deg']),
                'ra_hms': str(ra_hms),
                'dec_dms': str(dec_dms),
                'dm': best_dm,
                'max_snr': float(best['snr']),
                'beam_ids': ','.join(beams_unique),
                'scan_ids': ','.join([s for s in scans if s]),
                'fields': ','.join(sorted({r.get('field','') for r in sub_sorted})),
                'sbids' : ','.join(sorted({r.get('sbid','') for r in sub_sorted})),
                'n_beams': len(beams_unique),
                'n_detections': len(sub_sorted),
                'max_snr_beam': str(best.get('beam_id', '')),
                'max_snr_time_center': float(best['time_center']),
                'beam_all': ','.join(beams_all_lst),
                'snr_all': ','.join([f"{float(r.get('snr', 0.0)):.2f}" for r in sub_sorted]),
                'dm_all': ','.join([f"{float(r.get('dm', 0.0) if r.get('dm') is not None and not np.isnan(float(r.get('dm', 0.0))) else 0.0):.2f}" for r in sub_sorted]),
            }
            if kind == 'boxcar':
                widths_unique = sorted({int(r.get('width_samples', -1)) for r in sub_sorted
                                 if r.get('width_samples', None) is not None and int(r.get('width_samples', -1)) >= 0})
                ev['width_samples'] = ','.join([str(w) for w in widths_unique])
                try:
                    ev['max_snr_width'] = int(best.get('width_samples', -1))
                except Exception:
                    ev['max_snr_width'] = -1                
                ev['width_all'] = ','.join([str(int(r.get('width_samples', 1))) for r in sub_sorted])
            events.append(ev)

    events.sort(key=lambda r: float(r['time_center']))
    _ensure_sexagesimal_on_rows(events, ra_key="ra_deg", dec_key="dec_deg",
                                ra_hms_key="ra_hms", dec_dms_key="dec_dms", src_key="srcname")

    ra_all = np.array([float(r["ra_deg"]) for r in events])
    dec_all = np.array([float(r["dec_deg"]) for r in events])
    comps = _cluster_indices_by_sky(ra_all, dec_all, seplimit_arcsec=sky_tol_arcsec)

    super_rows = []
    thinned_events = []
    for comp in comps:
        sub = [events[i] for i in comp]
        best = max(sub, key=lambda r: float(r.get("max_snr", -np.inf)))
        beams_union = set()
        scans_union = set()
        for r in sub:
            beams_union |= set(_split_list_str(r.get("beam_ids","")))
            scans_union |= set(_split_list_str(r.get("scan_ids","")))
        best_dm = float(best.get("dm", 0.0) if best.get("dm") is not None and not np.isnan(float(best.get("dm", 0.0))) else 0.0)
        row = {
            "srcname": best["srcname"],
            "ra_deg": float(best["ra_deg"]),
            "dec_deg": float(best["dec_deg"]),
            "ra_hms": best["ra_hms"],
            "dec_dms": best["dec_dms"],
            "dm": best_dm,
            "n_events": len(sub),
            "n_detections_total": int(np.nansum([float(r.get("n_detections", 1)) for r in sub])),
            "beams_all": ",".join(sorted([b for b in beams_union if b])),
            "scans_all": ",".join(sorted([s for s in scans_union if s])),
            "max_snr": float(best["max_snr"]),
            "max_snr_time_center": float(best.get("max_snr_time_center", best.get("time_center", np.nan))),
            "max_snr_event_beams": best.get("beam_ids", ""),
            "max_snr_beam": str(best.get("max_snr_beam", "")),            
        }
        dms_union = set()
        for r in sub:
            dms_union |= set(_split_list_str(r.get("dm_all", "") or str(r.get("dm", ""))))
        row["dm_all"] = ",".join(sorted([d for d in dms_union if d]))
        if kind == "boxcar":
            w_union = set()
            for r in sub:
                w_union |= set(_split_list_int(r.get("width_samples","")))
            row["width_samples_all"] = ",".join(str(w) for w in sorted(w_union))
            if "max_snr_width" in best and best["max_snr_width"] not in (None, ""):
                row["max_snr_width"] = int(best["max_snr_width"])
            else:
                ws_best = _split_list_int(best.get("width_samples",""))
                row["max_snr_width"] = (ws_best[0] if len(ws_best) == 1 else -1)

        super_rows.append(row)

        def _ev_snr(r):
            v = r.get("max_snr", r.get("snr", -np.inf))
            try:
                f = float(v)
                return f if not np.isnan(f) else -np.inf
            except Exception:
                return -np.inf

        sub_sorted = sorted(sub, key=_ev_snr, reverse=True)
        if max_n_candidates_per_source is not None and max_n_candidates_per_source > 0:
            thinned_events.extend(sub_sorted[:max_n_candidates_per_source])
        else:
            thinned_events.extend(sub_sorted)

    thinned_events.sort(key=lambda r: float(r.get('time_center', 0.0)))
    for i, ev in enumerate(thinned_events):
        ev['event_id'] = i

    colnames = [
        'event_id', 'srcname', 'time_center', 'ra_deg', 'dec_deg', 'ra_hms', 'dec_dms',
        'dm', 'max_snr', 'max_snr_time_center', 'max_snr_beam'
    ]
    if kind == 'boxcar':
        colnames.append('max_snr_width')
        colnames.append('width_samples')
        colnames.append('width_all')
    colnames += ['beam_ids', 'beam_all', 'snr_all', 'dm_all', 'scan_ids', 'n_beams', 'n_detections']
    event_tab = Table(rows=[[ev.get(c) for c in colnames] for ev in thinned_events], names=colnames)
    if out_csv: 
        event_tab.write(out_csv, format='csv', overwrite=True)
    if out_vot: 
        event_tab.write(out_vot, format='votable', overwrite=True)

    super_rows.sort(key=lambda r: float(r.get("max_snr", -np.inf)), reverse=True)
    for i, r in enumerate(super_rows):
        r["source_id"] = i

    super_cols = [
        "source_id","srcname","ra_deg","dec_deg","ra_hms","dec_dms","dm","dm_all",
        "n_events","n_detections_total","beams_all","scans_all",
        "max_snr","max_snr_time_center","max_snr_event_beams",
    ]
    if kind == "boxcar":
        super_cols += ["width_samples_all", "max_snr_width", "max_snr_beam"]
    else:
        super_cols += ["max_snr_beam"]

    super_tab = Table(rows=[[r.get(c) for c in super_cols] for r in super_rows], names=super_cols)

    try:
        super_csv = out_csv.replace("_summary.csv", "_super_summary.csv") if out_csv else None
        super_vot = out_vot.replace("_summary.vot", "_super_summary.vot") if out_vot else None
    except Exception:
        super_csv = None 
        super_vot = None
    if not super_csv:
        super_csv = os.path.join(out_dir, f"field_{kind}_super_summary.csv")
    if not super_vot:
        super_vot = os.path.join(out_dir, f"field_{kind}_super_summary.vot")
    super_tab.write(super_csv, format="csv", overwrite=True)
    super_tab.write(super_vot, format="votable", overwrite=True)

    return event_tab


def aggregate_observation(
    obs_root: str,
    *,
    kind: str = 'boxcar',
    time_tol_s: float = 0.3,
    sky_tol_arcsec: float = 35.0,
    out_dir: Optional[str] = None,
    pattern: Optional[str] = None,
    max_n_candidates_per_source: Optional[int] = 100,
) -> Table:
    """Discover per-beam candidate tables under obs_root and write a field-level summary."""
    kind = kind.lower().strip()
    if pattern is None:
        suf = '*_boxcar_all.vot' if kind == 'boxcar' else '*_variance_all.vot'
        pattern = os.path.join(obs_root, 'candidates', suf)

    files = sorted(glob.glob(pattern, recursive=True))

    if out_dir is None:
        out_dir = obs_root
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"[Aggregate] Found {len(files)} '{kind}' tables under '{obs_root}'")

    return aggregate_beam_candidate_tables(
        files,
        kind=kind,
        time_tol_s=time_tol_s,
        sky_tol_arcsec=sky_tol_arcsec,
        out_dir=out_dir,
        max_n_candidates_per_source=max_n_candidates_per_source,
    )


def aggregate_observation_from_super_summaries(
    obs_root: str,
    *,
    kind: str = "variance",
    sky_tol_arcsec: float = 35.0,
    out_dir: Optional[str] = None,
    pattern: Optional[str] = None,
    psrcat_csv_path: Optional[str] = None,
    racs_vot_path: Optional[str] = None,
    match_radius_arcsec: float = 40.0,
    simbad_enable: bool = False,
    organize: bool = True,
    link_mode: str = "symlink",
    organize_structure: str = "categorized",
    max_n_candidates_per_source: Optional[int] = 100,
) -> Table:    
    """
    Read per-scan cross-beam super summaries under obs_root,
    cluster by sky position across all scans, and write an observation-level super summary.
    """
    kind = kind.lower().strip()
    if kind not in ("variance", "boxcar"):
        raise ValueError("kind must be 'variance' or 'boxcar'")

    if pattern is None:
        pattern = os.path.join(obs_root, "*", "candidates", f"*_{kind}_super_summary.vot")
    files = sorted(glob.glob(pattern, recursive=True))

    if out_dir is None:
        out_dir = os.path.join(obs_root, "candidates")
    os.makedirs(out_dir, exist_ok=True)

    if len(files) == 0:
        empty_cols = [
            "source_id","srcname","ra_deg","dec_deg","ra_hms","dec_dms","dm",
            "max_snr","max_snr_time_center","max_snr_beam","n_scans",
            "scan_ids","beams_all",
        ]
        if kind == "boxcar":
            empty_cols += ["max_snr_width","width_samples_all"]
        T = Table(names=empty_cols)
        field = "field"
        sbid = os.path.basename(os.path.abspath(obs_root))
        csv_out = os.path.join(out_dir, f"{field}.{sbid}_obs_{kind}_super_summary.csv")
        vot_out = os.path.join(out_dir, f"{field}.{sbid}_obs_{kind}_super_summary.vot")
        T.write(csv_out, format="csv", overwrite=True)
        T.write(vot_out, format="votable", overwrite=True)
        logger.info(f"[ObsSuper] No input VOTs found. Wrote empty outputs to {out_dir}")
        return T

    rows: List[Dict[str, Any]] = []
    field_for_name = None
    sbid_for_name = None

    required = {"ra_deg", "dec_deg", "max_snr"}
    optional_strings = {
        "srcname","ra_hms","dec_dms",
        "beams_all","width_samples_all",
        "max_snr_event_beams","max_snr_event_widths",
        "max_snr_beam",
    }
    optional_numbers = {"max_snr_width", "max_snr_time_center", "dm"}

    for p in files:
        meta = parse_candidate_filename(p, require='super')
        f_field = meta['field'] 
        f_sbid = meta['sbid']
        f_scan = meta.get("scan_id", "")
        if not f_scan or (str(f_scan).isdigit() and len(str(f_scan)) != 14):
            m_path_scan = re.search(r'(?<!\d)(\d{14})(?!\d)', os.path.abspath(p))
            if m_path_scan:
                f_scan = m_path_scan.group(1)
            else:
                f_scan = f_scan or "1"
        f_kind = meta['kind']
        if f_kind != kind:
            continue
        if field_for_name is None and f_field:
            field_for_name = f_field
        if sbid_for_name is None and f_sbid:
            sbid_for_name = f_sbid

        try:
            tab = Table.read(p, format="votable")
            tab = _unmask_table(tab)
        except Exception as e:
            logger.warning(f"[ObsSuper] Skipping '{p}' (read error: {e})")
            continue

        missing = [c for c in required if c not in tab.colnames]
        if missing:
            logger.warning(f"[ObsSuper] '{p}' missing required columns {missing}; skipping.")
            continue

        for r in tab:
            d = {name: _row_value_to_python(r[name]) if name in tab.colnames else None
                 for name in itertools.chain(required, optional_strings, optional_numbers)}
            row_dm = d.get("dm", None)
            meta_dm = meta.get("dm", 0.0)
            if row_dm is not None and not (isinstance(row_dm, float) and np.isnan(row_dm)):
                try:
                    d["dm"] = float(getattr(row_dm, "value", row_dm))
                except Exception:
                    d["dm"] = float(meta_dm if (meta_dm not in (None, "", "nan")) else 0.0)
            else:
                d["dm"] = float(meta_dm if (meta_dm not in (None, "", "nan")) else 0.0)
            r_scan = str(r.get("scan_id", "")).strip() if "scan_id" in tab.colnames else ""
            if len(str(f_scan)) == 14:
                d["scan_id"] = str(f_scan)
            elif len(r_scan) == 14:
                d["scan_id"] = r_scan
            elif f_scan:
                d["scan_id"] = str(f_scan)
            elif r_scan:
                d["scan_id"] = r_scan
            else:
                d["scan_id"] = "1"
            rows.append(d)

    if len(rows) == 0:
        logger.info("[ObsSuper] No valid rows after reading inputs; writing empty outputs.")
        empty_cols = [
            "source_id","srcname","ra_deg","dec_deg","ra_hms","dec_dms","dm","dm_all",
            "max_snr","max_snr_time_center","max_snr_beam","n_scans",
            "scan_ids","beams_all",
        ]
        if kind == "boxcar":
            empty_cols += ["max_snr_width","width_samples_all"]
        T = Table(names=empty_cols)
        field = field_for_name or "field"
        sbid = sbid_for_name or os.path.basename(os.path.abspath(obs_root))
        csv_out = os.path.join(out_dir, f"{field}.{sbid}_obs_{kind}_super_summary.csv")
        vot_out = os.path.join(out_dir, f"{field}.{sbid}_obs_{kind}_super_summary.vot")
        T.write(csv_out, format="csv", overwrite=True)
        T.write(vot_out, format="votable", overwrite=True)
        return T

    _ensure_sexagesimal_on_rows(rows)
    ra = np.array([float(r["ra_deg"]) for r in rows], dtype=float)
    dec = np.array([float(r["dec_deg"]) for r in rows], dtype=float)
    comps = _cluster_indices_by_sky(ra, dec, seplimit_arcsec=sky_tol_arcsec)

    obs_rows = []
    obs_rows_sub = []
    for comp_idxs in comps:
        sub = [rows[i] for i in comp_idxs]

        best = max(sub, key=lambda d: float(d.get("max_snr", -np.inf)))
        best_ra = float(best["ra_deg"])
        best_dec = float(best["dec_deg"])
        best_snr = float(best.get("max_snr", np.nan))
        best_time = float(best.get("max_snr_time_center", np.nan)) if best.get("max_snr_time_center", None) is not None else np.nan
        best_scan = str(best.get("scan_id", ""))
        best_dm = float(best.get("dm", 0.0) if best.get("dm") is not None and not np.isnan(float(best.get("dm", 0.0))) else 0.0)

        ra_hms = str(best.get("ra_hms", "") or "")
        dec_dms = str(best.get("dec_dms", "") or "")
        srcname = str(best.get("srcname", "") or "")
        if not (ra_hms and dec_dms and srcname):
            ra_hms_, dec_dms_, srcname_ = _sexagesimal_from_deg(best_ra, best_dec)
            ra_hms = ra_hms or ra_hms_
            dec_dms = dec_dms or dec_dms_
            srcname = srcname or srcname_

        scan_ids = sorted({str(r.get("scan_id", "")) for r in sub if r.get("scan_id", "")})
        beams_union = set()
        for r in sub:
            beams_union |= set(_split_list_str(r.get("beams_all", "")))

        dms_union = set()
        for r in sub:
            dms_union |= set(_split_list_str(r.get("dm_all", "") or str(r.get("dm", ""))))

        row_out = {
            "srcname": srcname,
            "ra_deg": best_ra,
            "dec_deg": best_dec,
            "ra_hms": ra_hms,
            "dec_dms": dec_dms,
            "dm": best_dm,
            "dm_all": ",".join(sorted([d for d in dms_union if d])),
            "max_snr": best_snr,
            "max_snr_time_center": best_time,
            "max_snr_beam": str(best.get("max_snr_beam", "")) if best.get("max_snr_beam", "") else "",
            "n_scans": len(scan_ids),
            "scan_ids": ",".join(scan_ids),
            "beams_all": ",".join(sorted([b for b in beams_union if b])),
            "max_snr_scan_id": best_scan,
        }

        if kind == "boxcar":
            widths_union = set()
            for r in sub:
                widths_union |= set(_split_list_int(r.get("width_samples_all", "")))
            max_w = -1
            if best.get("max_snr_width", None) not in (None, ""):
                try:
                    max_w = int(best["max_snr_width"])
                except Exception:
                    max_w = -1
            elif best.get("max_snr_event_widths", ""):
                ws = _split_list_int(best.get("max_snr_event_widths", ""))
                if len(ws) == 1:
                    max_w = ws[0]
            row_out["max_snr_width"] = int(max_w) if max_w is not None else -1
            row_out["width_samples_all"] = ",".join(str(w) for w in sorted(widths_union))

        obs_rows.append(row_out)
        obs_rows_sub.append(sub)

    paired = sorted(zip(obs_rows, obs_rows_sub), key=lambda p: float(p[0].get("max_snr", -np.inf)), reverse=True)
    obs_rows = [p[0] for p in paired]
    obs_rows_sub = [p[1] for p in paired]
    for i, d in enumerate(obs_rows):
        d["source_id"] = i

    cols = [
        "source_id","srcname","ra_deg","dec_deg","ra_hms","dec_dms","dm","dm_all",
        "max_snr","max_snr_time_center","max_snr_beam",
        "max_snr_scan_id",
        "n_scans","scan_ids","beams_all",
    ]
    if kind == "boxcar":
        cols += ["max_snr_width","width_samples_all"]
    out_tab = _unmask_table(Table(rows=[[r.get(c) for c in cols] for r in obs_rows], names=cols))

    try:
        out_tab = annotate_observation_with_catalogs(
            out_tab,
            psrcat_csv_path=psrcat_csv_path,
            racs_vot_path=racs_vot_path,
            match_radius_arcsec=match_radius_arcsec,
            simbad_enable=simbad_enable,
        )
    except Exception as e:
        logger.warning(f"[ObsSuper] Crossmatch step failed. Continuing without: {e}")

    field = field_for_name or "field"
    sbid = sbid_for_name or os.path.basename(os.path.abspath(obs_root))
    csv_out = os.path.join(out_dir, f"{field}.{sbid}_obs_{kind}_super_summary.csv")
    vot_out = os.path.join(out_dir, f"{field}.{sbid}_obs_{kind}_super_summary.vot")
    out_tab.write(csv_out, format="csv", overwrite=True)
    out_tab.write(vot_out, format="votable", overwrite=True)
    logger.info(f"[ObsSuper] Wrote {len(out_tab)} sources -> {csv_out}, {vot_out}")

    write_obs_region_files(
        out_tab,
        field=field,
        sbid=sbid,
        kind=kind,
        out_dir=out_dir,
        ds9_radius_arcsec=28.0,
        kvis_radius_arcsec=28.0,
        ds9_color="green",
        kvis_color="GREEN",
        label_mode="srcname+snr+match",
    )

    if organize:
        try:
            organize_candidate_outputs(
                obs_root=obs_root,
                kind=kind,
                out_tab=out_tab,
                out_dir=out_dir,
                link_mode=link_mode,
                structure=organize_structure,
                obs_rows_sub=obs_rows_sub,
                max_n_candidates_per_source=max_n_candidates_per_source,
            )
        except Exception as e:
            logger.warning(f"[ObsSuper] Candidate output organization failed: {e}")

    return out_tab
