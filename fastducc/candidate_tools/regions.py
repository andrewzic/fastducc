# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from astropy.table import Table


def write_obs_region_files(
    out_tab: Table,
    *,
    field: str,
    sbid: str,
    kind: str,
    out_dir: str,
    ds9_radius_arcsec: float = 28.0,
    kvis_radius_arcsec: float = 28.0,
    ds9_color: str = "green",
    kvis_color: str = "GREEN",
    label_mode: str = "srcname+snr+match"
) -> dict:
    """
    Create a SAOImage DS9 .reg and Karma kvis .ann file for all unique candidates
    contained in the observation-level table (out_tab).
    """
    os.makedirs(out_dir, exist_ok=True)

    base = f"{field}.{sbid}_obs_{kind}_super_summary"
    reg_path = os.path.join(out_dir, f"{base}.reg")
    ann_path = os.path.join(out_dir, f"{base}.ann")

    if not (("ra_deg" in out_tab.colnames) and ("dec_deg" in out_tab.colnames)):
        raise ValueError("Expected 'ra_deg' and 'dec_deg' columns in observation table")

    has_srcname = "srcname" in out_tab.colnames
    has_snr = "max_snr" in out_tab.colnames
    has_match = "source_name" in out_tab.colnames

    def _label(row):
        parts = []
        if has_srcname and ("srcname" in label_mode):
            parts.append(str(row["srcname"]))
        if has_snr and ("snr" in label_mode):
            try:
                parts.append(f"S/N={float(row['max_snr']):.1f}")
            except Exception:
                pass
        if has_match and ("match" in label_mode):
            name = str(row.get("source_name", "")).strip()
            if name:
                parts.append(name.replace("unknown", "??"))
        txt = " ".join(p for p in parts if p)
        return txt.replace("{", "(").replace("}", ")")

    # ----------------- DS9 .reg -----------------
    ds9_lines = [
        "# Region file format: DS9 version 4.1",
        f"global color={ds9_color} width=2",
        "fk5",
    ]

    r_arcsec = float(ds9_radius_arcsec)
    for row in out_tab:
        try:
            ra = float(row["ra_deg"])
            dec = float(row["dec_deg"])
        except Exception:
            continue
        label = _label(row)
        if label:
            ds9_lines.append(f"circle({ra:.6f},{dec:.6f},{r_arcsec:.2f}\") # text={{{label}}}")
        else:
            ds9_lines.append(f"circle({ra:.6f},{dec:.6f},{r_arcsec:.2f}\")")

    with open(reg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(ds9_lines) + "\n")

    # ----------------- Karma kvis .ann -----------------
    ann_lines = [
        "# Karma kvis annotation file (auto-generated)",
        "# COORD=W means world coordinates; units are degrees per the image WCS.",
        "COORD W",
        f"COLOR {kvis_color}",
    ]

    r_deg = float(kvis_radius_arcsec) / 3600.0
    for row in out_tab:
        try:
            ra = float(row["ra_deg"])
            dec = float(row["dec_deg"])
        except Exception:
            continue
        label = _label(row)
        ann_lines.append(f"CIRCLE {ra:.6f} {dec:.6f} {r_deg:.6f}")
        if label:
            x_left = ra + (r_deg * 1.2)
            y_left = dec
            ann_lines.append(f"TEXT {x_left:.6f} {y_left:.6f} {label}")

    with open(ann_path, "w", encoding="utf-8") as f:
        f.write("\n".join(ann_lines) + "\n")

    return {"ds9_reg": reg_path, "kvis_ann": ann_path}
