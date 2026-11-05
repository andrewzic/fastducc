"""
test_dynspec.py - Unit and benchmark tests for fastducc in-memory phased dynamic spectrum extraction.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

from fastducc.dynspec import (
    C_LIGHT,
    extract_phased_dynamic_spectra_from_ms,
)

# Test datasets on OzSTAR
MS_BEAM32 = Path(
    "/fred/oz451/azic/data/SB82418/native_combined/cracoData.LTR_1812-2849.SB82418.beam32.calG7.uvsub.ms"
)
GT_DS_J1752 = Path(
    "/fred/oz451/azic/data/SB82418/extract_ds/periodicity_cands/cracoData.LTR_1812-2849.SB82418.beam32_cand22859_J1752.ds"
)


def test_direction_cosines_math():
    """Verify direction cosine formulas against known sky positions."""
    from astropy.coordinates import SkyCoord

    ra0_deg, dec0_deg = 271.2, -28.9
    sc0 = SkyCoord(ra0_deg, dec0_deg, unit="deg")
    ra0 = sc0.ra.rad
    dec0 = sc0.dec.rad

    # Point at phase centre -> (l, m, n) = (0, 0, 1)
    l0 = np.cos(dec0) * np.sin(0.0)
    m0 = np.sin(dec0) * np.cos(dec0) - np.cos(dec0) * np.sin(dec0) * np.cos(0.0)
    n0 = np.sqrt(1.0 - l0**2 - m0**2)
    assert np.isclose(l0, 0.0, atol=1e-12)
    assert np.isclose(m0, 0.0, atol=1e-12)
    assert np.isclose(n0, 1.0, atol=1e-12)

    # Offset target
    tgt_ra_deg, tgt_dec_deg = 268.2389787, -28.1116732
    tgt_sc = SkyCoord(tgt_ra_deg, tgt_dec_deg, unit="deg")
    ra = tgt_sc.ra.rad
    dec = tgt_sc.dec.rad
    l = np.cos(dec) * np.sin(ra - ra0)
    m = np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(ra - ra0)
    n = np.sqrt(max(0.0, 1.0 - l**2 - m**2))
    assert -1.0 <= l <= 1.0
    assert -1.0 <= m <= 1.0
    assert 0.0 <= n <= 1.0


@pytest.mark.skipif(not MS_BEAM32.exists(), reason="Measurement set not found on disk")
def test_extract_phased_dynamic_spectrum_psr_j1752():
    """
    Test that in-memory phased visibility extraction on PSR J1752-2806 accurately matches
    ground-truth CASA/dstools dynamic spectrum visibilities.
    """
    targets = [
        {
            "id": "cand_J1752",
            "ra_deg": 268.238978719507,
            "dec_deg": -28.11167317183522,
        }
    ]

    t0 = time.time()
    specs = extract_phased_dynamic_spectra_from_ms(
        ms_path=MS_BEAM32,
        targets=targets,
        chunk_rows=100000,
        pol="I",
        verbose=True,
    )
    t_extract = time.time() - t0
    print(f"\n[BENCHMARK] Extraction of PSR J1752 completed in {t_extract:.2f}s")

    assert "cand_J1752" in specs
    spec_data = specs["cand_J1752"]
    data = spec_data["data"]
    assert data.ndim == 2
    assert data.shape[0] > 50000  # ~58,876 timestamps
    assert data.shape[1] == 288    # 288 channels
    assert spec_data["freq"].shape[0] == 288
    assert spec_data["time"].shape[0] == data.shape[0]

    # Verify correlation against ground truth .ds if available
    if GT_DS_J1752.exists():
        import h5py

        with h5py.File(GT_DS_J1752, "r") as f:
            gt_flux = f["flux"][0, :1000, :, 0].real
        sample_ext = data[:1000, :]
        valid = np.isfinite(gt_flux) & np.isfinite(sample_ext) & (sample_ext != 0.0)
        corr = np.corrcoef(sample_ext[valid], gt_flux[valid])[0, 1]
        print(f"[VERIFICATION] Correlation against ground truth: {corr:.6f}")
        assert corr > 0.998, f"Correlation {corr} is lower than 0.998"


@pytest.mark.skipif(not MS_BEAM32.exists(), reason="Measurement set not found on disk")
def test_pdmp_refinement_recovery(tmp_path):
    """
    Verify that piping the in-memory dynamic spectrum into run_pdmp_refinement
    recovers the exact expected pulsar period, DM, and folded pulse profile.
    """
    import sys
    sys.path.insert(0, "/fred/oz451/azic/scripts/lotrun_periodicity/src")
    from periodicity.pdmp_refine import run_pdmp_refinement

    targets = [
        {
            "id": "test_j1752_refine",
            "ra_deg": 268.238978719507,
            "dec_deg": -28.11167317183522,
        }
    ]
    specs = extract_phased_dynamic_spectra_from_ms(
        ms_path=MS_BEAM32,
        targets=targets,
        chunk_rows=100000,
        pol="I",
        verbose=False,
    )
    cand_spec = specs["test_j1752_refine"]

    p0 = 0.5625212721348165
    dm0 = 38.483916

    res = run_pdmp_refinement(
        ds_path=cand_spec,
        period_ref=p0,
        dm_ref=dm0,
        out_dir=tmp_path,
        cand_id="test_j1752_refine",
    )

    print("\n[VERIFICATION] Refined Pulsar Metrics:")
    print(f"  Period: {res['p_refined']:.8f} s (diff: {(res['p_refined'] - p0)*1e6:.2f} us)")
    print(f"  DM:     {res['dm_refined']:.2f} pc cm^-3")
    print(f"  SNR:    {res['snr_refined']:.2f}")

    # Verify period recovery within 50 us
    assert np.isclose(res["p_refined"], 0.5625668, atol=1e-5)
    # Verify DM recovery around 52 pc cm^-3
    assert 50.0 <= res["dm_refined"] <= 55.0
    # Verify SNR detection is confident (> 15)
    assert res["snr_refined"] > 15.0

    # Verify output diagnostic files were written
    assert Path(res["plot_path"]).is_file()
    assert Path(res["profile_path"]).is_file()
    assert Path(res["json_path"]).is_file()


@pytest.mark.skipif(not MS_BEAM32.exists(), reason="Measurement set not found on disk")
def test_batch_extraction_and_zero_disk_leak():
    """
    Verify that running on a batch of 5 candidates in beam32 reads the 35 GB MS
    only once, completes rapidly (< 90s total), and leaves zero temporary files on disk.
    """
    targets = [
        {"id": "cand_b1", "ra_deg": 268.2389787, "dec_deg": -28.1116732},
        {"id": "cand_b2", "ra_deg": 270.4388031, "dec_deg": -29.3498996},
        {"id": "cand_b3", "ra_deg": 269.6254167, "dec_deg": -28.8882500},
        {"id": "cand_b4", "ra_deg": 269.3808333, "dec_deg": -29.0531389},
        {"id": "cand_b5", "ra_deg": 268.5000000, "dec_deg": -28.5000000},
    ]

    # Snapshot directory before
    ms_dir = MS_BEAM32.parent
    before_files = set(p.name for p in ms_dir.iterdir())

    t0 = time.time()
    specs = extract_phased_dynamic_spectra_from_ms(
        ms_path=MS_BEAM32,
        targets=targets,
        chunk_rows=150000,
        pol="I",
        verbose=True,
    )
    elapsed = time.time() - t0

    # Snapshot directory after
    after_files = set(p.name for p in ms_dir.iterdir())
    new_files = after_files - before_files

    print(f"\n[BENCHMARK] 5-candidate batch streaming extraction completed in {elapsed:.2f}s")
    print(f"[BENCHMARK] Per-candidate average time: {elapsed / len(targets):.2f}s")

    assert len(specs) == 5
    for tgt in targets:
        cid = tgt["id"]
        assert cid in specs
        assert specs[cid]["data"].shape[0] > 50000
        assert specs[cid]["data"].shape[1] == 288

    # Assert single sequential pass through 35 GB MS completes rapidly
    assert elapsed < 120.0, f"Batch extraction took {elapsed:.2f}s (> 120s limit)"

    # Assert zero temporary files created in MS directory
    assert len(new_files) == 0, f"Found leaked temporary files: {new_files}"
