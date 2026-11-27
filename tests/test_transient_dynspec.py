"""Unit tests for on-the-fly chunk dynamic spectrum extraction and transient summary plotting."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

from fastducc.dynspec import (
    C_LIGHT,
    extract_phased_dynamic_spectra_from_chunk,
)
from fastducc.candidate_tools.plotting import save_candidate_summary
from fastducc.core import _export_boxcar_chunk_products
from fastducc.fd_types import Config


def _make_synthetic_chunk(
    nbl: int = 12,
    nchan: int = 16,
    nt: int = 32,
    ra0_deg: float = 270.0,
    dec0_deg: float = -28.0,
    src_ra_deg: float = 270.15,
    src_dec_deg: float = -27.90,
):
    """Synthesise visibilities for a single offset point source with a time- and frequency-varying flux."""
    rng = np.random.default_rng(42)
    times = 5.2e9 + np.arange(nt, dtype=np.float64) * 0.1  # raw seconds since MJD 0
    chan_freq = np.linspace(800e6, 1088e6, nchan, dtype=np.float64)

    ra0_rad = math.radians(ra0_deg)
    dec0_rad = math.radians(dec0_deg)
    ra_rad = math.radians(src_ra_deg)
    dec_rad = math.radians(src_dec_deg)

    l_src = math.cos(dec_rad) * math.sin(ra_rad - ra0_rad)
    m_src = math.sin(dec_rad) * math.cos(dec0_rad) - math.cos(dec_rad) * math.sin(dec0_rad) * math.cos(ra_rad - ra0_rad)
    n_src = math.sqrt(max(0.0, 1.0 - l_src**2 - m_src**2))

    # Baselines with varying lengths (50 m to 2000 m)
    uvw_3d = rng.uniform(-1000.0, 1000.0, size=(nbl, 3, nt)).astype(np.float64)
    # Guarantee baseline 0 is short (< 30 m) and others are long (> 200 m)
    uvw_3d[0, 0, :] = 10.0
    uvw_3d[0, 1, :] = 10.0
    uvw_3d[1:, 0, :] = np.where(np.abs(uvw_3d[1:, 0, :]) < 200.0, 300.0, uvw_3d[1:, 0, :])

    # True dynamic spectrum: baseline 0.1 Jy + transient burst around t=16
    true_spec = np.full((nt, nchan), 0.1, dtype=np.float32)
    true_spec[14:19, :] += np.linspace(1.0, 2.5, nchan, dtype=np.float32)[None, :]

    # Visibility model: V(b, c, t) = S(t, c) * exp(+i * 2*pi*nu/c * (u*l + v*m + w*(n-1)))
    delay = uvw_3d[:, 0, :] * l_src + uvw_3d[:, 1, :] * m_src + uvw_3d[:, 2, :] * (n_src - 1.0)  # (nbl, nt)
    phase = (2.0 * np.pi / C_LIGHT) * chan_freq[None, :, None] * delay[:, None, :]  # (nbl, nchan, nt)
    vis_3d = (true_spec.T[None, :, :] * np.exp(1j * phase)).astype(np.complex64)
    wgt_3d = np.ones((nbl, nchan, nt), dtype=np.float32)

    return times, chan_freq, vis_3d, wgt_3d, uvw_3d, ra0_rad, dec0_rad, ra_rad, dec_rad, true_spec


def test_extract_phased_dynamic_spectra_from_chunk_recovers_flux():
    """Verify exact recovery of an offset point-source dynamic spectrum from in-memory chunk visibilities."""
    times, chan_freq, vis_3d, wgt_3d, uvw_3d, ra0_rad, dec0_rad, ra_rad, dec_rad, true_spec = _make_synthetic_chunk()

    # Flag channel 3 completely and verify it zeroes out cleanly
    wgt_3d[:, 3, :] = 0.0

    targets = [
        {"id": "SRC_OFFSET", "ra_rad": ra_rad, "dec_rad": dec_rad},
        {"id": "SRC_CENTER", "ra_rad": ra0_rad, "dec_rad": dec0_rad},
    ]
    res = extract_phased_dynamic_spectra_from_chunk(
        times=times,
        chan_freq=chan_freq,
        vis_3d=vis_3d,
        wgt_3d=wgt_3d,
        uvw_3d=uvw_3d,
        ra0_rad=ra0_rad,
        dec0_rad=dec0_rad,
        targets=targets,
        pol="I",
    )

    assert "SRC_OFFSET" in res
    assert "SRC_CENTER" in res

    spec_off = res["SRC_OFFSET"]["data"]
    assert spec_off.shape == true_spec.shape
    # Flagged channel should be zero
    assert np.all(spec_off[:, 3] == 0.0)
    # Unflagged channels should match injected spectrum closely
    unflagged = [c for c in range(true_spec.shape[1]) if c != 3]
    assert np.allclose(spec_off[:, unflagged], true_spec[:, unflagged], rtol=1e-4, atol=1e-4)
    assert np.allclose(res["SRC_OFFSET"]["freq"], chan_freq / 1e6)
    assert np.allclose(res["SRC_OFFSET"]["time_raw"], times)


def test_extract_phased_dynamic_spectra_from_chunk_min_uv_cut():
    """Verify short baselines are excluded when min_uv_distance_m is set."""
    times, chan_freq, vis_3d, wgt_3d, uvw_3d, ra0_rad, dec0_rad, ra_rad, dec_rad, true_spec = _make_synthetic_chunk()

    # Corrupt short baseline 0 with huge spurious amplitude
    vis_3d[0, :, :] = 1e5 + 0j

    targets = [{"id": "SRC_OFFSET", "ra_rad": ra_rad, "dec_rad": dec_rad}]
    res = extract_phased_dynamic_spectra_from_chunk(
        times=times,
        chan_freq=chan_freq,
        vis_3d=vis_3d,
        wgt_3d=wgt_3d,
        uvw_3d=uvw_3d,
        ra0_rad=ra0_rad,
        dec0_rad=dec0_rad,
        targets=targets,
        min_uv_distance_m=100.0,
    )

    assert np.allclose(res["SRC_OFFSET"]["data"], true_spec, rtol=1e-4, atol=1e-4)


def test_save_candidate_summary_with_on_the_fly_dynspec(tmp_path):
    """Verify save_candidate_summary renders a 3-row summary PDF with an in-memory chunk dynspec."""
    times, chan_freq, vis_3d, wgt_3d, uvw_3d, ra0_rad, dec0_rad, ra_rad, dec_rad, _ = _make_synthetic_chunk()
    nt = len(times)
    ny, nx = 32, 32
    rng = np.random.default_rng(7)
    cube = rng.normal(0.0, 1.0, size=(nt, ny, nx)).astype(np.float64)
    cube[14:19, 16, 16] += 20.0

    dynspecs = extract_phased_dynamic_spectra_from_chunk(
        times=times,
        chan_freq=chan_freq,
        vis_3d=vis_3d,
        wgt_3d=wgt_3d,
        uvw_3d=uvw_3d,
        ra0_rad=ra0_rad,
        dec0_rad=dec0_rad,
        targets=[{"id": "J1800-2800", "ra_rad": ra_rad, "dec_rad": dec_rad}],
    )

    cand = {
        "srcname": "J1800-2800",
        "x": 16,
        "y": 16,
        "width_samples": 4,
        "t0_idx": 14,
        "t1_idx_excl": 18,
        "time_start": float(times[14]),
        "time_center": float(times[16]),
        "time_end": float(times[18]),
        "snr": 15.2,
        "dm": 50.0,
        "ra_rad": ra_rad,
        "dec_rad": dec_rad,
        "ra_deg": math.degrees(ra_rad),
        "dec_deg": math.degrees(dec_rad),
    }

    out_prefix = str(tmp_path / "test_chunk_boxcar_cand_J1800-2800_w4")
    out = save_candidate_summary(
        times=times,
        cube=cube,
        candidate=cand,
        out_prefix=out_prefix,
        spatial_size=16,
        npix_x=nx,
        npix_y=ny,
        ra0_rad=ra0_rad,
        dec0_rad=dec0_rad,
        pix_rad=math.radians(22.0 / 3600.0),
        ra_sign=-1,
        dec_sign=-1,
        method="boxcar",
        dpi=72,
        dynspec=dynspecs["J1800-2800"],
    )

    assert "figure" in out and Path(out["figure"]).is_file()
    assert Path(out["figure"]).stat().st_size > 10_000


def test_export_boxcar_chunk_products_wires_dynspec(tmp_path, monkeypatch):
    """Verify _export_boxcar_chunk_products extracts chunk dynspec on the fly and passes it to plotting."""
    times, chan_freq, vis_3d, wgt_3d, uvw_3d, ra0_rad, dec0_rad, ra_rad, dec_rad, true_spec = _make_synthetic_chunk()
    nt = len(times)
    ny, nx = 32, 32
    cube = np.zeros((nt, ny, nx), dtype=np.float64)
    cube[14:19, 16, 16] = 10.0

    captured = {}

    def fake_save_summary(*args, **kwargs):
        captured["dynspec"] = kwargs.get("dynspec")
        return {}

    import fastducc.core as fd_core
    monkeypatch.setattr(fd_core.candidates, "save_candidate_summary", fake_save_summary)
    monkeypatch.setattr(fd_core.candidates, "save_candidate_lightcurves", lambda *a, **k: {})

    cfg = Config(
        msname="dummy.ms",
        npix_x=nx,
        npix_y=ny,
        pix_rad=math.radians(22.0 / 3600.0),
        ra0_rad=ra0_rad,
        dec0_rad=dec0_rad,
        epsilon=1e-5,
        do_wgridding=False,
        nthreads=1,
        verbosity=0,
        corr_mode="stokesI",
        basis="linear",
        single_pol="XX",
        data_column="DATA",
        enable_var=False,
        do_var_search=False,
        enable_var_chunk=False,
        enable_var_scan=False,
        enable_var_obs=False,
        enable_boxcar=True,
        do_boxcar_search=True,
        var_threshold=8.0,
        var_keep_k=None,
        rms_clip_sigma=3.0,
        nms_radius=4,
        boxcar_widths=[4],
        boxcar_threshold=8.0,
        do_plot=False,
        plot_cands_only=False,
        ms_base="dummy",
        candidates_dir=str(tmp_path),
        chunk_prefix_root=str(tmp_path / "dummy"),
        all_prefix_root=str(tmp_path / "dummy"),
        save_var_lightcurves=False,
        save_full_var_lightcurves=False,
        save_var_snippets=False,
        save_box_lightcurves=True,
        save_box_snippets=False,
        continuum_dir=None,
        var_highpass_cutoff_sec=0.0,
    )

    cand = {
        "srcname": "J1800-2800",
        "x": 16,
        "y": 16,
        "width_samples": 4,
        "t0_idx": 14,
        "t1_idx_excl": 18,
        "time_center": float(times[16]),
        "snr": 12.0,
        "dm": 0.0,
        "ra_rad": ra_rad,
        "dec_rad": dec_rad,
    }

    _export_boxcar_chunk_products(
        cfg=cfg,
        times=times,
        cube=cube,
        annotated=[cand],
        box_root=str(tmp_path / "dummy_chunk_000000_boxcar"),
        chan_freq=chan_freq,
        vis_3d=vis_3d,
        wgt_3d=wgt_3d,
        uvw_3d=uvw_3d,
    )

    assert captured.get("dynspec") is not None
    assert np.allclose(captured["dynspec"]["data"], true_spec, rtol=1e-4, atol=1e-4)


def test_orchestrator_cli_and_per_scan_build_tasks(tmp_path):
    """Verify lotrun_processing extract_ds_orchestrator CLI defaults and per-scan task building."""
    orch_dir = "/fred/oz451/azic/scripts/lotrun_processing/src/dstools"
    if orch_dir not in sys.path:
        sys.path.insert(0, orch_dir)
    import extract_ds_orchestrator as orch

    args = orch.parse_args([
        "--sbid", "SB99999",
        "--data-root", str(tmp_path),
        "--datacolumn", "DATA",
        "--no-baseline-average",
    ])
    assert args.engine == "fastducc"
    assert args.datacolumn == "data"
    assert args.baseline_average is False

    # Verify per-scan mode builds tasks without UnboundLocalError
    sbid_dir = tmp_path / "SB99999"
    scan_dir = sbid_dir / "20260101000000"
    ms_dir = scan_dir / "cracoData.FIELD.SB99999.beam05.uvsub.ms"
    ms_dir.mkdir(parents=True)

    rows = [{
        "source_id": "1",
        "srcname": "J1800-2800",
        "ra_deg": "270.0",
        "dec_deg": "-28.0",
        "max_snr": 12.0,
        "scan_ids": ["20260101000000"],
        "beams_all": ["beam05"],
        "max_snr_beam": "beam05",
        "max_snr_scan_id": "20260101000000",
        "max_snr_time_center": 5.2e9,
        "max_snr_width": 4,
        "method": "boxcar",
        "dm": "10.0",
    }]
    tasks = orch.build_tasks(
        rows,
        sbid_dir,
        ms_glob_template="cracoData*%s*uvsub.ms",
        beam_scope="union",
        kind="boxcar",
        match_arcsec=35.0,
        fieldname="FIELD",
        sbid="SB99999",
        scan_scope="catalogue",
        mode="per-scan",
        combined_dir="native_combined",
    )
    assert len(tasks) == 1
    assert tasks[0].out_file.name == "FIELD.SB99999.beam05.20260101000000_cand_J1800-2800.ds"


def test_fastducc_batch_extract_catalogue_beam_batching_and_dedup(tmp_path, monkeypatch):
    """Verify fastducc.dynspec.batch_extract_catalogue groups by beam, deduplicates sky coords, and writes .ds without dstools."""
    from astropy.table import Table
    import fastducc.dynspec as dynspec_mod
    from fastducc.cli import main as cli_main

    sbid_dir = tmp_path / "SB99999"
    ms_dir = sbid_dir / "native_combined"
    cand_dir = sbid_dir / "candidates"
    ms_b05 = ms_dir / "cracoData.FIELD.SB99999.beam05.uvsub.ms"
    ms_b12 = ms_dir / "cracoData.FIELD.SB99999.beam12.uvsub.ms"
    ms_b05.mkdir(parents=True)
    ms_b12.mkdir(parents=True)
    cand_dir.mkdir(parents=True)

    cat_path = cand_dir / "FIELD.SB99999_obs_boxcar_super_summary.csv"
    tbl = Table({
        "source_id": ["1", "2", "3", "4"],
        "srcname": ["J1800-2800", "J1800-2800_dup", "J1805-2810", "J1810-2820"],
        "ra_deg": [270.0, 270.000001, 271.25, 272.5],
        "dec_deg": [-28.0, -28.000001, -28.16, -28.33],
        "max_snr": [15.0, 11.0, 12.0, 6.0],  # row 4 below min_snr=8.0
        "max_snr_beam": ["beam05", "5", "beam12", "beam12"],
        "dm": [50.0, 50.0, 100.0, 0.0],
    })
    tbl.write(cat_path, format="ascii.csv", overwrite=True)

    calls = []

    def fake_extract(ms_path, targets, **kwargs):
        calls.append((Path(ms_path).name, list(targets), kwargs))
        out = {}
        for tgt in targets:
            out[tgt["id"]] = {
                "data": np.ones((8, 4), dtype=np.float32),
                "time": np.arange(8, dtype=np.float64) * 0.1,
                "time_raw": 5.2e9 + np.arange(8, dtype=np.float64) * 0.1,
                "time_mjd": (5.2e9 + np.arange(8, dtype=np.float64) * 0.1) / 86400.0,
                "freq": np.linspace(800.0, 1088.0, 4, dtype=np.float64),
                "ra_str": "18h00m00s",
                "dec_str": "-28d00m00s",
                "ra_deg": tgt["ra_deg"],
                "dec_deg": tgt["dec_deg"],
                "telescope": "ASKAP",
                "pol": "Stokes I",
                "cand_id": tgt["id"],
            }
        return out

    monkeypatch.setattr(dynspec_mod, "extract_phased_dynamic_spectra_from_ms", fake_extract)

    rc = cli_main([
        "extract-ds",
        "--sbid", "SB99999",
        "--data-root", str(tmp_path),
        "--kind", "boxcar",
        "--min-snr", "8.0",
        "--max-batch-cands-per-beam", "10",
        "--parallel-mode", "serial",
        "--no-update-summary-plots",
    ])
    assert rc == 0

    # Two beam MS passes (beam05 + beam12); beam05 deduplicated 2 identical coords to 1 target
    assert len(calls) == 2
    by_ms = {ms_name: tgts for ms_name, tgts, _ in calls}
    assert len(by_ms["cracoData.FIELD.SB99999.beam05.uvsub.ms"]) == 1
    assert len(by_ms["cracoData.FIELD.SB99999.beam12.uvsub.ms"]) == 1

    # Verify .ds HDF5 files were written for all 3 passing candidates (row 4 filtered by min_snr)
    out_cands_dir = ms_dir / "candidates"
    assert (out_cands_dir / "FIELD.SB99999.beam05.combined_cand_J1800-2800.ds").is_file()
    assert (out_cands_dir / "FIELD.SB99999.beam05.combined_cand_J1800-2800_dup.ds").is_file()
    assert (out_cands_dir / "FIELD.SB99999.beam12.combined_cand_J1805-2810.ds").is_file()
    assert not (out_cands_dir / "FIELD.SB99999.beam12.combined_cand_J1810-2820.ds").exists()

