"""Comprehensive tests for fastducc.candidate_store, merging, consolidation,
ergonomic CandidateStore API, summary plot integration, and transient HDF5 wiring.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from astropy.table import Table
from astropy.wcs import WCS

from fastducc import candidates as cand_mod
from fastducc import cli as fd_cli
from fastducc import core as fd_core
from fastducc.candidate_store import (
    SCHEMA_VERSION,
    CandidateHDF5Writer,
    CandidateStore,
    DynamicSpectrum,
    Lightcurve,
    build_consolidated_h5_path,
    build_trial_h5_path,
    consolidate_candidate_h5,
    discover_trial_h5_files,
    is_valid_candidates_h5,
    merge_candidate_h5,
    open_candidates,
    parse_candidate_h5_filename,
)
from fastducc.fd_types import Config


def _make_wcs(ra0: float = 180.0, dec0: float = -30.0, npix: int = 384) -> WCS:
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    w.wcs.cunit = ["deg", "deg"]
    w.wcs.crpix = [npix / 2.0 + 1.0, npix / 2.0 + 1.0]
    w.wcs.crval = [ra0, dec0]
    w.wcs.cdelt = [-22.0 / 3600.0, 22.0 / 3600.0]
    return w


def test_naming_and_discovery(tmp_path):
    p_nufft = build_trial_h5_path(
        tmp_path,
        method="nufft",
        beam="beam00",
        dm=25.0,
        prefix="cracoData.FIELD.SB82418.beam00.uvsub",
    )
    assert p_nufft.name == "cracoData.FIELD.SB82418.beam00.uvsub_nufft_dm025.00_candidates.h5"

    p_box = build_trial_h5_path(
        tmp_path,
        method="boxcar",
        beam="beam03",
        dm=12.5,
        prefix="FIELD.SB82418",
    )
    assert p_box.name == "FIELD.SB82418_boxcar_beam03_dm012.50_candidates.h5"

    p_ffa = build_trial_h5_path(
        tmp_path,
        method="uv_ffa",
        beam="5",
        dm=0.0,
        prefix="FIELD.SB82418",
    )
    assert p_ffa.name == "FIELD.SB82418_ffa_beam05_dm000.00_candidates.h5"

    parsed = parse_candidate_h5_filename(p_box.name)
    assert parsed["method"] == "boxcar"
    assert parsed["beam"] == "beam03"
    assert parsed["dm"] == pytest.approx(12.5)
    assert parsed["sbid"] == "SB82418"
    assert parsed["is_consolidated"] is False

    p_cons = build_consolidated_h5_path(tmp_path, method="boxcar", sbid="SB82418", field="FIELD")
    assert p_cons.name == "FIELD.SB82418_obs_boxcar_consolidated_candidates.h5"
    assert parse_candidate_h5_filename(p_cons.name)["is_consolidated"] is True

    # Create dummy files and test discovery filtering
    for p in (p_nufft, p_box, p_ffa, p_cons):
        p.write_bytes(b"placeholder")
    assert discover_trial_h5_files([tmp_path], method="boxcar") == [p_box.resolve()]
    assert discover_trial_h5_files([tmp_path], method="ffa") == [p_ffa.resolve()]
    assert discover_trial_h5_files([tmp_path], method="nufft") == [p_nufft.resolve()]
    assert len(discover_trial_h5_files([tmp_path], include_consolidated=True)) == 4


def test_roundtrip_all_four_methods_and_dict_api(tmp_path):
    rng = np.random.default_rng(7)
    w0 = _make_wcs(ra0=185.0, dec0=-25.0)

    # 1. Boxcar (with variable-length lightcurves & dynamic spectrum)
    box_h5 = build_trial_h5_path(tmp_path, method="boxcar", beam="beam00", dm=10.0, prefix="FIELD.SB82418")
    t1 = np.linspace(5.2e9, 5.2e9 + 50.0, 64, dtype=np.float64)
    f1 = rng.standard_normal(64).astype(np.float32)
    e1 = np.full(64, 0.25, dtype=np.float32)
    cut1 = rng.standard_normal((50, 50)).astype(np.float32)
    ds1 = DynamicSpectrum(
        data=rng.standard_normal((32, 16)).astype(np.float32),
        time=np.linspace(0.0, 50.0, 32, dtype=np.float64),
        freq=np.linspace(800.0, 1088.0, 16, dtype=np.float64),
    )

    with CandidateHDF5Writer(
        box_h5, method="boxcar", cutout_size=50, wcs_full=w0, sbid="SB82418", field="FIELD", beam="beam00", dm=10.0
    ) as w:
        w.append_candidate(
            {
                "srcname": "J123456.7-250000.0",
                "x": 192,
                "y": 192,
                "ra_deg": 185.0,
                "dec_deg": -25.0,
                "snr": 12.4,
                "dm": 10.0,
                "width_samples": 4,
                "time_center": float(t1[32]),
            },
            cutout=cut1,
            lightcurve=Lightcurve(time=t1, flux=f1, flux_err=e1, width_samples=4),
            dynspec=ds1,
        )
        # Duplicate srcname with lower S/N -> dict lookup should return highest-S/N exemplar
        w.append_candidate(
            {
                "srcname": "J123456.7-250000.0",
                "x": 192,
                "y": 192,
                "ra_deg": 185.0,
                "dec_deg": -25.0,
                "snr": 9.1,
                "dm": 10.0,
                "width_samples": 2,
                "time_center": float(t1[10]),
            },
            cutout=cut1 * 0.5,
            lightcurve=Lightcurve(time=t1[:32], flux=f1[:32], flux_err=e1[:32], width_samples=2),
        )

    assert is_valid_candidates_h5(box_h5)
    with open_candidates(box_h5) as candidates:
        assert candidates.schema_version == SCHEMA_VERSION
        assert candidates.method == "boxcar"
        assert candidates.n_candidates == 2
        assert len(candidates) == 1  # unique srcnames
        assert "J123456.7-250000.0" in candidates

        c = candidates["J123456.7-250000.0"]
        assert c.snr == pytest.approx(12.4, abs=1e-4)
        assert c.width_samples == 4
        np.testing.assert_allclose(c.cutout, cut1)
        assert c.cutout_wcs is not None
        assert c.lightcurve is not None
        np.testing.assert_allclose(c.lightcurve.time, t1)
        np.testing.assert_allclose(c.lightcurve.flux, f1)
        t_sm, f_sm, e_sm = c.lightcurve.smoothed()
        assert t_sm.size == 64 - 4 + 1
        assert f_sm.size == t_sm.size
        assert c.dynspec is not None
        np.testing.assert_allclose(c.dynspec.data, ds1.data)

        all_hits = candidates.get_all("J123456.7-250000.0")
        assert len(all_hits) == 2
        assert all_hits[0].snr > all_hits[1].snr

        # Plot summary directly from CandidateRecord -> 1 PDF, 0 loose FITS/NPZ
        plot_dir = tmp_path / "box_plots"
        pdf_path = plot_dir / "J123456.7-250000.0_summary.pdf"
        res = c.plot_summary(pdf_path, dpi=100)
        assert Path(res["summary_plot"]).is_file()
        assert list(plot_dir.glob("*.fits")) == []
        assert list(plot_dir.glob("*.npz")) == []

    # 2. Variance (with std_map)
    var_h5 = build_trial_h5_path(tmp_path, method="variance", beam="beam00", dm=0.0, prefix="FIELD.SB82418")
    std_map = rng.random((64, 64)).astype(np.float32)
    with CandidateHDF5Writer(
        var_h5, method="variance", cutout_size=32, wcs_full=w0, sbid="SB82418", beam="beam00", dm=0.0
    ) as w:
        w.append_candidate(
            {"srcname": "J120001.0-250001.0", "x": 32, "y": 32, "ra_deg": 185.0, "dec_deg": -25.0, "snr": 11.2, "dm": 0.0},
            cutout=rng.random((32, 32)).astype(np.float32),
            lightcurve=Lightcurve(time=t1, flux=f1, flux_err=e1),
        )
        w.finalize(std_map=std_map)

    with open_candidates(var_h5) as cands:
        assert cands.method == "variance"
        np.testing.assert_allclose(cands.std_map, std_map)
        c = cands["J120001.0-250001.0"]
        res = c.plot_summary(tmp_path / "var_summary.pdf", dpi=100)
        assert Path(res["summary_plot"]).is_file()

    # 3. FFA (variable-length folded profiles + on/off cutouts)
    ffa_h5 = build_trial_h5_path(tmp_path, method="ffa", beam="beam01", dm=55.0, prefix="FIELD.SB82418")
    prof32 = rng.standard_normal(32).astype(np.float32)
    prof64 = rng.standard_normal(64).astype(np.float32)
    with CandidateHDF5Writer(
        ffa_h5, method="ffa", cutout_size=32, wcs_full=w0, sbid="SB82418", beam="beam01", dm=55.0
    ) as w:
        w.append_candidate(
            {
                "srcname": "J121000.0-251000.0",
                "x": 100,
                "y": 120,
                "ra_deg": 185.1,
                "dec_deg": -25.1,
                "snr": 14.5,
                "dm": 55.0,
                "period_s": 1.234,
                "f_hz": 1.0 / 1.234,
                "width_bins": 3,
                "duty_cycle": 3.0 / 32.0,
            },
            cutout=rng.standard_normal((32, 32)).astype(np.float32),
            cutout_off=rng.standard_normal((32, 32)).astype(np.float32),
            profile=prof32,
        )
        w.append_candidate(
            {
                "srcname": "J121100.0-251100.0",
                "x": 140,
                "y": 150,
                "ra_deg": 185.2,
                "dec_deg": -25.2,
                "snr": 9.8,
                "dm": 55.0,
                "period_s": 2.5,
                "f_hz": 0.4,
                "width_bins": 5,
                "duty_cycle": 5.0 / 64.0,
            },
            cutout=rng.standard_normal((32, 32)).astype(np.float32),
            cutout_off=rng.standard_normal((32, 32)).astype(np.float32),
            profile=prof64,
        )

    with open_candidates(ffa_h5) as cands:
        assert cands.method == "ffa"
        np.testing.assert_allclose(cands["J121000.0-251000.0"].profile, prof32)
        np.testing.assert_allclose(cands["J121100.0-251100.0"].profile, prof64)
        ffa_plot_dir = tmp_path / "ffa_plots"
        res = cands["J121000.0-251000.0"].plot_summary(ffa_plot_dir / "ffa_summary.pdf", dpi=100)
        assert Path(res["summary_plot"]).is_file()
        assert list(ffa_plot_dir.glob("*.fits")) == []
        assert list(ffa_plot_dir.glob("*.npz")) == []

    # 4. NUFFT (on/off cutouts + power spectrum)
    nufft_h5 = build_trial_h5_path(tmp_path, method="nufft", beam="beam02", dm=40.0, prefix="FIELD.SB82418")
    fkeep = np.linspace(0.5, 4.0, 128, dtype=np.float64)
    spec_arr = rng.random(128).astype(np.float32)
    with CandidateHDF5Writer(
        nufft_h5, method="nufft", freqs_hz=fkeep, cutout_size=32, wcs_full=w0, sbid="SB82418", beam="beam02", dm=40.0
    ) as w:
        w.append_candidate(
            {
                "srcname": "J122000.0-252000.0",
                "x": 160,
                "y": 170,
                "ra_deg": 185.3,
                "dec_deg": -25.3,
                "snr": 18.0,
                "dm": 40.0,
                "period_s": 0.5,
                "f_hz": 2.0,
                "f_off_res": 2.02,
                "nharm_used": 4,
                "spec_isolation_ratio": 4.2,
            },
            cutout=rng.standard_normal((32, 32)).astype(np.float32),
            cutout_off=rng.standard_normal((32, 32)).astype(np.float32),
            spectrum=spec_arr,
        )

    with open_candidates(nufft_h5) as cands:
        c = cands["J122000.0-252000.0"]
        assert c.spectrum is not None
        np.testing.assert_allclose(c.spectrum.power, spec_arr)
        np.testing.assert_allclose(c.spectrum.freqs_hz, fkeep)
        nufft_plot_dir = tmp_path / "nufft_plots"
        res = c.plot_summary(nufft_plot_dir / "nufft_summary.pdf", dpi=100)
        assert Path(res["summary_plot"]).is_file()
        assert list(nufft_plot_dir.glob("*.fits")) == []
        assert list(nufft_plot_dir.glob("*.npz")) == []


def test_merge_and_consolidate_exemplars_across_beams_and_dms(tmp_path):
    rng = np.random.default_rng(123)
    w_b0 = _make_wcs(ra0=180.0, dec0=-30.0)
    w_b1 = _make_wcs(ra0=181.5, dec0=-31.0)
    fkeep = np.linspace(0.5, 3.0, 64, dtype=np.float64)

    # Beam 00, DM 20
    h5_b0_dm20 = build_trial_h5_path(tmp_path, method="nufft", beam="beam00", dm=20.0, prefix="FIELD.SB82418")
    cut_b0_srcA = rng.standard_normal((32, 32)).astype(np.float32)
    with CandidateHDF5Writer(
        h5_b0_dm20, method="nufft", freqs_hz=fkeep, cutout_size=32, wcs_full=w_b0, sbid="SB82418", field="FIELD", beam="beam00", dm=20.0
    ) as w:
        w.append_candidate(
            {"srcname": "J120000.0-300000.0", "x": 192, "y": 192, "ra_deg": 180.0, "dec_deg": -30.0, "snr": 10.5, "dm": 20.0, "period_s": 0.8, "f_hz": 1.25},
            cutout=cut_b0_srcA,
            cutout_off=rng.standard_normal((32, 32)).astype(np.float32),
            spectrum=rng.random(64).astype(np.float32),
        )
        w.append_candidate(
            {"srcname": "J120500.0-300500.0", "x": 100, "y": 100, "ra_deg": 180.5, "dec_deg": -30.1, "snr": 16.2, "dm": 20.0, "period_s": 0.4, "f_hz": 2.5},
            cutout=rng.standard_normal((32, 32)).astype(np.float32),
            cutout_off=rng.standard_normal((32, 32)).astype(np.float32),
            spectrum=rng.random(64).astype(np.float32),
        )

    # Beam 01, DM 40 (same source J120000.0-300000.0 at higher S/N -> exemplar!)
    h5_b1_dm40 = build_trial_h5_path(tmp_path, method="nufft", beam="beam01", dm=40.0, prefix="FIELD.SB82418")
    cut_b1_srcA_win = rng.standard_normal((32, 32)).astype(np.float32)
    spec_b1_srcA_win = rng.random(64).astype(np.float32)
    with CandidateHDF5Writer(
        h5_b1_dm40, method="nufft", freqs_hz=fkeep, cutout_size=32, wcs_full=w_b1, sbid="SB82418", field="FIELD", beam="beam01", dm=40.0
    ) as w:
        w.append_candidate(
            {"srcname": "J120000.0-300000.0", "x": 180, "y": 200, "ra_deg": 180.0, "dec_deg": -30.0, "snr": 22.8, "dm": 40.0, "period_s": 0.8, "f_hz": 1.25},
            cutout=cut_b1_srcA_win,
            cutout_off=rng.standard_normal((32, 32)).astype(np.float32),
            spectrum=spec_b1_srcA_win,
        )

    # 1. Full merge preserves all 3 candidates and both beam WCS headers
    merged_path = tmp_path / "merged_nufft.h5"
    merge_candidate_h5([h5_b0_dm20, h5_b1_dm40], merged_path)
    with CandidateStore(merged_path) as st:
        assert st.n_candidates == 3
        assert st.is_consolidated is False
        # Per-beam WCS preserved
        assert st[0].wcs_full.wcs.crval[0] == pytest.approx(180.0)
        assert st[2].wcs_full.wcs.crval[0] == pytest.approx(181.5)
        # Dict lookup picks highest S/N exemplar for J120000.0-300000.0
        assert st["J120000.0-300000.0"].snr == pytest.approx(22.8, abs=1e-4)

    # 2. Exemplar consolidation retains ONLY the 2 unique sky-source exemplars
    exemplar_tab = Table(
        rows=[
            {
                "srcname": "J120000.0-300000.0",
                "max_snr": 22.8,
                "max_snr_beam": "beam01",
                "best_dm": 40.0,
                "period_s": 0.8,
                "beams_all": "beam00,beam01",
                "dm_all": "20.00,40.00",
                "n_detections_total": 2,
            },
            {
                "srcname": "J120500.0-300500.0",
                "max_snr": 16.2,
                "max_snr_beam": "beam00",
                "best_dm": 20.0,
                "period_s": 0.4,
                "beams_all": "beam00",
                "dm_all": "20.00",
                "n_detections_total": 1,
            },
        ]
    )
    cons_path = build_consolidated_h5_path(tmp_path, method="nufft", sbid="SB82418", field="FIELD")
    out_p, matched = consolidate_candidate_h5([h5_b0_dm20, h5_b1_dm40], exemplar_tab, cons_path, method="nufft")
    assert out_p == cons_path
    assert len(matched) == 2

    with CandidateStore(cons_path) as st:
        assert st.is_consolidated is True
        assert st.n_candidates == 2
        assert len(st) == 2
        c_win = st["J120000.0-300000.0"]
        assert c_win.beam == "beam01"
        assert c_win.dm == pytest.approx(40.0)
        assert c_win.meta["beams_all"] == "beam00,beam01"
        assert c_win.meta["dm_all"] == "20.00,40.00"
        np.testing.assert_allclose(c_win.cutout, cut_b1_srcA_win)
        np.testing.assert_allclose(c_win.spectrum.power, spec_b1_srcA_win)
        assert c_win.wcs_full.wcs.crval[0] == pytest.approx(181.5)


def test_cli_merge_h5_and_plot_h5(tmp_path):
    rng = np.random.default_rng(9)
    w0 = _make_wcs()
    t = np.linspace(5.2e9, 5.2e9 + 20.0, 32, dtype=np.float64)
    p1 = build_trial_h5_path(tmp_path, method="boxcar", beam="beam00", dm=0.0, prefix="FIELD.SB82418")
    with CandidateHDF5Writer(p1, method="boxcar", cutout_size=32, wcs_full=w0, sbid="SB82418", beam="beam00", dm=0.0) as w:
        w.append_candidate(
            {"srcname": "J120000.0-300000.0", "x": 192, "y": 192, "ra_deg": 180.0, "dec_deg": -30.0, "snr": 11.0, "dm": 0.0, "width_samples": 2, "time_center": float(t[16])},
            cutout=rng.standard_normal((32, 32)).astype(np.float32),
            lightcurve=Lightcurve(time=t, flux=rng.standard_normal(32).astype(np.float32), flux_err=np.full(32, 0.1, dtype=np.float32), width_samples=2),
        )

    merged = tmp_path / "cli_merged.h5"
    rc = fd_cli.main(["merge_h5", "--search-dir", str(tmp_path), "--method", "boxcar", "--out", str(merged)])
    assert rc == 0
    assert is_valid_candidates_h5(merged)

    plot_dir = tmp_path / "cli_plots"
    rc2 = fd_cli.main(["plot_h5", "--h5", str(merged), "--outdir", str(plot_dir), "--top", "5", "--dpi", "100"])
    assert rc2 == 0
    assert len(list(plot_dir.glob("*_summary.pdf"))) == 1


def test_transient_core_hdf5_chunk_export_and_merge(tmp_path):
    rng = np.random.default_rng(99)
    cand_dir = tmp_path / "candidates"
    cand_dir.mkdir()
    ms_base = "cracoData.FIELD.SB82418.beam00.uvsub"

    cfg = Config(
        msname="/tmp/dummy.ms",
        npix_x=64,
        npix_y=64,
        pix_rad=np.deg2rad(22.0 / 3600.0),
        ra0_rad=np.deg2rad(180.0),
        dec0_rad=np.deg2rad(-30.0),
        epsilon=1e-5,
        do_wgridding=False,
        nthreads=1,
        verbosity=0,
        corr_mode="single",
        basis="linear",
        single_pol="XX",
        data_column="DATA",
        enable_var=True,
        do_var_search=True,
        enable_var_chunk=True,
        enable_var_scan=False,
        enable_var_obs=False,
        enable_boxcar=True,
        do_boxcar_search=True,
        var_threshold=8.0,
        var_keep_k=None,
        rms_clip_sigma=3.0,
        nms_radius=4,
        boxcar_widths=[1, 2],
        boxcar_threshold=8.0,
        do_plot=False,
        plot_cands_only=False,
        ms_base=ms_base,
        candidates_dir=str(cand_dir),
        chunk_prefix_root=str(cand_dir / ms_base),
        all_prefix_root=str(cand_dir / ms_base),
        save_var_lightcurves=True,
        save_full_var_lightcurves=False,
        save_var_snippets=False,
        save_box_lightcurves=True,
        save_box_snippets=False,
        continuum_dir=None,
        var_highpass_cutoff_sec=0.0,
        current_dm=15.0,
        is_dm_search=True,
        product_format="hdf5",
    )

    times = np.linspace(5.2e9, 5.2e9 + 10.0, 20, dtype=np.float64)
    cube = rng.standard_normal((20, 64, 64)).astype(np.float32)
    std_map = np.std(cube, axis=0).astype(np.float32)

    box_cands = [
        {"srcname": "J120000.0-300000.0", "x": 32, "y": 32, "ra_deg": 180.0, "dec_deg": -30.0, "snr": 12.0, "dm": 15.0, "width_samples": 2, "time_center": float(times[10]), "t0_idx": 9},
    ]
    var_cands = [
        {"srcname": "J120001.0-300001.0", "x": 20, "y": 25, "ra_deg": 180.1, "dec_deg": -30.1, "snr": 10.5, "dm": 15.0, "time_center": float(times[5])},
    ]

    chunk_root = str(cand_dir / f"{ms_base}_dm015.00_chunk_000000")
    fd_core._export_boxcar_chunk_products(cfg, times, cube, box_cands, box_root=chunk_root + "_boxcar")
    fd_core._export_variance_chunk_products(cfg, times, cube, std_map, var_cands, var_root=chunk_root + "_var")

    # Zero loose FITS/NPZ/PDF files created
    assert list(cand_dir.glob("*.fits")) == []
    assert list(cand_dir.glob("*.npz")) == []
    assert list(cand_dir.glob("*.pdf")) == []

    merged = fd_core.merge_chunk_candidate_h5s(cfg, dm=15.0)
    assert set(merged.keys()) == {"boxcar", "variance"}
    # Exactly 2 canonical per-(method, beam, DM) HDF5 files remain
    remaining_h5 = sorted(cand_dir.glob("*.h5"))
    assert len(remaining_h5) == 2

    with open_candidates(merged["boxcar"]) as st_b:
        assert st_b["J120000.0-300000.0"].cutout.shape == (50, 50)
        assert len(st_b["J120000.0-300000.0"].lightcurve) == 20
    with open_candidates(merged["variance"]) as st_v:
        assert st_v["J120001.0-300001.0"].cutout.shape == (50, 50)
        assert len(st_v["J120001.0-300001.0"].lightcurve) == 20
