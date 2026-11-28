"""Unit tests for on-the-fly chunk dynamic spectrum extraction and transient summary plotting."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

from fastducc.dynspec import (
    C_LIGHT,
    DEFAULT_MIN_UV_DISTANCE_M,
    dedisperse_dynamic_spectrum,
    extract_phased_dynamic_spectra_from_chunk,
    flag_dynspec_rfi_mad,
)
from fastducc.candidate_tools.plotting import (
    plot_summaries_from_catalogue_and_h5,
    save_candidate_summary,
)
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
        product_format="legacy",
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


def test_flag_dynspec_rfi_mad_preserves_narrow_burst_and_flags_rfi_channels():
    """Frequency-only MAD flagging flags narrowband RFI channels while preserving bright narrow bursts."""
    rng = np.random.default_rng(123)
    nt, nchan = 256, 64
    ds = rng.normal(0.0, 1.0, size=(nt, nchan)).astype(np.float32)

    # Inject a very bright, 1-sample broadband transient burst at t=120
    burst_t = 120
    ds[burst_t, :] += 25.0

    # Inject strong narrowband RFI into 3 channels and a dead zero channel
    rfi_chans = [5, 27, 51]
    for c in rfi_chans:
        ds[:, c] += rng.normal(0.0, 15.0, size=nt).astype(np.float32)
    dead_chan = 40
    ds[:, dead_chan] = 0.0

    cleaned, mask = flag_dynspec_rfi_mad(ds, rfi_mad_thresh=3.0)

    assert mask.shape == (nchan,)
    # All narrowband RFI channels and the dead channel must be flagged
    for c in rfi_chans + [dead_chan]:
        assert bool(mask[c]) is True
        assert np.all(np.isnan(cleaned[:, c]))

    # Majority of clean channels remain unflagged and preserve the narrow transient burst
    clean_chans = [c for c in range(nchan) if c not in rfi_chans + [dead_chan]]
    n_false_pos = int(np.sum(mask[clean_chans]))
    assert n_false_pos <= 2

    # Frequency-averaged beamformed time series preserves the burst peak at burst_t
    beam_lc = np.nanmean(cleaned, axis=1)
    assert int(np.argmax(beam_lc)) == burst_t
    assert float(beam_lc[burst_t]) > 20.0

    # Disabling via rfi_mad_thresh <= 0 leaves data untouched
    unflagged, mask_off = flag_dynspec_rfi_mad(ds, rfi_mad_thresh=0.0)
    assert not np.any(mask_off)
    assert np.array_equal(unflagged, ds)


def test_dedisperse_dynamic_spectrum_aligns_pulse_and_masks_edges():
    """Padded dedispersion aligns a dispersed pulse to f_max and masks shifted edges with NaN."""
    from fastducc.dedisp import get_dm_delays

    nt, nchan = 200, 32
    tsamp_s = 0.11
    freqs_mhz = np.linspace(800.0, 1088.0, nchan, dtype=np.float64)
    dm_true = 500.0
    t_peak = 80

    delays = get_dm_delays(freqs_mhz * 1e6, dm_true, tsamp_s, align_to="fmax")
    max_delay = int(np.max(delays))
    assert max_delay > 10

    # Inject a pulse dispersed according to fastducc.dedisp (align_to="fmax")
    # plus a bright marker at t=0 to verify no wrap-around into trailing edge bins
    ds = np.zeros((nt, nchan), dtype=np.float32)
    ds[0, :] = 999.0
    for c in range(nchan):
        t_arr = t_peak + int(delays[c])
        if 0 <= t_arr < nt:
            ds[t_arr, c] = 10.0

    dedisp = dedisperse_dynamic_spectrum(ds, freqs_mhz, tsamp_s, dm_true, align_to="fmax")
    assert dedisp.shape == ds.shape

    # All channels should align at t_peak
    assert np.allclose(dedisp[t_peak, :], 10.0)

    # Trailing shifted edge bins for delayed channels must be NaN (never wrapped from t=0)
    for c in range(nchan):
        d = int(delays[c])
        if d > 0:
            assert np.all(np.isnan(dedisp[nt - d :, c]))
            assert not np.any(dedisp[nt - d :, c] == 999.0)


def test_extract_phased_dynamic_spectra_from_chunk_uv_clip_default_and_opt_out():
    """Dynamic spectrum extraction clips short baselines (<200 m) by default and supports opt-out."""
    times, chan_freq, vis_3d, wgt_3d, uvw_3d, ra0_rad, dec0_rad, ra_rad, dec_rad, true_spec = _make_synthetic_chunk()
    assert DEFAULT_MIN_UV_DISTANCE_M == 200.0

    # Corrupt short baseline 0 (~14 m) with huge spurious amplitude
    vis_3d[0, :, :] = 1e5 + 0j
    targets = [{"id": "SRC_OFFSET", "ra_rad": ra_rad, "dec_rad": dec_rad}]

    # Default (200 m UV cut) automatically excludes baseline 0
    res_default = extract_phased_dynamic_spectra_from_chunk(
        times=times,
        chan_freq=chan_freq,
        vis_3d=vis_3d,
        wgt_3d=wgt_3d,
        uvw_3d=uvw_3d,
        ra0_rad=ra0_rad,
        dec0_rad=dec0_rad,
        targets=targets,
    )
    assert res_default["SRC_OFFSET"]["min_uv_distance_m"] == pytest.approx(200.0)
    assert np.allclose(res_default["SRC_OFFSET"]["data"], true_spec, rtol=1e-4, atol=1e-4)

    # Explicit opt-out (min_uv_distance_m=0.0) retains baseline 0
    res_unclipped = extract_phased_dynamic_spectra_from_chunk(
        times=times,
        chan_freq=chan_freq,
        vis_3d=vis_3d,
        wgt_3d=wgt_3d,
        uvw_3d=uvw_3d,
        ra0_rad=ra0_rad,
        dec0_rad=dec0_rad,
        targets=targets,
        min_uv_distance_m=0.0,
    )
    assert res_unclipped["SRC_OFFSET"]["min_uv_distance_m"] == 0.0
    assert not np.allclose(res_unclipped["SRC_OFFSET"]["data"], true_spec, rtol=1e-2, atol=1e-2)


def test_standalone_plot_summaries_from_catalogue_and_h5(tmp_path):
    """Regenerate transient summary PDF/PNG directly from a final catalogue and HDF5 container."""
    from astropy.table import Table
    from fastducc import wcs as ducc_wcs
    from fastducc.candidate_store import CandidateHDF5Writer
    from fastducc.cli import main as cli_main
    from fastducc.dedisp import get_dm_delays

    rng = np.random.default_rng(99)
    nt, nchan = 128, 32
    tsamp_s = 0.11
    t0_raw = 5278350400.0
    times = t0_raw + np.arange(nt, dtype=np.float64) * tsamp_s
    freqs_mhz = np.linspace(800.0, 1088.0, nchan, dtype=np.float64)
    dm_val = 230.90
    t_peak = 50

    ds_data = rng.normal(0.0, 0.02, size=(nt, nchan)).astype(np.float32)
    delays = get_dm_delays(freqs_mhz * 1e6, dm_val, tsamp_s, align_to="fmax")
    for c in range(nchan):
        idx = t_peak + int(delays[c])
        if 0 <= idx < nt:
            ds_data[idx : min(nt, idx + 3), c] += 0.25
    ds_data[:, 7] += rng.normal(0.0, 5.0, size=nt).astype(np.float32)

    lc_flux = rng.normal(0.0, 0.005, size=nt).astype(np.float32)
    lc_flux[t_peak : t_peak + 4] += 0.04
    lc_err = np.full(nt, 0.005, dtype=np.float32)

    cutout = rng.normal(0.0, 0.002, size=(50, 50)).astype(np.float32)
    cutout[24:27, 24:27] += 0.03
    fullframe = rng.normal(0.0, 0.002, size=(64, 64)).astype(np.float32)
    fullframe[31:34, 31:34] += 0.03

    ra0_rad = math.radians(272.29)
    dec0_rad = math.radians(-31.19)
    wcs_full = ducc_wcs._build_fullframe_wcs(
        npix_x=64,
        npix_y=64,
        ra0_rad=ra0_rad,
        dec0_rad=dec0_rad,
        pixscale_rad=math.radians(22.0 / 3600.0),
        ra_sign=-1,
        dec_sign=-1,
    )

    h5_path = tmp_path / "LTR_1812-2849.SB82418_obs_boxcar_consolidated_candidates.h5"
    with CandidateHDF5Writer(
        h5_path,
        method="boxcar",
        cutout_size=50,
        npix_x=64,
        npix_y=64,
        wcs_full=wcs_full,
        sbid="SB82418",
        field="LTR_1812-2849",
        is_consolidated=True,
    ) as writer:
        writer.append_candidate(
            {
                "srcname": "J180910.0-311123.4",
                "beam": "beam28",
                "sbid": "SB82418",
                "field": "LTR_1812-2849",
                "scan_id": "20260221001346",
                "x": 32,
                "y": 32,
                "ra_deg": 272.2917,
                "dec_deg": -31.1898,
                "ra_hms": "18:09:10.0",
                "dec_dms": "-31:11:23.4",
                "snr": 13.16,
                "dm": dm_val,
                "width_samples": 12,
                "time_center": float(times[t_peak]),
                "time_start": float(times[t_peak - 2]),
                "time_end": float(times[t_peak + 10]),
            },
            cutout=cutout,
            fullframe=fullframe,
            lightcurve={"time": times, "flux": lc_flux, "flux_err": lc_err},
            dynspec={
                "data": ds_data,
                "time": times - times[0],
                "time_raw": times,
                "freq": freqs_mhz,
            },
        )

    cat_csv = tmp_path / "SB82418_boxcar_super_summary.csv"
    cat_vot = tmp_path / "SB82418_boxcar_super_summary.vot"
    tbl = Table({
        "source_id": [776],
        "srcname": ["J180910.0-311123.4"],
        "ra_deg": [272.2917],
        "dec_deg": [-31.1898],
        "dm": [dm_val],
        "dm_all": ["0.00,230.90"],
        "n_events": [17],
        "beams_all": ["beam11,beam28"],
        "max_snr": [13.16],
        "max_snr_time_center": [float(times[t_peak])],
        "max_snr_width": [12],
        "max_snr_beam": ["beam28"],
    })
    tbl.write(cat_csv, format="ascii.csv", overwrite=True)
    tbl.write(cat_vot, format="votable", overwrite=True)

    # Test Python API with VOTable catalogue
    out_api = tmp_path / "plots_api"
    res = plot_summaries_from_catalogue_and_h5(
        h5_path=h5_path,
        out_dir=out_api,
        catalogue_path=cat_vot,
        dpi=72,
        save_png=True,
    )
    assert res["rendered"] == 1
    assert res["missing"] == 0
    assert Path(res["outputs"][0]["figure"]).is_file()
    assert Path(res["outputs"][0]["png"]).is_file()

    # Test standalone CLI subcommand with CSV catalogue
    out_cli = tmp_path / "plots_cli"
    rc = cli_main([
        "plot-summaries",
        "--h5", str(h5_path),
        "--catalogue", str(cat_csv),
        "--outdir", str(out_cli),
        "--dpi", "72",
        "--png",
    ])
    assert rc == 0
    assert len(list(out_cli.glob("*_summary.pdf"))) == 1
    assert len(list(out_cli.glob("*_summary.png"))) == 1


