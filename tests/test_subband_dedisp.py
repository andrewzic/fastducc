# -*- coding: utf-8 -*-
"""
Unit, numerical parity, and scientific fidelity tests for Two-Tier Subbanded
Dedispersion (Dense-in-Time, Sparse-in-(u, v) Active Cell Matrix) in fastducc.
"""

from __future__ import annotations

import numpy as np
import pytest

from fastducc import cli
from fastducc.constants import C_LIGHT
from fastducc.dedisp import DedispersionPlan, K_DM, get_dm_delays
from fastducc.fastducc_run import build_cli as build_cli_search
from fastducc.gpu_direct_cube import stream_ms_to_ragged_csr
from fastducc.gpu_raster import (
    build_uv_to_m_lookup,
    compute_raster_grid_geometry,
    dedisperse_active_cell_matrix_cpu,
    dedisperse_active_cell_matrix_gpu,
    dedisperse_and_rasterise_chunk_cpu,
    dedisperse_and_rasterise_subband_cpu,
    dedisperse_and_rasterise_subband_gpu,
    is_cuda_available,
)

try:
    from casacore.tables import makearrcoldesc, makescacoldesc, maketabdesc, table

    HAS_CASACORE = True
except ImportError:
    HAS_CASACORE = False


def _rel_l2_error(a: np.ndarray, b: np.ndarray) -> float:
    """Compute relative L2 error ||a - b||_2 / max(||b||_2, 1e-12)."""
    diff = np.asarray(a, dtype=np.complex128) - np.asarray(b, dtype=np.complex128)
    ref = np.asarray(b, dtype=np.complex128)
    return float(np.linalg.norm(diff.ravel())) / max(float(np.linalg.norm(ref.ravel())), 1e-12)


def _synthesise_dispersed_pulse(
    *,
    n_ant: int = 10,
    b_max_m: float = 2500.0,
    n_chan: int = 64,
    n_time: int = 160,
    t_samp_s: float = 0.005,
    pulse_dm: float = 150.0,
    pulse_t0_s: float = 0.20,
    pulse_width_s: float = 0.015,
    pulse_flux_jy: float = 10.0,
    source_l_deg: float = 0.0,
    source_m_deg: float = 0.0,
    seed: int = 42,
):
    """
    Generate synthetic dispersed point-source visibilities matching fastducc's
    non-inverted (u_m, v_m) and causal align_to='fmax' dispersion conventions.
    """
    rng = np.random.default_rng(seed)
    ant_xy = rng.uniform(-0.5 * b_max_m, 0.5 * b_max_m, size=(n_ant, 2))
    bls = []
    for i in range(n_ant):
        for j in range(i + 1, n_ant):
            bls.append(ant_xy[j] - ant_xy[i])
    bls_m = np.asarray(bls, dtype=np.float64)  # (nbl, 2)
    nbl = bls_m.shape[0]

    freqs_hz = np.linspace(1256.0e6, 1544.0e6, n_chan, dtype=np.float64)
    freqs_ghz = freqs_hz * 1e-9
    f_max_ghz = float(np.max(freqs_ghz))

    uvw_3d = np.zeros((nbl, 3, n_time), dtype=np.float64)
    uvw_3d[:, 0, :] = bls_m[:, 0:1]
    uvw_3d[:, 1, :] = bls_m[:, 1:2]

    t_axis = np.arange(n_time, dtype=np.float64) * t_samp_s
    sigma_t = pulse_width_s / 2.354820045  # FWHM -> sigma

    l0 = np.deg2rad(source_l_deg)
    m0 = np.deg2rad(source_m_deg)

    vis_3d = np.zeros((nbl, n_chan, n_time), dtype=np.complex64)
    wgt_3d = np.ones((nbl, n_chan, n_time), dtype=np.float32)

    for c in range(n_chan):
        tau_c = K_DM * pulse_dm * (1.0 / (freqs_ghz[c] ** 2) - 1.0 / (f_max_ghz ** 2))
        env = pulse_flux_jy * np.exp(-0.5 * ((t_axis - (pulse_t0_s + tau_c)) / sigma_t) ** 2)
        u_lam = bls_m[:, 0] * (freqs_hz[c] / C_LIGHT)
        v_lam = bls_m[:, 1] * (freqs_hz[c] / C_LIGHT)
        phase = np.exp(-2j * np.pi * (u_lam * l0 + v_lam * m0))
        vis_3d[:, c, :] = (phase[:, None] * env[None, :]).astype(np.complex64)

    return vis_3d, wgt_3d, uvw_3d, freqs_hz


def _invert_halfplane_to_dirty(uv_half: np.ndarray, grid_nu: int, npix_x: int, npix_y: int) -> np.ndarray:
    """
    Invert Hermitian half-plane UV cube ``(nt, nv, u_half)`` to real dirty image
    snapshots ``(nt, npix_y, npix_x)`` conforming to Stage 3 c2r imaging.
    """
    nt, nv, _ = uv_half.shape
    img_full = np.fft.irfft2(uv_half, s=(nv, grid_nu), axes=(1, 2))
    img_shifted = np.fft.fftshift(img_full, axes=(1, 2))
    cy, cx = nv // 2, grid_nu // 2
    hy, hx = npix_y // 2, npix_x // 2
    return img_shifted[:, cy - hy : cy - hy + npix_y, cx - hx : cx - hx + npix_x].astype(np.float32)


def _measure_fwhm_samps(lightcurve: np.ndarray) -> float:
    peak_idx = int(np.argmax(lightcurve))
    half_max = 0.5 * float(lightcurve[peak_idx])
    left = peak_idx
    while left > 0 and lightcurve[left] > half_max:
        left -= 1
    right = peak_idx
    while right < len(lightcurve) - 1 and lightcurve[right] > half_max:
        right += 1
    return max(1.0, float(right - left))


# =============================================================================
# 1. DedispersionPlan Unit Tests
# =============================================================================


def test_dedispersion_plan_subbanded_vs_fullband():
    freqs_hz = np.linspace(1256.0e6, 1544.0e6, 288, dtype=np.float64)
    tsamp_s = 0.005

    plan_sub = DedispersionPlan(
        freqs_hz=freqs_hz,
        tsamp_s=tsamp_s,
        dm_min=0.0,
        dm_max=500.0,
        dedisp_mode="subbanded",
        n_subbands=16,
    )
    plan_full = DedispersionPlan(
        freqs_hz=freqs_hz,
        tsamp_s=tsamp_s,
        dm_min=0.0,
        dm_max=500.0,
        dedisp_mode="fullband",
        n_subbands=16,
    )

    assert plan_sub.dedisp_mode == "subbanded"
    assert plan_full.dedisp_mode == "fullband"
    assert plan_sub.n_subbands == 16
    assert plan_sub.ndm == plan_full.ndm
    assert np.allclose(plan_sub.trial_dms, plan_full.trial_dms)

    # Coarse DM step formula: t_samp * nu_min^3 / (2 * k_DM * delta_nu_sub)
    sub_bw_ghz = (1.544 - 1.256) / 16.0
    expected_coarse_step = (tsamp_s * (1.256 ** 3)) / (2.0 * K_DM * sub_bw_ghz)
    assert np.isclose(plan_sub.coarse_dm_step, expected_coarse_step, rtol=1e-6)
    assert plan_sub.n_coarse < plan_sub.ndm
    assert plan_full.n_coarse == plan_full.ndm

    # Causal non-negative delay bounds
    assert plan_sub.intra_delays.shape == (plan_sub.n_coarse, 288)
    assert plan_sub.inter_delays.shape == (plan_sub.ndm, 16)
    assert int(np.min(plan_sub.intra_delays)) >= 0
    assert int(np.min(plan_sub.inter_delays)) >= 0

    # Combined Tier 1 + Tier 2 delay must match exact fullband channel delay within <= 1 sample
    max_err = 0
    for ifine, dm_f in enumerate(plan_sub.trial_dms):
        ic = int(plan_sub.fine_to_coarse_idx[ifine])
        exact = plan_sub.get_channel_delays(float(dm_f), align_to="fmax")
        for s in range(plan_sub.n_subbands):
            c0 = int(plan_sub.subband_offsets[s])
            c1 = int(plan_sub.subband_offsets[s + 1])
            combined = plan_sub.intra_delays[ic, c0:c1] + plan_sub.inter_delays[ifine, s]
            err = int(np.max(np.abs(combined - exact[c0:c1])))
            max_err = max(max_err, err)
    assert max_err <= 1, f"Intra+inter delay error exceeded 1 sample: {max_err}"

    assert "subbanded" in plan_sub.summary()
    assert "Subbands: 16" in plan_sub.summary()


def test_dedispersion_plan_validation():
    freqs_hz = np.linspace(1256.0e6, 1544.0e6, 32, dtype=np.float64)
    with pytest.raises(ValueError, match="dedisp_mode"):
        DedispersionPlan(freqs_hz, 0.005, dedisp_mode="invalid")
    with pytest.raises(ValueError, match="n_subbands"):
        DedispersionPlan(freqs_hz, 0.005, n_subbands=0)


# =============================================================================
# 2. Numerical Parity: Numba CPU vs NumPy CPU vs CUDA
# =============================================================================


@pytest.mark.parametrize("mode", ["nearest", "bilinear", "wgridder"])
def test_tier1_subband_numba_vs_numpy_parity(mode: str):
    vis_3d, wgt_3d, uvw_3d, freqs_hz = _synthesise_dispersed_pulse(
        n_ant=6, n_chan=32, n_time=40, t_samp_s=0.005, pulse_dm=120.0
    )
    geom = compute_raster_grid_geometry(npix_x=64, npix_y=64, pixsize_arcsec=60.0, mode=mode)
    nv, u_half = int(geom["nv"]), int(geom["u_half"])
    flat_indices = np.arange(nv * u_half, dtype=np.int64)
    uv_to_m = build_uv_to_m_lookup(nv=nv, u_half=u_half, flat_indices=flat_indices, device="cpu")

    plan = DedispersionPlan(freqs_hz, 0.005, dm_min=120.0, dm_max=120.0, n_subbands=8)
    intra = plan.get_intra_delays(120.0, align_to="fmax")

    out_numba = dedisperse_and_rasterise_subband_cpu(
        vis_3d,
        wgt_3d,
        uvw_3d,
        freqs_hz,
        intra_delays=intra,
        n_subbands=8,
        nt_out=28,
        npix_x=64,
        npix_y=64,
        pixsize_arcsec=60.0,
        mode=mode,
        min_valid_channels=4,
        uv_to_m=uv_to_m,
        m_active=len(flat_indices),
        use_numba=True,
    )
    out_numpy = dedisperse_and_rasterise_subband_cpu(
        vis_3d,
        wgt_3d,
        uvw_3d,
        freqs_hz,
        intra_delays=intra,
        n_subbands=8,
        nt_out=28,
        npix_x=64,
        npix_y=64,
        pixsize_arcsec=60.0,
        mode=mode,
        min_valid_channels=4,
        uv_to_m=uv_to_m,
        m_active=len(flat_indices),
        use_numba=False,
    )
    assert out_numba.shape == (len(flat_indices), 8, 28)
    assert _rel_l2_error(out_numba, out_numpy) < 1e-6


@pytest.mark.parametrize("with_weights", [False, True])
def test_tier2_active_cell_numba_vs_numpy_parity(with_weights: bool):
    rng = np.random.default_rng(123)
    m_active, n_sub, nt_in, nt_out, n_dm = 128, 16, 64, 48, 12
    active_dga = (
        rng.standard_normal((m_active, n_sub, nt_in)) + 1j * rng.standard_normal((m_active, n_sub, nt_in))
    ).astype(np.complex64)
    active_wgt = rng.uniform(0.5, 2.0, size=(m_active, n_sub, nt_in)).astype(np.float32) if with_weights else None
    inter_delays = rng.integers(0, 15, size=(n_dm, n_sub), dtype=np.int32)

    out_numba = dedisperse_active_cell_matrix_cpu(
        active_dga, inter_delays, nt_out=nt_out, active_wgt=active_wgt, uv_threshold=1e-4, use_numba=True
    )
    out_numpy = dedisperse_active_cell_matrix_cpu(
        active_dga, inter_delays, nt_out=nt_out, active_wgt=active_wgt, uv_threshold=1e-4, use_numba=False
    )
    assert out_numba.shape == (n_dm, m_active, nt_out)
    assert _rel_l2_error(out_numba, out_numpy) < 1e-6

    # 1D inter_delays squeeze parity
    out_1d = dedisperse_active_cell_matrix_cpu(
        active_dga, inter_delays[0], nt_out=nt_out, active_wgt=active_wgt, uv_threshold=1e-4, use_numba=True
    )
    assert out_1d.shape == (m_active, nt_out)
    assert _rel_l2_error(out_1d, out_numba[0]) < 1e-6


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA device unavailable on this node")
@pytest.mark.parametrize("mode", ["nearest", "bilinear", "wgridder"])
def test_tier1_and_tier2_cuda_vs_cpu_parity(mode: str):
    vis_3d, wgt_3d, uvw_3d, freqs_hz = _synthesise_dispersed_pulse(
        n_ant=8, n_chan=32, n_time=48, t_samp_s=0.005, pulse_dm=150.0
    )
    geom = compute_raster_grid_geometry(npix_x=64, npix_y=64, pixsize_arcsec=60.0, mode=mode)
    nv, u_half = int(geom["nv"]), int(geom["u_half"])
    flat_indices = np.arange(nv * u_half, dtype=np.int64)
    uv_to_m = build_uv_to_m_lookup(nv=nv, u_half=u_half, flat_indices=flat_indices, device="cpu")

    plan = DedispersionPlan(freqs_hz, 0.005, dm_min=100.0, dm_max=200.0, n_subbands=8)
    intra = plan.intra_delays[0]

    t1_cpu = dedisperse_and_rasterise_subband_cpu(
        vis_3d, wgt_3d, uvw_3d, freqs_hz, intra_delays=intra, n_subbands=8,
        nt_out=32, npix_x=64, npix_y=64, pixsize_arcsec=60.0, mode=mode,
        min_valid_channels=4, uv_to_m=uv_to_m, m_active=len(flat_indices),
    )
    t1_gpu = dedisperse_and_rasterise_subband_gpu(
        vis_3d, wgt_3d, uvw_3d, freqs_hz, intra_delays=intra, n_subbands=8,
        nt_out=32, npix_x=64, npix_y=64, pixsize_arcsec=60.0, mode=mode,
        min_valid_channels=4, uv_to_m=uv_to_m, m_active=len(flat_indices),
        return_host=True,
    )
    assert _rel_l2_error(t1_gpu, t1_cpu) < 1e-6

    t2_cpu = dedisperse_active_cell_matrix_cpu(t1_cpu, plan.inter_delays, nt_out=20, uv_threshold=1e-4)
    t2_gpu = dedisperse_active_cell_matrix_gpu(t1_gpu, plan.inter_delays, nt_out=20, uv_threshold=1e-4, return_host=True)
    assert _rel_l2_error(t2_gpu, t2_cpu) < 1e-6

    # Also verify weight-normalised Tier 2 CUDA vs CPU parity
    t1_cpu_v, t1_cpu_w = dedisperse_and_rasterise_subband_cpu(
        vis_3d, wgt_3d, uvw_3d, freqs_hz, intra_delays=intra, n_subbands=8,
        nt_out=32, npix_x=64, npix_y=64, pixsize_arcsec=60.0, mode=mode,
        min_valid_channels=4, uv_to_m=uv_to_m, m_active=len(flat_indices),
        return_weights=True,
    )
    t1_gpu_v, t1_gpu_w = dedisperse_and_rasterise_subband_gpu(
        vis_3d, wgt_3d, uvw_3d, freqs_hz, intra_delays=intra, n_subbands=8,
        nt_out=32, npix_x=64, npix_y=64, pixsize_arcsec=60.0, mode=mode,
        min_valid_channels=4, uv_to_m=uv_to_m, m_active=len(flat_indices),
        return_weights=True, return_host=True,
    )
    assert _rel_l2_error(t1_gpu_v, t1_cpu_v) < 1e-6
    assert _rel_l2_error(t1_gpu_w, t1_cpu_w) < 1e-6
    t2_w_cpu = dedisperse_active_cell_matrix_cpu(
        t1_cpu_v, plan.inter_delays, nt_out=20, active_wgt=t1_cpu_w, uv_threshold=1e-4
    )
    t2_w_gpu = dedisperse_active_cell_matrix_gpu(
        t1_gpu_v, plan.inter_delays, nt_out=20, active_wgt=t1_gpu_w, uv_threshold=1e-4, return_host=True
    )
    assert _rel_l2_error(t2_w_gpu, t2_w_cpu) < 1e-6


# =============================================================================
# 3. Scientific Fidelity: Subbanded vs Fullband on Synthetic Point Sources
# =============================================================================


@pytest.mark.parametrize(
    "source_l_deg,source_m_deg",
    [
        (0.0, 0.0),
        (0.05, 0.02),
    ],
)
def test_scientific_fidelity_subbanded_vs_fullband(source_l_deg: float, source_m_deg: float):
    """
    Verify on synthetic dispersed point sources (both on-axis and off-axis):
      - Peak flux recovery > 96%
      - Zero temporal broadening (1.00x FWHM) and 0 sample timing offset
      - Astrometric centroid error == 0.00 pixels
    """
    npix = 128
    pixsize_arcsec = 18.0  # FOV = 0.64 deg (+/- 0.32 deg)
    dm_val = 150.0
    t_samp = 0.005
    n_sub = 16

    vis_3d, wgt_3d, uvw_3d, freqs_hz = _synthesise_dispersed_pulse(
        n_ant=10,
        b_max_m=2500.0,
        n_chan=64,
        n_time=160,
        t_samp_s=t_samp,
        pulse_dm=dm_val,
        pulse_t0_s=0.20,
        pulse_width_s=0.015,
        pulse_flux_jy=10.0,
        source_l_deg=source_l_deg,
        source_m_deg=source_m_deg,
    )

    geom = compute_raster_grid_geometry(
        npix_x=npix, npix_y=npix, pixsize_arcsec=pixsize_arcsec, mode="wgridder"
    )
    grid_nu = int(geom["grid_nu"])
    nv = int(geom["nv"])
    u_half = int(geom["u_half"])

    plan = DedispersionPlan(
        freqs_hz=freqs_hz,
        tsamp_s=t_samp,
        dm_min=dm_val,
        dm_max=dm_val,
        dedisp_mode="subbanded",
        n_subbands=n_sub,
    )
    max_total_delay = int(np.max(get_dm_delays(freqs_hz, dm_val, t_samp, align_to="fmax")))
    nt_eval = 160 - max_total_delay - 2

    # 1. Fullband brute-force reference
    uv_full = dedisperse_and_rasterise_chunk_cpu(
        vis_3d=vis_3d,
        wgt_3d=wgt_3d,
        uvw_3d=uvw_3d,
        freqs_hz=freqs_hz,
        dm=dm_val,
        dt=t_samp,
        nt_out=nt_eval,
        npix_x=npix,
        npix_y=npix,
        pixsize_arcsec=pixsize_arcsec,
        mode="wgridder",
        do_wgridding=False,
        min_valid_channels=8,
    )
    dirty_full = _invert_halfplane_to_dirty(uv_full, grid_nu=grid_nu, npix_x=npix, npix_y=npix)

    # 2. Two-Tier Subbanded Dedispersion via Dense-in-Time Active UV Cell Matrix
    flat_indices = np.arange(nv * u_half, dtype=np.int64)
    uv_to_m = build_uv_to_m_lookup(nv=nv, u_half=u_half, flat_indices=flat_indices, device="cpu")
    intra_delays = plan.get_intra_delays(dm_val, align_to="fmax")
    inter_delays = plan.get_inter_delays(dm_val, align_to="fmax")
    nt_tier1 = nt_eval + int(np.max(inter_delays))

    active_dga = dedisperse_and_rasterise_subband_cpu(
        vis_3d=vis_3d,
        wgt_3d=wgt_3d,
        uvw_3d=uvw_3d,
        freqs_hz=freqs_hz,
        intra_delays=intra_delays,
        n_subbands=n_sub,
        nt_out=nt_tier1,
        npix_x=npix,
        npix_y=npix,
        pixsize_arcsec=pixsize_arcsec,
        mode="wgridder",
        min_valid_channels=8,
        uv_to_m=uv_to_m,
        m_active=len(flat_indices),
    )
    dedisp_active = dedisperse_active_cell_matrix_cpu(
        active_dga, inter_delays, nt_out=nt_eval
    )  # (M_active, nt_eval)
    uv_sub = dedisp_active.T.reshape((nt_eval, nv, u_half))
    dirty_sub = _invert_halfplane_to_dirty(uv_sub, grid_nu=grid_nu, npix_x=npix, npix_y=npix)

    # Compare peak flux, FWHM, timing, and astrometric centroid
    lc_full = np.max(dirty_full, axis=(1, 2))
    lc_sub = np.max(dirty_sub, axis=(1, 2))

    t_peak_full = int(np.argmax(lc_full))
    t_peak_sub = int(np.argmax(lc_sub))
    assert t_peak_sub == t_peak_full, f"Timing offset: sub={t_peak_sub}, full={t_peak_full}"

    peak_ratio = float(lc_sub[t_peak_sub] / lc_full[t_peak_full])
    assert peak_ratio > 0.96, f"Peak flux recovery {peak_ratio:.4f} <= 0.96"

    fwhm_full = _measure_fwhm_samps(lc_full)
    fwhm_sub = _measure_fwhm_samps(lc_sub)
    assert np.isclose(fwhm_sub / fwhm_full, 1.0, atol=1e-6), (
        f"Temporal broadening detected: fwhm_sub={fwhm_sub}, fwhm_full={fwhm_full}"
    )

    pos_full = np.unravel_index(int(np.argmax(dirty_full[t_peak_full])), dirty_full[t_peak_full].shape)
    pos_sub = np.unravel_index(int(np.argmax(dirty_sub[t_peak_sub])), dirty_sub[t_peak_sub].shape)
    centroid_err = float(np.hypot(pos_sub[0] - pos_full[0], pos_sub[1] - pos_full[1]))
    assert centroid_err == 0.0, f"Astrometric centroid error {centroid_err:.2f} pix != 0.00"


# =============================================================================
# 4. CLI Flags & Streaming Integration Tests
# =============================================================================


def test_cli_dedisp_mode_and_nsubbands():
    args_default = build_cli_search(["--msname", "test.ms"])
    assert args_default.dedisp_mode == "subbanded"
    assert args_default.nsubbands == 16

    args_full = build_cli_search(["--msname", "test.ms", "--dedisp-mode", "fullband", "--nsubbands", "8"])
    assert args_full.dedisp_mode == "fullband"
    assert args_full.nsubbands == 8

    args_per = cli.build_cli_periodicity(["--msname", "test.ms"])
    assert args_per.dedisp_mode == "subbanded"
    assert args_per.nsubbands == 16

    args_per_fb = cli.build_cli_periodicity(["--msname", "test.ms", "--dedisp-mode", "fullband", "--nsubbands", "4"])
    assert args_per_fb.dedisp_mode == "fullband"
    assert args_per_fb.nsubbands == 4


@pytest.mark.skipif(not HAS_CASACORE, reason="python-casacore unavailable")
def test_stream_ms_to_ragged_csr_subbanded_mode(tmp_path):
    """Verify end-to-end streaming in subbanded vs fullband mode on a synthetic MS."""
    vis_3d, wgt_3d, uvw_3d, freqs_hz = _synthesise_dispersed_pulse(
        n_ant=5, n_chan=32, n_time=40, t_samp_s=0.005, pulse_dm=80.0
    )
    nbl, nchan, nt = vis_3d.shape
    ncorr = 1
    nrows = nbl * nt

    ms_path = str(tmp_path / "synth_subband.ms")
    desc = maketabdesc(
        [
            makescacoldesc("TIME", 0.0, valuetype="double"),
            makescacoldesc("ANTENNA1", 0, valuetype="int"),
            makescacoldesc("ANTENNA2", 0, valuetype="int"),
            makescacoldesc("SCAN_NUMBER", 1, valuetype="int"),
            makescacoldesc("FLAG_ROW", False, valuetype="bool"),
            makearrcoldesc("UVW", 0.0, valuetype="double", shape=[3]),
            makearrcoldesc("DATA", complex(0.0, 0.0), valuetype="complex", shape=[nchan, ncorr]),
            makearrcoldesc("WEIGHT", 1.0, valuetype="float", shape=[ncorr]),
            makearrcoldesc("FLAG", False, valuetype="bool", shape=[nchan, ncorr]),
        ]
    )
    with table(ms_path, desc, nrow=nrows, ack=False) as tb:
        ant_pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
        times = np.repeat(5.0e9 + np.arange(nt, dtype=np.float64) * 0.005, nbl)
        a1 = np.tile([p[0] for p in ant_pairs], nt).astype(np.int32)
        a2 = np.tile([p[1] for p in ant_pairs], nt).astype(np.int32)
        uvw_rows = np.transpose(uvw_3d, (2, 0, 1)).reshape(nrows, 3)
        data_rows = np.transpose(vis_3d, (2, 0, 1)).reshape(nrows, nchan, ncorr)
        tb.putcol("TIME", times)
        tb.putcol("ANTENNA1", a1)
        tb.putcol("ANTENNA2", a2)
        tb.putcol("SCAN_NUMBER", np.ones(nrows, dtype=np.int32))
        tb.putcol("FLAG_ROW", np.zeros(nrows, dtype=bool))
        tb.putcol("UVW", uvw_rows)
        tb.putcol("DATA", data_rows)
        tb.putcol("WEIGHT", np.ones((nrows, ncorr), dtype=np.float32))
        tb.putcol("FLAG", np.zeros((nrows, nchan, ncorr), dtype=bool))

    # Minimal subtables required by metadata prescan
    spw_desc = maketabdesc([makearrcoldesc("CHAN_FREQ", 0.0, valuetype="double", shape=[nchan])])
    with table(f"{ms_path}/SPECTRAL_WINDOW", spw_desc, nrow=1, ack=False) as t_spw:
        t_spw.putcol("CHAN_FREQ", freqs_hz[None, :])

    pol_desc = maketabdesc([makearrcoldesc("CORR_TYPE", 9, valuetype="int", shape=[ncorr])])
    with table(f"{ms_path}/POLARIZATION", pol_desc, nrow=1, ack=False) as t_pol:
        t_pol.putcol("CORR_TYPE", np.array([[9]], dtype=np.int32))

    fld_desc = maketabdesc(
        [
            makescacoldesc("NAME", "SYNTH", valuetype="string"),
            makearrcoldesc("PHASE_DIR", 0.0, valuetype="double", shape=[1, 2]),
        ]
    )
    with table(f"{ms_path}/FIELD", fld_desc, nrow=1, ack=False) as t_fld:
        t_fld.putcol("NAME", ["SYNTH"])
        t_fld.putcol("PHASE_DIR", np.zeros((1, 1, 2), dtype=np.float64))

    res_sub = stream_ms_to_ragged_csr(
        ms_path,
        npix_x=64,
        npix_y=64,
        pixsize_arcsec=60.0,
        dm=80.0,
        dedisp_mode="subbanded",
        nsubbands=8,
        mode="wgridder",
        do_wgridding=False,
        min_valid_channels=4,
        uv_threshold=1e-6,
        device="cpu",
    )
    res_full = stream_ms_to_ragged_csr(
        ms_path,
        npix_x=64,
        npix_y=64,
        pixsize_arcsec=60.0,
        dm=80.0,
        dedisp_mode="fullband",
        nsubbands=1,
        mode="wgridder",
        do_wgridding=False,
        min_valid_channels=4,
        uv_threshold=1e-6,
        device="cpu",
    )

    assert res_sub.attrs["dedisp_mode"] == "subbanded"
    assert res_sub.attrs["nsubbands"] == 8
    assert res_full.attrs["dedisp_mode"] == "fullband"
    assert res_sub.nnz_total > 0
    assert res_sub.m_total > 0

    if is_cuda_available():
        from fastducc.gpu_nufft import gpu_store_to_host_ragged

        res_sub_gpu = stream_ms_to_ragged_csr(
            ms_path,
            npix_x=64,
            npix_y=64,
            pixsize_arcsec=60.0,
            dm=80.0,
            dedisp_mode="subbanded",
            nsubbands=8,
            mode="wgridder",
            do_wgridding=False,
            min_valid_channels=4,
            uv_threshold=1e-6,
            device="cuda",
        )
        assert res_sub_gpu.attrs["dedisp_mode"] == "subbanded"
        assert res_sub_gpu.m_total == res_sub.m_total
        assert res_sub_gpu.nnz_total == res_sub.nnz_total
        host_from_gpu = gpu_store_to_host_ragged(res_sub_gpu.gpu_store)
        assert np.array_equal(host_from_gpu.cell_uv, res_sub.store.cell_uv)
        assert np.array_equal(host_from_gpu.cell_offsets, res_sub.store.cell_offsets)
        assert _rel_l2_error(host_from_gpu.vis_data, res_sub.store.vis_data) < 1e-5
