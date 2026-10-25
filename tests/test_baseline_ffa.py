# -*- coding: utf-8 -*-
"""
Tests for baseline-domain FFA search and ducc0.wgridder integration.
"""

import os
import pytest
import numpy as np

from fastducc.baseline_ffa import (
    prepare_baseline_series,
    shift_profile_phase_fourier,
    fold_baselines_option_b,
    fold_baselines_option_a,
    image_phase_bins_wgridder,
    evaluate_period_snr,
    search_baseline_ffa,
)


@pytest.fixture
def synthetic_vis_data():
    """Create a small synthetic visibility dataset with an injected periodic pulse."""
    nbl = 8
    nchan = 4
    nt = 300
    dt = 0.05
    period = 1.05
    chan_freq = np.linspace(1.2e9, 1.4e9, nchan)
    c_light = 299792458.0

    t = np.arange(nt) * dt
    phase = (t / period) % 1.0
    pulse = np.exp(-0.5 * ((phase - 0.5) / 0.03)**2).astype(np.float32)

    vis_3d = np.zeros((nbl, nchan, nt), dtype=np.complex64)
    wgt_3d = np.ones((nbl, nchan, nt), dtype=np.float32)
    uvw_3d = np.zeros((nbl, 3, nt), dtype=np.float64)

    l0, m0 = 0.001, 0.002
    for b in range(nbl):
        u_m = 40.0 * (b + 1)
        v_m = 25.0 * (b + 1)
        uvw_3d[b, 0, :] = u_m + 0.02 * np.arange(nt)
        uvw_3d[b, 1, :] = v_m + 0.01 * np.arange(nt)
        uvw_3d[b, 2, :] = 0.0
        for c in range(nchan):
            lam = c_light / chan_freq[c]
            sp_phase = 2.0 * np.pi * (uvw_3d[b, 0, :] * l0 + uvw_3d[b, 1, :] * m0) / lam
            vis_3d[b, c, :] = 4.0 * pulse * np.exp(1j * sp_phase) + 0.02 * (np.random.randn(nt) + 1j * np.random.randn(nt))

    return vis_3d, wgt_3d, uvw_3d, chan_freq, dt, period


def test_prepare_baseline_series(synthetic_vis_data):
    vis_3d, wgt_3d, uvw_3d, chan_freq, dt, _ = synthetic_vis_data
    nbl, nchan, nt = vis_3d.shape

    # 1. Collapsed mode
    vis_2d, wgt_2d, uvw_track, eff_freq, _ = prepare_baseline_series(
        vis_3d, wgt_3d, uvw_3d, chan_freq, dt, collapse_channels=True
    )
    assert vis_2d.shape == (nbl, nt)
    assert wgt_2d.shape == (nbl, nt)
    assert uvw_track.shape == (nbl, 3, nt)

    # 2. Pseudo-baseline mode
    vis_pseudo, wgt_pseudo, uvw_pseudo, eff_freq_p, _ = prepare_baseline_series(
        vis_3d, wgt_3d, uvw_3d, chan_freq, dt, collapse_channels=False
    )
    assert vis_pseudo.shape == (nbl * nchan, nt)
    assert uvw_pseudo.shape == (nbl * nchan, 3, nt)

    # 3. Subband mode (Option 2)
    nsub = 2
    vis_sub, wgt_sub, uvw_sub, eff_freq_sub, _ = prepare_baseline_series(
        vis_3d, wgt_3d, uvw_3d, chan_freq, dt, collapse_channels=True, nsubbands=nsub, dm=10.0, exact_uvw=True
    )
    assert vis_sub.shape == (nbl * nsub, nt)
    assert wgt_sub.shape == (nbl * nsub, nt)
    assert uvw_sub.shape == (nbl * nsub, 3, nt)
    assert len(eff_freq_sub) == 1  # [c_light] for dimensionless wavelengths


def test_fourier_phase_shift():
    nbins = 32
    nperiods = 4
    # Profile with impulse at bin 10
    prof = np.zeros((nperiods, nbins), dtype=np.complex64)
    prof[:, 10] = 1.0

    delta_phase = np.array([0.0, 0.25, 0.5, 0.75])
    shifted = shift_profile_phase_fourier(prof, delta_phase)

    # Shift by 0.25 should shift peak by 0.25 * 32 = 8 bins -> bin 18
    peak_0 = np.argmax(np.abs(shifted[0]))
    peak_1 = np.argmax(np.abs(shifted[1]))
    peak_2 = np.argmax(np.abs(shifted[2]))

    assert peak_0 == 10
    assert peak_1 == 18
    assert peak_2 == 26

    # 3D test
    M = 5
    prof_3d = np.repeat(prof[None, :, :], M, axis=0)  # (M, nperiods, nbins)
    shifted_3d = shift_profile_phase_fourier(prof_3d, delta_phase)
    assert shifted_3d.shape == (M, nperiods, nbins)
    for m in range(M):
        assert np.argmax(np.abs(shifted_3d[m, 0])) == 10
        assert np.argmax(np.abs(shifted_3d[m, 1])) == 18


def test_fold_options(synthetic_vis_data):
    vis_3d, wgt_3d, uvw_3d, chan_freq, dt, _ = synthetic_vis_data
    vis_2d, wgt_2d, uvw_track, eff_freq, _ = prepare_baseline_series(
        vis_3d, wgt_3d, uvw_3d, chan_freq, dt, collapse_channels=True
    )

    # Option B
    uvw_b, wgt_b, folded_b, periods_b = fold_baselines_option_b(
        vis_2d, wgt_2d, uvw_track, base_period_bins=16, tsamp=dt
    )
    assert uvw_b.shape == (vis_2d.shape[0], 3)
    assert folded_b.shape[0] == vis_2d.shape[0]
    assert folded_b.shape[2] == 16
    assert len(periods_b) > 0

    # Option A (segmented)
    uvw_a, wgt_a, folded_a, periods_a = fold_baselines_option_a(
        vis_2d, wgt_2d, uvw_track, base_period_bins=16, tsamp=dt, segment_duration_s=5.0
    )
    assert uvw_a.shape[1] == 3
    assert folded_a.shape[2] == 16
    assert len(periods_a) > 0


def test_imaging_and_snr(synthetic_vis_data):
    vis_3d, wgt_3d, uvw_3d, chan_freq, dt, _ = synthetic_vis_data
    vis_2d, wgt_2d, uvw_track, eff_freq, _ = prepare_baseline_series(
        vis_3d, wgt_3d, uvw_3d, chan_freq, dt, collapse_channels=True
    )
    uvw_b, wgt_b, folded_b, periods_b = fold_baselines_option_b(
        vis_2d, wgt_2d, uvw_track, base_period_bins=16, tsamp=dt
    )

    phase_imgs = image_phase_bins_wgridder(
        uvw=uvw_b, freq=eff_freq, vis_period=folded_b[:, 0, :], wgt=wgt_b,
        npix_x=32, npix_y=32, pixsize_x=2e-4, pixsize_y=2e-4
    )
    assert phase_imgs.shape == (16, 32, 32)

    snr_plane, best_snr, cand = evaluate_period_snr(phase_imgs, periods_b[0], snr_thresh=3.0)
    assert snr_plane.shape == (32, 32)
    assert isinstance(best_snr, float)


def test_search_baseline_ffa_end_to_end(synthetic_vis_data):
    vis_3d, wgt_3d, uvw_3d, chan_freq, dt, true_period = synthetic_vis_data

    res = search_baseline_ffa(
        (vis_3d, wgt_3d, uvw_3d, chan_freq, dt),
        period_min=1.0,
        period_max=1.1,
        bins_min=16,
        bins_max=20,
        npix_x=32,
        npix_y=32,
        pixsize_x=2e-4,
        pixsize_y=2e-4,
        snr_thresh=4.0
    )
    assert len(res.periods) > 0
    assert len(res.snr_periodogram) == len(res.periods)
    assert res.best_snr_plane.shape == (32, 32)


def test_dispersed_offaxis_recovery():
    """Verify that off-axis dispersed pulsar is recovered with exact UVW tracking."""
    nbl = 12
    nchan = 8
    nt = 300
    dt = 0.05
    period = 1.05
    dm = 30.0
    chan_freq = np.linspace(1.1e9, 1.4e9, nchan)
    c_light = 299792458.0

    # Dispersion delays
    from fastducc.dedisp import get_dm_delays
    delays = get_dm_delays(chan_freq, dm, dt, align_to="fmax")

    vis_3d = np.zeros((nbl, nchan, nt), dtype=np.complex64)
    wgt_3d = np.ones((nbl, nchan, nt), dtype=np.float32)
    uvw_3d = np.zeros((nbl, 3, nt), dtype=np.float64)

    # Place source off-axis: l0 = 0.003 rad (~10 arcmin), m0 = 0.002 rad
    l0, m0 = 0.003, 0.002
    for b in range(nbl):
        u_m = 80.0 * (b + 1)
        v_m = 60.0 * (b + 1)
        uvw_3d[b, 0, :] = u_m + 0.05 * np.arange(nt)
        uvw_3d[b, 1, :] = v_m + 0.03 * np.arange(nt)
        uvw_3d[b, 2, :] = 0.0

    for c in range(nchan):
        lam = c_light / chan_freq[c]
        d_c = delays[c]
        # Emit time: t_emit = t_obs - d_c * dt
        t_obs = np.arange(nt) * dt
        t_emit = t_obs - d_c * dt
        phase = (t_emit / period) % 1.0
        pulse = np.exp(-0.5 * ((phase - 0.5) / 0.03)**2).astype(np.float32)

        for b in range(nbl):
            sp_phase = 2.0 * np.pi * (uvw_3d[b, 0, :] * l0 + uvw_3d[b, 1, :] * m0) / lam
            vis_3d[b, c, :] = 6.0 * pulse * np.exp(-1j * sp_phase)

    # 1. Search with exact UVW pseudo-baselines (Option 1)
    res_exact = search_baseline_ffa(
        (vis_3d, wgt_3d, uvw_3d, chan_freq, dt),
        period_min=1.0,
        period_max=1.1,
        bins_min=16,
        bins_max=20,
        dm=dm,
        collapse_channels=False,
        exact_uvw=True,
        segment_duration_s=5.0,
        npix_x=48,
        npix_y=48,
        pixsize_x=2e-4,
        pixsize_y=2e-4,
        snr_thresh=4.0
    )

    # 2. Search with collapsed channels (Option B / collapsed)
    res_collapsed = search_baseline_ffa(
        (vis_3d, wgt_3d, uvw_3d, chan_freq, dt),
        period_min=1.0,
        period_max=1.1,
        bins_min=16,
        bins_max=20,
        dm=dm,
        collapse_channels=True,
        segment_duration_s=5.0,
        npix_x=48,
        npix_y=48,
        pixsize_x=2e-4,
        pixsize_y=2e-4,
        snr_thresh=4.0
    )

    peak_exact = np.max(res_exact.snr_periodogram)
    peak_collapsed = np.max(res_collapsed.snr_periodogram)

    # Exact tracking preserves candidate coherence without bandwidth smearing
    assert peak_exact > 8.0, f"Expected high SNR with exact UVW, got {peak_exact}"
    assert peak_exact > 1.25 * peak_collapsed, f"Exact SNR ({peak_exact}) should significantly exceed collapsed ({peak_collapsed})"
