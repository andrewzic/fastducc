# -*- coding: utf-8 -*-
"""
Unit tests for GPU/CPU direct half-plane rasterisation and C2R imaging in
fastducc transient searches (fastducc.gpu_c2r and fastducc.gpu_transient).
"""

from __future__ import annotations

import numpy as np
import pytest

from fastducc import fastducc_run, gpu_c2r, gpu_raster, gpu_transient, imaging
from fastducc.constants import C_LIGHT, K_DM


def _synthesise_transient_chunk(
    *,
    n_ant: int = 8,
    n_chan: int = 24,
    n_time: int = 32,
    dt: float = 0.01,
    pulse_dm: float = 0.0,
    pulse_t0_idx: int = 10,
    pulse_flux_jy: float = 5.0,
    source_l_deg: float = 0.15,
    source_m_deg: float = -0.10,
    seed: int = 42,
):
    """Synthesise a clean visibility chunk with a dispersed point-source pulse."""
    rng = np.random.default_rng(seed)
    ant_xyz = rng.uniform(-800.0, 800.0, size=(n_ant, 3))
    ant_xyz[:, 2] *= 0.05  # nearly coplanar for 2D wgridder reference comparison

    bl_pairs = [(i, j) for i in range(n_ant) for j in range(i + 1, n_ant)]
    nbl = len(bl_pairs)
    freqs_hz = np.linspace(0.80e9, 1.08e9, n_chan, dtype=np.float64)

    uvw_3d = np.zeros((nbl, 3, n_time), dtype=np.float64)
    for b, (i, j) in enumerate(bl_pairs):
        base = ant_xyz[j] - ant_xyz[i]
        # Slight Earth-rotation angle across the chunk
        angles = np.linspace(0.0, 0.02, n_time)
        ca, sa = np.cos(angles), np.sin(angles)
        uvw_3d[b, 0, :] = base[0] * ca - base[1] * sa
        uvw_3d[b, 1, :] = base[0] * sa + base[1] * ca
        uvw_3d[b, 2, :] = base[2]

    l_rad = np.deg2rad(source_l_deg)
    m_rad = np.deg2rad(source_m_deg)
    n_m1 = np.sqrt(max(0.0, 1.0 - l_rad * l_rad - m_rad * m_rad)) - 1.0

    # Causal dispersion delays relative to highest frequency
    freqs_mhz = freqs_hz * 1e-6
    fmax_mhz = float(np.max(freqs_mhz))
    delays_s = K_DM * pulse_dm * (1.0 / (freqs_mhz ** 2) - 1.0 / (fmax_mhz ** 2))
    delays_samp = np.round(delays_s / dt).astype(np.int64)

    vis_3d = np.zeros((nbl, n_chan, n_time), dtype=np.complex64)
    wgt_3d = np.ones((nbl, n_chan, n_time), dtype=np.float32)

    for c in range(n_chan):
        t_arr = pulse_t0_idx + int(delays_samp[c])
        if 0 <= t_arr < n_time:
            lam = C_LIGHT / freqs_hz[c]
            phase = -2.0j * np.pi * (
                uvw_3d[:, 0, t_arr] * l_rad
                + uvw_3d[:, 1, t_arr] * m_rad
                + uvw_3d[:, 2, t_arr] * n_m1
            ) / lam
            vis_3d[:, c, t_arr] = (pulse_flux_jy * np.exp(phase)).astype(np.complex64)

    return vis_3d, wgt_3d, uvw_3d, freqs_hz


def test_direct_raster_c2r_matches_vis2dirty_off_centre():
    """
    Verify direct half-plane wgridder rasterisation + C2R imaging with DUCC
    deapodisation matches ducc0.wgridder.vis2dirty for an off-centre point source.
    """
    npix = 64
    pix_arcsec = 60.0
    pix_rad = np.deg2rad(pix_arcsec / 3600.0)
    dt = 0.01

    vis_3d, wgt_3d, uvw_3d, freqs_hz = _synthesise_transient_chunk(
        n_ant=8,
        n_chan=16,
        n_time=12,
        dt=dt,
        pulse_dm=0.0,
        pulse_t0_idx=5,
        pulse_flux_jy=4.0,
        source_l_deg=0.18,
        source_m_deg=-0.12,
    )

    ref_cube = imaging.grid_chunk_cube(
        vis_3d=vis_3d,
        wgt_3d=wgt_3d,
        uvw_3d=uvw_3d,
        chan_freq=freqs_hz,
        dt=dt,
        dm=0.0,
        npix_x=npix,
        npix_y=npix,
        pixsize_x=pix_rad,
        pixsize_y=pix_rad,
        epsilon=1e-5,
        do_wgridding=False,
        nthreads=1,
        min_valid_channels=1,
    )

    results = list(
        gpu_transient.iter_chunk_dirty_cubes_direct_raster(
            vis_3d=vis_3d,
            wgt_3d=wgt_3d,
            uvw_3d=uvw_3d,
            chan_freq=freqs_hz,
            dt=dt,
            dm_trials=[0.0],
            npix_x=npix,
            npix_y=npix,
            pixsize_x=pix_rad,
            pixsize_y=pix_rad,
            dedisp_mode="fullband",
            raster_mode="wgridder",
            epsilon=1e-5,
            do_wgridding=False,
            min_valid_channels=1,
            device="cpu",
            nthreads=1,
        )
    )
    assert len(results) == 1
    dm_out, direct_cube = results[0]
    assert dm_out == 0.0
    assert direct_cube.shape == (12, npix, npix)

    ref_frame = ref_cube[5].astype(np.float64)
    dir_frame = direct_cube[5].astype(np.float64)

    ref_peak = np.unravel_index(int(np.argmax(ref_frame)), ref_frame.shape)
    dir_peak = np.unravel_index(int(np.argmax(dir_frame)), dir_frame.shape)
    assert dir_peak == ref_peak

    rel_l2 = np.linalg.norm(dir_frame - ref_frame) / max(1e-12, np.linalg.norm(ref_frame))
    assert rel_l2 < 1e-3


def test_subbanded_multi_dm_transient_recovery_and_dm0_first():
    """
    Verify two-tier subbanded active-cell gridding + C2R imaging yields DM=0.0
    first and maximises pulse peak flux at the injected dispersion measure.
    """
    npix = 64
    pix_arcsec = 60.0
    pix_rad = np.deg2rad(pix_arcsec / 3600.0)
    dt = 0.01
    true_dm = 40.0

    vis_3d, wgt_3d, uvw_3d, freqs_hz = _synthesise_transient_chunk(
        n_ant=8,
        n_chan=24,
        n_time=48,
        dt=dt,
        pulse_dm=true_dm,
        pulse_t0_idx=8,
        pulse_flux_jy=6.0,
        source_l_deg=-0.10,
        source_m_deg=0.14,
        seed=123,
    )

    dm_trials = [true_dm, 0.0, 2.0 * true_dm]
    cubes_by_dm = {}
    yielded_order = []
    for dm_val, cube in gpu_transient.iter_chunk_dirty_cubes_direct_raster(
        vis_3d=vis_3d,
        wgt_3d=wgt_3d,
        uvw_3d=uvw_3d,
        chan_freq=freqs_hz,
        dt=dt,
        dm_trials=dm_trials,
        npix_x=npix,
        npix_y=npix,
        pixsize_x=pix_rad,
        pixsize_y=pix_rad,
        dedisp_mode="subbanded",
        nsubbands=6,
        raster_mode="wgridder",
        epsilon=1e-5,
        do_wgridding=False,
        min_valid_channels=1,
        pad_radius=6,
        device="cpu",
        nthreads=1,
    ):
        yielded_order.append(dm_val)
        cubes_by_dm[dm_val] = cube

    assert yielded_order[0] == 0.0
    assert set(yielded_order) == set(dm_trials)

    peak_dm0 = float(np.max(cubes_by_dm[0.0]))
    peak_true = float(np.max(cubes_by_dm[true_dm]))
    peak_high = float(np.max(cubes_by_dm[2.0 * true_dm]))

    assert peak_true > peak_dm0 * 2.0
    assert peak_true > peak_high * 1.5

    t_peak, _, _ = np.unravel_index(int(np.argmax(cubes_by_dm[true_dm])), cubes_by_dm[true_dm].shape)
    assert abs(int(t_peak) - 8) <= 1


def test_sparse_and_dense_c2r_cpu_exact_parity():
    """Verify sparse active-cell C2R imaging matches dense half-plane C2R imaging."""
    rng = np.random.default_rng(7)
    nt, nv, grid_nu = 6, 48, 48
    u_half = grid_nu // 2 + 1
    ny, nx = 32, 32

    v_coords = rng.integers(0, nv, size=40, dtype=np.int32)
    u_coords = rng.integers(0, u_half, size=40, dtype=np.int32)
    cell_uv = np.unique(np.column_stack([v_coords, u_coords]), axis=0).astype(np.int32)
    m_active = cell_uv.shape[0]

    vis_active = (
        rng.standard_normal((m_active, nt)) + 1j * rng.standard_normal((m_active, nt))
    ).astype(np.complex64)

    dense_grid = np.zeros((nt, nv, u_half), dtype=np.complex64)
    dense_grid[:, cell_uv[:, 0], cell_uv[:, 1]] = vis_active.T

    cfu, cfv = gpu_c2r.compute_deapod_vectors(nx, ny, grid_nu, nv, raster_mode="wgridder", epsilon=1e-5)
    img_dense = gpu_c2r.halfplane_grid_to_dirty_cube_cpu(
        dense_grid, grid_nu=grid_nu, npix_y=ny, npix_x=nx, corfunc_u=cfu, corfunc_v=cfv
    )
    img_sparse = gpu_c2r.sparse_active_to_dirty_cube_cpu(
        vis_active, cell_uv, nv=nv, grid_nu=grid_nu, npix_y=ny, npix_x=nx, corfunc_u=cfu, corfunc_v=cfv
    )

    np.testing.assert_allclose(img_sparse, img_dense, rtol=1e-5, atol=1e-6)


def test_cli_gpu_transient_flags_and_auto_engine_promotion(monkeypatch):
    """Verify new GPU/direct_raster CLI flags parse cleanly and promote engine on CUDA."""
    args = fastducc_run.build_cli([
        "--msname", "dummy.ms",
        "--device", "cuda",
        "--raster-mode", "wgridder",
        "--dedisp-mode", "subbanded",
        "--nsubbands", "16",
        "--pad-radius", "8",
        "--gpu-c2r-batch-size", "128",
        "--min-uv-distance-m", "200.0",
        "--raster-uv-threshold", "6e-6",
    ])
    assert args.device == "cuda"
    assert args.engine == "wgridder"
    assert args.raster_mode == "wgridder"
    assert args.pad_radius == 8
    assert args.c2r_batch_size == 128
    assert args.min_uv_distance_m == 200.0
    assert args.raster_uv_threshold == pytest.approx(6e-6)

    monkeypatch.setattr(
        fastducc_run.ms_utils,
        "get_phase_center",
        lambda msname, field_name=None: (0.0, 0.0, "FIELD0"),
    )
    cfg = fastducc_run.make_config(args, ("dummy", "/tmp/cands", "/tmp/chunk", "/tmp/all"))
    assert cfg.device == "cuda"
    assert cfg.engine == "direct_raster"
    assert cfg.raster_mode == "wgridder"
    assert cfg.pad_radius == 8
    assert cfg.c2r_batch_size == 128
    assert cfg.min_uv_distance_m == 200.0
    assert cfg.raster_uv_threshold == pytest.approx(6e-6)


@pytest.mark.skipif(not gpu_raster.is_cuda_available(), reason="CUDA device unavailable")
def test_gpu_vs_cpu_direct_raster_transient_parity():
    """Verify CUDA GPU subbanded and fullband C2R transient imaging matches CPU."""
    npix = 64
    pix_rad = np.deg2rad(60.0 / 3600.0)
    dt = 0.01
    vis_3d, wgt_3d, uvw_3d, freqs_hz = _synthesise_transient_chunk(
        n_ant=8,
        n_chan=16,
        n_time=24,
        dt=dt,
        pulse_dm=25.0,
        pulse_t0_idx=6,
        pulse_flux_jy=5.0,
    )

    for dedisp_mode in ("subbanded", "fullband"):
        cpu_out = dict(
            gpu_transient.iter_chunk_dirty_cubes_direct_raster(
                vis_3d, wgt_3d, uvw_3d, freqs_hz, dt, [0.0, 25.0],
                npix_x=npix, npix_y=npix, pixsize_x=pix_rad, pixsize_y=pix_rad,
                dedisp_mode=dedisp_mode, nsubbands=4, raster_mode="wgridder",
                min_valid_channels=1, device="cpu",
            )
        )
        gpu_out = dict(
            gpu_transient.iter_chunk_dirty_cubes_direct_raster(
                vis_3d, wgt_3d, uvw_3d, freqs_hz, dt, [0.0, 25.0],
                npix_x=npix, npix_y=npix, pixsize_x=pix_rad, pixsize_y=pix_rad,
                dedisp_mode=dedisp_mode, nsubbands=4, raster_mode="wgridder",
                min_valid_channels=1, device="cuda",
            )
        )
        for dm_val in (0.0, 25.0):
            np.testing.assert_allclose(gpu_out[dm_val], cpu_out[dm_val], rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not gpu_raster.is_cuda_available(), reason="CUDA device unavailable")
def test_gpu_welford_and_boxcar_detection_parity():
    """Verify GPU Welford variance, local noise thresholding, and boxcar NMS match CPU."""
    import cupy as cp
    from fastducc import detection, filters, gpu_detection, kernels

    rng = np.random.default_rng(99)
    T, Ny, Nx = 64, 96, 96
    times = np.arange(T, dtype=np.float64) * 0.11
    cube = rng.standard_normal((T, Ny, Nx)).astype(np.float32)

    # Inject a bright 4-sample transient pulse at (y=40, x=55, t=20..23)
    cube[20:24, 40, 55] += 12.0

    # 1. Welford parity (with EMA high-pass filter)
    d_cube = cp.asarray(cube)
    c_gpu, m_gpu, M2_gpu, std_gpu = gpu_detection.welford_update_cube_gpu(
        d_cube, times, var_highpass_cutoff_sec=30.0
    )

    dt_med = float(np.median(np.diff(times)))
    alphas = np.array([1.0 - np.exp(-dt_med / 30.0)] * T, dtype=np.float64)
    ema_cpu = np.nanmean(cube, axis=0, dtype=np.float64)
    c_cpu = np.zeros((Ny, Nx), dtype=np.int64)
    m_cpu = np.zeros((Ny, Nx), dtype=np.float64)
    M2_cpu = np.zeros((Ny, Nx), dtype=np.float64)
    kernels.welford_update_cube(c_cpu, m_cpu, M2_cpu, ema_cpu, cube, alphas, do_highpass=True, ignore_nan=True)
    std_cpu = kernels.welford_finalise_std(c_cpu, M2_cpu, ddof=1)

    np.testing.assert_array_equal(c_gpu, c_cpu)
    np.testing.assert_allclose(m_gpu, m_cpu, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(M2_gpu, M2_cpu, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(std_gpu, std_cpu, rtol=1e-5, atol=1e-5)

    # 2. Variance local-threshold search parity
    var_gpu, snr_gpu = gpu_detection.variance_search_welford_gpu(
        std_gpu, threshold_sigma=6.0, use_local_threshold=True, local_window_size=32
    )
    var_cpu, snr_cpu = detection.variance_search_welford(
        std_cpu, threshold_sigma=6.0, return_snr_image=True, use_local_threshold=True, local_window_size=32
    )
    np.testing.assert_allclose(snr_gpu, snr_cpu, rtol=1e-4, atol=1e-4)
    assert len(var_gpu) == len(var_cpu) and len(var_gpu) >= 1
    assert (var_gpu[0]["y"], var_gpu[0]["x"]) == (40, 55)

    # 3. Boxcar cumsum + local threshold + NMS parity
    dets_gpu_by_w = gpu_detection.boxcar_search_and_nms_gpu(
        times, d_cube, widths=[1, 2, 4, 8], threshold_sigma=7.0,
        spatial_radius=4, time_radius=2, use_local_threshold=True, local_window_size=32,
    )
    final_gpu = filters.group_filter_across_widths(
        dets_gpu_by_w, times, spatial_radius=4, time_radius=8, ny_nx=(Ny, Nx), cube=cube
    )
    assert len(final_gpu) >= 1
    top = final_gpu[0]
    assert (top["y"], top["x"]) == (40, 55)
    assert top["width_samples"] == 4
    assert top["snr"] > 15.0
