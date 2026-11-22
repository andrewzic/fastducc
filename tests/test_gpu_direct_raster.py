# -*- coding: utf-8 -*-
"""
Unit and numerical parity tests for fastducc GPU direct-to-grid Hermitian
half-plane rasterisation, double-buffered MS streaming, and zero-copy GPU CSR
bridging (:mod:`fastducc.gpu_raster`, :mod:`fastducc.gpu_direct_cube`,
and :mod:`fastducc.gpu_nufft`).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
import numpy as np
import pytest

from fastducc.constants import C_LIGHT
from fastducc.dedisp import BruteForceDedisperser, DedispersionPlan, get_dm_delays
from fastducc.gpu_direct_cube import stream_ms_to_ragged_csr
from fastducc.gpu_nufft import (
    build_gpu_csr_matrix,
    from_gpu_buffers,
    gpu_store_to_host_ragged,
)
from fastducc.gpu_raster import (
    build_uv_to_m_lookup,
    compute_raster_grid_geometry,
    compute_subband_slices,
    dedisperse_and_rasterise_chunk_cpu,
    dedisperse_and_rasterise_chunk_gpu,
    dedisperse_rasterise_to_ragged,
    is_cuda_available,
    stage_ms_rows_cpu,
    stage_ms_rows_gpu,
)
from fastducc.ragged import RaggedDGAStore

try:
    import cupy as cp
    import cupyx.scipy.sparse as cupy_sparse
except ImportError:
    cp = None  # type: ignore[assignment]
    cupy_sparse = None  # type: ignore[assignment]

try:
    from casacore.tables import (
        makearrcoldesc,
        makescacoldesc,
         maketabdesc,
        table,
    )
    HAS_CASACORE = True
except ImportError:
    HAS_CASACORE = False


def _rel_l2_error(a: np.ndarray, b: np.ndarray) -> float:
    """Compute relative L2 error ||a - b||_2 / max(||b||_2, 1e-12)."""
    diff = np.asarray(a, dtype=np.complex128) - np.asarray(b, dtype=np.complex128)
    ref = np.asarray(b, dtype=np.complex128)
    num = float(np.linalg.norm(diff.ravel()))
    den = max(float(np.linalg.norm(ref.ravel())), 1e-12)
    return num / den


def _analytical_reference_raster(
    vis_3d: np.ndarray,
    wgt_3d: np.ndarray,
    uvw_3d: np.ndarray,
    freqs_hz: np.ndarray,
    dt: float,
    dm: float,
    nt_out: int,
    *,
    collapse_channels: bool,
    nsubbands: int,
    exact_uvw: bool,
    npix_x: int,
    npix_y: int,
    pixsize_arcsec: float,
    mode: str,
    min_valid_channels: int,
    min_uv_distance_m: Optional[float],
    uv_threshold: float,
) -> np.ndarray:
    """
    Independent analytical reference combining :class:`BruteForceDedisperser`
    channel shifting with explicit baseline-by-baseline Hermitian half-plane
    cell assignment and per-cell weight normalisation.
    """
    nbl, nchan, nt_in = vis_3d.shape
    geom = compute_raster_grid_geometry(
        npix_x=npix_x, npix_y=npix_y, pixsize_arcsec=pixsize_arcsec
    )
    nv = int(geom["nv"])
    u_half = int(geom["u_half"])
    du = float(geom["du"])
    dv = float(geom["dv"])

    min_uv = 0.0 if min_uv_distance_m is None else float(min_uv_distance_m)
    valid_mask = (
        (wgt_3d > 0.0)
        & np.isfinite(wgt_3d)
        & np.isfinite(vis_3d.real)
        & np.isfinite(vis_3d.imag)
    )
    if min_uv > 0.0:
        uv_dist = np.hypot(uvw_3d[:, 0, :], uvw_3d[:, 1, :])
        valid_mask &= (
            np.isfinite(uvw_3d[:, 0, :])
            & np.isfinite(uvw_3d[:, 1, :])
            & (uv_dist >= min_uv)
        )[:, None, :]

    vis_clean = np.where(valid_mask, vis_3d, np.complex64(0.0 + 0.0j))
    wgt_clean = np.where(valid_mask, wgt_3d, np.float32(0.0))

    raw_valid_chans = np.count_nonzero(
        np.any(wgt_clean[:, :, :nt_out] > 0.0, axis=0), axis=0
    )

    plan = DedispersionPlan(freqs_hz, dt, dm_min=dm, dm_max=dm)
    dedisperser = BruteForceDedisperser(plan, dm, align_to="fmax")
    delays = dedisperser.delays.astype(np.int32)

    # Shift uncollapsed channels first to count dedispersed valid channels
    vis_shifted, wgt_shifted, _ = dedisperser.dedisperse_chunk(
        vis_clean, wgt_clean, collapse_channels=False, nsubbands=1, use_history=False
    )
    vis_shifted = vis_shifted[:, :, :nt_out]
    wgt_shifted = wgt_shifted[:, :, :nt_out]
    dedisp_valid_chans = np.count_nonzero(
        np.any(wgt_shifted > 0.0, axis=0), axis=0
    )

    req_chans = min(min_valid_channels, nchan) if min_valid_channels > 0 else 1
    sub_offsets, nsub_eff = compute_subband_slices(
        nchan=nchan, collapse_channels=collapse_channels, nsubbands=nsubbands
    )
    is_full_collapse = bool(collapse_channels and nsub_eff == 1)

    out_grid = np.zeros((nt_out, nv, u_half), dtype=np.complex64)

    for t in range(nt_out):
        if collapse_channels:
            if raw_valid_chans[t] < req_chans or dedisp_valid_chans[t] < 1:
                continue
        else:
            if dedisp_valid_chans[t] < req_chans:
                continue

        acc_re = np.zeros((nv, u_half), dtype=np.float32)
        acc_im = np.zeros((nv, u_half), dtype=np.float32)
        acc_w = np.zeros((nv, u_half), dtype=np.float32)

        for s in range(nsub_eff):
            c0 = int(sub_offsets[s])
            c1 = int(sub_offsets[s + 1])
            f_sub = float(np.mean(freqs_hz[c0:c1]))
            d_sub = int(delays[c0]) if (c1 - c0 == 1) else int(np.rint(np.mean(delays[c0:c1])))

            t_uvw = t if (is_full_collapse or not exact_uvw) else min(max(0, t + d_sub), nt_in - 1)

            for bl in range(nbl):
                w_sub_vec = wgt_shifted[bl, c0:c1, t].astype(np.float64)
                v_sub_vec = vis_shifted[bl, c0:c1, t].astype(np.complex128)
                sum_w = float(np.sum(w_sub_vec))
                if sum_w <= 0.0:
                    continue
                sum_re = np.float32(np.sum(v_sub_vec.real * w_sub_vec))
                sum_im = np.float32(np.sum(v_sub_vec.imag * w_sub_vec))
                w_f32 = np.float32(sum_w)

                u_m = float(uvw_3d[bl, 0, t_uvw])
                v_m = float(uvw_3d[bl, 1, t_uvw])
                u_f = (u_m * (f_sub / C_LIGHT)) / du
                v_f = (v_m * (f_sub / C_LIGHT)) / dv

                if mode == "nearest":
                    u_cell = int(np.rint(u_f))
                    v_cell = int(np.rint(v_f))
                    im_val = sum_im
                    if u_cell < 0:
                        u_cell = -u_cell
                        v_cell = -v_cell
                        im_val = np.float32(-im_val)
                    if 0 <= u_cell < u_half:
                        v_idx = v_cell % nv
                        acc_re[v_idx, u_cell] = np.float32(acc_re[v_idx, u_cell] + sum_re)
                        acc_im[v_idx, u_cell] = np.float32(acc_im[v_idx, u_cell] + im_val)
                        acc_w[v_idx, u_cell] = np.float32(acc_w[v_idx, u_cell] + w_f32)
                else:
                    uf = u_f
                    vf = v_f
                    im_val = sum_im
                    if uf < 0.0:
                        uf = -uf
                        vf = -vf
                        im_val = np.float32(-im_val)
                    u0 = int(np.floor(uf))
                    v0 = int(np.floor(vf))
                    if 0 <= u0 < u_half:
                        au = np.float32(uf - float(u0))
                        av = np.float32(vf - float(v0))
                        c00 = np.float32((np.float32(1.0) - au) * (np.float32(1.0) - av))
                        c10 = np.float32(au * (np.float32(1.0) - av))
                        c01 = np.float32((np.float32(1.0) - au) * av)
                        c11 = np.float32(au * av)
                        v0w = v0 % nv
                        v1w = (v0w + 1) % nv
                        for cu, cv, cw in (
                            (u0, v0w, c00),
                            (u0 + 1, v0w, c10),
                            (u0, v1w, c01),
                            (u0 + 1, v1w, c11),
                        ):
                            if 0 <= cu < u_half and cw > 0.0:
                                acc_re[cv, cu] = np.float32(acc_re[cv, cu] + np.float32(sum_re * cw))
                                acc_im[cv, cu] = np.float32(acc_im[cv, cu] + np.float32(im_val * cw))
                                acc_w[cv, cu] = np.float32(acc_w[cv, cu] + np.float32(w_f32 * cw))

        pos = acc_w > 0.0
        inv_w = np.where(pos, np.float32(1.0) / np.where(pos, acc_w, np.float32(1.0)), np.float32(0.0))
        nr = acc_re * inv_w
        ni = acc_im * inv_w
        if uv_threshold > 0.0:
            thr_sq = np.float32(uv_threshold * uv_threshold)
            keep = pos & ((nr * nr + ni * ni) > thr_sq)
        else:
            keep = pos
        out_grid[t].real[keep] = nr[keep]
        out_grid[t].imag[keep] = ni[keep]

    return out_grid


def _simulate_288chan_chunk(seed: int = 12345):
    """
    Generate a realistic 288-channel synthetic visibility dataset with both
    positive and negative u baselines, short baselines below min_uv_distance_m,
    varying weights, flagged channels, and one snapshot below min_valid_channels.
    """
    rng = np.random.default_rng(seed)
    nbl = 14
    nchan = 288
    nt_in = 40
    nt_out = 12
    dt = 0.013824
    dm = 125.0
    freqs_hz = np.linspace(856.0e6, 1144.0e6, nchan, dtype=np.float64)

    vis_3d = (
        rng.standard_normal((nbl, nchan, nt_in))
        + 1j * rng.standard_normal((nbl, nchan, nt_in))
    ).astype(np.complex64)
    wgt_3d = rng.uniform(0.25, 2.5, size=(nbl, nchan, nt_in)).astype(np.float32)

    # Flag ~15% of channels randomly
    flag_mask = rng.random((nbl, nchan, nt_in)) < 0.15
    wgt_3d[flag_mask] = 0.0

    # Make snapshot t=4 have only 10 valid channels (< min_valid_channels=24)
    wgt_3d[:, 10:, 4] = 0.0

    # Construct baselines spanning positive u, negative u, and short baselines (< 150 m)
    uvw_3d = np.zeros((nbl, 3, nt_in), dtype=np.float64)
    base_u = np.array(
        [-1400.0, -950.0, -620.0, -310.0, -80.0, -25.0, 40.0, 190.0, 340.0, 520.0, 780.0, 1050.0, -1210.0, 890.0],
        dtype=np.float64,
    )
    base_v = np.array(
        [600.0, -820.0, 410.0, -290.0, 50.0, -30.0, 90.0, -310.0, 640.0, -510.0, 730.0, -940.0, -450.0, 380.0],
        dtype=np.float64,
    )
    t_axis = np.arange(nt_in, dtype=np.float64)
    # Slow Earth-rotation drift across time
    uvw_3d[:, 0, :] = base_u[:, None] + 1.5 * t_axis[None, :]
    uvw_3d[:, 1, :] = base_v[:, None] - 1.2 * t_axis[None, :]
    uvw_3d[:, 2, :] = 0.2 * base_u[:, None]

    return {
        "vis_3d": vis_3d,
        "wgt_3d": wgt_3d,
        "uvw_3d": uvw_3d,
        "freqs_hz": freqs_hz,
        "dt": dt,
        "dm": dm,
        "nt_in": nt_in,
        "nt_out": nt_out,
    }


@pytest.mark.parametrize("mode", ["nearest", "bilinear"])
@pytest.mark.parametrize(
    "collapse_channels,nsubbands,exact_uvw",
    [
        (True, 1, True),
        (False, 16, True),
        (False, 16, False),
        (False, 288, True),
        (False, 288, False),
    ],
)
def test_dedisperse_and_rasterise_288chan_parity(
    mode: str,
    collapse_channels: bool,
    nsubbands: int,
    exact_uvw: bool,
):
    """
    Verify < 1e-6 relative error across analytical BruteForceDedisperser reference,
    Numba CPU, pure NumPy CPU, and CUDA GPU kernels on 288-channel DM=125 pc cm^-3 data.
    """
    sim = _simulate_288chan_chunk(seed=2026)
    npix_x, npix_y = 128, 128
    pixsize_arcsec = 22.0
    min_valid_channels = 24
    min_uv_distance_m = 150.0
    uv_threshold = 1e-4

    ref_grid = _analytical_reference_raster(
        vis_3d=sim["vis_3d"],
        wgt_3d=sim["wgt_3d"],
        uvw_3d=sim["uvw_3d"],
        freqs_hz=sim["freqs_hz"],
        dt=sim["dt"],
        dm=sim["dm"],
        nt_out=sim["nt_out"],
        collapse_channels=collapse_channels,
        nsubbands=nsubbands,
        exact_uvw=exact_uvw,
        npix_x=npix_x,
        npix_y=npix_y,
        pixsize_arcsec=pixsize_arcsec,
        mode=mode,
        min_valid_channels=min_valid_channels,
        min_uv_distance_m=min_uv_distance_m,
        uv_threshold=uv_threshold,
    )
    assert np.count_nonzero(ref_grid) > 0
    # Snapshot t=4 had only 10 raw valid channels (< 24)
    if collapse_channels:
        assert np.all(ref_grid[4] == 0.0)

    cpu_numba = dedisperse_and_rasterise_chunk_cpu(
        vis_3d=sim["vis_3d"],
        wgt_3d=sim["wgt_3d"],
        uvw_3d=sim["uvw_3d"],
        freqs_hz=sim["freqs_hz"],
        nt_out=sim["nt_out"],
        dm=sim["dm"],
        dt=sim["dt"],
        collapse_channels=collapse_channels,
        nsubbands=nsubbands,
        exact_uvw=exact_uvw,
        npix_x=npix_x,
        npix_y=npix_y,
        pixsize_arcsec=pixsize_arcsec,
        mode=mode,
        min_valid_channels=min_valid_channels,
        min_uv_distance_m=min_uv_distance_m,
        uv_threshold=uv_threshold,
        use_numba=True,
    )
    cpu_numpy = dedisperse_and_rasterise_chunk_cpu(
        vis_3d=sim["vis_3d"],
        wgt_3d=sim["wgt_3d"],
        uvw_3d=sim["uvw_3d"],
        freqs_hz=sim["freqs_hz"],
        nt_out=sim["nt_out"],
        dm=sim["dm"],
        dt=sim["dt"],
        collapse_channels=collapse_channels,
        nsubbands=nsubbands,
        exact_uvw=exact_uvw,
        npix_x=npix_x,
        npix_y=npix_y,
        pixsize_arcsec=pixsize_arcsec,
        mode=mode,
        min_valid_channels=min_valid_channels,
        min_uv_distance_m=min_uv_distance_m,
        uv_threshold=uv_threshold,
        use_numba=False,
    )

    assert _rel_l2_error(cpu_numba, ref_grid) < 1e-6
    assert _rel_l2_error(cpu_numpy, ref_grid) < 1e-6
    assert _rel_l2_error(cpu_numba, cpu_numpy) < 1e-6

    if is_cuda_available():
        gpu_grid = dedisperse_and_rasterise_chunk_gpu(
            vis_3d=sim["vis_3d"],
            wgt_3d=sim["wgt_3d"],
            uvw_3d=sim["uvw_3d"],
            freqs_hz=sim["freqs_hz"],
            nt_out=sim["nt_out"],
            dm=sim["dm"],
            dt=sim["dt"],
            collapse_channels=collapse_channels,
            nsubbands=nsubbands,
            exact_uvw=exact_uvw,
            npix_x=npix_x,
            npix_y=npix_y,
            pixsize_arcsec=pixsize_arcsec,
            mode=mode,
            min_valid_channels=min_valid_channels,
            min_uv_distance_m=min_uv_distance_m,
            uv_threshold=uv_threshold,
            return_host=True,
        )
        assert _rel_l2_error(gpu_grid, ref_grid) < 1e-6
        assert _rel_l2_error(gpu_grid, cpu_numba) < 1e-6


def test_hermitian_conjugation_and_sparse_lookup_parity():
    """
    Explicitly verify that a baseline at (-u, -v) produces the complex conjugate
    at the same half-plane cell (u_cell, (-v_cell) mod nv) and that sparse
    active-cell rasterisation (return_sparse=True) matches dense slicing.
    """
    nbl, nchan, nt = 2, 16, 4
    freqs_hz = np.full(nchan, 1.0e9, dtype=np.float64)
    vis_3d = np.zeros((nbl, nchan, nt), dtype=np.complex64)
    wgt_3d = np.ones((nbl, nchan, nt), dtype=np.float32)
    uvw_3d = np.zeros((nbl, 3, nt), dtype=np.float64)

    # Baseline 0 has u > 0; baseline 1 has (-u, -v) and conjugated visibility
    v0 = np.complex64(3.25 - 1.75j)
    vis_3d[0, :, :] = v0
    vis_3d[1, :, :] = np.conj(v0)
    uvw_3d[0, 0, :] = 450.0
    uvw_3d[0, 1, :] = -220.0
    uvw_3d[1, 0, :] = -450.0
    uvw_3d[1, 1, :] = 220.0

    dense_cpu = dedisperse_and_rasterise_chunk_cpu(
        vis_3d, wgt_3d, uvw_3d, freqs_hz,
        npix_x=64, npix_y=64, pixsize_arcsec=22.0,
        mode="nearest", min_valid_channels=1,
    )
    nz_v, nz_u = np.nonzero(np.abs(dense_cpu[0]) > 0)
    assert len(nz_v) == 1
    # Both baselines coherently reinforce v0 at the single half-plane cell
    assert np.allclose(dense_cpu[:, nz_v[0], nz_u[0]], v0, rtol=1e-6, atol=1e-6)

    u_half = 64 // 2 + 1
    flat_indices = np.array([nz_v[0] * u_half + nz_u[0], 0, 15], dtype=np.int64)
    uv_to_m_cpu = build_uv_to_m_lookup(64, u_half, flat_indices=flat_indices, device="cpu")
    sparse_cpu = dedisperse_and_rasterise_chunk_cpu(
        vis_3d, wgt_3d, uvw_3d, freqs_hz,
        npix_x=64, npix_y=64, pixsize_arcsec=22.0,
        mode="nearest", min_valid_channels=1,
        uv_to_m=uv_to_m_cpu, return_sparse=True,
    )
    assert sparse_cpu.shape == (3, nt)
    assert np.allclose(sparse_cpu[0], v0, rtol=1e-6, atol=1e-6)
    assert np.all(sparse_cpu[1:] == 0.0)

    if is_cuda_available():
        dense_gpu = dedisperse_and_rasterise_chunk_gpu(
            vis_3d, wgt_3d, uvw_3d, freqs_hz,
            npix_x=64, npix_y=64, pixsize_arcsec=22.0,
            mode="nearest", min_valid_channels=1,
        )
        sparse_gpu = dedisperse_and_rasterise_chunk_gpu(
            vis_3d, wgt_3d, uvw_3d, freqs_hz,
            npix_x=64, npix_y=64, pixsize_arcsec=22.0,
            mode="nearest", min_valid_channels=1,
            flat_indices=flat_indices, return_sparse=True,
        )
        assert _rel_l2_error(dense_gpu, dense_cpu) < 1e-6
        assert _rel_l2_error(sparse_gpu, sparse_cpu) < 1e-6

        t_chunk = np.arange(nt, dtype=np.float64) * 0.1
        c_idx_cpu, c_cnt_cpu, v_dat_cpu, v_tim_cpu = dedisperse_rasterise_to_ragged(
            vis_3d, wgt_3d, uvw_3d, freqs_hz, t_chunk,
            flat_indices=flat_indices, npix_x=64, npix_y=64,
            min_valid_channels=1, device="cpu",
        )
        c_idx_gpu, c_cnt_gpu, v_dat_gpu, v_tim_gpu = dedisperse_rasterise_to_ragged(
            vis_3d, wgt_3d, uvw_3d, freqs_hz, t_chunk,
            flat_indices=flat_indices, npix_x=64, npix_y=64,
            min_valid_channels=1, device="cuda", return_host=True,
        )
        np.testing.assert_array_equal(c_idx_gpu, c_idx_cpu)
        np.testing.assert_array_equal(c_cnt_gpu, c_cnt_cpu)
        np.testing.assert_allclose(v_dat_gpu, v_dat_cpu, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(v_tim_gpu, v_tim_cpu, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("corr_mode", ["single", "average", "stokesI"])
@pytest.mark.parametrize("wgt_3d_input", [True, False])
def test_stage_ms_rows_cpu_gpu_parity(corr_mode: str, wgt_3d_input: bool):
    """Verify stage_ms_rows_gpu and stage_ms_rows_cpu match across all correlation modes."""
    rng = np.random.default_rng(99)
    nbl, nt, nchan, ncorr = 10, 6, 64, 4
    nrow = nbl * nt
    bl_idx = np.tile(np.arange(nbl, dtype=np.int32), nt)
    t_rel = np.repeat(np.arange(nt, dtype=np.int32), nbl)
    uvw_rows = rng.uniform(-800.0, 800.0, size=(nrow, 3)).astype(np.float64)
    data_rows = (
        rng.standard_normal((nrow, nchan, ncorr))
        + 1j * rng.standard_normal((nrow, nchan, ncorr))
    ).astype(np.complex64)
    if wgt_3d_input:
        wgt_rows = rng.uniform(0.5, 2.0, size=(nrow, nchan, ncorr)).astype(np.float32)
    else:
        wgt_rows = rng.uniform(0.5, 2.0, size=(nrow, ncorr)).astype(np.float32)
    flag_rows = rng.random((nrow, nchan, ncorr)) < 0.2
    flag_row = rng.random(nrow) < 0.1
    labels = ["XX", "XY", "YX", "YY"]
    lbl2idx = {k: i for i, k in enumerate(labels)}

    v_cpu, w_cpu, u_cpu = stage_ms_rows_cpu(
        bl_idx, t_rel, uvw_rows, data_rows, wgt_rows, flag_rows,
        flag_row=flag_row, nbl=nbl, nt_stage=nt + 2, t_offset=2,
        corr_mode=corr_mode, basis="linear", single_pol="YY",
        labels=labels, lbl2idx=lbl2idx, min_uv_distance_m=120.0,
    )
    if is_cuda_available():
        v_gpu, w_gpu, u_gpu = stage_ms_rows_gpu(
            bl_idx, t_rel, uvw_rows, data_rows, wgt_rows, flag_rows,
            flag_row=flag_row, nbl=nbl, nt_stage=nt + 2, t_offset=2,
            corr_mode=corr_mode, basis="linear", single_pol="YY",
            labels=labels, lbl2idx=lbl2idx, min_uv_distance_m=120.0,
            return_host=True,
        )
        assert _rel_l2_error(v_gpu, v_cpu) < 1e-6
        assert _rel_l2_error(w_gpu, w_cpu) < 1e-6
        np.testing.assert_allclose(u_gpu, u_cpu, rtol=1e-12, atol=1e-12)


def _create_synthetic_ms(
    ms_path: Path,
    *,
    nbl: int = 8,
    nchan: int = 288,
    nt: int = 30,
    dt: float = 0.1,
) -> None:
    """Create a valid Casacore Measurement Set on disk with 288 channels for streaming tests."""
    ms_dir = str(ms_path)
    desc = maketabdesc(
        [
            makescacoldesc("TIME", 0.0, valuetype="double"),
            makescacoldesc("TIME_CENTROID", 0.0, valuetype="double"),
            makescacoldesc("ANTENNA1", 0, valuetype="int"),
            makescacoldesc("ANTENNA2", 0, valuetype="int"),
            makescacoldesc("SCAN_NUMBER", 1, valuetype="int"),
            makescacoldesc("FLAG_ROW", False, valuetype="boolean"),
            makearrcoldesc("UVW", 0.0, ndim=1, shape=[3], valuetype="double"),
            makearrcoldesc("DATA", 0j, ndim=2, shape=[nchan, 2], valuetype="complex"),
            makearrcoldesc("WEIGHT", 1.0, ndim=1, shape=[2], valuetype="float"),
            makearrcoldesc("FLAG", False, ndim=2, shape=[nchan, 2], valuetype="boolean"),
        ]
    )
    nrow = nbl * nt
    rng = np.random.default_rng(777)
    t0 = 5.1e9
    times = t0 + np.repeat(np.arange(nt, dtype=np.float64) * dt, nbl)
    a1 = np.tile(np.zeros(nbl, dtype=np.int32), nt)
    a2 = np.tile(np.arange(1, nbl + 1, dtype=np.int32), nt)

    base_u = np.linspace(-900.0, 950.0, nbl, dtype=np.float64)
    base_v = np.linspace(750.0, -820.0, nbl, dtype=np.float64)
    # Guarantee all baselines exceed 200 m
    base_u[np.abs(base_u) < 250.0] += 350.0
    uvw = np.zeros((nrow, 3), dtype=np.float64)
    uvw[:, 0] = np.tile(base_u, nt) + 0.5 * np.repeat(np.arange(nt), nbl)
    uvw[:, 1] = np.tile(base_v, nt) - 0.4 * np.repeat(np.arange(nt), nbl)

    data = (
        rng.standard_normal((nrow, nchan, 2))
        + 1j * rng.standard_normal((nrow, nchan, 2))
    ).astype(np.complex64)
    weight = rng.uniform(0.8, 1.5, size=(nrow, 2)).astype(np.float32)
    flags = rng.random((nrow, nchan, 2)) < 0.05

    with table(ms_dir, desc, nrow=nrow, ack=False) as t_main:
        t_main.putcol("TIME", times)
        t_main.putcol("TIME_CENTROID", times)
        t_main.putcol("ANTENNA1", a1)
        t_main.putcol("ANTENNA2", a2)
        t_main.putcol("SCAN_NUMBER", np.ones(nrow, dtype=np.int32))
        t_main.putcol("FLAG_ROW", np.zeros(nrow, dtype=bool))
        t_main.putcol("UVW", uvw)
        t_main.putcol("DATA", data)
        t_main.putcol("WEIGHT", weight)
        t_main.putcol("FLAG", flags)

    spw_desc = maketabdesc(
        [
            makearrcoldesc("CHAN_FREQ", 0.0, ndim=1, shape=[nchan], valuetype="double"),
        ]
    )
    freqs_hz = np.linspace(856.0e6, 1144.0e6, nchan, dtype=np.float64)
    with table(os.path.join(ms_dir, "SPECTRAL_WINDOW"), spw_desc, nrow=1, ack=False) as t_spw:
        t_spw.putcell("CHAN_FREQ", 0, freqs_hz)

    pol_desc = maketabdesc(
        [
            makearrcoldesc("CORR_TYPE", 0, ndim=1, shape=[2], valuetype="int"),
        ]
    )
    with table(os.path.join(ms_dir, "POLARIZATION"), pol_desc, nrow=1, ack=False) as t_pol:
        t_pol.putcell("CORR_TYPE", 0, np.array([9, 12], dtype=np.int32))  # XX, YY

    field_desc = maketabdesc(
        [
            makescacoldesc("NAME", "TEST_FIELD", valuetype="string"),
            makearrcoldesc("PHASE_DIR", 0.0, ndim=2, shape=[1, 2], valuetype="double"),
        ]
    )
    with table(os.path.join(ms_dir, "FIELD"), field_desc, nrow=1, ack=False) as t_fld:
        t_fld.putcell("NAME", 0, "J1801-2920")
        t_fld.putcell("PHASE_DIR", 0, np.array([[4.72004, -0.51225]], dtype=np.float64))


@pytest.mark.skipif(not HAS_CASACORE, reason="python-casacore required for MS streaming test")
def test_multi_chunk_streaming_lookahead_parity(tmp_path: Path):
    """
    Confirm that stream_ms_to_ragged_csr with a small row_chunk_size (forcing
    multiple chunks narrower than or comparable to max_delay) produces identical
    consolidated CSR stores to a single-chunk reference on both CPU and GPU.
    """
    ms_path = tmp_path / "synth_lookahead.ms"
    nbl, nchan, nt, dt, dm = 8, 288, 48, 0.013824, 125.0
    _create_synthetic_ms(ms_path, nbl=nbl, nchan=nchan, nt=nt, dt=dt)

    freqs_hz = np.linspace(856.0e6, 1144.0e6, nchan, dtype=np.float64)
    delays = get_dm_delays(freqs_hz, dm, dt, align_to="fmax")
    assert int(np.max(delays)) > 5  # Non-trivial causal delay sweep (23 samples) across chunk boundaries

    # Single-chunk CPU reference (all 240 rows in one chunk)
    res_single_cpu = stream_ms_to_ragged_csr(
        ms_path,
        dm=dm,
        npix_x=64,
        npix_y=64,
        pixsize_arcsec=22.0,
        collapse_channels=False,
        nsubbands=16,
        min_valid_channels=16,
        min_uv_distance_m=150.0,
        uv_threshold=1e-4,
        row_chunk_size=100_000,
        device="cpu",
    )
    # Multi-chunk CPU (32 rows = 4 timestamps per chunk -> 8 chunks across 30 timestamps)
    res_multi_cpu = stream_ms_to_ragged_csr(
        ms_path,
        dm=dm,
        npix_x=64,
        npix_y=64,
        pixsize_arcsec=22.0,
        collapse_channels=False,
        nsubbands=16,
        min_valid_channels=16,
        min_uv_distance_m=150.0,
        uv_threshold=1e-4,
        row_chunk_size=32,
        device="cpu",
    )

    s_ref = res_single_cpu.store
    s_mcpu = res_multi_cpu.store
    assert s_ref is not None and s_mcpu is not None
    assert s_ref.m_total > 0 and s_ref.nnz > 0
    np.testing.assert_array_equal(s_mcpu.cell_uv, s_ref.cell_uv)
    np.testing.assert_array_equal(s_mcpu.cell_offsets, s_ref.cell_offsets)
    np.testing.assert_allclose(s_mcpu.vis_times, s_ref.vis_times, rtol=1e-12, atol=1e-12)
    assert _rel_l2_error(s_mcpu.vis_data, s_ref.vis_data) < 1e-6

    if is_cuda_available():
        res_single_gpu = stream_ms_to_ragged_csr(
            ms_path,
            dm=dm,
            npix_x=64,
            npix_y=64,
            pixsize_arcsec=22.0,
            collapse_channels=False,
            nsubbands=16,
            min_valid_channels=16,
            min_uv_distance_m=150.0,
            uv_threshold=1e-4,
            row_chunk_size=100_000,
            device="cuda",
            return_csr_matrix=True,
        )
        res_multi_gpu = stream_ms_to_ragged_csr(
            ms_path,
            dm=dm,
            npix_x=64,
            npix_y=64,
            pixsize_arcsec=22.0,
            collapse_channels=False,
            nsubbands=16,
            min_valid_channels=16,
            min_uv_distance_m=150.0,
            uv_threshold=1e-4,
            row_chunk_size=32,
            device="cuda",
            return_csr_matrix=True,
        )

        g_single = gpu_store_to_host_ragged(res_single_gpu.gpu_store)
        g_multi = gpu_store_to_host_ragged(res_multi_gpu.gpu_store)

        np.testing.assert_array_equal(g_single.cell_uv, s_ref.cell_uv)
        np.testing.assert_array_equal(g_multi.cell_uv, s_ref.cell_uv)
        np.testing.assert_array_equal(g_single.cell_offsets, s_ref.cell_offsets)
        np.testing.assert_array_equal(g_multi.cell_offsets, s_ref.cell_offsets)
        np.testing.assert_allclose(g_multi.vis_times, s_ref.vis_times, rtol=1e-12, atol=1e-12)
        assert _rel_l2_error(g_single.vis_data, s_ref.vis_data) < 1e-6
        assert _rel_l2_error(g_multi.vis_data, s_ref.vis_data) < 1e-6
        assert _rel_l2_error(g_multi.vis_data, g_single.vis_data) < 1e-6

        csr_mat = res_multi_gpu.to_csr_matrix(validate=True)
        assert csr_mat.shape == (s_ref.m_total, nt)
        assert csr_mat.nnz == s_ref.nnz


def test_gpu_buffers_and_csr_matrix_bridge(tmp_path: Path):
    """
    Test RaggedDGAStore.from_gpu_buffers, GPURaggedDGAStore.to_csr_matrix, and
    build_gpu_csr_matrix on CPU and CUDA.
    """
    cell_offsets = np.array([0, 3, 5, 6], dtype=np.uint64)
    cell_uv = np.array([[10, 4], [22, 9], [31, 15]], dtype=np.int32)
    vis_data = np.array([1.0 + 0.5j, 2.0 - 1.0j, 0.5 + 0.25j, -1.5 + 0.75j, 3.0 + 0.0j, 0.25 - 0.5j], dtype=np.complex64)
    dt = 0.25
    vis_times = np.array([0.0, 0.5, 1.25, 0.25, 1.75, 1.0], dtype=np.float64)
    total_planes = 10
    attrs = {"tsamp": dt, "total_planes": total_planes, "npix_x": 64, "npix_y": 64, "nv": 64, "nu": 33}

    # Host construction & on-disk round-trip via RaggedDGAStore.from_gpu_buffers
    disk_dir = str(tmp_path / "host_csr_store")
    host_store = RaggedDGAStore.from_gpu_buffers(
        cell_offsets=cell_offsets,
        cell_uv=cell_uv,
        vis_data=vis_data,
        vis_times=vis_times,
        attrs=attrs,
        store_dir=disk_dir,
        copy_to_host=True,
        write_to_disk=True,
    )
    assert isinstance(host_store, RaggedDGAStore)
    assert host_store.m_total == 3
    assert host_store.nnz == 6
    np.testing.assert_array_equal(host_store.cell_offsets, cell_offsets)
    np.testing.assert_array_equal(host_store.cell_uv, cell_uv)
    np.testing.assert_allclose(host_store.vis_data, vis_data)
    np.testing.assert_allclose(host_store.vis_times, vis_times)

    if is_cuda_available() and cp is not None and cupy_sparse is not None:
        d_off = cp.asarray(cell_offsets)
        d_uv = cp.asarray(cell_uv)
        d_vis = cp.asarray(vis_data)
        d_times = cp.asarray(vis_times)

        gpu_store = RaggedDGAStore.from_gpu_buffers(
            cell_offsets=d_off,
            cell_uv=d_uv,
            vis_data=d_vis,
            vis_times=d_times,
            attrs=attrs,
            copy_to_host=False,
        )
        assert gpu_store.m_total == 3
        assert gpu_store.nnz == 6
        # Verify zero-copy pointer sharing in VRAM
        assert int(gpu_store.vis_data.data.ptr) == int(d_vis.data.ptr)

        csr = gpu_store.to_csr_matrix(total_planes=total_planes, dt=dt, validate=True)
        assert isinstance(csr, cupy_sparse.csr_matrix)
        assert csr.shape == (3, total_planes)
        assert csr.nnz == 6
        np.testing.assert_array_equal(cp.asnumpy(csr.indptr), cell_offsets.astype(np.int32))
        np.testing.assert_array_equal(
            cp.asnumpy(csr.indices),
            np.rint(vis_times / dt).astype(np.int32),
        )
        np.testing.assert_allclose(cp.asnumpy(csr.data), vis_data)

        csr_direct = build_gpu_csr_matrix(
            gpu_store, total_planes=total_planes, dt=dt, validate=True
        )
        np.testing.assert_allclose(cp.asnumpy(csr_direct.toarray()), cp.asnumpy(csr.toarray()))

        roundtrip_host = gpu_store.to_cpu()
        np.testing.assert_allclose(roundtrip_host.vis_data, vis_data)
        gpu_store.release()


def test_wgridder_cpu_gpu_and_ducc_vis2grid_parity():
    """
    Verify that mode='wgridder' (both direct 2D DUCC piecewise polynomial kernel +
    w-phase preconditioning and full 3D DUCC w-stacking) matches CPU
    ducc0.wgridder.vis2grid to < 1e-4 relative difference (actually < 1e-6)
    across Numba CPU, pure NumPy CPU, and CUDA GPU, as well as sparse CSR output.
    """
    ducc0 = pytest.importorskip("ducc0")
    sim = _simulate_288chan_chunk(seed=4242)
    npix_x, npix_y = 128, 128
    pixsize_arcsec = 22.0
    pix_rad = pixsize_arcsec / 206265.0
    nt_out = 4
    vis_3d = sim["vis_3d"][:, :, :nt_out]
    wgt_3d = sim["wgt_3d"][:, :, :nt_out]
    uvw_3d = sim["uvw_3d"][:, :, :nt_out]
    freqs_hz = sim["freqs_hz"]

    # 1. Compare 2D DUCC polynomial kernel (apply_w_phase=False) against ducc0.wgridder.vis2grid(do_wgridding=False)
    from fastducc import imaging

    ref_2d = imaging.grid_chunk_uv_cube(
        vis_3d,
        wgt_3d,
        uvw_3d,
        freqs_hz,
        sim["dt"],
        dm=0.0,
        collapse_channels=False,
        nsubbands=1,
        npix_x=npix_x,
        npix_y=npix_y,
        pixsize_x=pix_rad,
        pixsize_y=pix_rad,
        epsilon=1e-5,
        do_wgridding=False,
        min_valid_channels=1,
    )
    u_half_2d = ref_2d.shape[2] // 2 + 1
    ref_2d_hp = ref_2d[:, :, :u_half_2d]

    cpu_numba_2d = dedisperse_and_rasterise_chunk_cpu(
        vis_3d,
        wgt_3d,
        uvw_3d,
        freqs_hz,
        npix_x=npix_x,
        npix_y=npix_y,
        pixsize_arcsec=pixsize_arcsec,
        mode="wgridder",
        epsilon=1e-5,
        do_wgridding=False,
        apply_w_phase=False,
        normalise_mode="vis2grid",
        min_valid_channels=1,
        use_numba=True,
    )
    cpu_numpy_2d = dedisperse_and_rasterise_chunk_cpu(
        vis_3d,
        wgt_3d,
        uvw_3d,
        freqs_hz,
        npix_x=npix_x,
        npix_y=npix_y,
        pixsize_arcsec=pixsize_arcsec,
        mode="wgridder",
        epsilon=1e-5,
        do_wgridding=False,
        apply_w_phase=False,
        normalise_mode="vis2grid",
        min_valid_channels=1,
        use_numba=False,
    )
    assert _rel_l2_error(cpu_numba_2d, ref_2d_hp) < 1e-5
    assert _rel_l2_error(cpu_numpy_2d, ref_2d_hp) < 1e-5
    assert _rel_l2_error(cpu_numba_2d, cpu_numpy_2d) < 1e-6

    # 2. Direct 2D DUCC kernel + w-phase preconditioning (default mode="wgridder"): CPU vs GPU & sparse CSR
    wg_cpu_numba = dedisperse_and_rasterise_chunk_cpu(
        vis_3d=sim["vis_3d"],
        wgt_3d=sim["wgt_3d"],
        uvw_3d=sim["uvw_3d"],
        freqs_hz=freqs_hz,
        nt_out=sim["nt_out"],
        dm=sim["dm"],
        dt=sim["dt"],
        collapse_channels=False,
        nsubbands=16,
        exact_uvw=True,
        npix_x=npix_x,
        npix_y=npix_y,
        pixsize_arcsec=pixsize_arcsec,
        mode="wgridder",
        epsilon=1e-5,
        min_valid_channels=24,
        min_uv_distance_m=150.0,
        uv_threshold=1e-4,
        use_numba=True,
    )
    wg_cpu_numpy = dedisperse_and_rasterise_chunk_cpu(
        vis_3d=sim["vis_3d"],
        wgt_3d=sim["wgt_3d"],
        uvw_3d=sim["uvw_3d"],
        freqs_hz=freqs_hz,
        nt_out=sim["nt_out"],
        dm=sim["dm"],
        dt=sim["dt"],
        collapse_channels=False,
        nsubbands=16,
        exact_uvw=True,
        npix_x=npix_x,
        npix_y=npix_y,
        pixsize_arcsec=pixsize_arcsec,
        mode="wgridder",
        epsilon=1e-5,
        min_valid_channels=24,
        min_uv_distance_m=150.0,
        uv_threshold=1e-4,
        use_numba=False,
    )
    assert _rel_l2_error(wg_cpu_numba, wg_cpu_numpy) < 1e-6

    # 3. Compare full 3D DUCC w-stacking (do_wgridding=True) against ducc0.wgridder.vis2grid(do_wgridding=True)
    ref_3d = imaging.grid_chunk_uv_cube(
        vis_3d[:, :, :2],
        wgt_3d[:, :, :2],
        uvw_3d[:, :, :2],
        freqs_hz,
        sim["dt"],
        dm=0.0,
        collapse_channels=False,
        nsubbands=1,
        npix_x=npix_x,
        npix_y=npix_y,
        pixsize_x=pix_rad,
        pixsize_y=pix_rad,
        epsilon=1e-5,
        do_wgridding=True,
        min_valid_channels=1,
    )
    u_half_3d = ref_3d.shape[2] // 2 + 1
    ref_3d_hp = ref_3d[:, :, :u_half_3d]

    cpu_3d = dedisperse_and_rasterise_chunk_cpu(
        vis_3d[:, :, :2],
        wgt_3d[:, :, :2],
        uvw_3d[:, :, :2],
        freqs_hz,
        npix_x=npix_x,
        npix_y=npix_y,
        pixsize_arcsec=pixsize_arcsec,
        mode="wgridder",
        epsilon=1e-5,
        do_wgridding=True,
        normalise_mode="vis2grid",
        min_valid_channels=1,
    )
    assert _rel_l2_error(cpu_3d, ref_3d_hp) < 1e-5

    if is_cuda_available():
        gpu_2d = dedisperse_and_rasterise_chunk_gpu(
            vis_3d,
            wgt_3d,
            uvw_3d,
            freqs_hz,
            npix_x=npix_x,
            npix_y=npix_y,
            pixsize_arcsec=pixsize_arcsec,
            mode="wgridder",
            epsilon=1e-5,
            do_wgridding=False,
            apply_w_phase=False,
            normalise_mode="vis2grid",
            min_valid_channels=1,
            return_host=True,
        )
        assert _rel_l2_error(gpu_2d, ref_2d_hp) < 1e-5
        assert _rel_l2_error(gpu_2d, cpu_numba_2d) < 1e-6

        wg_gpu = dedisperse_and_rasterise_chunk_gpu(
            vis_3d=sim["vis_3d"],
            wgt_3d=sim["wgt_3d"],
            uvw_3d=sim["uvw_3d"],
            freqs_hz=freqs_hz,
            nt_out=sim["nt_out"],
            dm=sim["dm"],
            dt=sim["dt"],
            collapse_channels=False,
            nsubbands=16,
            exact_uvw=True,
            npix_x=npix_x,
            npix_y=npix_y,
            pixsize_arcsec=pixsize_arcsec,
            mode="wgridder",
            epsilon=1e-5,
            min_valid_channels=24,
            min_uv_distance_m=150.0,
            uv_threshold=1e-4,
            return_host=True,
        )
        assert _rel_l2_error(wg_gpu, wg_cpu_numba) < 1e-6

        gpu_3d = dedisperse_and_rasterise_chunk_gpu(
            vis_3d[:, :, :2],
            wgt_3d[:, :, :2],
            uvw_3d[:, :, :2],
            freqs_hz,
            npix_x=npix_x,
            npix_y=npix_y,
            pixsize_arcsec=pixsize_arcsec,
            mode="wgridder",
            epsilon=1e-5,
            do_wgridding=True,
            normalise_mode="vis2grid",
            min_valid_channels=1,
            return_host=True,
        )
        assert _rel_l2_error(gpu_3d, ref_3d_hp) < 1e-5
        assert _rel_l2_error(gpu_3d, cpu_3d) < 1e-6
