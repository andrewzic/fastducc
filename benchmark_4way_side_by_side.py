# -*- coding: utf-8 -*-
"""
Four-Way Side-by-Side Periodicity Search Benchmark on Real CRACO Data with Injected Pulsar.
Evaluated on native CRACO grid dimensions: 384x384 pixels, 22 arcsec/pixel.

Compares:
1. Baseline-based FFA (fastducc + riptide Option A)
2. Image Cube + Time-domain FFT (with harmonic summing)
3. NUFFT-based 3D Imager (fastducc ducc0.nufft)
4. FFA on Gridded UV Cubes (Legacy UV-plane FFA with proper Hermitian gridding)
"""

from __future__ import annotations

import os
import sys
import time
import math
import logging
from typing import Tuple, Dict, Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Package paths
sys.path.insert(0, os.path.abspath("fastducc"))
sys.path.insert(0, os.path.abspath("riptide/src"))

import ducc0
import riptide.libcpp as libcpp
from riptide.ffautils import plan_ffa
from fastducc.imaging import load_chunk_data
from fastducc.baseline_ffa import (
    prepare_baseline_series,
    fold_baselines_option_a,
    image_phase_bins_wgridder,
    evaluate_period_snr
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("4way_benchmark")


def wrap_pi(x: np.ndarray) -> np.ndarray:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def next_pow2(n: int) -> int:
    return 1 << int(math.ceil(math.log2(max(1, n))))


def harmonic_sum_1d(power_spec: np.ndarray, num_harmonics: int = 16) -> np.ndarray:
    n = len(power_spec)
    out = np.zeros_like(power_spec)
    harmonics = [1, 2, 4, 8, 16] if num_harmonics == 16 else [1, 2, 4, 8]
    harmonics = [h for h in harmonics if h <= num_harmonics]

    for h in harmonics:
        idx = np.arange(n) * h
        valid = (idx < n)
        out[valid] += power_spec[idx[valid]]
    return out


def main():
    msname = "riptide/tests/data/cracoData.LTR_1733-2344.SB77974.beam01.20251015072402.calG9.uvsub.ms"
    artifact_dir = "/Users/zic006/.gemini/antigravity/brain/62efa71d-bc44-490e-ba26-4fbd9d0d085c"
    os.makedirs(artifact_dir, exist_ok=True)

    logger.info("=" * 75)
    logger.info("FOUR-WAY BENCHMARK ON REAL DATA (CRACO NATIVE 384x384 GRID, 22'' PIXELS)")
    logger.info("=" * 75)

    # 1. Load Real Data Chunk
    nt_target = 1000
    npix_x = 384
    npix_y = 384
    pixsize_x = 22.0 / 206265.0
    pixsize_y = 22.0 / 206265.0

    t_read_0 = time.time()
    logger.info(f"Loading {nt_target} time integrations from {msname}...")
    u_times, chan_freq, dt, vis_3d, wgt_3d, uvw_3d = load_chunk_data(
        msname, start_time_idx=0, end_time_idx=nt_target, corr_mode='average', average_correlations=True
    )
    t_read = time.time() - t_read_0
    nbl, nchan, nt = vis_3d.shape
    total_obs_s = nt * dt
    logger.info(f"Data loaded in {t_read:.2f}s: {nbl} baselines, {nchan} channels, {nt} integrations ({total_obs_s:.2f} s).")

    # Collapse channels at DM=0
    vis_2d, wgt_2d, uvw_track, eff_freq, dt = prepare_baseline_series(
        vis_3d, wgt_3d, uvw_3d, chan_freq, dt, dm=0.0, collapse_channels=True
    )

    # 2. Inject Pulsar Signal
    p_inj = 2.50000        # seconds
    duty_cycle = 0.05      # 5% FWHM
    amp_inj = 0.08         # coherent peak flux
    dx_pix = +10           # +10 pixels along X from center (192, 192) -> x=202
    dy_pix = +6            # +6 pixels along Y from center (192, 192) -> y=198
    target_x = (npix_x // 2) + dx_pix
    target_y = (npix_y // 2) + dy_pix

    dl = dx_pix * pixsize_x
    dm = dy_pix * pixsize_y
    f_inj = 1.0 / p_inj

    logger.info(f"Injecting fake pulsar: P = {p_inj:.5f} s (f = {f_inj:.4f} Hz), Duty = {duty_cycle*100:.1f}%")
    logger.info(f"Target pixel: ({target_x}, {target_y}) in {npix_x}x{npix_y} image grid")

    t_axis = np.arange(nt) * dt
    phase_t = (t_axis / p_inj) % 1.0

    sigma_phi = duty_cycle / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    pulse_t = np.exp(-0.5 * ((phase_t - 0.5) / sigma_phi) ** 2).astype(np.float32)

    c_light = 299792458.0
    lam = c_light / eff_freq[0]
    u_lam = uvw_track[:, 0, :] / lam  # (nbl, nt)
    v_lam = uvw_track[:, 1, :] / lam  # (nbl, nt)

    geom_phase = -2.0 * np.pi * (u_lam * dl + v_lam * dm)
    pulse_signal = (amp_inj * pulse_t[None, :] * np.exp(1j * geom_phase)).astype(np.complex64)
    vis_injected = vis_2d + pulse_signal
    logger.info("Signal injection complete.")

    p_min = 2.20
    p_max = 2.80
    b_min = 16
    b_max = 24

    results_summary = {}
    best_snr_plane_1 = None
    best_snr_plane_4 = None

    # =========================================================================
    # METHOD 1: Baseline-based FFA (Option A)
    # =========================================================================
    logger.info("-" * 60)
    logger.info("METHOD 1: Baseline-based FFA (fastducc + riptide Option A)")
    logger.info("-" * 60)
    t0 = time.time()

    plan_1 = plan_ffa(nt, dt, p_min, p_max, b_min, b_max)
    periods_1 = []
    snrs_1 = []
    best_cand_1 = None
    best_snr_1 = -1e9
    best_cube_1 = None
    best_period_1 = None

    for bp in plan_1:
        b_bins = int(bp["bins"])
        ds = float(bp["downsample_factor"])
        rows_eval = int(bp["rows_eval"])

        uvw_eff, wgt_eff, folded_vis, trial_periods = fold_baselines_option_a(
            vis_injected, wgt_2d, uvw_track, b_bins, dt,
            segment_duration_s=30.0, rows_eval=rows_eval, downsample_factor=ds
        )

        for r, period in enumerate(trial_periods):
            vis_p = folded_vis[:, r, :]
            phase_imgs = image_phase_bins_wgridder(
                uvw=uvw_eff, freq=eff_freq, vis_period=vis_p, wgt=wgt_eff,
                npix_x=npix_x, npix_y=npix_y, pixsize_x=pixsize_x, pixsize_y=pixsize_y
            )
            snr_plane, max_snr, cand = evaluate_period_snr(phase_imgs, period, snr_thresh=5.0)

            target_pixel_snr = float(snr_plane[target_y, target_x])
            periods_1.append(period)
            snrs_1.append(target_pixel_snr)

            if target_pixel_snr > best_snr_1:
                best_snr_1 = target_pixel_snr
                best_period_1 = period
                best_cand_1 = cand
                best_cube_1 = phase_imgs.copy()
                best_snr_plane_1 = snr_plane.copy()

    t_m1 = time.time() - t0
    periods_1 = np.array(periods_1)
    snrs_1 = np.array(snrs_1)
    sort_idx1 = np.argsort(periods_1)
    periods_1 = periods_1[sort_idx1]
    snrs_1 = snrs_1[sort_idx1]

    best_idx1 = int(np.argmax(snrs_1))
    recovered_p1 = float(periods_1[best_idx1])
    logger.info(f"Method 1 Complete in {t_m1:.2f}s! Evaluated {len(periods_1)} periods.")
    logger.info(f"  Recovered P = {recovered_p1:.6f} s (Error: {abs(recovered_p1 - p_inj)*1000:.3f} ms), Peak SNR = {best_snr_1:.2f} sigma")

    results_summary["Baseline FFA"] = {
        "runtime_s": t_m1,
        "recovered_period": recovered_p1,
        "period_error_ms": abs(recovered_p1 - p_inj) * 1000.0,
        "peak_snr": best_snr_1,
        "peak_pixel": (best_cand_1.x if best_cand_1 else target_x, best_cand_1.y if best_cand_1 else target_y),
        "periods": periods_1,
        "snrs": snrs_1,
    }

    # =========================================================================
    # METHOD 2: Image Cube + Time-Domain FFT
    # =========================================================================
    logger.info("-" * 60)
    logger.info("METHOD 2: Image Cube + Time-Domain FFT (with harmonic summing)")
    logger.info("-" * 60)
    t0 = time.time()

    logger.info(f"Synthesizing {nt} snapshot dirty images at {npix_x}x{npix_y}...")
    img_cube = np.empty((nt, npix_y, npix_x), dtype=np.float32)
    freq_arr = eff_freq

    for it in range(nt):
        vis_it = np.ascontiguousarray(vis_injected[:, it:it+1], dtype=np.complex64)
        wgt_it = np.ascontiguousarray(wgt_2d[:, it:it+1], dtype=np.float32)
        uvw_it = np.ascontiguousarray(uvw_track[:, :, it], dtype=np.float64)

        dirty_it = ducc0.wgridder.vis2dirty(
            uvw=uvw_it, freq=freq_arr, vis=vis_it, wgt=wgt_it,
            npix_x=npix_x, npix_y=npix_y, pixsize_x=pixsize_x, pixsize_y=pixsize_y,
            epsilon=1e-5, do_wgridding=True, nthreads=0
        )
        img_cube[it] = dirty_it.T

    logger.info("Computing 1D FFT along time axis across all pixels...")
    fft_cube = np.fft.rfft(img_cube, axis=0)
    freqs_fft = np.fft.rfftfreq(nt, d=dt)
    power_cube = (np.abs(fft_cube) ** 2).astype(np.float32)

    target_spec = power_cube[:, target_y, target_x]
    harm_1 = target_spec.copy()
    harm_16 = harmonic_sum_1d(target_spec, num_harmonics=16)

    f_min = 1.0 / p_max
    f_max = 1.0 / p_min
    f_mask = (freqs_fft >= f_min) & (freqs_fft <= f_max)
    sub_f = freqs_fft[f_mask]
    sub_periods_fft = 1.0 / sub_f

    def compute_fft_snr(spec_1d):
        amp = np.sqrt(np.maximum(0.0, spec_1d))
        bg = np.median(amp)
        mad = 1.4826 * np.median(np.abs(amp - bg))
        return (amp - bg) / max(1e-12, mad)

    snr_harm1 = compute_fft_snr(harm_1)[f_mask]
    snr_harm16 = compute_fft_snr(harm_16)[f_mask]

    peak_sub_idx = int(np.argmax(snr_harm16))
    recovered_p2 = float(sub_periods_fft[peak_sub_idx])
    peak_snr_2_fund = float(snr_harm1[peak_sub_idx])
    peak_snr_2_harm = float(snr_harm16[peak_sub_idx])

    t_m2 = time.time() - t0
    logger.info(f"Method 2 Complete in {t_m2:.2f}s!")
    logger.info(f"  Recovered P = {recovered_p2:.6f} s (Error: {abs(recovered_p2 - p_inj)*1000:.3f} ms)")
    logger.info(f"  Fundamental (1-harm) SNR = {peak_snr_2_fund:.2f} sigma, 16-Harmonic SNR = {peak_snr_2_harm:.2f} sigma")

    results_summary["Image Cube + FFT"] = {
        "runtime_s": t_m2,
        "recovered_period": recovered_p2,
        "period_error_ms": abs(recovered_p2 - p_inj) * 1000.0,
        "peak_snr": peak_snr_2_harm,
        "fundamental_snr": peak_snr_2_fund,
        "peak_pixel": (target_x, target_y),
        "periods": sub_periods_fft,
        "snrs": snr_harm16,
    }

    # =========================================================================
    # METHOD 3: NUFFT-Based 3D Imager in fastducc
    # =========================================================================
    logger.info("-" * 60)
    logger.info("METHOD 3: NUFFT-Based 3D Imager (fastducc ducc0.nufft at 384x384)")
    logger.info("-" * 60)
    t0 = time.time()

    Nf = next_pow2(nt)
    df = 1.0 / (Nf * dt)
    dl_rad = pixsize_x

    u_pts = uvw_track[:, 0, :].flatten() / lam
    v_pts = uvw_track[:, 1, :].flatten() / lam
    t_pts = np.repeat(t_axis[None, :], nbl, axis=0).flatten()
    vis_pts = vis_injected.flatten()

    omega_u = 2.0 * math.pi * u_pts * dl_rad
    omega_v = 2.0 * math.pi * v_pts * dl_rad
    omega_f = -2.0 * math.pi * t_pts * df
    coord_3d = np.stack([wrap_pi(omega_u), wrap_pi(omega_v), wrap_pi(omega_f)], axis=1).astype(np.float64)

    logger.info(f"Executing 3D adjoint NUFFT (u,v,t) -> (l,m,f) for {len(vis_pts)} points...")
    out_buf_3d = np.zeros((npix_x, npix_y, Nf), dtype=np.complex64)
    ducc0.nufft.nu2u(
        points=vis_pts,
        coord=coord_3d,
        forward=False,
        epsilon=1e-5,
        nthreads=0,
        out=out_buf_3d,
        fft_order=True
    )

    target_nx = dx_pix if dx_pix >= 0 else npix_x + dx_pix
    target_ny = dy_pix if dy_pix >= 0 else npix_y + dy_pix

    spec_3d = (np.abs(out_buf_3d[target_nx, target_ny, :Nf//2]) ** 2).astype(np.float32)
    freqs_3d = np.arange(Nf // 2) * df

    harm_3d_1 = spec_3d.copy()
    harm_3d_16 = harmonic_sum_1d(spec_3d, num_harmonics=16)

    mask_3d = (freqs_3d >= f_min) & (freqs_3d <= f_max)
    sub_f_3d = freqs_3d[mask_3d]
    sub_periods_3d = 1.0 / sub_f_3d

    snr_3d_fund = compute_fft_snr(harm_3d_1)[mask_3d]
    snr_3d_harm16 = compute_fft_snr(harm_3d_16)[mask_3d]

    peak_3d_idx = int(np.argmax(snr_3d_harm16))
    recovered_p3 = float(sub_periods_3d[peak_3d_idx])
    peak_snr_3_fund = float(snr_3d_fund[peak_3d_idx])
    peak_snr_3_harm = float(snr_3d_harm16[peak_3d_idx])

    t_m3 = time.time() - t0
    logger.info(f"Method 3 Complete in {t_m3:.2f}s!")
    logger.info(f"  Recovered P = {recovered_p3:.6f} s (Error: {abs(recovered_p3 - p_inj)*1000:.3f} ms)")
    logger.info(f"  Fundamental (1-harm) SNR = {peak_snr_3_fund:.2f} sigma, 16-Harmonic SNR = {peak_snr_3_harm:.2f} sigma")

    results_summary["3D NUFFT Imager"] = {
        "runtime_s": t_m3,
        "recovered_period": recovered_p3,
        "period_error_ms": abs(recovered_p3 - p_inj) * 1000.0,
        "peak_snr": peak_snr_3_harm,
        "fundamental_snr": peak_snr_3_fund,
        "peak_pixel": (target_x, target_y),
        "periods": sub_periods_3d,
        "snrs": snr_3d_harm16,
    }

    # =========================================================================
    # METHOD 4: FFA on Gridded UV Cubes (Legacy UV-plane FFA at 384x384)
    # =========================================================================
    logger.info("-" * 60)
    logger.info("METHOD 4: FFA on Gridded UV Cubes (Legacy UV-plane FFA at 384x384)")
    logger.info("-" * 60)
    t0 = time.time()

    du = 1.0 / (npix_x * pixsize_x)
    dv = 1.0 / (npix_y * pixsize_y)

    grid_uv = np.zeros((npix_y, npix_x, nt), dtype=np.complex64)
    wgt_uv = np.zeros((npix_y, npix_x, nt), dtype=np.float32)

    logger.info(f"Gridding baseline tracks into {npix_x}x{npix_y} UV cells (du = {du:.3f} lam)...")
    for it in range(nt):
        u_t = u_lam[:, it]
        v_t = v_lam[:, it]
        vis_t = vis_injected[:, it]
        w_t = wgt_2d[:, it]

        # col = iu (u), row = iv (v)
        iu = np.round(u_t / du).astype(int) + npix_x // 2
        iv = np.round(v_t / dv).astype(int) + npix_y // 2

        # Forward points (+u, +v)
        valid = (iu >= 0) & (iu < npix_x) & (iv >= 0) & (iv < npix_y)
        np.add.at(grid_uv[:, :, it], (iv[valid], iu[valid]), vis_t[valid] * w_t[valid])
        np.add.at(wgt_uv[:, :, it], (iv[valid], iu[valid]), w_t[valid])

        # Hermitian conjugate points (-u, -v) -> conj(V)
        iu_c = np.round(-u_t / du).astype(int) + npix_x // 2
        iv_c = np.round(-v_t / dv).astype(int) + npix_y // 2
        valid_c = (iu_c >= 0) & (iu_c < npix_x) & (iv_c >= 0) & (iv_c < npix_y)
        np.add.at(grid_uv[:, :, it], (iv_c[valid_c], iu_c[valid_c]), np.conj(vis_t[valid_c]) * w_t[valid_c])
        np.add.at(wgt_uv[:, :, it], (iv_c[valid_c], iu_c[valid_c]), w_t[valid_c])

    # Normalize active UV cells
    has_w = (wgt_uv > 0)
    grid_uv[has_w] /= wgt_uv[has_w]

    # Active UV cells
    uv_mask = (np.sum(wgt_uv, axis=2) > 0)
    active_iv, active_iu = np.nonzero(uv_mask)
    n_active_uv = len(active_iv)
    logger.info(f"Total active UV cells: {n_active_uv} / {npix_x * npix_y} cells")

    dga = np.ascontiguousarray(grid_uv[active_iv, active_iu, :], dtype=np.complex64)

    periods_4 = []
    snrs_4 = []
    best_snr_4 = -1e9
    best_period_4 = None
    best_cand_4 = None
    best_img_4 = None

    for bp in plan_1:
        b_bins = int(bp["bins"])
        ds = float(bp["downsample_factor"])
        rows_eval = int(bp["rows_eval"])
        eff_tau = dt * ds

        folded_uv, trial_p = libcpp.vis_ffa_transform_basep(
            dga,
            b_bins,
            eff_tau,
            int(rows_eval),
            float(ds)
        )

        n_trial = len(trial_p)
        for r in range(n_trial):
            p = trial_p[r]
            phase_imgs_4 = np.empty((b_bins, npix_y, npix_x), dtype=np.float32)
            for k in range(b_bins):
                g2d = np.zeros((npix_y, npix_x), dtype=np.complex64)
                g2d[active_iv, active_iu] = folded_uv[:, r, k]
                img2d = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(g2d))).real
                phase_imgs_4[k] = img2d

            snr_plane_4, max_snr_4, cand_4 = evaluate_period_snr(phase_imgs_4, p, snr_thresh=5.0)
            target_snr_4 = float(snr_plane_4[target_y, target_x])
            periods_4.append(p)
            snrs_4.append(target_snr_4)

            if target_snr_4 > best_snr_4:
                best_snr_4 = target_snr_4
                best_period_4 = p
                best_cand_4 = cand_4
                best_img_4 = phase_imgs_4.copy()
                best_snr_plane_4 = snr_plane_4.copy()

    t_m4 = time.time() - t0
    periods_4 = np.array(periods_4)
    snrs_4 = np.array(snrs_4)
    sort_idx4 = np.argsort(periods_4)
    periods_4 = periods_4[sort_idx4]
    snrs_4 = snrs_4[sort_idx4]

    best_idx4 = int(np.argmax(snrs_4))
    recovered_p4 = float(periods_4[best_idx4])
    logger.info(f"Method 4 Complete in {t_m4:.2f}s! Evaluated {len(periods_4)} periods.")
    logger.info(f"  Recovered P = {recovered_p4:.6f} s (Error: {abs(recovered_p4 - p_inj)*1000:.3f} ms), Peak SNR = {best_snr_4:.2f} sigma")

    results_summary["Gridded UV FFA"] = {
        "runtime_s": t_m4,
        "recovered_period": recovered_p4,
        "period_error_ms": abs(recovered_p4 - p_inj) * 1000.0,
        "peak_snr": best_snr_4,
        "peak_pixel": (target_x, target_y),
        "periods": periods_4,
        "snrs": snrs_4,
    }

    # =========================================================================
    # 5. SIDE-BY-SIDE PLOTS & DIAGNOSTICS
    # =========================================================================
    logger.info("Generating side-by-side diagnostic figures...")

    # Plot 1: Periodograms Comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=150, sharex=True)
    methods = [
        ("Baseline-based FFA (fastducc)", results_summary["Baseline FFA"], "#1f77b4", axes[0, 0]),
        ("Image Cube + FFT (16 Harmonics)", results_summary["Image Cube + FFT"], "#ff7f0e", axes[0, 1]),
        ("3D NUFFT Imager (16 Harmonics)", results_summary["3D NUFFT Imager"], "#2ca02c", axes[1, 0]),
        ("Gridded UV FFA (Legacy 384x384)", results_summary["Gridded UV FFA"], "#d62728", axes[1, 1]),
    ]

    for title, res_dict, color, ax in methods:
        ax.plot(res_dict["periods"], res_dict["snrs"], color=color, lw=1.3, label="Periodogram")
        ax.axvline(p_inj, color="black", ls="--", lw=1.2, label=f"Injected $P={p_inj:.2f}$s")
        ax.scatter([res_dict["recovered_period"]], [res_dict["peak_snr"]], color="gold", edgecolor="black", s=70, zorder=5,
                   label=f"Peak: {res_dict['recovered_period']:.4f}s ({res_dict['peak_snr']:.1f}$\sigma$)")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Detection S/N")
        ax.grid(True, alpha=0.3, ls=":")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_xlim(p_min, p_max)

    axes[1, 0].set_xlabel("Trial Period (s)", fontsize=11)
    axes[1, 1].set_xlabel("Trial Period (s)", fontsize=11)
    plt.suptitle(f"Side-by-Side Periodogram Comparison (Injected $P = {p_inj:.4f}$s, Duty = 5%, Offset = +10, +6 pix on 384x384)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    pgram_path = os.path.join(artifact_dir, "comparison_periodograms.png")
    plt.savefig(pgram_path)
    plt.close()
    logger.info(f"Saved periodogram comparison to {pgram_path}")

    # Plot 2: Reconstructed Pulse Profiles & Harmonic Spectra
    fig, (ax_prof, ax_harm) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # Left: Pulse Profile from Baseline FFA and Gridded UV FFA vs Injected
    if best_cube_1 is not None:
        nb1 = best_cube_1.shape[0]
        prof_ffa = best_cube_1[:, target_y, target_x]
        phase_ax = np.linspace(0, 1, nb1, endpoint=False)
        phase_2 = np.concatenate([phase_ax, phase_ax + 1.0])
        prof_2 = np.concatenate([prof_ffa, prof_ffa])
        true_prof = np.exp(-0.5 * ((phase_2 - 0.5) / sigma_phi) ** 2)
        true_prof = (true_prof / np.max(true_prof)) * (np.max(prof_2) - np.median(prof_2)) + np.median(prof_2)

        ax_prof.step(phase_2, prof_2, where="mid", color="#1f77b4", lw=1.8, label="Baseline FFA Recovered")
        ax_prof.plot(phase_2, true_prof, "k--", lw=1.5, alpha=0.8, label="Injected 5% Duty Gaussian")

    if best_img_4 is not None:
        nb4 = best_img_4.shape[0]
        prof_uv = best_img_4[:, target_y, target_x]
        phase_ax4 = np.linspace(0, 1, nb4, endpoint=False)
        phase_4_2 = np.concatenate([phase_ax4, phase_ax4 + 1.0])
        prof_4_2 = np.concatenate([prof_uv, prof_uv])
        # Scale to match display
        scale_4 = (np.max(prof_2) - np.median(prof_2)) / max(1e-12, np.max(prof_4_2) - np.median(prof_4_2))
        prof_4_norm = (prof_4_2 - np.median(prof_4_2)) * scale_4 + np.median(prof_2)
        ax_prof.step(phase_4_2, prof_4_norm, where="mid", color="#d62728", lw=1.4, ls=":", label="Gridded UV FFA Recovered")

    ax_prof.set_title("Reconstructed Pulse Profiles (FFA Methods)", fontsize=11, fontweight="bold")
    ax_prof.set_xlabel("Rotational Phase")
    ax_prof.set_ylabel("Normalized Intensity")
    ax_prof.grid(True, alpha=0.3, ls=":")
    ax_prof.legend(loc="upper right")

    # Right: Fourier Harmonic Power Distribution
    fund_bin = int(round(f_inj / (freqs_fft[1] - freqs_fft[0])))
    h_idx = np.arange(1, 17)
    fft_harm_powers = [float(target_spec[min(len(target_spec)-1, fund_bin * h)]) for h in h_idx]
    fft_harm_powers = np.array(fft_harm_powers) / max(1e-12, fft_harm_powers[0])

    ax_harm.bar(h_idx - 0.15, fft_harm_powers, width=0.3, color="#ff7f0e", alpha=0.85, label="Image Cube FFT Harmonics")
    theo_harm = np.exp(-2.0 * (np.pi ** 2) * (h_idx ** 2) * (sigma_phi ** 2))
    ax_harm.plot(h_idx, theo_harm, "ko-", ms=4, lw=1.5, label="Theoretical Harmonic Envelope")
    ax_harm.set_title("Fourier Harmonic Power Distribution (5% Duty Cycle)", fontsize=11, fontweight="bold")
    ax_harm.set_xlabel("Harmonic Number ($h$)")
    ax_harm.set_ylabel("Relative Power ($P_h / P_1$)")
    ax_harm.grid(True, alpha=0.3, ls=":")
    ax_harm.legend(loc="upper right")

    plt.tight_layout()
    prof_harm_path = os.path.join(artifact_dir, "comparison_profiles_harmonics.png")
    plt.savefig(prof_harm_path)
    plt.close()
    logger.info(f"Saved profiles & harmonics to {prof_harm_path}")

    # Plot 3: 4-Panel Consistent Spatial Detection S/N Maps (in units of sigma)
    # 2. Compute 16-harmonic spatial S/N map for Image Cube FFT
    fft_harm16_spatial = np.zeros((npix_y, npix_x), dtype=np.float32)
    fund_idx_fft = int(round(f_inj / (freqs_fft[1] - freqs_fft[0])))
    for h in [1, 2, 4, 8, 16]:
        k = fund_idx_fft * h
        if k < len(freqs_fft):
            fft_harm16_spatial += power_cube[k]
    fft_amp_spatial = np.sqrt(fft_harm16_spatial)
    bg_fft = np.median(fft_amp_spatial)
    mad_fft = 1.4826 * np.median(np.abs(fft_amp_spatial - bg_fft))
    fft_snr_plane = (fft_amp_spatial - bg_fft) / max(1e-12, mad_fft)

    # 3. Compute 16-harmonic spatial S/N map for 3D NUFFT
    nufft_harm16_spatial = np.zeros((npix_x, npix_y), dtype=np.float32)
    fund_idx_nufft = int(round(f_inj / df))
    for h in [1, 2, 4, 8, 16]:
        k = fund_idx_nufft * h
        if k < Nf // 2:
            nufft_harm16_spatial += np.abs(out_buf_3d[:, :, k]) ** 2
    nufft_amp_spatial = np.sqrt(nufft_harm16_spatial)
    bg_nufft = np.median(nufft_amp_spatial)
    mad_nufft = 1.4826 * np.median(np.abs(nufft_amp_spatial - bg_nufft))
    nufft_snr_plane_raw = (nufft_amp_spatial - bg_nufft) / max(1e-12, mad_nufft)
    nufft_snr_plane = np.roll(np.roll(nufft_snr_plane_raw, npix_x // 2, axis=0), npix_y // 2, axis=1).T

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=150)
    half_w = 20
    x0, x1 = target_x - half_w, target_x + half_w
    y0, y1 = target_y - half_w, target_y + half_w

    vmax_common = 28.0

    # 1. Baseline FFA SNR Map
    im0 = axes[0].imshow(best_snr_plane_1[y0:y1, x0:x1], origin="lower", cmap="inferno", vmin=0, vmax=vmax_common, extent=[x0, x1, y0, y1])
    axes[0].plot(target_x, target_y, "co", mfc="none", ms=10, mew=2, label="Injected")
    axes[0].set_title(f"Baseline FFA (S/N Map)\nPeak = {best_snr_1:.1f}$\\sigma$", fontsize=10, fontweight="bold")
    axes[0].legend(loc="upper right", fontsize=8)

    # 2. Image Cube FFT SNR Map
    im1 = axes[1].imshow(fft_snr_plane[y0:y1, x0:x1], origin="lower", cmap="inferno", vmin=0, vmax=vmax_common, extent=[x0, x1, y0, y1])
    axes[1].plot(target_x, target_y, "co", mfc="none", ms=10, mew=2)
    axes[1].set_title(f"Image Cube FFT (S/N Map)\nPeak = {peak_snr_2_harm:.1f}$\\sigma$ (16 harm)", fontsize=10, fontweight="bold")

    # 3. 3D NUFFT SNR Map
    im2 = axes[2].imshow(nufft_snr_plane[y0:y1, x0:x1], origin="lower", cmap="inferno", vmin=0, vmax=vmax_common, extent=[x0, x1, y0, y1])
    axes[2].plot(target_x, target_y, "co", mfc="none", ms=10, mew=2)
    axes[2].set_title(f"3D NUFFT Imager (S/N Map)\nPeak = {peak_snr_3_harm:.1f}$\\sigma$ (16 harm)", fontsize=10, fontweight="bold")

    # 4. Gridded UV FFA SNR Map
    im3 = axes[3].imshow(best_snr_plane_4[y0:y1, x0:x1], origin="lower", cmap="inferno", vmin=0, vmax=vmax_common, extent=[x0, x1, y0, y1])
    axes[3].plot(target_x, target_y, "co", mfc="none", ms=10, mew=2)
    axes[3].set_title(f"Gridded UV FFA (S/N Map)\nPeak = {best_snr_4:.1f}$\\sigma$", fontsize=10, fontweight="bold")

    for ax in axes:
        ax.set_xlabel("Pixel X")
        ax.set_ylabel("Pixel Y")

    cbar = fig.colorbar(im0, ax=axes, orientation="vertical", fraction=0.015, pad=0.02)
    cbar.set_label("Detection Significance ($\\sigma$)", fontsize=10, fontweight="bold")

    plt.suptitle("Spatial Detection Cutouts: Unified Detection S/N Maps in Units of $\\sigma$ (Common Color Scale 0 - 28$\\sigma$)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    imgs_path = os.path.join(artifact_dir, "comparison_detection_images.png")
    plt.savefig(imgs_path)
    plt.close()
    logger.info(f"Saved detection images to {imgs_path}")

    # Plot 4: Performance & S/N Benchmark Bar Chart
    fig, (ax_snr, ax_time) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    method_names = ["Baseline FFA", "Image Cube FFT", "3D NUFFT", "Gridded UV FFA"]
    snrs_plot = [
        results_summary["Baseline FFA"]["peak_snr"],
        results_summary["Image Cube + FFT"]["peak_snr"],
        results_summary["3D NUFFT Imager"]["peak_snr"],
        results_summary["Gridded UV FFA"]["peak_snr"],
    ]
    times_plot = [
        results_summary["Baseline FFA"]["runtime_s"],
        results_summary["Image Cube + FFT"]["runtime_s"],
        results_summary["3D NUFFT Imager"]["runtime_s"],
        results_summary["Gridded UV FFA"]["runtime_s"],
    ]
    colors_bar = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    bars_snr = ax_snr.bar(method_names, snrs_plot, color=colors_bar, alpha=0.85, width=0.55)
    ax_snr.set_ylabel("Recovered Peak S/N ($\sigma$)", fontsize=11)
    ax_snr.set_title("Detection Sensitivity Comparison", fontsize=11, fontweight="bold")
    ax_snr.grid(True, alpha=0.3, ls=":", axis="y")
    for bar in bars_snr:
        yval = bar.get_height()
        ax_snr.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f}$\sigma$", ha="center", va="bottom", fontweight="bold")

    bars_time = ax_time.bar(method_names, times_plot, color=colors_bar, alpha=0.85, width=0.55)
    ax_time.set_ylabel("Wall-Clock Execution Time (s)", fontsize=11)
    ax_time.set_title("Computational Runtime Comparison", fontsize=11, fontweight="bold")
    ax_time.grid(True, alpha=0.3, ls=":", axis="y")
    for bar in bars_time:
        yval = bar.get_height()
        ax_time.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.2f}s", ha="center", va="bottom", fontweight="bold")

    plt.suptitle("4-Way Benchmark on Native CRACO 384x384 Grid (1000 Integrations, 253 Baselines)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    perf_path = os.path.join(artifact_dir, "comparison_snr_runtime.png")
    plt.savefig(perf_path)
    plt.close()
    logger.info(f"Saved performance benchmark chart to {perf_path}")

    # Write Text Benchmark Report
    summary_txt = os.path.join(artifact_dir, "comparison_4way_benchmark_results.txt")
    with open(summary_txt, "w") as f:
        f.write("Side-by-Side 4-Way Method Benchmark Summary (384x384 Grid)\n")
        f.write("=" * 65 + "\n")
        f.write(f"Dataset: CRACO LTR_1733-2344 (1001 time integrations, 253 baselines)\n")
        f.write(f"Injected Pulsar: P = {p_inj:.5f} s, Duty Cycle = {duty_cycle*100:.1f}%, Target Pixel = ({target_x}, {target_y})\n\n")
        f.write(f"{'Method':<22} | {'Runtime (s)':<12} | {'Recov P (s)':<12} | {'P Error (ms)':<14} | {'Peak SNR':<10}\n")
        f.write("-" * 80 + "\n")
        for mname, mres in results_summary.items():
            f.write(
                f"{mname:<22} | {mres['runtime_s']:<12.2f} | {mres['recovered_period']:<12.5f} | "
                f"{mres['period_error_ms']:<14.3f} | {mres['peak_snr']:<10.2f}\n"
            )
        f.write("=" * 80 + "\n")

    logger.info(f"Summary written to {summary_txt}")
    logger.info("=" * 75)
    logger.info("All 4 Methods Benchmarked Successfully!")
    logger.info("=" * 75)


if __name__ == "__main__":
    main()
