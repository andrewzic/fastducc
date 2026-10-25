# -*- coding: utf-8 -*-
"""
Benchmark Script: Coupled Wideband Dedispersion & UVW Tracking for Baseline-Domain FFA in FastDUCC.

Evaluates and compares:
1. Traditional Channel-Collapsed FFA (collapse_channels=True, nsubbands=1)
2. Option 2: Coarse Subbanding FFA (collapse_channels=True, nsubbands=16, exact_uvw=True)
3. Option 1: Full Pseudo-Baselines FFA with Exact UVW Tracking (collapse_channels=False, exact_uvw=True)
4. Option 1: Pseudo-Baselines without Dispersion Baseline Migration (collapse_channels=False, exact_uvw=False)

Evaluated on real CRACO visibility data (253 baselines, 288 channels) with an injected dispersed
pulsar (DM = 50 pc/cm^3, Period = 2.5s) at an off-axis sky position.
"""

from __future__ import annotations

import os
import sys
import time
import math
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure local paths are accessible
sys.path.insert(0, os.path.abspath("fastducc"))
sys.path.insert(0, os.path.abspath("riptide/src"))

from fastducc.baseline_ffa import search_baseline_ffa
from fastducc import ms_utils, imaging, dedisp
from fastducc.constants import c as c_light

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_coupled_dedisp")


def run_benchmark():
    msname = "riptide/tests/data/cracoData.LTR_1733-2344.SB77974.beam01.20251015072402.calG9.uvsub.ms"
    artifact_dir = "/Users/zic006/.gemini/antigravity/brain/c963d92d-3f21-4d92-921e-59ed18a8bda1"
    os.makedirs(artifact_dir, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Coupled Wideband Dedispersion & UVW Tracking Benchmark for Baseline FFA")
    logger.info("=" * 80)

    # 1. Load chunk of CRACO data: 253 baselines, 288 channels, 600 timestamps (~66.4 seconds)
    nt_chunk = 600
    logger.info(f"Loading {nt_chunk} time samples from CRACO MS: {msname}")
    t0_load = time.time()
    u_times, chan_freq, dt, vis_3d, wgt_3d, uvw_3d = imaging.load_chunk_data(
        msname, start_time_idx=0, end_time_idx=nt_chunk, corr_mode='average', average_correlations=True
    )
    t_load = time.time() - t0_load
    nbl, nchan, nt = vis_3d.shape
    total_dur_s = nt * dt
    logger.info(f"Loaded {nbl} baselines, {nchan} channels, {nt} time steps ({total_dur_s:.2f}s, dt={dt:.4f}s) in {t_load:.2f}s")
    logger.info(f"Frequency band: {chan_freq[0]/1e6:.2f} to {chan_freq[-1]/1e6:.2f} MHz (Total BW: {(chan_freq[-1]-chan_freq[0])/1e6:.2f} MHz)")

    # 2. Inject Dispersed Off-Axis Pulsar
    true_period = 2.50000  # seconds
    true_dm = 50.0         # pc/cm^3
    pulse_fwhm_frac = 0.04
    pulse_amp = 4.0        # Jy/beam equivalent
    pulse_phase_0 = 0.35

    # Off-axis location: +12 pixels in x, -8 pixels in y relative to image center (32, 32)
    npix_x, npix_y = 64, 64
    pixsize_x = 22.0 / 206265.0  # radians
    pixsize_y = 22.0 / 206265.0
    dx_inj = +12.0
    dy_inj = -8.0
    l0 = dx_inj * pixsize_x
    m0 = dy_inj * pixsize_y
    n0_minus_1 = np.sqrt(max(0.0, 1.0 - l0**2 - m0**2)) - 1.0

    inj_x = npix_x // 2 + int(round(dx_inj))
    inj_y = npix_y // 2 + int(round(dy_inj))

    theta_rad = math.sqrt(l0**2 + m0**2)
    theta_arcmin = theta_rad * (180.0 / math.pi) * 60.0
    logger.info(f"Injecting pulsar: P={true_period:.5f}s, DM={true_dm:.2f} pc/cm^3")
    logger.info(f"Target position: offset ({dx_inj:+.1f}, {dy_inj:+.1f}) pix -> pixel ({inj_x}, {inj_y}), radius {theta_arcmin:.2f} arcmin")

    delays = dedisp.get_dm_delays(chan_freq, true_dm, dt, align_to="fmax")
    max_delay = int(np.max(delays))
    logger.info(f"Dispersion delay sweep across band: {max_delay} samples ({max_delay * dt * 1e3:.1f} ms)")

    # Standard interferometric Fourier kernel: exp(-2pi i (u*l + v*m + w*(n-1)))
    # Because vis2dirty computes inverse transform with exp(+2pi i (u*l + v*m)),
    # a point source at (l0, m0) must have exp(-2pi i (u*l0 + v*m0))
    vis_injected = vis_3d.copy()
    t_obs = np.arange(nt) * dt
    for c in range(nchan):
        d_c = delays[c]
        t_emit = t_obs - d_c * dt
        rot_phase = (t_emit / true_period) % 1.0
        pulse = pulse_amp * np.exp(-0.5 * ((rot_phase - pulse_phase_0) / pulse_fwhm_frac)**2).astype(np.float32)

        lam = c_light / chan_freq[c]
        geom_phase = 2.0 * np.pi * (uvw_3d[:, 0, :] * l0 + uvw_3d[:, 1, :] * m0 + uvw_3d[:, 2, :] * n0_minus_1) / lam
        vis_injected[:, c, :] += pulse[None, :] * np.exp(-1j * geom_phase)

    # 3. Configurations to Benchmark
    configs = [
        {
            "name": "Config 1: Traditional Channel-Collapsed (Nsub=1)",
            "short": "Collapsed (Nsub=1)",
            "collapse_channels": True,
            "nsubbands": 1,
            "exact_uvw": False,
            "color": "#d62728",  # Red
        },
        {
            "name": "Config 2: Option 2 Coarse Subbanding (Nsub=16, Exact UVW)",
            "short": "Option 2 (Nsub=16)",
            "collapse_channels": True,
            "nsubbands": 16,
            "exact_uvw": True,
            "color": "#ff7f0e",  # Orange
        },
        {
            "name": "Config 3: Option 1 Pseudo-Baselines (Nchan=288, Exact UVW)",
            "short": "Option 1 Exact (Nchan=288)",
            "collapse_channels": False,
            "nsubbands": 1,
            "exact_uvw": True,
            "color": "#2ca02c",  # Green
        },
        {
            "name": "Config 4: Option 1 Pseudo-Baselines (Nchan=288, No-Exact UVW)",
            "short": "Option 1 No-Exact (Nchan=288)",
            "collapse_channels": False,
            "nsubbands": 1,
            "exact_uvw": False,
            "color": "#1f77b4",  # Blue
        },
    ]

    results = []
    p_min, p_max = 2.20, 2.80
    seg_dur = 15.0  # Option A segment duration: 15s

    for cfg in configs:
        logger.info("-" * 70)
        logger.info(f"Running {cfg['name']}...")
        t0_run = time.time()
        res = search_baseline_ffa(
            (vis_injected.copy(), wgt_3d.copy(), uvw_3d.copy(), chan_freq, dt),
            period_min=p_min,
            period_max=p_max,
            bins_min=16,
            bins_max=16,
            dm=true_dm,
            collapse_channels=cfg["collapse_channels"],
            nsubbands=cfg["nsubbands"],
            exact_uvw=cfg["exact_uvw"],
            segment_duration_s=seg_dur,
            npix_x=npix_x,
            npix_y=npix_y,
            pixsize_x=pixsize_x,
            pixsize_y=pixsize_y,
            snr_thresh=4.0,
            save_best_dirty_cube=True,
            nthreads=0
        )
        t_run = time.time() - t0_run

        peak_idx = int(np.argmax(res.snr_periodogram))
        peak_snr = float(res.snr_periodogram[peak_idx])
        peak_p = float(res.periods[peak_idx])
        cand = res.best_candidate

        cand_x = cand.x if cand is not None else -1
        cand_y = cand.y if cand is not None else -1
        cand_w = cand.width_bins if cand is not None else -1

        logger.info(
            f"Finished {cfg['short']}: Wall time = {t_run:.3f} s, Peak SNR = {peak_snr:.2f} sigma, "
            f"Period = {peak_p:.5f} s, Peak Pixel = ({cand_x}, {cand_y}), Width = {cand_w} bins"
        )

        results.append({
            "cfg": cfg,
            "res": res,
            "time_s": t_run,
            "peak_snr": peak_snr,
            "peak_p": peak_p,
            "candidate": cand
        })

    # Benchmark Summary Table
    ref_snr = results[2]["peak_snr"]  # Config 3: Option 1 Exact
    logger.info("=" * 80)
    logger.info("BENCHMARK COMPARISON SUMMARY:")
    logger.info(f"{'Configuration':<45} | {'Wall Time':<10} | {'Peak SNR':<10} | {'Rel. S/N':<10} | {'Pixel':<10}")
    logger.info("-" * 92)
    for r in results:
        cfg = r["cfg"]
        rel_snr = (r["peak_snr"] / ref_snr) * 100.0
        cand_pos = f"({r['candidate'].x},{r['candidate'].y})" if r['candidate'] is not None else "None"
        logger.info(
            f"{cfg['name']:<45} | {r['time_s']:8.3f} s | {r['peak_snr']:8.2f} \u03c3 | {rel_snr:8.1f} % | {cand_pos:<10}"
        )
    logger.info("=" * 80)

    # 4. Generate Diagnostic Figures

    # Figure 1: Combined 4-Panel Master Summary
    fig = plt.figure(figsize=(16, 12), dpi=180)
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.25)

    # Subplot A: Periodogram Comparison
    ax0 = fig.add_subplot(gs[0, 0])
    for r in results:
        cfg = r["cfg"]
        res = r["res"]
        ax0.plot(res.periods, res.snr_periodogram, lw=1.8, color=cfg["color"], label=f"{cfg['short']}: Peak={r['peak_snr']:.1f}\u03c3 ({r['time_s']:.2f}s)")
    ax0.axvline(true_period, color="black", ls=":", alpha=0.8, lw=1.5, label=f"True Period ({true_period:.4f}s)")
    ax0.set_title("(A) Detection S/N vs Trial Period", fontsize=12, fontweight="bold")
    ax0.set_xlabel("Trial Period (s)", fontsize=11)
    ax0.set_ylabel("Detection S/N (MAD-normalized)", fontsize=11)
    ax0.grid(True, alpha=0.3, ls=":")
    ax0.legend(frameon=True, loc="upper right", fontsize=9)

    # Subplot B: SNR Loss & Runtime Tradeoff
    ax1 = fig.add_subplot(gs[0, 1])
    names = [r["cfg"]["short"] for r in results]
    snrs = [r["peak_snr"] for r in results]
    times = [r["time_s"] for r in results]
    colors = [r["cfg"]["color"] for r in results]

    x_pos = np.arange(len(names))
    width = 0.35
    ax1_twin = ax1.twinx()

    b1 = ax1.bar(x_pos - width/2, snrs, width, label="Peak SNR (\u03c3)", color=colors, alpha=0.85, edgecolor="black")
    b2 = ax1_twin.bar(x_pos + width/2, times, width, label="Runtime (s)", color="gray", alpha=0.5, hatch="//", edgecolor="black")

    ax1.set_ylabel("Peak Detection S/N (\u03c3)", fontsize=11, color="black")
    ax1_twin.set_ylabel("Wall-Clock Execution Time (s)", fontsize=11, color="dimgray")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax1.set_title("(B) Sensitivity Recovery vs Computational Cost", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.2, ls=":", axis="y")

    for i, snr in enumerate(snrs):
        ax1.text(x_pos[i] - width/2, snr + 0.8, f"{snr:.1f}\u03c3", ha="center", fontsize=8, fontweight="bold")
    for i, t_val in enumerate(times):
        ax1_twin.text(x_pos[i] + width/2, t_val + 0.05, f"{t_val:.2f}s", ha="center", fontsize=8)

    # Subplot C: Spatial SNR Dirty Plane for Option 1 Exact
    ax2 = fig.add_subplot(gs[1, 0])
    snr_plane_exact = results[2]["res"].best_snr_plane
    im_c = ax2.imshow(snr_plane_exact, origin="lower", cmap="inferno")
    ax2.plot(inj_x, inj_y, "m+", ms=14, mew=2, label=f"Injected Target ({inj_x},{inj_y})")
    best_c = results[2]["candidate"]
    if best_c is not None:
        ax2.plot(best_c.x, best_c.y, "gx", ms=12, mew=2, label=f"Option 1 Peak ({best_c.x},{best_c.y})")
    ax2.set_title(f"(C) Option 1 Spatial S/N Map: Peak={results[2]['peak_snr']:.1f}\u03c3 at ({inj_x},{inj_y})", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Pixel X", fontsize=11)
    ax2.set_ylabel("Pixel Y", fontsize=11)
    plt.colorbar(im_c, ax=ax2, fraction=0.046, pad=0.04, label="Detection S/N (\u03c3)")
    ax2.legend(loc="upper left", fontsize=9)

    # Subplot D: Folded Pulse Profile Comparison
    ax3 = fig.add_subplot(gs[1, 1])
    nbins = 16
    phase_axis = np.linspace(0, 1, nbins, endpoint=False)
    phase_2 = np.concatenate([phase_axis, phase_axis + 1.0])

    for r in results:
        cfg = r["cfg"]
        cube = r["res"].best_dirty_cube
        if cube is not None:
            # Profile at detected candidate position
            cand_x = r["candidate"].x if r["candidate"] is not None else inj_x
            cand_y = r["candidate"].y if r["candidate"] is not None else inj_y
            prof = cube[:, cand_y, cand_x]
            prof_norm = (prof - np.mean(prof))
            prof_norm = prof_norm / (np.max(prof_norm) if np.max(prof_norm) > 0 else 1.0)
            prof_2 = np.concatenate([prof_norm, prof_norm])
            ax3.step(phase_2, prof_2, where="mid", lw=1.8, color=cfg["color"], label=f"{cfg['short']} at ({cand_x},{cand_y})")

    ax3.set_title("(D) Folded Pulse Profile at Candidate Position", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Rotational Phase", fontsize=11)
    ax3.set_ylabel("Normalized Amplitude", fontsize=11)
    ax3.grid(True, alpha=0.3, ls=":")
    ax3.legend(frameon=True, loc="upper right", fontsize=9)

    plt.suptitle(
        f"FastDUCC Baseline FFA: Coupled Wideband Dedispersion Benchmark\n"
        f"CRACO LTR_1733-2344 (253 Baselines, 288 Chans, DM=50 pc/cm\u00b3, P=2.5s, Off-Axis \u03b8={theta_arcmin:.1f}')",
        fontsize=14, fontweight="bold", y=0.98
    )

    summary_plot_path = os.path.join(artifact_dir, "craco_coupled_dedisp_benchmark.png")
    plt.savefig(summary_plot_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved master benchmark diagnostic figure to: {summary_plot_path}")

    # Figure 2: Detailed Cutout & PSF Comparison
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), dpi=180)
    for i, r in enumerate(results):
        ax = axes[i]
        cube = r["res"].best_dirty_cube
        if cube is not None and r["candidate"] is not None:
            cand = r["candidate"]
            on_pulse = cube[cand.phase_bin]
            half = 10
            y0, y1 = max(0, cand.y - half), min(npix_y, cand.y + half)
            x0, x1 = max(0, cand.x - half), min(npix_x, cand.x + half)
            cut = on_pulse[y0:y1, x0:x1]
            im = ax.imshow(cut, origin="lower", cmap="viridis", extent=[x0, x1, y0, y1])
            ax.plot(inj_x, inj_y, "m+", ms=12, mew=2, label=f"Inj ({inj_x},{inj_y})")
            ax.plot(cand.x, cand.y, "rx", ms=10, mew=2, label=f"Peak ({cand.x},{cand.y})")
            ax.set_title(f"{r['cfg']['short']}\nPeak S/N = {r['peak_snr']:.1f}\u03c3", fontsize=10, fontweight="bold")
            ax.set_xlabel("Pixel X")
            if i == 0:
                ax.set_ylabel("Pixel Y")
                ax.legend(loc="upper right", fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle("On-Pulse Cutout (20x20 Pixels) Centered at Injected Off-Axis Target", fontsize=12, fontweight="bold")
    cutout_plot_path = os.path.join(artifact_dir, "craco_coupled_dedisp_cutouts.png")
    plt.savefig(cutout_plot_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved cutout diagnostic figure to: {cutout_plot_path}")

    logger.info("Benchmark complete!")


if __name__ == "__main__":
    run_benchmark()
