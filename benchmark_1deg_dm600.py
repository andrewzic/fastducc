# -*- coding: utf-8 -*-
"""
Benchmark Script: Coupled Wideband Dedispersion & UVW Tracking at 1 Degree Offset and DM 600 pc/cm^3.

Tests extreme chromatic smearing and dispersion sweep Earth rotation displacement:
- Field offset: 1.0 degree (60 arcmin) from phase center
- Dispersion Measure: DM = 600 pc/cm^3 (2.16s dispersion delay sweep, ~20 time samples)
- Grid size: 384 x 384 (2.35 degree full field of view)
- Configurations:
  1. Traditional Channel-Collapsed (Nsub=1)
  2. Option 2: Coarse Subbanding (Nsub=16, Exact UVW)
  3. Option 2: Fine Subbanding (Nsub=64, Exact UVW)
  4. Option 1: Full Pseudo-Baselines (Nchan=288, Exact UVW)
  5. Option 1: Pseudo-Baselines without Dispersion Baseline Migration (Nchan=288, No-Exact UVW)
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
logger = logging.getLogger("benchmark_1deg_dm600")


def run_benchmark():
    msname = "riptide/tests/data/cracoData.LTR_1733-2344.SB77974.beam01.20251015072402.calG9.uvsub.ms"
    artifact_dir = "/Users/zic006/.gemini/antigravity/brain/c963d92d-3f21-4d92-921e-59ed18a8bda1"
    os.makedirs(artifact_dir, exist_ok=True)

    logger.info("=" * 80)
    logger.info("CRACO Baseline FFA Benchmark: 1.0 Degree Off-Axis with DM = 600 pc/cm^3")
    logger.info("=" * 80)

    # 1. Load chunk of CRACO data: 253 baselines, 288 channels, 450 timestamps (~50.0 seconds)
    nt_chunk = 450
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

    # 2. Inject Dispersed Off-Axis Pulsar at 1.0 Degree
    true_period = 2.50000  # seconds
    true_dm = 600.0        # pc/cm^3
    pulse_fwhm_frac = 0.04
    pulse_amp = 6.0        # Jy/beam equivalent
    pulse_phase_0 = 0.35

    # Target position: 1.0 degree offset along x-axis
    npix_x, npix_y = 384, 384
    pixsize_x = 22.0 / 206265.0  # radians
    pixsize_y = 22.0 / 206265.0
    dx_inj = +164.0  # +164 pixels * 22 arcsec = 3608 arcsec = 1.002 degrees
    dy_inj = 0.0
    l0 = dx_inj * pixsize_x
    m0 = dy_inj * pixsize_y
    n0_minus_1 = np.sqrt(max(0.0, 1.0 - l0**2 - m0**2)) - 1.0

    inj_x = npix_x // 2 + int(round(dx_inj))  # 192 + 164 = 356
    inj_y = npix_y // 2 + int(round(dy_inj))  # 192

    theta_deg = math.sqrt(l0**2 + m0**2) * (180.0 / math.pi)
    logger.info(f"Injecting pulsar: P={true_period:.5f}s, DM={true_dm:.2f} pc/cm^3")
    logger.info(f"Target position: offset ({dx_inj:+.1f}, {dy_inj:+.1f}) pix -> pixel ({inj_x}, {inj_y}), radius {theta_deg:.3f} degrees ({theta_deg*60:.1f} arcmin)")

    delays = dedisp.get_dm_delays(chan_freq, true_dm, dt, align_to="fmax")
    max_delay = int(np.max(delays))
    sweep_dur_s = max_delay * dt
    logger.info(f"Dispersion delay sweep across band: {max_delay} samples ({sweep_dur_s:.3f} s)")

    # Standard interferometric Fourier kernel: exp(-2pi i (u*l + v*m + w*(n-1)))
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
            "name": "Config 3: Option 2 Fine Subbanding (Nsub=64, Exact UVW)",
            "short": "Option 2 (Nsub=64)",
            "collapse_channels": True,
            "nsubbands": 64,
            "exact_uvw": True,
            "color": "#9467bd",  # Purple
        },
        {
            "name": "Config 4: Option 1 Pseudo-Baselines (Nchan=288, Exact UVW)",
            "short": "Option 1 Exact (Nchan=288)",
            "collapse_channels": False,
            "nsubbands": 1,
            "exact_uvw": True,
            "color": "#2ca02c",  # Green
        },
        {
            "name": "Config 5: Option 1 Pseudo-Baselines (Nchan=288, No-Exact UVW)",
            "short": "Option 1 No-Exact (Nchan=288)",
            "collapse_channels": False,
            "nsubbands": 1,
            "exact_uvw": False,
            "color": "#1f77b4",  # Blue
        },
    ]

    results = []
    p_min, p_max = 2.45, 2.55
    seg_dur = 12.0  # Option A segment duration: 12s

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
    ref_snr = results[3]["peak_snr"]  # Config 4: Option 1 Exact
    logger.info("=" * 80)
    logger.info("BENCHMARK COMPARISON SUMMARY (1.0 Degree Offset, DM=600):")
    logger.info(f"{'Configuration':<45} | {'Wall Time':<10} | {'Peak SNR':<10} | {'Rel. S/N':<10} | {'Pixel':<10} | {'Offset Error':<12}")
    logger.info("-" * 105)
    for r in results:
        cfg = r["cfg"]
        rel_snr = (r["peak_snr"] / ref_snr) * 100.0
        cand = r['candidate']
        if cand is not None:
            cand_pos = f"({cand.x},{cand.y})"
            err_pix = math.sqrt((cand.x - inj_x)**2 + (cand.y - inj_y)**2)
            err_str = f"{err_pix:.1f} pix"
        else:
            cand_pos = "None"
            err_str = "N/A"
        logger.info(
            f"{cfg['name']:<45} | {r['time_s']:8.3f} s | {r['peak_snr']:8.2f} \u03c3 | {rel_snr:8.1f} % | {cand_pos:<10} | {err_str:<12}"
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
        ax0.plot(res.periods, res.snr_periodogram, lw=1.8, color=cfg["color"],
                 label=f"{cfg['short']}: Peak={r['peak_snr']:.1f}\u03c3 ({r['time_s']:.2f}s)")
    ax0.axvline(true_period, color="black", ls=":", alpha=0.8, lw=1.5, label=f"True Period ({true_period:.4f}s)")
    ax0.set_title("(A) Detection S/N vs Trial Period at 1.0\u00b0 Off-Axis", fontsize=12, fontweight="bold")
    ax0.set_xlabel("Trial Period (s)", fontsize=11)
    ax0.set_ylabel("Detection S/N (MAD-normalized)", fontsize=11)
    ax0.grid(True, alpha=0.3, ls=":")
    ax0.legend(frameon=True, loc="upper right", fontsize=8.5)

    # Subplot B: Sensitivity Recovery vs Computational Cost
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
    ax1.set_xticklabels(names, rotation=20, ha="right", fontsize=8.5)
    ax1.set_title("(B) S/N Recovery vs Computational Runtime", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.2, ls=":", axis="y")

    for i, snr in enumerate(snrs):
        ax1.text(x_pos[i] - width/2, snr + 1.5, f"{snr:.1f}\u03c3", ha="center", fontsize=8, fontweight="bold")
    for i, t_val in enumerate(times):
        ax1_twin.text(x_pos[i] + width/2, t_val + 0.1, f"{t_val:.2f}s", ha="center", fontsize=8)

    # Subplot C: Full 384x384 Spatial Dirty SNR Map (Option 1 Exact)
    ax2 = fig.add_subplot(gs[1, 0])
    snr_plane_exact = results[3]["res"].best_snr_plane
    im_c = ax2.imshow(snr_plane_exact, origin="lower", cmap="inferno")
    ax2.plot(npix_x//2, npix_y//2, "w+", ms=10, mew=1.5, label="Phase Centre (192, 192)")
    ax2.plot(inj_x, inj_y, "c+", ms=14, mew=2, label=f"Injected 1.0\u00b0 Target ({inj_x},{inj_y})")
    best_c = results[3]["candidate"]
    if best_c is not None:
        ax2.plot(best_c.x, best_c.y, "gx", ms=12, mew=2, label=f"Option 1 Peak ({best_c.x},{best_c.y})")
    ax2.set_title(f"(C) Full 384x384 Field (2.35\u00b0): Option 1 Peak={results[3]['peak_snr']:.1f}\u03c3", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Pixel X (22\"/pix)", fontsize=11)
    ax2.set_ylabel("Pixel Y (22\"/pix)", fontsize=11)
    plt.colorbar(im_c, ax=ax2, fraction=0.046, pad=0.04, label="Detection S/N (\u03c3)")
    ax2.legend(loc="upper left", fontsize=8.5)

    # Subplot D: Folded Pulse Profile Comparison
    ax3 = fig.add_subplot(gs[1, 1])
    nbins = 16
    phase_axis = np.linspace(0, 1, nbins, endpoint=False)
    phase_2 = np.concatenate([phase_axis, phase_axis + 1.0])

    for r in results:
        cfg = r["cfg"]
        cube = r["res"].best_dirty_cube
        if cube is not None:
            cand = r["candidate"]
            cx = cand.x if cand is not None else inj_x
            cy = cand.y if cand is not None else inj_y
            prof = cube[:, cy, cx]
            prof_norm = (prof - np.mean(prof))
            prof_norm = prof_norm / (np.max(prof_norm) if np.max(prof_norm) > 0 else 1.0)
            prof_2 = np.concatenate([prof_norm, prof_norm])
            ax3.step(phase_2, prof_2, where="mid", lw=1.8, color=cfg["color"],
                     label=f"{cfg['short']} at ({cx},{cy})")

    ax3.set_title("(D) Folded Pulse Profile at Detected Candidates", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Rotational Phase", fontsize=11)
    ax3.set_ylabel("Normalized Amplitude", fontsize=11)
    ax3.grid(True, alpha=0.3, ls=":")
    ax3.legend(frameon=True, loc="upper right", fontsize=8.5)

    plt.suptitle(
        f"FastDUCC Baseline FFA: Extreme Chromatic & Dispersion Benchmark\n"
        f"CRACO LTR_1733-2344 (253 Baselines, 288 Chans, DM=600 pc/cm\u00b3, P=2.5s, Off-Axis \u03b8=1.002\u00b0 / 60.1')",
        fontsize=14, fontweight="bold", y=0.98
    )

    summary_plot_path = os.path.join(artifact_dir, "craco_1deg_dm600_benchmark.png")
    plt.savefig(summary_plot_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved master 1-deg benchmark diagnostic figure to: {summary_plot_path}")

    # Figure 2: Detailed Cutout & PSF Comparison (5 panels)
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), dpi=180)
    for i, r in enumerate(results):
        ax = axes[i]
        cube = r["res"].best_dirty_cube
        if cube is not None and r["candidate"] is not None:
            cand = r["candidate"]
            on_pulse = cube[cand.phase_bin]
            half = 14
            y0, y1 = max(0, inj_y - half), min(npix_y, inj_y + half)
            x0, x1 = max(0, inj_x - half), min(npix_x, inj_x + half)
            cut = on_pulse[y0:y1, x0:x1]
            im = ax.imshow(cut, origin="lower", cmap="viridis", extent=[x0, x1, y0, y1])
            ax.plot(inj_x, inj_y, "m+", ms=12, mew=2, label=f"Inj ({inj_x},{inj_y})")
            ax.plot(cand.x, cand.y, "rx", ms=10, mew=2, label=f"Peak ({cand.x},{cand.y})")
            ax.set_title(f"{r['cfg']['short']}\nPeak S/N = {r['peak_snr']:.1f}\u03c3", fontsize=9.5, fontweight="bold")
            ax.set_xlabel("Pixel X")
            if i == 0:
                ax.set_ylabel("Pixel Y")
                ax.legend(loc="upper right", fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle("On-Pulse Cutout (28x28 Pixels) Centered at 1.0\u00b0 Injected Target (356, 192)", fontsize=12, fontweight="bold")
    cutout_plot_path = os.path.join(artifact_dir, "craco_1deg_dm600_cutouts.png")
    plt.savefig(cutout_plot_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved 1-deg cutout diagnostic figure to: {cutout_plot_path}")

    logger.info("1-deg DM 600 benchmark complete!")


if __name__ == "__main__":
    run_benchmark()
