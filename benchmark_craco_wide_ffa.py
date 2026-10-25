# -*- coding: utf-8 -*-
"""
Benchmark script: Baseline-Domain FFA on CRACO MS at DM=0 up to 120s.
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

# Ensure local packages are in path
sys.path.insert(0, os.path.abspath("fastducc"))
sys.path.insert(0, os.path.abspath("riptide/src"))

from fastducc.baseline_ffa import search_baseline_ffa
from fastducc import ms_utils, wcs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")

def main():
    msname = "riptide/tests/data/cracoData.LTR_1733-2344.SB77974.beam01.20251015072402.calG9.uvsub.ms"
    artifact_dir = "/Users/zic006/.gemini/antigravity/brain/62efa71d-bc44-490e-ba26-4fbd9d0d085c"
    os.makedirs(artifact_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Starting CRACO Baseline FFA Benchmark (DM=0, Period 1.5s - 120s)")
    logger.info("=" * 70)

    t0 = time.time()
    res = search_baseline_ffa(
        msname_or_data=msname,
        period_min=1.5,
        period_max=120.0,
        bins_min=16,
        bins_max=24,
        dm=0.0,
        collapse_channels=True,
        segment_duration_s=30.0,
        npix_x=64,
        npix_y=64,
        pixsize_x=22.0 / 206265.0,
        pixsize_y=22.0 / 206265.0,
        snr_thresh=5.0,
        save_best_dirty_cube=True,
        nthreads=0
    )
    t_total = time.time() - t0

    logger.info(f"Total benchmark wall-clock time: {t_total:.2f} s")
    logger.info(f"Number of periods evaluated: {len(res.periods)}")
    logger.info(f"Average time per trial period: {t_total / max(1, len(res.periods)) * 1000:.2f} ms")

    best_idx = int(np.argmax(res.snr_periodogram))
    best_p = float(res.periods[best_idx])
    best_snr = float(res.snr_periodogram[best_idx])
    logger.info(f"Global peak: Period = {best_p:.6f} s, SNR = {best_snr:.2f} sigma")

    # Inspect top candidates
    logger.info(f"Total candidates with SNR >= 5.0: {len(res.candidates)}")
    for i, c in enumerate(res.candidates[:10]):
        ra_str = f"{c.ra_deg:.4f} deg" if c.ra_deg is not None else "N/A"
        dec_str = f"{c.dec_deg:.4f} deg" if c.dec_deg is not None else "N/A"
        logger.info(
            f"  Candidate #{i+1:2d}: P={c.period:.5f} s, SNR={c.snr:.2f}, "
            f"pixel=({c.x:2d}, {c.y:2d}), RA={ra_str}, Dec={dec_str}, width={c.width_bins} bins, phase={c.phase_bin}"
        )

    # 1. Plot Full Periodogram
    plt.figure(figsize=(12, 5), dpi=150)
    plt.plot(res.periods, res.snr_periodogram, color="#1f77b4", lw=1.2, label=f"Baseline FFA (DM=0)")
    plt.axhline(5.0, color="crimson", ls="--", lw=1.0, label="5.0-sigma Threshold")
    plt.scatter([best_p], [best_snr], color="gold", edgecolor="black", s=90, zorder=5,
                label=f"Peak: P={best_p:.4f}s (SNR={best_snr:.2f})")
    plt.title(f"CRACO LTR_1733-2344 Baseline FFA Periodogram (1.5s - 120s)", fontsize=13, fontweight="bold")
    plt.xlabel("Trial Period (s)", fontsize=11)
    plt.ylabel("Detection S/N (MAD-normalized)", fontsize=11)
    plt.xlim(1.5, 120.0)
    plt.grid(True, alpha=0.3, ls=":")
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    periodogram_path = os.path.join(artifact_dir, "craco_wide_periodogram.png")
    plt.savefig(periodogram_path)
    plt.close()
    logger.info(f"Saved periodogram to {periodogram_path}")

    # 2. Extract profile and dirty images for best candidate
    if res.best_candidate is not None and res.best_dirty_cube is not None:
        cand = res.best_candidate
        cube = res.best_dirty_cube  # (nbins, ny, nx)
        nbins, ny, nx = cube.shape

        prof = cube[:, cand.y, cand.x]
        phase_axis = np.linspace(0, 1, nbins, endpoint=False)

        # Plot Recovered Profile
        plt.figure(figsize=(8, 4), dpi=150)
        # Duplicate 2 pulses for clarity
        phase_2 = np.concatenate([phase_axis, phase_axis + 1.0])
        prof_2 = np.concatenate([prof, prof])
        plt.step(phase_2, prof_2, where="mid", color="#2ca02c", lw=1.8)
        plt.axvspan(phase_axis[cand.phase_bin], phase_axis[(cand.phase_bin + cand.width_bins) % nbins] if (cand.phase_bin + cand.width_bins) < nbins else 1.0,
                    alpha=0.2, color="orange", label=f"Boxcar Match (w={cand.width_bins} bins)")
        plt.title(f"Peak Candidate Folded Profile: P = {cand.period:.5f} s (SNR = {cand.snr:.2f})", fontsize=12, fontweight="bold")
        plt.xlabel("Rotational Phase", fontsize=11)
        plt.ylabel("Flux (Jy/beam arbitrary)", fontsize=11)
        plt.grid(True, alpha=0.3, ls=":")
        plt.legend(loc="upper right")
        plt.tight_layout()
        profile_path = os.path.join(artifact_dir, "craco_wide_pulse_profile.png")
        plt.savefig(profile_path)
        plt.close()
        logger.info(f"Saved pulse profile to {profile_path}")

        # On-pulse vs Off-pulse images
        on_pulse_img = cube[cand.phase_bin]
        off_bin = (cand.phase_bin + nbins // 2) % nbins
        off_pulse_img = cube[off_bin]

        fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=150)
        im0 = axes[0].imshow(on_pulse_img, origin="lower", cmap="viridis")
        axes[0].plot(cand.x, cand.y, "rx", ms=10, mew=2, label=f"Cand ({cand.x},{cand.y})")
        axes[0].set_title(f"On-Pulse Dirty Image (Phase bin {cand.phase_bin})", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("Pixel X")
        axes[0].set_ylabel("Pixel Y")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        axes[0].legend(loc="upper right")

        im1 = axes[1].imshow(off_pulse_img, origin="lower", cmap="viridis")
        axes[1].set_title(f"Off-Pulse Dirty Image (Phase bin {off_bin})", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Pixel X")
        axes[1].set_ylabel("Pixel Y")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        dirty_img_path = os.path.join(artifact_dir, "craco_wide_dirty_images.png")
        plt.savefig(dirty_img_path)
        plt.close()
        logger.info(f"Saved dirty images to {dirty_img_path}")

        # Combined 4-Panel Summary Diagnostic
        fig = plt.figure(figsize=(14, 10), dpi=150)
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])

        # Top: Wide Periodogram
        ax_p = fig.add_subplot(gs[0, :])
        ax_p.plot(res.periods, res.snr_periodogram, color="#1f77b4", lw=1.2)
        ax_p.axhline(5.0, color="crimson", ls="--", lw=1.0, label="5-sigma")
        ax_p.scatter([best_p], [best_snr], color="gold", edgecolor="black", s=80, zorder=5,
                     label=f"Peak: P={best_p:.4f}s, SNR={best_snr:.2f}")
        ax_p.set_title("CRACO Baseline-Domain FFA Periodogram: P in [1.5, 120.0] s", fontsize=12, fontweight="bold")
        ax_p.set_xlabel("Trial Period (s)")
        ax_p.set_ylabel("Detection S/N")
        ax_p.set_xlim(1.5, 120.0)
        ax_p.grid(True, alpha=0.3, ls=":")
        ax_p.legend(loc="upper right")

        # Bottom Left: Folded Profile
        ax_prof = fig.add_subplot(gs[1, 0])
        ax_prof.step(phase_2, prof_2, where="mid", color="#2ca02c", lw=1.8)
        ax_prof.set_title(f"Peak Candidate Profile (P = {cand.period:.5f} s)", fontsize=11, fontweight="bold")
        ax_prof.set_xlabel("Pulse Phase")
        ax_prof.set_ylabel("Intensity")
        ax_prof.grid(True, alpha=0.3, ls=":")

        # Bottom Right: On-pulse image with WCS annotation
        ax_im = fig.add_subplot(gs[1, 1])
        im_plot = ax_im.imshow(on_pulse_img, origin="lower", cmap="inferno")
        ax_im.plot(cand.x, cand.y, "co", mfc="none", ms=12, mew=2, label=f"Candidate: SNR={cand.snr:.2f}")
        ax_im.set_title(f"DUCC0 wgridder On-Pulse Image (bin {cand.phase_bin})", fontsize=11, fontweight="bold")
        ax_im.set_xlabel("Pixel X")
        ax_im.set_ylabel("Pixel Y")
        plt.colorbar(im_plot, ax=ax_im, fraction=0.046, pad=0.04)
        ax_im.legend(loc="upper right")

        plt.suptitle("CRACO LTR_1733-2344 Baseline FFA Integration Benchmark (fastducc + riptide)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        summary_path = os.path.join(artifact_dir, "craco_wide_summary.png")
        plt.savefig(summary_path)
        plt.close()
        logger.info(f"Saved summary to {summary_path}")

    # Output text benchmark results
    benchmark_txt_path = os.path.join(artifact_dir, "craco_wide_benchmark_results.txt")
    with open(benchmark_txt_path, "w") as f:
        f.write("CRACO Baseline FFA Benchmark Summary\n")
        f.write("=" * 45 + "\n")
        f.write(f"MS Name: {msname}\n")
        f.write(f"Period Range: [1.5, 120.0] s\n")
        f.write(f"Total Periods Evaluated: {len(res.periods)}\n")
        f.write(f"Total Wall-Clock Time: {t_total:.2f} s\n")
        f.write(f"Throughput: {len(res.periods) / t_total:.2f} periods/sec\n")
        f.write(f"Global Peak Period: {best_p:.6f} s\n")
        f.write(f"Global Peak SNR: {best_snr:.2f} sigma\n")
        f.write(f"Total Candidates >= 5.0 sigma: {len(res.candidates)}\n\n")
        f.write("Top Candidates:\n")
        for i, c in enumerate(res.candidates[:15]):
            f.write(
                f"  #{i+1:2d}: P={c.period:10.5f}s, SNR={c.snr:6.2f}, "
                f"pixel=({c.x:2d}, {c.y:2d}), RA={c.ra_deg}, Dec={c.dec_deg}, width={c.width_bins}, phase={c.phase_bin}\n"
            )
    logger.info(f"Benchmark summary written to {benchmark_txt_path}")
    logger.info("=" * 70)
    logger.info("Benchmark Complete!")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
