# -*- coding: utf-8 -*-
"""
fastducc unified CLI dispatcher.

Provides a single modern entrypoint with subcommands and backwards-compatible
direct execution:
  fastducc                     -> displays overview help
  fastducc search [options]    -> chunk imaging & transient search pipeline
  fastducc cube [options]      -> dirty image FITS cube generation with fitscube
  fastducc aggregate [options] -> field-level candidate aggregation across beams
  fastducc aggregate_obs [...] -> cross-beam super summaries aggregation across SBID
  fastducc periodicity [...]   -> 3D DUCC NUFFT periodicity search

Direct execution fallback:
  fastducc --msname <MS> ...   -> automatically routed to 'search'
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Sequence

from fastducc import __version__

logger = logging.getLogger(__name__)


def build_cli_aggregate(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """CLI parser for 'fastducc aggregate'."""
    p = argparse.ArgumentParser(
        prog="fastducc aggregate",
        description="Aggregate per-beam candidate catalogues across an observation/scan",
    )
    p.add_argument("--obs-root", required=True, help="Root directory containing per-beam outputs")
    p.add_argument("--kind", choices=["boxcar", "variance"], default="boxcar",
                   help="Which candidate type to aggregate (default: boxcar)")
    p.add_argument("--time-tol", type=float, default=0.3,
                   help="Time tolerance in seconds (default: 0.3)")
    p.add_argument("--sky-tol-arcsec", type=float, default=35.0,
                   help="Sky tolerance in arcsec (default: 35)")
    p.add_argument("--pattern", default=None,
                   help="Optional glob pattern to discover input CSVs (overrides default)")
    p.add_argument("--outdir", default=None,
                   help="Output directory (default: obs-root)")
    p.add_argument("--max-n-candidates-per-source", "--max-candidates-per-source", "--max-N-candidates-per-source",
                   type=int, default=100,
                   help="Maximum events per candidate source to retain in summary (default: 100)")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging")
    return p.parse_args(argv)


def build_cli_aggregate_obs(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """CLI parser for 'fastducc aggregate_obs'."""
    p = argparse.ArgumentParser(
        prog="fastducc aggregate_obs",
        description="Aggregate per-scan cross-beam super summaries across an observation (SBID)",
    )
    p.add_argument("--obs-root", required=True, help="Path to SBID directory (e.g. /path/to/SB77974)")
    p.add_argument("--kind", choices=["variance", "boxcar"], default="variance",
                   help="Which family of super summaries to aggregate (default: variance)")
    p.add_argument("--sky-tol-arcsec", type=float, default=35.0,
                   help="Sky clustering tolerance in arcsec across scans (default: 35)")
    p.add_argument("--outdir", default=None,
                   help="Output directory (default: <obs-root>/candidates)")
    p.add_argument("--pattern", default=None,
                   help="Override VOT glob (default: <obs-root>/**/candidates/*_<kind>_super_summary.vot)")
    p.add_argument("--organize", action="store_true", default=True,
                   help="Organize candidate outputs into per-candidate directory trees under <outdir>/<kind>/<srcname>/ (default: True)")
    p.add_argument("--no-organize", action="store_false", dest="organize",
                   help="Disable candidate output directory tree organization")
    p.add_argument("--link-mode", choices=["symlink", "copy"], default="symlink",
                   help="How to place artefact files into candidate directories: absolute symlink or copy (default: symlink)")
    p.add_argument("--organize-structure", choices=["categorized", "flat"], default="categorized",
                   help="Directory tree structure for candidate subfolders (default: categorized)")
    p.add_argument("--max-n-candidates-per-source", "--max-candidates-per-source", "--max-N-candidates-per-source",
                   type=int, default=100,
                   help="Maximum events per candidate source to output in full candidate list (default: 100)")
    p.add_argument("--psrcat-path", default=None, help="Override path to PSRCAT CSV catalogue (default: packaged file)")
    p.add_argument("--racs-path", default=None, help="Override path to RACS VOTable catalogue (default: packaged file)")
    p.add_argument("--render-summary-plots", dest="render_summary_plots", action="store_true", default=False,
                   help="Render summary PDFs from HDF5 containers for each source exemplar and top-N events")
    p.add_argument("--no-render-summary-plots", dest="render_summary_plots", action="store_false",
                   help="Disable exemplar and top-N event summary PDF rendering during aggregate_obs")
    p.add_argument("--summary-top-events-per-source", type=int, default=5,
                   help="Number of top-S/N candidate events per source to render summary PDFs for (default: 5)")
    p.add_argument("--continuum-dir", default=None,
                   help="Directory containing continuum FITS images for summary plot overlays (default: <obs-root>/continuum_images)")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging")
    return p.parse_args(argv)


def build_cli_periodicity(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """CLI parser for 'fastducc periodicity'."""
    p = argparse.ArgumentParser(
        prog="fastducc periodicity",
        description="3-D DUCC NUFFT (u,v,t)->(l,m,f) periodicity search with S/N detection, catalogues and FITS products",
    )
    p.add_argument("--msname", required=True, help="Path to Measurement Set")
    p.add_argument("--spw", type=int, default=0, help="SPW index (default: 0)")
    p.add_argument("--pol", type=int, default=0, help="Pol index in DATA (default: 0)")
    p.add_argument("--data-col", default="DATA", help="MS data column (default: DATA)")
    p.add_argument("--chan-start", type=int, default=None)
    p.add_argument("--chan-end", type=int, default=None)

    p.add_argument("--npix", type=int, default=384)
    p.add_argument("--pixscale-arcsec", type=float, default=22.0)
    p.add_argument("--fmin", type=float, default=0.5, help="Minimum spin frequency to keep (Hz)")
    p.add_argument("--dm", type=float, default=0.0, help="Dispersion Measure (pc cm^-3)")
    p.add_argument(
        "--dedisp-mode",
        choices=["subbanded", "fullband"],
        default="subbanded",
        help="Dedispersion engine mode: 'subbanded' (two-tier subbanded) or 'fullband' (brute-force) (default: subbanded)",
    )
    p.add_argument(
        "--nsubbands",
        type=int,
        default=16,
        help="Number of coarse frequency subbands for two-tier subbanded dedispersion (--dedisp-mode subbanded, default: 16)",
    )
    p.add_argument(
        "--grid-freq-bins",
        type=int,
        default=0,
        help="Optional pre-gridding frequency channel averaging bins in fullband mode (0 = disabled, grid all native channels; default: 0)",
    )
    p.add_argument("--eps", type=float, default=1e-5, help="DUCC NUFFT tolerance")
    p.add_argument("--do-wstacking", action="store_true", default=True, help="Enable 3-D W-stacking to correct non-coplanar baseline phase errors (default: True)")
    p.add_argument("--no-wstacking", dest="do_wstacking", action="store_false", help="Disable W-stacking (pure coplanar 3-D NUFFT)")
    p.add_argument("--supp-w", type=int, default=4, help="Kernel support along w axis for w-stacking (default: 4)")
    p.add_argument("--w-oversampling", type=float, default=1.5, help="Oversampling factor for w-plane spacing (default: 1.5)")
    p.add_argument("--nfft-t", type=int, default=None, help="Length of frequency axis (power of 2)")
    p.add_argument("--block-rows", type=int, default=20000)
    p.add_argument("--partials-dir", default="partials_periodicity")
    p.add_argument("--sum-memmap", default="periodicity_sum_complex.dat")
    p.add_argument("--nthreads", type=int, default=0)

    # outputs
    p.add_argument("--out-cube-fits", default="periodicity_cube_pow.fits")
    p.add_argument("--write-products", action="store_true", help="Write additional 2D products (maxpow, bestfreq, PNG)")
    p.add_argument("--no-write-products", dest="write_products", action="store_false")
    p.set_defaults(write_products=True)

    # detection flags
    p.add_argument("--detect", action="store_true", help="Run spatial S/N detection on the spin-frequency cube")
    p.add_argument("--no-detect", dest="detect", action="store_false")
    p.set_defaults(detect=True)

    p.add_argument("--threshold-sigma", type=float, default=7.0)
    p.add_argument("--nharm", type=int, default=2, help="Number of harmonics to sum (default: 2)")
    p.add_argument("--keep-top-k", type=int, default=None)
    p.add_argument("--spatial-estimator", choices=["clipped_rms", "mad"], default="clipped_rms")
    p.add_argument("--clip-sigma", type=float, default=3.0)

    # catalogue outputs
    p.add_argument("--out-csv", default=None, help="Override candidate CSV output path")
    p.add_argument("--out-vot", default=None, help="Override candidate VOT output path")

    # parallel execution & limits
    p.add_argument("--max-blocks", type=int, default=None, help="Limit number of blocks processed for testing")
    p.add_argument("--parallel-mode", choices=["serial", "dask-local", "dask-slurm"], default="dask-local")
    p.add_argument("--dask-workers", type=int, default=0)
    p.add_argument("--threads-per-worker", type=int, default=1)
    p.add_argument("--dask-scheduler", choices=["processes", "threads"], default="processes")
    p.add_argument("--scheduler-address", default=None, help="Existing Dask scheduler address")
    p.add_argument("--slurm-partition", default=None, help="SLURM partition/queue")
    p.add_argument("--slurm-account", default=None, help="SLURM account/project")
    p.add_argument("--slurm-cores-per-worker", type=int, default=1, help="CPUs/cores per SLURM worker")
    p.add_argument("--slurm-mem", default="128GB", help="Memory limit per SLURM worker (e.g. 128GB)")
    p.add_argument("--slurm-walltime", default="02:00:00", help="SLURM job walltime (e.g. 02:00:00)")
    p.add_argument("--slurm-job-extra", nargs="*", default=None, help="Extra SLURM directives")
    p.add_argument("--slurm-interface", default=None, help="Network interface for SLURM workers")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging")

    return p.parse_args(argv)


def build_cli_extract_ds(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """CLI parser for 'fastducc extract_ds'."""
    p = argparse.ArgumentParser(
        prog="fastducc extract_ds",
        description="Extract in-memory phased visibility dynamic spectra from Measurement Set(s) to HDF5 (.ds)",
    )
    # Single-MS mode
    p.add_argument("--ms", "--msname", dest="ms", default=None, help="Path to Measurement Set (single-MS mode)")
    p.add_argument("--out", "-o", "--outdir", dest="out", default=None, help="Output .ds file or directory")
    p.add_argument("--ra", type=float, default=None, help="Right Ascension in degrees (single-MS mode)")
    p.add_argument("--dec", type=float, default=None, help="Declination in degrees (single-MS mode)")
    p.add_argument("--cand-id", default=None, help="Candidate identifier (single-MS mode)")
    p.add_argument("--dm", type=float, default=0.0, help="Dispersion Measure (pc cm^-3)")

    # Catalogue / observation batch orchestration mode (mirrors lotrun_periodicity pdmp_refine)
    p.add_argument("--sbid", default=None, help="SBID for observation batch extraction (e.g. SB77974)")
    p.add_argument("--data-root", default=None, help="Root directory holding SBIDs")
    p.add_argument("--catalogue", default=None, help="Path to candidate catalogue (VOTable or CSV)")
    p.add_argument("--kind", default="boxcar", help="Candidate family for catalogue auto-discovery (default: boxcar)")
    p.add_argument("--sbid-subdir", default=None, help="Optional subdirectory under <SBID> containing candidates/")
    p.add_argument("--ms-dir", default=None, help="Directory containing per-beam Measurement Sets")
    p.add_argument("--combined-dir", default="native_combined", help="Subdirectory under <SBID> for combined MS files")
    p.add_argument("--ms-glob-template", default="**/cracoData*%s*uvsub.ms", help="Glob template for beam MS discovery")
    p.add_argument("--min-snr", type=float, default=None, help="Minimum candidate S/N threshold")
    p.add_argument("--min-dm", type=float, default=None, help="Minimum candidate DM threshold (pc cm^-3)")
    p.add_argument("--max-cands", type=int, default=None, help="Maximum number of top candidates (by S/N) to extract")
    p.add_argument("--source-id", default=None, help="Restrict extraction to a single source_id or srcname")
    p.add_argument("--max-batch-cands-per-beam", "--max-cands-per-pass", dest="max_batch_cands_per_beam", type=int, default=10,
                   help="Maximum candidates batched per beam MS streaming pass (default: 10)")

    # Extraction options
    p.add_argument("--pol", default="I", help="Polarisation to extract ('I', 'XX', 'YY', etc.)")
    p.add_argument("--datacolumn", default="DATA", help="MS data column (DATA, CORRECTED_DATA, MODEL_DATA)")
    p.add_argument("--chunk-rows", type=int, default=100000, help="Chunk size for MS streaming (default: 100000)")
    p.add_argument("--min-uv-distance-m", "--minuvdist", dest="min_uv_distance_m", type=float, default=200.0,
                   help="Minimum baseline UV distance in metres (default: 200.0, matching transient search; set 0 to disable)")
    p.add_argument("--no-uv-clip", action="store_true",
                   help="Disable minimum baseline UV distance clipping (equivalent to --min-uv-distance-m 0.0)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing .ds files")
    p.add_argument("--dry-run", action="store_true", help="Print planned batches without extracting")
    p.add_argument("--update-summary-plots", dest="update_summary_plots", action="store_true", default=True,
                   help="Update candidate summary plots with dynamic spectrum (default: True)")
    p.add_argument("--no-update-summary-plots", dest="update_summary_plots", action="store_false",
                   help="Disable updating candidate summary plots")

    # Dask parallelisation (matches lotrun_periodicity pdmp_refine)
    p.add_argument("--parallel-mode", choices=["serial", "dask-local", "dask-slurm"], default="serial",
                   help="Execution mode for catalogue extraction (default: serial)")
    p.add_argument("--dask-scheduler-address", default=None, help="Existing Dask scheduler address")
    p.add_argument("--dask-min-workers", type=int, default=1, help="Minimum Dask workers (default: 1)")
    p.add_argument("--dask-max-workers", type=int, default=48, help="Maximum Dask workers (default: 48)")
    p.add_argument("--dask-local-workers", type=int, default=4, help="Workers for dask-local mode (default: 4)")
    p.add_argument("--dask-local-mem", default="6GB", help="Memory limit per local worker (default: 6GB)")
    p.add_argument("--dask-slurm-partition", default=None, help="Slurm partition for Dask workers")
    p.add_argument("--dask-slurm-account", default=None, help="Slurm account for Dask workers")
    p.add_argument("--dask-slurm-cores", type=int, default=2, help="CPUs per Slurm Dask worker (default: 2)")
    p.add_argument("--dask-slurm-mem", default="6GB", help="Memory per Slurm Dask worker (default: 6GB)")
    p.add_argument("--dask-slurm-walltime", default="00:30:00", help="Walltime per Slurm Dask worker (default: 00:30:00)")
    p.add_argument("--dask-slurm-interface", default=None, help="Network interface for Slurm Dask workers")
    p.add_argument("--dask-slurm-log-dir", default=None, help="Log directory for Slurm Dask workers")
    p.add_argument("--dask-job-name", default=None, help="Custom Slurm job name for Dask workers")
    p.add_argument("--dask-target-duration", default="1500s", help="Adaptive scaling target duration (default: 1500s)")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging")
    return p.parse_args(argv)


def _setup_logging(argv: Sequence[str]) -> None:
    """Initialize structured logging once at CLI entry based on verbosity flags."""
    is_verbose = any(arg in argv for arg in ("-v", "--verbose"))
    level = logging.DEBUG if is_verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)


def build_main_parser() -> argparse.ArgumentParser:
    """Construct top-level fastducc argument parser and command documentation."""
    parser = argparse.ArgumentParser(
        prog="fastducc",
        description="Fast transient imaging, dedispersion, and detection pipeline using DUCC0.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Subcommands:
  search (run)          Image MS in time chunks and search for transients (default)
  cube (make-cube)      Generate dirty FITS image cubes with fitscube or UV visibilities
  aggregate             Aggregate per-beam candidate catalogues across an observation/scan
  aggregate_obs         Aggregate cross-beam super summaries across an SBID observation
  periodicity           3D DUCC NUFFT periodicity search on MS visibilities
  extract_ds (extract-ds) High-performance in-memory phased visibility dynamic spectrum extraction
  merge_h5 (merge-h5)   Merge or consolidate per-(method, beam, DM) candidate HDF5 containers
  plot_h5 (plot-h5, plot-summaries) Render candidate summary PDFs/PNGs from an HDF5 container and optional final catalogue

Quick Examples:
  fastducc --msname /path/to/MS --chunk-size 512
  fastducc search --msname /path/to/MS --chunk-size 512
  fastducc cube --msname /path/to/MS --out-cube cube.fits
  fastducc aggregate --obs-root /path/to/SB12345/scan_00
  fastducc aggregate_obs --obs-root /path/to/SB12345
  fastducc periodicity --msname /path/to/MS --fmin 1.0
  fastducc extract_ds --ms /path/to/MS --ra 260.5 --dec -24.2 --cand-id cand01

Run 'fastducc <command> --help' for detailed options on any command.
""",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def run_search(argv: Sequence[str]) -> int:
    """Forward arguments to the chunk transient search pipeline."""
    from fastducc import fastducc_run
    res = fastducc_run.main(list(argv))
    return res if isinstance(res, int) else 0


def run_cube(argv: Sequence[str]) -> int:
    """Forward arguments to dirty FITS cube generation."""
    from fastducc import cube
    res = cube.main(list(argv))
    return res if isinstance(res, int) else 0


def run_aggregate(argv: Sequence[str]) -> int:
    """Execute per-beam candidate aggregation."""
    args = build_cli_aggregate(list(argv))
    from fastducc import candidates
    candidates.aggregate_observation(
        obs_root=args.obs_root,
        kind=args.kind,
        time_tol_s=args.time_tol,
        sky_tol_arcsec=args.sky_tol_arcsec,
        out_dir=args.outdir,
        pattern=args.pattern,
        max_n_candidates_per_source=args.max_n_candidates_per_source,
    )
    return 0


def run_aggregate_obs(argv: Sequence[str]) -> int:
    """Execute observation-level (SBID) candidate aggregation."""
    args = build_cli_aggregate_obs(list(argv))
    from fastducc import candidates
    from fastducc.catalogues import get_psrcat_csv_path, get_racs_vot_path
    candidates.aggregate_observation_from_super_summaries(
        obs_root=args.obs_root,
        kind=args.kind,
        sky_tol_arcsec=args.sky_tol_arcsec,
        out_dir=args.outdir,
        pattern=args.pattern,
        psrcat_csv_path=args.psrcat_path or get_psrcat_csv_path(),
        racs_vot_path=args.racs_path or get_racs_vot_path(),
        match_radius_arcsec=40.0,
        simbad_enable=False,
        organize=args.organize,
        link_mode=args.link_mode,
        organize_structure=args.organize_structure,
        max_n_candidates_per_source=args.max_n_candidates_per_source,
        render_summary_plots=args.render_summary_plots,
        summary_top_events_per_source=args.summary_top_events_per_source,
        continuum_dir=args.continuum_dir,
    )
    return 0


def run_periodicity(argv: Sequence[str]) -> int:
    """Execute 3-D DUCC NUFFT periodicity search."""
    args = build_cli_periodicity(list(argv))
    from fastducc import nufft_periodicity
    nufft_periodicity.run_periodicity(args)
    return 0


def run_extract_ds(argv: Sequence[str]) -> int:
    """Execute dynamic spectrum extraction via fastducc.dynspec (single MS or beam-batched catalogue)."""
    from pathlib import Path

    args = build_cli_extract_ds(list(argv))
    eff_min_uv = 0.0 if getattr(args, "no_uv_clip", False) else float(args.min_uv_distance_m)

    if args.catalogue or args.sbid:
        from fastducc.dynspec import batch_extract_catalogue

        summary = batch_extract_catalogue(
            catalogue_path=args.catalogue,
            sbid=args.sbid,
            data_root=args.data_root,
            kind=args.kind,
            sbid_subdir=args.sbid_subdir,
            ms_dir=args.ms_dir,
            combined_dir=args.combined_dir,
            ms_glob_template=args.ms_glob_template,
            out_dir=args.out,
            min_snr=args.min_snr,
            min_dm=args.min_dm,
            max_cands=args.max_cands,
            source_id=args.source_id,
            max_batch_cands_per_beam=args.max_batch_cands_per_beam,
            chunk_rows=args.chunk_rows,
            pol=args.pol,
            datacolumn=args.datacolumn,
            min_uv_distance_m=eff_min_uv,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            update_summary_plots=args.update_summary_plots,
            verbose=args.verbose,
            parallel_mode=args.parallel_mode,
            dask_scheduler_address=args.dask_scheduler_address,
            dask_min_workers=args.dask_min_workers,
            dask_max_workers=args.dask_max_workers,
            dask_local_workers=args.dask_local_workers,
            dask_local_mem=args.dask_local_mem,
            dask_slurm_partition=args.dask_slurm_partition,
            dask_slurm_account=args.dask_slurm_account,
            dask_slurm_cores=args.dask_slurm_cores,
            dask_slurm_mem=args.dask_slurm_mem,
            dask_slurm_walltime=args.dask_slurm_walltime,
            dask_slurm_interface=args.dask_slurm_interface,
            dask_slurm_log_dir=args.dask_slurm_log_dir,
            dask_job_name=args.dask_job_name,
            dask_target_duration=args.dask_target_duration,
        )
        return 0 if summary.get("failed", 0) == 0 else 1

    if not args.ms:
        logger.error("Provide --ms (single-MS mode) or --catalogue / --sbid (catalogue batch mode).")
        return 1

    from fastducc.dynspec import extract_phased_dynamic_spectra_from_ms, write_dynamic_spectrum_hdf5
    from fastducc.candidate_tools.plotting import update_candidate_summary_with_dynspec

    ms_path = Path(args.ms)
    targets = []
    if args.ra is not None and args.dec is not None:
        cand_id = args.cand_id or "cand_0"
        targets.append({"id": cand_id, "ra_deg": args.ra, "dec_deg": args.dec, "dm": args.dm})

    if not targets:
        logger.error("No targets provided. Specify --ra and --dec, or --catalogue / --sbid.")
        return 1

    dynspecs = extract_phased_dynamic_spectra_from_ms(
        ms_path=ms_path,
        targets=targets,
        chunk_rows=args.chunk_rows,
        pol=args.pol,
        datacolumn=args.datacolumn,
        min_uv_distance_m=eff_min_uv,
        verbose=args.verbose,
    )

    for tgt in targets:
        c_id = tgt["id"]
        if c_id not in dynspecs:
            continue
        spec = dynspecs[c_id]
        if args.out:
            out_p = Path(args.out)
            if out_p.is_dir() or str(args.out).endswith("/"):
                out_p.mkdir(parents=True, exist_ok=True)
                out_file = out_p / f"{c_id}.ds"
            else:
                out_file = out_p
        else:
            out_file = ms_path.parent / f"{c_id}.ds"

        extra_attrs = {"dm": float(tgt.get("dm", 0.0))}
        write_dynamic_spectrum_hdf5(spec, out_file, datacolumn=args.datacolumn, attrs=extra_attrs)
        logger.info(f"Saved dynamic spectrum to {out_file}")

        if args.update_summary_plots:
            for cand_dir in [out_file.parent, out_file.parent / "plots", out_file.parent.parent / "plots"]:
                if not cand_dir.is_dir():
                    continue
                for plot_file in cand_dir.glob(f"*{c_id}*_summary.pdf"):
                    try:
                        update_candidate_summary_with_dynspec(plot_file, out_file)
                        logger.info(f"Updated candidate summary plot: {plot_file}")
                    except Exception as e:
                        logger.warning(f"Failed to update plot {plot_file}: {e}")

    return 0


def build_cli_merge_h5(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """CLI parser for 'fastducc merge_h5'."""
    p = argparse.ArgumentParser(
        prog="fastducc merge_h5",
        description="Merge or consolidate per-(method, beam, DM) candidate HDF5 containers",
    )
    p.add_argument("--inputs", nargs="+", default=None, help="Explicit input candidate HDF5 file paths")
    p.add_argument("--search-dir", nargs="+", default=None, help="Directories to discover per-(method, beam, DM) HDF5 files")
    p.add_argument("--method", choices=["boxcar", "variance", "ffa", "nufft"], default=None, help="Filter/label search method")
    p.add_argument("--exemplars", default=None, help="Optional exemplar catalogue (CSV or VOTable) to consolidate only exemplar candidates")
    p.add_argument("--out", "-o", required=True, help="Output merged or consolidated HDF5 file path")
    p.add_argument("--sbid", default="", help="Observation SBID metadata")
    p.add_argument("--field", default="", help="Observation field metadata")
    p.add_argument("--remove-inputs", action="store_true", help="Remove input HDF5 files after successful merge (ignored in consolidate mode)")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging")
    return p.parse_args(argv)


def build_cli_plot_h5(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """CLI parser for 'fastducc plot_h5' / 'fastducc plot-summaries'."""
    p = argparse.ArgumentParser(
        prog="fastducc plot_h5",
        description="Render FastDUCC candidate summary PDFs/PNGs directly from a candidate HDF5 container and optional final catalogue",
    )
    p.add_argument("--h5", required=True, help="Path to consolidated or per-trial candidate HDF5 container")
    p.add_argument("--catalogue", "--catalog", dest="catalogue", default=None,
                   help="Optional final candidate catalogue (CSV or VOTable) to select, order, and annotate candidates")
    p.add_argument("--outdir", "-o", required=True, help="Output directory for rendered summary plots")
    p.add_argument("--srcname", nargs="+", default=None, help="Specific candidate srcname(s) or source_id(s) to plot")
    p.add_argument("--beam", default=None, help="Filter candidates to a specific beam (e.g. beam28)")
    p.add_argument("--min-snr", type=float, default=None, help="Minimum candidate S/N threshold")
    p.add_argument("--min-dm", type=float, default=None, help="Minimum candidate DM threshold (pc cm^-3)")
    p.add_argument("--top", type=int, default=20,
                   help="Maximum number of top-S/N candidates to plot when --srcname is omitted (default: 20; <=0 for all)")
    p.add_argument("--continuum-dir", default=None, help="Directory containing continuum reference FITS images")
    p.add_argument("--dynspec-dir", default=None, help="Optional directory containing external .ds files when not embedded in HDF5")
    p.add_argument("--rfi-mad-thresh", type=float, default=3.0,
                   help="Frequency-axis MAD variance RFI flagging threshold (default: 3.0; <=0 to disable)")
    p.add_argument("--no-dedisperse", action="store_true",
                   help="Disable dedispersion of the dynamic spectrum to the candidate DM")
    p.add_argument("--png", action="store_true", help="Also save PNG copies alongside summary PDFs")
    p.add_argument("--cmap", default="viridis", help="Matplotlib colourmap name (default: viridis)")
    p.add_argument("--dpi", type=int, default=200, help="Output figure DPI (default: 200)")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging")
    return p.parse_args(argv)


def run_merge_h5(argv: Sequence[str]) -> int:
    """Execute HDF5 candidate merge or exemplar consolidation."""
    from astropy.table import Table
    from fastducc.candidate_store import (
        consolidate_candidate_h5,
        discover_trial_h5_files,
        merge_candidate_h5,
    )

    args = build_cli_merge_h5(list(argv))
    in_paths = list(args.inputs or [])
    if args.search_dir:
        in_paths.extend(str(p) for p in discover_trial_h5_files(args.search_dir, method=args.method))
    if not in_paths:
        logger.error("No input candidate HDF5 files found.")
        return 1

    if args.exemplars:
        fmt = "votable" if str(args.exemplars).endswith((".vot", ".xml")) else "ascii.csv"
        tab = Table.read(args.exemplars, format=fmt)
        out_p, matched = consolidate_candidate_h5(
            in_paths,
            tab,
            args.out,
            method=args.method,
            sbid=args.sbid,
            field=args.field,
        )
        logger.info("Wrote consolidated exemplar HDF5 (%d/%d matched) -> %s", len(matched), len(tab), out_p)
    else:
        out_p = merge_candidate_h5(
            in_paths,
            args.out,
            method=args.method,
            sbid=args.sbid,
            field=args.field,
            remove_inputs=args.remove_inputs,
        )
        logger.info("Wrote merged candidate HDF5 -> %s", out_p)
    return 0


def run_plot_h5(argv: Sequence[str]) -> int:
    """Render candidate summary plots directly from a candidate HDF5 container and optional final catalogue."""
    from fastducc.candidate_tools.plotting import plot_summaries_from_catalogue_and_h5

    args = build_cli_plot_h5(list(argv))
    top_val = None if (args.srcname or (args.top is not None and args.top <= 0)) else args.top
    summary = plot_summaries_from_catalogue_and_h5(
        h5_path=args.h5,
        out_dir=args.outdir,
        catalogue_path=args.catalogue,
        continuum_dir=args.continuum_dir,
        dynspec_dir=args.dynspec_dir,
        srcnames=args.srcname,
        beam=args.beam,
        min_snr=args.min_snr,
        min_dm=args.min_dm,
        top=top_val,
        dpi=args.dpi,
        cmap=args.cmap,
        rfi_mad_thresh=args.rfi_mad_thresh,
        dedisperse_dynspec=not args.no_dedisperse,
        save_png=args.png,
    )
    logger.info(
        "Rendered %d summary plot(s) from %s%s -> %s (%d missing)",
        summary["rendered"],
        args.h5,
        f" + {args.catalogue}" if args.catalogue else "",
        args.outdir,
        summary["missing"],
    )
    return 0


# Subcommand dispatch table
_SUBCOMMAND_DISPATCH = {
    "search": run_search,
    "run": run_search,
    "cube": run_cube,
    "make-cube": run_cube,
    "make_cube": run_cube,
    "aggregate": run_aggregate,
    "aggregate_obs": run_aggregate_obs,
    "aggregate-obs": run_aggregate_obs,
    "periodicity": run_periodicity,
    "extract_ds": run_extract_ds,
    "extract-ds": run_extract_ds,
    "dynspec": run_extract_ds,
    "merge_h5": run_merge_h5,
    "merge-h5": run_merge_h5,
    "plot_h5": run_plot_h5,
    "plot-h5": run_plot_h5,
    "plot_summaries": run_plot_h5,
    "plot-summaries": run_plot_h5,
}


def main(argv: Sequence[str] | None = None) -> int:
    """
    Unified CLI entry point for 'fastducc'.

    Supports explicit subcommands:
      fastducc search [options]
      fastducc cube [options]
      fastducc aggregate [options]
      fastducc aggregate_obs [options]
      fastducc periodicity [options]

    Maintains 100% backward compatibility for direct invocations:
      fastducc --msname <MS> ... -> routed to search
    """
    if argv is None:
        argv = sys.argv[1:]

    _setup_logging(argv)

    if not argv:
        build_main_parser().print_help()
        return 0

    first_arg = argv[0].strip()

    # Top-level help and version
    if first_arg in ("-h", "--help"):
        build_main_parser().print_help()
        return 0
    if first_arg in ("-V", "--version"):
        print(f"fastducc {__version__}")
        return 0

    # Subcommand dispatch
    cmd_lower = first_arg.lower()
    if cmd_lower in _SUBCOMMAND_DISPATCH:
        handler = _SUBCOMMAND_DISPATCH[cmd_lower]
        return handler(argv[1:])

    # Shorthand: options passed directly without subcommand (e.g. 'fastducc --msname ...')
    if first_arg.startswith("-"):
        return run_search(argv)

    # Unknown subcommand
    logger.error(
        f"Unknown command '{first_arg}'. Available commands: "
        f"{', '.join(sorted(set(_SUBCOMMAND_DISPATCH.keys())))}. "
        "Run 'fastducc --help' for usage."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
