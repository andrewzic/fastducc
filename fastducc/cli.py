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

Quick Examples:
  fastducc --msname /path/to/MS --chunk-size 512
  fastducc search --msname /path/to/MS --chunk-size 512
  fastducc cube --msname /path/to/MS --out-cube cube.fits
  fastducc aggregate --obs-root /path/to/SB12345/scan_00
  fastducc aggregate_obs --obs-root /path/to/SB12345
  fastducc periodicity --msname /path/to/MS --fmin 1.0

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
    )
    return 0


def run_periodicity(argv: Sequence[str]) -> int:
    """Execute 3-D DUCC NUFFT periodicity search."""
    args = build_cli_periodicity(list(argv))
    from fastducc import nufft_periodicity
    nufft_periodicity.run_periodicity(args)
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
