from __future__ import annotations
import argparse
import logging
import os
import sys
import threading
import time
import numpy as np

logger = logging.getLogger(__name__)
import dask
try:
    from dask.distributed import Client, LocalCluster, as_completed
except ImportError:
    Client = LocalCluster = as_completed = None

try:
    from dask_jobqueue import SLURMCluster
except ImportError:
    SLURMCluster = None

try:
    from casacore.tables import table
except ImportError:
    table = None

try:
    import ducc0
except ImportError:
    ducc0 = None
    
from fastducc.fd_types import Config
from fastducc import candidates, ms_utils, imaging, dedisp
from fastducc import core as fd_core
from fastducc.catalogues import get_psrcat_csv_path, get_racs_vot_path

psrcat_path = get_psrcat_csv_path()
racs_path   = get_racs_vot_path()

def log_memory(stage: str):
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        self_mb = process.memory_info().rss / (1024 * 1024)
        try:
            children = process.children(recursive=True)
            child_mb = sum(c.memory_info().rss for c in children) / (1024 * 1024)
        except Exception:
            child_mb = 0.0
        virtual_mem = psutil.virtual_memory()
        sys_used = virtual_mem.used / (1024 * 1024)
        sys_total = virtual_mem.total / (1024 * 1024)
        logger.debug(f"[MEM PROFILE] Stage: {stage} | Self RSS: {self_mb:.2f} MB | Children RSS: {child_mb:.2f} MB | System Used: {sys_used:.2f}/{sys_total:.2f} MB")
    except Exception as e:
        logger.debug(f"[MEM PROFILE ERROR] Could not log memory at {stage}: {e}")


def _add_bool_flag(
    parser: argparse.ArgumentParser,
    true_flag: str,
    false_flag: str,
    dest: str,
    default: bool,
    help_true: str | None = None,
    help_false: str | None = None,
):
    """Register mutually exclusive true/false boolean flags with a default."""
    kw_true = {"help": help_true} if help_true else {}
    kw_false = {"help": help_false} if help_false else {}
    parser.add_argument(true_flag, dest=dest, action="store_true", **kw_true)
    parser.add_argument(false_flag, dest=dest, action="store_false", **kw_false)
    parser.set_defaults(**{dest: default})


def build_cli(argv=None):

    parser = argparse.ArgumentParser(prog='fastducc search', description='Image MS in time chunks using ducc0 wgridder')
    parser.add_argument('--msname', required=True, help='Path to Measurement Set')
    parser.add_argument('--chunk-size', type=int, default=512,
                        help='Number of time samples per chunk (default: 512)')
    parser.add_argument('--corr-mode', choices=['average','stokesI','single'], default='single',
                        help='Correlation handling mode (default: single)')
    parser.add_argument('--basis', choices=['auto','linear','circular'], default='linear',
                        help='Basis for stokesI (default: auto)')
    parser.add_argument('--single-pol', default='XX',
                        help='Single pol to image when corr-mode=single (default: XX)')
    parser.add_argument('--data-column', default='DATA', help='Which data column to image from the measurement set (default: DATA)')
    parser.add_argument('--npix-x', type=int, default=384)
    parser.add_argument('--npix-y', type=int, default=384)
    parser.add_argument('--pixsize-arcsec', type=float, default=22.0,
                        help='Pixel size (arcsec); applied to both axes (default: 22.0)')
    parser.add_argument('--epsilon', type=float, default=1e-6)
    _add_bool_flag(parser, '--do-wgridding', '--no-wgridding', 'do_wgridding', True)
    parser.add_argument('--nthreads', type=int, default=0)
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging")
    parser.add_argument('--verbosity', type=int, default=0)
    parser.add_argument('--boxcar-widths', nargs='+', type=int,
                        help='Widths (in samples) to perform boxcar search over (default: [1, 2, 4, 8, 16, 32, 64])',
                        default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument('--threshold-sigma', type=float, default=8.0, help="S/N threshold to use for boxcar detections (default: 8.0)")
    parser.add_argument('--do-plot', action='store_true')
    parser.add_argument('--var-threshold-sigma', type=float, default=8.0, help="S/N threshold to use for boxcar detections (default: 8.0)")
    parser.add_argument('--var-keep-top-k', type=int, default=None)
    parser.add_argument('--var-highpass-cutoff', type=float, default=30.0, help="Cutoff timescale in seconds for the EMA high-pass filter, useful for sidelobe suppression in variance search. Set to 0 to disable (default: 0.0)")
    parser.add_argument('--rms-clip-sigma', type=float, default=3.0)
    parser.add_argument('--nms-radius', type=int, default=6)
    parser.add_argument('--local-box-size', type=int, default=64, help="Sliding 2D window size in pixels for local background & stdev thresholding (default: 64)")
    _add_bool_flag(
        parser, '--enable-local-stats', '--disable-local-stats', 'use_local_threshold', True,
        help_true="Enable 2D local background & stdev thresholding (default: enabled)",
        help_false="Disable 2D local thresholding, fallback to global statistics",
    )
    # --- Search toggles ---
    _add_bool_flag(
        parser, '--enable-boxcar', '--disable-boxcar', 'enable_boxcar', True,
        help_true='Enable per-width boxcar transient search (default: enabled)',
        help_false='Disable per-width boxcar transient search',
    )
    _add_bool_flag(
        parser, '--do-boxcar-search', '--no-boxcar-search', 'do_boxcar_search', True,
        help_true='Perform boxcar transient search (default: enabled)',
        help_false='Disable boxcar transient search',
    )
    _add_bool_flag(
        parser, '--enable-var', '--disable-var', 'enable_var', True,
        help_true='Enable variance (std-map/Welford) variability search — master switch for all variance modes (default: enabled)',
        help_false='Disable all variance (std-map/Welford) variability searches',
    )
    _add_bool_flag(
        parser, '--enable-var-obs', '--disable-var-obs', 'enable_var_obs', False,
        help_true='Enable whole-observation variance search on the final merged std-map (default: disabled)',
        help_false='Disable whole-observation variance search',
    )
    _add_bool_flag(
        parser, '--do-var-search', '--no-var-search', 'do_var_search', True,
        help_true='Perform variance (std-map/Welford) variability search (default: enabled)',
        help_false='Disable variance (std-map/Welford) variability search',
    )
    _add_bool_flag(
        parser, '--enable-var-chunk', '--disable-var-chunk', 'enable_var_chunk', True,
        help_true='Enable variance (std-map/Welford) variability search per chunk (default: enabled)',
        help_false='Disable variance (std-map/Welford) variability search per chunk',
    )
    _add_bool_flag(
        parser, '--enable-var-scan', '--disable-var-scan', 'enable_var_scan', True,
        help_true='Enable variance (std-map/Welford) variability search per scan (default: enabled)',
        help_false='Disable variance (std-map/Welford) variability search per scan',
    )
    parser.add_argument('--plot-cands-only', dest='plot_cands_only', action='store_true', help='When plotting is enabled, only read and plot candidates instead of searching for them')
    _add_bool_flag(parser, '--save-var-lightcurves', '--no-save-var-lightcurves', 'save_var_lightcurves', True, help_true='Save lightcurves for variance candidates')
    _add_bool_flag(parser, '--save-full-var-lightcurves', '--no-save-full-var-lightcurves', 'save_full_var_lightcurves', False, help_true='Save full lightcurves (over all chunks) for variance candidates')
    _add_bool_flag(parser, '--save-var-snippets', '--no-save-var-snippets', 'save_var_snippets', False, help_true='Save snippet products (PNG/GIF/FITS) for variance candidates')
    _add_bool_flag(parser, '--save-box-lightcurves', '--no-save-box-lightcurves', 'save_box_lightcurves', True, help_true='Save lightcurves for boxcar candidates')
    _add_bool_flag(parser, '--save-box-snippets', '--no-save-box-snippets', 'save_box_snippets', False, help_true='Save snippet products (PNG/GIF/FITS) for boxcar candidates')
    parser.add_argument('--continuum-dir', default=None, help='Path to directory containing deep per-beam continuum fits images.')

    # --- Dedispersion & DM search parameters ---
    parser.add_argument('--dm', type=float, default=None,
                        help='Single trial DM (pc cm^-3). Default is 0.0 if not specified.')
    parser.add_argument('--dm-min', type=float, default=0.0,
                        help='Minimum trial DM (pc cm^-3, default: 0.0)')
    parser.add_argument('--dm-max', type=float, default=0.0,
                        help='Maximum trial DM (pc cm^-3, default: 0.0)')
    parser.add_argument('--dm-step', type=float, default=None,
                        help='Trial DM step size (pc cm^-3). If None and dm-max > dm-min, computed using --dm-tol.')
    parser.add_argument('--dm-tol', type=float, default=1.0,
                        help='Smearing tolerance factor in samples for optimal DM trial grid generation (default: 1.0).')
    _add_bool_flag(
        parser, '--collapse-channels', '--no-collapse-channels', 'collapse_channels', False,
        help_true='Average/collapse channel dimension after dedispersion before imaging (default: False).',
        help_false='Retain channel dimension when imaging (default: True).',
    )
    parser.add_argument('--nsubbands', type=int, default=1,
                        help='Number of subbands to divide the bandwidth into before imaging (default: 1).')
    parser.add_argument('--dm-list', type=float, nargs='+', default=None,
                        help='Explicit list of trial DMs (pc cm^-3), e.g. --dm-list 0.0 100.0 160.0 186.4 210.0')
    parser.add_argument('--scan-id', '--scan', dest='scan_id', default=None,
                        help='Filter chunk processing to a specific scan ID (e.g. 20251015085704)')
    parser.add_argument('--max-chunks', type=int, default=0,
                        help='Limit the number of chunks to process (0 = all)')
    _add_bool_flag(
        parser, '--exact-uvw', '--no-exact-uvw', 'exact_uvw', True,
        help_true='Calculate exact per-channel UVW coordinates accounting for dispersion delay (default: True).',
        help_false='Use baseline UVW coordinates at the snapshot reference timestamp without channel migration.',
    )
    parser.add_argument('--min-valid-channels', type=int, default=24,
                        help='Minimum number of non-zero-weight channels across band before imaging timestep (default: 24).')
    _add_bool_flag(
        parser, '--filter-ips-agn', '--no-filter-ips-agn', 'filter_ips_agn', True,
        help_true='Suppress DM>0 candidates at positions of known (RACS - PSRCAT) continuum sources if S/N <= threshold * DM0 S/N (default: True).',
        help_false='Disable (RACS - PSRCAT) IPS AGN suppression at DM>0.',
    )
    parser.add_argument('--ips-snr-ratio-threshold', type=float, default=0.8,
                        help='S/N ratio threshold vs DM=0 for keeping candidates coincident with (RACS - PSRCAT) sources (default: 0.8).')
    parser.add_argument('--ips-match-radius-arcsec', type=float, default=30.0,
                        help='Spatial match radius in arcsec for crossmatching candidates with RACS/PSRCAT (default: 30.0).')

    # --- Parallel execution flags ---
    parser.add_argument(
        '--parallel-mode',
        choices=['serial', 'dask-local', 'dask-slurm'],
        default='serial',
        help='Serial, Dask LocalCluster, or Dask SLURMCluster execution'
    )
    parser.add_argument(
        '--dask-workers', type=int, default=0,
        help='Number of Dask workers when --parallel-mode=dask; 0 => auto'
    )
    parser.add_argument(
        '--threads-per-worker', type=int, default=1,
        help='Threads per Dask worker (default: 1)'
    )
    parser.add_argument(
        '--dask-scheduler', choices=['processes', 'threads'], default='processes',
        help='Use multiprocessing or multithreading workers (default: processes)'
    )
    parser.add_argument(
        '--dask-worker-timeout', type=int, default=0,
        help='Timeout in seconds to wait for Dask workers before aborting (0 = wait indefinitely; default: 0)'
    )
    # --- SLURM-only cluster params ---
    parser.add_argument('--slurm-partition', default=None, help='SLURM partition/queue')
    parser.add_argument('--slurm-account',  default=None, help='SLURM account/project')
    parser.add_argument('--slurm-cores-per-worker', type=int, default=1)
    parser.add_argument('--slurm-mem', default='32GB', help='Memory per worker (e.g., 32GB)')
    parser.add_argument('--slurm-walltime', default='02:30:00')
    parser.add_argument('--slurm-job-extra', nargs='*', default=[],
                        help='Extra SLURM directives, e.g. ["--exclusive"]')
    parser.add_argument('--slurm-interface', default=None,
                        help='Network interface name (if needed for TCP comms)')

    # --- Cube creation mode ---
    parser.add_argument(
        '--make-cube', '--cube-only', dest='make_cube', action='store_true', default=False,
        help='Write out dirty FITS cube with no further detection'
    )
    parser.add_argument(
        '--use-fitscube', action='store_true', default=False,
        help='Combine via Alec Thomson fitscube file-based workflow instead of direct memory-mapped streaming'
    )
    parser.add_argument('--out-cube', default=None, help='Output path for dirty FITS cube')
    parser.add_argument('--start-time-idx', '--start-idx', type=int, default=None, help='0-based start time index')
    parser.add_argument('--end-time-idx', '--end-idx', type=int, default=None, help='0-based end time index')
    parser.add_argument('--start-time', type=float, default=None, help='Start time in seconds or MJD')
    parser.add_argument('--end-time', type=float, default=None, help='End time in seconds or MJD')
    parser.add_argument('--create-blanks', action='store_true', default=False,
                        help='Create blank NaN planes for missing time intervals on a regular grid (following fitscube)')
    parser.add_argument('--regularize-time', '--grid-time', dest='regularize_time', action='store_true', default=False,
                        help='Grid data onto a strictly regular time grid, padding inter-scan gaps and shifting out-of-sync blocks')
    parser.add_argument('--sync-tolerance', type=float, default=0.05,
                        help='Offset tolerance in units of offset/time_resolution (default: 0.05 = 5%%). Blocks with offset > tolerance are shifted')
    parser.add_argument('--shift-method', choices=['fft', 'sinc'], default='fft',
                        help='Method to shift out-of-sync blocks: Fourier domain phase gradient shift (\'fft\') or windowed sinc interpolation (\'sinc\')')
    parser.add_argument('--shift-domain', choices=['native', 'gridded'], default='native',
                        help='Domain to apply shifting: native visibilities (\'native\', faster) or gridded UV/dirty cubes (\'gridded\')')
    parser.add_argument('--time-step', '--dt', dest='time_step', type=float, default=None,
                        help='Nominal time step / integration time in seconds (auto-detected if omitted)')
    parser.add_argument('--max-blank-factor', type=float, default=10.0,
                        help='Maximum allowed ratio of regular grid frames to observed frames when --create-blanks is enabled (default: 10.0)')
    parser.add_argument('--no-timestamps-table', dest='write_timestamps_table', action='store_false', default=True,
                        help='Do not include TIMESTAMPS binary table extension HDU')
    parser.add_argument('--save-uv', '-save-uv', dest='save_uv', action='store_true', default=False,
                        help='Save gridded UV visibilities (2D FFT of dirty images) into separate real and imaginary FITS cubes (following wsclean -save-uv)')
    parser.add_argument('--uv-only', dest='uv_only', action='store_true', default=False,
                        help='Only output real and imaginary UV cubes, skipping the dirty image cube')
    parser.add_argument('--out-uv-real', default=None,
                        help='Explicit path for output real UV cube (default: <prefix>-uv-real.cube.fits)')
    parser.add_argument('--out-uv-imag', default=None,
                        help='Explicit path for output imaginary UV cube (default: <prefix>-uv-imag.cube.fits)')
    parser.add_argument('--uv-4d', '--uv-stokes-axis', dest='uv_4d', action='store_true', default=False,
                        help='Output UV cubes with 4D geometry (STOKES, TIME, V, U)')
    parser.add_argument(
        '--coord', '--cubelet-coord', '--target-coord', '--target', dest='coord', default=None,
        help="Target celestial coordinates for cubelet mode (e.g. '17:33:00 -23:44:00', '17h33m00s -23d44m00s', '263.25, -23.73')"
    )
    parser.add_argument(
        '--ra', '--cubelet-ra', dest='ra', default=None,
        help='Target Right Ascension for cubelet mode (sexagesimal or degrees)'
    )
    parser.add_argument(
        '--dec', '--cubelet-dec', dest='dec', default=None,
        help='Target Declination for cubelet mode (sexagesimal or degrees)'
    )
    parser.add_argument(
        '--radius', '--cubelet-radius', dest='radius', default=None,
        help="Cubelet cutout radius (e.g. '45arcsec', '1.5arcmin', '32pix', or float in arcsec)"
    )
    parser.add_argument(
        '--radius-arcsec', type=float, default=None,
        help='Cubelet cutout radius in arcseconds'
    )
    parser.add_argument(
        '--radius-pix', type=float, default=None,
        help='Cubelet cutout radius in pixels'
    )
    parser.add_argument(
        '--box-size', '--box-size-pix', dest='box_size_pix', type=int, default=None,
        help='Explicit cubelet diameter / box size in pixels (e.g. 64)'
    )
    parser.add_argument(
        '--box-size-arcsec', type=float, default=None,
        help='Explicit cubelet diameter / box size in arcseconds'
    )
    parser.add_argument(
        '--radius-unit', choices=['arcsec', 'pix'], default='arcsec',
        help='Default unit when radius is passed as a bare number (default: arcsec)'
    )

    return parser.parse_args(argv)

from fastducc.cli import (
    build_cli_aggregate,
    build_cli_aggregate_obs,
    build_cli_periodicity,
)


def make_config(args, paths) -> Config:
    ms_base, candidates_dir, chunk_prefix_root, all_prefix_root = paths
    ra0_rad, dec0_rad, _ = ms_utils.get_phase_center(args.msname, field_name=None)
    pix_rad = np.deg2rad(args.pixsize_arcsec / 3600.0)
    current_dm = getattr(args, "dm", 0.0)
    if current_dm is None:
        current_dm = 0.0
    collapse_channels = getattr(args, "collapse_channels", False)
    nsubbands = getattr(args, "nsubbands", 1)
    nthreads = args.nthreads if args.nthreads > 0 else (args.slurm_cores_per_worker if getattr(args, "parallel_mode", "") == 'dask-slurm' else 0)
    return Config(
        msname=args.msname,
        npix_x=args.npix_x, npix_y=args.npix_y,
        pix_rad=pix_rad, ra0_rad=ra0_rad, dec0_rad=dec0_rad,
        epsilon=args.epsilon, do_wgridding=args.do_wgridding,
        nthreads=nthreads, verbosity=args.verbosity,
        corr_mode=args.corr_mode, basis=args.basis, single_pol=args.single_pol,
        data_column=args.data_column,
        enable_var=args.enable_var, enable_boxcar=args.enable_boxcar,
        do_var_search=args.do_var_search, do_boxcar_search=args.do_boxcar_search,
        plot_cands_only=args.plot_cands_only,
        enable_var_chunk=args.enable_var_chunk,
        enable_var_scan=args.enable_var_scan,
        enable_var_obs=args.enable_var_obs,
        save_var_lightcurves=args.save_var_lightcurves,
        save_full_var_lightcurves=args.save_full_var_lightcurves,
        save_var_snippets=args.save_var_snippets,
        save_box_lightcurves=args.save_box_lightcurves,
        save_box_snippets=args.save_box_snippets,
        var_threshold=args.var_threshold_sigma, var_keep_k=args.var_keep_top_k,
        rms_clip_sigma=args.rms_clip_sigma, nms_radius=args.nms_radius,
        boxcar_widths=args.boxcar_widths, boxcar_threshold=args.threshold_sigma,
        do_plot=args.do_plot,
        ms_base=ms_base, candidates_dir=candidates_dir,
        chunk_prefix_root=chunk_prefix_root, all_prefix_root=all_prefix_root,
        continuum_dir=args.continuum_dir,
        var_highpass_cutoff_sec=args.var_highpass_cutoff,
        use_local_threshold=getattr(args, "use_local_threshold", True),
        local_window_size=getattr(args, "local_box_size", 64),
        current_dm=current_dm,
        collapse_channels=collapse_channels,
        nsubbands=nsubbands,
        exact_uvw=getattr(args, "exact_uvw", True),
        scan_id=getattr(args, "scan_id", None),
        min_valid_channels=getattr(args, "min_valid_channels", 24),
        filter_ips_agn=getattr(args, "filter_ips_agn", True),
        ips_snr_ratio_threshold=getattr(args, "ips_snr_ratio_threshold", 0.8),
        ips_match_radius_arcsec=getattr(args, "ips_match_radius_arcsec", 30.0),
    )

def main_serial(args):
    args.parallel_mode = "serial"
    return run_pipeline(args)

def aggregate_main(argv=None):

    args = build_cli_aggregate(argv)
    
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

def aggregate_obs_main(argv=None):
    args = build_cli_aggregate_obs(argv)
    candidates.aggregate_observation_from_super_summaries(
        obs_root=args.obs_root,
        kind=args.kind,
        sky_tol_arcsec=args.sky_tol_arcsec,
        out_dir=args.outdir,
        pattern=args.pattern,
        psrcat_csv_path=getattr(args, "psrcat_path", None) or psrcat_path,
        racs_vot_path=getattr(args, "racs_path", None) or racs_path,
        match_radius_arcsec=40.0,
        simbad_enable=False,
        organize=args.organize,
        link_mode=args.link_mode,
        organize_structure=args.organize_structure,
        max_n_candidates_per_source=args.max_n_candidates_per_source,
    )
    return 0

    

def run_pipeline(args):
    log_memory("Before derive_paths and make_config")
    ms_base, candidates_dir, chunk_prefix_root, all_prefix_root = ms_utils.derive_paths(args.msname)
    cfg = make_config(args, (ms_base, candidates_dir, chunk_prefix_root, all_prefix_root))

    # Compute chunk bounds (using casacore once in the driver)
    log_memory("Before opening t_main (casacore)")
    t_main = table(cfg.msname, readonly=True)
    time_col = 'TIME_CENTROID' if 'TIME_CENTROID' in set(t_main.colnames()) else 'TIME'
    dt = ms_utils.get_timebin(t_main, time_col)
    t_main.close()
    log_memory("After closing t_main (casacore)")

    unique_times, scan_per_time_idx = ms_utils.get_unique_times(cfg.msname, time_col)
    log_memory("After get_unique_times (casacore)")
    total_chunks = len(unique_times)
        
    logger.info(f"Found {total_chunks} time chunks in MS: {cfg.msname}")
    logger.info(f"Found time resolution {dt}s")

    # Determine DM trial grid
    if args.dm_list is not None and len(args.dm_list) > 0:
        dm_trials = [float(d) for d in args.dm_list]
        logger.info(f"Explicit DM Trial List: {len(dm_trials)} trial DMs: {dm_trials}")
    elif args.dm is not None:
        dm_trials = [float(args.dm)]
    elif args.dm_max > args.dm_min:
        _, channel_freqs, _ = ms_utils.get_channel_lambdas(cfg.msname)
        plan = dedisp.DedispersionPlan(
            freqs_hz=channel_freqs,
            tsamp_s=dt,
            dm_min=args.dm_min,
            dm_max=args.dm_max,
            dm_step=args.dm_step,
            tolerance_samples=args.dm_tol,
        )
        dm_trials = [float(d) for d in plan.trial_dms]
        logger.info(f"Dedispersion Plan: {len(dm_trials)} trial DMs from {dm_trials[0]:.2f} to {dm_trials[-1]:.2f} pc cm^-3 (step ~ {plan.dm_step:.3f})")
    else:
        dm_trials = [0.0]

    cfg.dm_trials = dm_trials
    cfg.is_dm_search = (args.dm_max > args.dm_min) or (args.dm_list is not None and len(args.dm_list) > 1) or (len(dm_trials) > 1)

    # MAKE CHUNKS HAVE OVERLAP TO ACCOUNT FOR DISPERSIVE DELAYS
    # for DM = 1000 and 0.6-1GHz band, DM delay is ~7 s.
    # So a 10s buffer around each chunk should be sufficient
    buffer_overlap = 10.0 #s
    buffer_overlap_samps = int(buffer_overlap / dt)
    max_dm = max(dm_trials)
    if max_dm > 0.0:
        _, channel_freqs, _ = ms_utils.get_channel_lambdas(cfg.msname)
        max_delay_samps = dedisp.get_dm_samps(channel_freqs, max_dm, dt)
        if max_delay_samps + 1 > buffer_overlap_samps:
            buffer_overlap_samps = max_delay_samps + 1
            logger.info(f"Adjusted buffer_overlap_samps to {buffer_overlap_samps} samples ({buffer_overlap_samps * dt:.2f} s) to accommodate max trial DM {max_dm:.2f}")

    chunk_size = max(1, args.chunk_size)
    if chunk_size < 2 * buffer_overlap_samps:
        logger.warning(f"chunk_size {chunk_size} is smaller than 2 * buffer_overlap_samps {2*buffer_overlap_samps}.")
    
    chunk_bounds, scan_per_time_idx = ms_utils.get_scan_aware_chunk_bounds(
        cfg.msname, time_col, chunk_size, buffer_overlap_samps
    )
    log_memory("After get_scan_aware_chunk_bounds (casacore)")
    logger.info(f"Defined {len(chunk_bounds)} scan-aware chunks across {len(np.unique(scan_per_time_idx))} scans.")

    # Pre-compute scan_id_str for each chunk
    chunk_scan_ids = []
    for (s, e) in chunk_bounds:
        chunk_scans = scan_per_time_idx[s:e+1]
        chunk_times = unique_times[s:e+1]
        chunk_scan_ids.append(ms_utils.derive_scan_id(
            chunk_scans,
            times=chunk_times,
            msname=cfg.msname,
            unique_times=unique_times,
            scan_per_time_idx=scan_per_time_idx,
        ))

    # Filter by scan_id or max_chunks if requested
    if args.scan_id:
        filtered = [
            (b, s) for b, s in zip(chunk_bounds, chunk_scan_ids)
            if str(args.scan_id) in str(s)
        ]
        if len(filtered) == 0:
            logger.warning(f"No chunks matched scan_id '{args.scan_id}'. Available scans: {set(chunk_scan_ids)}")
        else:
            chunk_bounds = [b for b, s in filtered]
            chunk_scan_ids = [s for b, s in filtered]
            logger.info(f"Filtered to {len(chunk_bounds)} chunks matching scan '{args.scan_id}'.")

    if args.max_chunks > 0 and len(chunk_bounds) > args.max_chunks:
        chunk_bounds = chunk_bounds[:args.max_chunks]
        chunk_scan_ids = chunk_scan_ids[:args.max_chunks]
        logger.info(f"Limited to first {len(chunk_bounds)} chunks.")

    def _process_chunks(client=None):
        logger.info(f"Processing {len(chunk_bounds)} chunk(s) across {len(dm_trials)} DM trial(s) ({dm_trials[0]:.2f}..{dm_trials[-1]:.2f} pc cm^-3)")
        if client is None:
            # Serial execution
            agg_list = []
            for (start, end), scan_id_str in zip(chunk_bounds, chunk_scan_ids):
                logger.info(f"[Serial] Chunk {start}..{end} (scan {scan_id_str}) across {len(dm_trials)} DM trials...")
                chunk_times = unique_times[start:end+1]
                res = fd_core.process_chunk_task(
                    cfg, ms_base, candidates_dir, start, end, scan_id_str, chunk_times
                )
                times, cube, welford_data = res[0], res[1], res[2]
                sid = res[3] if len(res) > 3 else scan_id_str
                agg_list.append((times, cube if cfg.save_full_var_lightcurves else None, welford_data, sid))
            return agg_list
        else:
            # Dask parallel workers execution
            futures = [client.submit(fd_core.process_chunk_task, cfg, ms_base, candidates_dir, s, e, scan_id_str, unique_times[s:e+1])
                       for ((s, e), scan_id_str) in zip(chunk_bounds, chunk_scan_ids)]
            total = len(futures)
            logger.info(f"[Dask] Submitted {total} chunk tasks across {len(dm_trials)} DM trials. Monitoring progress...")
            completed = 0
            t_start = time.time()
            fut_to_idx = {f: i for i, f in enumerate(futures)}
            results = [None] * total

            stop_heartbeat = threading.Event()
            def _heartbeat():
                while not stop_heartbeat.wait(60):
                    try:
                        n_workers = len(client.scheduler_info().get("workers", {}))
                        elapsed = int(time.time() - t_start)
                        logger.info(f"[Dask Heartbeat] {completed}/{total} chunks completed | {n_workers} worker(s) online ({elapsed}s elapsed)")
                    except Exception:
                        pass

            hb_thread = threading.Thread(target=_heartbeat, daemon=True)
            hb_thread.start()

            try:
                for f in as_completed(futures):
                    idx = fut_to_idx.pop(f)
                    results[idx] = f.result()
                    try:
                        client.cancel(f, force=True)
                    except Exception:
                        pass
                    f.release()
                    completed += 1
                    elapsed = time.time() - t_start
                    (s, e), sid = chunk_bounds[idx], chunk_scan_ids[idx]
                    n_workers = len(client.scheduler_info().get("workers", {}))
                    logger.info(f"[Dask Progress] [{completed}/{total}] Chunk {s}..{e} (scan {sid}) finished ({elapsed:.1f}s elapsed, {n_workers} worker(s) online)")
            finally:
                stop_heartbeat.set()

            return results
    if args.parallel_mode == 'serial':
        agg_list = _process_chunks()
        if cfg.enable_var:
            for trial_dm in dm_trials:
                log_memory(f"Before finalise_welford_parallel DM={trial_dm:.2f}")
                _ = fd_core.finalise_welford_parallel(cfg, agg_list, dm=trial_dm)
                log_memory(f"After finalise_welford_parallel DM={trial_dm:.2f}")

    elif args.parallel_mode == 'dask-local':
        n_workers = args.dask_workers if args.dask_workers > 0 else None
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=args.threads_per_worker,
            processes=(args.dask_scheduler == 'processes')
        )
        with Client(cluster) as client:
            agg_list = _process_chunks(client=client)
            if cfg.enable_var:
                for trial_dm in dm_trials:
                    log_memory(f"Before finalise_welford_parallel DM={trial_dm:.2f}")
                    _ = fd_core.finalise_welford_parallel(cfg, agg_list, dm=trial_dm)
                    log_memory(f"After finalise_welford_parallel DM={trial_dm:.2f}")
        cluster.close()

    elif args.parallel_mode == 'dask-slurm':
        # SLURMCluster (requires dask_jobqueue)
        if SLURMCluster is None:
            raise RuntimeError("dask_jobqueue is required for --parallel-mode=dask-slurm")
        slurm_interface = args.slurm_interface
        if slurm_interface is None:
            try:
                import psutil
                if 'ib0' in psutil.net_if_addrs():
                    slurm_interface = 'ib0'
            except Exception:
                pass
        scheduler_opts = {"dashboard_address": ":0"}
        if slurm_interface:
            scheduler_opts["interface"] = slurm_interface
        prologue = [
            "module load python-scientific/3.11.5-foss-2023b 2>/dev/null || true",
            "unset PYTHONPATH",
            f"source {os.path.dirname(sys.executable)}/activate 2>/dev/null || true",
            f"export OMP_NUM_THREADS={args.slurm_cores_per_worker}",
            "export OPENBLAS_NUM_THREADS=1"
        ]

        # Allow tasks to be retried if workers retire or fail without aborting cluster
        # Configure adaptive target-duration to match coarse-grained batch tasks (~1 hour)
        dask.config.set({
            "distributed.scheduler.allowed-failures": 25,
            "distributed.adaptive.target-duration": "3600s",
        })

        worker_extra = ["--nthreads", "1", "--memory-limit", "0"]
        if args.slurm_walltime:
            try:
                parts = args.slurm_walltime.split(":")
                if len(parts) >= 2:
                    wt_min = int(parts[0]) * 60 + int(parts[1])
                    # Retire gracefully 45 minutes before SLURM hard walltime, but at least 30m lifetime
                    lifetime_min = max(30, wt_min - 45)
                    worker_extra.extend(["--lifetime", f"{lifetime_min}m", "--lifetime-stagger", "5m"])
            except Exception:
                pass

        log_memory("Before creating SLURMCluster")
        cluster = SLURMCluster(
            queue=args.slurm_partition,
            account=args.slurm_account,
            cores=args.slurm_cores_per_worker,
            memory=args.slurm_mem,
            walltime=args.slurm_walltime,
            job_extra_directives=args.slurm_job_extra,
            interface=slurm_interface,
            python=sys.executable,
            job_script_prologue=prologue,
            scheduler_options=scheduler_opts, #avoid dashboard port conflicts
            processes=1,
            worker_extra_args=worker_extra,
        )
        log_memory("After creating SLURMCluster")
        # adapt to number of workers: scale dynamically to match expected number of time chunks if 0 or not specified
        if args.dask_workers and args.dask_workers > 0:
            n_workers = args.dask_workers
        else:
            n_workers = max(1, len(chunk_bounds))

        cluster.adapt(minimum=0, maximum=n_workers)
        logger.info(f"[Dask-SLURM] Dynamically adapted cluster (0..{n_workers} workers).")
        log_memory(f"After cluster.adapt ({n_workers})")

        # Connect the client and run
        log_memory("Before Client(cluster) context creation")
        with Client(cluster) as client:
            log_memory("After Client(cluster) context creation")
            n_init = len(client.scheduler_info()["workers"])
            logger.info(f"[Dask-SLURM] Connected to cluster. {n_init} worker(s) initially ready.")
            agg_list = _process_chunks(client=client)
            if cfg.enable_var:
                for trial_dm in dm_trials:
                    log_memory(f"Before finalise_welford_parallel DM={trial_dm:.2f}")
                    _ = fd_core.finalise_welford_parallel(cfg, agg_list, dm=trial_dm)
                    log_memory(f"After finalise_welford_parallel DM={trial_dm:.2f}")

        log_memory("After Client closed")
        # Graceful stop after the client context exits
        try:
            cluster.scale(0)  # retire workers first
        except BaseException as e:
            logger.warning(f"Error scaling cluster down: {e}")
        try:
            cluster.close()
        except BaseException as e:
            logger.warning(f"Error closing cluster: {e}")
        log_memory("After SLURMCluster closed")

    else:
        raise ValueError(f"Unknown parallel_mode: {args.parallel_mode}")

    # Consolidate per-chunk catalogues into candidates/
    # var_pattern = os.path.join(cfg.candidates_dir, f"{cfg.ms_base}*_var_candidates.csv")
    # box_pattern = os.path.join(cfg.candidates_dir, f"{cfg.ms_base}*_chunk_*_boxcar_candidates.csv")
    # candidates.consolidate_chunk_catalogues(
    #     ms_base=cfg.ms_base,
    #     out_dir=cfg.candidates_dir,
    #     var_csv_pattern=var_pattern,
    #     box_csv_pattern=box_pattern,
    #     remove_chunk_catalogues=True
    # )

    for trial_dm in dm_trials:
        cfg.current_dm = trial_dm
        fd_core.consolidate_catalogues(cfg)
    log_memory("After consolidate_chunk_catalogues (Finished)")


def main(argv: list[str] | None = None):
    if argv is None:
        argv = sys.argv[1:]

    log_level = logging.INFO
    if "-v" in argv or "--verbose" in argv or any(arg.startswith("--verbosity") for arg in argv):
        log_level = logging.DEBUG
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log_memory("Main Startup")

    if len(argv) > 0 and argv[0] == "aggregate":
        return aggregate_main(argv[1:])
    elif len(argv) > 0 and argv[0] in ("aggregate_obs", "aggregate-obs"):
        return aggregate_obs_main(argv[1:])
    elif len(argv) > 0 and argv[0] == "periodicity":
        from fastducc import nufft_periodicity
        args = build_cli_periodicity(argv[1:])
        nufft_periodicity.run_periodicity(args)
        return 0
    elif len(argv) > 0 and argv[0] in ("cube", "make-cube", "make_cube"):
        from fastducc import cube
        return cube.main(argv[1:])
    
    args = build_cli(argv)
    if getattr(args, "make_cube", False):
        from fastducc import cube
        from fastducc.fd_types import CubeConfig
        dm_list = getattr(args, "dm_list", None)
        dm_trials = dm_list if (dm_list is not None and len(dm_list) > 0) else [getattr(args, "dm", None) if getattr(args, "dm", None) is not None else 0.0]
        for trial_dm in dm_trials:
            out_cube = getattr(args, "out_cube", None) if len(dm_trials) == 1 else None
            cfg = CubeConfig.from_namespace(args, dm=float(trial_dm), out_cube=out_cube)
            cube.make_dirty_cube(cfg=cfg)
        return 0
    res = run_pipeline(args)
    return res if isinstance(res, int) else 0


    
if __name__ == '__main__':
    main()
