import argparse
import glob
import os
import shutil
import sys
from typing import Iterable, Tuple, List, Dict, Any, Optional
from dataclasses import dataclass

from numba import njit, prange
import numpy as np
import math
from tqdm import tqdm

import astropy.constants as const
import astropy.units as u
from astropy.visualization.wcsaxes import WCSAxes
from astropy.wcs import WCS
from astropy.io import fits
from astropy.table import Table, vstack

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from scipy.ndimage import maximum_filter

from casacore.tables import table
try:
    import ducc0
except Exception as e:
    raise RuntimeError('ducc0 is required') from e


from fastducc import wcs as ducc_wcs
from fastducc.types import Config, WelfordState
from fastducc import filters, kernels, candidates, ms_utils, detection, imaging
from fastducc import core as fd_core
from fastducc.catalogues import get_psrcat_csv_path, get_racs_vot_path

psrcat_path = get_psrcat_csv_path()
racs_path   = get_racs_vot_path()

def build_cli():

    parser = argparse.ArgumentParser(description='Image MS in time chunks using ducc0 wgridder')
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
    parser.add_argument('--do-wgridding', dest='do_wgridding', action='store_true', default=True)
    parser.add_argument('--no-wgridding', dest='do_wgridding', action='store_false')
    parser.add_argument('--nthreads', type=int, default=0)
    parser.add_argument('--verbosity', type=int, default=0)
    parser.add_argument('--boxcar-widths', nargs='+', type=int,
                        help='Widths (in samples) to perform boxcar search over (default: [1, 2, 4, 8, 16, 32, 64])',
                        default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument('--threshold-sigma', type=float, default=8.0, help="S/N threshold to use for boxcar detections (default: 8.0)")
    parser.add_argument('--do-plot', action='store_true')
    parser.add_argument('--var-threshold-sigma', type=float, default=8.0, help="S/N threshold to use for boxcar detections (default: 8.0)")
    parser.add_argument('--var-keep-top-k', type=int, default=None)
    parser.add_argument('--rms-clip-sigma', type=float, default=3.0)
    parser.add_argument('--nms-radius', type=int, default=6)
    # --- Search toggles ---
    parser.add_argument(
        '--enable-boxcar', dest='enable_boxcar', action='store_true',
        help='Enable per-width boxcar transient search (default: enabled)'
    )
    parser.add_argument(
        '--disable-boxcar', dest='enable_boxcar', action='store_false',
        help='Disable per-width boxcar transient search'
    )
    parser.set_defaults(enable_boxcar=True)
    parser.add_argument(
        '--enable-var', dest='enable_var', action='store_true',
        help='Enable variance (std-map/Welford) variability search (default: enabled)'
    )
    parser.add_argument(
        '--disable-var', dest='enable_var', action='store_false',
        help='Disable variance (std-map/Welford) variability search'
    )
    parser.set_defaults(enable_var=True)
    parser.add_argument(
        '--enable-var-chunk', dest='enable_var_chunk', action='store_true',
        help='Enable variance (std-map/Welford) variability search per chunk (default: disabled)'
    )
    parser.add_argument(
        '--disable-var-chunk', dest='enable_var_chunk', action='store_false',
        help='Disable variance (std-map/Welford) variability search per chunk'
    )
    parser.set_defaults(enable_var_chunk=False)
    parser.add_argument('--save-var-lightcurves',  dest='save_var_lightcurves',  action='store_true', help='Save lightcurves for variance candidates')
    parser.add_argument('--no-save-var-lightcurves',  dest='save_var_lightcurves',  action='store_false')
    parser.set_defaults(save_var_lightcurves=True)
    parser.add_argument('--save-full-var-lightcurves',  dest='save_full_var_lightcurves',  action='store_true', help='Save full lightcurves (over all chunks) for variance candidates')
    parser.add_argument('--no-save-full-var-lightcurves',  dest='save_full_var_lightcurves',  action='store_false')
    parser.set_defaults(save_full_var_lightcurves=False)    
    parser.add_argument('--save-var-snippets',     dest='save_var_snippets',     action='store_true', help='Save snippet products (PNG/GIF/FITS) for variance candidates')
    parser.add_argument('--no-save-var-snippets',  dest='save_var_snippets',     action='store_false')
    parser.set_defaults(save_var_snippets=True)
    parser.add_argument('--save-box-lightcurves',  dest='save_box_lightcurves',  action='store_true', help='Save lightcurves for boxcar candidates')
    parser.add_argument('--no-save-box-lightcurves',  dest='save_box_lightcurves',  action='store_false')
    parser.set_defaults(save_box_lightcurves=True)
    parser.add_argument('--save-box-snippets',     dest='save_box_snippets',     action='store_true', help='Save snippet products (PNG/GIF/FITS) for boxcar candidates')
    parser.add_argument('--no-save-box-snippets',  dest='save_box_snippets',     action='store_false')
    parser.set_defaults(save_box_snippets=True)

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
    # --- SLURM-only cluster params ---
    parser.add_argument('--slurm-partition', default=None, help='SLURM partition/queue')
    parser.add_argument('--slurm-account',  default=None, help='SLURM account/project')
    parser.add_argument('--slurm-cores-per-worker', type=int, default=1)
    parser.add_argument('--slurm-mem', default='8GB', help='Memory per worker (e.g., 8GB)')
    parser.add_argument('--slurm-walltime', default='01:00:00')
    parser.add_argument('--slurm-job-extra', nargs='*', default=[],
                        help='Extra SLURM directives, e.g. ["--exclusive"]')
    parser.add_argument('--slurm-interface', default=None,
                        help='Network interface name (if needed for TCP comms)')


    args = parser.parse_args()

    return args

def build_cli_aggregate(argv=None):
    p = argparse.ArgumentParser(
        prog="fastducc aggregate",
        description="Aggregate per-beam candidate catalogues",
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

    args = p.parse_args(argv)
    return args


def build_cli_aggregate_obs(argv=None):
    p = argparse.ArgumentParser(
        prog="aggregate_obs",
        description="Aggregate per-scan cross-beam super summaries across an observation (SBID)."
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
    return p.parse_args(argv)


def make_config(args, paths) -> Config:
    ms_base, candidates_dir, chunk_prefix_root, all_prefix_root = paths
    ra0_rad, dec0_rad, _ = ms_utils.get_phase_center(args.msname, field_name=None)
    pix_rad = args.pixsize_arcsec / 206265.0
    return Config(
        msname=args.msname,
        npix_x=args.npix_x, npix_y=args.npix_y,
        pix_rad=pix_rad, ra0_rad=ra0_rad, dec0_rad=dec0_rad,
        epsilon=args.epsilon, do_wgridding=args.do_wgridding,
        nthreads=args.nthreads, verbosity=args.verbosity,
        corr_mode=args.corr_mode, basis=args.basis, single_pol=args.single_pol,
        data_column=args.data_column,
        enable_var=args.enable_var, enable_boxcar=args.enable_boxcar,
        enable_var_chunk=args.enable_var_chunk,
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
        chunk_prefix_root=chunk_prefix_root, all_prefix_root=all_prefix_root
    )

def main_serial(args):
    #args = build_cli()

    ms_base, candidates_dir, chunk_prefix_root, all_prefix_root = ms_utils.derive_paths(args.msname)
    cfg = make_config(args, (ms_base, candidates_dir, chunk_prefix_root, all_prefix_root))
    t_main, total_chunks, time_col = ms_utils.open_ms(args.msname)
    # Discover total number of time chunks via iter
    print(f"Found {total_chunks} time chunks in MS: {args.msname}")

    # Convert arcsec to radians
    pix_rad = args.pixsize_arcsec / 206265.0
    ra0_rad, dec0_rad, used_field = ms_utils.get_phase_center(args.msname, field_name=None)

    start = 0
    chunk_size = max(1, args.chunk_size)
    chunk_id = 0

    wf = fd_core.init_welford(cfg)

    #t_main = table(args.msname, readonly=True)
    while start < total_chunks:
        end = min(start + chunk_size - 1, total_chunks - 1)

        print(f"[Chunk {chunk_id}] imaging time_idx {start}..{end}")
        times, cube = imaging.image_time_samples(msname=args.msname,
                                         t_main=t_main,
                                         start_time_idx=start,
                                         end_time_idx=end,
                                         corr_mode=args.corr_mode,
                                         basis=args.basis,
                                         single_pol=args.single_pol,
                                         data_column=args.data_column,
                                         npix_x=args.npix_x,
                                         npix_y=args.npix_y,
                                         pixsize_x=pix_rad,
                                         pixsize_y=pix_rad,
                                         epsilon=args.epsilon,
                                         do_wgridding=args.do_wgridding,
                                         nthreads=args.nthreads,
                                         verbosity=args.verbosity,
                                         do_plot=args.do_plot,
                                         )


        # Variance/Welford chunk
        var_ann = fd_core.process_variance_cube_chunk(cfg, times, cube, wf, start)

        # Boxcar chunk
        box_ann = fd_core.process_boxcar_chunk(cfg, times, cube, start)

        start = end + 1
        chunk_id += 1

    # Finalisation
    fd_core.finalise_welford(cfg, wf, times, cube)
    fd_core.consolidate_catalogues(cfg)

def aggregate_main(argv=None):

    args = build_cli_aggregate(argv)
    
    candidates.aggregate_observation(
        obs_root=args.obs_root,
        kind=args.kind,
        time_tol_s=args.time_tol,
        sky_tol_arcsec=args.sky_tol_arcsec,
        out_dir=args.outdir,
        pattern=args.pattern,
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
        psrcat_csv_path=psrcat_path, #TODO: make these cli enabled
        racs_vot_path=racs_path,
        match_radius_arcsec=40.0,
        simbad_enable=False,
    )
    return 0

    

def main():

    if sys.argv[1] == "aggregate":
        aggregate_main(sys.argv)
        return None

    elif  sys.argv[1] == "aggregate_obs":
        aggregate_obs_main(sys.argv)
        return None
    
    args = build_cli()

    if args.parallel_mode == "serial":
        main_serial(args)
        return None

    ms_base, candidates_dir, chunk_prefix_root, all_prefix_root = ms_utils.derive_paths(args.msname)
    cfg = make_config(args, (ms_base, candidates_dir, chunk_prefix_root, all_prefix_root))

    # Compute chunk bounds (using casacore once in the driver)
    t_main = table(cfg.msname, readonly=True)
    time_col = 'TIME_CENTROID' if 'TIME_CENTROID' in set(t_main.colnames()) else 'TIME'
    it = t_main.iter([time_col], sort=True)
    total_chunks = sum(1 for _ in it)
    print(f"Found {total_chunks} time chunks in MS: {cfg.msname}")
    t_main.close()

    chunk_size = max(1, args.chunk_size)
    chunk_bounds = []
    start = 0
    while start < total_chunks:
        end = min(start + chunk_size - 1, total_chunks - 1)
        chunk_bounds.append((start, end))
        start = end + 1

    # Execute chunks: serial, Dask-Local, or Dask-SLURM
    agg_list = []
    if args.parallel_mode == 'serial':
        for (start, end) in chunk_bounds:
            print(f"[Serial] Chunk {start}..{end}")
            times, cube, c, m, M2 = fd_core.process_chunk_task(cfg, ms_base, candidates_dir, start, end)
            agg_list.append((times, cube if cfg.save_full_var_lightcurves else None, c, m, M2))

    elif args.parallel_mode == 'dask-local':
        from dask.distributed import Client, LocalCluster
        n_workers = args.dask_workers if args.dask_workers > 0 else None
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=args.threads_per_worker,
            processes=(args.dask_scheduler == 'processes')
        )
        with Client(cluster) as client:
            futures = [client.submit(fd_core.process_chunk_task, cfg, ms_base, candidates_dir, s, e)
                       for (s, e) in chunk_bounds]
            agg_list = client.gather(futures)
        cluster.close()

    elif args.parallel_mode == 'dask-slurm':
        # SLURMCluster (requires dask_jobqueue)
        from dask_jobqueue import SLURMCluster
        cluster = SLURMCluster(
            queue=args.slurm_partition,
            account=args.slurm_account,
            cores=args.slurm_cores_per_worker,
            memory=args.slurm_mem,
            walltime=args.slurm_walltime,
            job_extra_directives=args.slurm_job_extra,
            interface=args.slurm_interface
        )
        # scale to number of workers (or adapt if dask_workers==0)
        if args.dask_workers and args.dask_workers > 0:
            cluster.scale(args.dask_workers)
        else:
            # adaptively allocate between 1 and len(chunk_bounds) workers
            cluster.adapt(minimum=1, maximum=max(1, len(chunk_bounds)))
        from dask.distributed import Client
        with Client(cluster) as client:
            futures = [client.submit(fd_core.process_chunk_task, cfg, ms_base, candidates_dir, s, e)
                       for (s, e) in chunk_bounds]
            agg_list = client.gather(futures)
        cluster.close()

    else:
        raise ValueError(f"Unknown parallel_mode: {args.parallel_mode}")

    _ = fd_core.finalise_welford_parallel(cfg, agg_list, run_final_variance=True)

    # Consolidate per-chunk catalogues into candidates/
    var_pattern = os.path.join(cfg.candidates_dir, f"{cfg.ms_base}*_var_candidates.csv")
    box_pattern = os.path.join(cfg.candidates_dir, f"{cfg.ms_base}_chunk_*_boxcar_candidates.csv")
    candidates.consolidate_chunk_catalogues(
        ms_base=cfg.ms_base,
        out_dir=cfg.candidates_dir,
        var_csv_pattern=var_pattern,
        box_csv_pattern=box_pattern,
        remove_chunk_catalogues=True
    )

    
if __name__ == '__main__':
    main()
