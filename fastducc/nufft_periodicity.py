# fastducc/nufft_periodicity.py
"""
3-D DUCC NUFFT periodicity search (u,v,t) -> (l,m,f).

- Block-wise adjoint NUFFT using ducc0.nufft (nonuniform -> uniform).
- Optional Dask Futures for parallel per-block work.
- Each worker writes complex partials X(l,m,f>=fmin) to disk with t0 (block ref. time).
- Driver phase-stitches via exp(-i 2pi f t0) and accumulates into a complex memmap.
- Writes a FITS cube of power |sum X|^2 with spatial TAN WCS (from ducc_wcs)
  and a spectral axis CTYPE3='FREQ' (Hz).
- Detection options: spatial S/N detection on the spin-frequency cube with harmonic summing.
- WCS annotation of candidates (RA/Dec + srcname) using ducc_wcs + ms_utils.get_phase_center.
- Outputs periodicity candidates catalogues (CSV + VOT).
- Outputs quicklook products: max-over-f power map, best-freq map, and optional PNG.
"""

from __future__ import annotations

import os
import math
import time
import uuid
from typing import Optional, Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from casacore.tables import table
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt

from ducc0 import nufft as ducc_nufft

from fastducc import ms_utils
from fastducc import wcs as ducc_wcs
from fastducc import detection
from fastducc import candidates as cand_mod
from fastducc import constants

C = constants.c
K_DM = constants.K_DM


def next_pow2(n: int) -> int:
    return 1 << int(math.ceil(math.log2(max(1, n))))



def wrap_pi(x: np.ndarray) -> np.ndarray:
    """Map radians to [-pi,pi)."""
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def _process_block_and_save(
    ms_path: str,
    row_ids_block: np.ndarray,
    spw: int,
    pol: int,
    data_col: str,
    chan_sel: Optional[Tuple[int, Optional[int]]],
    freqs_hz: np.ndarray,
    npix: int,
    dl: float,            # rad/pix (spatial)
    Nf: int,
    dt: float,
    df: float,
    fgrid: np.ndarray,    # full f-grid
    kmin: int,            # index on positive side where f>=fmin
    eps: float,
    nthreads: int,
    dm_pc_cm3: float,
    out_dir: Optional[str],
    block_id: int,
) -> Dict:
    """
    Per-block worker:
      - Load rows for this block.
      - For each channel: DM time shift; UV scale by lambda; build omega in radians.
      - Run adjoint 3-D NUFFT to (l,m,f), keep f>=fmin on +ve side; sum across channels.
    """
    T = table(ms_path, readonly=True)
    colnames = set(T.colnames())
    time_col = 'TIME_CENTROID' if 'TIME_CENTROID' in colnames else 'TIME'

    min_row = int(np.min(row_ids_block))
    max_row = int(np.max(row_ids_block))
    num_rows = max_row - min_row + 1
    rel_rows = row_ids_block - min_row

    uvw_m = T.getcol('UVW', startrow=min_row, nrow=num_rows)[rel_rows]
    times = T.getcol(time_col, startrow=min_row, nrow=num_rows)[rel_rows]
    data = T.getcol(data_col, startrow=min_row, nrow=num_rows)[rel_rows]
    flags = T.getcol('FLAG', startrow=min_row, nrow=num_rows)[rel_rows]
    T.close()

    if chan_sel is None:
        vis = data[:, :, pol].astype(np.complex64)
        flg = flags[:, :, pol]
    else:
        s, e = chan_sel
        vis = data[:, s:e, pol].astype(np.complex64)
        flg = flags[:, s:e, pol]

    nrow_blk, nchan = vis.shape
    if nrow_blk == 0:
        raise RuntimeError("Empty block.")

    t0 = float(times[0])
    t_rel = (times - t0).astype(np.float64)

    pos = (fgrid >= 0.0)
    fkeep = fgrid[pos][kmin:]
    nkeep = fkeep.size

    X_blk_keep = np.zeros((npix, npix, nkeep), dtype=np.complex64)
    out_buf = np.zeros((npix, npix, Nf), dtype=np.complex64)
    eps_use = max(float(eps), 1e-5)

    nu_mhz = freqs_hz * 1e-6
    tau_dm = (K_DM * dm_pc_cm3 / (nu_mhz ** 2)) * 1e-3  # seconds

    for ic in range(nchan):
        good = ~flg[:, ic]
        if not np.any(good):
            continue

        u_m = uvw_m[good, 0]
        v_m = uvw_m[good, 1]
        nu = float(freqs_hz[ic])
        lam = C / nu
        u = u_m / lam
        v = v_m / lam

        t_rel_dm = t_rel[good] - float(tau_dm[ic])

        omega_u = 2.0 * math.pi * u * dl
        omega_v = 2.0 * math.pi * v * dl
        omega_f = -2.0 * math.pi * t_rel_dm * df
        coord = np.stack([wrap_pi(omega_u), wrap_pi(omega_v), wrap_pi(omega_f)], axis=1).astype(np.float64)

        y = vis[good, ic]

        out_buf.fill(0.0)
        ducc_nufft.nu2u(
            points=y,
            coord=coord,
            forward=False,
            epsilon=eps_use,
            nthreads=1,
            out=out_buf,
            fft_order=True
        )

        # Use basic slicing [kmin : Nf//2] to create a zero-copy view instead of boolean advanced indexing
        X_blk_keep += out_buf[:, :, kmin : Nf // 2]

    part_path = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        part_name = f"periodicity_block_{block_id:06d}_{uuid.uuid4().hex}.npy"
        part_path = os.path.join(out_dir, part_name)
        np.save(part_path, X_blk_keep)
        del out_buf, X_blk_keep
        import gc
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        return {
            "path": part_path,
            "block_id": int(block_id),
            "t0": float(t0),
            "npix": int(npix),
            "nkeep": int(nkeep),
        }

    return {
        "X": X_blk_keep,
        "path": None,
        "block_id": int(block_id),
        "t0": float(t0),
        "npix": int(npix),
        "nkeep": int(nkeep),
    }

def _process_block_helper(kw: dict) -> dict:
    import faulthandler
    faulthandler.enable()
    try:
        return _process_block_and_save(**kw)
    except Exception as e:
        import traceback
        print(f"[Worker Error Block {kw.get('block_id')}] Exception: {e}", flush=True)
        traceback.print_exc()
        raise e

def run_periodicity(args=None) -> str:
    """
    Driver for periodicity search.
    Returns path to the power cube FITS written (primary product).
    """
    if args is None:
        from fastducc.fastducc_run import build_cli_periodicity
        args = build_cli_periodicity()

    # --- prepare MS row ordering ---
    T = table(args.msname, readonly=True, lockoptions='usernoread')
    colnames = set(T.colnames())
    time_col = 'TIME_CENTROID' if 'TIME_CENTROID' in colnames else 'TIME'
    all_times = T.getcol(time_col)
    row_ids_sorted = np.argsort(all_times)
    T.close()

    # --- global time/frequency grid for the spin axis ---
    ut = np.unique(all_times[row_ids_sorted])
    if ut.size < 2:
        raise RuntimeError("Not enough time samples for periodicity axis.")
    dt = float(np.median(np.diff(ut)))
    Nf = args.nfft_t if args.nfft_t is not None else next_pow2(ut.size)
    df = 1.0 / (Nf * dt)
    fgrid = np.fft.fftfreq(Nf, d=dt)
    pos = (fgrid >= 0.0)
    kmin = int(np.searchsorted(fgrid[pos], float(args.fmin), side='left'))
    fkeep = fgrid[pos][kmin:]     # Hz, f >= fmin
    nkeep = fkeep.size

    # --- spatial grid ---
    npix = int(args.npix)
    dl = ducc_wcs.arcsec_to_rad(float(args.pixscale_arcsec))

    # phase center & output paths consistent with your module
    ra0_rad, dec0_rad, used_field = ms_utils.get_phase_center(args.msname, field_name=None)
    ms_base, candidates_dir, chunk_prefix_root, all_prefix_root = ms_utils.derive_paths(args.msname)

    partials_dir = os.path.join(candidates_dir, args.partials_dir)
    os.makedirs(partials_dir, exist_ok=True)

    # block row ranges
    nrows = row_ids_sorted.size
    bounds = []
    start = 0
    while start < nrows:
        end = min(nrows - 1, start + args.block_rows - 1)
        bounds.append((start, end))
        start = end + 1

    if getattr(args, 'max_blocks', None) is not None and args.max_blocks > 0:
        bounds = bounds[:int(args.max_blocks)]

    # channel selection & SPW frequency loading
    chan_sel = None
    if args.chan_start is not None or args.chan_end is not None:
        chan_sel = (0 if args.chan_start is None else int(args.chan_start),
                    None if args.chan_end is None else int(args.chan_end))

    spw_table_path = os.path.join(args.msname, "SPECTRAL_WINDOW")
    Tspw = table(spw_table_path, readonly=True)
    if Tspw.nrows() == 0 and args.msname.endswith(".uvsub.ms"):
        alt_ms = args.msname[:-9] + ".ms"
        if os.path.exists(alt_ms):
            Tspw.close()
            spw_table_path = os.path.join(alt_ms, "SPECTRAL_WINDOW")
            Tspw = table(spw_table_path, readonly=True)
    freqs_full = Tspw.getcol('CHAN_FREQ')[int(args.spw)]
    Tspw.close()

    if chan_sel is None:
        freqs_hz = freqs_full
    else:
        s, e = chan_sel
        freqs_hz = freqs_full[s:e]

    metas: List[Dict] = []
    num_blocks = len(bounds)

    print(f"\n==================================================", flush=True)
    print(f"[Periodicity] Starting NUFFT Periodicity Search", flush=True)
    print(f"[Periodicity] Target MS         : {args.msname}", flush=True)
    print(f"[Periodicity] Total Rows        : {nrows}", flush=True)
    print(f"[Periodicity] Unique Timestamps : {ut.size} (dt = {dt:.4f} s, total T = {ut[-1]-ut[0]:.2f} s)", flush=True)
    print(f"[Periodicity] Frequency Grid   : Nf = {Nf}, df = {df:.6f} Hz", flush=True)
    print(f"[Periodicity] Keeping Spin Freqs: {nkeep} bins in range [{fkeep[0]:.4f}, {fkeep[-1]:.4f}] Hz (fmin={args.fmin} Hz)", flush=True)
    print(f"[Periodicity] Spatial Grid      : {npix}x{npix} @ {args.pixscale_arcsec}\" ({dl*206265:.1f}\"/pix)", flush=True)
    print(f"[Periodicity] DM Search Value   : {args.dm} pc cm^-3", flush=True)
    print(f"[Periodicity] Partitioning      : {num_blocks} block(s) of ~{args.block_rows} rows (nthreads={args.nthreads})", flush=True)
    print(f"==================================================\n", flush=True)

    t_start = time.time()

    sum_mmap_path = os.path.join(candidates_dir, args.sum_memmap)
    mm = np.memmap(sum_mmap_path, mode='w+', dtype=np.complex64, shape=(npix, npix, nkeep))
    mm[:] = 0.0

    # --- execute per-block NUFFT ---
    import shutil
    jobfs = os.environ.get('JOBFS')
    
    # Check if JOBFS is available AND has at least 10 GB free space.
    # Slurm interactive jobs without --tmp often default JOBFS to a tiny 250MB partition!
    has_enough_jobfs = False
    if jobfs and os.path.exists(jobfs):
        try:
            free_bytes = shutil.disk_usage(jobfs).free
            if free_bytes >= 10 * (1024**3):
                has_enough_jobfs = True
            else:
                print(f"[Periodicity] Notice: $JOBFS ({jobfs}) has only {free_bytes / (1024**2):.1f} MB free. Falling back to project storage on /fred.", flush=True)
        except Exception:
            pass

    is_distributed = args.parallel_mode in ["dask-local", "dask-slurm"] and args.scheduler_address is not None
    
    if has_enough_jobfs and not is_distributed:
        out_partials_dir = os.path.join(jobfs, f"fastducc_partials_{uuid.uuid4().hex}")
    else:
        out_partials_dir = os.path.join(os.path.dirname(sum_mmap_path), f"partials_{uuid.uuid4().hex}")
    os.makedirs(out_partials_dir, exist_ok=True)

    if args.parallel_mode == "serial":
        workers = max(1, int(args.nthreads))
        try:
            import psutil
            total_ram_gb = psutil.virtual_memory().total / (1024**3)
            # Each worker needs ~18-20 GB peak RAM (9.6GB out_buf + 4.6GB X_blk + C++ NUFFT workspace).
            # The main accumulation process needs ~8 GB.
            main_ram_gb = 8.0
            worker_ram_gb = 20.0
            avail_ram = max(0.0, total_ram_gb - main_ram_gb)
            max_safe_workers = max(1, int(avail_ram / worker_ram_gb))
            if workers > max_safe_workers:
                print(f"[Periodicity] Auto-scaling worker processes from {workers} to {max_safe_workers} to fit {total_ram_gb:.1f} GB RAM safely ({worker_ram_gb} GB/worker)...", flush=True)
                workers = max_safe_workers
        except Exception:
            pass
        # Never spin up more workers than there are blocks — idle workers waste base memory
        workers = min(workers, num_blocks)

        if workers > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            tasks = []
            for i, (lo, hi) in enumerate(bounds):
                kw = dict(
                    ms_path=args.msname,
                    row_ids_block=row_ids_sorted[lo:hi + 1],
                    spw=int(args.spw),
                    pol=int(args.pol),
                    data_col=args.data_col,
                    chan_sel=chan_sel,
                    freqs_hz=freqs_hz,
                    npix=npix, dl=dl, Nf=Nf, dt=dt, df=df,
                    fgrid=fgrid, kmin=kmin,
                    eps=float(args.eps),
                    nthreads=1,
                    dm_pc_cm3=float(args.dm),
                    out_dir=out_partials_dir,
                    block_id=i,
                )
                tasks.append(kw)

            import multiprocessing as mp
            ctx = mp.get_context('forkserver')
            print(f"[Periodicity] Launching {workers} parallel worker processes (mp_context=forkserver) for {num_blocks} blocks...", flush=True)
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
                futures = {executor.submit(_process_block_helper, kw): kw["block_id"] for kw in tasks}
                completed = 0
                for fut in as_completed(futures):
                    meta = fut.result()
                    path = meta["path"]
                    t0 = meta["t0"]
                    blk_id = meta["block_id"]
                    
                    # Load full array into RAM. Main thread only holds one 4.6GB block at a
                    # time; chunked accumulation keeps temporary arrays to ~300 MB each.
                    X_blk = np.load(path)
                    phase = np.exp(1j * 2.0 * np.pi * fkeep * t0).astype(np.complex64)
                    _chunk = 256
                    for _ks in range(0, X_blk.shape[2], _chunk):
                        _ke = min(_ks + _chunk, X_blk.shape[2])
                        mm[:, :, _ks:_ke] += X_blk[:, :, _ks:_ke] * phase[None, None, _ks:_ke]
                    
                    # Cleanup
                    del X_blk, meta
                    if os.path.exists(path):
                        os.remove(path)
                    completed += 1
                    elapsed_tot = time.time() - t_start
                    eta_tot = (elapsed_tot / completed) * (num_blocks - completed)
                    print(f"[Periodicity] Completed {completed:02d}/{num_blocks:02d} blocks (block_id={blk_id:02d}) | Elapsed: {elapsed_tot:6.1f}s | ETA: {eta_tot:6.1f}s", flush=True)
        else:
            for i, (lo, hi) in enumerate(bounds):
                t_blk_0 = time.time()
                row_ids_block = row_ids_sorted[lo:hi + 1]
                meta = _process_block_and_save(
                    ms_path=args.msname,
                    row_ids_block=row_ids_block,
                    spw=int(args.spw),
                    pol=int(args.pol),
                    data_col=args.data_col,
                    chan_sel=chan_sel,
                    freqs_hz=freqs_hz,
                    npix=npix, dl=dl, Nf=Nf, dt=dt, df=df,
                    fgrid=fgrid, kmin=kmin,
                    eps=float(args.eps),
                    nthreads=1,
                    dm_pc_cm3=float(args.dm),
                    out_dir=None,
                    block_id=i,
                )
                X_blk = meta["X"]
                t0 = meta["t0"]
                phase = np.exp(1j * 2.0 * np.pi * fkeep * t0).astype(np.complex64)
                mm += X_blk * phase[None, None, :]
                del X_blk, meta

                dt_blk = time.time() - t_blk_0
                elapsed_tot = time.time() - t_start
                eta_tot = (elapsed_tot / (i + 1)) * (num_blocks - (i + 1))
                print(f"[Periodicity] Block {i+1:02d}/{num_blocks:02d} done (rows {lo:7d}..{hi:7d}) in {dt_blk:5.2f}s | Elapsed: {elapsed_tot:6.1f}s | ETA: {eta_tot:6.1f}s", flush=True)

    elif args.parallel_mode in ["dask-local", "dask-slurm"]:
        from dask.distributed import Client, LocalCluster, as_completed

        cluster = None
        if args.scheduler_address is not None:
            print(f"[Periodicity] Connecting to existing Dask scheduler at {args.scheduler_address}...", flush=True)
            client = Client(args.scheduler_address)
        elif args.parallel_mode == "dask-slurm":
            try:
                from dask_jobqueue import SLURMCluster
            except ImportError:
                raise RuntimeError("dask_jobqueue is required for --parallel-mode=dask-slurm. Install with `pip install dask-jobqueue`.")
            
            import sys
            slurm_interface = args.slurm_interface
            if slurm_interface is None:
                try:
                    import psutil
                    if 'ib0' in psutil.net_if_addrs():
                        slurm_interface = 'ib0'
                except Exception:
                    pass
            print(f"[Periodicity] Launching Dask SLURMCluster (mem={args.slurm_mem}, cores={args.slurm_cores_per_worker}, interface={slurm_interface})...", flush=True)
            scheduler_opts = {"dashboard_address": ":0"}
            if slurm_interface:
                scheduler_opts["interface"] = slurm_interface
            prologue = [
                "module load python-scientific/3.11.5-foss-2023b 2>/dev/null || true",
                "unset PYTHONPATH",
                f"source {os.path.dirname(sys.executable)}/activate 2>/dev/null || true",
                "export OMP_NUM_THREADS=1",
                "export OPENBLAS_NUM_THREADS=1"
            ]
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
                scheduler_options=scheduler_opts,
                processes=1,
                worker_extra_args=["--nthreads", "1", "--memory-limit", "0"],
            )
            if args.dask_workers and args.dask_workers > 0:
                cluster.scale(args.dask_workers)
            else:
                cluster.adapt(minimum=1, maximum=max(1, num_blocks))
            client = Client(cluster)
        else:
            n_workers = args.dask_workers if args.dask_workers > 0 else None
            if n_workers is None:
                try:
                    import psutil
                    total_ram_gb = psutil.virtual_memory().total / (1024**3)
                    n_workers = max(1, int((total_ram_gb - 8.0) / 20.0))
                    if args.nthreads > 0:
                        n_workers = min(int(args.nthreads), n_workers)
                except Exception:
                    n_workers = 2
            n_workers = min(n_workers, num_blocks)
            print(f"[Periodicity] Launching Dask LocalCluster with {n_workers} worker processes...", flush=True)
            cluster = LocalCluster(
                n_workers=n_workers,
                threads_per_worker=args.threads_per_worker,
                processes=(args.dask_scheduler == 'processes'),
                memory_limit='30GB'
            )
            client = Client(cluster)

        with client:
            futures = []
            for i, (lo, hi) in enumerate(bounds):
                row_ids_block = row_ids_sorted[lo:hi + 1]
                fut = client.submit(
                    _process_block_and_save,
                    args.msname, row_ids_block, int(args.spw), int(args.pol),
                    args.data_col, chan_sel, freqs_hz,
                    npix, dl, Nf, dt, df, fgrid, kmin,
                    float(args.eps), 1,
                    float(args.dm),
                    out_partials_dir,
                    i
                )
                futures.append(fut)

            completed = 0
            for fut in as_completed(futures):
                meta = fut.result()
                path = meta["path"]
                t0 = meta["t0"]
                blk_id = meta["block_id"]
                
                # Load full array into RAM. Main thread only holds one 4.6GB block at a
                # time; chunked accumulation keeps temporary arrays to ~300 MB each.
                X_blk = np.load(path)
                phase = np.exp(1j * 2.0 * np.pi * fkeep * t0).astype(np.complex64)
                _chunk = 256
                for _ks in range(0, X_blk.shape[2], _chunk):
                    _ke = min(_ks + _chunk, X_blk.shape[2])
                    mm[:, :, _ks:_ke] += X_blk[:, :, _ks:_ke] * phase[None, None, _ks:_ke]
                
                # Cleanup
                del X_blk, meta
                if os.path.exists(path):
                    os.remove(path)
                completed += 1
                elapsed_tot = time.time() - t_start
                eta_tot = (elapsed_tot / completed) * (num_blocks - completed)
                print(f"[Periodicity] Completed {completed:02d}/{num_blocks:02d} blocks (block_id={blk_id:02d}) | Elapsed: {elapsed_tot:6.1f}s | ETA: {eta_tot:6.1f}s", flush=True)

        if cluster is not None:
            cluster.close()
    else:
        raise ValueError(f"Unknown parallel-mode: {args.parallel_mode}")

    print(f"\n[Periodicity] NUFFT block computations & accumulation completed in {time.time() - t_start:.2f}s.", flush=True)

    t_stitch_0 = time.time()
    print(f"[Periodicity] Computing 3D power cube |X(l,m,f)|^2 and spatial FFTShift...", flush=True)
    # power cube (fftshift spatial axes so zero-frequency is centered at (npix/2, npix/2))
    P = np.fft.fftshift((np.abs(mm) ** 2), axes=(0, 1)).astype(np.float32)
    del mm
    print(f"[Periodicity] Phase stitching and power cube construction done in {time.time() - t_stitch_0:.2f}s.", flush=True)

    # --- write 3D FITS cube (power) ---
    w2 = ducc_wcs._build_fullframe_wcs(
        npix_x=npix, npix_y=npix,
        ra0_rad=ra0_rad, dec0_rad=dec0_rad,
        pixscale_rad=dl,
        ra_sign=-1, dec_sign=-1,
        radesys="ICRS", equinox=None
    )
    w3 = WCS(naxis=3)
    w3.wcs.ctype[0] = w2.wcs.ctype[0]
    w3.wcs.ctype[1] = w2.wcs.ctype[1]
    w3.wcs.cunit[0] = w2.wcs.cunit[0]
    w3.wcs.cunit[1] = w2.wcs.cunit[1]
    w3.wcs.crpix[0] = w2.wcs.crpix[0]
    w3.wcs.crpix[1] = w2.wcs.crpix[1]
    w3.wcs.crval[0] = w2.wcs.crval[0]
    w3.wcs.crval[1] = w2.wcs.crval[1]
    w3.wcs.cdelt[0] = w2.wcs.cdelt[0]
    w3.wcs.cdelt[1] = w2.wcs.cdelt[1]
    w3.wcs.ctype[2] = 'FREQ'
    w3.wcs.cunit[2] = 'Hz'
    w3.wcs.crpix[2] = 1.0
    w3.wcs.crval[2] = float(fkeep[0]) if nkeep > 0 else 0.0
    w3.wcs.cdelt[2] = float(fkeep[1] - fkeep[0]) if nkeep > 1 else float(df)

    data_for_fits = np.transpose(P, (2, 1, 0)).copy()  # (freq, y, x)
    hdr = w3.to_header()
    hdu = fits.PrimaryHDU(data=data_for_fits, header=hdr)
    hdu.header['BUNIT'] = 'arb'
    hdu.header['COMMENT'] = 'Periodicity cube: |X(l,m,f)|^2'
    hdu.header['FMIN_HZ'] = float(fkeep[0]) if nkeep > 0 else 0.0
    hdu.header['DM'] = float(args.dm)

    out_cube_fits = os.path.join(candidates_dir, args.out_cube_fits)
    hdu.writeto(out_cube_fits, overwrite=True)
    print(f"[Periodicity] Wrote 3D power FITS cube -> {out_cube_fits}", flush=True)

    # --- optional quicklook products ---
    if args.write_products:
        print(f"[Periodicity] Generating quicklook max-power & best-frequency maps...", flush=True)
        # max over f
        maxpow = np.max(P, axis=2) if P.shape[2] > 0 else np.zeros((npix, npix), dtype=np.float32)
        argmax = np.argmax(P, axis=2).astype(np.int32) if P.shape[2] > 0 else np.zeros((npix, npix), dtype=np.int32)
        bestfreq = (fkeep[argmax]).astype(np.float32) if P.shape[2] > 0 else np.zeros((npix, npix), dtype=np.float32)

        hdr2 = w2.to_header()
        fits.writeto(os.path.join(candidates_dir, f"{ms_base}_periodicity_maxpow.fits"),
                     maxpow.astype(np.float32), header=hdr2, overwrite=True)
        hdrf = w2.to_header()
        hdrf['BUNIT'] = 'Hz'
        fits.writeto(os.path.join(candidates_dir, f"{ms_base}_periodicity_bestfreq.fits"),
                     bestfreq.astype(np.float32), header=hdrf, overwrite=True)

        # optional PNG
        try:
            plt.figure(figsize=(6, 5), dpi=150)
            plt.imshow(maxpow, origin='lower', cmap='inferno')
            plt.colorbar(label='max power')
            plt.title(f"{ms_base} periodicity max power")
            plt.tight_layout()
            plt.savefig(os.path.join(candidates_dir, f"{ms_base}_periodicity_maxpow.png"))
            plt.close()
        except Exception:
            pass

    # --- detection + catalogues ---
    if args.detect:
        print(f"[Periodicity] Running spatial SNR detection (threshold={args.threshold_sigma} sigma, nharm={args.nharm})...", flush=True)
        dets, _ = detection.detect_periodicity_spatial_snr(
            P, fkeep,
            threshold_sigma=float(args.threshold_sigma),
            nharm=int(args.nharm),
            spatial_estimator=str(args.spatial_estimator),
            clip_sigma=float(args.clip_sigma),
            keep_top_k=(int(args.keep_top_k) if args.keep_top_k is not None else None),
            valid_mask=None
        )

        # annotate with WCS (adds l,m, ra/dec, srcname)
        annotated = ducc_wcs.annotate_candidates_with_sky_coords(
            args.msname, dets,
            npix_x=npix, npix_y=npix,
            pixsize_x=dl, pixsize_y=dl,
            flip_u=True, flip_v=True,
            field_name=None
        )

        # add phase center field for provenance (matches other tables)
        for r in annotated:
            r["phase_center_field"] = used_field

        # write candidate table
        tab = cand_mod.candidates_to_astropy_table_periodicity(annotated)

        out_csv = args.out_csv or os.path.join(candidates_dir, f"{ms_base}_periodicity_all.csv")
        out_vot = args.out_vot or os.path.join(candidates_dir, f"{ms_base}_periodicity_all.vot")
        cand_mod.save_periodicity_candidates_table(tab, out_csv, out_vot)
        print(f"[Periodicity] Detection complete: Found {len(annotated)} candidate(s). Written to {out_csv}", flush=True)

    total_time = time.time() - t_start
    print(f"[Periodicity] Finished fastducc periodicity search in {total_time:.2f}s.\n", flush=True)
    return out_cube_fits
