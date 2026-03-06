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
import uuid
from typing import Optional, Tuple, List, Dict

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
    chan_sel: Optional[Tuple[int, int]],
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
    out_dir: str,
    block_id: int,
) -> Dict:
    """
    Per-block worker:
      - Load rows for this block, fetch SPW freqs.
      - For each channel: DM time shift; UV scale by lambda; build omega in radians.
      - Run adjoint 3-D NUFFT to (l,m,f), keep f>=fmin on +ve side; sum across channels.
      - Save complex partial + t0 for stitching.
    """
    T = table(ms_path, readonly=True)
    Ts = T[row_ids_block]
    uvw_m = Ts.getcol('UVW')                       # (R,3)
    colnames = set(T.colnames())
    time_col = 'TIME_CENTROID' if 'TIME_CENTROID' in colnames else 'TIME'
    times = Ts.getcol(time_col)
    data = Ts.getcol(data_col)                     # (R,nchan,npol)
    flags = Ts.getcol('FLAG')                      # (R,nchan,npol)
    T.close()

    Tspw = table(ms_path + '::SPECTRAL_WINDOW', readonly=True)
    freqs_full = Tspw.getcol('CHAN_FREQ')[spw]     # (nchan_all,)
    Tspw.close()

    if chan_sel is None:
        freqs_hz = freqs_full
        vis = data[:, :, pol].astype(np.complex64)
        flg = flags[:, :, pol]
    else:
        s, e = chan_sel
        freqs_hz = freqs_full[s:e]
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

        # DM as time shift (equivalent to a phasor in f-domain)
        t_rel_dm = t_rel[good] - float(tau_dm[ic])

        omega_u = 2.0 * math.pi * u * dl
        omega_v = 2.0 * math.pi * v * dl
        omega_f = -2.0 * math.pi * t_rel_dm * df
        om = np.stack([wrap_pi(omega_u), wrap_pi(omega_v), wrap_pi(omega_f)], axis=0)

        y = vis[good, ic]

        # Adjoint NUFFT: nonuniform -> uniform (npix,npix,Nf)
        X_full = ducc_nufft.nufft(
            data=y,
            om=om,
            shape=(npix, npix, Nf),
            isign=-1,
            eps=eps,
            nthreads=nthreads,
            forward=False
        )

        X_pos = X_full[:, :, pos]
        X_keep = X_pos[:, :, kmin:].astype(np.complex64)
        X_blk_keep += X_keep

    os.makedirs(out_dir, exist_ok=True)
    part_name = f"periodicity_block_{block_id:06d}_{uuid.uuid4().hex}.npz"
    part_path = os.path.join(out_dir, part_name)
    np.savez_compressed(part_path, X=X_blk_keep, t0=np.float64(t0))

    return {
        "path": part_path,
        "block_id": int(block_id),
        "t0": float(t0),
        "npix": int(npix),
        "nkeep": int(nkeep),
    }

def run_periodicity(args=None) -> str:
    """
    Driver for periodicity search.
    Returns path to the power cube FITS written (primary product).
    """
    if args is None:
        args = build_cli_periodicity()

    # --- prepare MS row ordering ---
    T = table(args.msname, readonly=True)
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

    # channel selection
    chan_sel = None
    if args.chan_start is not None or args.chan_end is not None:
        chan_sel = (0 if args.chan_start is None else int(args.chan_start),
                    None if args.chan_end is None else int(args.chan_end))

    metas: List[Dict] = []

    # --- execute per-block NUFFT ---
    if args.parallel_mode == "serial":
        for i, (lo, hi) in enumerate(bounds):
            row_ids_block = row_ids_sorted[lo:hi + 1]
            meta = _process_block_and_save(
                ms_path=args.msname,
                row_ids_block=row_ids_block,
                spw=int(args.spw),
                pol=int(args.pol),
                data_col=args.data_col,
                chan_sel=chan_sel,
                npix=npix, dl=dl, Nf=Nf, dt=dt, df=df,
                fgrid=fgrid, kmin=kmin,
                eps=float(args.eps),
                nthreads=int(args.nthreads),
                dm_pc_cm3=float(args.dm),
                out_dir=partials_dir,
                block_id=i,
            )
            metas.append(meta)

    elif args.parallel_mode == "dask-local":
        from dask.distributed import Client, LocalCluster, as_completed

        cluster = LocalCluster(
            n_workers=(args.dask_workers if args.dask_workers > 0 else None),
            threads_per_worker=args.threads_per_worker,
            processes=(args.dask_scheduler == 'processes')
        ) if args.scheduler_address is None else None

        with Client(cluster if cluster is not None else args.scheduler_address) as client:
            futures = []
            for i, (lo, hi) in enumerate(bounds):
                row_ids_block = row_ids_sorted[lo:hi + 1]
                fut = client.submit(
                    _process_block_and_save,
                    args.msname, row_ids_block, int(args.spw), int(args.pol),
                    args.data_col, chan_sel,
                    npix, dl, Nf, dt, df, fgrid, kmin,
                    float(args.eps), int(args.nthreads),
                    float(args.dm),
                    partials_dir,
                    i
                )
                futures.append(fut)

            for fut in as_completed(futures):
                metas.append(fut.result())

        if cluster is not None:
            cluster.close()
    else:
        raise ValueError(f"Unknown parallel-mode: {args.parallel_mode}")

    # --- phase stitch + accumulate ---
    sum_mmap_path = os.path.join(candidates_dir, args.sum_memmap)
    mm = np.memmap(sum_mmap_path, mode='w+', dtype=np.complex64, shape=(npix, npix, nkeep))
    mm[:] = 0.0

    for meta in metas:
        with np.load(meta["path"]) as z:
            X_blk = z["X"]
            t0 = float(z["t0"])
        phase = np.exp(-1j * 2.0 * np.pi * fkeep * t0).astype(np.complex64)
        mm += X_blk * phase[None, None, :]

    # power cube
    P = (np.abs(mm) ** 2).astype(np.float32)
    del mm

    # --- write 3D FITS cube (power) ---
    w2 = ducc_wcs._build_fullframe_wcs(
        npix_x=npix, npix_y=npix,
        ra0_rad=ra0_rad, dec0_rad=dec0_rad,
        pixscale_rad=dl,
        ra_sign=-1, dec_sign=+1,
        radesys="ICRS", equinox=None
    )
    w3 = WCS(naxis=3)
    w3.wcs.ctype[0:2] = w2.wcs.ctype
    w3.wcs.cunit[0:2] = w2.wcs.cunit
    w3.wcs.crpix[0:2] = w2.wcs.crpix
    w3.wcs.crval[0:2] = w2.wcs.crval
    w3.wcs.cdelt[0:2] = w2.wcs.cdelt
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

    # --- optional quicklook products ---
    if args.write_products:
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
            flip_u=False, flip_v=False,
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

    return out_cube_fits
