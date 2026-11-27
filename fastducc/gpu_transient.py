# -*- coding: utf-8 -*-
"""
fastducc.gpu_transient: GPU-accelerated direct half-plane gridding and C2R
imaging for multi-DM transient searches.

Ports the direct Hermitian half-plane rasterisation and two-tier subbanded
active-cell dedispersion from ``fastducc.gpu_raster`` together with batched
2D C2R imaging from ``fastducc.gpu_c2r`` into the chunked transient search
pipeline. Supports both CUDA (CuPy) and CPU fallback execution.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import binary_dilation

try:
    import cupy as cp

    HAS_CUPY = True
except Exception:
    cp = None  # type: ignore[assignment]
    HAS_CUPY = False

from fastducc import dedisp, gpu_c2r, gpu_raster
from fastducc.constants import C_LIGHT

logger = logging.getLogger(__name__)


def build_chunk_active_uv_plan(
    uvw_3d: np.ndarray,
    chan_freq: np.ndarray,
    *,
    nv: int,
    grid_nu: int,
    u_half: int,
    pix_rad: float,
    pad_radius: int = 6,
    min_uv_distance_m: Optional[float] = None,
    n_freq_samples: int = 16,
    time_stride: int = 4,
) -> Dict[str, Any]:
    """
    Build a compact chunk-local active UV-cell coordinate table on the Hermitian
    half-plane ``(nv, u_half)`` from ``uvw_3d (nbl, 3, nt)``.

    Applies periodic dilation along ``v`` (mod ``nv``) and bounded dilation along
    ``u`` (``0 <= u < u_half``) to capture full gridding kernel support.
    """
    nbl, _, nt = uvw_3d.shape
    nv_i, nu_i, uh_i = int(nv), int(grid_nu), int(u_half)
    if nbl == 0 or nt == 0 or len(chan_freq) == 0:
        empty_uv = np.zeros((0, 2), dtype=np.int32)
        empty_flat = np.zeros((0,), dtype=np.int64)
        return {
            "uv_indices": empty_uv,
            "flat_indices": empty_flat,
            "m_total": 0,
            "nv": nv_i,
            "grid_nu": nu_i,
            "u_half": uh_i,
        }

    stride = max(1, int(time_stride))
    t_sel = np.arange(0, nt, stride, dtype=np.int64)
    if t_sel[-1] != nt - 1:
        t_sel = np.append(t_sel, nt - 1)

    u_m = uvw_3d[:, 0, t_sel].ravel()
    v_m = uvw_3d[:, 1, t_sel].ravel()
    finite = np.isfinite(u_m) & np.isfinite(v_m)
    if min_uv_distance_m is not None and float(min_uv_distance_m) > 0.0:
        finite &= np.hypot(u_m, v_m) >= float(min_uv_distance_m)
    u_m = u_m[finite]
    v_m = v_m[finite]

    if u_m.size == 0:
        empty_uv = np.zeros((0, 2), dtype=np.int32)
        empty_flat = np.zeros((0,), dtype=np.int64)
        return {
            "uv_indices": empty_uv,
            "flat_indices": empty_flat,
            "m_total": 0,
            "nv": nv_i,
            "grid_nu": nu_i,
            "u_half": uh_i,
        }

    freqs = np.asarray(chan_freq, dtype=np.float64)
    if freqs.size > n_freq_samples:
        sample_freqs = np.linspace(float(freqs.min()), float(freqs.max()), int(n_freq_samples), dtype=np.float64)
    else:
        sample_freqs = freqs

    raw_mask = np.zeros((nv_i, uh_i), dtype=bool)
    scale_u = float(nu_i) * float(pix_rad)
    scale_v = float(nv_i) * float(pix_rad)

    for f in sample_freqs:
        inv_lam = float(f) / C_LIGHT
        u_pix = np.round(u_m * inv_lam * scale_u).astype(np.int64) % nu_i
        v_pix = np.round(v_m * inv_lam * scale_v).astype(np.int64) % nv_i

        # Mark both (u, v) and its Hermitian conjugate (-u, -v) in the u >= 0 half-plane
        in_hp = u_pix < uh_i
        if np.any(in_hp):
            raw_mask[v_pix[in_hp], u_pix[in_hp]] = True
        u_neg = (-u_pix) % nu_i
        v_neg = (-v_pix) % nv_i
        in_hp_neg = u_neg < uh_i
        if np.any(in_hp_neg):
            raw_mask[v_neg[in_hp_neg], u_neg[in_hp_neg]] = True

    pad = max(0, int(pad_radius))
    if pad > 0:
        # Periodic wrap along v (axis 0) before binary dilation
        v_pad = min(pad, nv_i)
        padded = np.concatenate([raw_mask[-v_pad:, :], raw_mask, raw_mask[:v_pad, :]], axis=0)
        struct = np.ones((2 * pad + 1, 2 * pad + 1), dtype=bool)
        dilated = binary_dilation(padded, structure=struct)
        active_mask = dilated[v_pad : v_pad + nv_i, :uh_i]
    else:
        active_mask = raw_mask

    v_idx, u_idx = np.where(active_mask)
    order = np.lexsort((u_idx, v_idx))
    v_idx = v_idx[order].astype(np.int32)
    u_idx = u_idx[order].astype(np.int32)
    uv_indices = np.ascontiguousarray(np.column_stack([v_idx, u_idx]), dtype=np.int32)
    flat_indices = np.ascontiguousarray(
        v_idx.astype(np.int64) * np.int64(uh_i) + u_idx.astype(np.int64), dtype=np.int64
    )
    return {
        "uv_indices": uv_indices,
        "flat_indices": flat_indices,
        "m_total": int(flat_indices.size),
        "nv": nv_i,
        "grid_nu": nu_i,
        "u_half": uh_i,
    }


def _order_coarse_groups_dm0_first(
    dm_list: List[float],
    coarse_dms: np.ndarray,
) -> List[Tuple[int, List[float]]]:
    """Group fine DMs by nearest coarse DM, placing DM=0.0 first if present."""
    groups: Dict[int, List[float]] = {}
    for dm_f in dm_list:
        ic = int(np.argmin(np.abs(coarse_dms - dm_f)))
        groups.setdefault(ic, []).append(float(dm_f))

    ordered: List[Tuple[int, List[float]]] = []
    dm0_group_idx: Optional[int] = None
    for ic in sorted(groups.keys()):
        dms = groups[ic]
        if 0.0 in dms:
            dms = [0.0] + [d for d in dms if d != 0.0]
            dm0_group_idx = len(ordered)
        ordered.append((ic, dms))

    if dm0_group_idx is not None and dm0_group_idx > 0:
        ordered.insert(0, ordered.pop(dm0_group_idx))
    return ordered


def iter_chunk_dirty_cubes_direct_raster(
    vis_3d: np.ndarray,
    wgt_3d: np.ndarray,
    uvw_3d: np.ndarray,
    chan_freq: np.ndarray,
    dt: float,
    dm_trials: Sequence[float],
    *,
    npix_x: int = 384,
    npix_y: int = 384,
    pixsize_x: float = 22.0 / 206265.0,
    pixsize_y: float = 22.0 / 206265.0,
    dedisp_mode: str = "subbanded",
    nsubbands: int = 16,
    collapse_channels: bool = False,
    exact_uvw: bool = True,
    raster_mode: str = "wgridder",
    epsilon: float = 1e-5,
    do_wgridding: bool = False,
    sigma_min: float = 1.1,
    sigma_max: float = 2.6,
    min_valid_channels: int = 24,
    min_uv_distance_m: Optional[float] = None,
    uv_threshold: float = 0.0,
    pad_radius: int = 6,
    c2r_batch_size: int = 256,
    device: str = "auto",
    device_id: int = 0,
    nthreads: int = 1,
    tolerance_samples: float = 1.25,
    align_to: str = "fmax",
    return_device: bool = False,
) -> Iterator[Tuple[float, Any]]:
    """
    Generate dirty image cubes ``(nt, npix_y, npix_x)`` across all ``dm_trials``
    for an in-memory chunk using direct Hermitian half-plane rasterisation and
    batched 2D C2R imaging.

    When ``dedisp_mode == "subbanded"`` and multiple DM trials are requested
    (with ``do_wgridding=False``), executes two-tier subbanded dedispersion:
      - Tier 1: Gridding into compact active UV cells ``(M_active, n_subbands, nt)``
        once per coarse DM bin.
      - Tier 2: Vectorised active-cell shift-and-sum across subbands per fine DM,
        followed by batched half-plane C2R imaging with DUCC deapodisation.

    Otherwise executes fullband direct half-plane rasterisation per trial DM
    followed by batched C2R imaging.

    Yields
    ------
    dm_val : float
        Trial dispersion measure (pc cm^-3). DM=0.0 is yielded first if present.
    cube : np.ndarray (float32), shape ``(nt, npix_y, npix_x)``
        Real dirty image time cube.
    """
    dm_list = [float(d) for d in dm_trials]
    if not dm_list:
        return
    if 0.0 in dm_list:
        dm_list = [0.0] + [d for d in dm_list if d != 0.0]

    eff_device = gpu_raster.resolve_device(device)
    eff_dedisp_mode = str(dedisp_mode).strip().lower()
    if eff_dedisp_mode not in ("subbanded", "fullband"):
        raise ValueError(
            f"Unsupported dedisp_mode='{dedisp_mode}'; expected 'subbanded' or 'fullband'."
        )

    nbl, nchan, nt = vis_3d.shape
    eff_pix_rad = float(pixsize_x)
    use_subbanded = (
        eff_dedisp_mode == "subbanded"
        and len(dm_list) > 1
        and not collapse_channels
        and int(nsubbands) > 1
        and nchan > 1
    )
    eff_do_wgridding = bool(do_wgridding)
    if use_subbanded and eff_do_wgridding:
        logger.info(
            "Subbanded two-tier direct_raster uses 2D wgridder in Tier 1; "
            "disabling 3D w-stacking for subbanded pass (use --dedisp-mode fullband for 3D w-gridding)."
        )
        eff_do_wgridding = False

    geom = gpu_raster.compute_raster_grid_geometry(
        npix_x=int(npix_x),
        npix_y=int(npix_y),
        pix_rad=eff_pix_rad,
        mode=str(raster_mode),
        epsilon=float(epsilon),
        do_wgridding=eff_do_wgridding,
        sigma_min=float(sigma_min),
        sigma_max=float(sigma_max),
    )
    grid_nu = int(geom["grid_nu"])
    nv = int(geom["nv"])
    u_half = int(geom["u_half"])

    cfu_host, cfv_host = gpu_c2r.compute_deapod_vectors(
        npix_x=int(npix_x),
        npix_y=int(npix_y),
        grid_nu=grid_nu,
        nv=nv,
        raster_mode=str(raster_mode),
        epsilon=float(epsilon),
        do_wgridding=eff_do_wgridding,
        sigma_min=float(sigma_min),
        sigma_max=float(sigma_max),
    )

    if eff_device == "cuda":
        t0_gpu = time.time()
        with cp.cuda.Device(int(device_id)):
            d_vis = cp.ascontiguousarray(cp.asarray(vis_3d, dtype=np.complex64))
            d_wgt = cp.ascontiguousarray(cp.asarray(wgt_3d, dtype=np.float32))
            d_uvw = cp.ascontiguousarray(cp.asarray(uvw_3d, dtype=np.float64))
            d_cfu = cp.asarray(cfu_host, dtype=cp.float32) if cfu_host is not None else None
            d_cfv = cp.asarray(cfv_host, dtype=cp.float32) if cfv_host is not None else None
            ws = gpu_c2r.HalfPlaneC2RWorkspace.create(
                max_batch=max(1, min(int(c2r_batch_size), max(1, nt))),
                nv=nv,
                nu=grid_nu,
                ny=int(npix_y),
                nx=int(npix_x),
                device_id=int(device_id),
            )
        try:
            if use_subbanded:
                eff_nsub = max(1, min(int(nsubbands), nchan))
                plan = dedisp.DedispersionPlan(
                    freqs_hz=chan_freq,
                    tsamp_s=float(dt),
                    dm_min=float(min(dm_list)),
                    dm_max=float(max(dm_list)),
                    tolerance_samples=float(tolerance_samples),
                    dedisp_mode="subbanded",
                    n_subbands=eff_nsub,
                )
                coarse_groups = _order_coarse_groups_dm0_first(dm_list, plan.coarse_dms)
                eff_pad = int(pad_radius)
                if str(raster_mode).strip().lower() in ("wgridder", "ducc_wgridder"):
                    kpar = gpu_raster.select_ducc_kernel_params(
                        epsilon=float(epsilon),
                        do_wgridding=False,
                        sigma_min=float(sigma_min),
                        sigma_max=float(sigma_max),
                    )
                    eff_pad = max(eff_pad, (int(kpar["W"]) + 1) // 2 + 1)
                active_plan = build_chunk_active_uv_plan(
                    uvw_3d=uvw_3d,
                    chan_freq=chan_freq,
                    nv=nv,
                    grid_nu=grid_nu,
                    u_half=u_half,
                    pix_rad=eff_pix_rad,
                    pad_radius=eff_pad,
                    min_uv_distance_m=min_uv_distance_m,
                )
                m_active = int(active_plan["m_total"])
                if m_active == 0:
                    for dm_val in dm_list:
                        yield dm_val, np.zeros((nt, int(npix_y), int(npix_x)), dtype=np.float32)
                    return

                with cp.cuda.Device(int(device_id)):
                    d_uv_to_m = gpu_raster.build_uv_to_m_lookup(
                        nv=nv,
                        u_half=u_half,
                        flat_indices=active_plan["flat_indices"],
                        device="cuda",
                    )
                    d_cell_uv = cp.asarray(active_plan["uv_indices"], dtype=cp.int32)

                logger.debug(
                    "[GPU Transient Subbanded] Chunk nt=%d, M_active=%d/%d (%.1f%%), %d coarse -> %d fine DMs",
                    nt,
                    m_active,
                    nv * u_half,
                    100.0 * m_active / max(1, nv * u_half),
                    len(coarse_groups),
                    len(dm_list),
                )

                for ic, fine_dms in coarse_groups:
                    dm_coarse = float(plan.coarse_dms[ic])
                    intra_delays = plan.get_intra_delays(dm_coarse, align_to=align_to).astype(np.int32)
                    with cp.cuda.Device(int(device_id)):
                        d_active_dga = gpu_raster.dedisperse_and_rasterise_subband_gpu(
                            d_vis,
                            d_wgt,
                            d_uvw,
                            chan_freq,
                            intra_delays=intra_delays,
                            n_subbands=eff_nsub,
                            nt_out=nt,
                            dm_coarse=dm_coarse,
                            dt=float(dt),
                            align_to=align_to,
                            exact_uvw=bool(exact_uvw),
                            npix_x=int(npix_x),
                            npix_y=int(npix_y),
                            pix_rad=eff_pix_rad,
                            grid_nu=grid_nu,
                            nv=nv,
                            mode=str(raster_mode),
                            epsilon=float(epsilon),
                            sigma_min=float(sigma_min),
                            sigma_max=float(sigma_max),
                            min_valid_channels=int(min_valid_channels),
                            min_uv_distance_m=min_uv_distance_m,
                            uv_to_m=d_uv_to_m,
                            m_active=m_active,
                            return_host=False,
                        )
                    try:
                        for dm_val in fine_dms:
                            inter_delays = plan.get_inter_delays(dm_val, align_to=align_to).astype(np.int32)
                            with cp.cuda.Device(int(device_id)):
                                d_vis_active = gpu_raster.dedisperse_active_cell_matrix_gpu(
                                    d_active_dga,
                                    inter_delays,
                                    nt_out=nt,
                                    uv_threshold=float(uv_threshold),
                                    return_host=False,
                                )
                                cube = gpu_c2r.sparse_active_to_dirty_cube_gpu(
                                    d_vis_active,
                                    d_cell_uv,
                                    nv=nv,
                                    grid_nu=grid_nu,
                                    npix_y=int(npix_y),
                                    npix_x=int(npix_x),
                                    corfunc_u=d_cfu,
                                    corfunc_v=d_cfv,
                                    c2r_batch_size=int(c2r_batch_size),
                                    workspace=ws,
                                    device_id=int(device_id),
                                    return_host=not bool(return_device),
                                )
                                del d_vis_active
                            yield dm_val, cube
                    finally:
                        del d_active_dga
            else:
                for dm_val in dm_list:
                    with cp.cuda.Device(int(device_id)):
                        d_uv_half = gpu_raster.dedisperse_and_rasterise_chunk_gpu(
                            d_vis,
                            d_wgt,
                            d_uvw,
                            chan_freq,
                            nt_out=nt,
                            dm=float(dm_val),
                            dt=float(dt),
                            align_to=align_to,
                            collapse_channels=bool(collapse_channels),
                            nsubbands=int(nsubbands),
                            exact_uvw=bool(exact_uvw),
                            npix_x=int(npix_x),
                            npix_y=int(npix_y),
                            pix_rad=eff_pix_rad,
                            grid_nu=grid_nu,
                            nv=nv,
                            mode=str(raster_mode),
                            epsilon=float(epsilon),
                            do_wgridding=eff_do_wgridding,
                            sigma_min=float(sigma_min),
                            sigma_max=float(sigma_max),
                            min_valid_channels=int(min_valid_channels),
                            min_uv_distance_m=min_uv_distance_m,
                            uv_threshold=float(uv_threshold),
                            return_sparse=False,
                            return_host=False,
                        )
                        cube = gpu_c2r.halfplane_grid_to_dirty_cube_gpu(
                            d_uv_half,
                            grid_nu=grid_nu,
                            npix_y=int(npix_y),
                            npix_x=int(npix_x),
                            corfunc_u=d_cfu,
                            corfunc_v=d_cfv,
                            c2r_batch_size=int(c2r_batch_size),
                            workspace=ws,
                            device_id=int(device_id),
                            return_host=not bool(return_device),
                        )
                        del d_uv_half
                    yield dm_val, cube
        finally:
            ws.release()
            del d_vis, d_wgt, d_uvw, d_cfu, d_cfv
            logger.debug(
                "[GPU Transient] Finished chunk nt=%d across %d DM(s) in %.2fs",
                nt,
                len(dm_list),
                time.time() - t0_gpu,
            )
        return

    # CPU fallback path (exact parity with GPU kernels for testing / CPU nodes)
    if use_subbanded:
        eff_nsub = max(1, min(int(nsubbands), nchan))
        plan = dedisp.DedispersionPlan(
            freqs_hz=chan_freq,
            tsamp_s=float(dt),
            dm_min=float(min(dm_list)),
            dm_max=float(max(dm_list)),
            tolerance_samples=float(tolerance_samples),
            dedisp_mode="subbanded",
            n_subbands=eff_nsub,
        )
        coarse_groups = _order_coarse_groups_dm0_first(dm_list, plan.coarse_dms)
        eff_pad = int(pad_radius)
        if str(raster_mode).strip().lower() in ("wgridder", "ducc_wgridder"):
            kpar = gpu_raster.select_ducc_kernel_params(
                epsilon=float(epsilon),
                do_wgridding=False,
                sigma_min=float(sigma_min),
                sigma_max=float(sigma_max),
            )
            eff_pad = max(eff_pad, (int(kpar["W"]) + 1) // 2 + 1)
        active_plan = build_chunk_active_uv_plan(
            uvw_3d=uvw_3d,
            chan_freq=chan_freq,
            nv=nv,
            grid_nu=grid_nu,
            u_half=u_half,
            pix_rad=eff_pix_rad,
            pad_radius=eff_pad,
            min_uv_distance_m=min_uv_distance_m,
        )
        m_active = int(active_plan["m_total"])
        if m_active == 0:
            for dm_val in dm_list:
                yield dm_val, np.zeros((nt, int(npix_y), int(npix_x)), dtype=np.float32)
            return

        uv_to_m = gpu_raster.build_uv_to_m_lookup(
            nv=nv,
            u_half=u_half,
            flat_indices=active_plan["flat_indices"],
            device="cpu",
        )
        cell_uv = active_plan["uv_indices"]

        for ic, fine_dms in coarse_groups:
            dm_coarse = float(plan.coarse_dms[ic])
            intra_delays = plan.get_intra_delays(dm_coarse, align_to=align_to).astype(np.int32)
            active_dga = gpu_raster.dedisperse_and_rasterise_subband_cpu(
                vis_3d,
                wgt_3d,
                uvw_3d,
                chan_freq,
                intra_delays=intra_delays,
                n_subbands=eff_nsub,
                nt_out=nt,
                dm_coarse=dm_coarse,
                dt=float(dt),
                align_to=align_to,
                exact_uvw=bool(exact_uvw),
                npix_x=int(npix_x),
                npix_y=int(npix_y),
                pix_rad=eff_pix_rad,
                grid_nu=grid_nu,
                nv=nv,
                mode=str(raster_mode),
                epsilon=float(epsilon),
                sigma_min=float(sigma_min),
                sigma_max=float(sigma_max),
                min_valid_channels=int(min_valid_channels),
                min_uv_distance_m=min_uv_distance_m,
                uv_to_m=uv_to_m,
                m_active=m_active,
            )
            for dm_val in fine_dms:
                inter_delays = plan.get_inter_delays(dm_val, align_to=align_to).astype(np.int32)
                vis_active = gpu_raster.dedisperse_active_cell_matrix_cpu(
                    active_dga,
                    inter_delays,
                    nt_out=nt,
                    uv_threshold=float(uv_threshold),
                )
                cube = gpu_c2r.sparse_active_to_dirty_cube_cpu(
                    vis_active,
                    cell_uv,
                    nv=nv,
                    grid_nu=grid_nu,
                    npix_y=int(npix_y),
                    npix_x=int(npix_x),
                    corfunc_u=cfu_host,
                    corfunc_v=cfv_host,
                    nthreads=max(1, int(nthreads)),
                )
                yield dm_val, cube
            del active_dga
    else:
        for dm_val in dm_list:
            uv_half = gpu_raster.dedisperse_and_rasterise_chunk_cpu(
                vis_3d,
                wgt_3d,
                uvw_3d,
                chan_freq,
                nt_out=nt,
                dm=float(dm_val),
                dt=float(dt),
                align_to=align_to,
                collapse_channels=bool(collapse_channels),
                nsubbands=int(nsubbands),
                exact_uvw=bool(exact_uvw),
                npix_x=int(npix_x),
                npix_y=int(npix_y),
                pix_rad=eff_pix_rad,
                grid_nu=grid_nu,
                nv=nv,
                mode=str(raster_mode),
                epsilon=float(epsilon),
                do_wgridding=eff_do_wgridding,
                sigma_min=float(sigma_min),
                sigma_max=float(sigma_max),
                min_valid_channels=int(min_valid_channels),
                min_uv_distance_m=min_uv_distance_m,
                uv_threshold=float(uv_threshold),
                return_sparse=False,
            )
            cube = gpu_c2r.halfplane_grid_to_dirty_cube_cpu(
                uv_half,
                grid_nu=grid_nu,
                npix_y=int(npix_y),
                npix_x=int(npix_x),
                corfunc_u=cfu_host,
                corfunc_v=cfv_host,
                nthreads=max(1, int(nthreads)),
            )
            yield dm_val, cube


__all__ = [
    "build_chunk_active_uv_plan",
    "iter_chunk_dirty_cubes_direct_raster",
]
