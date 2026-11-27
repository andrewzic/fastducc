#!/usr/bin/env python3
"""
dynspec.py - High-Performance In-Memory Phased Visibility Streaming Dynamic Spectrum Extraction

Replaces disk-based CASA workflows with a fast in-memory streaming visibility engine:
1. Reads Measurement Sets in streaming row chunks via casacore.tables.
2. Applies conjugate geometric phasors to rotate the phase centre to candidate (l, m, n).
3. Sums weighted unflagged visibilities on-the-fly to form phased Stokes I (or requested pol) dynamic spectra.
4. Batches multiple candidates per beam simultaneously in a single sequential pass.
"""

from __future__ import annotations

import collections
import gc
import logging
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from casacore.tables import table
except ImportError:
    table = None

try:
    import numba as nb
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    nb = None

from astropy.coordinates import SkyCoord
import astropy.units as u

logger = logging.getLogger("fastducc.dynspec")

# Speed of light in m/s
C_LIGHT = 299792458.0

_CORR_CODE_TO_NAME = {
    5: "RR", 6: "RL", 7: "LR", 8: "LL",
    9: "XX", 10: "XY", 11: "YX", 12: "YY",
}


# =============================================================================
# Numba JIT Kernels for Chunk Visibility Phasing and Accumulation
# =============================================================================

if HAS_NUMBA:
    @nb.njit(parallel=True, fastmath=True)
    def _accumulate_chunk_1d_weights(
        starts: np.ndarray,
        ends: np.ndarray,
        P: np.ndarray,
        data_r: np.ndarray,
        data_i: np.ndarray,
        weights: np.ndarray,
        flags: np.ndarray,
        t_idx: np.ndarray,
        freq_coeffs: np.ndarray,
        numerators: np.ndarray,
        denom: np.ndarray,
    ) -> None:
        """
        Parallel accumulation across non-overlapping time slices with 1D row weights.
        P: (B, K) geometric delays (u*l + v*m + w*(n-1)) in meters.
        data_r, data_i: (B, Nchan) real and imag parts of visibilities.
        weights: (B,) float32 weights per row.
        flags: (B, Nchan) boolean flags.
        t_idx: (B,) int32 time bin indices.
        freq_coeffs: (Nchan,) float32 array (2*pi*nu/c).
        numerators: (K, Ntime, Nchan) float32 accumulator.
        denom: (Ntime, Nchan) float32 accumulator.
        """
        n_slices = len(starts)
        K = P.shape[1]
        Nchan = len(freq_coeffs)

        for s in nb.prange(n_slices):
            r_start = starts[s]
            r_end = ends[s]
            for r in range(r_start, r_end):
                t = t_idx[r]
                w = weights[r]
                if w <= 0.0:
                    continue
                # 1. Accumulate denominator once per valid visibility
                for c in range(Nchan):
                    if not flags[r, c]:
                        denom[t, c] += w

                # 2. Accumulate phased visibilities for each target
                for k in range(K):
                    p = P[r, k]
                    for c in range(Nchan):
                        if not flags[r, c]:
                            phi = p * freq_coeffs[c]
                            # rot = exp(-i*phi) = cos(phi) - i*sin(phi)
                            # Re((dr + i*di) * (cos(phi) - i*sin(phi))) = dr*cos(phi) + di*sin(phi)
                            cos_p = np.cos(phi)
                            sin_p = np.sin(phi)
                            numerators[k, t, c] += w * (data_r[r, c] * cos_p + data_i[r, c] * sin_p)

    @nb.njit(parallel=True, fastmath=True)
    def _accumulate_chunk_2d_weights(
        starts: np.ndarray,
        ends: np.ndarray,
        P: np.ndarray,
        data_r: np.ndarray,
        data_i: np.ndarray,
        weights: np.ndarray,
        flags: np.ndarray,
        t_idx: np.ndarray,
        freq_coeffs: np.ndarray,
        numerators: np.ndarray,
        denom: np.ndarray,
    ) -> None:
        """
        Parallel accumulation across non-overlapping time slices with 2D channel weights.
        weights: (B, Nchan) float32 weights.
        """
        n_slices = len(starts)
        K = P.shape[1]
        Nchan = len(freq_coeffs)

        for s in nb.prange(n_slices):
            r_start = starts[s]
            r_end = ends[s]
            for r in range(r_start, r_end):
                t = t_idx[r]
                for c in range(Nchan):
                    if not flags[r, c]:
                        w = weights[r, c]
                        if w > 0.0:
                            denom[t, c] += w

                for k in range(K):
                    p = P[r, k]
                    for c in range(Nchan):
                        if not flags[r, c]:
                            w = weights[r, c]
                            if w > 0.0:
                                phi = p * freq_coeffs[c]
                                cos_p = np.cos(phi)
                                sin_p = np.sin(phi)
                                numerators[k, t, c] += w * (data_r[r, c] * cos_p + data_i[r, c] * sin_p)
    @nb.njit(parallel=True, fastmath=True)
    def _accumulate_chunk_cube(
        P: np.ndarray,
        vis_r: np.ndarray,
        vis_i: np.ndarray,
        wgt: np.ndarray,
        freq_coeffs: np.ndarray,
        numerators: np.ndarray,
        denom: np.ndarray,
    ) -> None:
        """
        Parallel accumulation over time bins for an in-memory chunk visibility cube.

        Parameters
        ----------
        P : (K, nbl, nt) float32 geometric delays (u*l + v*m + w*(n-1)) in metres.
        vis_r, vis_i : (nbl, nchan, nt) float32 real and imaginary visibilities.
        wgt : (nbl, nchan, nt) float32 weights (zero for flagged samples).
        freq_coeffs : (nchan,) float32 array (2*pi*nu/c).
        numerators : (K, nt, nchan) float32 accumulator.
        denom : (nt, nchan) float32 accumulator.
        """
        K = P.shape[0]
        nbl = vis_r.shape[0]
        nchan = vis_r.shape[1]
        nt = vis_r.shape[2]

        for t in nb.prange(nt):
            for c in range(nchan):
                fc = freq_coeffs[c]
                d_acc = np.float32(0.0)
                for b in range(nbl):
                    w = wgt[b, c, t]
                    if w > 0.0:
                        d_acc += w
                denom[t, c] = d_acc
                if d_acc <= 0.0:
                    continue

                for k in range(K):
                    n_acc = np.float32(0.0)
                    for b in range(nbl):
                        w = wgt[b, c, t]
                        if w > 0.0:
                            phi = P[k, b, t] * fc
                            cos_p = np.cos(phi)
                            sin_p = np.sin(phi)
                            n_acc += w * (vis_r[b, c, t] * cos_p + vis_i[b, c, t] * sin_p)
                    numerators[k, t, c] = n_acc
else:
    def _accumulate_chunk_cube(
        P, vis_r, vis_i, wgt, freq_coeffs, numerators, denom
    ):
        K = P.shape[0]
        nbl, nchan, nt = vis_r.shape
        denom[:, :] = np.sum(np.where(wgt > 0.0, wgt, 0.0), axis=0).T
        pos_wgt = np.where(wgt > 0.0, wgt, 0.0)
        for k in range(K):
            phi = P[k, :, None, :] * freq_coeffs[None, :, None]
            phased = vis_r * np.cos(phi) + vis_i * np.sin(phi)
            numerators[k, :, :] = np.sum(pos_wgt * phased, axis=0).T

    def _accumulate_chunk_1d_weights(
        starts, ends, P, data_r, data_i, weights, flags, t_idx, freq_coeffs, numerators, denom
    ):
        K = P.shape[1]
        Nchan = len(freq_coeffs)
        B = len(t_idx)
        for r in range(B):
            t = t_idx[r]
            w = weights[r]
            if w <= 0.0:
                continue
            for c in range(Nchan):
                if not flags[r, c]:
                    denom[t, c] += w
                    fc = freq_coeffs[c]
                    dr = data_r[r, c]
                    di = data_i[r, c]
                    for k in range(K):
                        phi = P[r, k] * fc
                        cos_p = math.cos(phi)
                        sin_p = math.sin(phi)
                        numerators[k, t, c] += w * (dr * cos_p + di * sin_p)

    def _accumulate_chunk_2d_weights(
        starts, ends, P, data_r, data_i, weights, flags, t_idx, freq_coeffs, numerators, denom
    ):
        K = P.shape[1]
        Nchan = len(freq_coeffs)
        B = len(t_idx)
        for r in range(B):
            t = t_idx[r]
            for c in range(Nchan):
                if not flags[r, c]:
                    w = weights[r, c]
                    if w > 0.0:
                        denom[t, c] += w
                        fc = freq_coeffs[c]
                        dr = data_r[r, c]
                        di = data_i[r, c]
                        for k in range(K):
                            phi = P[r, k] * fc
                            cos_p = math.cos(phi)
                            sin_p = math.sin(phi)
                            numerators[k, t, c] += w * (dr * cos_p + di * sin_p)


def _compute_disjoint_slices(
    t_chunk_idx: np.ndarray,
    n_threads: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Partition rows into n_threads slices such that no two slices share any timestamp bin.
    Eliminates race conditions when writing into global accumulator arrays.
    """
    B = len(t_chunk_idx)
    n_sl = min(max(1, n_threads), B)
    target_split = np.linspace(0, B, n_sl + 1).astype(int)
    starts = []
    ends = []
    cur = 0

    for i in range(n_sl):
        start = cur
        if i == n_sl - 1:
            end = B
        else:
            raw_end = target_split[i + 1]
            t_val = t_chunk_idx[raw_end]
            # Advance split boundary to first row with t_val + 1
            split_pos = int(np.searchsorted(t_chunk_idx, t_val + 1))
            end = min(split_pos, B)

        if end > start:
            starts.append(start)
            ends.append(end)
            cur = end

    return np.array(starts, dtype=np.int32), np.array(ends, dtype=np.int32)


# =============================================================================
# Core Phased Visibility Streaming Engine
# =============================================================================

def extract_phased_dynamic_spectra_from_ms(
    ms_path: Union[str, Path],
    targets: List[Dict[str, Any]],
    chunk_rows: int = 100000,
    pol: str = "I",
    verbose: bool = True,
    n_threads: Optional[int] = None,
    datacolumn: str = "DATA",
    min_uv_distance_m: float = 0.0,
) -> Dict[str, Dict[str, Any]]:
    """
    Extract phased, baseline-averaged dynamic spectra directly from a Measurement Set in RAM
    for a list of targets (candidates) simultaneously in a single sequential streaming pass.

    Parameters
    ----------
    ms_path : str or Path
        Path to the Measurement Set.
    targets : list of dict
        Each dict must contain:
          - 'id': candidate identifier (str or int)
          - 'ra_deg': Right Ascension in degrees (float)
          - 'dec_deg': Declination in degrees (float)
    chunk_rows : int, default 100000
        Number of MS rows to process per streaming iteration.
    pol : str, default "I"
        Polarization to extract ("I", "XX", "YY", etc.).
    verbose : bool, default True
        Enable progress reporting.
    n_threads : int, optional
        Number of parallel worker slices (defaults to min(16, os.cpu_count())).

    Returns
    -------
    dict
        Mapping cand_id -> dict with keys:
          - 'data': 2D float32 array (N_time, N_chan) of phased dynamic spectrum.
          - 'time': 1D float64 array of elapsed seconds from observation start.
          - 'time_raw': 1D float64 array of raw casacore timestamps (seconds since MJD 0).
          - 'time_mjd': 1D float64 array of MJD timestamps.
          - 'freq': 1D float64 array of channel frequencies in MHz.
          - 'ra_str': Formatted RA string for barycentring.
          - 'dec_str': Formatted Dec string for barycentring.
          - 'telescope': Telescope name (default: "ASKAP").
          - 'pol': Polarization label (e.g. "Stokes I").
          - 'cand_id': Target identifier.
    """
    if table is None:
        raise ImportError(
            "casacore.tables is required for extract_phased_dynamic_spectra_from_ms."
        )

    ms_path = Path(ms_path).resolve()
    if not ms_path.is_dir():
        raise FileNotFoundError(f"Measurement Set not found: {ms_path}")

    if not targets:
        return {}

    t_start = time.time()
    if verbose:
        logger.info(
            f"Starting in-memory streaming extraction for {len(targets)} candidate(s) "
            f"from {ms_path.name} (pol={pol}, chunk_rows={chunk_rows})..."
        )

    if n_threads is None:
        n_threads = min(16, max(1, os.cpu_count() or 1))

    # 1. Read Phase Centre from FIELD subtable
    field_tb = table(f"{ms_path}/FIELD", readonly=True)
    try:
        pdir = np.array(field_tb.getcol("PHASE_DIR"))
        flat_pdir = pdir.reshape(-1, 2)
        ra0 = float(flat_pdir[0, 0])
        dec0 = float(flat_pdir[0, 1])
    finally:
        field_tb.close()

    # 2. Read Channel Frequencies from SPECTRAL_WINDOW subtable
    spw_tb = table(f"{ms_path}/SPECTRAL_WINDOW", readonly=True)
    try:
        freq_hz = np.array(spw_tb.getcell("CHAN_FREQ", 0), dtype=np.float64)
    finally:
        spw_tb.close()

    nchan = len(freq_hz)
    freq_mhz = freq_hz / 1e6
    freq_coeffs = (2.0 * np.pi / C_LIGHT * freq_hz).astype(np.float32)

    # 3. Read Polarizations and determine combination weights
    pol_tb = table(f"{ms_path}/POLARIZATION", readonly=True)
    try:
        corr_types = pol_tb.getcell("CORR_TYPE", 0)
    finally:
        pol_tb.close()

    corr_labels = [_CORR_CODE_TO_NAME.get(int(c), str(c)) for c in corr_types]
    lbl2idx = {lab: i for i, lab in enumerate(corr_labels)}

    pol_req = pol.upper()
    if pol_req == "I":
        if len(corr_labels) == 1:
            pol_indices = [0]
            pol_factors = [1.0]
        elif "XX" in lbl2idx and "YY" in lbl2idx:
            pol_indices = [lbl2idx["XX"], lbl2idx["YY"]]
            pol_factors = [0.5, 0.5]
        elif "RR" in lbl2idx and "LL" in lbl2idx:
            pol_indices = [lbl2idx["RR"], lbl2idx["LL"]]
            pol_factors = [0.5, 0.5]
        else:
            pol_indices = [0]
            pol_factors = [1.0]
    elif pol_req in lbl2idx:
        pol_indices = [lbl2idx[pol_req]]
        pol_factors = [1.0]
    else:
        # Default fallback to first correlation
        pol_indices = [0]
        pol_factors = [1.0]

    # 4. Compute Direction Cosines (l, m, n-1) for each target
    K = len(targets)
    target_vecs = np.zeros((K, 3), dtype=np.float32)
    cand_metadata = []

    for k, tgt in enumerate(targets):
        c_id = tgt.get("id", f"cand_{k}")
        ra_deg = float(tgt["ra_deg"])
        dec_deg = float(tgt["dec_deg"])

        sc = SkyCoord(ra_deg, dec_deg, unit="deg")
        ra_rad = sc.ra.rad
        dec_rad = sc.dec.rad

        l_k = np.cos(dec_rad) * np.sin(ra_rad - ra0)
        m_k = np.sin(dec_rad) * np.cos(dec0) - np.cos(dec_rad) * np.sin(dec0) * np.cos(ra_rad - ra0)
        n_k = np.sqrt(max(0.0, 1.0 - l_k**2 - m_k**2))
        target_vecs[k] = [l_k, m_k, n_k - 1.0]

        ra_str, dec_str = sc.to_string("hmsdms").split()
        cand_metadata.append({
            "id": c_id,
            "ra_deg": ra_deg,
            "dec_deg": dec_deg,
            "ra_str": ra_str,
            "dec_str": dec_str,
        })

    # 5. Open MS Main Table & Build Unique Timestamps Mapping
    tb = table(str(ms_path), readonly=True)
    try:
        nrows = tb.nrows()
        all_times = tb.getcol("TIME")
        unique_times, time_indices = np.unique(all_times, return_inverse=True)
        time_indices = time_indices.astype(np.int32)
        ntime = len(unique_times)

        if verbose:
            logger.info(
                f"MS contains {nrows:,} rows across {ntime:,} unique timestamps "
                f"({nchan} channels, dt={np.median(np.diff(unique_times)):.3f}s)."
            )

        # Allocate Accumulators in RAM
        numerators = np.zeros((K, ntime, nchan), dtype=np.float32)
        denom = np.zeros((ntime, nchan), dtype=np.float32)

        # Determine data column (accept both short aliases and canonical MS column names)
        _col_aliases = {
            "DATA": "DATA",
            "CORRECTED": "CORRECTED_DATA",
            "CORRECTED_DATA": "CORRECTED_DATA",
            "MODEL": "MODEL_DATA",
            "MODEL_DATA": "MODEL_DATA",
        }
        data_col = _col_aliases.get(datacolumn.upper(), datacolumn.upper())
        if data_col not in tb.colnames():
            if "DATA" in tb.colnames():
                logger.warning(f"Requested column {datacolumn} not in MS; falling back to DATA")
                data_col = "DATA"
            else:
                raise RuntimeError(f"Column {datacolumn} not found in {ms_path}")

        # Determine weight column layout
        has_weight_spectrum = "WEIGHT_SPECTRUM" in tb.colnames()
        has_weight = "WEIGHT" in tb.colnames()

        use_1d_weights = False
        if has_weight:
            if not has_weight_spectrum:
                use_1d_weights = True
            else:
                # Fast check: verify if WEIGHT_SPECTRUM is just broadcast WEIGHT
                n_sample = min(100, nrows)
                sample_ws = tb.getcol("WEIGHT_SPECTRUM", 0, n_sample)
                sample_w = tb.getcol("WEIGHT", 0, n_sample)
                diff = sample_ws[:, :, 0] - sample_w[:, 0:1]
                if np.max(np.abs(diff)) < 1e-6:
                    use_1d_weights = True

        if verbose:
            w_mode = "1D WEIGHT" if use_1d_weights else ("WEIGHT_SPECTRUM" if has_weight_spectrum else "UNIFORM")
            logger.info(f"Using {w_mode} weighting strategy (parallel threads={n_threads}).")

        # 6. Streaming Chunk Loop
        n_chunks = (nrows + chunk_rows - 1) // chunk_rows
        for chunk_idx, row_start in enumerate(range(0, nrows, chunk_rows)):
            row_end = min(row_start + chunk_rows, nrows)
            B = row_end - row_start

            t_chunk_idx = time_indices[row_start:row_end]
            uvw = tb.getcol("UVW", row_start, B)
            data_raw = tb.getcol(data_col, row_start, B)
            flag_raw = tb.getcol("FLAG", row_start, B)

            # Extract / combine requested polarizations
            if len(pol_indices) == 1:
                p_idx = pol_indices[0]
                data_r = data_raw[:, :, p_idx].real.astype(np.float32)
                data_i = data_raw[:, :, p_idx].imag.astype(np.float32)
                flags = flag_raw[:, :, p_idx]
            else:
                # Combine dual/quad pols (e.g. Stokes I = (XX + YY) / 2)
                p0, p1 = pol_indices[0], pol_indices[1]
                f0, f1 = pol_factors[0], pol_factors[1]
                data_r = (f0 * data_raw[:, :, p0].real + f1 * data_raw[:, :, p1].real).astype(np.float32)
                data_i = (f0 * data_raw[:, :, p0].imag + f1 * data_raw[:, :, p1].imag).astype(np.float32)
                flags = flag_raw[:, :, p0] | flag_raw[:, :, p1]

            # Apply minimum baseline UV-distance cut (matching main periodicity search)
            if min_uv_distance_m and min_uv_distance_m > 0.0:
                uv_dist2 = uvw[:, 0] ** 2 + uvw[:, 1] ** 2
                short_mask = uv_dist2 < (float(min_uv_distance_m) ** 2)
                if np.any(short_mask):
                    flags = flags | short_mask[:, np.newaxis]

            # Compute geometric dot products P = UVW @ target_vecs.T: shape (B, K)
            P = (uvw @ target_vecs.T).astype(np.float32)

            # Compute non-overlapping slice partitions for multi-threaded accumulation
            starts, ends = _compute_disjoint_slices(t_chunk_idx, n_threads=n_threads)

            if use_1d_weights:
                weights_raw = tb.getcol("WEIGHT", row_start, B)
                if len(pol_indices) == 1:
                    weights = weights_raw[:, pol_indices[0]].astype(np.float32)
                else:
                    weights = (0.5 * (weights_raw[:, pol_indices[0]] + weights_raw[:, pol_indices[1]])).astype(np.float32)
                _accumulate_chunk_1d_weights(
                    starts, ends, P, data_r, data_i, weights, flags, t_chunk_idx, freq_coeffs, numerators, denom
                )
            elif has_weight_spectrum:
                ws_raw = tb.getcol("WEIGHT_SPECTRUM", row_start, B)
                if len(pol_indices) == 1:
                    weights = ws_raw[:, :, pol_indices[0]].astype(np.float32)
                else:
                    weights = (0.5 * (ws_raw[:, :, pol_indices[0]] + ws_raw[:, :, pol_indices[1]])).astype(np.float32)
                _accumulate_chunk_2d_weights(
                    starts, ends, P, data_r, data_i, weights, flags, t_chunk_idx, freq_coeffs, numerators, denom
                )
            else:
                weights = np.ones(B, dtype=np.float32)
                _accumulate_chunk_1d_weights(
                    starts, ends, P, data_r, data_i, weights, flags, t_chunk_idx, freq_coeffs, numerators, denom
                )

            if verbose and (chunk_idx + 1) % 20 == 0:
                pct = (row_end / nrows) * 100.0
                logger.info(f"Processed chunk {chunk_idx + 1}/{n_chunks} ({pct:.1f}% rows)...")

    finally:
        tb.close()

    # 7. Form Final Dynamic Spectra for all Targets
    t_sec = unique_times - unique_times[0]
    time_mjd = unique_times / 86400.0

    results: Dict[str, Dict[str, Any]] = {}
    for k, meta in enumerate(cand_metadata):
        c_id = meta["id"]
        num_k = numerators[k]

        with np.errstate(divide="ignore", invalid="ignore"):
            spec = np.where(denom > 0.0, num_k / denom, 0.0)
        spec[~np.isfinite(spec)] = 0.0

        results[c_id] = {
            "data": spec,
            "time": t_sec,
            "time_raw": unique_times,
            "time_mjd": time_mjd,
            "freq": freq_mhz,
            "ra_str": meta["ra_str"],
            "dec_str": meta["dec_str"],
            "ra_deg": meta["ra_deg"],
            "dec_deg": meta["dec_deg"],
            "telescope": "ASKAP",
            "pol": f"Stokes {pol}",
            "cand_id": c_id,
        }

    t_total = time.time() - t_start
    if verbose:
        logger.info(
            f"Successfully extracted {len(results)} dynamic spectra in {t_total:.2f}s "
            f"({t_total / max(1, len(results)):.2f}s per candidate)."
        )

    return results


def extract_phased_dynamic_spectra_from_chunk(
    times: np.ndarray,
    chan_freq: np.ndarray,
    vis_3d: np.ndarray,
    wgt_3d: np.ndarray,
    uvw_3d: np.ndarray,
    ra0_rad: float,
    dec0_rad: float,
    targets: List[Dict[str, Any]],
    *,
    pol: str = "I",
    min_uv_distance_m: float = 0.0,
    telescope: str = "ASKAP",
) -> Dict[str, Dict[str, Any]]:
    """
    Extract phased, baseline-averaged dynamic spectra directly from an in-memory
    visibility chunk cube (already loaded by ``imaging.load_chunk_data``) with
    zero additional Measurement Set disk I/O.

    Parameters
    ----------
    times : np.ndarray
        1D float64 array of chunk timestamps (seconds since MJD 0), shape ``(nt,)``.
    chan_freq : np.ndarray
        1D float64 array of channel frequencies in Hz, shape ``(nchan,)``.
    vis_3d : np.ndarray
        Complex visibilities, shape ``(nbl, nchan, nt)``.
    wgt_3d : np.ndarray
        Real weights (zero for flagged samples), shape ``(nbl, nchan, nt)``.
    uvw_3d : np.ndarray
        Baseline UVW coordinates in metres, shape ``(nbl, 3, nt)``.
    ra0_rad, dec0_rad : float
        Phase centre RA and Dec in radians.
    targets : list of dict
        Each dict must contain ``'id'`` and either ``('ra_rad', 'dec_rad')`` or
        ``('ra_deg', 'dec_deg')``.
    pol : str, default "I"
        Polarisation label for metadata.
    min_uv_distance_m : float, default 0.0
        Optional minimum baseline UV distance in metres.
    telescope : str, default "ASKAP"
        Telescope name for metadata.

    Returns
    -------
    dict
        Mapping ``cand_id -> dict`` matching ``extract_phased_dynamic_spectra_from_ms``.
    """
    if not targets:
        return {}

    times_arr = np.asarray(times, dtype=np.float64)
    freq_hz = np.asarray(chan_freq, dtype=np.float64)
    nbl, nchan, nt = vis_3d.shape
    if len(times_arr) != nt:
        raise ValueError(f"times length ({len(times_arr)}) does not match vis_3d time axis ({nt})")
    if len(freq_hz) != nchan:
        raise ValueError(f"chan_freq length ({len(freq_hz)}) does not match vis_3d channel axis ({nchan})")

    freq_mhz = freq_hz / 1e6
    freq_coeffs = (2.0 * np.pi / C_LIGHT * freq_hz).astype(np.float32)

    K = len(targets)
    target_vecs = np.zeros((K, 3), dtype=np.float64)
    cand_metadata = []

    for k, tgt in enumerate(targets):
        c_id = tgt.get("id", f"cand_{k}")
        if "ra_rad" in tgt and "dec_rad" in tgt:
            ra_rad = float(tgt["ra_rad"])
            dec_rad = float(tgt["dec_rad"])
            ra_deg = float(np.degrees(ra_rad))
            dec_deg = float(np.degrees(dec_rad))
        else:
            ra_deg = float(tgt["ra_deg"])
            dec_deg = float(tgt["dec_deg"])
            ra_rad = float(np.radians(ra_deg))
            dec_rad = float(np.radians(dec_deg))

        l_k = math.cos(dec_rad) * math.sin(ra_rad - ra0_rad)
        m_k = math.sin(dec_rad) * math.cos(dec0_rad) - math.cos(dec_rad) * math.sin(dec0_rad) * math.cos(ra_rad - ra0_rad)
        n_k = math.sqrt(max(0.0, 1.0 - l_k * l_k - m_k * m_k))
        target_vecs[k] = [l_k, m_k, n_k - 1.0]

        sc = SkyCoord(ra_deg, dec_deg, unit="deg")
        ra_str, dec_str = sc.to_string("hmsdms").split()
        cand_metadata.append({
            "id": c_id,
            "ra_deg": ra_deg,
            "dec_deg": dec_deg,
            "ra_str": ra_str,
            "dec_str": dec_str,
        })

    # Geometric delays P[k, b, t] = u*l_k + v*m_k + w*(n_k - 1)
    P = np.einsum("kd,bdt->kbt", target_vecs, uvw_3d).astype(np.float32, copy=False)

    vis_r = np.ascontiguousarray(vis_3d.real, dtype=np.float32)
    vis_i = np.ascontiguousarray(vis_3d.imag, dtype=np.float32)
    wgt_f = np.ascontiguousarray(wgt_3d, dtype=np.float32)

    if min_uv_distance_m and min_uv_distance_m > 0.0:
        uv_dist2 = uvw_3d[:, 0, :] ** 2 + uvw_3d[:, 1, :] ** 2
        short_mask = uv_dist2 < (float(min_uv_distance_m) ** 2)
        if np.any(short_mask):
            wgt_f = np.where(short_mask[:, None, :], np.float32(0.0), wgt_f)

    numerators = np.zeros((K, nt, nchan), dtype=np.float32)
    denom = np.zeros((nt, nchan), dtype=np.float32)

    _accumulate_chunk_cube(P, vis_r, vis_i, wgt_f, freq_coeffs, numerators, denom)

    t_sec = times_arr - times_arr[0] if nt > 0 else np.array([], dtype=np.float64)
    time_mjd = times_arr / 86400.0

    results: Dict[str, Dict[str, Any]] = {}
    for k, meta in enumerate(cand_metadata):
        c_id = meta["id"]
        num_k = numerators[k]
        with np.errstate(divide="ignore", invalid="ignore"):
            spec = np.where(denom > 0.0, num_k / denom, np.float32(0.0))
        spec[~np.isfinite(spec)] = 0.0

        results[c_id] = {
            "data": spec,
            "time": t_sec,
            "time_raw": times_arr,
            "time_mjd": time_mjd,
            "freq": freq_mhz,
            "ra_str": meta["ra_str"],
            "dec_str": meta["dec_str"],
            "ra_deg": meta["ra_deg"],
            "dec_deg": meta["dec_deg"],
            "telescope": telescope,
            "pol": f"Stokes {pol}" if pol.upper() == "I" else pol,
            "cand_id": c_id,
        }

    return results


# =============================================================================
# HDF5 I/O Utilities for Dynamic Spectra (.ds)
# =============================================================================

def write_dynamic_spectrum_hdf5(
    spec_data: Dict[str, Any],
    out_path: Union[str, Path],
    datacolumn: str = "data",
    attrs: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Write an in-memory dynamic spectrum dictionary to a standard HDF5 (.ds) file
    compatible with dstools, pdmp_refine, and downstream transient/pulsar pipelines.

    Parameters
    ----------
    spec_data : dict
        Dynamic spectrum dictionary as returned by extract_phased_dynamic_spectra_from_ms.
    out_path : str or Path
        Output file path for the .ds HDF5 file.
    datacolumn : str, default "data"
        Data column identifier (e.g. "DATA", "CORRECTED_DATA").
    attrs : dict, optional
        Extra header attributes to record in the HDF5 root group.

    Returns
    -------
    Path
        Resolved path to the saved .ds file.
    """
    import h5py

    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.asarray(spec_data["data"], dtype=np.float32)
    time_raw = np.asarray(spec_data.get("time_raw", spec_data.get("time", [])), dtype=np.float64)
    freq_hz = np.asarray(spec_data["freq"], dtype=np.float64)
    # If freq is in MHz (< 1e5), convert to Hz
    if np.max(freq_hz) < 1e6:
        freq_hz = freq_hz * 1e6

    # Shape is (1, Ntime, Nchan, 1) to match dstools single-pol baseline-averaged format
    flux_4d = data[np.newaxis, :, :, np.newaxis].astype(np.complex64)

    with h5py.File(out_path, "w", track_order=True) as f:
        f.attrs["generator"] = "fastducc.dynspec"
        f.attrs["phasecentre"] = f"{spec_data.get('ra_str', '')} {spec_data.get('dec_str', '')}".strip()
        f.attrs["telescope"] = spec_data.get("telescope", "ASKAP")
        f.attrs["datacolumn"] = datacolumn.upper()
        f.attrs["pb_scale"] = 1.0
        f.attrs["polarisations"] = 1
        f.attrs["cand_id"] = str(spec_data.get("cand_id", ""))
        f.attrs["ra_deg"] = float(spec_data.get("ra_deg", 0.0))
        f.attrs["dec_deg"] = float(spec_data.get("dec_deg", 0.0))

        if attrs:
            for k, v in attrs.items():
                if k not in f.attrs:
                    f.attrs[k] = v

        f.create_dataset("time", data=time_raw)
        f.create_dataset("frequency", data=freq_hz)
        f.create_dataset("uvdist", data=np.array([0.0], dtype=np.float64))
        f.create_dataset("flux", data=flux_4d)

    return out_path


def load_dynamic_spectrum_hdf5(
    ds_path: Union[str, Path],
    pol: str = "I",
) -> Dict[str, Any]:
    """
    Load a standard .ds HDF5 dynamic spectrum file into a normalized in-memory dictionary.

    Parameters
    ----------
    ds_path : str or Path
        Path to the .ds HDF5 file.
    pol : str, default "I"
        Polarization to extract ("I", "XX", etc.).

    Returns
    -------
    dict
        Dictionary containing:
          - 'data': 2D float32 array (Ntime, Nchan)
          - 'time': 1D float64 array of elapsed seconds
          - 'time_raw': 1D float64 array of timestamps
          - 'time_mjd': 1D float64 array of MJD timestamps
          - 'freq': 1D float64 array of channel frequencies in MHz
          - 'ra_str', 'dec_str': coordinates string if recorded
          - 'attrs': dictionary of HDF5 root attributes
    """
    import h5py

    ds_path = Path(ds_path).resolve()
    if not ds_path.is_file():
        raise FileNotFoundError(f"Dynamic spectrum file not found: {ds_path}")

    with h5py.File(ds_path, "r") as f:
        attrs = dict(f.attrs)
        time_raw = np.array(f["time"][:], dtype=np.float64)
        freq_raw = np.array(f["frequency"][:], dtype=np.float64)
        freq_mhz = freq_raw / 1e6 if np.max(freq_raw) > 1e6 else freq_raw
        flux_raw = f["flux"][:]

        # Extract Stokes I or requested polarization
        if flux_raw.ndim == 4:
            if flux_raw.shape[-1] >= 2 and pol.upper() in ["I", "STOKES I"]:
                data = 0.5 * np.real(flux_raw[0, :, :, 0] + flux_raw[0, :, :, -1])
            else:
                data = np.real(flux_raw[0, :, :, 0])
        elif flux_raw.ndim == 3:
            data = np.real(flux_raw[0, :, :])
        elif flux_raw.ndim == 2:
            data = np.real(flux_raw)
        else:
            data = np.real(np.squeeze(flux_raw))

        data = data.astype(np.float32)

    t_sec = time_raw - time_raw[0] if len(time_raw) > 0 else np.array([], dtype=np.float64)
    # Check if time_raw looks like MJD (e.g. ~60000) or seconds (> 1e7)
    if len(time_raw) > 0 and time_raw[0] < 1e5:
        time_mjd = time_raw
    else:
        time_mjd = time_raw / 86400.0 if len(time_raw) > 0 else np.array([], dtype=np.float64)

    phasecentre = str(attrs.get("phasecentre", ""))
    ra_str, dec_str = "", ""
    if phasecentre:
        parts = phasecentre.split()
        if len(parts) >= 2:
            ra_str, dec_str = parts[0], parts[1]

    return {
        "data": data,
        "time": t_sec,
        "time_raw": time_raw,
        "time_mjd": time_mjd,
        "freq": freq_mhz,
        "ra_str": ra_str,
        "dec_str": dec_str,
        "telescope": str(attrs.get("telescope", "ASKAP")),
        "pol": f"Stokes {pol}",
        "cand_id": str(attrs.get("cand_id", ds_path.stem)),
        "path": ds_path,
        "attrs": attrs,
    }


# =============================================================================
# Beam-Batched Catalogue Orchestration (Shared by Periodicity & Transient Pipelines)
# =============================================================================

def resolve_worker_threads(default: int = 2) -> int:
    """Resolve thread count for a Dask/Slurm worker from environment variables."""
    for env_var in ("SLURM_CPUS_PER_TASK", "NUMBA_NUM_THREADS", "OMP_NUM_THREADS"):
        val = os.environ.get(env_var, "").strip()
        if val.isdigit() and int(val) > 0:
            return int(val)
    return max(1, int(default))


def normalise_beam_name(beam_val: str) -> str:
    """Normalise a beam label (e.g. '5', 'beam5', 'beam05') to canonical 'beamNN'."""
    val = str(beam_val or "").strip()
    if not val or val == "None":
        return "beam00"
    first = val.split(",")[0].split()[0]
    digits = re.sub(r"[^0-9]", "", first)
    return f"beam{int(digits):02d}" if digits else first


def _safe_cand_name(name: str) -> str:
    """Sanitise a candidate name for filesystem paths."""
    cleaned = re.sub(r"[^A-Za-z0-9._+-]+", "_", str(name).strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "SRC"


def find_ms_for_beam(
    ms_dir: Union[str, Path],
    beam_name: str,
    ms_glob_template: str = "**/cracoData*%s*uvsub.ms",
) -> Optional[Path]:
    """Locate the Measurement Set corresponding to a beam under ``ms_dir``."""
    ms_root = Path(ms_dir)
    if not ms_root.is_dir():
        return None

    b_norm = normalise_beam_name(beam_name)
    digits = re.sub(r"[^0-9]", "", b_norm)

    patterns: List[str] = []
    if "%s" in ms_glob_template:
        patterns.append(ms_glob_template % b_norm)
        if digits:
            patterns.append(ms_glob_template % f"beam{int(digits)}")
            patterns.append(ms_glob_template % digits)
    else:
        patterns.append(ms_glob_template)

    patterns.extend([
        f"*{b_norm}*uvsub.ms",
        f"*{b_norm}*.ms",
    ])

    for pat in patterns:
        matches = sorted(ms_root.glob(pat))
        if matches:
            return matches[0]
        stripped = re.sub(r"^[*/]+", "", pat)
        if stripped != pat:
            matches = sorted(ms_root.glob(stripped))
            if matches:
                return matches[0]
    return None


def discover_obs_catalogue(
    sbid_dir: Union[str, Path],
    sbid: str,
    kind: str = "boxcar",
    sbid_subdir: Optional[str] = None,
) -> Path:
    """Auto-discover the observation-level candidate summary catalogue for an SBID."""
    sbid_path = Path(sbid_dir)
    cand_dir = (sbid_path / sbid_subdir / "candidates") if sbid_subdir else (sbid_path / "candidates")
    for stem_pat in (
        f"*.{sbid}_obs_{kind}_super_summary",
        f"*.{sbid}_field_{kind}_super_summary",
        f"*_obs_{kind}_top_candidates",
        f"*_obs_{kind}_super_summary",
    ):
        for ext in ("vot", "xml", "csv"):
            matches = sorted(cand_dir.glob(f"{stem_pat}.{ext}"))
            if matches:
                return matches[0]
    raise FileNotFoundError(f"No candidate catalogue found under {cand_dir} for sbid={sbid}, kind={kind}")


def _parse_field_from_catalogue_path(cat_path: Path) -> str:
    m = re.match(
        r"^(?P<field>[^.]+)\.SB\d{5,}_(?:obs|field)_[A-Za-z0-9_]+\.(?:vot|xml|csv)$",
        cat_path.name,
        re.IGNORECASE,
    )
    return m.group("field") if m else ""


def _extract_sbid_from_path(path: Path) -> str:
    for token in (path.name, *[p.name for p in path.parents]):
        m = re.search(r"(SB\d{5,})", token, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return ""


def _update_transient_summary_plots_for_candidate(
    cand: Dict[str, Any],
    ds_file: Path,
    max_plots: int = 10,
) -> bool:
    """Locate existing transient summary plots (or cutouts) for a candidate and update them with ``ds_file``."""
    from fastducc.candidate_tools.plotting import update_candidate_summary_with_dynspec

    srcname = str(cand["srcname"])
    search_dirs: List[Path] = [
        ds_file.parent,
        ds_file.parent / "plots",
        ds_file.parent.parent / "candidates",
        ds_file.parent.parent / "plots",
    ]

    sbid_dir = None
    for parent in ds_file.parents:
        if re.match(r"^SB\d+$", parent.name):
            sbid_dir = parent
            break
    if sbid_dir is not None:
        for k in ("boxcar", "variance"):
            cand_plot_dir = sbid_dir / "candidates" / k / srcname / "plots"
            if cand_plot_dir.is_dir():
                search_dirs.append(cand_plot_dir)

    found_plots: List[Path] = []
    seen = set()
    for sdir in search_dirs:
        if not sdir.is_dir():
            continue
        matches = (
            list(sdir.glob(f"*{srcname}*_summary.pdf"))
            + list(sdir.glob(f"*cand_{srcname}*_summary.pdf"))
        )
        for p in matches:
            resolved = p.resolve()
            if resolved not in seen and resolved.is_file():
                seen.add(resolved)
                found_plots.append(resolved)

    if not found_plots:
        for sdir in search_dirs:
            if not sdir.is_dir():
                continue
            for cd in (sdir, sdir / "cutouts", sdir.parent / "cutouts"):
                if not cd.is_dir():
                    continue
                cut_matches = (
                    list(cd.glob(f"*{srcname}*cutout.fits"))
                    + list(cd.glob(f"*cand_{srcname}*cutout.fits"))
                )
                for c_file in cut_matches:
                    if c_file.is_file():
                        base_cand_dir = c_file.parent.parent if c_file.parent.name == "cutouts" else c_file.parent
                        target_plot_dir = base_cand_dir / "plots"
                        target_plot_dir.mkdir(parents=True, exist_ok=True)
                        synth_plot = target_plot_dir / c_file.name.replace("_cutout.fits", "_summary.pdf")
                        if synth_plot not in seen:
                            seen.add(synth_plot)
                            found_plots.append(synth_plot)

    if not found_plots:
        return False

    target_mjd = None
    if cand.get("max_snr_time_center") is not None:
        try:
            target_mjd = float(cand["max_snr_time_center"]) / 86400.0
        except (TypeError, ValueError):
            target_mjd = None
    target_width = cand.get("max_snr_width")
    target_scan = str(cand.get("max_snr_scan_id") or "")
    target_beam = normalise_beam_name(str(cand.get("beam_name") or ""))
    target_method = str(cand.get("method") or "")

    def _sort_key(p: Path) -> float:
        name = p.name
        score = 0.0
        if target_mjd is not None:
            mjd_m = re.search(r"MJD([\d.]+)", name)
            if mjd_m:
                try:
                    delta_sec = abs(float(mjd_m.group(1)) - target_mjd) * 86400.0
                    if delta_sec < 1.0:
                        score += 1000.0
                    elif delta_sec < 60.0:
                        score += max(0.0, 500.0 - delta_sec * 5.0)
                except ValueError:
                    pass
        w_m = re.search(r"_w(\d+)", name)
        if w_m and target_width is not None:
            file_w = int(w_m.group(1))
            score += 200.0 if file_w == int(target_width) else max(0.0, 50.0 - abs(file_w - int(target_width)))
        if target_scan and target_scan != "combined" and target_scan in name:
            score += 100.0
        if target_beam and target_beam in name:
            score += 80.0
        if target_method and target_method in name:
            score += 50.0
        return -score

    found_plots.sort(key=_sort_key)

    ds_mtime = ds_file.stat().st_mtime if ds_file.exists() else 0.0
    cand_info = {
        "srcname": srcname,
        "snr": cand.get("snr", np.nan),
        "dm": cand.get("dm", 0.0),
        "sbid": cand.get("sbid", ""),
        "beam": target_beam,
        "scan": target_scan or "combined",
        "field": cand.get("field", ""),
        "ra_deg": cand.get("ra_deg"),
        "dec_deg": cand.get("dec_deg"),
        "width": target_width,
        "method": target_method,
    }

    updated_any = False
    count = 0
    for p in found_plots:
        png_p = p.with_suffix(".png")
        if png_p.is_file() and png_p.stat().st_mtime >= ds_mtime:
            continue
        try:
            update_candidate_summary_with_dynspec(p, ds_file, candidate_info=cand_info)
            updated_any = True
            count += 1
            if max_plots > 0 and count >= max_plots:
                break
        except Exception as exc:
            logger.warning("Failed to update summary plot %s with %s: %s", p, ds_file.name, exc)

    return updated_any


def _extract_beam_batch_task(task_spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute batched streaming dynamic spectrum extraction for one beam batch.
    Top-level picklable worker entry point for Dask and serial execution.
    """
    t0 = time.time()
    beam_name = str(task_spec["beam_name"])
    batch_index = int(task_spec["batch_index"])
    ms_file_str = task_spec.get("ms_file")
    targets_to_extract: List[Dict[str, Any]] = list(task_spec.get("targets_to_extract") or [])
    candidates: List[Dict[str, Any]] = list(task_spec.get("candidates") or [])
    cfg: Dict[str, Any] = dict(task_spec["extract_kwargs"])

    overwrite = bool(cfg.get("overwrite", False))
    dry_run = bool(cfg.get("dry_run", False))
    update_summary_plots = bool(cfg.get("update_summary_plots", True))
    datacolumn = str(cfg.get("datacolumn", "DATA"))
    pol = str(cfg.get("pol", "I"))
    chunk_rows = int(cfg.get("chunk_rows", 100000))
    min_uv_distance_m = float(cfg.get("min_uv_distance_m", 0.0) or 0.0)

    results: List[Dict[str, Any]] = []
    pending_cands: List[Dict[str, Any]] = []
    needed_target_ids = set()

    for cand in candidates:
        out_file = Path(cand["out_file"])
        if cand.get("skip_reason"):
            results.append({
                "srcname": cand["srcname"],
                "out_file": str(out_file),
                "status": "SKIPPED",
                "message": str(cand["skip_reason"]),
            })
            continue

        if (not overwrite) and out_file.is_file() and out_file.stat().st_size > 0:
            plot_updated = False
            if update_summary_plots and not dry_run:
                plot_updated = _update_transient_summary_plots_for_candidate(cand, out_file)
            msg = "exists" + (" (summary plot updated)" if plot_updated else "")
            results.append({
                "srcname": cand["srcname"],
                "out_file": str(out_file),
                "status": "EXISTS",
                "message": msg,
            })
            continue

        pending_cands.append(cand)
        if cand.get("target_id"):
            needed_target_ids.add(cand["target_id"])

    if not pending_cands:
        return {
            "beam_name": beam_name,
            "batch_index": batch_index,
            "results": results,
            "elapsed_sec": time.time() - t0,
        }

    if dry_run:
        for cand in pending_cands:
            results.append({
                "srcname": cand["srcname"],
                "out_file": str(cand["out_file"]),
                "status": "DRY_RUN",
                "message": f"would extract from {ms_file_str}",
            })
        return {
            "beam_name": beam_name,
            "batch_index": batch_index,
            "results": results,
            "elapsed_sec": time.time() - t0,
        }

    pending_targets = [t for t in targets_to_extract if t["id"] in needed_target_ids]
    extracted_specs: Dict[str, Dict[str, Any]] = {}
    extraction_error: Optional[str] = None

    if not ms_file_str:
        extraction_error = f"Measurement Set not found for beam {beam_name}"
        logger.error("%s", extraction_error)
    elif pending_targets:
        ms_file = Path(ms_file_str)
        n_threads = resolve_worker_threads()
        logger.info(
            "Streaming extraction for %d unique position(s) in %s batch %d from %s (chunk_rows=%d, threads=%d)...",
            len(pending_targets),
            beam_name,
            batch_index,
            ms_file.name,
            chunk_rows,
            n_threads,
        )
        try:
            extracted_specs = extract_phased_dynamic_spectra_from_ms(
                ms_path=ms_file,
                targets=pending_targets,
                chunk_rows=chunk_rows,
                pol=pol,
                datacolumn=datacolumn,
                min_uv_distance_m=min_uv_distance_m,
                n_threads=n_threads,
                verbose=bool(cfg.get("verbose", False)),
            )
        except Exception as exc:
            extraction_error = f"{type(exc).__name__}: {exc}"
            logger.error(
                "Streaming extraction failed for %s batch %d: %s",
                beam_name,
                batch_index,
                exc,
                exc_info=True,
            )
            extracted_specs = {}

    try:
        written_files = set()
        for cand in pending_cands:
            out_file = Path(cand["out_file"])
            t_id = cand.get("target_id")
            spec_data = extracted_specs.get(t_id) if t_id else None
            if spec_data is None:
                results.append({
                    "srcname": cand["srcname"],
                    "out_file": str(out_file),
                    "status": "FAILED",
                    "message": extraction_error or f"Target {t_id} missing from extraction output",
                })
                continue

            try:
                if out_file not in written_files:
                    spec_copy = dict(spec_data)
                    spec_copy["cand_id"] = cand["srcname"]
                    extra_attrs = {"dm": float(cand.get("dm", 0.0))} if cand.get("dm") else {}
                    write_dynamic_spectrum_hdf5(
                        spec_data=spec_copy,
                        out_path=out_file,
                        datacolumn=datacolumn,
                        attrs=extra_attrs,
                    )
                    written_files.add(out_file)

                plot_updated = False
                if update_summary_plots:
                    plot_updated = _update_transient_summary_plots_for_candidate(cand, out_file)

                msg = "ok" + (" (summary plot updated)" if plot_updated else "")
                results.append({
                    "srcname": cand["srcname"],
                    "out_file": str(out_file),
                    "status": "OK",
                    "message": msg,
                })
            except Exception as exc:
                logger.error("Failed writing/plotting %s: %s", out_file, exc, exc_info=True)
                results.append({
                    "srcname": cand["srcname"],
                    "out_file": str(out_file),
                    "status": "FAILED",
                    "message": f"{type(exc).__name__}: {exc}",
                })
    finally:
        extracted_specs.clear()
        gc.collect()

    return {
        "beam_name": beam_name,
        "batch_index": batch_index,
        "results": results,
        "elapsed_sec": time.time() - t0,
    }


def batch_extract_catalogue(
    *,
    catalogue_path: Optional[Union[str, Path]] = None,
    sbid: Optional[str] = None,
    data_root: Optional[Union[str, Path]] = None,
    kind: str = "boxcar",
    sbid_subdir: Optional[str] = None,
    ms_dir: Optional[Union[str, Path]] = None,
    combined_dir: str = "native_combined",
    ms_glob_template: str = "**/cracoData*%s*uvsub.ms",
    out_dir: Optional[Union[str, Path]] = None,
    min_snr: Optional[float] = None,
    min_dm: Optional[float] = None,
    max_cands: Optional[int] = None,
    source_id: Optional[str] = None,
    max_batch_cands_per_beam: int = 10,
    chunk_rows: int = 100000,
    pol: str = "I",
    datacolumn: str = "DATA",
    min_uv_distance_m: float = 0.0,
    overwrite: bool = False,
    dry_run: bool = False,
    update_summary_plots: bool = True,
    verbose: bool = False,
    parallel_mode: str = "serial",
    dask_scheduler_address: Optional[str] = None,
    dask_min_workers: int = 1,
    dask_max_workers: int = 48,
    dask_local_workers: int = 4,
    dask_local_mem: str = "6GB",
    dask_slurm_partition: Optional[str] = None,
    dask_slurm_account: Optional[str] = None,
    dask_slurm_cores: int = 2,
    dask_slurm_mem: str = "6GB",
    dask_slurm_walltime: str = "00:30:00",
    dask_slurm_interface: Optional[str] = None,
    dask_slurm_log_dir: Optional[str] = None,
    dask_job_name: Optional[str] = None,
    dask_target_duration: str = "1500s",
) -> Dict[str, Any]:
    """
    Extract phased dynamic spectra for a candidate catalogue in per-beam batches using
    fastducc's in-memory visibility streaming engine, with unified serial / Dask-local /
    Dask-SLURM execution matching ``lotrun_periodicity``.
    """
    from astropy.table import Table

    mode = str(parallel_mode or "serial").strip().lower()
    if mode not in ("serial", "dask-local", "dask-slurm"):
        raise ValueError(
            f"Unsupported parallel_mode '{parallel_mode}'. Expected 'serial', 'dask-local', or 'dask-slurm'."
        )

    if catalogue_path is not None:
        cat_path = Path(catalogue_path).resolve()
        sbid_dir = Path(data_root) / sbid if (data_root and sbid) else cat_path.parent.parent
    elif sbid and data_root:
        sbid_dir = (Path(data_root) / sbid).resolve()
        cat_path = discover_obs_catalogue(sbid_dir, sbid, kind=kind, sbid_subdir=sbid_subdir)
    else:
        raise ValueError("Provide either catalogue_path or both sbid and data_root.")

    if not cat_path.is_file():
        raise FileNotFoundError(f"Catalogue file not found: {cat_path}")

    sbid_lbl = sbid or _extract_sbid_from_path(cat_path)
    fieldname = _parse_field_from_catalogue_path(cat_path) or "field"

    if ms_dir is not None:
        ms_dir_path = Path(ms_dir).resolve()
    else:
        cand_ms_dirs = [sbid_dir / combined_dir, sbid_dir, cat_path.parent]
        ms_dir_path = next((d for d in cand_ms_dirs if d.is_dir() and list(d.glob("*.ms"))), sbid_dir / combined_dir)

    target_out_dir = Path(out_dir).resolve() if out_dir is not None else (ms_dir_path / "candidates")
    target_out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Reading candidate catalogue: %s", cat_path)
    if cat_path.suffix.lower() in (".vot", ".xml"):
        tbl = Table.read(cat_path, format="votable")
    else:
        tbl = Table.read(cat_path, format="ascii.csv")

    if len(tbl) == 0:
        logger.info("Catalogue is empty; nothing to extract.")
        return {"completed": 0, "ok": 0, "exists": 0, "skipped": 0, "failed": 0, "results": []}

    def _get_col(*names: str) -> Optional[str]:
        return next((n for n in names if n in tbl.colnames), None)

    col_sid = _get_col("source_id", "srcname", "source_name", "name")
    col_name = _get_col("srcname", "source_name", "name", "source_id")
    col_ra = _get_col("ra_deg")
    col_dec = _get_col("dec_deg")
    col_snr = _get_col("max_snr", "snr", "SNR", "peak_snr")
    col_beam = _get_col("max_snr_beam", "beam", "beams_all", "primary_beam")
    col_dm = _get_col("best_dm", "dm", "DM", "dm_all")
    col_time = _get_col("max_snr_time_center", "time_center")
    col_width = _get_col("max_snr_width", "width_samples")
    col_scan = _get_col("max_snr_scan_id", "scan_id")
    col_method = _get_col("method")

    if not (col_ra and col_dec and col_name):
        raise ValueError(f"Catalogue {cat_path} missing required columns (need srcname/name, ra_deg, dec_deg)")

    selected_rows: List[Dict[str, Any]] = []
    for row in tbl:
        sid = str(row[col_sid]).strip() if col_sid else str(row[col_name]).strip()
        if source_id and sid != str(source_id) and str(row[col_name]).strip() != str(source_id):
            continue

        snr_val = float("nan")
        if col_snr:
            try:
                snr_val = float(str(row[col_snr]).split(",")[0])
            except (TypeError, ValueError):
                snr_val = float("nan")
        if min_snr is not None and math.isfinite(snr_val) and snr_val < float(min_snr):
            continue

        dm_val = 0.0
        if col_dm:
            try:
                dm_val = float(str(row[col_dm]).split(",")[0])
            except (TypeError, ValueError):
                dm_val = 0.0
        if min_dm is not None and dm_val < float(min_dm):
            continue

        ra_deg = float(row[col_ra]) % 360.0
        dec_deg = float(row[col_dec])
        srcname = _safe_cand_name(str(row[col_name]))
        beam_name = normalise_beam_name(str(row[col_beam]) if col_beam else "")

        max_time = None
        if col_time:
            try:
                t_v = float(row[col_time])
                if math.isfinite(t_v):
                    max_time = t_v
            except (TypeError, ValueError):
                pass

        max_width = None
        if col_width:
            try:
                max_width = int(float(row[col_width]))
            except (TypeError, ValueError):
                pass

        max_scan = str(row[col_scan]).strip().split(",")[0].split()[0] if col_scan else ""
        cand_method = str(row[col_method]).strip() if col_method else kind

        selected_rows.append({
            "source_id": sid,
            "srcname": srcname,
            "ra_deg": ra_deg,
            "dec_deg": dec_deg,
            "snr": snr_val,
            "dm": dm_val,
            "beam_name": beam_name,
            "field": fieldname,
            "sbid": sbid_lbl,
            "max_snr_time_center": max_time,
            "max_snr_width": max_width,
            "max_snr_scan_id": max_scan,
            "method": cand_method,
        })

    selected_rows.sort(
        key=lambda r: r["snr"] if math.isfinite(r["snr"]) else -1e9,
        reverse=True,
    )
    if max_cands is not None and len(selected_rows) > int(max_cands):
        selected_rows = selected_rows[: int(max_cands)]

    if not selected_rows:
        logger.info("No candidates remain after filtering; nothing to extract.")
        return {"completed": 0, "ok": 0, "exists": 0, "skipped": 0, "failed": 0, "results": []}

    beam_groups: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in selected_rows:
        beam_groups[r["beam_name"]].append(r)

    batch_size = max(1, int(max_batch_cands_per_beam))
    shared_kwargs = {
        "overwrite": overwrite,
        "dry_run": dry_run,
        "update_summary_plots": update_summary_plots,
        "datacolumn": datacolumn,
        "pol": pol,
        "chunk_rows": int(chunk_rows),
        "min_uv_distance_m": float(min_uv_distance_m or 0.0),
        "verbose": verbose,
    }

    batch_tasks: List[Dict[str, Any]] = []
    for beam_name in sorted(beam_groups.keys()):
        beam_rows = beam_groups[beam_name]
        ms_file = find_ms_for_beam(ms_dir_path, beam_name, ms_glob_template=ms_glob_template)
        ms_file_str = str(ms_file.resolve()) if ms_file is not None else None

        for b_idx, b_start in enumerate(range(0, len(beam_rows), batch_size)):
            chunk_rows_list = beam_rows[b_start : b_start + batch_size]
            targets_to_extract: List[Dict[str, Any]] = []
            batch_cands: List[Dict[str, Any]] = []
            coord_map: Dict[Tuple[float, float], str] = {}

            for r in chunk_rows_list:
                out_name = f"{fieldname}.{sbid_lbl}.{beam_name}.combined_cand_{r['srcname']}.ds" if sbid_lbl else f"{fieldname}.{beam_name}.combined_cand_{r['srcname']}.ds"
                out_file = target_out_dir / out_name

                coord_key = (round(r["ra_deg"], 5), round(r["dec_deg"], 5))
                if coord_key in coord_map:
                    t_id = coord_map[coord_key]
                else:
                    t_id = f"cand_t{len(targets_to_extract)}_{r['srcname']}"
                    coord_map[coord_key] = t_id
                    targets_to_extract.append({
                        "id": t_id,
                        "ra_deg": r["ra_deg"],
                        "dec_deg": r["dec_deg"],
                    })

                c_entry = dict(r)
                c_entry["out_file"] = str(out_file)
                c_entry["target_id"] = t_id
                batch_cands.append(c_entry)

            batch_tasks.append({
                "beam_name": beam_name,
                "batch_index": b_idx,
                "ms_file": ms_file_str,
                "targets_to_extract": targets_to_extract,
                "candidates": batch_cands,
                "extract_kwargs": shared_kwargs,
            })

    n_tasks = len(batch_tasks)
    logger.info(
        "Prepared %d beam-batch extraction task(s) for %d candidate(s) across %d beam(s) (mode=%s).",
        n_tasks,
        len(selected_rows),
        len(beam_groups),
        mode,
    )

    batch_outputs: List[Dict[str, Any]] = []
    if mode == "serial":
        for i_t, spec in enumerate(batch_tasks, start=1):
            logger.info(
                "[%d/%d] Extracting %s batch %d (%d candidate(s))...",
                i_t,
                n_tasks,
                spec["beam_name"],
                spec["batch_index"],
                len(spec["candidates"]),
            )
            batch_outputs.append(_extract_beam_batch_task(spec))
    else:
        from dask.distributed import as_completed
        from fastducc.dask_utils import get_dask_client

        eff_max = max(1, min(max(1, int(dask_max_workers)), n_tasks))
        eff_min = max(1, min(max(1, int(dask_min_workers)), eff_max))
        local_w = max(1, min(int(dask_local_workers), n_tasks))

        client = None
        cluster = None
        try:
            client, cluster = get_dask_client(
                parallel_mode=mode,
                scheduler_address=dask_scheduler_address,
                stage_name="dynspec",
                sbid=sbid_lbl or None,
                dask_job_name=dask_job_name,
                slurm_log_dir=dask_slurm_log_dir,
                output_dir=str(target_out_dir),
                slurm_partition=dask_slurm_partition,
                slurm_account=dask_slurm_account,
                slurm_cores_per_worker=int(dask_slurm_cores),
                slurm_mem=str(dask_slurm_mem),
                slurm_walltime=str(dask_slurm_walltime),
                slurm_interface=dask_slurm_interface,
                min_workers=eff_min,
                max_workers=eff_max,
                local_workers=local_w,
                local_memory_limit=str(dask_local_mem),
                threads_per_worker=1,
                target_duration=str(dask_target_duration),
            )
            future_to_spec = {}
            for spec in batch_tasks:
                key = f"dynspec-{spec['beam_name']}-b{spec['batch_index']:03d}"
                fut = client.submit(_extract_beam_batch_task, spec, key=key, pure=False)
                future_to_spec[fut] = spec

            n_done = 0
            for fut in as_completed(list(future_to_spec.keys())):
                n_done += 1
                spec = future_to_spec[fut]
                try:
                    out = fut.result()
                    batch_outputs.append(out)
                    logger.info(
                        "[%d/%d] Completed %s batch %d (%d candidate(s)) in %.1f s.",
                        n_done,
                        n_tasks,
                        spec["beam_name"],
                        spec["batch_index"],
                        len(out["results"]),
                        float(out.get("elapsed_sec", 0.0)),
                    )
                except Exception as exc:
                    err_msg = f"{type(exc).__name__}: {exc}"
                    logger.error(
                        "[%d/%d] Dask task failed for %s batch %d: %s",
                        n_done,
                        n_tasks,
                        spec["beam_name"],
                        spec["batch_index"],
                        exc,
                        exc_info=True,
                    )
                    batch_outputs.append({
                        "beam_name": spec["beam_name"],
                        "batch_index": spec["batch_index"],
                        "results": [
                            {
                                "srcname": c["srcname"],
                                "out_file": str(c["out_file"]),
                                "status": "FAILED",
                                "message": f"DaskTaskError: {err_msg}",
                            }
                            for c in spec["candidates"]
                        ],
                        "elapsed_sec": 0.0,
                    })
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception as exc:
                    logger.warning("Error closing Dask client: %s", exc)
            if cluster is not None:
                try:
                    cluster.close()
                except Exception as exc:
                    logger.warning("Error closing Dask cluster: %s", exc)

    all_results: List[Dict[str, Any]] = []
    counts = {"OK": 0, "EXISTS": 0, "DRY_RUN": 0, "SKIPPED": 0, "FAILED": 0}
    for out in batch_outputs:
        for item in out["results"]:
            all_results.append(item)
            st = str(item.get("status", "FAILED"))
            counts[st] = counts.get(st, 0) + 1
            logger.info("[%s] %s: %s", st, item.get("out_file", ""), item.get("message", ""))

    summary = {
        "completed": len(all_results),
        "ok": counts["OK"],
        "exists": counts["EXISTS"],
        "dry_run": counts["DRY_RUN"],
        "skipped": counts["SKIPPED"],
        "failed": counts["FAILED"],
        "results": all_results,
    }
    logger.info(
        "Extraction complete: completed=%d ok=%d exists=%d skipped=%d failed=%d",
        summary["completed"],
        summary["ok"],
        summary["exists"],
        summary["skipped"],
        summary["failed"],
    )
    return summary
