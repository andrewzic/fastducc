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

import logging
import math
import os
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
else:
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

        # Determine data column
        data_col = datacolumn.upper()
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
