# -*- coding: utf-8 -*-
"""Unified monolithic HDF5 candidate container and API for fastducc.

Supports all fastducc transient and periodicity search methods:
- ``boxcar``   : 2D detection cutouts, 1D lightcurves (with local RMS), optional dynamic spectra & snippets
- ``variance`` : 2D std-map cutouts, 1D lightcurves, optional full std-map & dynamic spectra
- ``ffa``      : 2D on/off-pulse cutouts, variable-length 1D folded profiles
- ``nufft``    : 2D on/off-resonance cutouts, 1D spin-frequency power spectra, optional summary maps

File naming conventions
-----------------------
Per-(method, beam, DM) trial container:
    ``<prefix>[_scan_<scan_id>]_<method>[_<beam>]_dm<dm:06.2f>_candidates.h5``

Observation-level consolidated exemplar container (one per method across all beams & DMs):
    ``<field>.<sbid>_obs_<method>_consolidated_candidates.h5``

Schema (``schema_version = "2.0"``)
-----------------------------------
Root attributes:
    schema_version, method, ms_base, sbid, field, beam, dm, is_consolidated,
    cutout_size, npix_x, npix_y, n_candidates, n_freq

Datasets:
    /metadata           : compound structured array (N,)
    /wcs_headers        : variable-length UTF-8 FITS WCS header strings (M_wcs,)
    /extra_json         : variable-length UTF-8 JSON provenance strings (N,)
    /cutouts            : float32 (N, Y, X)                 primary / on-pulse / on-resonance cutout
    /cutouts_off        : float32 (N, Y, X) [optional]      off-pulse / off-resonance cutout
    /freqs_hz           : float64 (N_freq,) [optional]      shared spin-frequency axis [Hz]
    /spectra            : float32 (N, N_freq) [optional]    uniform-length 1D power spectra
    /spectra_ragged/    : CSR variable-length spectra (offsets, values, freqs) when merged across heterogeneous grids
    /lightcurves/       : CSR variable-length 1D lightcurves (offsets, time, flux, flux_err)
    /profiles/          : CSR variable-length 1D folded profiles (offsets, values)
    /fullframes/<idx>   : float32 (Ny, Nx) [sparse, optional]
    /dynspecs/<idx>/    : data, time, freq, time_raw [sparse, optional]
    /snippets/<idx>/    : cube, times [sparse, optional]
    /maps/              : optional 2D summary maps (max_snr_map, best_freq_map, std_map)

Backwards-compatible reading of ``schema_version = "1.0"`` NUFFT containers is supported transparently.
"""

from __future__ import annotations

import collections
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import glob
import json
import logging
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

logger = logging.getLogger("fastducc.candidate_store")

SCHEMA_VERSION = "2.0"

VALID_METHODS: Tuple[str, ...] = ("boxcar", "variance", "ffa", "nufft")

_METHOD_ALIASES: Dict[str, str] = {
    "boxcar": "boxcar",
    "box": "boxcar",
    "variance": "variance",
    "var": "variance",
    "ffa": "ffa",
    "uv_ffa": "ffa",
    "uv-ffa": "ffa",
    "uvffa": "ffa",
    "nufft": "nufft",
    "periodicity": "nufft",
}

_UTF8_VLEN = h5py.string_dtype(encoding="utf-8")

CANDIDATE_META_DTYPE = np.dtype(
    [
        # Identification & provenance
        ("cand_idx", np.int32),
        ("srcname", "S48"),
        ("method", "S16"),
        ("beam", "S16"),
        ("sbid", "S24"),
        ("field", "S48"),
        ("scan_id", "S32"),
        ("wcs_idx", np.int32),
        # Sky & pixel coordinates
        ("x", np.int32),
        ("y", np.int32),
        ("ra_deg", np.float64),
        ("dec_deg", np.float64),
        ("ra_hms", "S24"),
        ("dec_dms", "S24"),
        # Core detection metrics
        ("snr", np.float32),
        ("dm", np.float32),
        ("dm_idx", np.int32),
        # Transient search fields (boxcar / variance)
        ("width_samples", np.int32),
        ("time_center", np.float64),
        ("time_start", np.float64),
        ("time_end", np.float64),
        ("t0_idx", np.int32),
        ("t1_idx_excl", np.int32),
        ("center_idx", np.int32),
        ("var_highpass_cutoff_sec", np.float32),
        # Periodicity search fields (FFA / NUFFT)
        ("period_s", np.float64),
        ("f_hz", np.float64),
        ("f_off_res", np.float64),
        ("nharm_used", np.int16),
        ("k_idx", np.int32),
        ("width_bins", np.int32),
        ("duty_cycle", np.float32),
        ("best_b", np.int32),
        ("plan_bins", np.int32),
        ("spec_isolation_ratio", np.float32),
        ("spec_local_contrast", np.float32),
    ]
)

_META_FIELD_NAMES = frozenset(CANDIDATE_META_DTYPE.names or ())


# ---------------------------------------------------------------------------
# Normalisation, naming, and atomic I/O helpers
# ---------------------------------------------------------------------------


def normalise_method(method: Optional[str], *, default: str = "nufft") -> str:
    """Normalise a search method label to one of ``boxcar``, ``variance``, ``ffa``, ``nufft``."""
    if not method:
        return default
    key = str(method).strip().lower()
    if key in _METHOD_ALIASES:
        return _METHOD_ALIASES[key]
    for alias, canon in _METHOD_ALIASES.items():
        if alias in key:
            return canon
    raise ValueError(f"Unsupported search method '{method}'. Expected one of {VALID_METHODS}.")


def normalise_beam(beam: Any) -> str:
    """Normalise a beam identifier to lowercase ``beamNN`` (or empty string if absent)."""
    if beam is None:
        return ""
    s = str(beam).strip()
    if not s or s.lower() in ("none", "nan", "--"):
        return ""
    m = re.search(r"beam(\d+)", s, re.IGNORECASE)
    if m:
        return f"beam{int(m.group(1)):02d}"
    if s.isdigit():
        return f"beam{int(s):02d}"
    return s.lower()


def _is_tmp_name(path: Union[str, os.PathLike]) -> bool:
    name = os.path.basename(str(path))
    return name.startswith(".") or ".tmp." in name or name.endswith(".tmp")


def _make_temp_path(dest_path: Union[str, os.PathLike], suffix: str = ".h5") -> Path:
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dest.stem}.tmp.", suffix=suffix, dir=str(dest.parent))
    os.close(fd)
    return Path(tmp_name)


def _encode_fixed_str(val: Any, max_len: int) -> bytes:
    if val is None:
        return b""
    s = str(val).strip()
    if s.lower() in ("none", "nan", "--"):
        return b""
    return s.encode("utf-8", errors="replace")[:max_len]


def _decode_fixed_str(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode("utf-8", errors="replace").rstrip("\x00").strip()
    s = str(val).strip()
    return "" if s.lower() in ("none", "nan", "--") else s


def _safe_float(val: Any, default: float = float("nan")) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val: Any, default: int = -1) -> int:
    if val is None:
        return default
    try:
        f = float(val)
        if not np.isfinite(f):
            return default
        return int(f)
    except (TypeError, ValueError):
        return default


def _crop_or_pad_2d(img: np.ndarray, target_size: int) -> np.ndarray:
    """Centre-crop or NaN-pad a 2D array to ``(target_size, target_size)`` float32."""
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D cutout array, got shape {arr.shape}")
    ny, nx = arr.shape
    if ny == target_size and nx == target_size:
        return np.ascontiguousarray(arr, dtype=np.float32)
    out = np.full((target_size, target_size), np.nan, dtype=np.float32)
    sy = min(ny, target_size)
    sx = min(nx, target_size)
    y0_src = (ny - sy) // 2
    x0_src = (nx - sx) // 2
    y0_dst = (target_size - sy) // 2
    x0_dst = (target_size - sx) // 2
    out[y0_dst : y0_dst + sy, x0_dst : x0_dst + sx] = arr[y0_src : y0_src + sy, x0_src : x0_src + sx]
    return out


def _build_cutout_wcs(
    w2: Optional[WCS],
    x0: int,
    y0: int,
    cutout_size: int,
) -> Optional[WCS]:
    """Shift a full-frame 2D celestial WCS to a ``cutout_size x cutout_size`` stamp centred at ``(x0, y0)``."""
    if w2 is None:
        return None
    half_c = int(cutout_size) // 2
    w_cut = w2.deepcopy()
    w_cut.wcs.crpix[0] = float(w2.wcs.crpix[0]) - float(x0 - half_c)
    w_cut.wcs.crpix[1] = float(w2.wcs.crpix[1]) - float(y0 - half_c)
    return w_cut


def _wcs_to_header_str(wcs_obj: Optional[Union[WCS, str, fits.Header]]) -> str:
    if wcs_obj is None:
        return ""
    if isinstance(wcs_obj, str):
        return wcs_obj.strip()
    if isinstance(wcs_obj, fits.Header):
        return wcs_obj.tostring()
    if isinstance(wcs_obj, WCS):
        return wcs_obj.celestial.to_header().tostring()
    raise TypeError(f"Unsupported WCS type: {type(wcs_obj)!r}")


def _header_str_to_wcs(hdr_str: str) -> Optional[WCS]:
    if not hdr_str or not hdr_str.strip():
        return None
    hdr = fits.Header.fromstring(hdr_str)
    return WCS(hdr).celestial


def build_trial_h5_path(
    out_dir: Union[str, os.PathLike],
    *,
    method: str,
    beam: str = "",
    dm: float = 0.0,
    prefix: str = "",
    scan_id: Optional[str] = None,
) -> Path:
    """Build canonical path for a per-(method, beam, DM) candidate HDF5 container.

    Examples
    --------
    >>> build_trial_h5_path("/data/SB82418/candidates", method="nufft", beam="beam00", dm=25.0, prefix="cracoData.FIELD.SB82418.beam00.uvsub")
    PosixPath('/data/SB82418/candidates/cracoData.FIELD.SB82418.beam00.uvsub_nufft_dm025.00_candidates.h5')
    >>> build_trial_h5_path("/data/SB82418/candidates", method="boxcar", beam="beam03", dm=10.5, prefix="FIELD.SB82418")
    PosixPath('/data/SB82418/candidates/FIELD.SB82418_boxcar_beam03_dm010.50_candidates.h5')
    """
    m = normalise_method(method)
    b = normalise_beam(beam)
    dm_tag = f"dm{float(dm):06.2f}"

    parts: List[str] = []
    clean_prefix = str(prefix).strip().rstrip("_")
    # Strip trailing dm tag from prefix if caller passed a dm-suffixed root
    clean_prefix = re.sub(r"_dm\d+(?:\.\d+)?$", "", clean_prefix)
    if clean_prefix:
        parts.append(clean_prefix)

    if scan_id:
        s_str = str(scan_id).strip()
        if s_str and f"scan_{s_str}" not in clean_prefix:
            parts.append(f"scan_{s_str}")

    parts.append(m)

    if b and not re.search(rf"(?:^|[._-]){re.escape(b)}(?:$|[._-])", clean_prefix, re.IGNORECASE):
        parts.append(b)

    parts.append(dm_tag)
    parts.append("candidates")
    fname = "_".join(parts) + ".h5"
    return Path(out_dir) / fname


def build_consolidated_h5_path(
    out_dir: Union[str, os.PathLike],
    *,
    method: str,
    prefix: str = "",
    sbid: str = "",
    field: str = "",
) -> Path:
    """Build canonical path for a method-level consolidated exemplar HDF5 container."""
    m = normalise_method(method)
    clean_prefix = str(prefix).strip().rstrip("_")
    if not clean_prefix:
        if field and sbid:
            clean_prefix = f"{field}.{sbid}"
        elif sbid:
            clean_prefix = sbid
        elif field:
            clean_prefix = field
        else:
            clean_prefix = "obs"
    return Path(out_dir) / f"{clean_prefix}_obs_{m}_consolidated_candidates.h5"


def parse_candidate_h5_filename(path: Union[str, os.PathLike]) -> Dict[str, Any]:
    """Extract method, beam, DM, SBID, scan_id, and consolidation flag from a candidate HDF5 filename."""
    base = os.path.basename(str(path))
    info: Dict[str, Any] = {
        "method": "",
        "beam": "",
        "dm": None,
        "sbid": "",
        "scan_id": "",
        "is_consolidated": "_consolidated_candidates.h5" in base,
    }
    m_sbid = re.search(r"(SB\d+)", base)
    if m_sbid:
        info["sbid"] = m_sbid.group(1)

    m_beam = re.search(r"beam(\d+)", base, re.IGNORECASE)
    if m_beam:
        info["beam"] = f"beam{int(m_beam.group(1)):02d}"

    m_dm = re.search(r"_dm(\d+(?:\.\d+)?)", base)
    if m_dm:
        info["dm"] = float(m_dm.group(1))

    m_scan = re.search(r"_scan_([^_]+)", base)
    if m_scan:
        info["scan_id"] = m_scan.group(1)

    for tok in ("boxcar", "variance", "uv_ffa", "ffa", "nufft"):
        if re.search(rf"(?:^|_){tok}(?:_|\.h5$)", base, re.IGNORECASE):
            info["method"] = normalise_method(tok)
            break

    return info


def discover_trial_h5_files(
    search_dirs: Sequence[Union[str, os.PathLike]],
    *,
    method: Optional[str] = None,
    beam: Optional[str] = None,
    include_consolidated: bool = False,
) -> List[Path]:
    """Discover candidate HDF5 files across *search_dirs*, optionally filtered by method and beam."""
    req_method = normalise_method(method) if method else None
    req_beam = normalise_beam(beam) if beam else None

    found: List[Path] = []
    seen: set[Path] = set()
    for d in search_dirs:
        if not d:
            continue
        dp = Path(d)
        if not dp.is_dir():
            continue
        for fp in sorted(dp.glob("*_candidates.h5")):
            if _is_tmp_name(fp):
                continue
            rp = fp.resolve()
            if rp in seen:
                continue
            info = parse_candidate_h5_filename(fp.name)
            if info["is_consolidated"] and not include_consolidated:
                continue
            if req_method:
                file_m = info["method"]
                if file_m and file_m != req_method:
                    continue
                if not file_m:
                    # Legacy schema 1.0 files without method token in filename are NUFFT
                    if req_method != "nufft":
                        continue
            if req_beam and info["beam"] and info["beam"] != req_beam:
                continue
            seen.add(rp)
            found.append(rp)
    return found


# ---------------------------------------------------------------------------
# Typed artefact containers returned by CandidateRecord properties
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Lightcurve:
    """1D candidate lightcurve with timestamps and per-sample local RMS uncertainties."""

    time: np.ndarray        # float64 (T,)
    flux: np.ndarray        # float32 (T,)
    flux_err: np.ndarray    # float32 (T,)
    width_samples: int = 1

    def __len__(self) -> int:
        return int(self.time.size)

    def smoothed(
        self,
        width: Optional[int] = None,
        *,
        center_policy: str = "right",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return boxcar-smoothed ``(time_sm, flux_sm, err_sm)`` for *width* (defaults to ``self.width_samples``)."""
        w = max(1, int(width if width is not None else self.width_samples))
        t = np.asarray(self.time, dtype=np.float64)
        f = np.asarray(self.flux, dtype=np.float64)
        e = np.asarray(self.flux_err, dtype=np.float64)
        n = t.size
        if n == 0 or w > n:
            return (
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
            )
        if w == 1:
            return t.copy(), f.astype(np.float32), e.astype(np.float32)

        from fastducc import kernels

        lc_sm, n_eff = kernels._compute_boxcar_1d(f, w)
        offset = (w // 2) if center_policy == "right" else ((w - 1) // 2)
        times_sm = t[offset : offset + n_eff]
        csum_var = np.zeros(n + 1, dtype=np.float64)
        csum_var[1:] = np.cumsum(np.where(np.isfinite(e), e * e, 0.0))
        err_sm = np.sqrt(np.maximum(0.0, csum_var[w:] - csum_var[:-w])) / float(w)
        return times_sm, lc_sm.astype(np.float32), err_sm.astype(np.float32)


@dataclass(frozen=True)
class Spectrum:
    """1D spin-frequency power spectrum for a periodicity (NUFFT) candidate."""

    freqs_hz: np.ndarray    # float64 (N_freq,)
    power: np.ndarray       # float32 (N_freq,)
    f_hz: float = float("nan")
    f_off_res: float = float("nan")
    spec_isolation_ratio: float = float("nan")
    spec_local_contrast: float = float("nan")

    def __len__(self) -> int:
        return int(self.power.size)


@dataclass(frozen=True)
class DynamicSpectrum:
    """Phased visibility dynamic spectrum (time x frequency)."""

    data: np.ndarray                  # float32 (Nt, Nf)
    time: np.ndarray                  # float64 (Nt,)
    freq: np.ndarray                  # float64 (Nf,) [MHz]
    time_raw: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "data": self.data,
            "time": self.time,
            "freq": self.freq,
        }
        if self.time_raw is not None:
            d["time_raw"] = self.time_raw
        return d


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def _append_csr_1d(
    grp: h5py.Group,
    arrays: Dict[str, np.ndarray],
) -> None:
    """Append one variable-length 1D record to a CSR group with ``offsets`` dataset."""
    ds_off = grp["offsets"]
    prev_end = int(ds_off[-1])
    first_arr = next(iter(arrays.values()))
    n_el = int(first_arr.size)
    new_end = prev_end + n_el

    if n_el > 0:
        for name, arr in arrays.items():
            ds = grp[name]
            ds.resize((new_end,))
            ds[prev_end:new_end] = arr

    n_off = int(ds_off.shape[0])
    ds_off.resize((n_off + 1,))
    ds_off[n_off] = new_end


class CandidateHDF5Writer:
    """Incrementally write candidates and their artefacts into a monolithic HDF5 container.

    Writes to a hidden temporary file in the destination directory and atomically
    commits via ``os.replace`` on :meth:`finalize`.
    """

    def __init__(
        self,
        dest_path: Union[str, os.PathLike],
        *,
        method: str = "nufft",
        freqs_hz: Optional[np.ndarray] = None,
        cutout_size: Optional[int] = None,
        npix_x: int = 384,
        npix_y: int = 384,
        wcs_full: Optional[Union[WCS, str, fits.Header]] = None,
        ms_base: str = "",
        sbid: str = "",
        field: str = "",
        beam: str = "",
        dm: Optional[float] = None,
        has_off_cutout: Optional[bool] = None,
        is_consolidated: bool = False,
        compression: Optional[str] = "lzf",
        compression_opts: Optional[int] = None,
    ) -> None:
        self.dest_path = Path(dest_path)
        self.dest_path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp_path = _make_temp_path(self.dest_path, suffix=".h5")

        self.method = normalise_method(method)
        default_cut = 50 if self.method in ("boxcar", "variance") else 32
        self.cutout_size = int(cutout_size) if cutout_size is not None else default_cut
        self.npix_x = int(npix_x)
        self.npix_y = int(npix_y)
        self.ms_base = str(ms_base)
        self.sbid = str(sbid)
        self.field = str(field)
        self.beam = normalise_beam(beam)
        self.dm = float(dm) if dm is not None and np.isfinite(float(dm)) else float("nan")
        self.is_consolidated = bool(is_consolidated)
        self.compression = compression
        self.compression_opts = compression_opts

        if freqs_hz is not None and np.asarray(freqs_hz).size > 0:
            self.freqs_hz: Optional[np.ndarray] = np.ascontiguousarray(freqs_hz, dtype=np.float64).ravel()
            self.n_freq = int(self.freqs_hz.size)
        else:
            self.freqs_hz = None
            self.n_freq = 0

        if has_off_cutout is None:
            self.has_off_cutout = self.method in ("nufft", "ffa")
        else:
            self.has_off_cutout = bool(has_off_cutout)

        self._wcs_map: Dict[str, int] = {}
        self._count = 0
        self._closed = False
        self._finalized = False

        self._h5 = h5py.File(str(self._tmp_path), "w")
        self._init_datasets()

        self._default_wcs_idx = -1
        if wcs_full is not None:
            self._default_wcs_idx = self.register_wcs(wcs_full)

    def _comp_kwargs(self) -> Dict[str, Any]:
        if not self.compression:
            return {}
        kw: Dict[str, Any] = {"compression": self.compression}
        if self.compression == "gzip" and self.compression_opts is not None:
            kw["compression_opts"] = int(self.compression_opts)
        return kw

    def _init_datasets(self) -> None:
        f = self._h5
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["method"] = self.method
        f.attrs["ms_base"] = self.ms_base
        f.attrs["sbid"] = self.sbid
        f.attrs["field"] = self.field
        f.attrs["beam"] = self.beam
        if np.isfinite(self.dm):
            f.attrs["dm"] = float(self.dm)
        f.attrs["is_consolidated"] = bool(self.is_consolidated)
        f.attrs["cutout_size"] = self.cutout_size
        f.attrs["npix_x"] = self.npix_x
        f.attrs["npix_y"] = self.npix_y
        f.attrs["n_freq"] = self.n_freq
        f.attrs["n_candidates"] = 0

        ckw = self._comp_kwargs()

        f.create_dataset(
            "metadata",
            shape=(0,),
            maxshape=(None,),
            chunks=(64,),
            dtype=CANDIDATE_META_DTYPE,
        )
        f.create_dataset(
            "wcs_headers",
            shape=(0,),
            maxshape=(None,),
            chunks=(16,),
            dtype=_UTF8_VLEN,
        )
        f.create_dataset(
            "extra_json",
            shape=(0,),
            maxshape=(None,),
            chunks=(64,),
            dtype=_UTF8_VLEN,
        )
        f.create_dataset(
            "cutouts",
            shape=(0, self.cutout_size, self.cutout_size),
            maxshape=(None, self.cutout_size, self.cutout_size),
            chunks=(16, self.cutout_size, self.cutout_size),
            dtype=np.float32,
            **ckw,
        )
        if self.has_off_cutout:
            f.create_dataset(
                "cutouts_off",
                shape=(0, self.cutout_size, self.cutout_size),
                maxshape=(None, self.cutout_size, self.cutout_size),
                chunks=(16, self.cutout_size, self.cutout_size),
                dtype=np.float32,
                **ckw,
            )

        if self.freqs_hz is not None and self.n_freq > 0:
            f.create_dataset("freqs_hz", data=self.freqs_hz, dtype=np.float64)
            spec_chunk_f = max(1, min(self.n_freq, 4096))
            f.create_dataset(
                "spectra",
                shape=(0, self.n_freq),
                maxshape=(None, self.n_freq),
                chunks=(16, spec_chunk_f),
                dtype=np.float32,
                **ckw,
            )

    def _ensure_csr_group(self, group_name: str, fields: Dict[str, np.dtype]) -> h5py.Group:
        if group_name in self._h5:
            return self._h5[group_name]
        grp = self._h5.create_group(group_name)
        # Back-fill zero-length offsets for any candidates already written
        init_offsets = np.zeros(self._count + 1, dtype=np.int64)
        grp.create_dataset("offsets", data=init_offsets, maxshape=(None,), chunks=(64,), dtype=np.int64)
        ckw = self._comp_kwargs()
        for fname, fdtype in fields.items():
            grp.create_dataset(fname, shape=(0,), maxshape=(None,), chunks=(4096,), dtype=fdtype, **ckw)
        return grp

    def _ensure_off_cutout_dataset(self) -> h5py.Dataset:
        if "cutouts_off" in self._h5:
            return self._h5["cutouts_off"]
        ckw = self._comp_kwargs()
        ds = self._h5.create_dataset(
            "cutouts_off",
            shape=(self._count, self.cutout_size, self.cutout_size),
            maxshape=(None, self.cutout_size, self.cutout_size),
            chunks=(16, self.cutout_size, self.cutout_size),
            dtype=np.float32,
            fillvalue=np.nan,
            **ckw,
        )
        if self._count > 0:
            ds[:] = np.nan
        self.has_off_cutout = True
        return ds

    def register_wcs(self, wcs_obj: Optional[Union[WCS, str, fits.Header]]) -> int:
        """Register a full-frame WCS header string and return its integer index."""
        hdr_str = _wcs_to_header_str(wcs_obj)
        if not hdr_str:
            return -1
        if hdr_str in self._wcs_map:
            return self._wcs_map[hdr_str]
        ds = self._h5["wcs_headers"]
        idx = int(ds.shape[0])
        ds.resize((idx + 1,))
        ds[idx] = hdr_str
        self._wcs_map[hdr_str] = idx
        if idx == 0 and "wcs_header" not in self._h5.attrs:
            self._h5.attrs["wcs_header"] = hdr_str
        return idx

    @property
    def final_path(self) -> str:
        return str(self.dest_path)

    @property
    def count(self) -> int:
        return self._count

    def _pack_metadata_row(
        self,
        c: Dict[str, Any],
        arr: Dict[str, Any],
        default_cand_idx: int,
        wcs_idx: int,
    ) -> Tuple[np.void, str]:
        row = np.zeros(1, dtype=CANDIDATE_META_DTYPE)[0]
        row["cand_idx"] = _safe_int(c.get("cand_idx"), default_cand_idx)
        row["srcname"] = _encode_fixed_str(c.get("srcname"), 48)
        row["method"] = _encode_fixed_str(normalise_method(c.get("method", c.get("algo", self.method)), default=self.method), 16)
        row["beam"] = _encode_fixed_str(normalise_beam(c.get("beam", c.get("max_snr_beam", self.beam))), 16)
        row["sbid"] = _encode_fixed_str(c.get("sbid", self.sbid), 24)
        row["field"] = _encode_fixed_str(c.get("field", c.get("phase_center_field", self.field)), 48)
        row["scan_id"] = _encode_fixed_str(c.get("scan_id", c.get("scan", "")), 32)
        row["wcs_idx"] = np.int32(wcs_idx)

        row["x"] = _safe_int(c.get("x"), -1)
        row["y"] = _safe_int(c.get("y"), -1)
        row["ra_deg"] = _safe_float(c.get("ra_deg"), float("nan"))
        row["dec_deg"] = _safe_float(c.get("dec_deg"), float("nan"))
        row["ra_hms"] = _encode_fixed_str(c.get("ra_hms"), 24)
        row["dec_dms"] = _encode_fixed_str(c.get("dec_dms"), 24)

        row["snr"] = np.float32(_safe_float(c.get("snr", c.get("max_snr")), 0.0))
        default_dm = self.dm if np.isfinite(self.dm) else 0.0
        row["dm"] = np.float32(_safe_float(c.get("dm", c.get("best_dm", c.get("trial_dm", default_dm))), default_dm))
        row["dm_idx"] = _safe_int(c.get("dm_idx"), -1)

        row["width_samples"] = _safe_int(c.get("width_samples", c.get("max_snr_width")), 1)
        row["time_center"] = _safe_float(c.get("time_center", c.get("max_snr_time_center")), float("nan"))
        row["time_start"] = _safe_float(c.get("time_start"), float("nan"))
        row["time_end"] = _safe_float(c.get("time_end"), float("nan"))
        row["t0_idx"] = _safe_int(c.get("t0_idx"), -1)
        row["t1_idx_excl"] = _safe_int(c.get("t1_idx_excl"), -1)
        row["center_idx"] = _safe_int(c.get("center_idx"), -1)
        row["var_highpass_cutoff_sec"] = np.float32(_safe_float(c.get("var_highpass_cutoff_sec"), 0.0))

        row["period_s"] = _safe_float(c.get("period_s", c.get("period")), float("nan"))
        row["f_hz"] = _safe_float(c.get("f_hz"), float("nan"))
        row["f_off_res"] = _safe_float(arr.get("f_off_res", c.get("f_off_res")), float("nan"))
        row["nharm_used"] = np.int16(_safe_int(c.get("nharm_used"), 1))
        row["k_idx"] = _safe_int(c.get("k_idx"), -1)
        row["width_bins"] = _safe_int(c.get("width_bins", c.get("width")), 1)
        row["duty_cycle"] = np.float32(_safe_float(c.get("duty_cycle"), float("nan")))
        row["best_b"] = _safe_int(c.get("best_b"), -1)
        row["plan_bins"] = _safe_int(c.get("plan_bins"), -1)
        row["spec_isolation_ratio"] = np.float32(
            _safe_float(c.get("spec_isolation_ratio", arr.get("spec_isolation_ratio")), float("nan"))
        )
        row["spec_local_contrast"] = np.float32(
            _safe_float(c.get("spec_local_contrast", arr.get("spec_local_contrast")), float("nan"))
        )

        # Preserve extra scalar/string provenance fields (e.g. beams_all, dm_all, psrcat_match_name)
        extra_payload: Dict[str, Any] = {}
        _skip_keys = _META_FIELD_NAMES | {
            "cut_on", "cut_off", "cutout", "cutout_on", "cutout_off", "cutout_on_res", "cutout_off_res",
            "spec_1d", "spectrum_1d", "spectrum_freqs", "profile", "fullframe", "harm_img",
            "dynspec", "lightcurve", "snippet", "wcs_full", "wcs_cutout",
            "image_yx", "snr_image_yx", "algo", "trial_dm", "period", "width",
        }
        for k, v in c.items():
            if k in _skip_keys or v is None:
                continue
            if isinstance(v, (np.ndarray, WCS, fits.Header)):
                continue
            if isinstance(v, (np.generic,)):
                v = v.item()
            if isinstance(v, float) and not np.isfinite(v):
                continue
            if isinstance(v, (str, int, float, bool)):
                extra_payload[k] = v

        extra_str = json.dumps(extra_payload, separators=(",", ":")) if extra_payload else ""
        return row, extra_str

    def append_candidate(
        self,
        meta: Dict[str, Any],
        *,
        cutout: Optional[np.ndarray] = None,
        cutout_off: Optional[np.ndarray] = None,
        fullframe: Optional[np.ndarray] = None,
        lightcurve: Optional[Union[Lightcurve, Dict[str, Any]]] = None,
        profile: Optional[np.ndarray] = None,
        spectrum: Optional[Union[Spectrum, np.ndarray]] = None,
        freqs_hz: Optional[np.ndarray] = None,
        dynspec: Optional[Union[DynamicSpectrum, Dict[str, Any]]] = None,
        snippet: Optional[Dict[str, Any]] = None,
        wcs_full: Optional[Union[WCS, str, fits.Header]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Append a single candidate and its artefacts, returning its 0-based container index."""
        c_dict = dict(meta)
        if extra:
            c_dict.update(extra)
        arr_dict: Dict[str, Any] = {}
        if cutout is not None:
            arr_dict["cut_on"] = cutout
        if cutout_off is not None:
            arr_dict["cut_off"] = cutout_off
        if fullframe is not None:
            arr_dict["fullframe"] = fullframe
        if lightcurve is not None:
            arr_dict["lightcurve"] = lightcurve
        if profile is not None:
            arr_dict["profile"] = profile
        if spectrum is not None:
            if isinstance(spectrum, Spectrum):
                arr_dict["spec_1d"] = spectrum.power
                arr_dict["freqs_hz"] = spectrum.freqs_hz
                if np.isfinite(spectrum.f_off_res):
                    arr_dict["f_off_res"] = spectrum.f_off_res
            else:
                arr_dict["spec_1d"] = spectrum
        if freqs_hz is not None:
            arr_dict["freqs_hz"] = freqs_hz
        if dynspec is not None:
            arr_dict["dynspec"] = dynspec
        if snippet is not None:
            arr_dict["snippet"] = snippet

        idxs = self.append_batch([c_dict], [arr_dict], wcs_full=wcs_full)
        return idxs[0]

    def append_batch(
        self,
        candidates: Sequence[Dict[str, Any]],
        cand_arrays: Sequence[Dict[str, Any]],
        *,
        wcs_full: Optional[Union[WCS, str, fits.Header]] = None,
    ) -> List[int]:
        """Append a batch of candidates and their extracted artefact arrays.

        Returns the list of 0-based indices assigned inside the container.
        """
        if self._closed:
            raise RuntimeError("Cannot append to a closed CandidateHDF5Writer.")
        n = min(len(candidates), len(cand_arrays))
        if n <= 0:
            return []

        batch_wcs_idx = self.register_wcs(wcs_full) if wcs_full is not None else self._default_wcs_idx
        start = self._count
        end = start + n

        meta_buf = np.zeros(n, dtype=CANDIDATE_META_DTYPE)
        extra_buf: List[str] = []
        cut_buf = np.full((n, self.cutout_size, self.cutout_size), np.nan, dtype=np.float32)

        need_off = self.has_off_cutout or any(
            (arr.get("cut_off") is not None or arr.get("cutout_off") is not None)
            for arr in cand_arrays[:n]
        )
        cut_off_buf = (
            np.full((n, self.cutout_size, self.cutout_size), np.nan, dtype=np.float32)
            if need_off
            else None
        )

        # Check if spectra match fixed /spectra or need ragged storage
        has_fixed_spec = "spectra" in self._h5 and self.n_freq > 0
        spec_buf = np.zeros((n, self.n_freq), dtype=np.float32) if has_fixed_spec else None

        for i in range(n):
            c = candidates[i]
            arr = cand_arrays[i]
            cand_wcs = arr.get("wcs_full")
            w_idx = self.register_wcs(cand_wcs) if cand_wcs is not None else batch_wcs_idx

            row, extra_str = self._pack_metadata_row(c, arr, default_cand_idx=start + i + 1, wcs_idx=w_idx)
            meta_buf[i] = row
            extra_buf.append(extra_str)

            c_on = arr.get("cut_on")
            if c_on is None:
                c_on = arr.get("cutout")
            if c_on is not None and np.asarray(c_on).size > 0:
                cut_buf[i] = _crop_or_pad_2d(c_on, self.cutout_size)

            if cut_off_buf is not None:
                c_off = arr.get("cut_off")
                if c_off is None:
                    c_off = arr.get("cutout_off")
                if c_off is not None and np.asarray(c_off).size > 0:
                    cut_off_buf[i] = _crop_or_pad_2d(c_off, self.cutout_size)

            if spec_buf is not None:
                sp = arr.get("spec_1d")
                if sp is not None and np.asarray(sp).size == self.n_freq:
                    spec_buf[i] = np.asarray(sp, dtype=np.float32).ravel()

        ds_meta = self._h5["metadata"]
        ds_extra = self._h5["extra_json"]
        ds_cut = self._h5["cutouts"]

        ds_meta.resize((end,))
        ds_extra.resize((end,))
        ds_cut.resize((end, self.cutout_size, self.cutout_size))

        ds_meta[start:end] = meta_buf
        ds_extra[start:end] = extra_buf
        ds_cut[start:end] = cut_buf

        if cut_off_buf is not None:
            ds_off = self._ensure_off_cutout_dataset()
            ds_off.resize((end, self.cutout_size, self.cutout_size))
            ds_off[start:end] = cut_off_buf

        if spec_buf is not None:
            ds_spec = self._h5["spectra"]
            ds_spec.resize((end, self.n_freq))
            ds_spec[start:end] = spec_buf

        # Variable-length CSR datasets (lightcurves, profiles, ragged spectra) and sparse groups
        any_lc = "lightcurves" in self._h5 or any(arr.get("lightcurve") is not None for arr in cand_arrays[:n])
        any_prof = "profiles" in self._h5 or any(arr.get("profile") is not None for arr in cand_arrays[:n])
        any_ragged_spec = "spectra_ragged" in self._h5 or any(
            arr.get("spec_1d") is not None and (not has_fixed_spec or np.asarray(arr["spec_1d"]).size != self.n_freq)
            for arr in cand_arrays[:n]
        )

        if any_lc:
            lc_grp = self._ensure_csr_group(
                "lightcurves",
                {"time": np.float64, "flux": np.float32, "flux_err": np.float32},
            )
            for i in range(n):
                lc = cand_arrays[i].get("lightcurve")
                if lc is None:
                    _append_csr_1d(
                        lc_grp,
                        {
                            "time": np.empty(0, dtype=np.float64),
                            "flux": np.empty(0, dtype=np.float32),
                            "flux_err": np.empty(0, dtype=np.float32),
                        },
                    )
                else:
                    if isinstance(lc, Lightcurve):
                        t_arr, f_arr, e_arr = lc.time, lc.flux, lc.flux_err
                    else:
                        t_arr = np.asarray(lc["time"], dtype=np.float64).ravel()
                        f_arr = np.asarray(lc["flux"], dtype=np.float32).ravel()
                        e_raw = lc.get("flux_err", lc.get("error"))
                        e_arr = (
                            np.asarray(e_raw, dtype=np.float32).ravel()
                            if e_raw is not None
                            else np.zeros_like(f_arr, dtype=np.float32)
                        )
                    _append_csr_1d(
                        lc_grp,
                        {
                            "time": np.asarray(t_arr, dtype=np.float64).ravel(),
                            "flux": np.asarray(f_arr, dtype=np.float32).ravel(),
                            "flux_err": np.asarray(e_arr, dtype=np.float32).ravel(),
                        },
                    )

        if any_prof:
            prof_grp = self._ensure_csr_group("profiles", {"values": np.float32})
            for i in range(n):
                p = cand_arrays[i].get("profile")
                p_arr = np.asarray(p, dtype=np.float32).ravel() if p is not None else np.empty(0, dtype=np.float32)
                _append_csr_1d(prof_grp, {"values": p_arr})

        if any_ragged_spec:
            sp_grp = self._ensure_csr_group("spectra_ragged", {"values": np.float32, "freqs": np.float64})
            for i in range(n):
                sp = cand_arrays[i].get("spec_1d")
                fq = cand_arrays[i].get("freqs_hz", self.freqs_hz)
                if sp is not None and fq is not None:
                    sp_a = np.asarray(sp, dtype=np.float32).ravel()
                    fq_a = np.asarray(fq, dtype=np.float64).ravel()
                    m_len = min(sp_a.size, fq_a.size)
                    _append_csr_1d(sp_grp, {"values": sp_a[:m_len], "freqs": fq_a[:m_len]})
                else:
                    _append_csr_1d(
                        sp_grp,
                        {"values": np.empty(0, dtype=np.float32), "freqs": np.empty(0, dtype=np.float64)},
                    )

        # Sparse per-candidate heavy artefacts (fullframe, dynspec, snippet)
        ckw = self._comp_kwargs()
        for i in range(n):
            idx = start + i
            arr = cand_arrays[i]

            ff = arr.get("fullframe")
            if ff is not None and np.asarray(ff).size > 0:
                ff_grp = self._h5.require_group("fullframes")
                ff_grp.create_dataset(str(idx), data=np.asarray(ff, dtype=np.float32), **ckw)

            ds_obj = arr.get("dynspec")
            if ds_obj is not None:
                ds_dict = ds_obj.to_dict() if isinstance(ds_obj, DynamicSpectrum) else ds_obj
                if isinstance(ds_dict, dict) and ds_dict.get("data") is not None:
                    d_grp = self._h5.require_group("dynspecs").create_group(str(idx))
                    d_grp.create_dataset("data", data=np.asarray(ds_dict["data"], dtype=np.float32), **ckw)
                    d_grp.create_dataset("time", data=np.asarray(ds_dict["time"], dtype=np.float64))
                    d_grp.create_dataset("freq", data=np.asarray(ds_dict["freq"], dtype=np.float64))
                    if ds_dict.get("time_raw") is not None:
                        d_grp.create_dataset("time_raw", data=np.asarray(ds_dict["time_raw"], dtype=np.float64))

            snip = arr.get("snippet")
            if isinstance(snip, dict) and snip.get("snippet_cube") is not None:
                s_grp = self._h5.require_group("snippets").create_group(str(idx))
                s_grp.create_dataset("cube", data=np.asarray(snip["snippet_cube"], dtype=np.float32), **ckw)
                if snip.get("snippet_times") is not None:
                    s_grp.create_dataset("times", data=np.asarray(snip["snippet_times"], dtype=np.float64))

        self._count = end
        self._h5.attrs["n_candidates"] = self._count
        return list(range(start, end))

    def finalize(
        self,
        *,
        max_snr_map: Optional[np.ndarray] = None,
        best_freq_map: Optional[np.ndarray] = None,
        std_map: Optional[np.ndarray] = None,
    ) -> Path:
        """Optionally attach 2D summary maps, close, and atomically commit the HDF5 file."""
        if self._finalized:
            return self.dest_path
        if self._closed:
            raise RuntimeError("CandidateHDF5Writer was aborted/closed before finalize().")

        ckw = self._comp_kwargs()
        if any(m is not None for m in (max_snr_map, best_freq_map, std_map)):
            maps_grp = self._h5.require_group("maps")
            if max_snr_map is not None:
                maps_grp.create_dataset("max_snr_map", data=np.asarray(max_snr_map, dtype=np.float32), **ckw)
            if best_freq_map is not None:
                maps_grp.create_dataset("best_freq_map", data=np.asarray(best_freq_map, dtype=np.float32), **ckw)
            if std_map is not None:
                maps_grp.create_dataset("std_map", data=np.asarray(std_map, dtype=np.float32), **ckw)

        self._h5.attrs["n_candidates"] = self._count
        self._h5.flush()
        self._h5.close()
        self._closed = True

        os.replace(str(self._tmp_path), str(self.dest_path))
        self._finalized = True
        logger.info(
            "Committed candidate HDF5 container [%s] (%d candidate(s)) -> %s",
            self.method,
            self._count,
            self.dest_path,
        )
        return self.dest_path

    def abort(self) -> None:
        """Close and remove the uncommitted temporary HDF5 file."""
        if not self._closed:
            try:
                self._h5.close()
            finally:
                self._closed = True
        if not self._finalized and self._tmp_path.exists():
            try:
                self._tmp_path.unlink()
            except OSError:
                pass

    def __enter__(self) -> "CandidateHDF5Writer":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None:
            self.abort()
        elif not self._finalized:
            self.finalize()


# ---------------------------------------------------------------------------
# Lazy CandidateRecord & dictionary-like CandidateStore API
# ---------------------------------------------------------------------------


class CandidateRecord:
    """Lazy-loading handle for a single candidate stored in a :class:`CandidateStore`.

    Scalar metadata is immediately available via attributes (e.g. ``cand.snr``,
    ``cand.dm``, ``cand.ra_deg``, ``cand.method``); heavy array products
    (``cand.cutout``, ``cand.cutout_off``, ``cand.lightcurve``, ``cand.profile``,
    ``cand.spectrum``, ``cand.dynspec``, ``cand.fullframe``) are read on demand
    from HDF5 and cached on the record.
    """

    def __init__(self, store: "CandidateStore", index: int, meta_dict: Dict[str, Any]) -> None:
        self._store = store
        self._index = int(index)
        self._meta = meta_dict
        self._cache: Dict[str, Any] = {}

    @property
    def h5_index(self) -> int:
        return self._index

    @property
    def store_path(self) -> Path:
        return self._store.path

    @property
    def meta(self) -> Dict[str, Any]:
        return self._meta

    # --- Core scalar attributes ---
    @property
    def srcname(self) -> str:
        return str(self._meta.get("srcname", ""))

    @property
    def method(self) -> str:
        return str(self._meta.get("method", self._store.method))

    @property
    def beam(self) -> str:
        return str(self._meta.get("beam", ""))

    @property
    def sbid(self) -> str:
        return str(self._meta.get("sbid", ""))

    @property
    def field(self) -> str:
        return str(self._meta.get("field", ""))

    @property
    def scan_id(self) -> str:
        return str(self._meta.get("scan_id", ""))

    @property
    def cand_idx(self) -> int:
        return int(self._meta.get("cand_idx", self._index + 1))

    @property
    def x(self) -> int:
        return int(self._meta.get("x", -1))

    @property
    def y(self) -> int:
        return int(self._meta.get("y", -1))

    @property
    def ra_deg(self) -> float:
        return float(self._meta.get("ra_deg", float("nan")))

    @property
    def dec_deg(self) -> float:
        return float(self._meta.get("dec_deg", float("nan")))

    @property
    def ra_hms(self) -> str:
        return str(self._meta.get("ra_hms", ""))

    @property
    def dec_dms(self) -> str:
        return str(self._meta.get("dec_dms", ""))

    @property
    def snr(self) -> float:
        return float(self._meta.get("snr", float("nan")))

    @property
    def dm(self) -> float:
        return float(self._meta.get("dm", float("nan")))

    @property
    def dm_idx(self) -> int:
        return int(self._meta.get("dm_idx", -1))

    # --- Transient scalar attributes ---
    @property
    def width_samples(self) -> int:
        return int(self._meta.get("width_samples", 1))

    @property
    def time_center(self) -> float:
        return float(self._meta.get("time_center", float("nan")))

    @property
    def time_start(self) -> float:
        return float(self._meta.get("time_start", float("nan")))

    @property
    def time_end(self) -> float:
        return float(self._meta.get("time_end", float("nan")))

    @property
    def t0_idx(self) -> int:
        return int(self._meta.get("t0_idx", -1))

    @property
    def t1_idx_excl(self) -> int:
        return int(self._meta.get("t1_idx_excl", -1))

    @property
    def center_idx(self) -> int:
        return int(self._meta.get("center_idx", -1))

    # --- Periodicity scalar attributes ---
    @property
    def period_s(self) -> float:
        return float(self._meta.get("period_s", float("nan")))

    @property
    def f_hz(self) -> float:
        return float(self._meta.get("f_hz", float("nan")))

    @property
    def f_off_res(self) -> float:
        return float(self._meta.get("f_off_res", float("nan")))

    @property
    def nharm_used(self) -> int:
        return int(self._meta.get("nharm_used", 1))

    @property
    def width_bins(self) -> int:
        return int(self._meta.get("width_bins", 1))

    @property
    def duty_cycle(self) -> float:
        return float(self._meta.get("duty_cycle", float("nan")))

    @property
    def spec_isolation_ratio(self) -> float:
        return float(self._meta.get("spec_isolation_ratio", float("nan")))

    @property
    def spec_local_contrast(self) -> float:
        return float(self._meta.get("spec_local_contrast", float("nan")))

    # --- Lazy-loaded array & WCS artefacts ---
    @property
    def cutout(self) -> Optional[np.ndarray]:
        """Primary 2D detection cutout (on-pulse / on-resonance / std-map stamp)."""
        if "cutout" not in self._cache:
            self._cache["cutout"] = self._store._read_cutout(self._index)
        return self._cache["cutout"]

    @property
    def cutout_on(self) -> Optional[np.ndarray]:
        """Alias for :attr:`cutout`."""
        return self.cutout

    @property
    def cutout_off(self) -> Optional[np.ndarray]:
        """Off-pulse / off-resonance 2D cutout (if stored)."""
        if "cutout_off" not in self._cache:
            self._cache["cutout_off"] = self._store._read_cutout_off(self._index)
        return self._cache["cutout_off"]

    @property
    def wcs_full(self) -> Optional[WCS]:
        """Full-frame 2D celestial WCS for this candidate's parent beam."""
        if "wcs_full" not in self._cache:
            wcs_idx = int(self._meta.get("wcs_idx", -1))
            self._cache["wcs_full"] = self._store._get_wcs_by_index(wcs_idx)
        return self._cache["wcs_full"]

    @property
    def cutout_wcs(self) -> Optional[WCS]:
        """2D celestial WCS shifted to :attr:`cutout`."""
        if "cutout_wcs" not in self._cache:
            w_full = self.wcs_full
            if w_full is not None and self.x >= 0 and self.y >= 0:
                self._cache["cutout_wcs"] = _build_cutout_wcs(
                    w_full, x0=self.x, y0=self.y, cutout_size=self._store.cutout_size
                )
            else:
                self._cache["cutout_wcs"] = None
        return self._cache["cutout_wcs"]

    @property
    def lightcurve(self) -> Optional[Lightcurve]:
        """1D candidate lightcurve (boxcar / variance)."""
        if "lightcurve" not in self._cache:
            self._cache["lightcurve"] = self._store._read_lightcurve(self._index, self.width_samples)
        return self._cache["lightcurve"]

    @property
    def profile(self) -> Optional[np.ndarray]:
        """1D folded pulse profile (FFA)."""
        if "profile" not in self._cache:
            self._cache["profile"] = self._store._read_profile(self._index)
        return self._cache["profile"]

    @property
    def spectrum(self) -> Optional[Spectrum]:
        """1D spin-frequency power spectrum (NUFFT)."""
        if "spectrum" not in self._cache:
            self._cache["spectrum"] = self._store._read_spectrum(self._index, self._meta)
        return self._cache["spectrum"]

    @property
    def fullframe(self) -> Optional[np.ndarray]:
        """Full-field 2D detection image (if stored)."""
        if "fullframe" not in self._cache:
            self._cache["fullframe"] = self._store._read_fullframe(self._index)
        return self._cache["fullframe"]

    @property
    def dynspec(self) -> Optional[DynamicSpectrum]:
        """Phased visibility dynamic spectrum (if stored)."""
        if "dynspec" not in self._cache:
            self._cache["dynspec"] = self._store._read_dynspec(self._index)
        return self._cache["dynspec"]

    @property
    def snippet(self) -> Optional[Dict[str, np.ndarray]]:
        """Time-resolved 3D cutout snippet (if stored)."""
        if "snippet" not in self._cache:
            self._cache["snippet"] = self._store._read_snippet(self._index)
        return self._cache["snippet"]

    def to_dict(self, *, load_arrays: bool = True) -> Dict[str, Any]:
        """Return a candidate dict compatible with FastDUCC plotting and PDMP helpers."""
        d = dict(self._meta)
        d["h5_index"] = self._index
        d["products_h5"] = str(self._store.path)
        d.setdefault("algo", self.method)
        if np.isfinite(self.ra_deg) and "ra_rad" not in d:
            d["ra_rad"] = float(np.deg2rad(self.ra_deg))
        if np.isfinite(self.dec_deg) and "dec_rad" not in d:
            d["dec_rad"] = float(np.deg2rad(self.dec_deg))
        if load_arrays:
            if self.cutout is not None:
                d["cutout"] = self.cutout
                if self.method == "nufft":
                    d["cutout_on_res"] = self.cutout
            if self.cutout_off is not None:
                d["cutout_off"] = self.cutout_off
                if self.method == "nufft":
                    d["cutout_off_res"] = self.cutout_off
            if self.fullframe is not None:
                d["fullframe"] = self.fullframe
            if self.profile is not None:
                d["profile"] = self.profile
            if self.spectrum is not None:
                d["spectrum_1d"] = self.spectrum.power
                d["spectrum_freqs"] = self.spectrum.freqs_hz
                if "profile" not in d:
                    d["profile"] = self.spectrum.power
            if self.dynspec is not None:
                d["dynspec"] = self.dynspec.to_dict()
        return d

    def plot_summary(
        self,
        out_path: Union[str, os.PathLike],
        *,
        continuum_dir: Optional[str] = None,
        dpi: int = 200,
        cmap: str = "viridis",
        **kwargs: Any,
    ) -> Dict[str, str]:
        """Render the FastDUCC diagnostic summary plot directly from this HDF5 candidate record."""
        from fastducc.candidate_tools.plotting import render_candidate_summary_from_record

        return render_candidate_summary_from_record(
            self,
            out_path=out_path,
            continuum_dir=continuum_dir,
            dpi=dpi,
            cmap=cmap,
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"CandidateRecord(srcname={self.srcname!r}, method={self.method!r}, "
            f"beam={self.beam!r}, dm={self.dm:.2f}, snr={self.snr:.2f}, idx={self._index})"
        )


class CandidateStore(Mapping):
    """Read-only dictionary-like interface to a monolithic candidate HDF5 container.

    Supports both ``schema_version = "2.0"`` (boxcar, variance, FFA, NUFFT, and
    consolidated exemplar files) and legacy ``schema_version = "1.0"`` NUFFT files.

    Examples
    --------
    >>> with CandidateStore("FIELD.SB82418_obs_boxcar_consolidated_candidates.h5") as candidates:
    ...     cand = candidates["J123456.7-876543.2"]
    ...     lc = cand.lightcurve
    ...     cut = cand.cutout
    ...     cand.plot_summary("J123456.7-876543.2_summary.pdf")
    """

    def __init__(self, path: Union[str, os.PathLike]) -> None:
        self.path = Path(path)
        self._h5: Optional[h5py.File] = h5py.File(str(self.path), "r")
        self._schema_version = _decode_fixed_str(self._h5.attrs.get("schema_version", "1.0")) or "1.0"
        self._is_v1 = self._schema_version.startswith("1.")
        self._meta_cache: Optional[np.ndarray] = None
        self._freqs_cache: Optional[np.ndarray] = None
        self._wcs_cache: Dict[int, Optional[WCS]] = {}
        self._srcname_index: Optional[Dict[str, List[int]]] = None
        self._record_cache: Dict[int, CandidateRecord] = {}

        # Legacy v1 root WCS attribute
        raw_hdr = self._h5.attrs.get("wcs_header")
        self._root_wcs: Optional[WCS] = None
        if raw_hdr:
            try:
                self._root_wcs = _header_str_to_wcs(_decode_fixed_str(raw_hdr))
            except Exception as exc:
                logger.warning("Failed to parse root wcs_header in %s: %s", self.path, exc)

    # --- File-level properties ---
    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def method(self) -> str:
        raw = _decode_fixed_str(self._h5.attrs.get("method", ""))
        if raw:
            return normalise_method(raw)
        return "nufft" if self._is_v1 else "nufft"

    @property
    def beam(self) -> str:
        return _decode_fixed_str(self._h5.attrs.get("beam", ""))

    @property
    def sbid(self) -> str:
        return _decode_fixed_str(self._h5.attrs.get("sbid", ""))

    @property
    def field(self) -> str:
        return _decode_fixed_str(self._h5.attrs.get("field", ""))

    @property
    def dm(self) -> float:
        return _safe_float(self._h5.attrs.get("dm"), float("nan"))

    @property
    def is_consolidated(self) -> bool:
        return bool(self._h5.attrs.get("is_consolidated", False))

    @property
    def n_candidates(self) -> int:
        return int(self._h5["metadata"].shape[0])

    @property
    def cutout_size(self) -> int:
        return int(self._h5.attrs.get("cutout_size", self._h5["cutouts"].shape[-1]))

    @property
    def npix_x(self) -> int:
        return int(self._h5.attrs.get("npix_x", 384))

    @property
    def npix_y(self) -> int:
        return int(self._h5.attrs.get("npix_y", 384))

    @property
    def freqs_hz(self) -> np.ndarray:
        if self._freqs_cache is None:
            if "freqs_hz" in self._h5:
                self._freqs_cache = np.asarray(self._h5["freqs_hz"][:], dtype=np.float64)
            else:
                self._freqs_cache = np.empty(0, dtype=np.float64)
        return self._freqs_cache

    @property
    def metadata(self) -> np.ndarray:
        if self._meta_cache is None:
            self._meta_cache = self._h5["metadata"][:]
        return self._meta_cache

    @property
    def wcs_full(self) -> Optional[WCS]:
        return self._get_wcs_by_index(0)

    def _read_map(self, name: str) -> Optional[np.ndarray]:
        if "maps" in self._h5 and name in self._h5["maps"]:
            return np.asarray(self._h5["maps"][name][:], dtype=np.float32)
        if name in self._h5:
            return np.asarray(self._h5[name][:], dtype=np.float32)
        return None

    @property
    def max_snr_map(self) -> Optional[np.ndarray]:
        return self._read_map("max_snr_map")

    @property
    def best_freq_map(self) -> Optional[np.ndarray]:
        return self._read_map("best_freq_map")

    @property
    def std_map(self) -> Optional[np.ndarray]:
        return self._read_map("std_map")

    # --- Internal readers ---
    def _get_wcs_by_index(self, wcs_idx: int) -> Optional[WCS]:
        if wcs_idx in self._wcs_cache:
            return self._wcs_cache[wcs_idx]
        wcs_obj: Optional[WCS] = None
        if "wcs_headers" in self._h5:
            ds = self._h5["wcs_headers"]
            if 0 <= wcs_idx < ds.shape[0]:
                hdr_str = _decode_fixed_str(ds[wcs_idx])
                if hdr_str:
                    wcs_obj = _header_str_to_wcs(hdr_str)
            elif wcs_idx < 0 and ds.shape[0] > 0:
                hdr_str = _decode_fixed_str(ds[0])
                if hdr_str:
                    wcs_obj = _header_str_to_wcs(hdr_str)
        if wcs_obj is None:
            wcs_obj = self._root_wcs
        self._wcs_cache[wcs_idx] = wcs_obj
        return wcs_obj

    def _row_to_meta_dict(self, index: int) -> Dict[str, Any]:
        row = self.metadata[index]
        names = row.dtype.names or ()
        d: Dict[str, Any] = {}
        for name in names:
            val = row[name]
            if isinstance(val, (bytes, np.bytes_)):
                d[name] = _decode_fixed_str(val)
            elif isinstance(val, np.integer):
                d[name] = int(val)
            elif isinstance(val, np.floating):
                d[name] = float(val)
            else:
                d[name] = val
        # Normalise legacy v1 field name "algo" -> "method"
        if "method" not in d:
            d["method"] = normalise_method(d.get("algo", self.method), default=self.method)
        d.setdefault("algo", d["method"])
        d.setdefault("wcs_idx", 0)

        if "extra_json" in self._h5 and index < self._h5["extra_json"].shape[0]:
            raw_extra = _decode_fixed_str(self._h5["extra_json"][index])
            if raw_extra:
                extra_dict = json.loads(raw_extra)
                for k, v in extra_dict.items():
                    if k not in d:
                        d[k] = v
        return d

    def _read_cutout(self, index: int) -> Optional[np.ndarray]:
        if "cutouts" not in self._h5:
            return None
        ds = self._h5["cutouts"]
        if ds.ndim == 4:
            # Schema 1.0: (N, 2, Y, X)
            arr = np.asarray(ds[index, 0], dtype=np.float32)
        else:
            arr = np.asarray(ds[index], dtype=np.float32)
        if np.all(np.isnan(arr)):
            return None
        return arr

    def _read_cutout_off(self, index: int) -> Optional[np.ndarray]:
        if "cutouts_off" in self._h5:
            arr = np.asarray(self._h5["cutouts_off"][index], dtype=np.float32)
            return None if np.all(np.isnan(arr)) else arr
        if "cutouts" in self._h5 and self._h5["cutouts"].ndim == 4:
            arr = np.asarray(self._h5["cutouts"][index, 1], dtype=np.float32)
            return None if np.all(np.isnan(arr)) else arr
        return None

    def _read_lightcurve(self, index: int, width_samples: int) -> Optional[Lightcurve]:
        if "lightcurves" not in self._h5:
            return None
        grp = self._h5["lightcurves"]
        offsets = grp["offsets"]
        if index + 1 >= offsets.shape[0]:
            return None
        i0 = int(offsets[index])
        i1 = int(offsets[index + 1])
        if i1 <= i0:
            return None
        return Lightcurve(
            time=np.asarray(grp["time"][i0:i1], dtype=np.float64),
            flux=np.asarray(grp["flux"][i0:i1], dtype=np.float32),
            flux_err=np.asarray(grp["flux_err"][i0:i1], dtype=np.float32),
            width_samples=max(1, int(width_samples)),
        )

    def _read_profile(self, index: int) -> Optional[np.ndarray]:
        if "profiles" not in self._h5:
            return None
        grp = self._h5["profiles"]
        offsets = grp["offsets"]
        if index + 1 >= offsets.shape[0]:
            return None
        i0 = int(offsets[index])
        i1 = int(offsets[index + 1])
        if i1 <= i0:
            return None
        return np.asarray(grp["values"][i0:i1], dtype=np.float32)

    def _read_spectrum(self, index: int, meta: Dict[str, Any]) -> Optional[Spectrum]:
        if "spectra_ragged" in self._h5:
            grp = self._h5["spectra_ragged"]
            offsets = grp["offsets"]
            if index + 1 < offsets.shape[0]:
                i0 = int(offsets[index])
                i1 = int(offsets[index + 1])
                if i1 > i0:
                    return Spectrum(
                        freqs_hz=np.asarray(grp["freqs"][i0:i1], dtype=np.float64),
                        power=np.asarray(grp["values"][i0:i1], dtype=np.float32),
                        f_hz=float(meta.get("f_hz", float("nan"))),
                        f_off_res=float(meta.get("f_off_res", float("nan"))),
                        spec_isolation_ratio=float(meta.get("spec_isolation_ratio", float("nan"))),
                        spec_local_contrast=float(meta.get("spec_local_contrast", float("nan"))),
                    )
        if "spectra" in self._h5:
            power = np.asarray(self._h5["spectra"][index], dtype=np.float32)
            if power.size > 0:
                return Spectrum(
                    freqs_hz=self.freqs_hz,
                    power=power,
                    f_hz=float(meta.get("f_hz", float("nan"))),
                    f_off_res=float(meta.get("f_off_res", float("nan"))),
                    spec_isolation_ratio=float(meta.get("spec_isolation_ratio", float("nan"))),
                    spec_local_contrast=float(meta.get("spec_local_contrast", float("nan"))),
                )
        return None

    def _read_fullframe(self, index: int) -> Optional[np.ndarray]:
        if "fullframes" not in self._h5:
            return None
        key = str(index)
        if key not in self._h5["fullframes"]:
            return None
        return np.asarray(self._h5["fullframes"][key][:], dtype=np.float32)

    def _read_dynspec(self, index: int) -> Optional[DynamicSpectrum]:
        if "dynspecs" not in self._h5:
            return None
        key = str(index)
        if key not in self._h5["dynspecs"]:
            return None
        grp = self._h5["dynspecs"][key]
        return DynamicSpectrum(
            data=np.asarray(grp["data"][:], dtype=np.float32),
            time=np.asarray(grp["time"][:], dtype=np.float64),
            freq=np.asarray(grp["freq"][:], dtype=np.float64),
            time_raw=np.asarray(grp["time_raw"][:], dtype=np.float64) if "time_raw" in grp else None,
        )

    def _read_snippet(self, index: int) -> Optional[Dict[str, np.ndarray]]:
        if "snippets" not in self._h5:
            return None
        key = str(index)
        if key not in self._h5["snippets"]:
            return None
        grp = self._h5["snippets"][key]
        out: Dict[str, np.ndarray] = {"snippet_cube": np.asarray(grp["cube"][:], dtype=np.float32)}
        if "times" in grp:
            out["snippet_times"] = np.asarray(grp["times"][:], dtype=np.float64)
        return out

    # --- Indexing & Mapping interface ---
    def _ensure_srcname_index(self) -> Dict[str, List[int]]:
        if self._srcname_index is None:
            idx_map: Dict[str, List[int]] = collections.defaultdict(list)
            meta = self.metadata
            for i in range(meta.shape[0]):
                name = _decode_fixed_str(meta[i]["srcname"]) or f"cand_{i:06d}"
                idx_map[name].append(i)
            # Order duplicate entries per srcname by descending S/N so [srcname] returns the exemplar
            snr_col = meta["snr"]
            for name, idxs in idx_map.items():
                if len(idxs) > 1:
                    idxs.sort(key=lambda ii: float(snr_col[ii]), reverse=True)
            self._srcname_index = dict(idx_map)
        return self._srcname_index

    def _get_record(self, index: int) -> CandidateRecord:
        n = self.n_candidates
        if index < 0:
            index += n
        if index < 0 or index >= n:
            raise IndexError(f"Candidate index {index} out of range [0, {n}) in {self.path}")
        if index not in self._record_cache:
            self._record_cache[index] = CandidateRecord(self, index, self._row_to_meta_dict(index))
        return self._record_cache[index]

    def __len__(self) -> int:
        return len(self._ensure_srcname_index())

    def __iter__(self) -> Iterator[str]:
        return iter(self._ensure_srcname_index())

    def __contains__(self, key: object) -> bool:
        if isinstance(key, (int, np.integer)):
            return 0 <= int(key) < self.n_candidates
        return str(key) in self._ensure_srcname_index()

    def __getitem__(self, key: Union[str, int, np.integer]) -> CandidateRecord:
        if isinstance(key, (int, np.integer)):
            return self._get_record(int(key))
        idx_map = self._ensure_srcname_index()
        k = str(key)
        if k not in idx_map:
            raise KeyError(f"Candidate '{k}' not found in {self.path}")
        return self._get_record(idx_map[k][0])

    def get_all(self, srcname: str) -> List[CandidateRecord]:
        """Return all :class:`CandidateRecord` rows matching *srcname*, ordered by descending S/N."""
        idxs = self._ensure_srcname_index().get(str(srcname), [])
        return [self._get_record(i) for i in idxs]

    @property
    def records(self) -> List[CandidateRecord]:
        """All candidate records in storage index order."""
        return [self._get_record(i) for i in range(self.n_candidates)]

    def filter(
        self,
        *,
        min_snr: Optional[float] = None,
        max_snr: Optional[float] = None,
        beam: Optional[str] = None,
        method: Optional[str] = None,
        dm_min: Optional[float] = None,
        dm_max: Optional[float] = None,
        srcname: Optional[str] = None,
    ) -> List[CandidateRecord]:
        """Filter candidates by metadata predicates and return matching records."""
        n = self.n_candidates
        if n == 0:
            return []
        meta = self.metadata
        mask = np.ones(n, dtype=bool)
        if min_snr is not None:
            mask &= meta["snr"] >= float(min_snr)
        if max_snr is not None:
            mask &= meta["snr"] <= float(max_snr)
        if dm_min is not None:
            mask &= meta["dm"] >= float(dm_min)
        if dm_max is not None:
            mask &= meta["dm"] <= float(dm_max)
        if beam is not None:
            b_bytes = _encode_fixed_str(normalise_beam(beam), 16)
            mask &= meta["beam"] == b_bytes
        if method is not None and "method" in (meta.dtype.names or ()):
            m_bytes = _encode_fixed_str(normalise_method(method), 16)
            mask &= meta["method"] == m_bytes
        if srcname is not None:
            s_bytes = _encode_fixed_str(srcname, 48)
            mask &= meta["srcname"] == s_bytes
        return [self._get_record(int(i)) for i in np.flatnonzero(mask)]

    def top(self, n: int = 20, *, by: str = "snr") -> List[CandidateRecord]:
        """Return the top *n* candidates sorted descending by metadata field *by*."""
        total = self.n_candidates
        if total == 0 or n <= 0:
            return []
        vals = np.asarray(self.metadata[by], dtype=np.float64)
        order = np.argsort(np.nan_to_num(vals, nan=-np.inf))[::-1][: min(int(n), total)]
        return [self._get_record(int(i)) for i in order]

    def to_table(self) -> Table:
        """Return an Astropy Table of all candidate metadata rows."""
        rows = [self._get_record(i).meta for i in range(self.n_candidates)]
        if not rows:
            return Table()
        all_keys: List[str] = []
        seen: set[str] = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    all_keys.append(k)
        return Table(rows=[{k: r.get(k, "") for k in all_keys} for r in rows])

    # --- Legacy CandidateHDF5Reader API compatibility ---
    def get_candidate(self, index: int) -> Dict[str, Any]:
        """Load cutouts, 1D spectrum/profile/lightcurve, WCS, and metadata for 0-based *index*."""
        rec = self._get_record(index)
        cut_on = rec.cutout
        cut_off = rec.cutout_off
        spec = rec.spectrum
        spec_1d = spec.power if spec is not None else np.empty(0, dtype=np.float32)
        freqs = spec.freqs_hz if spec is not None else self.freqs_hz
        return {
            "h5_index": int(index),
            "metadata": rec.meta,
            "cut_on": cut_on,
            "cut_off": cut_off,
            "spec_1d": spec_1d,
            "freqs_hz": freqs,
            "profile": rec.profile,
            "lightcurve": rec.lightcurve,
            "fullframe": rec.fullframe,
            "dynspec": rec.dynspec,
            "wcs_cutout": rec.cutout_wcs,
            "wcs_full": rec.wcs_full,
            "f_off_res": rec.f_off_res,
        }

    def find_candidate_index(
        self,
        *,
        h5_index: Optional[int] = None,
        cand_idx: Optional[int] = None,
        srcname: Optional[str] = None,
        beam: Optional[str] = None,
        dm: Optional[float] = None,
        period_s: Optional[float] = None,
        time_center: Optional[float] = None,
        width_samples: Optional[int] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
    ) -> Optional[int]:
        """Locate a candidate 0-based row index by direct ``h5_index`` or metadata predicates."""
        n = self.n_candidates
        if n == 0:
            return None
        if h5_index is not None and 0 <= int(h5_index) < n:
            return int(h5_index)

        meta = self.metadata
        names = set(meta.dtype.names or ())

        if cand_idx is not None and "cand_idx" in names:
            hits = np.flatnonzero(meta["cand_idx"] == int(cand_idx))
            if hits.size == 1:
                return int(hits[0])

        mask = np.ones(n, dtype=bool)
        used_predicate = False

        if beam and "beam" in names:
            b_enc = _encode_fixed_str(normalise_beam(beam), 16)
            if b_enc:
                mask &= meta["beam"] == b_enc
                used_predicate = True
        if dm is not None and np.isfinite(dm) and "dm" in names:
            mask &= np.isclose(meta["dm"], float(dm), rtol=1e-3, atol=0.05)
            used_predicate = True
        if period_s is not None and np.isfinite(period_s) and period_s > 0 and "period_s" in names:
            mask &= np.isclose(meta["period_s"], float(period_s), rtol=1e-3, atol=1e-5)
            used_predicate = True
        if time_center is not None and np.isfinite(time_center) and "time_center" in names:
            mask &= np.isclose(meta["time_center"], float(time_center), rtol=0.0, atol=1.0)
            used_predicate = True
        if width_samples is not None and int(width_samples) > 0 and "width_samples" in names:
            mask &= meta["width_samples"] == int(width_samples)
            used_predicate = True
        if x is not None and y is not None and int(x) >= 0 and int(y) >= 0:
            mask &= (meta["x"] == int(x)) & (meta["y"] == int(y))
            used_predicate = True
        if srcname and "srcname" in names:
            target = _encode_fixed_str(srcname, 48)
            s_mask = mask & (meta["srcname"] == target)
            if np.any(s_mask):
                hits = np.flatnonzero(s_mask)
                return int(hits[int(np.argmax(meta["snr"][hits]))])

        if used_predicate:
            hits = np.flatnonzero(mask)
            if hits.size > 0:
                return int(hits[int(np.argmax(meta["snr"][hits]))])

        if srcname and "srcname" in names:
            target = _encode_fixed_str(srcname, 48)
            hits = np.flatnonzero(meta["srcname"] == target)
            if hits.size > 0:
                return int(hits[int(np.argmax(meta["snr"][hits]))])

        return None

    def find(self, **kwargs: Any) -> Optional[CandidateRecord]:
        """Locate and return a :class:`CandidateRecord` matching the given metadata fields."""
        idx = self.find_candidate_index(**kwargs)
        return self._get_record(idx) if idx is not None else None

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __enter__(self) -> "CandidateStore":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"CandidateStore({str(self.path)!r}, method={self.method!r}, "
            f"n_candidates={self.n_candidates}, unique_sources={len(self)})"
        )


# Backwards-compatible alias
CandidateHDF5Reader = CandidateStore


def open_candidates(path: Union[str, os.PathLike]) -> CandidateStore:
    """Open a monolithic candidate HDF5 container and return a :class:`CandidateStore`."""
    return CandidateStore(path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def is_valid_candidates_h5(path: Union[str, os.PathLike]) -> bool:
    """Return True iff *path* is an intact candidate HDF5 container (schema 1.0 or 2.0)."""
    p = Path(path)
    if _is_tmp_name(p) or not p.is_file():
        return False
    try:
        if p.stat().st_size <= 0:
            return False
        with h5py.File(str(p), "r") as f:
            if "metadata" not in f or "cutouts" not in f:
                return False
            n = int(f["metadata"].shape[0])
            if f["cutouts"].shape[0] != n:
                return False
            if "spectra" in f and f["spectra"].shape[0] != n:
                return False
            if "cutouts_off" in f and f["cutouts_off"].shape[0] != n:
                return False
        return True
    except (OSError, KeyError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Merging & Exemplar Consolidation
# ---------------------------------------------------------------------------


def _record_to_writer_payload(rec: CandidateRecord, extra_meta: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    meta = dict(rec.meta)
    if extra_meta:
        meta.update(extra_meta)
    arr: Dict[str, Any] = {}
    if rec.cutout is not None:
        arr["cut_on"] = rec.cutout
    if rec.cutout_off is not None:
        arr["cut_off"] = rec.cutout_off
    if rec.lightcurve is not None:
        arr["lightcurve"] = rec.lightcurve
    if rec.profile is not None:
        arr["profile"] = rec.profile
    if rec.spectrum is not None:
        arr["spec_1d"] = rec.spectrum.power
        arr["freqs_hz"] = rec.spectrum.freqs_hz
        if np.isfinite(rec.spectrum.f_off_res):
            arr["f_off_res"] = rec.spectrum.f_off_res
    if rec.fullframe is not None:
        arr["fullframe"] = rec.fullframe
    if rec.dynspec is not None:
        arr["dynspec"] = rec.dynspec
    if rec.snippet is not None:
        arr["snippet"] = rec.snippet
    if rec.wcs_full is not None:
        arr["wcs_full"] = rec.wcs_full
    return meta, arr


def merge_candidate_h5(
    input_paths: Sequence[Union[str, os.PathLike]],
    dest_path: Union[str, os.PathLike],
    *,
    method: Optional[str] = None,
    sbid: str = "",
    field: str = "",
    beam: str = "",
    dm: Optional[float] = None,
    cutout_size: Optional[int] = None,
    compression: Optional[str] = "lzf",
    remove_inputs: bool = False,
) -> Path:
    """Merge multiple candidate HDF5 containers into a single HDF5 file.

    Preserves all candidates, per-beam celestial WCS headers, cutouts, lightcurves,
    folded profiles, spectra, and optional sparse artefacts.
    """
    valid_paths = [Path(p) for p in input_paths if is_valid_candidates_h5(p)]
    if not valid_paths:
        raise ValueError("No valid candidate HDF5 input files provided for merge.")

    detected_method: Optional[str] = normalise_method(method) if method else None
    max_cut = int(cutout_size) if cutout_size is not None else 0
    common_freqs: Optional[np.ndarray] = None
    uniform_freqs = True
    any_off = False
    inferred_sbid = sbid
    inferred_field = field

    for p in valid_paths:
        with CandidateStore(p) as st:
            if detected_method is None:
                detected_method = st.method
            max_cut = max(max_cut, st.cutout_size)
            if not inferred_sbid and st.sbid:
                inferred_sbid = st.sbid
            if not inferred_field and st.field:
                inferred_field = st.field
            if "cutouts_off" in st._h5 or (st._h5["cutouts"].ndim == 4):
                any_off = True
            fq = st.freqs_hz
            if fq.size > 0:
                if common_freqs is None:
                    common_freqs = fq.copy()
                elif fq.shape != common_freqs.shape or not np.allclose(fq, common_freqs, rtol=1e-9, atol=1e-12):
                    uniform_freqs = False

    resolved_method = detected_method or "nufft"
    resolved_cut = max_cut if max_cut > 0 else (50 if resolved_method in ("boxcar", "variance") else 32)
    writer_freqs = common_freqs if (uniform_freqs and common_freqs is not None) else None

    with CandidateHDF5Writer(
        dest_path,
        method=resolved_method,
        freqs_hz=writer_freqs,
        cutout_size=resolved_cut,
        sbid=inferred_sbid,
        field=inferred_field,
        beam=beam,
        dm=dm,
        has_off_cutout=any_off,
        is_consolidated=False,
        compression=compression,
    ) as writer:
        for p in valid_paths:
            with CandidateStore(p) as st:
                if st.n_candidates == 0:
                    continue
                c_batch: List[Dict[str, Any]] = []
                a_batch: List[Dict[str, Any]] = []
                for idx in range(st.n_candidates):
                    rec = st[idx]
                    c_meta, c_arr = _record_to_writer_payload(rec)
                    c_batch.append(c_meta)
                    a_batch.append(c_arr)
                writer.append_batch(c_batch, a_batch)

    if remove_inputs:
        dest_res = Path(dest_path).resolve()
        for p in valid_paths:
            if p.resolve() != dest_res:
                try:
                    p.unlink()
                except OSError as exc:
                    logger.warning("Could not remove merged input '%s': %s", p, exc)

    return Path(dest_path)


def consolidate_candidate_h5(
    input_paths: Sequence[Union[str, os.PathLike]],
    exemplars: Union[Table, Sequence[Dict[str, Any]]],
    dest_path: Union[str, os.PathLike],
    *,
    method: Optional[str] = None,
    sbid: str = "",
    field: str = "",
    cutout_size: Optional[int] = None,
    compression: Optional[str] = "lzf",
) -> Tuple[Path, List[Tuple[int, int]]]:
    """Build a consolidated observation-level HDF5 containing ONLY exemplar candidates.

    Parameters
    ----------
    input_paths : sequence of paths
        Per-(method, beam, DM) candidate HDF5 containers to draw artefacts from.
    exemplars : astropy.table.Table or sequence of dict
        Aggregated exemplar candidate rows (one per unique sky source). Each row
        may specify ``products_h5`` / ``h5_index`` or will be matched against the
        input HDF5s by ``beam``, ``dm``, ``srcname``, ``period_s``, ``time_center``,
        ``width_samples``, ``x``, ``y``.
    dest_path : path
        Output consolidated HDF5 file path.
    method : str, optional
        Search method label (``boxcar``, ``variance``, ``ffa``, ``nufft``).

    Returns
    -------
    (dest_path, matched_pairs)
        ``matched_pairs`` is a list of ``(exemplar_row_idx, consolidated_h5_idx)``
        for every exemplar resolved and written into the consolidated container.
    """
    valid_paths = [Path(p).resolve() for p in input_paths if is_valid_candidates_h5(p)]
    if isinstance(exemplars, Table):
        ex_rows = [{col: exemplars[col][i] for col in exemplars.colnames} for i in range(len(exemplars))]
    else:
        ex_rows = [dict(r) for r in exemplars]

    # Index input stores by beam and also allow explicit products_h5 paths
    stores: Dict[Path, CandidateStore] = {}
    by_beam: Dict[str, List[Path]] = collections.defaultdict(list)
    by_beam_dm: Dict[Tuple[str, float], List[Path]] = collections.defaultdict(list)

    def _get_store(p: Path) -> Optional[CandidateStore]:
        rp = p.resolve()
        if rp not in stores:
            if not is_valid_candidates_h5(rp):
                return None
            stores[rp] = CandidateStore(rp)
        return stores[rp]

    detected_method: Optional[str] = normalise_method(method) if method else None
    max_cut = int(cutout_size) if cutout_size is not None else 0
    common_freqs: Optional[np.ndarray] = None
    uniform_freqs = True
    any_off = False
    inferred_sbid = sbid
    inferred_field = field

    try:
        # Include any explicit products_h5 referenced by exemplar rows
        all_paths = list(valid_paths)
        seen_p = set(valid_paths)
        for row in ex_rows:
            ph5 = str(row.get("products_h5", "") or "").strip()
            if ph5 and ph5.lower() not in ("none", "nan", "--"):
                pp = Path(ph5)
                if pp.is_file():
                    rp = pp.resolve()
                    if rp not in seen_p and is_valid_candidates_h5(rp):
                        seen_p.add(rp)
                        all_paths.append(rp)

        for rp in all_paths:
            st = _get_store(rp)
            if st is None:
                continue
            if detected_method is None:
                detected_method = st.method
            max_cut = max(max_cut, st.cutout_size)
            if not inferred_sbid and st.sbid:
                inferred_sbid = st.sbid
            if not inferred_field and st.field:
                inferred_field = st.field
            if "cutouts_off" in st._h5 or (st._h5["cutouts"].ndim == 4):
                any_off = True
            fq = st.freqs_hz
            if fq.size > 0:
                if common_freqs is None:
                    common_freqs = fq.copy()
                elif fq.shape != common_freqs.shape or not np.allclose(fq, common_freqs, rtol=1e-9, atol=1e-12):
                    uniform_freqs = False

            finfo = parse_candidate_h5_filename(rp.name)
            b_key = normalise_beam(st.beam or finfo["beam"])
            by_beam[b_key].append(rp)
            dm_val = st.dm if np.isfinite(st.dm) else finfo["dm"]
            if dm_val is not None and np.isfinite(dm_val):
                by_beam_dm[(b_key, round(float(dm_val), 2))].append(rp)

        resolved_method = detected_method or "nufft"
        resolved_cut = max_cut if max_cut > 0 else (50 if resolved_method in ("boxcar", "variance") else 32)
        writer_freqs = common_freqs if (uniform_freqs and common_freqs is not None) else None

        matched_pairs: List[Tuple[int, int]] = []

        with CandidateHDF5Writer(
            dest_path,
            method=resolved_method,
            freqs_hz=writer_freqs,
            cutout_size=resolved_cut,
            sbid=inferred_sbid,
            field=inferred_field,
            has_off_cutout=any_off,
            is_consolidated=True,
            compression=compression,
        ) as writer:
            for ex_idx, row in enumerate(ex_rows):
                beam_str = normalise_beam(row.get("max_snr_beam") or row.get("beam") or "")
                dm_val = _safe_float(row.get("best_dm", row.get("dm")), float("nan"))
                period_val = _safe_float(row.get("period_s"), float("nan"))
                t_center = _safe_float(row.get("max_snr_time_center", row.get("time_center")), float("nan"))
                w_val = _safe_int(row.get("max_snr_width", row.get("width_samples")), -1)
                x_val = _safe_int(row.get("x"), -1)
                y_val = _safe_int(row.get("y"), -1)
                srcname = _decode_fixed_str(row.get("srcname"))

                # Build prioritised list of candidate source stores
                cand_stores: List[Path] = []
                explicit_h5 = str(row.get("products_h5", "") or "").strip()
                explicit_path: Optional[Path] = None
                if explicit_h5 and explicit_h5.lower() not in ("none", "nan", "--"):
                    ep = Path(explicit_h5)
                    if ep.is_file():
                        explicit_path = ep.resolve()
                        cand_stores.append(explicit_path)

                if beam_str and np.isfinite(dm_val):
                    for rp in by_beam_dm.get((beam_str, round(dm_val, 2)), []):
                        if rp not in cand_stores:
                            cand_stores.append(rp)
                if beam_str:
                    for rp in by_beam.get(beam_str, []):
                        if rp not in cand_stores:
                            cand_stores.append(rp)
                if not cand_stores:
                    cand_stores = list(all_paths)

                req_h5_idx = _safe_int(row.get("h5_index"), -1)
                found_store: Optional[CandidateStore] = None
                found_idx: Optional[int] = None

                for rp in cand_stores:
                    st = _get_store(rp)
                    if st is None or st.n_candidates == 0:
                        continue
                    idx = st.find_candidate_index(
                        h5_index=req_h5_idx if (explicit_path is not None and rp == explicit_path and req_h5_idx >= 0) else None,
                        srcname=srcname or None,
                        beam=beam_str or None,
                        dm=dm_val if np.isfinite(dm_val) else None,
                        period_s=period_val if (np.isfinite(period_val) and period_val > 0) else None,
                        time_center=t_center if np.isfinite(t_center) else None,
                        width_samples=w_val if w_val > 0 else None,
                        x=x_val if x_val >= 0 else None,
                        y=y_val if y_val >= 0 else None,
                    )
                    if idx is not None:
                        found_store = st
                        found_idx = idx
                        break

                if found_store is None or found_idx is None:
                    continue

                rec = found_store[found_idx]
                # Merge consolidated exemplar provenance metadata onto the record
                provenance: Dict[str, Any] = {
                    "source_h5": str(found_store.path),
                    "source_h5_index": int(found_idx),
                }
                for k in (
                    "source_id", "beams_all", "dm_all", "width_samples_all", "scan_ids", "scans_all",
                    "n_beams", "n_scans", "n_events", "n_detections", "n_detections_total",
                    "psrcat_match_name", "psrcat_sep_arcsec", "racs_match_name", "racs_sep_arcsec",
                    "max_snr", "max_snr_beam", "max_snr_width", "max_snr_time_center",
                    "classification", "vetting_score",
                ):
                    if k in row and row[k] is not None:
                        provenance[k] = row[k]
                if srcname:
                    provenance["srcname"] = srcname

                c_meta, c_arr = _record_to_writer_payload(rec, extra_meta=provenance)
                new_idx = writer.append_batch([c_meta], [c_arr])[0]
                matched_pairs.append((ex_idx, new_idx))

        logger.info(
            "Consolidated %d/%d exemplar candidate(s) [%s] -> %s",
            len(matched_pairs),
            len(ex_rows),
            resolved_method,
            dest_path,
        )
        return Path(dest_path), matched_pairs
    finally:
        for st in stores.values():
            st.close()


# ---------------------------------------------------------------------------
# On-demand materialisation (backwards-compatible with lotrun_periodicity)
# ---------------------------------------------------------------------------


def _atomic_write_fits(dest_path: Path, data: np.ndarray, header: fits.Header) -> Path:
    tmp = _make_temp_path(dest_path, suffix=".fits")
    try:
        fits.writeto(str(tmp), data, header=header, overwrite=True)
        os.replace(str(tmp), str(dest_path))
        return dest_path
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def _atomic_write_npz(dest_path: Path, **arrays: Any) -> Path:
    tmp = _make_temp_path(dest_path, suffix=".npz")
    try:
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrays)
        os.replace(str(tmp), str(dest_path))
        return dest_path
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def materialize_candidate_products_from_h5(
    h5_path: Union[str, os.PathLike],
    index: int,
    out_dir: Union[str, os.PathLike],
    *,
    stem: Optional[str] = None,
    render_plot: bool = False,
    continuum_dir: Optional[str] = None,
    reader: Optional[CandidateStore] = None,
) -> Dict[str, str]:
    """Extract a single candidate from an HDF5 container into standalone FITS/NPZ/PDF files."""
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    owns_reader = reader is None
    rdr = CandidateStore(h5_path) if owns_reader else reader
    try:
        rec = rdr[int(index)]
        meta = rec.meta
        method = rec.method
        cut_on = rec.cutout
        cut_off = rec.cutout_off
        wcs_cut = rec.cutout_wcs
        spec = rec.spectrum
        prof = rec.profile
        lc = rec.lightcurve

        if not stem:
            src = rec.srcname or f"cand{rec.cand_idx:04d}"
            beam = rec.beam or "beamXX"
            dm = rec.dm if np.isfinite(rec.dm) else 0.0
            if method in ("nufft", "ffa") and np.isfinite(rec.period_s) and rec.period_s > 0:
                stem = f"{src}_{beam}_dm{dm:.2f}_p{rec.period_s * 1e3:.2f}ms"
            else:
                stem = f"{src}_{beam}_dm{dm:.2f}_{method}"

        def _make_hdr(is_off: bool = False) -> fits.Header:
            hdr = wcs_cut.to_header() if wcs_cut is not None else fits.Header()
            hdr["SRCNAME"] = str(rec.srcname)[:68]
            hdr["SNR"] = float(rec.snr) if np.isfinite(rec.snr) else 0.0
            hdr["DM"] = float(rec.dm) if np.isfinite(rec.dm) else 0.0
            hdr["METHOD"] = str(method)[:16]
            if np.isfinite(rec.period_s):
                hdr["PERIOD"] = float(rec.period_s)
            if is_off and np.isfinite(rec.f_off_res):
                hdr["FREQ_HZ"] = float(rec.f_off_res)
            elif np.isfinite(rec.f_hz):
                hdr["FREQ_HZ"] = float(rec.f_hz)
            hdr["BEAM"] = str(rec.beam)[:16]
            hdr["SBID"] = str(rec.sbid)[:24]
            return hdr

        out_paths: Dict[str, str] = {
            "cutout_fits": "",
            "cutout_on_res_fits": "",
            "cutout_off_fits": "",
            "cutout_off_res_fits": "",
            "lightcurve_npz": "",
            "profile_npz": "",
            "spectrum_npz": "",
            "summary_plot": "",
        }

        if cut_on is not None:
            cut_on_path = out_root / f"{stem}_cutout.fits"
            _atomic_write_fits(cut_on_path, cut_on, _make_hdr(is_off=False))
            out_paths["cutout_fits"] = str(cut_on_path)
            if method == "nufft":
                out_paths["cutout_on_res_fits"] = str(cut_on_path)

        if cut_off is not None:
            cut_off_path = out_root / f"{stem}_cutout_off.fits"
            _atomic_write_fits(cut_off_path, cut_off, _make_hdr(is_off=True))
            out_paths["cutout_off_fits"] = str(cut_off_path)
            if method == "nufft":
                out_paths["cutout_off_res_fits"] = str(cut_off_path)

        if spec is not None and spec.power.size > 0:
            spec_npz_path = out_root / f"{stem}_nufft_spectrum.npz"
            _atomic_write_npz(
                spec_npz_path,
                spectrum=spec.power,
                fkeep=spec.freqs_hz,
                snr=np.float64(rec.snr),
                period_s=np.float64(rec.period_s),
                f_hz=np.float64(rec.f_hz),
                dm=np.float64(rec.dm),
                srcname=str(rec.srcname),
                spec_isolation_ratio=np.float64(rec.spec_isolation_ratio),
                spec_local_contrast=np.float64(rec.spec_local_contrast),
            )
            out_paths["spectrum_npz"] = str(spec_npz_path)
            out_paths["profile_npz"] = str(spec_npz_path)
            out_paths["lightcurve_npz"] = str(spec_npz_path)

        if prof is not None and prof.size > 0:
            prof_npz_path = out_root / f"{stem}_profile.npz"
            _atomic_write_npz(
                prof_npz_path,
                profile=prof,
                snr=np.float64(rec.snr),
                period_s=np.float64(rec.period_s),
                f_hz=np.float64(rec.f_hz),
                dm=np.float64(rec.dm),
                srcname=str(rec.srcname),
                width_bins=np.int32(rec.width_bins),
                duty_cycle=np.float32(rec.duty_cycle),
            )
            out_paths["profile_npz"] = str(prof_npz_path)
            out_paths["lightcurve_npz"] = str(prof_npz_path)

        if lc is not None and len(lc) > 0:
            lc_npz_path = out_root / f"{stem}_lc_full.npz"
            _atomic_write_npz(
                lc_npz_path,
                time=lc.time,
                flux=lc.flux,
                flux_err=lc.flux_err,
                error=lc.flux_err,
                width=np.int32(lc.width_samples),
            )
            out_paths["lightcurve_npz"] = str(lc_npz_path)

        if render_plot:
            dst_pdf = out_root / f"{stem}_summary.pdf"
            plot_res = rec.plot_summary(
                dst_pdf,
                continuum_dir=continuum_dir,
            )
            if plot_res.get("summary_plot") and os.path.exists(plot_res["summary_plot"]):
                out_paths["summary_plot"] = plot_res["summary_plot"]

        return out_paths
    finally:
        if owns_reader:
            rdr.close()


__all__ = [
    "SCHEMA_VERSION",
    "VALID_METHODS",
    "CANDIDATE_META_DTYPE",
    "normalise_method",
    "normalise_beam",
    "build_trial_h5_path",
    "build_consolidated_h5_path",
    "parse_candidate_h5_filename",
    "discover_trial_h5_files",
    "Lightcurve",
    "Spectrum",
    "DynamicSpectrum",
    "CandidateHDF5Writer",
    "CandidateRecord",
    "CandidateStore",
    "CandidateHDF5Reader",
    "open_candidates",
    "is_valid_candidates_h5",
    "merge_candidate_h5",
    "consolidate_candidate_h5",
    "materialize_candidate_products_from_h5",
]
