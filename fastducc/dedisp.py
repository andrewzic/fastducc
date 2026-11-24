"""
Brute-force visibility dedispersion module for fastducc.

Implements DedispersionPlan, BruteForceDedisperser, and frequency channel collapsing
before snapshot imaging, conforming to the CRACO dedispersion specification.
Also provides compatibility fallbacks / wrappers for tools from craco-python.
"""

from typing import Optional, Tuple, Union, List
import numpy as np

from fastducc.constants import K_DM as _K_DM_MHZ

# Dispersion constant: k_DM in s * GHz^2 * pc^-1 * cm^3
# (4.148808e3 MHz^2 pc^-1 cm^3 s = 4.148808e-3 GHz^2 pc^-1 cm^3 s)
K_DM = _K_DM_MHZ * 1e-6

# Attempt to import from craco if available
try:
    from craco.preprocess import Dedisp as CracoDedisp
    from craco.preprocess import get_dm_delays as craco_get_dm_delays
    from craco.preprocess import get_dm_samps as craco_get_dm_samps
    HAVE_CRACO = True
except ImportError:
    CracoDedisp = None
    craco_get_dm_delays = None
    craco_get_dm_samps = None
    HAVE_CRACO = False


def get_dm_delays(freqs_hz: np.ndarray, dm_pccc: float, tsamp_s: float, align_to: str = "fmax") -> np.ndarray:
    """
    Calculate integer sample delays per channel relative to reference frequency.

    Parameters
    ----------
    freqs_hz : np.ndarray
        Channel frequencies in Hz.
    dm_pccc : float
        Dispersion measure in pc cm^-3.
    tsamp_s : float
        Sampling interval in seconds.
    align_to : {'fmax', 'fmin'}
        Reference frequency alignment. 'fmax' ensures causal delays (delays >= 0).

    Returns
    -------
    delays_samp : np.ndarray (int64)
        Integer delays in samples for each channel.
    """
    freqs_ghz = np.asarray(freqs_hz, dtype=np.float64) * 1e-9
    fmin_ghz = np.min(freqs_ghz)
    fmax_ghz = np.max(freqs_ghz)

    if align_to == "fmax":
        ref_ghz = fmax_ghz
        delays_s = K_DM * dm_pccc * (1.0 / freqs_ghz**2 - 1.0 / ref_ghz**2)
    elif align_to == "fmin":
        ref_ghz = fmin_ghz
        delays_s = K_DM * dm_pccc * (1.0 / ref_ghz**2 - 1.0 / freqs_ghz**2)
    else:
        raise ValueError(f"Unknown align_to mode: {align_to}")

    return np.round(delays_s / tsamp_s).astype(np.int64)


def get_dm_samps(freqs_hz: np.ndarray, dm_pccc: float, tsamp_s: float) -> int:
    """
    Calculate total dispersion sweep in samples across the full band for a given DM.
    """
    freqs_ghz = np.asarray(freqs_hz, dtype=np.float64) * 1e-9
    fmin_ghz = np.min(freqs_ghz)
    fmax_ghz = np.max(freqs_ghz)
    delay_s = K_DM * dm_pccc * (1.0 / fmin_ghz**2 - 1.0 / fmax_ghz**2)
    return int(np.round(delay_s / tsamp_s))


class DedispersionPlan:
    """
    Computes an optimal DM trial grid and per-channel / two-tier subbanded
    delay profiles based on observation frequency and time resolution.
    """
    K_DM = K_DM

    def __init__(
        self,
        freqs_hz: np.ndarray,
        tsamp_s: float,
        dm_min: float = 0.0,
        dm_max: float = 1000.0,
        dm_step: Optional[float] = None,
        tolerance_samples: float = 1.0,
        dedisp_mode: str = "subbanded",
        n_subbands: int = 16,
    ):
        """
        Parameters
        ----------
        freqs_hz : array_like
            Channel frequencies in Hz.
        tsamp_s : float
            Sampling interval in seconds.
        dm_min : float
            Minimum DM trial in pc/cm^3.
        dm_max : float
            Maximum DM trial in pc/cm^3.
        dm_step : float, optional
            Explicit DM trial step size. If None, derived from tolerance_samples.
        tolerance_samples : float
            Maximum acceptable delay error across full band in sample units (default: 1.0).
        dedisp_mode : {'subbanded', 'fullband'}
            Dedispersion engine mode (default: 'subbanded').
        n_subbands : int
            Number of coarse frequency subbands for two-tier subbanded dedispersion
            (default: 16).
        """
        mode_norm = str(dedisp_mode).strip().lower()
        if mode_norm not in ("subbanded", "fullband"):
            raise ValueError(
                f"Unsupported dedisp_mode='{dedisp_mode}'; expected 'subbanded' or 'fullband'."
            )
        if int(n_subbands) < 1:
            raise ValueError(f"n_subbands must be >= 1; got {n_subbands}.")

        self.freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
        self.freqs_ghz = self.freqs_hz * 1e-9
        self.tsamp_s = float(tsamp_s)
        self.tolerance_samples = float(tolerance_samples)
        self.fmin_ghz = float(np.min(self.freqs_ghz))
        self.fmax_ghz = float(np.max(self.freqs_ghz))
        self.nchan = len(freqs_hz)
        self.dedisp_mode = mode_norm
        self.n_subbands = max(1, min(int(n_subbands), self.nchan)) if self.nchan > 0 else 0

        # Band dispersion factor: (1/fmin^2 - 1/fmax^2)
        self.band_disp_factor = (1.0 / self.fmin_ghz**2 - 1.0 / self.fmax_ghz**2)

        # Delay (in samples) across the band per 1 pc/cm^3 of DM
        self.samples_per_dm = (self.K_DM * self.band_disp_factor) / self.tsamp_s

        # Optimal fine DM step size
        if dm_step is not None and dm_step > 0:
            self.dm_step = float(dm_step)
        else:
            self.dm_step = float(tolerance_samples / max(1e-12, self.samples_per_dm))

        # Fine trial DMs array
        if dm_max <= dm_min:
            self.trial_dms = np.array([float(dm_min)], dtype=np.float64)
        else:
            self.trial_dms = np.arange(dm_min, dm_max + 0.5 * self.dm_step, self.dm_step, dtype=np.float64)
        self.ndm = len(self.trial_dms)

        # Maximum delay sweep across the entire plan in samples
        self.max_delay_samps = int(np.ceil(self.K_DM * max(dm_max, dm_min) * self.band_disp_factor / self.tsamp_s))

        # Subband partitioning matching np.array_split(np.arange(nchan), n_subbands)
        if self.n_subbands > 0:
            q, r = divmod(self.nchan, self.n_subbands)
            sizes = np.empty(self.n_subbands, dtype=np.int32)
            sizes[:r] = q + 1
            sizes[r:] = q
            self.subband_offsets = np.zeros(self.n_subbands + 1, dtype=np.int32)
            self.subband_offsets[1:] = np.cumsum(sizes, dtype=np.int32)
            self.subband_freqs_hz = np.array(
                [float(np.mean(self.freqs_hz[self.subband_offsets[s]:self.subband_offsets[s + 1]]))
                 for s in range(self.n_subbands)],
                dtype=np.float64,
            )
            self.subband_max_freqs_hz = np.array(
                [float(np.max(self.freqs_hz[self.subband_offsets[s]:self.subband_offsets[s + 1]]))
                 for s in range(self.n_subbands)],
                dtype=np.float64,
            )
        else:
            self.subband_offsets = np.zeros(1, dtype=np.int32)
            self.subband_freqs_hz = np.empty(0, dtype=np.float64)
            self.subband_max_freqs_hz = np.empty(0, dtype=np.float64)

        # Coarse DM step guaranteeing intra-subband dispersive smearing <= tolerance_samples
        bw_ghz = self.fmax_ghz - self.fmin_ghz
        sub_bw_ghz = (bw_ghz / float(self.n_subbands)) if (self.n_subbands > 0 and bw_ghz > 0.0) else 0.0
        if sub_bw_ghz > 0.0:
            raw_coarse_step = (
                self.tolerance_samples * self.tsamp_s * (self.fmin_ghz ** 3)
            ) / (2.0 * self.K_DM * sub_bw_ghz)
            self.coarse_dm_step = float(max(raw_coarse_step, self.dm_step))
        else:
            self.coarse_dm_step = float(self.dm_step)

        # Coarse DM trial grid and mapping between fine and coarse trials
        if self.dedisp_mode == "fullband" or self.ndm <= 1 or dm_max <= dm_min:
            self.coarse_dms = self.trial_dms.copy()
        else:
            dm_lo = float(self.trial_dms[0])
            dm_hi = float(self.trial_dms[-1])
            dm_span = dm_hi - dm_lo
            n_coarse = max(1, int(np.ceil(dm_span / self.coarse_dm_step)))
            if n_coarse >= self.ndm:
                self.coarse_dms = self.trial_dms.copy()
            else:
                interval = dm_span / float(n_coarse)
                self.coarse_dms = dm_lo + (np.arange(n_coarse, dtype=np.float64) + 0.5) * interval
        self.n_coarse = int(self.coarse_dms.size)

        # Assign each fine DM trial to its nearest coarse DM trial
        self.fine_to_coarse_idx = np.array(
            [int(np.argmin(np.abs(self.coarse_dms - dm_f))) for dm_f in self.trial_dms],
            dtype=np.int32,
        )
        self.coarse_to_fine_indices: List[np.ndarray] = [
            np.flatnonzero(self.fine_to_coarse_idx == ic).astype(np.int32)
            for ic in range(self.n_coarse)
        ]

        # Precompute causal (align_to="fmax") intra-subband and inter-subband delay tables
        self.intra_delays = np.zeros((self.n_coarse, self.nchan), dtype=np.int32)
        for ic, dm_c in enumerate(self.coarse_dms):
            self.intra_delays[ic] = self.get_intra_delays(float(dm_c), align_to="fmax")

        self.inter_delays = np.zeros((self.ndm, self.n_subbands), dtype=np.int32)
        for ifine, dm_f in enumerate(self.trial_dms):
            self.inter_delays[ifine] = self.get_inter_delays(float(dm_f), align_to="fmax")

        self.max_intra_delay_samps = int(np.max(self.intra_delays)) if self.intra_delays.size > 0 else 0
        self.max_inter_delay_samps = int(np.max(self.inter_delays)) if self.inter_delays.size > 0 else 0

    def get_channel_delays(self, dm_pccc: float, align_to: str = "fmax") -> np.ndarray:
        """
        Computes integer sample delays per channel for a given DM.
        """
        return get_dm_delays(self.freqs_hz, dm_pccc, self.tsamp_s, align_to=align_to)

    def get_intra_delays(self, dm_coarse: float, align_to: str = "fmax") -> np.ndarray:
        """
        Compute causal intra-subband channel delays (samples) relative to each
        subband's upper edge frequency ``nu_{s, max}`` (when ``align_to='fmax'``).
        """
        delays = np.zeros(self.nchan, dtype=np.int32)
        for s in range(self.n_subbands):
            c0 = int(self.subband_offsets[s])
            c1 = int(self.subband_offsets[s + 1])
            if c1 > c0:
                delays[c0:c1] = get_dm_delays(
                    self.freqs_hz[c0:c1], float(dm_coarse), self.tsamp_s, align_to=align_to
                ).astype(np.int32)
        return delays

    def get_inter_delays(self, dm_fine: float, align_to: str = "fmax") -> np.ndarray:
        """
        Compute causal inter-subband delays (samples) for each subband's upper
        edge frequency ``nu_{s, max}`` relative to the band reference frequency
        ``nu_ref`` (``nu_max`` when ``align_to='fmax'``).
        """
        if self.n_subbands == 0:
            return np.empty(0, dtype=np.int32)
        return get_dm_delays(
            self.subband_max_freqs_hz, float(dm_fine), self.tsamp_s, align_to=align_to
        ).astype(np.int32)

    def summary(self) -> str:
        base = (
            f"DedispersionPlan Summary:\n"
            f"  Mode: {self.dedisp_mode}\n"
            f"  Frequencies: {self.fmin_ghz*1e3:.2f} - {self.fmax_ghz*1e3:.2f} MHz ({self.nchan} channels)\n"
            f"  Time Resolution: {self.tsamp_s*1e3:.3f} ms\n"
            f"  DM Range: {self.trial_dms[0]:.2f} - {self.trial_dms[-1]:.2f} pc/cm^3\n"
            f"  DM Step (delta_DM): {self.dm_step:.3f} pc/cm^3 (Total trials: {self.ndm})\n"
            f"  Max Delay Sweep: {self.max_delay_samps} samples ({self.max_delay_samps * self.tsamp_s:.3f} s)"
        )
        if self.dedisp_mode == "subbanded":
            base += (
                f"\n  Subbands: {self.n_subbands} | Coarse DM Step: {self.coarse_dm_step:.3f} pc/cm^3 "
                f"(Coarse trials: {self.n_coarse})"
            )
        return base


class BruteForceDedisperser:
    """
    Applies brute-force channel shifting on visibility blocks (nbl, nchan, nt)
    for a specific DM trial, managing history buffers or edge boundaries across chunks.
    """
    def __init__(self, plan: DedispersionPlan, dm_pccc: float, align_to: str = "fmax"):
        self.plan = plan
        self.dm_pccc = float(dm_pccc)
        self.align_to = align_to
        self.delays = plan.get_channel_delays(self.dm_pccc, align_to=align_to)
        self.max_delay = int(np.max(self.delays)) if len(self.delays) > 0 else 0
        self.nchan = len(self.delays)
        self.history: Optional[np.ndarray] = None
        self.wgt_history: Optional[np.ndarray] = None

    def reset(self):
        """Resets the history buffer for a new observation/scan."""
        self.history = None
        self.wgt_history = None

    def dedisperse_chunk(
        self,
        vis_chunk: np.ndarray,
        wgt_chunk: Optional[np.ndarray] = None,
        iblock: int = 0,
        collapse_channels: bool = True,
        nsubbands: int = 1,
        use_history: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Dedisperses a chunk of visibilities.

        Parameters
        ----------
        vis_chunk : np.ndarray (complex64)
            Input visibility block with shape (nbl, nchan, nt).
        wgt_chunk : np.ndarray, optional
            Input weight block with shape (nbl, nchan, nt). If None, uniform 1.0 weights are assumed.
        iblock : int
            Block index (0 for initial chunk).
        collapse_channels : bool
            If True, collapses channels into 1 (or nsubbands) frequencies.
        nsubbands : int
            Number of subbands when collapsing (default: 1 for full collapse).
        use_history : bool
            If True, maintains an internal history buffer across sequential chunks.
            If False (default for independent chunks with overlap), shifts data and zeroes
            out unshifted edge samples to avoid wrap-around contamination.

        Returns
        -------
        vis_out : np.ndarray
            Dedispersed visibilities: (nbl, nsubbands, nt) if collapsed, else (nbl, nchan, nt).
        wgt_out : np.ndarray
            Dedispersed weights with same spatial/time shape.
        freq_out : np.ndarray
            Output channel frequencies in Hz: length nsubbands if collapsed, else nchan.
        """
        nbl, nchan, nt = vis_chunk.shape
        assert nchan == self.nchan, f"Expected {self.nchan} channels, got {nchan}"

        if wgt_chunk is None:
            wgt_chunk = np.ones((nbl, nchan, nt), dtype=np.float32)

        # Trivial case: DM = 0
        if self.max_delay == 0 or self.dm_pccc == 0.0:
            if collapse_channels:
                v_col, w_col, f_col = collapse_channel_axis(vis_chunk, wgt_chunk, self.plan.freqs_hz, nsubbands=nsubbands)
                return v_col, w_col, f_col
            return vis_chunk, wgt_chunk, self.plan.freqs_hz

        if use_history:
            # Streamed history buffer mode (Section 4.2 of spec)
            if iblock == 0 or self.history is None:
                self.history = np.zeros((nbl, nchan, self.max_delay), dtype=vis_chunk.dtype)
                self.wgt_history = np.zeros((nbl, nchan, self.max_delay), dtype=wgt_chunk.dtype)

            concatenated_vis = np.concatenate([self.history, vis_chunk], axis=-1)
            concatenated_wgt = np.concatenate([self.wgt_history, wgt_chunk], axis=-1)

            rolled_vis = np.empty_like(concatenated_vis)
            rolled_wgt = np.empty_like(concatenated_wgt)

            for ichan in range(nchan):
                d = int(self.delays[ichan])
                rolled_vis[:, ichan, :] = np.roll(concatenated_vis[:, ichan, :], -d, axis=-1)
                rolled_wgt[:, ichan, :] = np.roll(concatenated_wgt[:, ichan, :], -d, axis=-1)

            # Update history buffer
            self.history = concatenated_vis[..., -self.max_delay:]
            self.wgt_history = concatenated_wgt[..., -self.max_delay:]

            valid_vis = rolled_vis[..., :nt]
            valid_wgt = rolled_wgt[..., :nt]
        else:
            # Chunk-with-overlap mode (standard fastducc dask chunking)
            # Channel shift backward by d. The tail d samples lack future data and are zeroed/masked.
            valid_vis = np.zeros_like(vis_chunk)
            valid_wgt = np.zeros_like(wgt_chunk)

            for ichan in range(nchan):
                d = int(self.delays[ichan])
                if d == 0:
                    valid_vis[:, ichan, :] = vis_chunk[:, ichan, :]
                    valid_wgt[:, ichan, :] = wgt_chunk[:, ichan, :]
                elif d < nt:
                    valid_vis[:, ichan, :-d] = vis_chunk[:, ichan, d:]
                    valid_wgt[:, ichan, :-d] = wgt_chunk[:, ichan, d:]
                    # Trailing d samples remain 0 (weight=0)
                else:
                    # Delay exceeds entire chunk length
                    valid_vis[:, ichan, :] = 0.0
                    valid_wgt[:, ichan, :] = 0.0

        if collapse_channels:
            v_col, w_col, f_col = collapse_channel_axis(valid_vis, valid_wgt, self.plan.freqs_hz, nsubbands=nsubbands)
            return v_col, w_col, f_col

        return valid_vis, valid_wgt, self.plan.freqs_hz


def collapse_channel_axis(
    vis: np.ndarray,
    wgt: np.ndarray,
    freqs_hz: np.ndarray,
    nsubbands: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collapse/average the frequency channel axis (axis 1) of visibilities and weights.

    Parameters
    ----------
    vis : np.ndarray
        Array of shape (nbl, nchan, nt).
    wgt : np.ndarray
        Array of shape (nbl, nchan, nt).
    freqs_hz : np.ndarray
        Channel frequencies of shape (nchan,).
    nsubbands : int
        Number of output subbands (default 1). If 1, averages all channels.

    Returns
    -------
    vis_out : np.ndarray
        Collapsed visibilities of shape (nbl, nsubbands, nt).
    wgt_out : np.ndarray
        Collapsed weights of shape (nbl, nsubbands, nt).
    subband_freqs : np.ndarray
        Frequencies for each subband of length nsubbands.
    """
    nbl, nchan, nt = vis.shape
    nsubbands = max(1, min(nsubbands, nchan))

    if nsubbands == 1:
        # Full collapse to 1 channel
        wsum = np.sum(wgt, axis=1, keepdims=True)  # (nbl, 1, nt)
        with np.errstate(invalid='ignore', divide='ignore'):
            vsum = np.sum(vis * wgt, axis=1, keepdims=True) / np.where(wsum > 0, wsum, np.nan)
        v_collapsed = np.nan_to_num(vsum, nan=0.0)
        w_collapsed = wsum
        f_collapsed = np.array([float(np.mean(freqs_hz))], dtype=np.float64)
        return v_collapsed, w_collapsed, f_collapsed

    # Multi-subband collapse
    chan_splits = np.array_split(np.arange(nchan), nsubbands)
    v_out = np.zeros((nbl, nsubbands, nt), dtype=vis.dtype)
    w_out = np.zeros((nbl, nsubbands, nt), dtype=wgt.dtype)
    f_out = np.zeros(nsubbands, dtype=np.float64)

    for isub, chans in enumerate(chan_splits):
        v_sub = vis[:, chans, :]
        w_sub = wgt[:, chans, :]
        wsum = np.sum(w_sub, axis=1)  # (nbl, nt)
        with np.errstate(invalid='ignore', divide='ignore'):
            vsum = np.sum(v_sub * w_sub, axis=1) / np.where(wsum > 0, wsum, np.nan)
        v_out[:, isub, :] = np.nan_to_num(vsum, nan=0.0)
        w_out[:, isub, :] = wsum
        f_out[isub] = float(np.mean(freqs_hz[chans]))

    return v_out, w_out, f_out
