import os
import numpy as np
import astropy.constants as const
import astropy.units as u

from casacore.tables import table, taql
try:
    import ducc0
except Exception as e:
    raise RuntimeError('ducc0 is required') from e

# CASA POLARIZATION CORR_TYPE integer codes to labels
_CORR_CODE_TO_NAME = {
    5: 'RR', 6: 'RL', 7: 'LR', 8: 'LL',
    9: 'XX', 10: 'XY', 11: 'YX', 12: 'YY',
}

def get_corr_label_indices(msname: str):
    """Return (labels_list, label->index dict) from POLARIZATION/CORR_TYPE."""
    t_pol = table(f"{msname}/POLARIZATION", readonly=True)
    corr_types = t_pol.getcell('CORR_TYPE', 0)
    #print(f"found corr_types {corr_types} in {msname}")
    t_pol.close()
    labels = [ _CORR_CODE_TO_NAME.get(int(c), str(c)) for c in corr_types ]
    lbl2idx = {lab: i for i, lab in enumerate(labels)}
    return labels, lbl2idx


def get_channel_lambdas(ms):
    tf = table(f"{ms}/SPECTRAL_WINDOW")
    # get all channel frequencies and convert that to wavelengths
    channel_freqs = tf[0]["CHAN_FREQ"]
    nchan = len(channel_freqs)
    channel_lambdas = const.c.to(u.m/u.s).value / channel_freqs
    tf.close()
    return nchan, channel_freqs, channel_lambdas

def get_unique_times(msname: str, time_col: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Get sorted unique times and scan numbers efficiently using TaQL.
    """
    t = table(msname, readonly=True)
    has_scan = 'SCAN_NUMBER' in set(t.colnames())
    if has_scan:
        res = taql(f"SELECT DISTINCT {time_col}, SCAN_NUMBER FROM $1 ORDERBY {time_col}", tables=[t])
        unique_times = res.getcol(time_col)
        scan_per_time = res.getcol('SCAN_NUMBER')
        res.close()
    else:
        res = taql(f"SELECT DISTINCT {time_col} FROM $1 ORDERBY {time_col}", tables=[t])
        unique_times = res.getcol(time_col)
        scan_per_time = np.zeros(len(unique_times), dtype=int)
        res.close()
    t.close()
    return unique_times, scan_per_time

def get_scan_aware_chunk_bounds(
    msname: str,
    time_col: str,
    chunk_size: int,
    buffer_overlap_samps: int,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """
    Returns (chunk_bounds, scan_per_time_idx) where:
      - chunk_bounds: list of (start, end) into the sorted unique-time index sequence.
        Chunks never span scan boundaries. Overlap buffer is added within scans only.
      - scan_per_time_idx: int array of shape (total_unique_times,) giving the
        SCAN_NUMBER at each time index.
    """
    unique_times, scan_per_time = get_unique_times(msname, time_col)
    # Find where scan changes → these are hard boundaries
    scan_change = np.where(np.diff(scan_per_time) != 0)[0] + 1  # indices of first sample of new scan
    boundaries = np.concatenate(([0], scan_change, [len(unique_times)]))
    chunk_bounds = []
    for b_start, b_end in zip(boundaries[:-1], boundaries[1:]):
        # Slice of time indices belonging to this scan
        scan_end = b_end - 1   # inclusive last index in this scan
        pos = b_start
        while pos <= scan_end:
            end = min(pos + chunk_size + buffer_overlap_samps - 1, scan_end)
            chunk_bounds.append((pos, end))
            if end == scan_end:
                break
            pos = end + 1 - buffer_overlap_samps
    return chunk_bounds, scan_per_time

def derive_scan_id(scan_numbers: np.ndarray) -> str:
    """
    Given an array of scan numbers for a chunk, return the dominant scan ID as a string.
    Returns '' if no valid scan number is found (e.g. all zeros).
    """
    if len(scan_numbers) == 0:
        return ""
    
    # Filter out zeros
    valid_scans = scan_numbers[scan_numbers != 0]
    if len(valid_scans) == 0:
        return ""
    
    # Return the most common (mode)
    unique_scans, counts = np.unique(valid_scans, return_counts=True)
    return str(unique_scans[np.argmax(counts)])


def get_time(t):
    vis_time = t.getcol('TIME_CENTROID')
    unique_times = np.unique(vis_time)
    assert np.all(np.abs(np.diff(unique_times) - np.diff(unique_times)[0]) < 1e-2)
    nsub = unique_times.shape[0]
    return nsub, vis_time, unique_times

def get_uvwave_indices(t, channel_lambdas, uvcell_size, N_pixels):    
    #  get the uv positions (in lambdas) for each visibility sample
    uvws = t.getcol('UVW') # in units of metres
    #populate uvw_l grid
    uvws_l = np.ones((uvws.shape[0], uvws.shape[1], len(channel_lambdas))) * uvws[:, :, None]
    # duplicate each uv sample into a size such that we can multiply with the channel lambdas    
    #uvs_l = uvws_l[:, :2, :]        # gets rid of w axis. DONT FORGET: need to do w-projection! can't just ignore the w axis like im doing here
    
    chan_tiled = np.ones_like(uvws_l)*channel_lambdas[None, None, :] # arrange and repeat the channel lambdas in an array shape such that we can multiply it onto the uv samples
    
    uvws_l = uvws_l / chan_tiled    # get the uvw positions in lambdas


def get_phase_center(msname: str, field_name: str | None = None):
    """
    Returns (ra0_rad, dec0_rad, selected_field_name).
    If field_name is None, uses the first row in FIELD.
    """
    t_field = table(f"{msname}/FIELD", readonly=True)
    names = t_field.getcol("NAME")          # shape: (nfield,)
    phase_dir = t_field.getcol("PHASE_DIR") # shape: (nfield, 2, n_poly) in radians
    t_field.close()

    idx = 0
    if field_name is not None:
        matches = np.where(names == field_name)[0]
        if matches.size == 0:
            raise ValueError(f"FIELD name '{field_name}' not found in {msname}/FIELD.")
        idx = int(matches[0])

    # Usually polynomial degree n_poly=1. Use the 0th term.
    print(phase_dir[idx])
    ra0_rad  = float(phase_dir[idx][ 0, 0])
    dec0_rad = float(phase_dir[idx][ 0, 1])  #phase_dir[idx]: [[-1.67134852 -0.41158857]]
    return ra0_rad, dec0_rad, str(names[idx])

# --- 3) MS open & chunk count ---
def get_timebin(t_main, time_col: str) -> float:
    """
    Get the timebin of the MS from the INTERVAL column.
    """
    nrows = t_main.nrows()
    if nrows == 0:
        return 1.0
    
    # Read at most 10,000 rows to avoid loading a massive column into memory
    nrow_to_read = min(10000, nrows)
    try:
        intervals = t_main.getcol('INTERVAL', 0, nrow_to_read)
        timebin = float(np.median(intervals))
    except Exception:
        # Fallback to time_col if INTERVAL is somehow missing
        times = t_main.getcol(time_col, 0, nrow_to_read)
        times = np.sort(np.unique(times))
        if len(times) > 1:
            timebin = float(np.median(np.diff(times)))
        else:
            timebin = 1.0
    return timebin


def open_ms(msname: str):
    t_main = table(msname, readonly=True)
    colnames = set(t_main.colnames())
    time_col = 'TIME_CENTROID' if 'TIME_CENTROID' in colnames else 'TIME'
    # Use get_unique_times helper to avoid loading full time column in Python
    unique_times, _ = get_unique_times(msname, time_col)
    total_chunks = len(unique_times)
    return t_main, total_chunks, time_col


def derive_paths(msname: str):
    ms_path = os.path.abspath(msname.rstrip('/'))
    ms_dir  = os.path.dirname(ms_path)
    ms_tag  = os.path.basename(ms_path)
    ms_base = ms_tag[:-3] if ms_tag.endswith('.ms') else ms_tag
    candidates_dir = os.path.join(ms_dir, "candidates")
    os.makedirs(candidates_dir, exist_ok=True)

    def chunk_prefix_root(start_idx: int) -> str:
        return os.path.join(candidates_dir, f"{ms_base}_chunk_{start_idx:06d}")

    all_prefix_root = os.path.join(candidates_dir, f"{ms_base}_all")

    return ms_base, candidates_dir, chunk_prefix_root, all_prefix_root
