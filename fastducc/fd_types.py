from __future__ import annotations
from dataclasses import dataclass
from typing import List, Any, Optional
import numpy as np

@dataclass
class Config:
    msname: str
    npix_x: int
    npix_y: int
    pix_rad: float
    ra0_rad: float
    dec0_rad: float
    epsilon: float
    do_wgridding: bool
    nthreads: int
    verbosity: int
    corr_mode: str
    basis: str
    single_pol: str
    data_column: str
    enable_var: bool
    do_var_search: bool
    enable_var_chunk: bool
    enable_var_scan: bool
    enable_var_obs: bool
    enable_boxcar: bool
    do_boxcar_search: bool
    var_threshold: float
    var_keep_k: Optional[int]
    rms_clip_sigma: float
    nms_radius: int
    boxcar_widths: List[int]
    boxcar_threshold: float
    do_plot: bool
    plot_cands_only: bool
    ms_base: str
    candidates_dir: str
    chunk_prefix_root: Any
    all_prefix_root: str
    save_var_lightcurves: bool
    save_full_var_lightcurves: bool
    save_var_snippets: bool
    save_box_lightcurves: bool
    save_box_snippets: bool
    continuum_dir: Optional[str] | None
    var_highpass_cutoff_sec: float
    use_local_threshold: bool = True
    local_window_size: int = 64
    current_dm: float = 0.0
    dm_trials: Optional[List[float]] = None
    collapse_channels: bool = False
    nsubbands: int = 1
    exact_uvw: bool = True
    scan_id: Optional[str] = None
    is_dm_search: bool = False
    min_valid_channels: int = 24
    filter_ips_agn: bool = True
    ips_snr_ratio_threshold: float = 0.8
    ips_match_radius_arcsec: float = 30.0

@dataclass
class WelfordState:
    count: np.ndarray
    mean:  np.ndarray
    M2:    np.ndarray
    ema_mean: np.ndarray
    last_time: float


@dataclass
class CubeConfig:
    msname: str
    out_cube: Optional[str] = None
    coord: Optional[Any] = None
    ra: Optional[Any] = None
    dec: Optional[Any] = None
    radius: Optional[Any] = None
    radius_arcsec: Optional[float] = None
    radius_pix: Optional[float] = None
    box_size_pix: Optional[int] = None
    box_size_arcsec: Optional[float] = None
    radius_unit: str = "arcsec"
    start_time_idx: Optional[int] = None
    end_time_idx: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    chunk_size: int = 512
    time_step: Optional[float] = None
    time_tolerance: float = 1e-3
    max_blank_factor: float = 10.0
    write_timestamps_table: bool = True
    save_uv: bool = False
    uv_only: bool = False
    uv_mode: str = "auto"
    uv_format: str = "fits"
    uv_hermitian_half: bool = False
    uv_zarr_path: Optional[str] = None
    uv_threshold: float = 1e-4
    out_uv_real: Optional[str] = None
    out_uv_imag: Optional[str] = None
    uv_4d: bool = False
    npix_x: int = 384
    npix_y: int = 384
    pixsize_arcsec: float = 22.0
    corr_mode: str = "single"
    single_pol: str = "XX"
    basis: str = "linear"
    data_column: str = "DATA"
    field_name: Optional[str] = None
    ra_sign: int = -1
    dec_sign: int = -1
    epsilon: float = 1e-6
    do_wgridding: bool = True
    nthreads: int = 0
    verbosity: int = 0
    flip_u: bool = False
    flip_v: bool = False
    flip_w: bool = False
    divide_by_n: bool = True
    sigma_min: float = 1.1
    sigma_max: float = 2.6
    min_valid_channels: int = 24
    dm: float = 0.0
    collapse_channels: bool = False
    nsubbands: int = 1
    exact_uvw: bool = True
    align_to: str = "fmax"
    overwrite: bool = True
    parallel_mode: Optional[str] = None
    max_workers: Optional[int] = None
    threads_per_worker: int = 1
    dask_scheduler: str = "processes"
    slurm_partition: Optional[str] = None
    slurm_account: Optional[str] = None
    slurm_cores_per_worker: int = 1
    slurm_mem: str = "32GB"
    slurm_walltime: str = "02:30:00"
    slurm_job_extra: Optional[List[str]] = None
    slurm_interface: Optional[str] = None
    use_fitscube: bool = False
    create_blanks: bool = False
    floating: int = 32
    regularize_time: bool = False
    sync_tolerance: float = 0.05
    shift_method: str = "fft"
    shift_domain: str = "native"

    @classmethod
    def from_namespace(cls, args: Any, **overrides) -> CubeConfig:
        import dataclasses
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        kwargs = {}
        for f in valid_fields:
            if f in overrides:
                kwargs[f] = overrides[f]
            elif hasattr(args, f):
                kwargs[f] = getattr(args, f)
        for k, v in overrides.items():
            if k in valid_fields:
                kwargs[k] = v
        return cls(**kwargs)
