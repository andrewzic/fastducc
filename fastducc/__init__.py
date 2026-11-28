# fastducc/__init__.py


"""
fastducc: tools for imaging and transient detection with ducc0 and casacore.

Submodules:
- fastducc_run: CLI / orchestration to image MS in time chunks and run detection
- cube: utility to write dirty image FITS cubes using fitscube
- imaging: per-chunk dirty image generation (ducc0 wgridder)
- detection: boxcar/SNR maps
- dedisp: brute-force dedispersion utility
- kernels: numba-accelerated primitives (moving sums, MAD/RMS, temporal SNR, max-filter)
- filters: candidate filtering utils including  NMS, cross-width grouping
- ms_utils: Measurement Set helpers (corr labels, spectral window, times, field centres)
- wcs: sky projection/WCS utilities
- candidates: snippet extraction & packaging of candidate cutouts
- ducc_continuum_img: auxiliary continuum imaging tool
- types: type definitions for dataclasses
- gpu_raster: fused CUDA visibility staging, dedispersion, and direct Hermitian rasterisation
- gpu_direct_cube: double-buffered pinned-host MS streaming to consolidated ragged CSR stores
- gpu_nufft: zero-copy GPU CSR bridge and Stage 2/3/4 NUFFT hand-off
"""

__all__ = [
    "fastducc_run",
    "catalogues",
    "core",
    "cli",
    "constants",
    "cube",
    "dedisp",
    "imaging",
    "detection",
    "kernels",
    "filters",
    "ms_utils",
    "wcs",
    "candidates",
    "candidate_store",
    "fd_types",
    "timeshift",
    "baseline_ffa",
    "wgridded_nufft",
    "dask_utils",
    "dynspec",
    "prune",
    "ragged",
    "ragged_gpu",
    "gpu_raster",
    "gpu_c2r",
    "gpu_detection",
    "gpu_transient",
    "gpu_direct_cube",
    "gpu_nufft",
]

__version__ = "0.1.0"
