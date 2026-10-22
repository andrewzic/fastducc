# fastducc

Fast transient imaging, dedispersion, and detection utilities using DUCC0 (`ducc0.wgridder` / `ducc0.nufft`).

## Installation

```bash
git clone https://github.com/andrewzic/fastducc.git
cd fastducc
pip install -e .
```

## CLI Usage

`fastducc` provides a unified command-line interface with dedicated subcommands:

```bash
fastducc --help
```

```
usage: fastducc [-h] [-v] [--version]

Fast transient imaging, dedispersion, and detection pipeline using DUCC0.

Available Subcommands:
  search (run)          Image MS in time chunks and search for transients (default)
  cube (make-cube)      Generate dirty FITS image cubes with fitscube or UV visibilities
  aggregate             Aggregate per-beam candidate catalogues across an observation/scan
  aggregate_obs         Aggregate cross-beam super summaries across an SBID observation
  periodicity           3D DUCC NUFFT periodicity search on MS visibilities
```

### 1. Transient Search Pipeline (`fastducc search` or `fastducc`)

Run snapshot imaging, dedispersion, and boxcar/variance transient detection in time chunks:

```bash
# Shorthand direct invocation (backward-compatible)
fastducc --msname /path/to/observation.ms --chunk-size 512 --dm 15.0 --enable-boxcar

# Explicit subcommand invocation
fastducc search --msname /path/to/observation.ms --chunk-size 512 --threshold-sigma 8.0
```

### 2. Dirty FITS Cube & UV Generation (`fastducc cube`)

Stream dirty image cubes or UV visibilities directly into FITS without intermediate files:

```bash
# Generate full-field dirty FITS cube
fastducc cube --msname /path/to/observation.ms --out-cube dirty_cube.fits --chunk-size 512

# Generate a targeted cubelet around a celestial position
fastducc cube --msname /path/to/observation.ms --coord "17:33:00 -23:44:00" --radius 60arcsec

# Output gridded UV visibilities (real and imaginary planes)
fastducc cube --msname /path/to/observation.ms --save-uv --uv-only
```

### 3. Candidate Aggregation (`fastducc aggregate` & `fastducc aggregate_obs`)

Aggregate and crossmatch candidate detections across multiple beams and scans:

```bash
# Aggregate beam outputs within a single scan
fastducc aggregate --obs-root /path/to/scan_dir --kind boxcar

# Aggregate super summaries across all scans in an SBID observation
fastducc aggregate_obs --obs-root /path/to/SB77974 --organize
```

### 4. NUFFT Periodicity Search (`fastducc periodicity`)

Perform 3-D DUCC NUFFT $(u,v,t) \rightarrow (l,m,f)$ spin-frequency periodicity search:

```bash
fastducc periodicity --msname /path/to/observation.ms --fmin 0.5 --dm 25.0 --npix 384
```

---

## Programmatic API

Modules can be imported and executed directly in Python:

```python
from fastducc.cube import make_dirty_cube, CubeConfig

# 1. Direct keyword arguments (100% backward compatible)
make_dirty_cube(
    msname="/path/to/observation.ms",
    out_cube="dirty_cube.fits",
    chunk_size=512,
    npix_x=384,
    pixsize_arcsec=22.0,
    dm=0.0,
)

# 2. Structured configuration object
cfg = CubeConfig(
    msname="/path/to/observation.ms",
    out_cube="dirty_cube.fits",
    chunk_size=512,
    dm=45.0,
)
make_dirty_cube(cfg=cfg)
```

## Running Tests

Run the unit and integration test suite:

```bash
python -m unittest discover tests
```
