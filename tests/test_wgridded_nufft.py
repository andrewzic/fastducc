# tests/test_wgridded_nufft.py
import math
import numpy as np
import pytest

try:
    import ducc0
    HAS_DUCC = True
except ImportError:
    HAS_DUCC = False

from fastducc.wgridded_nufft import (
    compute_w_plane_geometry,
    kb_kernel_eval,
    compute_nm1_screen,
    prepare_visibilities_and_coords,
    filter_and_weight_w_plane,
    accumulate_w_screen_inplace,
    wstacked_3d_nufft,
    wstacked_3d_nufft_block,
    snapshot_wgridder_fft,
)


def test_geometry_and_kernel():
    npix = 128
    pixsize = 30.0 * (math.pi / (180.0 * 3600.0))
    dw, nplanes, w0 = compute_w_plane_geometry(npix, pixsize, -2000.0, 2000.0, supp_w=4)
    assert dw > 0
    assert nplanes >= 4
    assert w0 < 0

    x_dist = np.array([-1.0, 0.0, 1.0, 3.0])
    k_vals = kb_kernel_eval(x_dist, supp=4)
    assert k_vals[1] == pytest.approx(1.0, rel=1e-5)
    assert k_vals[0] > 0
    assert k_vals[3] == 0.0


def test_nm1_screen():
    npix = 64
    pixsize = 45.0 * (math.pi / (180.0 * 3600.0))
    screen = compute_nm1_screen(npix, pixsize)
    assert screen.shape == (npix, npix)
    # Mode 0 is index 0 -> nm1 should be 0.0 at (0, 0)
    assert screen[0, 0] == pytest.approx(0.0, abs=1e-12)
    # At non-zero indices, nm1 < 0
    assert screen[10, 10] < 0.0


def test_numba_prepare_and_filter():
    nrows = 100
    nchan = 8
    uvw_m = np.random.uniform(-500, 500, size=(nrows, 3))
    times = np.linspace(0, 10, nrows)
    vis = (np.random.normal(size=(nrows, nchan)) + 1j * np.random.normal(size=(nrows, nchan))).astype(np.complex64)
    flags = np.zeros((nrows, nchan), dtype=bool)
    flags[0, 0] = True  # flag 1 vis
    freqs_hz = np.linspace(1.2e9, 1.4e9, nchan)
    tau_dm = np.zeros(nchan)

    pts, coords, w_lam = prepare_visibilities_and_coords(
        uvw_m, times, vis, flags, freqs_hz, tau_dm,
        dl=1e-4, df=0.1, c_light=299792458.0
    )
    assert len(pts) == nrows * nchan - 1
    assert coords.shape == (len(pts), 3)
    assert len(w_lam) == len(pts)

    # Filter
    p_pts, p_coords = filter_and_weight_w_plane(
        pts, coords, w_lam, w_curr=0.0, dw=500.0, supp_w=4.0, beta=2.3*4.0
    )
    assert len(p_pts) <= len(pts)
    assert p_coords.shape == (len(p_pts), 3)


def test_numba_accumulation_matches_numpy():
    npix = 32
    nkeep = 16
    Nf = 64
    accum = np.zeros((npix, npix, nkeep), dtype=np.complex64)
    buf = np.random.normal(size=(npix, npix, Nf)).astype(np.complex64)
    screen = np.random.normal(size=(npix, npix)) * 1e-4
    w_curr = 2000.0
    kmin = 4

    accumulate_w_screen_inplace(accum, buf, screen, w_curr, kmin, nkeep)

    # Reference NumPy
    w_scr_np = np.exp(2j * math.pi * w_curr * screen).astype(np.complex64)
    ref = buf[:, :, kmin : kmin + nkeep] * w_scr_np[:, :, None]

    assert np.allclose(accum, ref, atol=1e-6)


@pytest.mark.skipif(not HAS_DUCC, reason="ducc0 not installed")
def test_wstacked_3d_nufft_execution():
    np.random.seed(42)
    npix = 64
    pixsize = 45.0 * (math.pi / (180.0 * 3600.0))
    nt = 64
    dt = 0.1
    t_sec = np.repeat(np.arange(nt) * dt, 10)
    nvis = len(t_sec)

    u_lam = np.random.uniform(-1000, 1000, size=nvis)
    v_lam = np.random.uniform(-1000, 1000, size=nvis)
    w_lam = np.random.uniform(-500, 500, size=nvis)
    vis = (np.random.normal(size=nvis) + 1j * np.random.normal(size=nvis)).astype(np.complex64)

    cube = wstacked_3d_nufft(
        u_lam=u_lam,
        v_lam=v_lam,
        w_lam=w_lam,
        t_sec=t_sec,
        vis=vis,
        npix=npix,
        pixsize_rad=pixsize,
        Nf=nt,
        dt=dt,
        epsilon=1e-5,
    )
    assert cube.shape == (npix, npix, nt // 2)
    assert np.all(np.isfinite(cube))


@pytest.mark.skipif(not HAS_DUCC, reason="ducc0 not installed")
def test_wstacked_3d_nufft_block_execution():
    np.random.seed(42)
    nrows = 50
    nchan = 4
    npix = 64
    pixsize = 45.0 * (math.pi / (180.0 * 3600.0))
    Nf = 64
    dt = 0.1
    df = 1.0 / (Nf * dt)

    uvw_m = np.random.uniform(-500, 500, size=(nrows, 3))
    times = np.linspace(0, 5, nrows)
    vis = (np.random.normal(size=(nrows, nchan)) + 1j * np.random.normal(size=(nrows, nchan))).astype(np.complex64)
    flags = np.zeros((nrows, nchan), dtype=bool)
    freqs_hz = np.linspace(1.2e9, 1.4e9, nchan)
    tau_dm = np.zeros(nchan)

    cube = wstacked_3d_nufft_block(
        uvw_m=uvw_m,
        times=times,
        vis=vis,
        flags=flags,
        freqs_hz=freqs_hz,
        tau_dm=tau_dm,
        npix=npix,
        pixsize_rad=pixsize,
        Nf=Nf,
        dt=dt,
        df=df,
        kmin=2,
        nkeep=20,
        supp_w=4,
        ofactor=1.5,
        epsilon=1e-5,
        nthreads=1,
    )
    assert cube.shape == (npix, npix, 20)
    assert np.all(np.isfinite(cube))
