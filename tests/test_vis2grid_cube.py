import unittest
from unittest.mock import patch
import numpy as np

from fastducc import imaging
from fastducc.fd_types import CubeConfig
from fastducc.cube import build_cli_cube, process_chunk_task


@unittest.skipUnless(imaging.ducc0 is not None, "ducc0 is required for vis2grid tests")
class TestVis2GridCube(unittest.TestCase):
    def test_get_uv_grid_dims(self):
        nu, nv = imaging.get_uv_grid_dims(
            npix_x=384, npix_y=256, pixsize_x=1e-4, pixsize_y=1e-4,
            epsilon=1e-5, do_wgridding=True
        )
        self.assertGreater(nu, 384)
        self.assertGreater(nv, 256)
        self.assertGreater(nu, nv)

    def test_grid_chunk_uv_cube_basic(self):
        nbl, nchan, nt = 12, 16, 4
        npix_x, npix_y = 64, 64
        pix_rad = 1e-4
        chan_freq = np.linspace(1.2e9, 1.4e9, nchan)
        dt = 0.1

        # Synthesize random visibilities and UVW
        np.random.seed(42)
        vis_3d = (np.random.randn(nbl, nchan, nt) + 1j * np.random.randn(nbl, nchan, nt)).astype(np.complex64)
        wgt_3d = np.ones((nbl, nchan, nt), dtype=np.float32)
        uvw_3d = np.random.randn(nbl, 3, nt) * 500.0

        uv_cube = imaging.grid_chunk_uv_cube(
            vis_3d=vis_3d,
            wgt_3d=wgt_3d,
            uvw_3d=uvw_3d,
            chan_freq=chan_freq,
            dt=dt,
            dm=0.0,
            npix_x=npix_x,
            npix_y=npix_y,
            pixsize_x=pix_rad,
            pixsize_y=pix_rad,
            epsilon=1e-4,
            do_wgridding=True,
        )

        nu, nv = imaging.get_uv_grid_dims(
            npix_x=npix_x, npix_y=npix_y, pixsize_x=pix_rad, pixsize_y=pix_rad,
            epsilon=1e-4, do_wgridding=True
        )
        self.assertEqual(uv_cube.shape, (nt, nv, nu))
        self.assertEqual(uv_cube.dtype, np.complex64)
        self.assertTrue(np.any(np.abs(uv_cube) > 0.0))

    def test_grid_chunk_uv_cube_dedispersion_and_exact_uvw(self):
        nbl, nchan, nt = 8, 32, 5
        npix_x, npix_y = 64, 64
        pix_rad = 1e-4
        chan_freq = np.linspace(1.2e9, 1.4e9, nchan)
        dt = 0.05

        vis_3d = (np.random.randn(nbl, nchan, nt) + 1j * np.random.randn(nbl, nchan, nt)).astype(np.complex64)
        wgt_3d = np.ones((nbl, nchan, nt), dtype=np.float32)
        uvw_3d = np.random.randn(nbl, 3, nt) * 500.0

        uv_cube = imaging.grid_chunk_uv_cube(
            vis_3d=vis_3d,
            wgt_3d=wgt_3d,
            uvw_3d=uvw_3d,
            chan_freq=chan_freq,
            dt=dt,
            dm=50.0,
            exact_uvw=True,
            npix_x=npix_x,
            npix_y=npix_y,
            pixsize_x=pix_rad,
            pixsize_y=pix_rad,
            epsilon=1e-4,
            do_wgridding=True,
        )

        nu, nv = imaging.get_uv_grid_dims(
            npix_x=npix_x, npix_y=npix_y, pixsize_x=pix_rad, pixsize_y=pix_rad,
            epsilon=1e-4, do_wgridding=True
        )
        self.assertEqual(uv_cube.shape, (nt, nv, nu))
        self.assertTrue(np.any(np.abs(uv_cube) > 0.0))

    def test_cli_and_config_uv_mode(self):
        # Explicit vis2grid
        args = build_cli_cube([
            "--msname", "test.ms",
            "--save-uv",
            "--uv-mode", "vis2grid",
        ])
        self.assertTrue(args.save_uv)
        self.assertEqual(args.uv_mode, "vis2grid")

        cfg = CubeConfig.from_namespace(args)
        self.assertTrue(cfg.save_uv)
        self.assertEqual(cfg.uv_mode, "vis2grid")

        # Default auto with --uv-only -> resolves to vis2grid
        args_uv_only = build_cli_cube([
            "--msname", "test.ms",
            "--uv-only",
        ])
        self.assertTrue(args_uv_only.uv_only)
        self.assertEqual(args_uv_only.uv_mode, "auto")
        cfg_uv_only = CubeConfig.from_namespace(args_uv_only)
        self.assertEqual(cfg_uv_only.uv_mode, "auto")

        # Default auto with --save-uv (imaging + save-uv) -> resolves to fft
        args_save_uv = build_cli_cube([
            "--msname", "test.ms",
            "--save-uv",
        ])
        self.assertTrue(args_save_uv.save_uv)
        self.assertFalse(args_save_uv.uv_only)
        self.assertEqual(args_save_uv.uv_mode, "auto")

    def test_make_dirty_cube_auto_resolution(self):
        # Verify make_dirty_cube resolves uv_mode="auto" properly
        from fastducc.cube import make_dirty_cube
        with patch("fastducc.cube.init_empty_fits_cube", return_value=0), \
             patch("fastducc.cube.process_chunk_task", return_value=(0, 1, 1, None)) as mock_task, \
             patch("fastducc.cube.table") as mock_tab, \
             patch("os.path.exists", return_value=True), \
             patch("fastducc.imaging.get_uv_grid_dims", return_value=(96, 96)), \
             patch("fastducc.ms_utils.get_phase_center", return_value=(0.0, 0.0, "field0")):

            mock_tab_instance = mock_tab.return_value
            mock_tab_instance.colnames.return_value = ["TIME", "SCAN_NUMBER"]
            mock_tab_instance.getcol.side_effect = lambda col: np.array([0.0]) if col == "TIME" else np.array([1])

            # 1. uv_only -> should resolve to vis2grid
            make_dirty_cube(
                msname="mock.ms",
                uv_only=True,
                overwrite=True,
                write_timestamps_table=False,
            )
            self.assertEqual(mock_task.call_args[1]["uv_mode"], "vis2grid")

            # 2. imaging + save_uv -> should resolve to fft
            make_dirty_cube(
                msname="mock.ms",
                save_uv=True,
                uv_only=False,
                overwrite=True,
                write_timestamps_table=False,
            )
            self.assertEqual(mock_task.call_args[1]["uv_mode"], "fft")

            # 3. explicit override with uv_only -> should remain fft
            make_dirty_cube(
                msname="mock.ms",
                uv_only=True,
                uv_mode="fft",
                overwrite=True,
                write_timestamps_table=False,
            )
            self.assertEqual(mock_task.call_args[1]["uv_mode"], "fft")

            # 4. explicit override with save_uv -> should remain vis2grid
            make_dirty_cube(
                msname="mock.ms",
                save_uv=True,
                uv_only=False,
                uv_mode="vis2grid",
                overwrite=True,
                write_timestamps_table=False,
            )
            self.assertEqual(mock_task.call_args[1]["uv_mode"], "vis2grid")

    def test_process_chunk_task_vis2grid_direct(self):
        nbl, nchan, nt = 6, 8, 3
        npix_x, npix_y = 48, 48
        pix_rad = 1e-4
        times = np.array([0.0, 0.1, 0.2])
        chan_freq = np.linspace(1.3e9, 1.4e9, nchan)
        dt = 0.1

        vis_3d = (np.random.randn(nbl, nchan, nt) + 1j * np.random.randn(nbl, nchan, nt)).astype(np.complex64)
        wgt_3d = np.ones((nbl, nchan, nt), dtype=np.float32)
        uvw_3d = np.random.randn(nbl, 3, nt) * 300.0

        with patch("fastducc.imaging.load_chunk_data", return_value=(times, chan_freq, dt, vis_3d, wgt_3d, uvw_3d)):
            res = process_chunk_task(
                msname="dummy.ms",
                out_cube=None,
                chunk_times=times,
                offset_start=0,
                offset_end=nt,
                npix_x=npix_x,
                npix_y=npix_y,
                pix_rad=pix_rad,
                save_uv=True,
                uv_mode="vis2grid",
                write_dirty=False,
                direct_write=False,
                epsilon=1e-4,
                do_wgridding=True,
            )
            offset_start, offset_end, n_samples, arrays = res
            self.assertEqual(offset_start, 0)
            self.assertEqual(offset_end, nt)
            self.assertEqual(n_samples, nt)

            ret_dirty, ret_real, ret_imag = arrays
            self.assertIsNone(ret_dirty)
            self.assertIsNotNone(ret_real)
            self.assertIsNotNone(ret_imag)

            nu, nv = imaging.get_uv_grid_dims(
                npix_x=npix_x, npix_y=npix_y, pixsize_x=pix_rad, pixsize_y=pix_rad,
                epsilon=1e-4, do_wgridding=True
            )
            self.assertEqual(ret_real.shape, (nt, nv, nu))
            self.assertEqual(ret_imag.shape, (nt, nv, nu))

    def test_grid_chunk_to_zarr_task(self):
        import tempfile
        import shutil
        import zarr
        from fastducc.cube import grid_chunk_to_zarr_task

        tmpdir = tempfile.mkdtemp()
        zarr_path = f"{tmpdir}/test_uv.zarr"
        try:
            nbl, nchan, nt = 4, 8, 3
            npix_x, npix_y = 32, 32
            pix_rad = 1e-4
            times = np.array([0.0, 0.1, 0.2])
            chan_freq = np.linspace(1.3e9, 1.4e9, nchan)
            dt = 0.1

            vis_3d = (np.random.randn(nbl, nchan, nt) + 1j * np.random.randn(nbl, nchan, nt)).astype(np.complex64)
            wgt_3d = np.ones((nbl, nchan, nt), dtype=np.float32)
            uvw_3d = np.random.randn(nbl, 3, nt) * 200.0

            nu, nv = imaging.get_uv_grid_dims(
                npix_x=npix_x, npix_y=npix_y, pixsize_x=pix_rad, pixsize_y=pix_rad,
                epsilon=1e-4, do_wgridding=True
            )
            nu_stored = nu // 2

            z_arr = zarr.open_array(
                zarr_path, mode="w", shape=(10, nv, nu_stored),
                chunks=(5, nv, nu_stored), dtype=np.complex64
            )

            with patch("fastducc.imaging.load_chunk_data", return_value=(times, chan_freq, dt, vis_3d, wgt_3d, uvw_3d)):
                res = grid_chunk_to_zarr_task(
                    msname="dummy.ms",
                    out_zarr=zarr_path,
                    chunk_times=times,
                    offset_start=2,
                    offset_end=5,
                    npix_x=npix_x,
                    npix_y=npix_y,
                    pix_rad=pix_rad,
                    grid_indices=np.array([2, 3, 4]),
                    uv_hermitian_half=True,
                    epsilon=1e-4,
                    do_wgridding=True,
                )
                off_s, off_e, n_done = res
                self.assertEqual(off_s, 2)
                self.assertEqual(off_e, 5)
                self.assertEqual(n_done, nt)

            z_read = zarr.open_array(zarr_path, mode="r")
            self.assertTrue(np.any(np.abs(z_read[2:5]) > 0.0))
            self.assertTrue(np.all(np.abs(z_read[0:2]) == 0.0))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_make_dirty_cube_out_zarr_routing_and_client_reuse(self):
        import tempfile
        import shutil
        import zarr
        from unittest.mock import MagicMock
        from fastducc.cube import make_dirty_cube

        tmpdir = tempfile.mkdtemp()
        zarr_path = f"{tmpdir}/test_direct.zarr"
        try:
            mock_client = MagicMock()
            mock_fut = MagicMock()
            mock_fut.result.return_value = (0, 2, 2)
            mock_client.submit.return_value = mock_fut

            with patch("fastducc.cube.table") as mock_tab, \
                 patch("os.path.exists", return_value=True), \
                 patch("fastducc.imaging.get_uv_grid_dims", return_value=(48, 48)), \
                 patch("fastducc.ms_utils.get_phase_center", return_value=(0.0, 0.0, "field0")), \
                 patch("dask.distributed.as_completed", return_value=[mock_fut]):

                mock_tab_inst = mock_tab.return_value
                mock_tab_inst.colnames.return_value = ["TIME", "SCAN_NUMBER"]
                mock_tab_inst.getcol.side_effect = lambda col: np.array([0.0, 0.1]) if col == "TIME" else np.array([1, 1])

                out = make_dirty_cube(
                    msname="mock.ms",
                    out_zarr=zarr_path,
                    client=mock_client,
                    uv_hermitian_half=True,
                    chunk_size=2,
                    npix_x=32,
                    npix_y=32,
                )
                self.assertEqual(out, zarr_path)
                self.assertTrue(mock_client.submit.called)
                # Verify external client was NOT closed
                self.assertFalse(mock_client.close.called)

                # Verify Zarr metadata
                z_check = zarr.open_array(zarr_path, mode="r")
                self.assertEqual(z_check.shape, (2, 48, 24))
                self.assertTrue(z_check.attrs["hermitian_half"])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_make_dirty_cube_hermitian_half_fits_dimension_match(self):
        import os
        import tempfile
        import shutil
        from astropy.io import fits
        from fastducc.cube import make_dirty_cube

        tmpdir = tempfile.mkdtemp()
        fits_real = f"{tmpdir}/test_real.fits"
        fits_imag = f"{tmpdir}/test_imag.fits"
        try:
            times = np.array([0.0, 0.1])
            chan_freq = np.linspace(1.2e9, 1.4e9, 8)
            dt = 0.1
            nbl = 4
            nt = len(times)
            vis_3d = (np.random.randn(nbl, 8, nt) + 1j * np.random.randn(nbl, 8, nt)).astype(np.complex64)
            wgt_3d = np.ones((nbl, 8, nt), dtype=np.float32)
            uvw_3d = np.random.randn(nbl, 3, nt) * 100.0

            orig_exists = os.path.exists
            with patch("fastducc.cube.table") as mock_tab, \
                 patch("os.path.exists", side_effect=lambda p: True if str(p) == "mock.ms" else orig_exists(p)), \
                 patch("fastducc.imaging.get_uv_grid_dims", return_value=(450, 450)), \
                 patch("fastducc.ms_utils.get_phase_center", return_value=(0.0, 0.0, "field0")), \
                 patch("fastducc.imaging.load_chunk_data", return_value=(times, chan_freq, dt, vis_3d, wgt_3d, uvw_3d)), \
                 patch("fastducc.imaging.grid_chunk_uv_cube", return_value=np.zeros((nt, 450, 450), dtype=np.complex64)):

                mock_tab_inst = mock_tab.return_value
                mock_tab_inst.colnames.return_value = ["TIME", "SCAN_NUMBER"]
                mock_tab_inst.getcol.side_effect = lambda col: times if col == "TIME" else np.array([1, 1])

                make_dirty_cube(
                    msname="mock.ms",
                    out_uv_real=fits_real,
                    out_uv_imag=fits_imag,
                    save_uv=True,
                    uv_only=True,
                    uv_mode="vis2grid",
                    uv_hermitian_half=True,
                    parallel_mode="serial",
                    chunk_size=2,
                    npix_x=384,
                    npix_y=384,
                )

                hdr_r = fits.getheader(fits_real)
                self.assertEqual(hdr_r["NAXIS1"], 225)
                self.assertEqual(hdr_r["NAXIS2"], 450)
                data_r = fits.getdata(fits_real)
                self.assertEqual(data_r.shape, (2, 450, 225))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
