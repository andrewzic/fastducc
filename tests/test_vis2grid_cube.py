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


if __name__ == "__main__":
    unittest.main()
