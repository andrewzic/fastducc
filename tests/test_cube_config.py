import unittest
import inspect
import argparse

from fastducc.fd_types import CubeConfig
from fastducc.cube import make_dirty_cube

class TestCubeConfig(unittest.TestCase):
    def test_default_values(self):
        cfg = CubeConfig(msname="sample.ms")
        self.assertEqual(cfg.msname, "sample.ms")
        self.assertIsNone(cfg.out_cube)
        self.assertEqual(cfg.npix_x, 384)
        self.assertEqual(cfg.npix_y, 384)
        self.assertEqual(cfg.pixsize_arcsec, 22.0)
        self.assertEqual(cfg.chunk_size, 512)
        self.assertEqual(cfg.dm, 0.0)
        self.assertTrue(cfg.exact_uvw)
        self.assertTrue(cfg.do_wgridding)
        self.assertEqual(cfg.floating, 32)

    def test_from_namespace(self):
        ns = argparse.Namespace(
            msname="input.ms",
            chunk_size=128,
            dm=45.5,
            npix_x=512,
            npix_y=512,
            unknown_arg="ignored",
        )
        cfg = CubeConfig.from_namespace(ns, out_cube="override.fits", floating=64)
        self.assertEqual(cfg.msname, "input.ms")
        self.assertEqual(cfg.chunk_size, 128)
        self.assertEqual(cfg.dm, 45.5)
        self.assertEqual(cfg.npix_x, 512)
        self.assertEqual(cfg.out_cube, "override.fits")
        self.assertEqual(cfg.floating, 64)
        self.assertEqual(cfg.pixsize_arcsec, 22.0)  # Preserved default

    def test_make_dirty_cube_signature_backward_compatibility(self):
        sig = inspect.signature(make_dirty_cube)
        expected_params = [
            "msname", "out_cube", "cfg", "config", "coord", "ra", "dec",
            "radius", "radius_arcsec", "radius_pix", "box_size_pix",
            "box_size_arcsec", "radius_unit", "start_time_idx", "end_time_idx",
            "start_time", "end_time", "chunk_size", "time_step", "time_tolerance",
            "max_blank_factor", "write_timestamps_table", "save_uv", "uv_only", "uv_mode",
            "uv_format", "uv_hermitian_half", "uv_zarr_path", "uv_threshold",
            "out_uv_real", "out_uv_imag", "uv_4d", "npix_x", "npix_y",
            "pixsize_arcsec", "corr_mode", "single_pol", "basis", "data_column",
            "field_name", "ra_sign", "dec_sign", "epsilon", "do_wgridding",
            "nthreads", "verbosity", "flip_u", "flip_v", "flip_w", "divide_by_n",
            "sigma_min", "sigma_max", "min_valid_channels", "dm", "collapse_channels",
            "nsubbands", "exact_uvw", "align_to", "overwrite", "parallel_mode",
            "max_workers", "threads_per_worker", "dask_scheduler", "slurm_partition",
            "slurm_account", "slurm_cores_per_worker", "slurm_mem", "slurm_walltime",
            "slurm_job_extra", "slurm_interface", "use_fitscube", "create_blanks",
            "floating", "regularize_time", "sync_tolerance", "shift_method", "shift_domain",
        ]
        for param in expected_params:
            self.assertIn(param, sig.parameters, f"Missing parameter {param} in make_dirty_cube signature")

    def test_calling_with_kwargs(self):
        # Calling with nonexistent MS path should raise FileNotFoundError, NOT TypeError
        with self.assertRaises(FileNotFoundError):
            make_dirty_cube(
                msname="/nonexistent/path/dummy.ms",
                out_cube="dummy.fits",
                npix_x=512,
                npix_y=512,
                dm=12.5,
            )

    def test_calling_with_cfg(self):
        cfg = CubeConfig(msname="/nonexistent/path/dummy.ms", npix_x=256)
        with self.assertRaises(FileNotFoundError):
            make_dirty_cube(cfg=cfg)

    def test_uv_hermitian_config_and_headers(self):
        from astropy.io import fits
        import numpy as np
        from fastducc.cube import build_uv_headers

        cfg = CubeConfig(
            msname="test.ms",
            uv_format="fits-half",
            uv_hermitian_half=True,
            uv_threshold=1e-3,
        )
        self.assertEqual(cfg.uv_format, "fits-half")
        self.assertTrue(cfg.uv_hermitian_half)
        self.assertEqual(cfg.uv_threshold, 1e-3)

        base_hdr = fits.Header()
        time_info = {"grid_times": np.array([0.0, 0.11]), "dt_nominal": 0.11}
        h_r, h_i = build_uv_headers(
            base_hdr,
            npix_x=588 // 2 + 1,
            npix_y=588,
            pix_rad=1e-4,
            total_planes=50,
            time_info=time_info,
            hermitian_half=True,
        )
        self.assertEqual(h_r["NAXIS1"], 295)
        self.assertEqual(h_r["NAXIS2"], 588)
        self.assertEqual(h_r["CRPIX1"], 1.0)
        self.assertTrue(h_r["HERMHALF"])
        self.assertEqual(h_i["WSCVTYPE"], "IMAG")

if __name__ == "__main__":
    unittest.main()
