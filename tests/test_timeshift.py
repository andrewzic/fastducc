import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from fastducc import timeshift
from fastducc.fd_types import CubeConfig
from fastducc.cube import build_cli_cube, make_dirty_cube


class TestTimeShift(unittest.TestCase):
    def test_shift_time_series_identity(self):
        data = np.ones((5, 10), dtype=np.float32)
        out = timeshift.shift_time_series(data, 0.0, method="fft", axis=-1)
        np.testing.assert_allclose(data, out)

        out_sinc = timeshift.shift_time_series(data, 0.0, method="sinc", axis=-1)
        np.testing.assert_allclose(data, out_sinc)

    def test_shift_time_series_single_sample(self):
        data = np.array([42.0], dtype=np.float32)
        out = timeshift.shift_time_series(data, 0.3, method="fft")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0], 42.0)

    def test_shift_time_series_real_fft_and_sinc(self):
        t = np.linspace(0, 10, 200, endpoint=False)
        dt = t[1] - t[0]
        signal = np.sin(2 * np.pi * 0.2 * t).astype(np.float64)

        shift_samps = 0.35
        # Delayed by shift_samps * dt
        exact = np.sin(2 * np.pi * 0.2 * (t - shift_samps * dt))

        s_fft = timeshift.shift_time_series(signal, shift_samps, method="fft")
        s_sinc = timeshift.shift_time_series(signal, shift_samps, method="sinc")

        # Compare away from boundaries (index 15 to -15)
        err_fft = np.max(np.abs(s_fft[15:-15] - exact[15:-15]))
        err_sinc = np.max(np.abs(s_sinc[15:-15] - exact[15:-15]))

        self.assertLess(err_fft, 0.02)
        self.assertLess(err_sinc, 0.005)
        self.assertEqual(s_fft.dtype, signal.dtype)
        self.assertEqual(s_sinc.dtype, signal.dtype)

    def test_shift_time_series_negative_shift(self):
        t = np.linspace(0, 10, 200, endpoint=False)
        dt = t[1] - t[0]
        signal = np.cos(2 * np.pi * 0.15 * t).astype(np.float32)

        shift_samps = -0.4
        exact = np.cos(2 * np.pi * 0.15 * (t - shift_samps * dt))

        s_fft = timeshift.shift_time_series(signal, shift_samps, method="fft")
        s_sinc = timeshift.shift_time_series(signal, shift_samps, method="sinc")

        err_fft = np.max(np.abs(s_fft[15:-15] - exact[15:-15]))
        err_sinc = np.max(np.abs(s_sinc[15:-15] - exact[15:-15]))

        self.assertLess(err_fft, 0.02)
        self.assertLess(err_sinc, 0.005)
        self.assertEqual(s_fft.dtype, np.float32)

    def test_shift_time_series_complex_native_vis(self):
        # Native visibilities shape: (nbl, nchan, nt)
        nbl, nchan, nt = 4, 3, 50
        t = np.linspace(0, 5, nt, endpoint=False)
        dt = t[1] - t[0]

        vis = np.zeros((nbl, nchan, nt), dtype=np.complex64)
        for b in range(nbl):
            for c in range(nchan):
                vis[b, c] = np.exp(1j * 2 * np.pi * 0.1 * (b + 1) * t)

        shift_samps = 0.25
        shifted_fft = timeshift.shift_time_series(vis, shift_samps, method="fft", axis=-1)
        shifted_sinc = timeshift.shift_time_series(vis, shift_samps, method="sinc", axis=-1)

        self.assertEqual(shifted_fft.shape, vis.shape)
        self.assertEqual(shifted_sinc.shape, vis.shape)
        self.assertEqual(shifted_fft.dtype, np.complex64)
        self.assertEqual(shifted_sinc.dtype, np.complex64)

        expected_00 = np.exp(1j * 2 * np.pi * 0.1 * 1 * (t - shift_samps * dt))
        err_fft = np.max(np.abs(shifted_fft[0, 0, 10:-10] - expected_00[10:-10]))
        err_sinc = np.max(np.abs(shifted_sinc[0, 0, 10:-10] - expected_00[10:-10]))
        self.assertLess(err_fft, 0.02)
        self.assertLess(err_sinc, 0.005)

    def test_shift_time_series_gridded_cube(self):
        # Gridded cube shape: (nt, ny, nx) along axis=0
        cube = np.random.randn(40, 16, 16).astype(np.float32)
        shifted = timeshift.shift_time_series(cube, -0.3, method="fft", axis=0)
        self.assertEqual(shifted.shape, cube.shape)
        self.assertEqual(shifted.dtype, np.float32)

    def test_shift_invalid_method(self):
        arr = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            timeshift.shift_time_series(arr, 0.1, method="invalid_method")


class TestTimeRegularization(unittest.TestCase):
    def test_identify_time_blocks_scans_and_gaps(self):
        # Scan 1: 5 samples at dt=0.1
        t1 = np.arange(0, 5) * 0.1
        # Gap of 10s, Scan 2: 5 samples at dt=0.1
        t2 = 10.0 + np.arange(0, 5) * 0.1
        times = np.concatenate([t1, t2])
        scans = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])

        blocks, dt = timeshift.identify_time_blocks(times, scan_numbers=scans)
        self.assertEqual(len(blocks), 2)
        self.assertAlmostEqual(dt, 0.1)
        self.assertEqual(blocks[0]["n_samples"], 5)
        self.assertEqual(blocks[1]["n_samples"], 5)
        self.assertEqual(blocks[0]["scan_id"], 1)
        self.assertEqual(blocks[1]["scan_id"], 2)

    def test_regularize_time_grid_out_of_sync_block(self):
        # Scan 1: in sync (0.0 to 0.4)
        t1 = np.arange(0, 5) * 0.1
        # Scan 2: start at 10.034s -> fractional offset is +0.34 dt
        t2 = 10.034 + np.arange(0, 5) * 0.1
        times = np.concatenate([t1, t2])
        scans = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])

        res = timeshift.regularize_time_grid(times, sync_tolerance=0.05, scan_numbers=scans)
        self.assertTrue(res["has_gaps"])
        self.assertEqual(res["total_planes"], 105)
        self.assertEqual(len(res["blank_indices"]), 95)
        self.assertEqual(res["n_shifted_blocks"], 1)

        b0 = res["blocks"][0]
        self.assertFalse(b0["is_shifted"])
        self.assertAlmostEqual(b0["shift_samps"], 0.0)
        self.assertEqual(b0["k_start"], 0)

        b1 = res["blocks"][1]
        self.assertTrue(b1["is_shifted"])
        self.assertAlmostEqual(b1["offset_frac"], 0.34, places=4)
        self.assertAlmostEqual(b1["shift_samps"], -0.34, places=4)
        self.assertEqual(b1["k_start"], 100)

        # Check grid indices assignment
        np.testing.assert_array_equal(res["grid_indices"][:5], np.arange(0, 5))
        np.testing.assert_array_equal(res["grid_indices"][5:], np.arange(100, 105))

    def test_regularize_time_grid_within_tolerance(self):
        # Offset is 0.02 dt, with sync_tolerance=0.05 -> should not shift
        t1 = np.arange(0, 5) * 0.1
        t2 = 5.002 + np.arange(0, 5) * 0.1
        times = np.concatenate([t1, t2])

        res = timeshift.regularize_time_grid(times, sync_tolerance=0.05)
        self.assertEqual(res["n_shifted_blocks"], 0)
        self.assertFalse(res["blocks"][1]["is_shifted"])
        self.assertAlmostEqual(res["blocks"][1]["shift_samps"], 0.0)


class TestCubeCLIAndIntegration(unittest.TestCase):
    def test_cli_parser_regularize_options(self):
        args = build_cli_cube([
            "--msname", "test.ms",
            "--regularize-time",
            "--sync-tolerance", "0.08",
            "--shift-method", "sinc",
            "--shift-domain", "gridded",
        ])
        self.assertTrue(args.regularize_time)
        self.assertAlmostEqual(args.sync_tolerance, 0.08)
        self.assertEqual(args.shift_method, "sinc")
        self.assertEqual(args.shift_domain, "gridded")

    def test_cube_config_from_namespace(self):
        args = build_cli_cube([
            "--msname", "test.ms",
            "--regularize-time",
            "--sync-tolerance", "0.1",
            "--shift-method", "sinc",
            "--shift-domain", "gridded",
        ])
        cfg = CubeConfig.from_namespace(args)
        self.assertTrue(cfg.regularize_time)
        self.assertAlmostEqual(cfg.sync_tolerance, 0.1)
        self.assertEqual(cfg.shift_method, "sinc")
        self.assertEqual(cfg.shift_domain, "gridded")


if __name__ == "__main__":
    unittest.main()
