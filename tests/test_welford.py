import unittest
import numpy as np
import tempfile
import os
from astropy.io import fits
from astropy.table import Table

from fastducc import kernels, core
from fastducc.candidate_tools import aggregation


class TestWelfordAggregation(unittest.TestCase):
    def test_within_chunk_variance_pools_without_cross_term(self):
        """
        Verify that pooling multi-chunk Welford aggregates with include_cross_term=False
        measures the pure within-chunk variance and rejects inter-chunk mean steps (e.g. sidelobes).
        """
        np.random.seed(42)
        Ny, Nx = 16, 16
        n_chunks = 4
        chunk_len = 50
        true_sigma = 0.05

        # Accumulate per-chunk statistics
        count_acc = np.zeros((Ny, Nx), dtype=np.int64)
        mean_acc = np.zeros((Ny, Nx), dtype=np.float64)
        M2_acc = np.zeros((Ny, Nx), dtype=np.float64)

        M2_chan_acc = np.zeros((Ny, Nx), dtype=np.float64)
        mean_chan_acc = np.zeros((Ny, Nx), dtype=np.float64)
        count_chan_acc = np.zeros((Ny, Nx), dtype=np.int64)

        for k in range(n_chunks):
            # Each chunk has noise + a large distinct DC offset (simulating rotating sidelobes / beam shift)
            dc_offset = (k + 1) * 10.0
            chunk_data = np.random.normal(loc=dc_offset, scale=true_sigma, size=(chunk_len, Ny, Nx))

            c = np.full((Ny, Nx), chunk_len, dtype=np.int64)
            # Within-chunk sample mean and sum-of-squared deviations
            m = np.mean(chunk_data, axis=0)
            M2 = np.sum((chunk_data - m)**2, axis=0)

            # Pooled (default)
            count_acc, mean_acc, M2_acc = core.welford_combine_aggregates(
                count_acc, mean_acc, M2_acc, c, m, M2, include_cross_term=False
            )
            # Chan cross-term
            count_chan_acc, mean_chan_acc, M2_chan_acc = core.welford_combine_aggregates(
                count_chan_acc, mean_chan_acc, M2_chan_acc, c, m, M2, include_cross_term=True
            )

        total_count = n_chunks * chunk_len
        self.assertTrue(np.all(count_acc == total_count))

        # Finalised pooled std map
        std_pooled = kernels.welford_finalise_std(count_acc, M2_acc, ddof=1)
        # Finalised Chan std map
        std_chan = kernels.welford_finalise_std(count_chan_acc, M2_chan_acc, ddof=1)

        # Pooled std matches true within-chunk noise sigma (~0.05)
        np.testing.assert_allclose(np.mean(std_pooled), true_sigma, rtol=0.05)

        # Chan std is corrupted by inter-chunk DC steps (explodes to > 10.0)
        self.assertGreater(np.mean(std_chan), 5.0)

    def test_welford_finalise_std_clamps_negative_m2(self):
        """Verify that tiny negative M2 values from float inaccuracy clamp to 0 instead of NaN."""
        count = np.array([[10, 10], [0, 1]], dtype=np.int64)
        M2 = np.array([[-1e-16, 2.5], [0.0, 1.0]], dtype=np.float64)

        std = kernels.welford_finalise_std(count, M2, ddof=1)
        self.assertEqual(std[0, 0], 0.0)
        self.assertAlmostEqual(std[0, 1], np.sqrt(2.5 / 9.0))
        self.assertTrue(np.isnan(std[1, 0]))
        self.assertTrue(np.isnan(std[1, 1]))

    def test_generate_obs_variance_candidate_products_resolution(self):
        """Verify that generate_obs_variance_candidate_products loads DM-tagged products correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ms_base = "test_obs"
            dm = 15.0
            dm_tag = f"_dm{dm:06.2f}"

            # Create mock std_map_full FITS
            hdr = fits.Header()
            hdr["CRPIX1"] = 16.0
            hdr["CRPIX2"] = 16.0
            hdr["CDELT1"] = -0.001
            hdr["CDELT2"] = -0.001
            hdr["CRVAL1"] = 180.0
            hdr["CRVAL2"] = -45.0
            hdr["CTYPE1"] = "RA---TAN"
            hdr["CTYPE2"] = "DEC--TAN"
            hdr["DM"] = dm
            data = np.full((32, 32), 0.05, dtype=np.float32)
            data[16, 16] = 0.50  # candidate peak
            std_fits = os.path.join(tmpdir, f"{ms_base}{dm_tag}_std_map_full.fits")
            fits.writeto(std_fits, data, hdr)

            # Create candidate table with time_center >= 9000
            tab = Table({
                "srcname": ["SRC_OBS_001"],
                "x": [16],
                "y": [16],
                "snr": [10.0],
                "ra_deg": [180.0],
                "dec_deg": [-45.0],
                "ra_hms": ["12:00:00.0"],
                "dec_dms": ["-45:00:00.0"],
                "time_center": [9999.0],
                "center_idx": [9999],
                "dm": [dm],
            })
            tab_path = os.path.join(tmpdir, f"{ms_base}{dm_tag}_var_candidates.vot")
            tab.write(tab_path, format="votable")

            generated = aggregation.generate_obs_variance_candidate_products(
                ms_base=ms_base,
                candidates_dir=tmpdir,
                spatial_size=10,
                dm=dm,
                std_fits_path=std_fits,
            )
            self.assertTrue(len(generated) > 0)
            cutout_fits = [p for p in generated if p.endswith("_cutout.fits")]
            summary_pdf = [p for p in generated if p.endswith("_summary.pdf")]
            self.assertEqual(len(cutout_fits), 1)
            self.assertEqual(len(summary_pdf), 1)
            self.assertTrue(os.path.exists(cutout_fits[0]))
            self.assertTrue(os.path.exists(summary_pdf[0]))

    def test_radio_sidelobe_rejection_and_transient_preservation(self):
        """
        Radio astronomy benchmark:
        Simulate an observation with 5 chunks containing:
        - Pixel (0, 0): Quiet sky with 10 mJy thermal noise.
        - Pixel (0, 1): Rotating PSF sidelobe drifting by hundreds of mJy across chunks.
        - Pixel (0, 2): A fast transient pulse (200 mJy burst) in chunk 2.

        Verify that pooled within-chunk variance rejects the rotating sidelobe (stays at noise floor),
        while Chan's cross-term formula corrupts the sidelobe pixel into a massive false positive.
        """
        np.random.seed(123)
        n_chunks = 5
        chunk_len = 80
        sigma_noise = 0.010  # 10 mJy thermal noise

        c_pool = np.zeros((1, 3), dtype=np.int64)
        m_pool = np.zeros((1, 3), dtype=np.float64)
        M2_pool = np.zeros((1, 3), dtype=np.float64)

        c_chan = np.zeros((1, 3), dtype=np.int64)
        m_chan = np.zeros((1, 3), dtype=np.float64)
        M2_chan = np.zeros((1, 3), dtype=np.float64)

        sidelobe_fluxes = [0.05, 0.45, 0.80, 0.30, 0.02]

        for k in range(n_chunks):
            chunk = np.random.normal(0.0, sigma_noise, size=(chunk_len, 1, 3))
            # Sidelobe rotation at pixel (0, 1)
            chunk[:, 0, 1] += sidelobe_fluxes[k]
            # Fast transient at pixel (0, 2) in chunk 2 (5 samples at 200 mJy)
            if k == 2:
                chunk[30:35, 0, 2] += 0.200

            c = np.full((1, 3), chunk_len, dtype=np.int64)
            m = np.mean(chunk, axis=0)
            M2 = np.sum((chunk - m)**2, axis=0)

            c_pool, m_pool, M2_pool = core.welford_combine_aggregates(
                c_pool, m_pool, M2_pool, c, m, M2, include_cross_term=False
            )
            c_chan, m_chan, M2_chan = core.welford_combine_aggregates(
                c_chan, m_chan, M2_chan, c, m, M2, include_cross_term=True
            )

        std_pool = kernels.welford_finalise_std(c_pool, M2_pool, ddof=1)
        std_chan = kernels.welford_finalise_std(c_chan, M2_chan, ddof=1)

        # 1. Quiet sky pixel matches thermal noise floor in both
        self.assertAlmostEqual(std_pool[0, 0], sigma_noise, places=2)
        self.assertAlmostEqual(std_pool[0, 0], std_chan[0, 0], places=3)

        # 2. Sidelobe pixel: pooled stays near noise floor (~10 mJy), Chan blows up (>250 mJy)
        self.assertLess(std_pool[0, 1], 0.015)
        self.assertGreater(std_chan[0, 1], 0.250)

        # 3. Transient pixel: cleanly elevated above noise floor (~20 mJy) in pooled map
        self.assertGreater(std_pool[0, 2], 0.020)


if __name__ == "__main__":
    unittest.main()
