import unittest
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

from fastducc import filters

class TestFilters(unittest.TestCase):
    def test_is_zero_flux_candidate(self):
        # 3D cube: (T=5, Ny=10, Nx=10)
        cube = np.ones((5, 10, 10), dtype=np.float32)

        det = {"y": 5, "x": 5, "width_samples": 1, "center_idx": 2, "t0_idx": 2, "t1_idx_excl": 3}
        self.assertFalse(filters.is_zero_flux_candidate(det, cube))

        # Test NaN flux at candidate location
        cube_nan = cube.copy()
        cube_nan[2, 5, 5] = np.nan
        self.assertTrue(filters.is_zero_flux_candidate(det, cube_nan))

        # Test zero-padded frame
        cube_zero_frame = cube.copy()
        cube_zero_frame[2, :, :] = 0.0
        self.assertTrue(filters.is_zero_flux_candidate(det, cube_zero_frame))

    def test_nms_snr_map_2d(self):
        # Create 2D SNR map with two adjacent peaks
        snr_map = np.zeros((20, 20), dtype=np.float32)
        snr_map[10, 10] = 12.0  # dominant peak
        snr_map[10, 11] = 10.0  # adjacent peak within radius=3
        snr_map[5, 5] = 9.0     # isolated secondary peak

        base_detections = [
            {"y": 10, "x": 10, "snr": 12.0},
            {"y": 10, "x": 11, "snr": 10.0},
            {"y": 5, "x": 5, "snr": 9.0},
        ]

        dets = filters.nms_snr_map_2d(
            snr_map,
            base_detections,
            threshold_sigma=8.0,
            spatial_radius=3,
        )

        # The adjacent peak at (10, 11) should be suppressed by (10, 10)
        self.assertEqual(len(dets), 2)
        coords = {(d["y"], d["x"]) for d in dets}
        self.assertIn((10, 10), coords)
        self.assertIn((5, 5), coords)
        self.assertNotIn((10, 11), coords)

    def test_filter_ips_agn_candidates(self):
        # Two DM>0 candidates: one near known RACS source with SNR ratio <= 0.8 * dm0_snr, one far
        dm0_dets = [
            {"ra_deg": 180.0, "dec_deg": -45.0, "snr": 10.0},
        ]
        candidates = [
            {"ra_deg": 180.0, "dec_deg": -45.0, "snr": 7.0},   # 7.0 <= 0.8 * 10.0 -> suppressed
            {"ra_deg": 190.0, "dec_deg": -45.0, "snr": 12.0},  # far away -> kept
        ]
        racs_coords = SkyCoord([180.0001], [-45.0001], unit="deg", frame="icrs")

        filtered = filters.filter_ips_agn_candidates(
            candidates_list=candidates,
            dm0_candidates=dm0_dets,
            racs_coords=racs_coords,
            psrcat_coords=None,
            ips_thresh=0.8,
            ips_radius_arcsec=30.0,
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["ra_deg"], 190.0)

    def test_evaluate_candidate_morphology_synthesis(self):
        # 1. Synthesized beam point source (FWHM = 3 to 5 pixels) MUST pass as is_point_like = True
        Y, X = np.ogrid[:32, :32]
        rng = np.random.default_rng(42)

        for fwhm in [3.0, 4.0, 5.0]:
            im_beam = np.exp(-4.0 * np.log(2.0) * ((X - 16)**2 + (Y - 16)**2) / fwhm**2)
            im_noisy = im_beam + rng.normal(0, 0.05, im_beam.shape)
            res = filters.evaluate_candidate_morphology_and_flux(
                im_noisy,
                is_synthesis_image=True,
                bmaj_pix=fwhm,
                bmin_pix=fwhm,
            )
            self.assertTrue(
                res["is_point_like"],
                f"Synthesized beam point source with FWHM={fwhm} failed morphology: {res['rejection_reasons']}",
            )
            self.assertEqual(res["morphology_verdict"], "POINT-LIKE")

        # 2. Extended diffuse Gaussian blob (FWHM = 15 pixels) MUST be rejected as is_point_like = False
        im_diff = np.exp(-4.0 * np.log(2.0) * ((X - 16)**2 + (Y - 16)**2) / 15.0**2)
        im_diff_noisy = im_diff + rng.normal(0, 0.05, im_diff.shape)
        res_diff = filters.evaluate_candidate_morphology_and_flux(
            im_diff_noisy,
            is_synthesis_image=True,
            bmaj_pix=4.0,
            bmin_pix=4.0,
        )
        self.assertFalse(res_diff["is_point_like"], "Extended diffuse blob should be rejected")
        self.assertEqual(res_diff["morphology_verdict"], "RESOLVED / EXTENDED")

        # 3a. Annular ring artifact MUST be rejected
        R = np.sqrt((X - 16)**2 + (Y - 16)**2)
        im_ring = np.exp(-0.5 * (R - 6.0)**2 / 1.5**2)
        res_ring = filters.evaluate_candidate_morphology_and_flux(
            im_ring,
            is_synthesis_image=True,
            bmaj_pix=4.0,
            bmin_pix=4.0,
        )
        self.assertFalse(res_ring["is_point_like"], "Annular ring artifact should be rejected")

        # 3b. Ridge artifact (elongated stripe) MUST be rejected
        im_ridge = np.exp(-0.5 * ((X - 16)**2 / 1.5**2 + (Y - 16)**2 / 12.0**2))
        res_ridge = filters.evaluate_candidate_morphology_and_flux(
            im_ridge,
            is_synthesis_image=True,
            bmaj_pix=4.0,
            bmin_pix=4.0,
        )
        self.assertFalse(res_ridge["is_point_like"], "Elongated ridge artifact should be rejected")

    def test_evaluate_candidate_morphology_backward_compatibility(self):
        # In non-synthesis mode, 1-pixel spike passes, diffuse / extended fails
        im_spike = np.zeros((32, 32))
        im_spike[16, 16] = 10.0
        res = filters.evaluate_candidate_morphology_and_flux(im_spike, is_synthesis_image=False)
        self.assertTrue(res["is_point_like"])

        # Wide source fails in non-synthesis mode
        Y, X = np.ogrid[:32, :32]
        im_wide = np.exp(-0.5 * ((X - 16)**2 + (Y - 16)**2) / 3.0**2)
        res_wide = filters.evaluate_candidate_morphology_and_flux(im_wide, is_synthesis_image=False)
        self.assertFalse(res_wide["is_point_like"])

if __name__ == "__main__":
    unittest.main()
