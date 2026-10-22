import unittest
import numpy as np
from fastducc import wcs

class TestWCS(unittest.TestCase):
    def test_arcsec_rad_conversion(self):
        arcsec = 3600.0
        rad = wcs.arcsec_to_rad(arcsec)
        self.assertAlmostEqual(rad, np.deg2rad(1.0), places=9)
        back_arcsec = np.rad2deg(rad) * 3600.0
        self.assertAlmostEqual(back_arcsec, arcsec, places=6)

    def test_rad_to_hmsdms_origin(self):
        ra_str, dec_str = wcs.rad_to_hmsdms(0.0, 0.0)
        self.assertEqual(ra_str, "00:00:00.0")
        self.assertEqual(dec_str, "+00:00:00.0")

    def test_rad_to_hmsdms_negative_dec(self):
        ra_rad = np.deg2rad(180.0)
        dec_rad = np.deg2rad(-45.5)
        ra_str, dec_str = wcs.rad_to_hmsdms(ra_rad, dec_rad)
        self.assertEqual(ra_str, "12:00:00.0")
        self.assertEqual(dec_str, "-45:30:00.0")

    def test_rad_to_hmsdms_format_pattern(self):
        import re
        ra_str, dec_str = wcs.rad_to_hmsdms(1.234567, -0.987654)
        ra_match = re.match(r"^\d{2}:\d{2}:\d{2}\.\d$", ra_str)
        dec_match = re.match(r"^[+-]\d{2}:\d{2}:\d{2}\.\d$", dec_str)
        self.assertIsNotNone(ra_match, f"Invalid RA format: {ra_str}")
        self.assertIsNotNone(dec_match, f"Invalid Dec format: {dec_str}")

if __name__ == "__main__":
    unittest.main()
