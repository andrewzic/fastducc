import unittest
import os
import tempfile
from astropy.table import Table

from fastducc import catalogues
from fastducc.candidate_tools import regions

class TestCandidateTools(unittest.TestCase):
    def test_catalogues_module_name(self):
        # Australian English spelling constraint: 'catalogues.py'
        self.assertTrue(hasattr(catalogues, "get_psrcat_csv_path"))
        self.assertTrue(hasattr(catalogues, "get_racs_vot_path"))

    def test_psrcat_path_exists(self):
        path = catalogues.get_psrcat_csv_path()
        self.assertTrue(os.path.exists(path), f"psrcat catalogue not found at {path}")

    def test_write_obs_region_files(self):
        tab = Table({
            "ra_deg": [180.0, 181.5],
            "dec_deg": [-45.0, -44.5],
            "srcname": ["SRC_A", "SRC_B"],
            "max_snr": [10.5, 8.2],
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            res = regions.write_obs_region_files(
                tab,
                field="FIELD01",
                sbid="SB12345",
                kind="boxcar",
                out_dir=tmpdir,
                ds9_radius_arcsec=30.0,
                kvis_radius_arcsec=30.0,
            )
            reg_path = res["ds9_reg"]
            ann_path = res["kvis_ann"]
            self.assertTrue(os.path.exists(reg_path))
            self.assertTrue(os.path.exists(ann_path))

            with open(reg_path, "r") as f:
                content = f.read()
                self.assertIn("# Region file format: DS9", content)
                self.assertIn("circle(180.000000,-45.000000,30.00\")", content)
                self.assertIn("SRC_A", content)

            with open(ann_path, "r") as f:
                content = f.read()
                self.assertIn("COORD W", content)
                self.assertIn("CIRCLE 180.000000 -45.000000", content)
                self.assertIn("TEXT", content)

if __name__ == "__main__":
    unittest.main()
