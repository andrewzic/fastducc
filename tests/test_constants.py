import unittest
from fastducc import constants

class TestConstants(unittest.TestCase):
    def test_speed_of_light(self):
        self.assertAlmostEqual(constants.c, 299792458.0)

    def test_dispersion_constant(self):
        # K_DM in MHz^2 pc^-1 cm^3 s
        self.assertAlmostEqual(constants.K_DM, 4.148808e3, places=2)

if __name__ == "__main__":
    unittest.main()
