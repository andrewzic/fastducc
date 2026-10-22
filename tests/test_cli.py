import unittest
import io
import sys
from unittest.mock import patch

from fastducc import __version__
from fastducc import cli
from fastducc.fastducc_run import build_cli as build_cli_search
from fastducc.cube import build_cli_cube

class TestCLI(unittest.TestCase):
    def test_version_output(self):
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            code = cli.main(["--version"])
            self.assertEqual(code, 0)
            self.assertIn(__version__, fake_out.getvalue())

    def test_top_level_help(self):
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            code = cli.main(["--help"])
            self.assertEqual(code, 0)
            out = fake_out.getvalue()
            self.assertIn("search", out)
            self.assertIn("cube", out)
            self.assertIn("aggregate", out)
            self.assertIn("aggregate_obs", out)
            self.assertIn("periodicity", out)

    def test_no_args_displays_help(self):
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            code = cli.main([])
            self.assertEqual(code, 0)
            self.assertIn("Available Subcommands", fake_out.getvalue())

    def test_unknown_subcommand(self):
        with patch("sys.stderr", new=io.StringIO()):
            code = cli.main(["unknown_command_xyz"])
            self.assertEqual(code, 2)

    def test_build_cli_aggregate(self):
        args = cli.build_cli_aggregate(["--obs-root", "/tmp/test_obs", "--kind", "variance"])
        self.assertEqual(args.obs_root, "/tmp/test_obs")
        self.assertEqual(args.kind, "variance")
        self.assertEqual(args.max_n_candidates_per_source, 100)

    def test_build_cli_aggregate_obs(self):
        args = cli.build_cli_aggregate_obs([
            "--obs-root", "/tmp/test_sbid",
            "--no-organize",
            "--psrcat-path", "/custom/psrcat.csv",
            "--racs-path", "/custom/racs.xml",
        ])
        self.assertEqual(args.obs_root, "/tmp/test_sbid")
        self.assertFalse(args.organize)
        self.assertEqual(args.link_mode, "symlink")
        self.assertEqual(args.psrcat_path, "/custom/psrcat.csv")
        self.assertEqual(args.racs_path, "/custom/racs.xml")

    def test_build_cli_periodicity(self):
        args = cli.build_cli_periodicity(["--msname", "test.ms", "--fmin", "1.5", "--dm", "25.0"])
        self.assertEqual(args.msname, "test.ms")
        self.assertEqual(args.fmin, 1.5)
        self.assertEqual(args.dm, 25.0)

    def test_build_cli_cube(self):
        args = build_cli_cube(["--msname", "test.ms", "--out-cube", "cube.fits", "--chunk-size", "256"])
        self.assertEqual(args.msname, "test.ms")
        self.assertEqual(args.out_cube, "cube.fits")
        self.assertEqual(args.chunk_size, 256)

    def test_build_cli_search(self):
        args = build_cli_search(["--msname", "test.ms", "--chunk-size", "1024", "--threshold-sigma", "9.0"])
        self.assertEqual(args.msname, "test.ms")
        self.assertEqual(args.chunk_size, 1024)
        self.assertEqual(args.threshold_sigma, 9.0)

    def test_shorthand_routing_to_search(self):
        with patch("fastducc.cli.run_search", return_value=0) as mock_search:
            code = cli.main(["--msname", "test.ms", "--chunk-size", "512"])
            self.assertEqual(code, 0)
            mock_search.assert_called_once_with(["--msname", "test.ms", "--chunk-size", "512"])

if __name__ == "__main__":
    unittest.main()
