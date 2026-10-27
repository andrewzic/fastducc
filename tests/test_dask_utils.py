import os
import unittest
from unittest.mock import MagicMock, patch

from fastducc.dask_utils import (
    derive_dask_job_name,
    resolve_dask_log_directory,
    setup_slurm_cluster,
    get_dask_client,
)


class TestDaskUtils(unittest.TestCase):
    def test_derive_dask_job_name_transient(self):
        ms_name = "cracoData.LTR_1733-2344.SB77974.beam01.20251015072402.calG9.uvsub.ms"
        job_name = derive_dask_job_name(stage="transient", ms_path=ms_name)
        self.assertEqual(job_name, "fd_tr_SB77974_b01")

    def test_derive_dask_job_name_uv_ffa(self):
        ms_name = "cracoData.LTR_1733-2344.SB77974.beam12.20251015072402.calG9.uvsub.ms"
        job_name = derive_dask_job_name(stage="uv_ffa", ms_path=ms_name)
        self.assertEqual(job_name, "dask_ffa_SB77974_b12")

    def test_derive_dask_job_name_nufft_explicit_beam_sbid(self):
        job_name = derive_dask_job_name(stage="nufft", beam=5, sbid="SB77974")
        self.assertEqual(job_name, "dask_nufft_SB77974_b05")

    def test_derive_dask_job_name_explicit_override(self):
        job_name = derive_dask_job_name(stage="transient", explicit_name="custom_worker_job_name_123")
        self.assertEqual(job_name, "custom_worker_job_name_123")

    def test_resolve_dask_log_directory_explicit(self):
        target = resolve_dask_log_directory(log_dir="/tmp/test_pipeline_logs")
        self.assertEqual(target, "/tmp/test_pipeline_logs/dask_workers")
        self.assertTrue(os.path.isdir(target))

    def test_resolve_dask_log_directory_env(self):
        with patch.dict(os.environ, {"LOG_DIR": "/tmp/test_env_logs"}):
            target = resolve_dask_log_directory()
            self.assertEqual(target, "/tmp/test_env_logs/dask_workers")
            self.assertTrue(os.path.isdir(target))

    @patch("fastducc.dask_utils.SLURMCluster")
    def test_setup_slurm_cluster_parameters(self, mock_slurm):
        mock_instance = MagicMock()
        mock_slurm.return_value = mock_instance

        cluster = setup_slurm_cluster(
            queue="milan",
            account="oz313",
            cores_per_worker=2,
            mem="16GB",
            walltime="03:00:00",
            job_name="fd_tr_SB77974_b01",
            log_directory="/tmp/dask_workers",
            min_workers=2,
            max_workers=8,
            adaptive=True,
        )

        mock_slurm.assert_called_once()
        call_kwargs = mock_slurm.call_args[1]
        self.assertEqual(call_kwargs["name"], "fd_tr_SB77974_b01")
        self.assertEqual(call_kwargs["queue"], "milan")
        self.assertEqual(call_kwargs["account"], "oz313")
        self.assertEqual(call_kwargs["cores"], 2)
        self.assertEqual(call_kwargs["memory"], "16GB")

        # Verify worker_extra_args has lifetime retirement
        worker_extra = call_kwargs["worker_extra_args"]
        self.assertIn("--lifetime", worker_extra)

        # Verify adaptive was called
        mock_instance.adapt.assert_called_once_with(minimum=2, maximum=8)

    def test_fastducc_run_cli_dask_arguments(self):
        from fastducc.fastducc_run import build_cli
        args = build_cli([
            "--msname", "test_craco.ms",
            "--parallel-mode", "dask-slurm",
            "--dask-job-name", "custom_tr_b01",
            "--slurm-log-dir", "/tmp/custom_dask_logs",
        ])
        self.assertEqual(args.parallel_mode, "dask-slurm")
        self.assertEqual(args.dask_job_name, "custom_tr_b01")
        self.assertEqual(args.slurm_log_dir, "/tmp/custom_dask_logs")


if __name__ == "__main__":
    unittest.main()
