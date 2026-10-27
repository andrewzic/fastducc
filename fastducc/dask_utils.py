# -*- coding: utf-8 -*-
"""
fastducc.dask_utils
===================
Unified, robust Dask cluster initialization and SLURMCluster management across
FastDUCC transient search, cube generation, NUFFT periodicity, and Lotrun periodicity pipelines.

Standardises:
1. Descriptive SLURM job names for Dask workers (e.g. 'fd_tr_SB77974_b01', 'dask_ffa_b01',
   'dask_nufft_b01') instead of the opaque generic 'dask-worker'.
2. Descriptive SLURM worker log files (e.g. '<log_dir>/dask_workers/<job_name>-%j.out')
   routed to the main pipeline's logging directory ($LOG_DIR/dask_workers).
3. Resilience configuration:
   - distributed.scheduler.allowed-failures: 25
   - distributed.adaptive.target-duration: 3600s
4. Worker lifetime calculation:
   - Retires workers gracefully 45 minutes before SLURM hard walltime (--lifetime / --lifetime-stagger).
5. Standard HPC cluster prologue (module load, venv activation, OMP/OPENBLAS threading control).
6. Dynamic adaptive scaling with minimum and maximum bounds.
"""

from __future__ import annotations

import inspect
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger("fastducc.dask_utils")

# Lazy / conditional imports for environments where dask/distributed are optional
try:
    import dask
    from dask.distributed import Client, LocalCluster
except ImportError:
    dask = None  # type: ignore[assignment]
    Client = None  # type: ignore[assignment, misc]
    LocalCluster = None  # type: ignore[assignment, misc]

try:
    from dask_jobqueue import SLURMCluster
except ImportError:
    SLURMCluster = None  # type: ignore[assignment, misc]


def derive_dask_job_name(
    stage: str = "worker",
    ms_path: Optional[str] = None,
    beam: Optional[Union[str, int]] = None,
    sbid: Optional[Union[str, int]] = None,
    explicit_name: Optional[str] = None,
    max_len: int = 48,
) -> str:
    """
    Derive a descriptive Slurm job name for Dask workers.
    
    Examples:
      - Transient search:  'fd_tr_SB77974_b01' or 'fd_tr_b01'
      - UV-FFA search:     'dask_ffa_SB77974_b01' or 'dask_ffa_b01'
      - NUFFT search:      'dask_nufft_SB77974_b01' or 'dask_nufft_b01'
      - Cube generation:   'fd_cube_b01'
    """
    if explicit_name and explicit_name.strip():
        clean_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", explicit_name.strip())
        return clean_name[:max_len]

    # Map stage to standard concise prefix
    stage_lower = stage.lower().replace("-", "_")
    if "transient" in stage_lower:
        prefix = "fd_tr"
    elif "uv_ffa" in stage_lower or "ffa" in stage_lower:
        prefix = "dask_ffa"
    elif "nufft" in stage_lower:
        prefix = "dask_nufft"
    elif "cube" in stage_lower:
        prefix = "fd_cube"
    else:
        prefix = f"dask_{stage_lower[:8]}"

    # Extract beam and SBID from MS path if not explicitly provided
    extracted_sbid = str(sbid).strip() if sbid is not None else ""
    extracted_beam = str(beam).strip() if beam is not None else ""

    if ms_path:
        fname = os.path.basename(ms_path.rstrip("/"))
        if not extracted_sbid:
            m_sb = re.search(r"(SB\d+)", fname, re.IGNORECASE)
            if m_sb:
                extracted_sbid = m_sb.group(1).upper()
        if not extracted_beam:
            m_b = re.search(r"beam(\d+)", fname, re.IGNORECASE)
            if not m_b:
                m_b = re.search(r"(?:^|[._\-])b(\d+)", fname, re.IGNORECASE)
            if m_b:
                extracted_beam = f"b{int(m_b.group(1)):02d}"

    if extracted_beam and not extracted_beam.startswith("b"):
        try:
            extracted_beam = f"b{int(extracted_beam):02d}"
        except ValueError:
            extracted_beam = f"b{extracted_beam}"

    parts = [prefix]
    if extracted_sbid:
        parts.append(extracted_sbid)
    if extracted_beam:
        parts.append(extracted_beam)

    if len(parts) == 1:
        parts.append("worker")

    job_name = "_".join(parts)
    job_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", job_name)
    return job_name[:max_len]


def resolve_dask_log_directory(
    log_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    subdir: str = "dask_workers",
) -> str:
    """
    Resolve and create the target directory for Dask worker Slurm stdout/stderr logs.
    Defaults to $LOG_DIR/dask_workers or $SLURM_LOG_DIR/dask_workers when running in pipeline.
    """
    target: Optional[str] = None

    if log_dir and log_dir.strip():
        clean_dir = os.path.abspath(log_dir.strip())
        if clean_dir.endswith(subdir) or "dask" in os.path.basename(clean_dir):
            target = clean_dir
        else:
            target = os.path.join(clean_dir, subdir)
    elif "LOG_DIR" in os.environ and os.environ["LOG_DIR"].strip():
        target = os.path.join(os.path.abspath(os.environ["LOG_DIR"].strip()), subdir)
    elif "SLURM_LOG_DIR" in os.environ and os.environ["SLURM_LOG_DIR"].strip():
        target = os.path.join(os.path.abspath(os.environ["SLURM_LOG_DIR"].strip()), subdir)
    elif output_dir and output_dir.strip():
        target = os.path.join(os.path.abspath(output_dir.strip()), "logs", subdir)
    else:
        target = os.path.join(os.getcwd(), "logs", subdir)

    try:
        os.makedirs(target, exist_ok=True)
    except Exception as e:
        logger.warning(f"Could not create Dask worker log directory '{target}': {e}. Falling back to /tmp/dask_workers")
        target = "/tmp/dask_workers"
        os.makedirs(target, exist_ok=True)

    return target


def setup_slurm_cluster(
    queue: Optional[str] = None,
    account: Optional[str] = None,
    cores_per_worker: int = 1,
    mem: str = "16GB",
    walltime: str = "02:30:00",
    job_name: Optional[str] = None,
    log_directory: Optional[str] = None,
    job_extra_directives: Optional[Sequence[str]] = None,
    interface: Optional[str] = None,
    python_path: Optional[str] = None,
    min_workers: int = 0,
    max_workers: int = 16,
    adaptive: bool = True,
    allowed_failures: int = 25,
    target_duration: str = "3600s",
) -> Any:
    """
    Construct and configure a dynamic dask_jobqueue.SLURMCluster with:
      - Descriptive job name (-J <job_name>)
      - Descriptive worker logs (-o <log_dir>/<job_name>-%j.out, -e <log_dir>/<job_name>-%j.err)
      - Standard OzSTAR / HPC prologue & graceful lifetime retirement
      - Adaptive cluster scaling
    """
    if SLURMCluster is None:
        raise RuntimeError("dask_jobqueue is required for SLURMCluster. Install with `pip install dask-jobqueue`.")

    py_exe = python_path or sys.executable

    # Detect high-speed InfiniBand network interface if not specified
    if interface is None:
        try:
            import psutil
            if "ib0" in psutil.net_if_addrs():
                interface = "ib0"
        except Exception:
            pass

    # Configure scheduler options
    scheduler_opts: Dict[str, Any] = {"dashboard_address": ":0"}
    if interface:
        scheduler_opts["interface"] = interface

    # Standard prologue across fastducc transient & periodicity searches
    prologue = [
        "module load python-scientific/3.11.5-foss-2023b 2>/dev/null || true",
        "unset PYTHONPATH",
        f"source {os.path.dirname(py_exe)}/activate 2>/dev/null || true",
        f"export OMP_NUM_THREADS={cores_per_worker}",
        "export OPENBLAS_NUM_THREADS=1",
        "export MKL_NUM_THREADS=1",
    ]

    # Global Dask resilience settings
    if dask is not None:
        dask.config.set({
            "distributed.scheduler.allowed-failures": int(allowed_failures),
            "distributed.adaptive.target-duration": target_duration,
        })

    # Worker extra arguments and walltime lifetime management
    worker_extra = ["--nthreads", "1", "--memory-limit", "0"]
    if walltime:
        try:
            parts = walltime.split(":")
            if len(parts) >= 2:
                wt_min = int(parts[0]) * 60 + int(parts[1])
                lifetime_min = max(30, wt_min - 45)
                worker_extra.extend(["--lifetime", f"{lifetime_min}m", "--lifetime-stagger", "5m"])
        except Exception:
            pass

    # Ensure job_name and log_directory are set
    effective_name = job_name or "dask_worker"
    effective_log_dir = log_directory or resolve_dask_log_directory()

    extra_directives = list(job_extra_directives or [])
    extra_directives.append(f"--job-name={effective_name}")

    # Check if SLURMCluster supports log_directory natively; otherwise append to directives
    cluster_kwargs: Dict[str, Any] = {
        "queue": queue,
        "account": account,
        "cores": int(cores_per_worker),
        "memory": mem,
        "walltime": walltime,
        "interface": interface,
        "python": py_exe,
        "job_script_prologue": prologue,
        "scheduler_options": scheduler_opts,
        "processes": 1,
        "worker_extra_args": worker_extra,
        "name": effective_name,
    }

    sig = inspect.signature(SLURMCluster.__init__)
    if "log_directory" in sig.parameters:
        cluster_kwargs["log_directory"] = effective_log_dir
    else:
        extra_directives.append(f"--output={effective_log_dir}/{effective_name}-%j.out")
        extra_directives.append(f"--error={effective_log_dir}/{effective_name}-%j.err")

    cluster_kwargs["job_extra_directives"] = extra_directives

    logger.info(
        f"[Dask-SLURM] Creating SLURMCluster: name='{effective_name}', queue='{queue}', "
        f"account='{account}', cores={cores_per_worker}, mem={mem}, walltime={walltime}, "
        f"log_dir='{effective_log_dir}'"
    )

    cluster = SLURMCluster(**cluster_kwargs)

    # Scale cluster
    if adaptive:
        cluster.adapt(minimum=int(min_workers), maximum=int(max_workers))
        logger.info(f"[Dask-SLURM] Dynamically adapted cluster ({min_workers} .. {max_workers} workers).")
    else:
        cluster.scale(int(max_workers))
        logger.info(f"[Dask-SLURM] Scaled cluster to fixed {max_workers} workers.")

    return cluster


def get_dask_client(
    parallel_mode: str = "dask-slurm",
    scheduler_address: Optional[str] = None,
    stage_name: str = "transient",
    ms_path: Optional[str] = None,
    beam: Optional[Union[str, int]] = None,
    sbid: Optional[Union[str, int]] = None,
    dask_job_name: Optional[str] = None,
    slurm_log_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    slurm_partition: Optional[str] = None,
    slurm_account: Optional[str] = None,
    slurm_cores_per_worker: int = 1,
    slurm_mem: str = "16GB",
    slurm_walltime: str = "02:30:00",
    slurm_job_extra: Optional[Sequence[str]] = None,
    slurm_interface: Optional[str] = None,
    min_workers: int = 0,
    max_workers: int = 16,
    local_workers: int = 4,
    local_memory_limit: str = "8GB",
    threads_per_worker: int = 1,
    dask_scheduler: str = "processes",
    allowed_failures: int = 25,
    target_duration: str = "3600s",
) -> Tuple[Any, Optional[Any]]:
    """
    Unified entrypoint to initialize a Dask Client + Cluster across FastDUCC
    and Lotrun Periodicity pipelines.
    
    Returns:
      (client, cluster) tuple. cluster is None if connecting to an existing scheduler.
    """
    if Client is None:
        raise ImportError(
            "dask.distributed is not installed. Please install dask[distributed] "
            "or run in an HPC virtual environment where distributed is available."
        )

    if scheduler_address is not None and str(scheduler_address).strip():
        addr = str(scheduler_address).strip()
        logger.info(f"Connecting to existing Dask scheduler at {addr}...")
        return Client(addr), None

    if parallel_mode == "dask-slurm":
        if SLURMCluster is None:
            logger.warning("dask_jobqueue not installed. Falling back to LocalCluster.")
            parallel_mode = "dask-local"

    if parallel_mode == "dask-slurm":
        job_name = derive_dask_job_name(
            stage=stage_name,
            ms_path=ms_path,
            beam=beam,
            sbid=sbid,
            explicit_name=dask_job_name,
        )
        log_directory = resolve_dask_log_directory(
            log_dir=slurm_log_dir,
            output_dir=output_dir,
            subdir="dask_workers",
        )

        cluster = setup_slurm_cluster(
            queue=slurm_partition,
            account=slurm_account,
            cores_per_worker=slurm_cores_per_worker,
            mem=slurm_mem,
            walltime=slurm_walltime,
            job_name=job_name,
            log_directory=log_directory,
            job_extra_directives=slurm_job_extra,
            interface=slurm_interface,
            python_path=sys.executable,
            min_workers=min_workers,
            max_workers=max_workers,
            adaptive=True,
            allowed_failures=allowed_failures,
            target_duration=target_duration,
        )
        client = Client(cluster)
        return client, cluster

    elif parallel_mode == "dask-local":
        if LocalCluster is None:
            raise ImportError("dask.distributed LocalCluster is not available.")
        n_w = min(int(local_workers), max(1, os.cpu_count() or 4))
        logger.info(f"Launching Dask LocalCluster ({n_w} workers, memory_limit={local_memory_limit})...")
        cluster = LocalCluster(
            n_workers=n_w,
            threads_per_worker=int(threads_per_worker),
            processes=(dask_scheduler == "processes"),
            memory_limit=local_memory_limit,
            dashboard_address=":0",
        )
        client = Client(cluster)
        return client, cluster

    else:
        raise ValueError(f"Unknown parallel_mode: '{parallel_mode}'. Expected 'dask-slurm' or 'dask-local'.")
