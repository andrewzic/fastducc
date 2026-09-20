# -*- coding: utf-8 -*-
"""
fastducc CLI dispatcher.

Provides:
  fastducc            -> existing fastducc.fastducc_run:main
  fastducc aggregate  -> field-level candidate aggregation across beams
"""

import sys

def _run_imaging_search(argv):
    """
    Forward argv to fastducc.fastducc_run.main() by temporarily setting sys.argv.
    This preserves the existing argparse behavior in fastducc_run.py.
    """
    from fastducc import fastducc_run

    old_argv = sys.argv
    try:
        sys.argv = ["fastducc"] + list(argv)
        fastducc_run.main()
        return 0
    finally:
        sys.argv = old_argv



def _run_aggregate(argv):
    """
    Parse aggregate args and run candidates.aggregate_observation().
    """
    from fastducc import fastducc_run
    return fastducc_run.aggregate_main(list(argv))

def _run_aggregate_obs(argv):
    """
    Parse aggregate args and run candidates.aggregate_observation().
    """
    from fastducc import fastducc_run
    return fastducc_run.aggregate_obs_main(list(argv))

def _run_periodicity(argv):
    """
    Parse periodicity args and run nufft_periodicity.run_periodicity().
    """
    from fastducc import nufft_periodicity, fastducc_run
    args = fastducc_run.build_cli_periodicity(list(argv))
    nufft_periodicity.run_periodicity(args)
    return 0

def main(argv=None):
    """
    Entry point for the console script "fastducc".

    If first argument is "aggregate", run aggregation.
    If first argument is "periodicity", run periodicity search.
    Otherwise, treat everything as arguments for the imaging/search pipeline.
    """
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) > 0 and argv[0] == "aggregate":
        return _run_aggregate(argv[1:])
    elif len(argv) > 0 and argv[0] == "aggregate_obs":
        return _run_aggregate_obs(argv[1:])
    elif len(argv) > 0 and argv[0] == "periodicity":
        return _run_periodicity(argv[1:])

    return _run_imaging_search(argv)


if __name__ == "__main__":
    raise SystemExit(main())
