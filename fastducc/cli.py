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

def main(argv=None):
    """
    Entry point for the console script "fastducc".

    If first argument is "aggregate", run aggregation.
    Otherwise, treat everything as arguments for the imaging/search pipeline.
    """
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) > 0 and argv[0] == "aggregate":
        return _run_aggregate(argv[1:])
    elif len(argv) > 0 and argv[0] == "aggregate_obs":
        return _run_aggregate_obs(argv[1:])

    return _run_imaging_search(argv)


if __name__ == "__main__":
    raise SystemExit(main())
