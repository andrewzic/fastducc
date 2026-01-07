import ducc_fast_imager_numba as ducc
import argparse
import glob
import os
import shutil
import sys
from typing import Iterable, Tuple, List, Dict, Any, Optional
from dataclasses import dataclass

from numba import njit, prange
import numpy as np
import math
from tqdm import tqdm

import astropy.constants as const
import astropy.units as u
from astropy.visualization.wcsaxes import WCSAxes
from astropy.wcs import WCS
from astropy.io import fits
from astropy.table import Table

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from scipy.ndimage import maximum_filter

from casacore.tables import table
try:
    import ducc0
except Exception as e:
    raise RuntimeError('ducc0 is required') from e


def main():
    parser = argparse.ArgumentParser(description='Image MS in time chunks using ducc0 wgridder')
    parser.add_argument('--msname', required=True, help='Path to Measurement Set')
    parser.add_argument('--corr-mode', choices=['average','stokesI','single'], default='single',
                        help='Correlation handling mode (default: single)')
    parser.add_argument('--basis', choices=['auto','linear','circular'], default='linear',
                        help='Basis for stokesI (default: auto)')
    parser.add_argument('--single-pol', default='XX',
                        help='Single pol to image when corr-mode=single (default: XX)')
    parser.add_argument('--data-column', default='DATA', help='Which data column to image from the measurement set (default: DATA)')
    parser.add_argument('--npix-x', type=int, default=384)
    parser.add_argument('--npix-y', type=int, default=384)
    parser.add_argument('--pixsize-arcsec', type=float, default=22.0,
                        help='Pixel size (arcsec); applied to both axes (default: 22.0)')
    parser.add_argument('--epsilon', type=float, default=1e-6)
    parser.add_argument('--do-wgridding', dest='do_wgridding', action='store_true', default=True)
    parser.add_argument('--no-wgridding', dest='do_wgridding', action='store_false')
    parser.add_argument('--nthreads', type=int, default=0)
    parser.add_argument('--verbosity', type=int, default=0)
    args = parser.parse_args()

    # Convert arcsec to radians
    pix_rad = args.pixsize_arcsec / 206265.0

    # Discover total number of time chunks via iter
    t_main = table(args.msname, readonly=True)
    colnames = set(t_main.colnames())    


    cont_img = ducc.continuum_image(msname=args.msname,
                                    t_main=t_main,
                                    corr_mode=args.corr_mode,
                                    basis=args.basis,
                                    single_pol=args.single_pol,
                                    data_column=args.data_column,
                                    npix_x=args.npix_x,
                                    npix_y=args.npix_y,
                                    pixsize_x=pix_rad,
                                    pixsize_y=pix_rad,
                                    epsilon=args.epsilon,
                                    do_wgridding=args.do_wgridding,
                                    nthreads=args.nthreads,
                                    verbosity=args.verbosity,
                                    )

    pix_rad = args.pixsize_arcsec / 206265.0
    ra0_rad, dec0_rad, used_field = ducc.get_phase_center(args.msname, field_name=None)

    _wcs = ducc._build_fullframe_wcs(npix_x=args.npix_x, npix_y=args.npix_y,
                                     ra0_rad=ra0_rad, dec0_rad=dec0_rad,
                                     pixscale_rad=pix_rad,
                                     ra_sign=-1, dec_sign=-1, radesys="ICRS", equinox=2000.0)
    
    fits.writeto(
        "./ducc.image_TNN.fits",
        data = cont_img, header=_wcs.to_header(),
        overwrite=True
    )

    
if __name__ == '__main__':
    main()
