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
from astropy.table import Table, vstack

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from scipy.ndimage import maximum_filter

from casacore.tables import table
try:
    import ducc0
except Exception as e:
    raise RuntimeError('ducc0 is required') from e


from fastducc import wcs as ducc_wcs
from fastducc import filters, kernels, candidates, ms_utils, detection, imaging

def main():
    parser = argparse.ArgumentParser(description='Image MS in time chunks using ducc0 wgridder')
    parser.add_argument('--msname', required=True, help='Path to Measurement Set')
    parser.add_argument('--chunk-size', type=int, default=1000,
                        help='Number of time samples per chunk (default: 1000)')
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
    parser.add_argument('--boxcar-widths', nargs='+', type=int,
                        help='Widths (in samples) to perform boxcar search over (default: [1, 2, 4, 8, 16, 32, 64])',
                        default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument('--threshold-sigma', type=float, default=8.0, help="S/N threshold to use for detections (default: 8.0)")
    parser.add_argument('--do-plot', action='store_true')
    parser.add_argument('--var-threshold-sigma', type=float, default=8.0)
    parser.add_argument('--var-keep-top-k', type=int, default=None)
    parser.add_argument('--var-clip-sigma', type=float, default=3.0)
    parser.add_argument('--var-nms-radius', type=int, default=10)
    # --- Search toggles ---
    parser.add_argument(
        '--enable-boxcar', dest='enable_boxcar', action='store_true',
        help='Enable per-width boxcar transient search (default: enabled)'
    )
    parser.add_argument(
        '--disable-boxcar', dest='enable_boxcar', action='store_false',
        help='Disable per-width boxcar transient search'
    )
    parser.set_defaults(enable_boxcar=True)
    parser.add_argument(
        '--enable-var', dest='enable_var', action='store_true',
        help='Enable variance (std-map/Welford) variability search (default: enabled)'
    )
    parser.add_argument(
        '--disable-var', dest='enable_var', action='store_false',
        help='Disable variance (std-map/Welford) variability search'
    )
    parser.set_defaults(enable_var=True)
    args = parser.parse_args()

    # Convert arcsec to radians
    pix_rad = args.pixsize_arcsec / 206265.0
    ra0_rad, dec0_rad, used_field = ms_utils.get_phase_center(args.msname, field_name=None)
    
    # Discover total number of time chunks via iter
    t_main = table(args.msname, readonly=True)
    colnames = set(t_main.colnames())    
    time_col = 'TIME_CENTROID' if 'TIME_CENTROID' in set(t_main.colnames()) else 'TIME'
    it = t_main.iter([time_col], sort=True)
    total_chunks = sum(1 for _ in it)
    #it.close()
    #t_main.close()

    print(f"Found {total_chunks} time chunks in MS: {args.msname}")

    start = 0
    chunk_size = max(1, args.chunk_size)
    chunk_id = 0

    # --- Running Welford aggregates (global across the whole observation) ---
    Ny, Nx = args.npix_y, args.npix_x
    w_count = np.zeros((Ny, Nx), dtype=np.int64)      # number of valid time samples seen per pixel
    w_mean  = np.zeros((Ny, Nx), dtype=np.float64)    # running mean per pixel
    w_M2    = np.zeros((Ny, Nx), dtype=np.float64)    # running sum of squares of differences

    # ---- Derive output directory: <MS parent>/candidates and a base tag from MS name ----
    ms_path = os.path.abspath(args.msname.rstrip('/'))
    ms_dir  = os.path.dirname(ms_path)             # e.g., /fred/oz451/azic/data/blahblah
    ms_tag  = os.path.basename(ms_path)            # e.g., example.ms
    ms_base = ms_tag[:-3] if ms_tag.endswith('.ms') else ms_tag  # e.g., example
    
    candidates_dir = os.path.join(ms_dir, "candidates")
    # Helper for per-chunk prefix roots (consistent naming)
    def chunk_prefix_root(start_idx: int) -> str:
        # e.g., /.../candidates/example_chunk_000000 (use start index for uniqueness)
        return os.path.join(candidates_dir, f"{ms_base}_chunk_{start_idx:06d}")
    all_prefix_root = os.path.join(candidates_dir, f"{ms_base}_all")
    
    os.makedirs(candidates_dir, exist_ok=True)
    
    #t_main = table(args.msname, readonly=True)
    while start < total_chunks:
        end = min(start + chunk_size - 1, total_chunks - 1)
        
        print(f"[Chunk {chunk_id}] imaging time_idx {start}..{end}")
        times, cube = imaging.image_time_samples(msname=args.msname,
                                         t_main=t_main,
                                         start_time_idx=start,
                                         end_time_idx=end,
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
                                         do_plot=args.do_plot,
                                         )
        
        print(f"[Chunk {chunk_id}] partial variance search (std-map) for long-timescale variability")
        
        # Update global aggregates with samples from this chunk
        kernels.welford_update_cube(w_count, w_mean, w_M2, cube, ignore_nan=True)

        # Optional: build the current (partial) global std-map to use for detection or diagnostics
        #std_map_partial = welford_finalize_std(w_count, w_M2, ddof=1)

        #print(len(times), cube.shape)
        widths = args.boxcar_widths
        print(f"[Chunk {chunk_id}] boxcar searching with widths {widths}")

        if args.enable_boxcar:
        
            detections, snr_cubes = detection.boxcar_search_time(times, cube,
                                                                 widths=widths,
                                                                 widths_in_seconds=False,      # set True if widths are in seconds
                                                                 threshold_sigma=args.threshold_sigma,
                                                                 return_snr_cubes=True,
                                                                 keep_top_k=50,                # keep top 50 per width
                                                                 std_mode="spatial_per_window",
                                                                 subtract_mean_per_pixel=True  # high-pass in time per pixel
                                                                 )


            # 1) NMS per width (spatial, optional temporal)
            print(f"[Chunk {chunk_id}] filtering {len(detections)} candidates with non-max suppression")
            detections_by_width = filters.nms_snr_maps_per_width(
                snr_cubes,
                times,
                threshold_sigma=args.threshold_sigma,
                spatial_radius=10,
                time_radius=2,          # set >0 to suppress across neighboring time slices too
                valid_mask=None
            )

            if len(detections_by_width) > 0:
                # 2) Merge across widths and apply cross-width NMS
                print(f"[Chunk {chunk_id}] filtering {len(detections)} grouping {len(detections_by_width)} filtered candidates by width")        
                final_detections = filters.group_filter_across_widths(
                    detections_by_width, times,
                    spatial_radius=10,
                    time_radius=0,         # set >0 to suppress across nearby center_idx groups
                    policy="max_snr",       # or "prefer_short"/"prefer_long"
                    max_per_time_group=1,   # keep only 1 per time center (typical)
                    ny_nx=(cube.shape[1], cube.shape[2])  # pass explicit Ny,Nx for speed
                )
            else:
                final_detections = []

            print(f"chunk start:{start} end:{end}: Found {len(final_detections)} candidates")

            if len(final_detections) > 0:

                annotated = ducc_wcs.annotate_candidates_with_sky_coords(
                    msname=args.msname,
                    final_detections=final_detections,
                    npix_x=args.npix_x,
                    npix_y=args.npix_y,
                    pixsize_x=pix_rad,
                    pixsize_y=pix_rad,
                    flip_u=True, flip_v=True,         # change to True if set flip_* in vis2dirty
                    field_name=None                     # or pass a specific field name
                )
                for c in annotated[:50]:  # preview a few
                    print(f"cand @ (x,y)=({c['x']},{c['y']})  l={c['l']:.6g}  m={c['m']:.6g}  "
                          f"RA={c['ra_hms']}  Dec={c['dec_dms']}  field={c['phase_center_field']}")

                for d in final_detections:
                    d["chunk_id"] = int(chunk_id)
                    d["algo"] = "boxcar"

                t = candidates.candidates_to_astropy_table(annotated)

                box_root = f"{chunk_prefix_root(start)}_boxcar"
                print(f"boxcar save prefix: {box_root}")
                box_csv = f"{box_root}_candidates.csv"
                box_vot = f"{box_root}_candidates.vot"

                candidates.save_candidates_table(t,
                                                 csv_path=box_csv,
                                                 vot_path=box_vot
                                      )

                print(f"Saved {len(t)} candidates to {box_csv} and {box_vot}")

                for i, cand in enumerate(final_detections):
                    paths = candidates.save_candidate_lightcurves(
                        times=times,
                        cube=cube,
                        candidate=cand,
                        out_prefix=f"{box_root}_cand_{i:03d}_lc",
                        spatial_size=50,
                        save_format="npz",                 # or "ascii"
                        center_policy="right",
                        cmap="gray",
                        dpi=180,
                        # WCS inputs:
                        npix_x=args.npix_x, npix_y=args.npix_y,
                        ra0_rad=ra0_rad, dec0_rad=dec0_rad,
                        pix_rad=pix_rad,
                        ra_sign=-1,                        # RA increases to the left (astronomical convention)
                        dec_sign=-1,
                        radesys="ICRS", equinox=None
                    )
                    print(f"[cand {i:03d}] lightcurve files:", paths)

                #import ipdb; ipdb.set_trace();
                snippets = candidates.extract_candidate_snippets(
                    times, cube, annotated,
                    spatial_size=50,
                    time_factor=5,
                    pad_mode="constant",   # or "edge"
                    pad_value=0.0,
                    return_indices=True
                )


                for i, s in enumerate(snippets):
                    snip_cand_prefix = f"{box_root}_cand_{i:03d}_snip"
                    cand = s["candidate"]
                    # Pull world coords from the annotated candidate dict
                    ra_rad = float(cand["ra_rad"])
                    dec_rad = float(cand["dec_rad"])
                    out = candidates.save_candidate_snippet_products(
                        snippet_rec=s,
                        out_prefix=snip_cand_prefix,
                        pixscale_rad=pix_rad,
                        ra_rad=ra_rad,
                        dec_rad=dec_rad,
                        ra_sign=-1,            # RA increases to the left (standard WCS). Set +1 if desired.
                        dec_sign=-1,
                        cmap="gray",
                        gif_fps=6,
                        dpi=180
                    )
                    print(f"[cand {i:03d}] saved:", out)

        
        start = end + 1
        chunk_id += 1

    std_map_full = kernels.welford_finalise_std(w_count, w_M2, ddof=1)

    # Full-frame WCS (you already have _build_fullframe_wcs and ms_base/candidates_dir)
    wcs_full = ducc_wcs._build_fullframe_wcs(
        npix_x=args.npix_x, npix_y=args.npix_y,
        ra0_rad=ra0_rad, dec0_rad=dec0_rad,
        pixscale_rad=pix_rad,
        ra_sign=-1, radesys="ICRS", equinox=None
    )
    hdr = wcs_full.to_header()
    hdr["BUNIT"] = "std"
    hdr["COMMENT"] = "Global per-pixel standard deviation across full observation"
    hdr["CDELT1"] = wcs_full.wcs.cdelt[0]
    hdr["CDELT2"] = wcs_full.wcs.cdelt[1]
    hdr["PC1_1"]  = 1.0; hdr["PC1_2"] = 0.0
    hdr["PC2_1"]  = 0.0; hdr["PC2_2"] = 1.0
    
    full_std_fits = os.path.join(candidates_dir, f"{ms_base}_std_map_full.fits")
    fits.writeto(full_std_fits, data=std_map_full.astype(np.float32), header=hdr, overwrite=True)
    print(f"[Final] wrote full-observation std-map -> {full_std_fits}")

    if args.enable_var:
        var_detections, var_snr = detection.variance_search_welford(
            std_map_full,
            threshold_sigma=args.var_threshold_sigma,
            return_snr_image=True,
            keep_top_k=args.var_keep_top_k,
            valid_mask=None,
            spatial_estimator="clipped_rms",
            clip_sigma=args.var_clip_sigma,
            subtract_mean_of_std_map=True
        )
        std_map = var_snr["std"].astype(np.float64) if (var_snr is not None and "std" in var_snr) else np.nanstd(cube, axis=0)

        if len(var_detections) > 0:
            print(f"[Chunk {chunk_id}] variance map: spatial NMS on SNR")
            var_detections_nms = filters.nms_snr_map_2d(
                snr_2d=std_map / max(np.nanstd(std_map), 1e-9),  # normalized SNR if desired
                base_detections=var_detections,
                threshold_sigma=args.var_threshold_sigma,
                spatial_radius=args.var_nms_radius,
                valid_mask=None,
                times=times, cube=cube,
                time_tag_policy="peak_absdev"  # or "none" / "peak_flux"
            )
        else:
            var_detections_nms = []

        print(f"Variance search found {len(var_detections_nms)} candidates after NMS")

        if len(var_detections_nms) > 0:
            # Annotate variance candidates with sky coordinates
            annotated_var = ducc_wcs.annotate_candidates_with_sky_coords(
                msname=args.msname,
                final_detections=var_detections_nms,
                npix_x=args.npix_x,
                npix_y=args.npix_y,
                pixsize_x=pix_rad,
                pixsize_y=pix_rad,
                flip_u=True, flip_v=True,
                field_name=None
            )

            # For each variance candidate:
            #  - make a std-map snippet (time length = 1)
            #  - save lightcurves & figure (using std-map images in top panels)
            #  - save snippet products (GIF=1 frame, PNG, FITS) from std-map
            print(f"[Chunk {chunk_id}] generating std-map snippets & lightcurves for variance candidates")

            var_root = f"{all_prefix_root}_var"

            for i, cand in enumerate(annotated_var):
                # 1) std-map snippet
                std_snip = candidates.make_stdmap_snippet(std_map_full, cand, spatial_size=50)
                # 2) lightcurves + figure (use std-map images in top panels)
                lc_paths = candidates.save_candidate_lightcurves(
                    times=times, cube=cube, candidate=cand,
                    out_prefix=f"{var_root}_cand_{i:03d}_lc",
                    spatial_size=50, save_format="npz",
                    center_policy="right", cmap="gray", dpi=180,
                    # WCS / scale
                    npix_x=args.npix_x, npix_y=args.npix_y,
                    ra0_rad=ra0_rad, dec0_rad=dec0_rad,
                    pix_rad=pix_rad,
                    ra_sign=-1, dec_sign=-1, radesys="ICRS", equinox=None,
                    std_map=std_map_full, use_std_images=True
                )
                # 3) snippet products (GIF/PNG/FITS) using std-map snippet
                #    (GIF will have a single frame; PNG+FITS get TAN WCS and correct CDELT)
                prod_paths = candidates.save_candidate_snippet_products(
                    snippet_rec=std_snip,
                    out_prefix=f"{var_root}_cand_{i:03d}_snip",
                    pixscale_rad=pix_rad,
                    ra_rad=float(cand["ra_rad"]),
                    dec_rad=float(cand["dec_rad"]),
                    ra_sign=-1, dec_sign=-1, cmap="gray", gif_fps=1, dpi=180
                )
                print(f"[Chunk {chunk_id}] variance cand {i:03d} ->", {**lc_paths, **prod_paths})

                for d in annotated_var:
                    d["chunk_id"] = int(chunk_id)
                    d["algo"] = "variance"

            t_var = candidates.candidates_to_astropy_table(annotated_var)
            var_csv = f"{var_root}_candidates.csv"
            var_vot = f"{var_root}_candidates.vot"
            candidates.save_candidates_table(t_var,
                                             csv_path=var_csv,
                                             vot_path=var_vot
                                             )

            print(f"[Chunk {chunk_id}] wrote {len(t_var)} variance candidates -> {var_csv}, {var_vot}")


    var_pattern = os.path.join(candidates_dir, f"{ms_base}*_var_candidates.csv")
    box_pattern = os.path.join(candidates_dir, f"{ms_base}_chunk_*_boxcar_candidates.csv")

    # Consolidated outputs go to the same candidates dir

    candidates.consolidate_chunk_catalogues(
        ms_base=ms_base,
        out_dir=candidates_dir,
        var_csv_pattern=var_pattern,
        box_csv_pattern=box_pattern,
        remove_chunk_catalogues=True
    )
        
if __name__ == '__main__':
    main()
