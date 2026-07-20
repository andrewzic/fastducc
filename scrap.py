


def _build_tan_wcs_for_snippet(spatial_size: int,
                               ra_rad: float,
                               dec_rad: float,
                               pixscale_rad: float,
                               ra_sign: int = -1) -> WCS:
    """
    Build a simple 2D TAN WCS centered at the detection pixel (snippet center).
    Parameters
    ----------
    spatial_size : int
        Size (N) for the square snippet (N x N).
    ra_rad, dec_rad : float
        Candidate RA/Dec in radians (world coordinate at the snippet center).
    pixscale_rad : float
        Pixel scale in radians/pixel (same as used in ducc wgridder).
    ra_sign : int
        Sign for RA pixel scale in CD matrix. Default -1 (RA increases to the left
        as per astronomical WCS convention). Set +1 if prefer RA to increase to the right.
    Returns
    -------
    w : astropy.wcs.WCS
        2D TAN WCS.
    """
    w = WCS(naxis=2)
    # Reference pixel (1-indexed in FITS): the center of the snippet
    crpix_x = (spatial_size + 1) / 2.0
    crpix_y = (spatial_size + 1) / 2.0
    w.wcs.crpix = [crpix_x, crpix_y]
    # Reference world coord at the snippet center
    w.wcs.crval = [math.degrees(ra_rad), math.degrees(dec_rad)]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cunit = ["deg", "deg"]
    # CD matrix with no rotation, only scale. Convert scale to deg/pix.
    sc = math.degrees(pixscale_rad)
    w.wcs.cd = np.array([[ra_sign * sc, 0.0],
                         [0.0,            sc]], dtype=float)
    return w


def extract_candidate_snippets(
    times: np.ndarray,
    cube: np.ndarray,                       
    detections: List[Dict[str, Any]],
    *,
    spatial_size: int = 50,                 
    time_factor: int = 5,
    pad_mode: str = "constant",             
    pad_value: float = 0.0,                 
    return_indices: bool = True,            
) -> List[Dict[str, Any]]:

    if cube.ndim != 3:
        raise ValueError("cube must be (T, Ny, Nx)")
    T, Ny, Nx = cube.shape
    if times.shape[0] != T:
        raise ValueError("times length must match cube time axis")
    if spatial_size < 1 or time_factor < 1:
        raise ValueError("spatial_size and time_factor must be >= 1")

    half_sp = spatial_size // 2
    out: List[Dict[str, Any]] = []

    # find center time index robustly
    def _resolve_center_idx(det: Dict[str, Any]) -> int:
        if "center_idx" in det:
            return int(det["center_idx"])
        # fallback: nearest time index to time_center
        tc = float(det["time_center"])
        k = np.searchsorted(times, tc)
        if k == 0:
            return 0
        if k >= T:
            return T - 1
        # choose nearest of k-1, k
        return (k - 1) if (abs(times[k-1] - tc) <= abs(times[k] - tc)) else k

    for det in detections:
        y = int(det["y"]); x = int(det["x"])
        w = max(1, int(det["width_samples"]))       # width in samples
        t_center = _resolve_center_idx(det)
        t_len = max(1, time_factor * w)
        half_t = t_len // 2

        # --- Compute desired index ranges (time, y, x) ---
        t0 = t_center - half_t
        t1 = t0 + t_len                             # exclusive

        y0 = y - half_sp
        y1 = y0 + spatial_size                      # exclusive

        x0 = x - half_sp
        x1 = x0 + spatial_size

        # --- Clip to valid indices ---
        t0_clip = max(0, t0)
        t1_clip = min(T, t1)

        y0_clip = max(0, y0)
        y1_clip = min(Ny, y1)

        x0_clip = max(0, x0)
        x1_clip = min(Nx, x1)

        # --- Compute required padding (front/back) to keep fixed size ---
        pad_t_front = t0_clip - t0          # if t0<0 -> positive padding needed
        pad_t_back  = t1 - t1_clip          # if t1>T -> positive padding needed

        pad_y_top   = y0_clip - y0
        pad_y_bot   = y1 - y1_clip

        pad_x_left  = x0_clip - x0
        pad_x_right = x1 - x1_clip

        # --- Slice the valid region ---
        sub_cube  = cube[t0_clip:t1_clip, y0_clip:y1_clip, x0_clip:x1_clip]
        sub_times = times[t0_clip:t1_clip]

        # --- Pad to fixed shapes (t_len, spatial_size, spatial_size) ---
        pad_widths = (
            (pad_t_front, pad_t_back),
            (pad_y_top,   pad_y_bot),
            (pad_x_left,  pad_x_right)
        )

        if pad_mode == "constant":
            snippet_cube = np.pad(sub_cube, pad_widths, mode="constant", constant_values=pad_value)
            # pad times with NaN (no extrapolation)
            snippet_times = np.pad(sub_times, (pad_t_front, pad_t_back), mode="constant", constant_values=np.nan)
        elif pad_mode == "edge":
            snippet_cube = np.pad(sub_cube, pad_widths, mode="edge")
            # for times, duplicate edges
            if sub_times.size == 0:
                # pathological case (w > T): fill with NaN
                snippet_times = np.full((t_len,), np.nan, dtype=times.dtype)
            else:
                front_vals = np.full((pad_t_front,), sub_times[0], dtype=sub_times.dtype)
                back_vals  = np.full((pad_t_back,),  sub_times[-1], dtype=sub_times.dtype)
                snippet_times = np.concatenate([front_vals, sub_times, back_vals], axis=0)
        else:
            raise ValueError(f"Unknown pad_mode='{pad_mode}'")

        # Sanity: enforce exact shapes
        if snippet_cube.shape != (t_len, spatial_size, spatial_size):
            raise RuntimeError(f"Snippet cube has unexpected shape {snippet_cube.shape}")
        if snippet_times.shape != (t_len,):
            raise RuntimeError(f"Snippet times has unexpected shape {snippet_times.shape}")



        lc = snippet_cube[:, half_sp, half_sp]  # detection pixel (center)
        # Restrict to the original window inside the snippet:
        # Map [t0, t1) in global indices to snippet indices:
        win_start = half_t - (w // 2)
        win_end   = win_start + w  # exclusive
        win_start = max(0, win_start)
        win_end   = min(t_len, win_end)
        
        if win_end > win_start:
            local_win = lc[win_start:win_end]
            rel_peak  = np.nanargmax(local_win)    # argmax inside window
            det_peak_snippet_idx = win_start + rel_peak
            # Optionally update the candidate time center to this peak:
            det_time_peak = snippet_times[det_peak_snippet_idx]
            det["time_center_peak"] = float(det_time_peak) if np.isfinite(det_time_peak) else float(det["time_center"])
            det["center_idx_peak"]  = det["center_idx"] + (det_peak_snippet_idx - half_t)



        
        # --- Prepare output record ---
        rec: Dict[str, Any] = {
            "candidate": det,
            "snippet_cube": snippet_cube,
            "snippet_times": snippet_times,
        }

        if return_indices:
            rec["meta"] = {
                "time_indices": {"desired": (t0, t1), "clipped": (t0_clip, t1_clip), "pad": (pad_t_front, pad_t_back)},
                "y_indices":    {"desired": (y0, y1), "clipped": (y0_clip, y1_clip), "pad": (pad_y_top, pad_y_bot)},
                "x_indices":    {"desired": (x0, x1), "clipped": (x0_clip, x1_clip), "pad": (pad_x_left, pad_x_right)},
                "center_idx": t_center,
                "snippet_shape": (t_len, spatial_size, spatial_size),
            }

        out.append(rec)

    return out



def save_candidate_lightcurves(
    times: np.ndarray,
    cube: np.ndarray,
    candidate: Dict[str, Any],
    out_prefix: str,
    *,
    spatial_size: int = 50,
    save_format: str = "npz",            # "npz" or "ascii"
    center_policy: str = "right",        # "right" uses w//2; "left" uses (w-1)//2
    cmap: str = "gray",
    dpi: int = 200,
) -> Dict[str, str]:
    """
    Extract and save the full-resolution and boxcar-smoothed light curves for a candidate,
    and produce a diagnostic matplotlib figure as requested.

    Parameters
    ----------
    times : (T,) ndarray
        Time stamps for the cube (float). Must be monotonic.
    cube : (T, Ny, Nx) ndarray
        Full-resolution image cube.
    candidate : dict
        Candidate record with (at minimum):
          - "x","y" (int) : pixel position in the full image
          - "width_samples" (int) : boxcar width
          - Preferred time tagging keys:
              * "t0_idx" (int) : window start index in full resolution (recommended)
              * OR "center_idx" (int) : original center sample (fallback)
    out_prefix : str
        Prefix for output files (e.g., "/path/run01_cand0007").
    spatial_size : int
        Size of snippet cutout (square).
    save_format : {"npz","ascii"}
        File format for light curves.
    center_policy : {"right","left"}
        Policy for mapping window start to center time:
          - "right": center = t0 + w//2 (default; consistent with prior fixes)
          - "left":  center = t0 + (w-1)//2
    cmap : str
        Colormap for images.
    dpi : int
        DPI for the saved figure.

    Returns
    -------
    dict with paths:
        {
          "lc_full": <file>,
          "lc_boxcar": <file>,
          "figure": <file>
        }
    """
    # --- Validate inputs ---
    if cube.ndim != 3:
        raise ValueError("cube must be (T, Ny, Nx)")
    T, Ny, Nx = cube.shape
    if times.shape[0] != T:
        raise ValueError("times length must match cube time axis")
    if spatial_size < 1:
        raise ValueError("spatial_size must be >= 1")
    if save_format not in ("npz", "ascii"):
        raise ValueError("save_format must be 'npz' or 'ascii'")

    # --- Candidate fields ---
    y = int(candidate["y"])
    x = int(candidate["x"])
    w = max(1, int(candidate.get("width_samples", 1)))

    # Where is the detection in time?
    # Prefer smoothed index (window start), else derive from center_idx.
    if "t0_idx" in candidate:
        t0_idx = int(candidate["t0_idx"])
    else:
        cidx = int(candidate.get("center_idx", 0))
        t0_idx = cidx - (w // 2) if (center_policy == "right") else cidx - ((w - 1) // 2)
        t0_idx = max(0, min(t0_idx, T - w))  # ensure valid start

    offset = (w // 2) if (center_policy == "right") else ((w - 1) // 2)
    # Smoothed center sample maps to full-res index:
    t_full_center = t0_idx + offset
    t_full_center = max(0, min(t_full_center, T - 1))

    # --- Full-resolution light curve at the candidate pixel ---
    lc_full = cube[:, y, x].astype(np.float64, copy=False)

    # --- Boxcar-smoothed light curve (mean over width w) ---
    lc_sm, T_eff = _compute_boxcar_1d(lc_full, w)
    if T_eff > 0:
        # Smoothed time stamps are the times at the center of each window
        times_sm = times[offset:offset + T_eff]
        k_center = t0_idx  # smoothed index
        k_center = max(0, min(k_center, T_eff - 1))
    else:
        times_sm = np.empty((0,), dtype=times.dtype)
        k_center = 0  # dummy

    # --- Determine display scaling for images from the full-res frame ---
    frame_full = cube[t_full_center]
    vmin = np.nanpercentile(frame_full, 5.0)
    vmax = np.nanpercentile(frame_full, 99.5)

    # --- Prepare snippet cutout around the candidate (from the same full-res frame) ---
    half_sp = spatial_size // 2
    y0 = max(0, y - half_sp)
    y1 = min(Ny, y0 + spatial_size)
    x0 = max(0, x - half_sp)
    x1 = min(Nx, x0 + spatial_size)
    snippet = frame_full[y0:y1, x0:x1]

    # --- Save light curves ---
    out_full = f"{out_prefix}_lc_full.{ 'npz' if save_format=='npz' else 'txt' }"
    out_box  = f"{out_prefix}_lc_boxcar_w{w}.{ 'npz' if save_format=='npz' else 'txt' }"

    if save_format == "npz":
        np.savez(out_full, time=times, flux=lc_full)
        np.savez(out_box, time=times_sm, flux=lc_sm, width=w)
    else:
        # ASCII: two columns "time flux" with header
        with open(out_full, "w") as f:
            f.write("# time  flux(full_res)\n")
            for t, v in zip(times, lc_full):
                f.write(f"{t:.9f} {v:.9e}\n")
        with open(out_box, "w") as f:
            f.write(f"# time  flux(boxcar_mean_w={w})\n")
            for t, v in zip(times_sm, lc_sm):
                f.write(f"{t:.9f} {v:.9e}\n")

    # --- Build the figure ---
    out_fig = f"{out_prefix}_lightcurves.png"
    fig = plt.figure(figsize=(10, 8), dpi=dpi)

    # Top-left: full image at detection time
    ax_full = fig.add_subplot(2, 2, 1)
    im_full = ax_full.imshow(frame_full, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax_full.plot([x], [y], marker="o", ms=5, color="cyan")
    ax_full.set_title(f"Full image @ t={times[t_full_center]:.3f}s (idx {t_full_center})")
    fig.colorbar(im_full, ax=ax_full, fraction=0.046, pad=0.04)

    # Top-right: snippet cutout around candidate
    ax_cut = fig.add_subplot(2, 2, 2)
    im_cut = ax_cut.imshow(snippet, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax_cut.plot([min(half_sp, x - x0)], [min(half_sp, y - y0)], marker="o", ms=5, color="cyan")
    ax_cut.set_title(f"Cutout (size {snippet.shape[0]}x{snippet.shape[1]})")
    fig.colorbar(im_cut, ax=ax_cut, fraction=0.046, pad=0.04)

    # Bottom-left (full width): full-res light curve
    ax_lc_full = fig.add_subplot(3, 1, 2)
    ax_lc_full.plot(times, lc_full, color="tab:blue", lw=1.5)
    ax_lc_full.axvline(times[t_full_center], color="red", ls="--", lw=1.0, label="Detection time")
    ax_lc_full.set_ylabel("Flux (full-res)")
    ax_lc_full.grid(True, alpha=0.3)
    ax_lc_full.legend(loc="best")

    # Bottom-right (full width): boxcar-smoothed light curve
    ax_lc_box = fig.add_subplot(3, 1, 3)
    if T_eff > 0:
        ax_lc_box.plot(times_sm, lc_sm, color="tab:green", lw=1.5)
        ax_lc_box.axvline(times_sm[k_center], color="red", ls="--", lw=1.0, label="Smoothed detection time")
        ax_lc_box.legend(loc="best")
    else:
        ax_lc_box.text(0.5, 0.5, f"Width {w} > T; no smoothed LC", transform=ax_lc_box.transAxes,
                       ha="center", va="center", color="red")
    ax_lc_box.set_xlabel("Time (s)")
    ax_lc_box.set_ylabel(f"Flux (boxcar mean, w={w})")
    ax_lc_box.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)

    return {"lc_full": out_full, "lc_boxcar": out_box, "figure": out_fig}





        var_detections, var_snr = detection.variance_search(
            times, cube,
            threshold_sigma=args.var_threshold_sigma if hasattr(args, "var_threshold_sigma") else 5.0,
            return_snr_cubes=True,
            keep_top_k=args.var_keep_top_k if hasattr(args, "var_keep_top_k") else None,
            valid_mask=None,
            subtract_mean_per_pixel=True,             # recommended for variability
            spatial_estimator="clipped_rms",
            clip_sigma=args.var_clip_sigma if hasattr(args, "var_clip_sigma") else 3.0,
        )
    
        
        
        wcs_full = ducc_wcs._build_fullframe_wcs(
            npix_x=args.npix_x, npix_y=args.npix_y,
            ra0_rad=ra0_rad, dec0_rad=dec0_rad,
            pixscale_rad=pix_rad,            # radians/pixel from CLI
            ra_sign=-1,                      # RA increases to the left (astronomical convention)
            dec_sign=-1,
            radesys="ICRS", equinox=2000.0
        )

        
        # Convert WCS to header and add helpful metadata
        hdr = wcs_full.to_header()
        hdr["BUNIT"]   = "std"                  # standard deviation units (arbitrary)
        hdr["COMMENT"] = "Per-pixel std across time (variance map)"
        # (Optional) ensure CDELT/PC are present even if to_header() varies between versions
        hdr["CDELT1"] = wcs_full.wcs.cdelt[0]
        hdr["CDELT2"] = wcs_full.wcs.cdelt[1]
        hdr["PC1_1"]  = 1.0; hdr["PC1_2"] = 0.0
        hdr["PC2_1"]  = 0.0; hdr["PC2_2"] = 1.0
        if getattr(wcs_full.wcs, "radesys", None):
            hdr["RADESYS"] = wcs_full.wcs.radesys
        #if getattr(wcs_full.wcs, "equinox", None) is not None:
        #    hdr["EQUINOX"] = wcs_full.wcs.equinox

        # Choose an output path (per chunk)
        
        std_fits_path = os.path.join(candidates_dir, f"{ms_base}_chunk_{start:06d}_std_map.fits")
        fits.writeto(std_fits_path, data=std_map.astype(np.float32), header=hdr, overwrite=True)
        print(f"[Chunk {chunk_id}] wrote variance std-map FITS -> {std_fits_path}")

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
        
        print(f"[Chunk {chunk_id}] variance search found {len(var_detections_nms)} candidates after NMS")

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
            ra0_rad, dec0_rad, used_field = ms_utils.get_phase_center(args.msname, field_name=None)
            
            var_root = f"{chunk_prefix_root(start)}_var"
            
            for i, cand in enumerate(annotated_var):
                # 1) std-map snippet
                std_snip = candidates.make_stdmap_snippet(std_map, cand, spatial_size=50)
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
                    std_map=std_map, use_std_images=True
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











            


    # Finalization
    finalize_welford(cfg, wf, times, cube)
    consolidate_catalogues(cfg)

        
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
            
