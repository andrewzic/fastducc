# -*- coding: utf-8 -*-
"""
fastducc.gpu_detection: CuPy/CUDA-accelerated Welford variance accumulation,
2D local/global noise thresholding, boxcar transient search, and spatial/temporal
non-maximum suppression (NMS).

Operates directly on VRAM-resident dirty image cubes ``(T, Ny, Nx)`` produced by
``fastducc.gpu_transient`` + ``fastducc.gpu_c2r`` to eliminate host transfers
and CPU ``scipy.ndimage.uniform_filter`` bottlenecks.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cnd

    HAS_CUPY = True
    _CUPY_IMPORT_ERROR: Optional[BaseException] = None
except Exception as exc:
    cp = None  # type: ignore[assignment]
    cnd = None  # type: ignore[assignment]
    HAS_CUPY = False
    _CUPY_IMPORT_ERROR = exc

logger = logging.getLogger(__name__)


def _require_cupy(device_id: int = 0) -> None:
    if not HAS_CUPY or cp is None or cnd is None:
        raise RuntimeError(
            "CuPy with cupyx.scipy.ndimage is required for GPU transient detection."
        ) from _CUPY_IMPORT_ERROR
    cp.cuda.Device(int(device_id)).use()


_WELFORD_UPDATE_CUBE_SRC = r"""
extern "C" __global__
void welford_update_cube_kernel(
    long long* __restrict__ count,
    double* __restrict__ mean,
    double* __restrict__ M2,
    double* __restrict__ ema_mean,
    const float* __restrict__ cube,
    const double* __restrict__ alphas,
    int T,
    int Ny,
    int Nx,
    int do_highpass,
    int ignore_nan
) {
    unsigned long long npix = (unsigned long long)Ny * (unsigned long long)Nx;
    unsigned long long gid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;

    for (unsigned long long yx = gid; yx < npix; yx += stride) {
        long long c = count[yx];
        double m = mean[yx];
        double m2 = M2[yx];
        double ema = ema_mean[yx];

        for (int t = 0; t < T; ++t) {
            double val = (double)cube[(unsigned long long)t * npix + yx];
            if ((ignore_nan && !isfinite(val)) || val == 0.0) {
                continue;
            }
            double val_to_use;
            if (do_highpass) {
                double alpha = alphas[t];
                if (isnan(ema)) {
                    ema = val;
                } else {
                    ema = ema * (1.0 - alpha) + val * alpha;
                }
                val_to_use = val - ema;
            } else {
                val_to_use = val;
            }
            c += 1;
            double delta = val_to_use - m;
            m += delta / (double)c;
            m2 += delta * (val_to_use - m);
        }

        count[yx] = c;
        mean[yx] = m;
        M2[yx] = m2;
        ema_mean[yx] = ema;
    }
}
"""

if HAS_CUPY:
    _WELFORD_UPDATE_CUBE_KERNEL = cp.RawKernel(
        _WELFORD_UPDATE_CUBE_SRC, "welford_update_cube_kernel"
    )
else:
    _WELFORD_UPDATE_CUBE_KERNEL = None


def welford_update_cube_gpu(
    d_cube: Any,
    times: np.ndarray,
    *,
    var_highpass_cutoff_sec: float = 0.0,
    ignore_nan: bool = True,
    device_id: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-pixel Welford aggregates ``(count, mean, M2)`` and partial
    standard deviation map ``std_map`` on GPU for a VRAM-resident dirty cube
    ``d_cube (T, Ny, Nx)``.

    Returns host NumPy arrays ``(count, mean, M2, std_map)`` matching
    ``fastducc.kernels.welford_update_cube`` and ``welford_finalise_std(ddof=1)``.
    """
    _require_cupy(device_id)
    with cp.cuda.Device(int(device_id)):
        d_c32 = cp.ascontiguousarray(cp.asarray(d_cube), dtype=cp.float32)
        if d_c32.ndim != 3:
            raise ValueError(f"d_cube must be 3D (T, Ny, Nx); got shape {d_c32.shape}.")
        T, Ny, Nx = (int(d_c32.shape[0]), int(d_c32.shape[1]), int(d_c32.shape[2]))

        do_highpass = float(var_highpass_cutoff_sec) > 0.0
        alphas_host = np.zeros(T, dtype=np.float64)
        if do_highpass:
            # Matches core._process_variance_chunk_dm EMA initialisation (nanmean over time)
            d_ema = cp.nanmean(d_c32, axis=0, dtype=cp.float64)
            dt_median = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
            last_t = np.nan
            for i, t_val in enumerate(times):
                if np.isnan(last_t):
                    alphas_host[i] = 1.0 - np.exp(-dt_median / float(var_highpass_cutoff_sec))
                else:
                    dt_step = max(0.0, float(t_val) - last_t)
                    alphas_host[i] = 1.0 - np.exp(-dt_step / float(var_highpass_cutoff_sec))
                last_t = float(t_val)
        else:
            d_ema = cp.full((Ny, Nx), cp.nan, dtype=cp.float64)

        d_alphas = cp.asarray(alphas_host, dtype=cp.float64)
        d_count = cp.zeros((Ny, Nx), dtype=cp.int64)
        d_mean = cp.zeros((Ny, Nx), dtype=cp.float64)
        d_M2 = cp.zeros((Ny, Nx), dtype=cp.float64)

        npix = Ny * Nx
        threads = 256
        blocks = min(65535, (npix + threads - 1) // threads)
        _WELFORD_UPDATE_CUBE_KERNEL(
            (blocks,),
            (threads,),
            (
                d_count,
                d_mean,
                d_M2,
                d_ema,
                d_c32,
                d_alphas,
                np.int32(T),
                np.int32(Ny),
                np.int32(Nx),
                np.int32(1 if do_highpass else 0),
                np.int32(1 if ignore_nan else 0),
            ),
        )

        denom = d_count - 1
        d_std = cp.where(denom > 0, cp.sqrt(cp.maximum(d_M2, 0.0) / cp.maximum(denom, 1)), cp.nan)

        count_h = cp.asnumpy(d_count)
        mean_h = cp.asnumpy(d_mean)
        M2_h = cp.asnumpy(d_M2)
        std_h = cp.asnumpy(d_std)
        del d_alphas, d_ema, d_count, d_mean, d_M2, d_std

    return count_h, mean_h, M2_h, std_h


def local_mean_std_2d_batch_gpu(
    d_batch: Any,
    *,
    valid_mask: Optional[Any] = None,
    window_size: Union[int, Tuple[int, int]] = 64,
    min_std: float = 1e-12,
    precomputed_sum_w: Optional[Any] = None,
) -> Tuple[Any, Any]:
    """
    Compute boundary-aware 2D sliding-window local mean and standard deviation
    on GPU for a batch of 2D images ``d_batch (B, Ny, Nx)``.

    Matches ``fastducc.kernels._local_mean_std_2d`` using ``float64`` uniform
    filtering to avoid catastrophic cancellation in ``E[x^2] - E[x]^2``.
    """
    if d_batch.ndim != 3:
        raise ValueError(f"d_batch must be 3D (B, Ny, Nx); got shape {d_batch.shape}.")
    B, Ny, Nx = (int(d_batch.shape[0]), int(d_batch.shape[1]), int(d_batch.shape[2]))
    if isinstance(window_size, int):
        wy, wx = int(window_size), int(window_size)
    else:
        wy, wx = int(window_size[0]), int(window_size[1])
    win_area = float(wy * wx)

    d64 = d_batch.astype(cp.float64, copy=False)
    finite_mask = cp.isfinite(d64)
    all_finite = bool(cp.all(finite_mask))

    if valid_mask is None and all_finite and precomputed_sum_w is not None:
        sum_w = precomputed_sum_w[None, :, :]
        data = d64
    else:
        if valid_mask is not None:
            vm = cp.asarray(valid_mask, dtype=cp.bool_)[None, :, :]
            w_arr = cp.where(vm & finite_mask, 1.0, 0.0)
        else:
            w_arr = cp.where(finite_mask, 1.0, 0.0)
        data = cp.where(w_arr > 0.0, d64, 0.0)
        sum_w = cnd.uniform_filter(w_arr, size=(1, wy, wx), mode="constant", cval=0.0) * win_area

    sum_d = cnd.uniform_filter(data, size=(1, wy, wx), mode="constant", cval=0.0) * win_area
    sum_d2 = cnd.uniform_filter(data * data, size=(1, wy, wx), mode="constant", cval=0.0) * win_area

    valid_count = cp.maximum(sum_w, 1.0)
    mean_map = sum_d / valid_count
    mean_sq_map = sum_d2 / valid_count
    var_map = cp.maximum(0.0, mean_sq_map - mean_map * mean_map)
    std_map = cp.sqrt(var_map)

    invalid = (sum_w < 1.0) | (~cp.isfinite(std_map)) | (std_map < float(min_std))
    if bool(cp.any(invalid)):
        # Per-frame fallback matching _local_mean_std_2d
        for b in range(B):
            if bool(cp.any(invalid[b])):
                if valid_mask is not None:
                    vp = d64[b][cp.asarray(valid_mask, dtype=cp.bool_) & finite_mask[b]]
                else:
                    vp = d64[b][finite_mask[b]]
                g_std = float(cp.nanstd(vp).get()) if vp.size > 0 else 1.0
                if (not np.isfinite(g_std)) or g_std <= float(min_std):
                    g_std = 1.0
                mean_map[b] = cp.where(invalid[b], 0.0, mean_map[b])
                std_map[b] = cp.where(invalid[b], g_std, std_map[b])

    return mean_map, std_map


def _global_sigma_batch_gpu(
    d_batch: Any,
    *,
    spatial_estimator: str = "mad",
    clip_sigma: float = 3.0,
    valid_mask: Optional[Any] = None,
    max_iter: int = 5,
) -> Any:
    """Compute per-frame global spatial sigma ``(B,)`` on GPU (MAD or clipped RMS)."""
    B = int(d_batch.shape[0])
    d64 = d_batch.astype(cp.float64, copy=False)
    if valid_mask is not None:
        vm = cp.asarray(valid_mask, dtype=cp.bool_)
        flat = d64[:, vm]
    else:
        flat = d64.reshape(B, -1)

    med = cp.nanmedian(flat, axis=1, keepdims=True)
    mad = cp.nanmedian(cp.abs(flat - med), axis=1, keepdims=True)
    std = 1.4826 * mad

    if spatial_estimator == "clipped_rms":
        mu = med
        for _ in range(int(max_iter)):
            thr = float(clip_sigma) * std
            keep = cp.isfinite(flat) & (cp.abs(flat - mu) <= thr)
            cnt = cp.sum(keep, axis=1, keepdims=True)
            safe_cnt = cp.maximum(cnt, 1)
            mu_new = cp.sum(cp.where(keep, flat, 0.0), axis=1, keepdims=True) / safe_cnt
            diff = cp.where(keep, flat - mu_new, 0.0)
            std_new = cp.sqrt(cp.sum(diff * diff, axis=1, keepdims=True) / cp.maximum(cnt - 1, 1))
            valid_step = (cnt > 1) & cp.isfinite(std_new) & (std_new > 0.0)
            mu = cp.where(valid_step, mu_new, mu)
            std = cp.where(valid_step, std_new, std)
    elif spatial_estimator != "mad":
        raise ValueError(f"Unknown spatial_estimator='{spatial_estimator}'")

    out = std.ravel()
    out = cp.where(cp.isfinite(out) & (out > 0.0), out, 1.0)
    return out


def boxcar_search_and_nms_gpu(
    times: np.ndarray,
    d_cube: Any,
    widths: Iterable[int],
    *,
    threshold_sigma: float = 8.0,
    spatial_radius: int = 6,
    time_radius: int = 2,
    valid_mask: Optional[np.ndarray] = None,
    subtract_mean_per_pixel: bool = True,
    std_mode: str = "spatial_per_window",
    spatial_estimator: str = "mad",
    clip_sigma: float = 3.0,
    use_local_threshold: bool = True,
    local_window_size: int = 64,
    dm: float = 0.0,
    t_batch_size: int = 64,
    device_id: int = 0,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    GPU-accelerated boxcar transient search + per-width spatial/temporal NMS
    directly on a VRAM-resident dirty image cube ``d_cube (T, Ny, Nx)``.

    Returns ``detections_by_width`` (dict ``width_samples -> list[det]``)
    matching ``detection.boxcar_search_time`` + ``filters.nms_snr_maps_per_width``,
    ready for ``filters.group_filter_across_widths``.
    """
    _require_cupy(device_id)
    if std_mode != "spatial_per_window":
        raise ValueError(f"GPU boxcar search currently supports std_mode='spatial_per_window'; got '{std_mode}'.")

    with cp.cuda.Device(int(device_id)):
        d_c32 = cp.ascontiguousarray(cp.asarray(d_cube), dtype=cp.float32)
        if d_c32.ndim != 3:
            raise ValueError(f"d_cube must be 3D (T, Ny, Nx); got shape {d_c32.shape}.")
        T, Ny, Nx = (int(d_c32.shape[0]), int(d_c32.shape[1]), int(d_c32.shape[2]))
        if len(times) != T:
            raise ValueError(f"times length ({len(times)}) != cube time axis ({T}).")

        dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
        widths_samples = [max(1, int(w)) for w in widths if 1 <= int(w) <= T]
        detections_by_width: Dict[int, List[Dict[str, Any]]] = {}
        if not widths_samples:
            return detections_by_width

        d_valid_mask = cp.asarray(valid_mask, dtype=cp.bool_) if valid_mask is not None else None

        # Precompute frame/pixel validity cumulative sums for O(1) window gating
        d_is_zero_frame = cp.all(d_c32 == 0.0, axis=(1, 2))
        d_zero_frame_csum = cp.zeros(T + 1, dtype=cp.int32)
        cp.cumsum(d_is_zero_frame.astype(cp.int32), out=d_zero_frame_csum[1:])

        d_invalid_pix = (d_c32 == 0.0) | (~cp.isfinite(d_c32))
        d_invalid_pix_csum = cp.zeros((T + 1, Ny, Nx), dtype=cp.int32)
        cp.cumsum(d_invalid_pix.astype(cp.int32), axis=0, out=d_invalid_pix_csum[1:])

        # Optional per-pixel temporal mean subtraction (matching detection.boxcar_search_time)
        d_data = d_c32.astype(cp.float64)
        if subtract_mean_per_pixel:
            d_mean_map = cp.mean(d_data, axis=0, keepdims=True)
            d_data = cp.where(d_invalid_pix, 0.0, d_data - d_mean_map)
            del d_mean_map

        d_csum = cp.zeros((T + 1, Ny, Nx), dtype=cp.float64)
        cp.cumsum(d_data, axis=0, out=d_csum[1:])
        del d_data, d_invalid_pix

        # Precompute boundary weight sum_w once for constant all-valid local stats
        precomp_sum_w = None
        if use_local_threshold and d_valid_mask is None:
            wy = wx = int(local_window_size)
            ones_2d = cp.ones((1, Ny, Nx), dtype=cp.float64)
            precomp_sum_w = (
                cnd.uniform_filter(ones_2d, size=(1, wy, wx), mode="constant", cval=0.0)[0]
                * float(wy * wx)
            )
            del ones_2d

        t_blk = max(1, int(t_batch_size))
        r_sp = max(1, int(spatial_radius))
        r_t = max(0, int(time_radius))
        thresh = float(threshold_sigma)

        for w in widths_samples:
            T_eff = T - w + 1
            raw_hits: List[Tuple[int, int, int, float]] = []

            for t0 in range(0, T_eff, t_blk):
                t1 = min(T_eff, t0 + t_blk)
                B = t1 - t0

                # Windowed boxcar sum S: (B, Ny, Nx)
                S_blk = d_csum[t0 + w : t1 + w] - d_csum[t0 : t1]

                if use_local_threshold:
                    mu_loc, sigma_loc = local_mean_std_2d_batch_gpu(
                        S_blk,
                        valid_mask=d_valid_mask,
                        window_size=local_window_size,
                        precomputed_sum_w=precomp_sum_w,
                    )
                    snr_blk = ((S_blk - mu_loc) / sigma_loc).astype(cp.float32)
                    del mu_loc, sigma_loc
                else:
                    sig_w = _global_sigma_batch_gpu(
                        S_blk,
                        spatial_estimator=spatial_estimator,
                        clip_sigma=clip_sigma,
                        valid_mask=d_valid_mask,
                    )
                    snr_blk = (S_blk / sig_w[:, None, None]).astype(cp.float32)
                    del sig_w
                del S_blk

                # Gate any window containing a zero-padded frame or zero/NaN pixel
                zero_frame_in_win = (d_zero_frame_csum[t0 + w : t1 + w] - d_zero_frame_csum[t0 : t1]) > 0
                invalid_pix_in_win = (d_invalid_pix_csum[t0 + w : t1 + w] - d_invalid_pix_csum[t0 : t1]) > 0
                bad_mask = zero_frame_in_win[:, None, None] | invalid_pix_in_win
                if d_valid_mask is not None:
                    bad_mask = bad_mask | (~d_valid_mask[None, :, :])
                snr_blk = cp.where(bad_mask, cp.float32(0.0), snr_blk)
                del zero_frame_in_win, invalid_pix_in_win, bad_mask

                # Fast-path: skip NMS & host copy if no pixel exceeds threshold
                if not bool(cp.any(snr_blk >= thresh)):
                    del snr_blk
                    continue

                # 2D spatial local-maximum pre-filter on GPU matching filters.nms_snr_maps_per_width
                work = cp.where(snr_blk >= thresh, snr_blk, cp.float32(-np.inf))
                local_max = cnd.maximum_filter(work, size=(1, r_sp, r_sp), mode="nearest")
                peaks_mask = (work >= local_max) & cp.isfinite(work)
                del local_max

                t_rel, ys, xs = cp.where(peaks_mask)
                if t_rel.size > 0:
                    snrs = work[t_rel, ys, xs]
                    t_rel_h = cp.asnumpy(t_rel)
                    ys_h = cp.asnumpy(ys)
                    xs_h = cp.asnumpy(xs)
                    snrs_h = cp.asnumpy(snrs)
                    for tr, y_i, x_i, s_v in zip(t_rel_h, ys_h, xs_h, snrs_h):
                        raw_hits.append((t0 + int(tr), int(y_i), int(x_i), float(s_v)))
                del work, peaks_mask, snr_blk

            # Greedy spatial + temporal NMS on sparse host hits (exact match to filters.nms_snr_maps_per_width)
            dets_w: List[Dict[str, Any]] = []
            if raw_hits:
                # Process chronologically (t0 ascending), descending SNR within each t0
                raw_hits.sort(key=lambda item: (item[0], -item[3]))
                occ_time: Dict[int, np.ndarray] = {}
                for t_hit, y, x, s_val in raw_hits:
                    # Check temporal + spatial occupancy
                    blocked = False
                    for dt_step in range(-r_t, r_t + 1):
                        t2 = t_hit + dt_step
                        occ_map = occ_time.get(t2)
                        if occ_map is not None and not occ_map[y, x]:
                            blocked = True
                            break
                    if blocked:
                        continue

                    t_end_excl = t_hit + w
                    time_start = float(times[t_hit]) - 0.5 * dt
                    time_end = float(times[t_end_excl - 1]) + 0.5 * dt
                    duration = time_end - time_start
                    center_idx = t_hit + (w // 2)
                    time_center = 0.5 * (time_start + time_end)

                    dets_w.append({
                        "y": int(y),
                        "x": int(x),
                        "snr": float(s_val),
                        "dm": float(dm),
                        "width_samples": int(w),
                        "t0_idx": int(t_hit),
                        "t1_idx_excl": int(t_end_excl),
                        "center_idx": int(center_idx),
                        "time_start": time_start,
                        "time_end": time_end,
                        "time_center": float(time_center),
                        "duration": float(duration),
                    })

                    y0 = max(0, y - r_sp)
                    y1 = min(Ny, y + r_sp + 1)
                    x0 = max(0, x - r_sp)
                    x1 = min(Nx, x + r_sp + 1)
                    for dt_step in range(-r_t, r_t + 1):
                        t2 = t_hit + dt_step
                        if 0 <= t2 < T_eff:
                            occ_map = occ_time.get(t2)
                            if occ_map is None:
                                occ_map = np.ones((Ny, Nx), dtype=bool)
                                occ_time[t2] = occ_map
                            occ_map[y0:y1, x0:x1] = False

            detections_by_width[w] = dets_w

        del d_csum, d_zero_frame_csum, d_invalid_pix_csum, d_is_zero_frame, precomp_sum_w

    return detections_by_width


def variance_search_welford_gpu(
    std_map: Any,
    *,
    threshold_sigma: float = 8.0,
    keep_top_k: Optional[int] = None,
    valid_mask: Optional[np.ndarray] = None,
    spatial_estimator: str = "clipped_rms",
    clip_sigma: float = 3.0,
    subtract_mean_of_std_map: bool = True,
    use_local_threshold: bool = True,
    local_window_size: int = 64,
    dm: float = 0.0,
    device_id: int = 0,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    GPU-accelerated variability detection on a 2D Welford std-map ``(Ny, Nx)``.

    Matches ``fastducc.detection.variance_search_welford``.
    """
    _require_cupy(device_id)
    with cp.cuda.Device(int(device_id)):
        d_std = cp.asarray(std_map, dtype=cp.float64)
        if d_std.ndim != 2:
            raise ValueError(f"std_map must be 2D (Ny, Nx); got shape {d_std.shape}.")
        Ny, Nx = int(d_std.shape[0]), int(d_std.shape[1])
        d_vm = cp.asarray(valid_mask, dtype=cp.bool_) if valid_mask is not None else None

        if use_local_threshold:
            mu_loc, sig_loc = local_mean_std_2d_batch_gpu(
                d_std[None, :, :],
                valid_mask=d_vm,
                window_size=local_window_size,
            )
            if subtract_mean_of_std_map:
                d_snr = (d_std - mu_loc[0]) / sig_loc[0]
            else:
                d_snr = d_std / sig_loc[0]
        else:
            sig_g = float(
                _global_sigma_batch_gpu(
                    d_std[None, :, :],
                    spatial_estimator=spatial_estimator,
                    clip_sigma=clip_sigma,
                    valid_mask=d_vm,
                )[0].get()
            )
            if subtract_mean_of_std_map:
                mu_g = float(cp.nanmean(d_std).get())
                d_snr = (d_std - mu_g) / sig_g
            else:
                d_snr = d_std / sig_g

        if d_vm is not None:
            d_snr = cp.where(d_vm, d_snr, 0.0)

        snr_host = cp.asnumpy(d_snr.astype(cp.float32))
        ys_d, xs_d = cp.where(d_snr >= float(threshold_sigma))
        if ys_d.size == 0:
            return [], snr_host

        snr_vals = cp.asnumpy(d_snr[ys_d, xs_d])
        std_vals = cp.asnumpy(d_std[ys_d, xs_d])
        ys_h = cp.asnumpy(ys_d)
        xs_h = cp.asnumpy(xs_d)

    order = np.argsort(snr_vals)[::-1]
    if keep_top_k is not None and len(order) > int(keep_top_k):
        order = order[: int(keep_top_k)]

    detections: List[Dict[str, Any]] = []
    for idx in order:
        detections.append({
            "y": int(ys_h[idx]),
            "x": int(xs_h[idx]),
            "dm": float(dm),
            "snr": float(snr_vals[idx]),
            "std": float(std_vals[idx]),
            "time_start": 0,
            "time_end": 99999,
            "time_center": 9999,
            "duration": 999,
            "center_idx": 9999,
            "width_samples": 999,
        })
    return detections, snr_host


__all__ = [
    "HAS_CUPY",
    "boxcar_search_and_nms_gpu",
    "local_mean_std_2d_batch_gpu",
    "variance_search_welford_gpu",
    "welford_update_cube_gpu",
]
