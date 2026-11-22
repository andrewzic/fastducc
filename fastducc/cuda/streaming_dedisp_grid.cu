// -*- coding: utf-8 -*-
// fastducc/cuda/streaming_dedisp_grid.cu
//
// CUDA C++17 kernels for fused visibility staging, brute-force channel
// dedispersion, direct-to-grid Hermitian half-plane rasterisation, and
// per-cell weight normalisation in fastducc.
//
// Sign and grid conventions (strictly matches ducc0.wgridder.vis2grid with
// flip_u=False, flip_v=False and fastducc Stage 3 c2r imaging):
//   - No sign inversion on MS (u_m, v_m).
//   - Grid dimensions: grid_nu (default npix_x), nv (default npix_y),
//     u_half = grid_nu / 2 + 1.
//   - Cell spacings: du = 1.0 / (grid_nu * pix_rad),
//                    dv = 1.0 / (nv * pix_rad).
//   - Continuous coordinates:
//       u_f = (u_m * (freq_hz / C_LIGHT)) / du
//       v_f = (v_m * (freq_hz / C_LIGHT)) / dv
//   - Hermitian half-plane folding maps u < 0 onto (-u, -v) with V -> conj(V),
//     and wraps v modulo nv into [0, nv).

namespace {

constexpr double kSpeedOfLightMps = 299792458.0;

__device__ __forceinline__ bool is_finite_vis(const float2 v, const float w) {
    return (w > 0.0f) && isfinite(w) && isfinite(v.x) && isfinite(v.y);
}

__device__ __forceinline__ bool passes_uv_distance(
    const double u_m,
    const double v_m,
    const double min_uv_distance_m
) {
    if (min_uv_distance_m <= 0.0) {
        return true;
    }
    if (!isfinite(u_m) || !isfinite(v_m)) {
        return false;
    }
    return hypot(u_m, v_m) >= min_uv_distance_m;
}

__device__ __forceinline__ bool is_snapshot_valid(
    const int t,
    const int nchan,
    const int collapse_channels,
    const int min_valid_channels,
    const int* __restrict__ raw_valid_chans,
    const int* __restrict__ dedisp_valid_chans
) {
    if (nchan <= 0) {
        return false;
    }
    const int req_chans = (min_valid_channels > 0)
        ? ((min_valid_channels < nchan) ? min_valid_channels : nchan)
        : 1;
    if (collapse_channels != 0) {
        return (raw_valid_chans[t] >= req_chans) && (dedisp_valid_chans[t] >= 1);
    }
    return dedisp_valid_chans[t] >= req_chans;
}

__device__ __forceinline__ void accumulate_grid_cell(
    const int t,
    const int nt_out,
    const int nv,
    const int u_half,
    const long long u_cell,
    const long long v_idx,
    const float re,
    const float im,
    const float w,
    const int* __restrict__ uv_to_m,
    const int sparse_output,
    float2* __restrict__ grid_vis,
    float* __restrict__ grid_wgt
) {
    if (w <= 0.0f || u_cell < 0 || u_cell >= static_cast<long long>(u_half)) {
        return;
    }
    const size_t uv_offset =
        static_cast<size_t>(v_idx) * static_cast<size_t>(u_half) + static_cast<size_t>(u_cell);

    size_t dst_idx = 0ULL;
    if (uv_to_m != nullptr) {
        const int m_idx = uv_to_m[uv_offset];
        if (m_idx < 0) {
            return;
        }
        if (sparse_output != 0) {
            dst_idx = static_cast<size_t>(m_idx) * static_cast<size_t>(nt_out) + static_cast<size_t>(t);
        } else {
            dst_idx = static_cast<size_t>(t) * static_cast<size_t>(nv) * static_cast<size_t>(u_half) + uv_offset;
        }
    } else {
        dst_idx = static_cast<size_t>(t) * static_cast<size_t>(nv) * static_cast<size_t>(u_half) + uv_offset;
    }

    atomicAdd(&grid_vis[dst_idx].x, re);
    atomicAdd(&grid_vis[dst_idx].y, im);
    atomicAdd(&grid_wgt[dst_idx], w);
}

__device__ __forceinline__ void splat_nearest_cell(
    const int t,
    const int nt_out,
    const int nv,
    const int u_half,
    const double u_f,
    const double v_f,
    const float re,
    const float im,
    const float w,
    const int* __restrict__ uv_to_m,
    const int sparse_output,
    float2* __restrict__ grid_vis,
    float* __restrict__ grid_wgt
) {
    if (w <= 0.0f || !isfinite(u_f) || !isfinite(v_f)) {
        return;
    }
    if (fabs(u_f) > 1.0e15 || fabs(v_f) > 1.0e15) {
        return;
    }

    long long u_cell = __double2ll_rn(u_f);
    long long v_cell = __double2ll_rn(v_f);
    float out_im = im;

    if (u_cell < 0) {
        u_cell = -u_cell;
        v_cell = -v_cell;
        out_im = -out_im;
    }
    if (u_cell < 0 || u_cell >= static_cast<long long>(u_half)) {
        return;
    }

    const long long nv_ll = static_cast<long long>(nv);
    const long long v_idx = ((v_cell % nv_ll) + nv_ll) % nv_ll;
    accumulate_grid_cell(
        t, nt_out, nv, u_half, u_cell, v_idx, re, out_im, w,
        uv_to_m, sparse_output, grid_vis, grid_wgt
    );
}

__device__ __forceinline__ void splat_bilinear_corners(
    const int t,
    const int nt_out,
    const int nv,
    const int u_half,
    const long long u0,
    const long long v0_wrap,
    const float re00, const float im00, const float w00,
    const float re10, const float im10, const float w10,
    const float re01, const float im01, const float w01,
    const float re11, const float im11, const float w11,
    const int* __restrict__ uv_to_m,
    const int sparse_output,
    float2* __restrict__ grid_vis,
    float* __restrict__ grid_wgt
) {
    if (u0 < 0 || u0 >= static_cast<long long>(u_half)) {
        return;
    }
    const long long nv_ll = static_cast<long long>(nv);
    const long long v1_wrap = (v0_wrap + 1 < nv_ll) ? (v0_wrap + 1) : 0;

    if (w00 > 0.0f) {
        accumulate_grid_cell(
            t, nt_out, nv, u_half, u0, v0_wrap, re00, im00, w00,
            uv_to_m, sparse_output, grid_vis, grid_wgt
        );
    }
    if (w10 > 0.0f && (u0 + 1) < static_cast<long long>(u_half)) {
        accumulate_grid_cell(
            t, nt_out, nv, u_half, u0 + 1, v0_wrap, re10, im10, w10,
            uv_to_m, sparse_output, grid_vis, grid_wgt
        );
    }
    if (w01 > 0.0f) {
        accumulate_grid_cell(
            t, nt_out, nv, u_half, u0, v1_wrap, re01, im01, w01,
            uv_to_m, sparse_output, grid_vis, grid_wgt
        );
    }
    if (w11 > 0.0f && (u0 + 1) < static_cast<long long>(u_half)) {
        accumulate_grid_cell(
            t, nt_out, nv, u_half, u0 + 1, v1_wrap, re11, im11, w11,
            uv_to_m, sparse_output, grid_vis, grid_wgt
        );
    }
}

}  // namespace


/**
 * Stage streamed Measurement Set row buffers directly into GPU 3D staging
 * tensors, performing flagging, minimum UV-distance masking, and correlation
 * collapse ('single', 'average', or 'stokesI') matching
 * ``fastducc.imaging._collapse_correlations``.
 *
 * Launch geometry:
 *   block = (32, 8, 1)  [x: channel within row, y: row within tile]
 *   grid  = (ceil(nrow / 8), min(65535, ceil(nchan / 32)), 1)
 */
extern "C" __global__ void stage_ms_rows_kernel(
    const int* __restrict__ bl_idx,
    const int* __restrict__ t_rel,
    const double* __restrict__ uvw_rows,
    const float2* __restrict__ data_rows,
    const float* __restrict__ wgt_rows,
    const unsigned char* __restrict__ flag_rows,
    const unsigned char* __restrict__ flag_row,
    float2* __restrict__ vis_3d,
    float* __restrict__ wgt_3d,
    double* __restrict__ uvw_3d,
    int nrow,
    int nbl,
    int nchan,
    int ncorr,
    int nt_stage,
    int t_offset,
    int wgt_is_spectrum,
    int corr_mode_id,
    int corr_idx_0,
    int corr_idx_1,
    double min_uv_distance_m
) {
    const int r = blockIdx.x * blockDim.y + threadIdx.y;
    if (r >= nrow) {
        return;
    }

    const int bl = bl_idx[r];
    const int t_dst = t_offset + t_rel[r];
    if (bl < 0 || bl >= nbl || t_dst < 0 || t_dst >= nt_stage) {
        return;
    }

    const size_t uvw_row_off = static_cast<size_t>(r) * 3ULL;
    const double u_m = uvw_rows[uvw_row_off + 0ULL];
    const double v_m = uvw_rows[uvw_row_off + 1ULL];
    const double w_m = uvw_rows[uvw_row_off + 2ULL];

    if (blockIdx.y == 0 && threadIdx.x == 0) {
        const size_t bl_uvw_off = static_cast<size_t>(bl) * 3ULL * static_cast<size_t>(nt_stage);
        uvw_3d[bl_uvw_off + 0ULL * static_cast<size_t>(nt_stage) + static_cast<size_t>(t_dst)] = u_m;
        uvw_3d[bl_uvw_off + 1ULL * static_cast<size_t>(nt_stage) + static_cast<size_t>(t_dst)] = v_m;
        uvw_3d[bl_uvw_off + 2ULL * static_cast<size_t>(nt_stage) + static_cast<size_t>(t_dst)] = w_m;
    }

    const bool row_masked =
        (flag_row != nullptr && flag_row[r] != 0) ||
        !passes_uv_distance(u_m, v_m, min_uv_distance_m);

    const size_t row_chan_stride = static_cast<size_t>(nchan) * static_cast<size_t>(ncorr);
    const size_t row_data_base = static_cast<size_t>(r) * row_chan_stride;
    const size_t row_wgt_base = (wgt_is_spectrum != 0)
        ? row_data_base
        : (static_cast<size_t>(r) * static_cast<size_t>(ncorr));
    const size_t bl_base = static_cast<size_t>(bl) * static_cast<size_t>(nchan);

    for (int c = blockIdx.y * blockDim.x + threadIdx.x; c < nchan; c += gridDim.y * blockDim.x) {
        float2 vis_out = make_float2(0.0f, 0.0f);
        float wgt_out = 0.0f;

        if (!row_masked) {
            const size_t cell_off = row_data_base + static_cast<size_t>(c) * static_cast<size_t>(ncorr);
            const size_t wgt_off = (wgt_is_spectrum != 0)
                ? cell_off
                : row_wgt_base;

            if (corr_mode_id == 0) {
                // Single correlation
                if (corr_idx_0 >= 0 && corr_idx_0 < ncorr) {
                    const size_t p_off = cell_off + static_cast<size_t>(corr_idx_0);
                    if (flag_rows[p_off] == 0) {
                        const float2 v = data_rows[p_off];
                        const float w = wgt_rows[wgt_off + static_cast<size_t>(corr_idx_0)];
                        if (is_finite_vis(v, w)) {
                            vis_out = v;
                            wgt_out = w;
                        }
                    }
                }
            } else if (corr_mode_id == 1) {
                // Weighted average across all correlations
                double sum_re = 0.0;
                double sum_im = 0.0;
                double sum_w = 0.0;
                for (int p = 0; p < ncorr; ++p) {
                    const size_t p_off = cell_off + static_cast<size_t>(p);
                    if (flag_rows[p_off] != 0) {
                        continue;
                    }
                    const float2 v = data_rows[p_off];
                    const float w = wgt_rows[wgt_off + static_cast<size_t>(p)];
                    if (is_finite_vis(v, w)) {
                        const double wd = static_cast<double>(w);
                        sum_re += static_cast<double>(v.x) * wd;
                        sum_im += static_cast<double>(v.y) * wd;
                        sum_w += wd;
                    }
                }
                if (sum_w > 0.0) {
                    const double inv_w = 1.0 / sum_w;
                    vis_out = make_float2(static_cast<float>(sum_re * inv_w), static_cast<float>(sum_im * inv_w));
                    wgt_out = static_cast<float>(sum_w);
                }
            } else if (corr_mode_id == 2) {
                // Stokes I from parallel hands (i1 = corr_idx_0, i2 = corr_idx_1)
                bool present1 = false;
                bool present2 = false;
                float2 v1 = make_float2(0.0f, 0.0f);
                float2 v2 = make_float2(0.0f, 0.0f);
                float w1 = 0.0f;
                float w2 = 0.0f;

                if (corr_idx_0 >= 0 && corr_idx_0 < ncorr) {
                    const size_t p1 = cell_off + static_cast<size_t>(corr_idx_0);
                    if (flag_rows[p1] == 0) {
                        const float2 vt = data_rows[p1];
                        const float wt = wgt_rows[wgt_off + static_cast<size_t>(corr_idx_0)];
                        if (is_finite_vis(vt, wt)) {
                            present1 = true;
                            v1 = vt;
                            w1 = wt;
                        }
                    }
                }
                if (corr_idx_1 >= 0 && corr_idx_1 < ncorr) {
                    const size_t p2 = cell_off + static_cast<size_t>(corr_idx_1);
                    if (flag_rows[p2] == 0) {
                        const float2 vt = data_rows[p2];
                        const float wt = wgt_rows[wgt_off + static_cast<size_t>(corr_idx_1)];
                        if (is_finite_vis(vt, wt)) {
                            present2 = true;
                            v2 = vt;
                            w2 = wt;
                        }
                    }
                }

                if (present1 && present2) {
                    vis_out = make_float2(0.5f * (v1.x + v2.x), 0.5f * (v1.y + v2.y));
                    const double inv_sum = 1.0 / static_cast<double>(w1) + 1.0 / static_cast<double>(w2);
                    wgt_out = (inv_sum > 0.0 && isfinite(inv_sum)) ? static_cast<float>(4.0 / inv_sum) : 0.0f;
                    if (wgt_out <= 0.0f) {
                        vis_out = make_float2(0.0f, 0.0f);
                    }
                } else if (present1) {
                    vis_out = v1;
                    wgt_out = w1;
                } else if (present2) {
                    vis_out = v2;
                    wgt_out = w2;
                }
            }
        }

        const size_t dst_idx = (bl_base + static_cast<size_t>(c)) * static_cast<size_t>(nt_stage) + static_cast<size_t>(t_dst);
        vis_3d[dst_idx] = vis_out;
        wgt_3d[dst_idx] = wgt_out;
    }
}


/**
 * Count raw and dedispersed valid channels per output snapshot t in [0, nt_out).
 * Both raw_valid_chans[nt_out] and dedisp_valid_chans[nt_out] must be
 * zero-initialised prior to launch.
 *
 * Launch geometry:
 *   block = (32, 8, 1)  [x: warp lanes over baselines, y: channel within tile]
 *   grid  = (nt_out, min(65535, ceil(nchan / 8)), 1)
 */
extern "C" __global__ void compute_valid_channels_kernel(
    const float2* __restrict__ vis_3d,
    const float* __restrict__ wgt_3d,
    const double* __restrict__ uvw_3d,
    const int* __restrict__ delays,
    int* __restrict__ raw_valid_chans,
    int* __restrict__ dedisp_valid_chans,
    int nbl,
    int nchan,
    int nt_in,
    int nt_out,
    double min_uv_distance_m
) {
    __shared__ int s_raw_cnt;
    __shared__ int s_dedisp_cnt;

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_raw_cnt = 0;
        s_dedisp_cnt = 0;
    }
    __syncthreads();

    const int t = blockIdx.x;
    if (t >= nt_out) {
        return;
    }

    const int lane = threadIdx.x;
    const size_t bl_vis_stride = static_cast<size_t>(nchan) * static_cast<size_t>(nt_in);
    const size_t bl_uvw_stride = 3ULL * static_cast<size_t>(nt_in);

    for (int c = blockIdx.y * blockDim.y + threadIdx.y; c < nchan; c += gridDim.y * blockDim.y) {
        const int d = delays[c];
        const int t_src = t + d;
        const bool src_in_bounds = (t_src >= 0 && t_src < nt_in);

        bool raw_found = false;
        for (int bl_base = 0; bl_base < nbl; bl_base += 32) {
            const int bl = bl_base + lane;
            bool ok = false;
            if (bl < nbl) {
                const size_t vis_idx =
                    static_cast<size_t>(bl) * bl_vis_stride +
                    static_cast<size_t>(c) * static_cast<size_t>(nt_in) +
                    static_cast<size_t>(t);
                const float w = wgt_3d[vis_idx];
                if (w > 0.0f && isfinite(w)) {
                    const float2 v = vis_3d[vis_idx];
                    if (isfinite(v.x) && isfinite(v.y)) {
                        if (min_uv_distance_m <= 0.0) {
                            ok = true;
                        } else {
                            const size_t u_idx = static_cast<size_t>(bl) * bl_uvw_stride + static_cast<size_t>(t);
                            ok = passes_uv_distance(uvw_3d[u_idx], uvw_3d[u_idx + static_cast<size_t>(nt_in)], min_uv_distance_m);
                        }
                    }
                }
            }
            if (__any_sync(0xffffffffu, ok)) {
                raw_found = true;
                break;
            }
        }

        bool dedisp_found = false;
        if (src_in_bounds) {
            if (d == 0) {
                dedisp_found = raw_found;
            } else {
                for (int bl_base = 0; bl_base < nbl; bl_base += 32) {
                    const int bl = bl_base + lane;
                    bool ok = false;
                    if (bl < nbl) {
                        const size_t vis_idx =
                            static_cast<size_t>(bl) * bl_vis_stride +
                            static_cast<size_t>(c) * static_cast<size_t>(nt_in) +
                            static_cast<size_t>(t_src);
                        const float w = wgt_3d[vis_idx];
                        if (w > 0.0f && isfinite(w)) {
                            const float2 v = vis_3d[vis_idx];
                            if (isfinite(v.x) && isfinite(v.y)) {
                                if (min_uv_distance_m <= 0.0) {
                                    ok = true;
                                } else {
                                    const size_t u_idx = static_cast<size_t>(bl) * bl_uvw_stride + static_cast<size_t>(t_src);
                                    ok = passes_uv_distance(
                                        uvw_3d[u_idx],
                                        uvw_3d[u_idx + static_cast<size_t>(nt_in)],
                                        min_uv_distance_m
                                    );
                                }
                            }
                        }
                    }
                    if (__any_sync(0xffffffffu, ok)) {
                        dedisp_found = true;
                        break;
                    }
                }
            }
        }

        if (lane == 0) {
            if (raw_found) {
                atomicAdd(&s_raw_cnt, 1);
            }
            if (dedisp_found) {
                atomicAdd(&s_dedisp_cnt, 1);
            }
        }
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        if (s_raw_cnt > 0) {
            atomicAdd(&raw_valid_chans[t], s_raw_cnt);
        }
        if (s_dedisp_cnt > 0) {
            atomicAdd(&dedisp_valid_chans[t], s_dedisp_cnt);
        }
    }
}


/**
 * Fused full-channel-collapse dedispersion and direct-to-grid rasterisation
 * kernel (Regime A: collapse_channels=True and nsubbands == 1).
 *
 * Launch geometry:
 *   block = (32, 8, 1)  [x: snapshot time t (coalesced), y: baseline]
 *   grid  = (ceil(nt_out / 32), min(65535, ceil(nbl / 8)), 1)
 */
extern "C" __global__ void dedisperse_and_rasterise_collapsed_kernel(
    const float2* __restrict__ vis_3d,
    const float* __restrict__ wgt_3d,
    const double* __restrict__ uvw_3d,
    const int* __restrict__ delays,
    const int* __restrict__ raw_valid_chans,
    const int* __restrict__ dedisp_valid_chans,
    const int* __restrict__ uv_to_m,
    float2* __restrict__ grid_vis,
    float* __restrict__ grid_wgt,
    int nbl,
    int nchan,
    int nt_in,
    int nt_out,
    int nv,
    int u_half,
    double du,
    double dv,
    double mean_freq_hz,
    double min_uv_distance_m,
    int min_valid_channels,
    int mode,
    int sparse_output
) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nt_out) {
        return;
    }
    if (!is_snapshot_valid(t, nchan, 1, min_valid_channels, raw_valid_chans, dedisp_valid_chans)) {
        return;
    }

    const size_t bl_vis_stride = static_cast<size_t>(nchan) * static_cast<size_t>(nt_in);
    const size_t bl_uvw_stride = 3ULL * static_cast<size_t>(nt_in);
    const double freq_over_c = mean_freq_hz / kSpeedOfLightMps;
    const long long nv_ll = static_cast<long long>(nv);

    for (int bl = blockIdx.y * blockDim.y + threadIdx.y; bl < nbl; bl += gridDim.y * blockDim.y) {
        const size_t base_vis = static_cast<size_t>(bl) * bl_vis_stride;
        const size_t base_uvw = static_cast<size_t>(bl) * bl_uvw_stride;

        const double u_m_t = uvw_3d[base_uvw + static_cast<size_t>(t)];
        const double v_m_t = uvw_3d[base_uvw + static_cast<size_t>(nt_in) + static_cast<size_t>(t)];

        double sum_re = 0.0;
        double sum_im = 0.0;
        double sum_w = 0.0;

        for (int c = 0; c < nchan; ++c) {
            const int d = delays[c];
            const int t_src = t + d;
            if (t_src < 0 || t_src >= nt_in) {
                continue;
            }
            const size_t vis_idx = base_vis + static_cast<size_t>(c) * static_cast<size_t>(nt_in) + static_cast<size_t>(t_src);
            const float w = wgt_3d[vis_idx];
            if (w <= 0.0f || !isfinite(w)) {
                continue;
            }
            const float2 v = vis_3d[vis_idx];
            if (!isfinite(v.x) || !isfinite(v.y)) {
                continue;
            }
            if (min_uv_distance_m > 0.0) {
                const double u_src = (d == 0) ? u_m_t : uvw_3d[base_uvw + static_cast<size_t>(t_src)];
                const double v_src = (d == 0) ? v_m_t : uvw_3d[base_uvw + static_cast<size_t>(nt_in) + static_cast<size_t>(t_src)];
                if (!passes_uv_distance(u_src, v_src, min_uv_distance_m)) {
                    continue;
                }
            }
            const double wd = static_cast<double>(w);
            sum_re += static_cast<double>(v.x) * wd;
            sum_im += static_cast<double>(v.y) * wd;
            sum_w += wd;
        }

        if (sum_w <= 0.0) {
            continue;
        }

        const double u_f = (u_m_t * freq_over_c) / du;
        const double v_f = (v_m_t * freq_over_c) / dv;
        if (!isfinite(u_f) || !isfinite(v_f) || fabs(u_f) > 1.0e15 || fabs(v_f) > 1.0e15) {
            continue;
        }

        const float out_re = static_cast<float>(sum_re);
        float out_im = static_cast<float>(sum_im);
        const float out_w = static_cast<float>(sum_w);

        if (mode == 0) {
            splat_nearest_cell(
                t, nt_out, nv, u_half, u_f, v_f, out_re, out_im, out_w,
                uv_to_m, sparse_output, grid_vis, grid_wgt
            );
        } else {
            double uf = u_f;
            double vf = v_f;
            if (uf < 0.0) {
                uf = -uf;
                vf = -vf;
                out_im = -out_im;
            }
            const long long u0 = static_cast<long long>(floor(uf));
            const long long v0 = static_cast<long long>(floor(vf));
            if (u0 < 0 || u0 >= static_cast<long long>(u_half)) {
                continue;
            }
            const float au = static_cast<float>(uf - static_cast<double>(u0));
            const float av = static_cast<float>(vf - static_cast<double>(v0));
            const float c00 = (1.0f - au) * (1.0f - av);
            const float c10 = au * (1.0f - av);
            const float c01 = (1.0f - au) * av;
            const float c11 = au * av;
            const long long v0_wrap = ((v0 % nv_ll) + nv_ll) % nv_ll;

            splat_bilinear_corners(
                t, nt_out, nv, u_half, u0, v0_wrap,
                out_re * c00, out_im * c00, out_w * c00,
                out_re * c10, out_im * c10, out_w * c10,
                out_re * c01, out_im * c01, out_w * c01,
                out_re * c11, out_im * c11, out_w * c11,
                uv_to_m, sparse_output, grid_vis, grid_wgt
            );
        }
    }
}


// =============================================================================
// DUCC W-Gridder Piecewise Polynomial & W-Phase CUDA Device Functions
// Adapted from DUCC (src/ducc0/math/gridding_kernel.h, wgridder_impl.h,
// and not_yet_integrated/wgridder_sycl.h).
// =============================================================================

namespace {

constexpr double kTwoPiD = 6.283185307179586476925286766559;
constexpr float kTwoPiF = 6.283185307179586476925286766559f;

/**
 * DUCC coordinate-to-grid index mapping (Wgridder::getpix in wgridder_impl.h:685-695).
 * Inputs u_lam, v_lam are in wavelengths.
 */
__device__ __forceinline__ void ducc_getpix(
    const double u_lam,
    const double v_lam,
    const double pixsize_x,
    const double pixsize_y,
    const int grid_nu,
    const int nv,
    const int supp,
    float& ufrac,
    float& vfrac,
    int& iu0,
    int& iv0
) {
    const double ushift = static_cast<double>(supp) * (-0.5) + 1.0 + static_cast<double>(grid_nu);
    const double vshift = static_cast<double>(supp) * (-0.5) + 1.0 + static_cast<double>(nv);
    const int maxiu0 = (grid_nu + supp) - ((supp + 1) / 2);
    const int maxiv0 = (nv + supp) - ((supp + 1) / 2);

    double u = u_lam * pixsize_x;
    u = (u - floor(u)) * static_cast<double>(grid_nu);
    iu0 = min(static_cast<int>(u + ushift) - grid_nu, maxiu0);
    ufrac = static_cast<float>(u - static_cast<double>(iu0));

    double v = v_lam * pixsize_y;
    v = (v - floor(v)) * static_cast<double>(nv);
    iv0 = min(static_cast<int>(v + vshift) - nv, maxiv0);
    vfrac = static_cast<float>(v - static_cast<double>(iv0));
}

/**
 * DUCC w-sign canonicalisation (FixW) and w-phase preconditioning
 * (Wgridder::compute_phases in wgridder_impl.h:1162-1176):
 *   exp(2 * pi * i * imflip * (u * lshift + v * mshift + w * nshift))
 * with range reduction (tmp - floor(tmp)) prior to single-precision trig evaluation.
 */
__device__ __forceinline__ void ducc_fixw_and_phase(
    double& u_lam,
    double& v_lam,
    double& w_lam,
    float& vr,
    float& vi,
    const double lshift,
    const double mshift,
    const double nshift,
    const int do_shift
) {
    float imflip = 1.0f;
    if (w_lam < 0.0) {
        u_lam = -u_lam;
        v_lam = -v_lam;
        w_lam = -w_lam;
        imflip = -1.0f;
    }
    if (do_shift != 0) {
        const double tmp = static_cast<double>(imflip) * (u_lam * lshift + v_lam * mshift + w_lam * nshift);
        const float ang = kTwoPiF * static_cast<float>(tmp - floor(tmp));
        float s = 0.0f;
        float c = 1.0f;
        __sincosf(ang, &s, &c);
        const float nvr = vr * c - vi * s;
        const float nvi = vr * s + vi * c;
        vr = nvr;
        vi = nvi;
    }
    vi *= imflip;
}

/**
 * Register-only separable 2D piecewise polynomial evaluation adapted from
 * DUCC TemplateKernel::eval2 (gridding_kernel.h:456-483) and
 * KernelComputer::compute_uv (wgridder_sycl.h:892-916).
 *
 * ``poly_coeffs`` has shape ``((supp + 1) / 2, deg + 1)`` where
 * ``deg = supp + 3 + (supp & 1)`` is always odd (deg in {7, 9, 11}).
 */
__device__ __forceinline__ void ducc_eval_poly_uv(
    const float ufrac,
    const float vfrac,
    const int supp,
    const int deg,
    const float* __restrict__ poly_coeffs,
    float* __restrict__ ku,
    float* __restrict__ kv
) {
    const float x0 = -2.0f * ufrac + static_cast<float>(supp - 1);
    const float y0 = -2.0f * vfrac + static_cast<float>(supp - 1);
    const float x2 = x0 * x0;
    const float y2 = y0 * y0;
    const int lim = (supp + 1) >> 1;
    const int stride = deg + 1;

    for (int i = 0; i < lim; ++i) {
        const float* __restrict__ row = poly_coeffs + i * stride;
        float tx = row[0];
        float tx2 = row[1];
        float ty = row[0];
        float ty2 = row[1];
        for (int j = 2; j < deg; j += 2) {
            const float c_even = row[j];
            const float c_odd = row[j + 1];
            tx = fmaf(tx, x2, c_even);
            ty = fmaf(ty, y2, c_even);
            tx2 = fmaf(tx2, x2, c_odd);
            ty2 = fmaf(ty2, y2, c_odd);
        }
        const float prod_x = tx * x0;
        const float prod_y = ty * y0;
        ku[i] = tx2 + prod_x;
        kv[i] = ty2 + prod_y;
        ku[supp - 1 - i] = tx2 - prod_x;
        kv[supp - 1 - i] = ty2 - prod_y;
    }
}

/**
 * Register-only separable 3D (u, v, w) piecewise polynomial evaluation adapted
 * from DUCC TemplateKernel::eval2s (gridding_kernel.h:375-434) and
 * KernelComputer::compute_uvw (wgridder_sycl.h:917-952).
 */
__device__ __forceinline__ void ducc_eval_poly_uvw(
    const float ufrac,
    const float vfrac,
    const float z_in,
    int nth,
    const int supp,
    const int deg,
    const float* __restrict__ poly_coeffs,
    float* __restrict__ ku,
    float* __restrict__ kv
) {
    const float x0 = -2.0f * ufrac + static_cast<float>(supp - 1);
    const float y0 = -2.0f * vfrac + static_cast<float>(supp - 1);
    float z0 = (z_in - static_cast<float>(nth)) * 2.0f + static_cast<float>(supp - 1);
    const float x2 = x0 * x0;
    const float y2 = y0 * y0;
    const float z2 = z0 * z0;
    const int lim = (supp + 1) >> 1;
    const int stride = deg + 1;

    if (nth >= lim) {
        nth = supp - 1 - nth;
        z0 = -z0;
    }
    const float* __restrict__ zrow = poly_coeffs + nth * stride;
    float tz = zrow[0];
    float tz2 = zrow[1];
    for (int j = 2; j < deg; j += 2) {
        tz = fmaf(tz, z2, zrow[j]);
        tz2 = fmaf(tz2, z2, zrow[j + 1]);
    }
    const float kw = fmaf(tz, z0, tz2);

    for (int i = 0; i < lim; ++i) {
        const float* __restrict__ row = poly_coeffs + i * stride;
        float tx = row[0];
        float tx2 = row[1];
        float ty = row[0];
        float ty2 = row[1];
        for (int j = 2; j < deg; j += 2) {
            const float c_even = row[j];
            const float c_odd = row[j + 1];
            tx = fmaf(tx, x2, c_even);
            ty = fmaf(ty, y2, c_even);
            tx2 = fmaf(tx2, x2, c_odd);
            ty2 = fmaf(ty2, y2, c_odd);
        }
        const float prod_x = tx * x0;
        const float prod_y = ty * y0;
        ku[i] = (tx2 + prod_x) * kw;
        kv[i] = ty2 + prod_y;
        ku[supp - 1 - i] = (tx2 - prod_x) * kw;
        kv[supp - 1 - i] = ty2 - prod_y;
    }
}

/**
 * Splat a visibility with DUCC 2D piecewise polynomial weights directly onto
 * the Hermitian half-plane u in [0, u_half) (or sparse CSR active cells).
 *
 * Implements the exact DUCC complex2hartley + hartley2complex Hermitian symmetry:
 *   G_sym(u, v) = 0.5 * (G(u, v) + conj(G((-u) mod nu, (-v) mod nv)))
 * so every half-plane cell (including the self-conjugate columns u = 0 and
 * u = grid_nu / 2) matches ducc0.wgridder.vis2grid without allocating the
 * negative half-plane u > grid_nu / 2.
 */
__device__ __forceinline__ void splat_ducc_wgridder_2d(
    const int t,
    const int nt_out,
    const int grid_nu,
    const int nv,
    const int u_half,
    const int supp,
    const int iu0,
    const int iv0,
    const float vr,
    const float vi,
    const float w_val,
    const float* __restrict__ ku,
    const float* __restrict__ kv,
    const int* __restrict__ uv_to_m,
    const int sparse_output,
    float2* __restrict__ grid_vis,
    float* __restrict__ grid_wgt
) {
    for (int cu = 0; cu < supp; ++cu) {
        int iu = iu0 + cu;
        if (iu < 0) {
            iu += grid_nu;
        } else if (iu >= grid_nu) {
            iu -= grid_nu;
        }
        const int xu = (iu == 0) ? 0 : (grid_nu - iu);
        const bool do_pos = (iu < u_half);
        const bool do_neg = (xu < u_half);
        if (!do_pos && !do_neg) {
            continue;
        }

        const float ku_val = 0.5f * ku[cu];
        const float vr_u = vr * ku_val;
        const float vi_u = vi * ku_val;
        const float w_u = w_val * ku_val;

        for (int cv = 0; cv < supp; ++cv) {
            int iv = iv0 + cv;
            if (iv < 0) {
                iv += nv;
            } else if (iv >= nv) {
                iv -= nv;
            }
            const float kv_val = kv[cv];
            const float kw = w_u * kv_val;
            if (kw <= 0.0f) {
                continue;
            }
            const float re_tap = vr_u * kv_val;
            const float im_tap = vi_u * kv_val;

            if (do_pos) {
                accumulate_grid_cell(
                    t, nt_out, nv, u_half,
                    static_cast<long long>(iu),
                    static_cast<long long>(iv),
                    re_tap, im_tap, kw,
                    uv_to_m, sparse_output, grid_vis, grid_wgt
                );
            }
            if (do_neg) {
                const int xv = (iv == 0) ? 0 : (nv - iv);
                accumulate_grid_cell(
                    t, nt_out, nv, u_half,
                    static_cast<long long>(xu),
                    static_cast<long long>(xv),
                    re_tap, -im_tap, kw,
                    uv_to_m, sparse_output, grid_vis, grid_wgt
                );
            }
        }
    }
}

}  // namespace


/**
 * Fused full-channel-collapse dedispersion and DUCC w-gridder direct-to-CSR /
 * half-plane rasterisation kernel (Regime A for mode="wgridder").
 *
 * Launch geometry:
 *   block = (32, 8, 1)  [x: snapshot time t (coalesced), y: baseline]
 *   grid  = (ceil(nt_out / 32), min(65535, ceil(nbl / 8)), 1)
 */
extern "C" __global__ void dedisperse_and_wgridder_collapsed_kernel(
    const float2* __restrict__ vis_3d,
    const float* __restrict__ wgt_3d,
    const double* __restrict__ uvw_3d,
    const int* __restrict__ delays,
    const int* __restrict__ raw_valid_chans,
    const int* __restrict__ dedisp_valid_chans,
    const int* __restrict__ uv_to_m,
    const float* __restrict__ poly_coeffs,
    float2* __restrict__ grid_vis,
    float* __restrict__ grid_wgt,
    int nbl,
    int nchan,
    int nt_in,
    int nt_out,
    int grid_nu,
    int nv,
    int u_half,
    double pixsize_x,
    double pixsize_y,
    double mean_freq_hz,
    double min_uv_distance_m,
    int min_valid_channels,
    int supp,
    int deg,
    double lshift,
    double mshift,
    double nshift,
    int do_shift,
    int flip_u,
    int flip_v,
    int flip_w,
    int sparse_output
) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nt_out) {
        return;
    }
    if (!is_snapshot_valid(t, nchan, 1, min_valid_channels, raw_valid_chans, dedisp_valid_chans)) {
        return;
    }

    const size_t bl_vis_stride = static_cast<size_t>(nchan) * static_cast<size_t>(nt_in);
    const size_t bl_uvw_stride = 3ULL * static_cast<size_t>(nt_in);
    const double freq_over_c = mean_freq_hz / kSpeedOfLightMps;
    const double su = (flip_u != 0) ? -1.0 : 1.0;
    const double sv = (flip_v != 0) ? -1.0 : 1.0;
    const double sw = (flip_w != 0) ? -1.0 : 1.0;

    float ku[8];
    float kv[8];

    for (int bl = blockIdx.y * blockDim.y + threadIdx.y; bl < nbl; bl += gridDim.y * blockDim.y) {
        const size_t base_vis = static_cast<size_t>(bl) * bl_vis_stride;
        const size_t base_uvw = static_cast<size_t>(bl) * bl_uvw_stride;

        const double u_m_t = uvw_3d[base_uvw + static_cast<size_t>(t)];
        const double v_m_t = uvw_3d[base_uvw + static_cast<size_t>(nt_in) + static_cast<size_t>(t)];
        const double w_m_t = uvw_3d[base_uvw + 2ULL * static_cast<size_t>(nt_in) + static_cast<size_t>(t)];

        double sum_re = 0.0;
        double sum_im = 0.0;
        double sum_w = 0.0;

        for (int c = 0; c < nchan; ++c) {
            const int d = delays[c];
            const int t_src = t + d;
            if (t_src < 0 || t_src >= nt_in) {
                continue;
            }
            const size_t vis_idx = base_vis + static_cast<size_t>(c) * static_cast<size_t>(nt_in) + static_cast<size_t>(t_src);
            const float w = wgt_3d[vis_idx];
            if (w <= 0.0f || !isfinite(w)) {
                continue;
            }
            const float2 v = vis_3d[vis_idx];
            if (!isfinite(v.x) || !isfinite(v.y)) {
                continue;
            }
            if (min_uv_distance_m > 0.0) {
                const double u_src = (d == 0) ? u_m_t : uvw_3d[base_uvw + static_cast<size_t>(t_src)];
                const double v_src = (d == 0) ? v_m_t : uvw_3d[base_uvw + static_cast<size_t>(nt_in) + static_cast<size_t>(t_src)];
                if (!passes_uv_distance(u_src, v_src, min_uv_distance_m)) {
                    continue;
                }
            }
            const double wd = static_cast<double>(w);
            sum_re += static_cast<double>(v.x) * wd;
            sum_im += static_cast<double>(v.y) * wd;
            sum_w += wd;
        }

        if (sum_w <= 0.0) {
            continue;
        }

        double u_lam = (u_m_t * su) * freq_over_c;
        double v_lam = (v_m_t * sv) * freq_over_c;
        double w_lam = (w_m_t * sw) * freq_over_c;
        if (!isfinite(u_lam) || !isfinite(v_lam) || !isfinite(w_lam)) {
            continue;
        }

        float vr = static_cast<float>(sum_re);
        float vi = static_cast<float>(sum_im);
        const float out_w = static_cast<float>(sum_w);

        ducc_fixw_and_phase(u_lam, v_lam, w_lam, vr, vi, lshift, mshift, nshift, do_shift);

        float ufrac = 0.0f;
        float vfrac = 0.0f;
        int iu0 = 0;
        int iv0 = 0;
        ducc_getpix(u_lam, v_lam, pixsize_x, pixsize_y, grid_nu, nv, supp, ufrac, vfrac, iu0, iv0);
        ducc_eval_poly_uv(ufrac, vfrac, supp, deg, poly_coeffs, ku, kv);

        splat_ducc_wgridder_2d(
            t, nt_out, grid_nu, nv, u_half, supp, iu0, iv0,
            vr, vi, out_w, ku, kv,
            uv_to_m, sparse_output, grid_vis, grid_wgt
        );
    }
}


/**
 * Fused subband/per-channel dedispersion and DUCC w-gridder direct-to-CSR /
 * half-plane rasterisation kernel (Regime B for mode="wgridder").
 *
 * Launch geometry:
 *   block = (32, 8, 1)  [x: snapshot time t (coalesced), y: baseline]
 *   grid  = (ceil(nt_out / 32), min(65535, ceil(nbl / 8)), 1)
 */
extern "C" __global__ void dedisperse_and_wgridder_subband_kernel(
    const float2* __restrict__ vis_3d,
    const float* __restrict__ wgt_3d,
    const double* __restrict__ uvw_3d,
    const int* __restrict__ delays,
    const int* __restrict__ sub_offsets,
    const double* __restrict__ sub_freqs_hz,
    const int* __restrict__ sub_delays,
    const int* __restrict__ raw_valid_chans,
    const int* __restrict__ dedisp_valid_chans,
    const int* __restrict__ uv_to_m,
    const float* __restrict__ poly_coeffs,
    float2* __restrict__ grid_vis,
    float* __restrict__ grid_wgt,
    int nbl,
    int nchan,
    int nsub_eff,
    int nt_in,
    int nt_out,
    int grid_nu,
    int nv,
    int u_half,
    double pixsize_x,
    double pixsize_y,
    double min_uv_distance_m,
    int min_valid_channels,
    int collapse_channels,
    int exact_uvw,
    int supp,
    int deg,
    double lshift,
    double mshift,
    double nshift,
    int do_shift,
    int flip_u,
    int flip_v,
    int flip_w,
    int sparse_output
) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nt_out) {
        return;
    }
    if (!is_snapshot_valid(t, nchan, collapse_channels, min_valid_channels, raw_valid_chans, dedisp_valid_chans)) {
        return;
    }

    const size_t bl_vis_stride = static_cast<size_t>(nchan) * static_cast<size_t>(nt_in);
    const size_t bl_uvw_stride = 3ULL * static_cast<size_t>(nt_in);
    const double su = (flip_u != 0) ? -1.0 : 1.0;
    const double sv = (flip_v != 0) ? -1.0 : 1.0;
    const double sw = (flip_w != 0) ? -1.0 : 1.0;

    float ku[8];
    float kv[8];

    for (int bl = blockIdx.y * blockDim.y + threadIdx.y; bl < nbl; bl += gridDim.y * blockDim.y) {
        const size_t base_vis = static_cast<size_t>(bl) * bl_vis_stride;
        const size_t base_uvw = static_cast<size_t>(bl) * bl_uvw_stride;

        for (int s = 0; s < nsub_eff; ++s) {
            const int c0 = sub_offsets[s];
            const int c1 = sub_offsets[s + 1];

            double sum_re = 0.0;
            double sum_im = 0.0;
            double sum_w = 0.0;

            for (int c = c0; c < c1; ++c) {
                const int d = delays[c];
                const int t_src = t + d;
                if (t_src < 0 || t_src >= nt_in) {
                    continue;
                }
                const size_t vis_idx = base_vis + static_cast<size_t>(c) * static_cast<size_t>(nt_in) + static_cast<size_t>(t_src);
                const float w = wgt_3d[vis_idx];
                if (w <= 0.0f || !isfinite(w)) {
                    continue;
                }
                const float2 v = vis_3d[vis_idx];
                if (!isfinite(v.x) || !isfinite(v.y)) {
                    continue;
                }
                if (min_uv_distance_m > 0.0) {
                    const double u_src = uvw_3d[base_uvw + static_cast<size_t>(t_src)];
                    const double v_src = uvw_3d[base_uvw + static_cast<size_t>(nt_in) + static_cast<size_t>(t_src)];
                    if (!passes_uv_distance(u_src, v_src, min_uv_distance_m)) {
                        continue;
                    }
                }
                const double wd = static_cast<double>(w);
                sum_re += static_cast<double>(v.x) * wd;
                sum_im += static_cast<double>(v.y) * wd;
                sum_w += wd;
            }

            if (sum_w <= 0.0) {
                continue;
            }

            int t_uvw = t;
            if (exact_uvw != 0) {
                const int ts = t + sub_delays[s];
                t_uvw = (ts < 0) ? 0 : ((ts >= nt_in) ? (nt_in - 1) : ts);
            }
            const double u_m = uvw_3d[base_uvw + static_cast<size_t>(t_uvw)];
            const double v_m = uvw_3d[base_uvw + static_cast<size_t>(nt_in) + static_cast<size_t>(t_uvw)];
            const double w_m = uvw_3d[base_uvw + 2ULL * static_cast<size_t>(nt_in) + static_cast<size_t>(t_uvw)];
            const double freq_over_c = sub_freqs_hz[s] / kSpeedOfLightMps;

            double u_lam = (u_m * su) * freq_over_c;
            double v_lam = (v_m * sv) * freq_over_c;
            double w_lam = (w_m * sw) * freq_over_c;
            if (!isfinite(u_lam) || !isfinite(v_lam) || !isfinite(w_lam)) {
                continue;
            }

            float vr = static_cast<float>(sum_re);
            float vi = static_cast<float>(sum_im);
            const float out_w = static_cast<float>(sum_w);

            ducc_fixw_and_phase(u_lam, v_lam, w_lam, vr, vi, lshift, mshift, nshift, do_shift);

            float ufrac = 0.0f;
            float vfrac = 0.0f;
            int iu0 = 0;
            int iv0 = 0;
            ducc_getpix(u_lam, v_lam, pixsize_x, pixsize_y, grid_nu, nv, supp, ufrac, vfrac, iu0, iv0);
            ducc_eval_poly_uv(ufrac, vfrac, supp, deg, poly_coeffs, ku, kv);

            splat_ducc_wgridder_2d(
                t, nt_out, grid_nu, nv, u_half, supp, iu0, iv0,
                vr, vi, out_w, ku, kv,
                uv_to_m, sparse_output, grid_vis, grid_wgt
            );
        }
    }
}


/**
 * 3D (u, v, w) DUCC piecewise polynomial w-stacking splat kernel for a single
 * w-plane ``p0`` (matching Wgridder::x2grid_c_helper<SUPP, true> in
 * wgridder_impl.h:1178-1273).
 *
 * Also records seed coordinates (uc, vc) and (-uc mod nu, -vc mod nv) into
 * ``fresnel_seed`` on the first plane pass when ``record_seed != 0``.
 *
 * Launch geometry:
 *   block = (32, 8, 1)  [x: subband/channel s, y: baseline bl]
 *   grid  = (ceil(nsub_eff / 32), min(65535, ceil(nbl / 8)), 1)
 */
extern "C" __global__ void ducc_wgridder_3d_plane_splat_kernel(
    const float2* __restrict__ vis_sub,
    const float* __restrict__ wgt_sub,
    const double* __restrict__ uvw_sub_m,
    const double* __restrict__ sub_freqs_hz,
    const float* __restrict__ poly_coeffs,
    float2* __restrict__ plane_grid,
    unsigned char* __restrict__ fresnel_seed,
    int nbl,
    int nsub_eff,
    int grid_nu,
    int nv,
    double pixsize_x,
    double pixsize_y,
    int supp,
    int deg,
    double lshift,
    double mshift,
    double nshift,
    int do_shift,
    int flip_u,
    int flip_v,
    int flip_w,
    int p0,
    double w0,
    double xdw,
    double wshift,
    int record_seed
) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nsub_eff) {
        return;
    }

    const double freq_over_c = sub_freqs_hz[s] / kSpeedOfLightMps;
    const double su = (flip_u != 0) ? -1.0 : 1.0;
    const double sv = (flip_v != 0) ? -1.0 : 1.0;
    const double sw = (flip_w != 0) ? -1.0 : 1.0;
    const int half_supp = supp >> 1;

    float ku[8];
    float kv[8];

    for (int bl = blockIdx.y * blockDim.y + threadIdx.y; bl < nbl; bl += gridDim.y * blockDim.y) {
        const size_t idx = static_cast<size_t>(bl) * static_cast<size_t>(nsub_eff) + static_cast<size_t>(s);
        const float w = wgt_sub[idx];
        if (w <= 0.0f || !isfinite(w)) {
            continue;
        }
        const float2 v = vis_sub[idx];
        if (!isfinite(v.x) || !isfinite(v.y)) {
            continue;
        }

        const size_t uvw_off = (static_cast<size_t>(bl) * static_cast<size_t>(nsub_eff) + static_cast<size_t>(s)) * 3ULL;
        double u_lam = (uvw_sub_m[uvw_off + 0ULL] * su) * freq_over_c;
        double v_lam = (uvw_sub_m[uvw_off + 1ULL] * sv) * freq_over_c;
        double w_lam = (uvw_sub_m[uvw_off + 2ULL] * sw) * freq_over_c;
        if (!isfinite(u_lam) || !isfinite(v_lam) || !isfinite(w_lam)) {
            continue;
        }

        float vr = v.x * w;
        float vi = v.y * w;
        if (vr == 0.0f && vi == 0.0f) {
            continue;
        }

        ducc_fixw_and_phase(u_lam, v_lam, w_lam, vr, vi, lshift, mshift, nshift, do_shift);

        float ufrac = 0.0f;
        float vfrac = 0.0f;
        int iu0 = 0;
        int iv0 = 0;
        ducc_getpix(u_lam, v_lam, pixsize_x, pixsize_y, grid_nu, nv, supp, ufrac, vfrac, iu0, iv0);

        if (record_seed != 0 && fresnel_seed != nullptr) {
            int uc = (iu0 + half_supp) % grid_nu;
            if (uc < 0) uc += grid_nu;
            int vc = (iv0 + half_supp) % nv;
            if (vc < 0) vc += nv;
            const int xuc = (uc == 0) ? 0 : (grid_nu - uc);
            const int xvc = (vc == 0) ? 0 : (nv - vc);
            fresnel_seed[static_cast<size_t>(uc) * static_cast<size_t>(nv) + static_cast<size_t>(vc)] = 1;
            fresnel_seed[static_cast<size_t>(xuc) * static_cast<size_t>(nv) + static_cast<size_t>(xvc)] = 1;
        }

        const int minplane = max(0, static_cast<int>((w_lam + wshift) * xdw));
        if (minplane <= p0 && (minplane + supp) > p0) {
            const int nth = p0 - minplane;
            const float z_in = static_cast<float>(xdw * (w0 - w_lam));
            ducc_eval_poly_uvw(ufrac, vfrac, z_in, nth, supp, deg, poly_coeffs, ku, kv);

            for (int cu = 0; cu < supp; ++cu) {
                int iu = iu0 + cu;
                if (iu < 0) iu += grid_nu;
                else if (iu >= grid_nu) iu -= grid_nu;
                const float ku_val = ku[cu];
                const float vr_u = vr * ku_val;
                const float vi_u = vi * ku_val;
                const size_t row_base = static_cast<size_t>(iu) * static_cast<size_t>(nv);
                for (int cv = 0; cv < supp; ++cv) {
                    int iv = iv0 + cv;
                    if (iv < 0) iv += nv;
                    else if (iv >= nv) iv -= nv;
                    const float kv_val = kv[cv];
                    const size_t cell = row_base + static_cast<size_t>(iv);
                    atomicAdd(&plane_grid[cell].x, vr_u * kv_val);
                    atomicAdd(&plane_grid[cell].y, vi_u * kv_val);
                }
            }
        }
    }
}


/**
 * Accumulate one w-plane's backward 2D FFT grid (grid_nu, nv) into the real
 * image-space accumulator ``rgrid`` (grid_nu, nv) with the exact DUCC w-screen
 * phase factor (Wgridder::grid2rgrid_c_overwrite_wscreen_add in
 * wgridder_impl.h:1750-1808):
 *   nm1 = (-x^2 - y^2) / (sqrt(1 - x^2 - y^2) + 1)
 *   phs = -w * (nm1 + nshift)
 *   ang = 2 * pi * (phs - floor(phs))
 *   rgrid(ix, jx) += grid(ix, jx).real * cos(ang) - grid(ix, jx).imag * sin(ang)
 *
 * Launch geometry:
 *   block = (32, 8, 1)  [x: jx in [0, nv), y: ix in [0, grid_nu)]
 *   grid  = (ceil(nv / 32), ceil(grid_nu / 8), 1)
 */
extern "C" __global__ void ducc_wscreen_accumulate_kernel(
    const float2* __restrict__ plane_grid,
    float* __restrict__ rgrid,
    int grid_nu,
    int nv,
    int nxdirty,
    int nydirty,
    double pixsize_x,
    double pixsize_y,
    double lshift,
    double mshift,
    double nshift,
    double w_plane
) {
    const int jx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ix = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= grid_nu || jx >= nv) {
        return;
    }

    const int hnx = nxdirty / 2;
    const int hny = nydirty / 2;
    int kx = (ix <= grid_nu / 2) ? ix : (ix - grid_nu);
    kx = max(-hnx, min(hnx, kx));
    int ky = (jx <= nv / 2) ? jx : (jx - nv);
    ky = max(-hny, min(hny, ky));

    const double x = lshift + static_cast<double>(kx) * pixsize_x;
    const double y = mshift + static_cast<double>(ky) * pixsize_y;
    const double xsq = x * x;
    const double ysq = y * y;
    const double tmp = 1.0 - xsq - ysq;

    float ang = 0.0f;
    if (tmp > 0.0) {
        const double nm1 = (-xsq - ysq) / (sqrt(tmp) + 1.0);
        const double phs = -w_plane * (nm1 + nshift);
        ang = kTwoPiF * static_cast<float>(phs - floor(phs));
    }

    float s = 0.0f;
    float c = 1.0f;
    __sincosf(ang, &s, &c);

    const size_t idx = static_cast<size_t>(ix) * static_cast<size_t>(nv) + static_cast<size_t>(jx);
    const float2 g = plane_grid[idx];
    rgrid[idx] += g.x * c - g.y * s;
}


/**
 * Fused subband/per-channel dedispersion and direct-to-grid rasterisation
 * kernel (Regime B: collapse_channels=False or nsubbands > 1) with in-register
 * coalescing across consecutive subbands/channels mapping to the same cell.
 *
 * Launch geometry:
 *   block = (32, 8, 1)  [x: snapshot time t (coalesced), y: baseline]
 *   grid  = (ceil(nt_out / 32), min(65535, ceil(nbl / 8)), 1)
 */
extern "C" __global__ void dedisperse_and_rasterise_subband_kernel(
    const float2* __restrict__ vis_3d,
    const float* __restrict__ wgt_3d,
    const double* __restrict__ uvw_3d,
    const int* __restrict__ delays,
    const int* __restrict__ sub_offsets,
    const double* __restrict__ sub_freqs_hz,
    const int* __restrict__ sub_delays,
    const int* __restrict__ raw_valid_chans,
    const int* __restrict__ dedisp_valid_chans,
    const int* __restrict__ uv_to_m,
    float2* __restrict__ grid_vis,
    float* __restrict__ grid_wgt,
    int nbl,
    int nchan,
    int nsub_eff,
    int nt_in,
    int nt_out,
    int nv,
    int u_half,
    double du,
    double dv,
    double min_uv_distance_m,
    int min_valid_channels,
    int collapse_channels,
    int exact_uvw,
    int mode,
    int sparse_output
) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nt_out) {
        return;
    }
    if (!is_snapshot_valid(t, nchan, collapse_channels, min_valid_channels, raw_valid_chans, dedisp_valid_chans)) {
        return;
    }

    const size_t bl_vis_stride = static_cast<size_t>(nchan) * static_cast<size_t>(nt_in);
    const size_t bl_uvw_stride = 3ULL * static_cast<size_t>(nt_in);
    const long long nv_ll = static_cast<long long>(nv);

    for (int bl = blockIdx.y * blockDim.y + threadIdx.y; bl < nbl; bl += gridDim.y * blockDim.y) {
        const size_t base_vis = static_cast<size_t>(bl) * bl_vis_stride;
        const size_t base_uvw = static_cast<size_t>(bl) * bl_uvw_stride;

        if (mode == 0) {
            long long acc_u = -1;
            long long acc_v = -1;
            float acc_re = 0.0f;
            float acc_im = 0.0f;
            float acc_w = 0.0f;

            for (int s = 0; s < nsub_eff; ++s) {
                const int c0 = sub_offsets[s];
                const int c1 = sub_offsets[s + 1];

                double sum_re = 0.0;
                double sum_im = 0.0;
                double sum_w = 0.0;

                for (int c = c0; c < c1; ++c) {
                    const int d = delays[c];
                    const int t_src = t + d;
                    if (t_src < 0 || t_src >= nt_in) {
                        continue;
                    }
                    const size_t vis_idx = base_vis + static_cast<size_t>(c) * static_cast<size_t>(nt_in) + static_cast<size_t>(t_src);
                    const float w = wgt_3d[vis_idx];
                    if (w <= 0.0f || !isfinite(w)) {
                        continue;
                    }
                    const float2 v = vis_3d[vis_idx];
                    if (!isfinite(v.x) || !isfinite(v.y)) {
                        continue;
                    }
                    if (min_uv_distance_m > 0.0) {
                        const double u_src = uvw_3d[base_uvw + static_cast<size_t>(t_src)];
                        const double v_src = uvw_3d[base_uvw + static_cast<size_t>(nt_in) + static_cast<size_t>(t_src)];
                        if (!passes_uv_distance(u_src, v_src, min_uv_distance_m)) {
                            continue;
                        }
                    }
                    const double wd = static_cast<double>(w);
                    sum_re += static_cast<double>(v.x) * wd;
                    sum_im += static_cast<double>(v.y) * wd;
                    sum_w += wd;
                }

                if (sum_w <= 0.0) {
                    continue;
                }

                int t_uvw = t;
                if (exact_uvw != 0) {
                    const int ts = t + sub_delays[s];
                    t_uvw = (ts < 0) ? 0 : ((ts >= nt_in) ? (nt_in - 1) : ts);
                }
                const double u_m = uvw_3d[base_uvw + static_cast<size_t>(t_uvw)];
                const double v_m = uvw_3d[base_uvw + static_cast<size_t>(nt_in) + static_cast<size_t>(t_uvw)];
                const double freq_over_c = sub_freqs_hz[s] / kSpeedOfLightMps;
                const double u_f = (u_m * freq_over_c) / du;
                const double v_f = (v_m * freq_over_c) / dv;
                if (!isfinite(u_f) || !isfinite(v_f) || fabs(u_f) > 1.0e15 || fabs(v_f) > 1.0e15) {
                    continue;
                }

                long long u_cell = __double2ll_rn(u_f);
                long long v_cell = __double2ll_rn(v_f);
                const float cur_re = static_cast<float>(sum_re);
                float cur_im = static_cast<float>(sum_im);
                const float cur_w = static_cast<float>(sum_w);

                if (u_cell < 0) {
                    u_cell = -u_cell;
                    v_cell = -v_cell;
                    cur_im = -cur_im;
                }
                if (u_cell < 0 || u_cell >= static_cast<long long>(u_half)) {
                    continue;
                }
                const long long v_idx = ((v_cell % nv_ll) + nv_ll) % nv_ll;

                if (u_cell == acc_u && v_idx == acc_v) {
                    acc_re += cur_re;
                    acc_im += cur_im;
                    acc_w += cur_w;
                } else {
                    if (acc_w > 0.0f) {
                        accumulate_grid_cell(
                            t, nt_out, nv, u_half, acc_u, acc_v, acc_re, acc_im, acc_w,
                            uv_to_m, sparse_output, grid_vis, grid_wgt
                        );
                    }
                    acc_u = u_cell;
                    acc_v = v_idx;
                    acc_re = cur_re;
                    acc_im = cur_im;
                    acc_w = cur_w;
                }
            }

            if (acc_w > 0.0f) {
                accumulate_grid_cell(
                    t, nt_out, nv, u_half, acc_u, acc_v, acc_re, acc_im, acc_w,
                    uv_to_m, sparse_output, grid_vis, grid_wgt
                );
            }
        } else {
            long long acc_u0 = -1;
            long long acc_v0 = -1;
            float acc_re00 = 0.0f, acc_im00 = 0.0f, acc_w00 = 0.0f;
            float acc_re10 = 0.0f, acc_im10 = 0.0f, acc_w10 = 0.0f;
            float acc_re01 = 0.0f, acc_im01 = 0.0f, acc_w01 = 0.0f;
            float acc_re11 = 0.0f, acc_im11 = 0.0f, acc_w11 = 0.0f;

            for (int s = 0; s < nsub_eff; ++s) {
                const int c0 = sub_offsets[s];
                const int c1 = sub_offsets[s + 1];

                double sum_re = 0.0;
                double sum_im = 0.0;
                double sum_w = 0.0;

                for (int c = c0; c < c1; ++c) {
                    const int d = delays[c];
                    const int t_src = t + d;
                    if (t_src < 0 || t_src >= nt_in) {
                        continue;
                    }
                    const size_t vis_idx = base_vis + static_cast<size_t>(c) * static_cast<size_t>(nt_in) + static_cast<size_t>(t_src);
                    const float w = wgt_3d[vis_idx];
                    if (w <= 0.0f || !isfinite(w)) {
                        continue;
                    }
                    const float2 v = vis_3d[vis_idx];
                    if (!isfinite(v.x) || !isfinite(v.y)) {
                        continue;
                    }
                    if (min_uv_distance_m > 0.0) {
                        const double u_src = uvw_3d[base_uvw + static_cast<size_t>(t_src)];
                        const double v_src = uvw_3d[base_uvw + static_cast<size_t>(nt_in) + static_cast<size_t>(t_src)];
                        if (!passes_uv_distance(u_src, v_src, min_uv_distance_m)) {
                            continue;
                        }
                    }
                    const double wd = static_cast<double>(w);
                    sum_re += static_cast<double>(v.x) * wd;
                    sum_im += static_cast<double>(v.y) * wd;
                    sum_w += wd;
                }

                if (sum_w <= 0.0) {
                    continue;
                }

                int t_uvw = t;
                if (exact_uvw != 0) {
                    const int ts = t + sub_delays[s];
                    t_uvw = (ts < 0) ? 0 : ((ts >= nt_in) ? (nt_in - 1) : ts);
                }
                const double u_m = uvw_3d[base_uvw + static_cast<size_t>(t_uvw)];
                const double v_m = uvw_3d[base_uvw + static_cast<size_t>(nt_in) + static_cast<size_t>(t_uvw)];
                const double freq_over_c = sub_freqs_hz[s] / kSpeedOfLightMps;
                double uf = (u_m * freq_over_c) / du;
                double vf = (v_m * freq_over_c) / dv;
                if (!isfinite(uf) || !isfinite(vf) || fabs(uf) > 1.0e15 || fabs(vf) > 1.0e15) {
                    continue;
                }

                const float cur_re = static_cast<float>(sum_re);
                float cur_im = static_cast<float>(sum_im);
                const float cur_w = static_cast<float>(sum_w);

                if (uf < 0.0) {
                    uf = -uf;
                    vf = -vf;
                    cur_im = -cur_im;
                }
                const long long u0 = static_cast<long long>(floor(uf));
                const long long v0 = static_cast<long long>(floor(vf));
                if (u0 < 0 || u0 >= static_cast<long long>(u_half)) {
                    continue;
                }
                const float au = static_cast<float>(uf - static_cast<double>(u0));
                const float av = static_cast<float>(vf - static_cast<double>(v0));
                const float c00 = (1.0f - au) * (1.0f - av);
                const float c10 = au * (1.0f - av);
                const float c01 = (1.0f - au) * av;
                const float c11 = au * av;
                const long long v0_wrap = ((v0 % nv_ll) + nv_ll) % nv_ll;

                if (u0 == acc_u0 && v0_wrap == acc_v0) {
                    acc_re00 += cur_re * c00; acc_im00 += cur_im * c00; acc_w00 += cur_w * c00;
                    acc_re10 += cur_re * c10; acc_im10 += cur_im * c10; acc_w10 += cur_w * c10;
                    acc_re01 += cur_re * c01; acc_im01 += cur_im * c01; acc_w01 += cur_w * c01;
                    acc_re11 += cur_re * c11; acc_im11 += cur_im * c11; acc_w11 += cur_w * c11;
                } else {
                    if (acc_u0 >= 0) {
                        splat_bilinear_corners(
                            t, nt_out, nv, u_half, acc_u0, acc_v0,
                            acc_re00, acc_im00, acc_w00,
                            acc_re10, acc_im10, acc_w10,
                            acc_re01, acc_im01, acc_w01,
                            acc_re11, acc_im11, acc_w11,
                            uv_to_m, sparse_output, grid_vis, grid_wgt
                        );
                    }
                    acc_u0 = u0;
                    acc_v0 = v0_wrap;
                    acc_re00 = cur_re * c00; acc_im00 = cur_im * c00; acc_w00 = cur_w * c00;
                    acc_re10 = cur_re * c10; acc_im10 = cur_im * c10; acc_w10 = cur_w * c10;
                    acc_re01 = cur_re * c01; acc_im01 = cur_im * c01; acc_w01 = cur_w * c01;
                    acc_re11 = cur_re * c11; acc_im11 = cur_im * c11; acc_w11 = cur_w * c11;
                }
            }

            if (acc_u0 >= 0) {
                splat_bilinear_corners(
                    t, nt_out, nv, u_half, acc_u0, acc_v0,
                    acc_re00, acc_im00, acc_w00,
                    acc_re10, acc_im10, acc_w10,
                    acc_re01, acc_im01, acc_w01,
                    acc_re11, acc_im11, acc_w11,
                    uv_to_m, sparse_output, grid_vis, grid_wgt
                );
            }
        }
    }
}


/**
 * Normalise dense UV raster grid (nt_out, nv, u_half) by per-cell weights,
 * enforce snapshot channel-validity masks, and zero entries with |V| <= uv_threshold.
 * ``out_uv`` may alias ``grid_vis`` for in-place normalisation.
 *
 * Launch geometry:
 *   block = (32, 8, 1)  [x: spatial cell s (coalesced), y: snapshot t]
 *   grid  = (ceil((nv * u_half) / 32), min(65535, ceil(nt_out / 8)), 1)
 */
extern "C" __global__ void normalise_raster_grid_kernel(
    const float2* __restrict__ grid_vis,
    const float* __restrict__ grid_wgt,
    const int* __restrict__ raw_valid_chans,
    const int* __restrict__ dedisp_valid_chans,
    float2* __restrict__ out_uv,
    int nt_out,
    int nv,
    int u_half,
    int nchan,
    int collapse_channels,
    int min_valid_channels,
    float uv_threshold
) {
    const size_t spatial = static_cast<size_t>(nv) * static_cast<size_t>(u_half);
    const size_t s = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    if (s >= spatial) {
        return;
    }

    const float thr_sq = (uv_threshold > 0.0f) ? (uv_threshold * uv_threshold) : 0.0f;

    for (int t = blockIdx.y * blockDim.y + threadIdx.y; t < nt_out; t += gridDim.y * blockDim.y) {
        const size_t idx = static_cast<size_t>(t) * spatial + s;
        float2 val = make_float2(0.0f, 0.0f);

        if (is_snapshot_valid(t, nchan, collapse_channels, min_valid_channels, raw_valid_chans, dedisp_valid_chans)) {
            const float w = grid_wgt[idx];
            if (w > 0.0f && isfinite(w)) {
                const float2 raw = grid_vis[idx];
                const float inv_w = 1.0f / w;
                const float rx = raw.x * inv_w;
                const float ry = raw.y * inv_w;
                if (isfinite(rx) && isfinite(ry)) {
                    if (thr_sq <= 0.0f || (rx * rx + ry * ry) > thr_sq) {
                        val = make_float2(rx, ry);
                    }
                }
            }
        }
        out_uv[idx] = val;
    }
}


/**
 * Normalise sparse active-cell DGA raster grid (M, nt_out) by per-cell weights,
 * enforce snapshot channel-validity masks, and zero entries with |V| <= uv_threshold.
 * ``out_dga`` may alias ``sparse_vis`` for in-place normalisation.
 *
 * Launch geometry:
 *   block = (32, 8, 1)  [x: snapshot t (coalesced), y: active cell m]
 *   grid  = (ceil(nt_out / 32), min(65535, ceil(M / 8)), 1)
 */
extern "C" __global__ void normalise_sparse_grid_kernel(
    const float2* __restrict__ sparse_vis,
    const float* __restrict__ sparse_wgt,
    const int* __restrict__ raw_valid_chans,
    const int* __restrict__ dedisp_valid_chans,
    float2* __restrict__ out_dga,
    int m_active,
    int nt_out,
    int nchan,
    int collapse_channels,
    int min_valid_channels,
    float uv_threshold
) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nt_out) {
        return;
    }

    const bool snap_ok = is_snapshot_valid(
        t, nchan, collapse_channels, min_valid_channels, raw_valid_chans, dedisp_valid_chans
    );
    const float thr_sq = (uv_threshold > 0.0f) ? (uv_threshold * uv_threshold) : 0.0f;

    for (int m = blockIdx.y * blockDim.y + threadIdx.y; m < m_active; m += gridDim.y * blockDim.y) {
        const size_t idx = static_cast<size_t>(m) * static_cast<size_t>(nt_out) + static_cast<size_t>(t);
        float2 val = make_float2(0.0f, 0.0f);

        if (snap_ok) {
            const float w = sparse_wgt[idx];
            if (w > 0.0f && isfinite(w)) {
                const float2 raw = sparse_vis[idx];
                const float inv_w = 1.0f / w;
                const float rx = raw.x * inv_w;
                const float ry = raw.y * inv_w;
                if (isfinite(rx) && isfinite(ry)) {
                    if (thr_sq <= 0.0f || (rx * rx + ry * ry) > thr_sq) {
                        val = make_float2(rx, ry);
                    }
                }
            }
        }
        out_dga[idx] = val;
    }
}


/**
 * Normalise dense UV raster grid (nt_out, nv, u_half) by per-snapshot natural
 * weight scale ``snap_inv_w[t]`` (e.g., ``1.0f / (wgt_max[t] * n_valid[t])``
 * matching ``fastducc.imaging.grid_chunk_uv_cube`` and ``ducc0.wgridder.vis2grid``).
 *
 * Launch geometry:
 *   block = (32, 8, 1)  [x: spatial cell s (coalesced), y: snapshot t]
 *   grid  = (ceil((nv * u_half) / 32), min(65535, ceil(nt_out / 8)), 1)
 */
extern "C" __global__ void normalise_raster_grid_snapshot_kernel(
    const float2* __restrict__ grid_vis,
    const float* __restrict__ grid_wgt,
    const float* __restrict__ snap_inv_w,
    const int* __restrict__ raw_valid_chans,
    const int* __restrict__ dedisp_valid_chans,
    float2* __restrict__ out_uv,
    int nt_out,
    int nv,
    int u_half,
    int nchan,
    int collapse_channels,
    int min_valid_channels,
    float uv_threshold
) {
    const size_t spatial = static_cast<size_t>(nv) * static_cast<size_t>(u_half);
    const size_t s = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    if (s >= spatial) {
        return;
    }

    const float thr_sq = (uv_threshold > 0.0f) ? (uv_threshold * uv_threshold) : 0.0f;

    for (int t = blockIdx.y * blockDim.y + threadIdx.y; t < nt_out; t += gridDim.y * blockDim.y) {
        const size_t idx = static_cast<size_t>(t) * spatial + s;
        float2 val = make_float2(0.0f, 0.0f);

        if (is_snapshot_valid(t, nchan, collapse_channels, min_valid_channels, raw_valid_chans, dedisp_valid_chans)) {
            const float inv_w = snap_inv_w[t];
            const float w = grid_wgt[idx];
            if (inv_w > 0.0f && isfinite(inv_w) && w > 0.0f && isfinite(w)) {
                const float2 raw = grid_vis[idx];
                const float rx = raw.x * inv_w;
                const float ry = raw.y * inv_w;
                if (isfinite(rx) && isfinite(ry)) {
                    if (thr_sq <= 0.0f || (rx * rx + ry * ry) > thr_sq) {
                        val = make_float2(rx, ry);
                    }
                }
            }
        }
        out_uv[idx] = val;
    }
}


/**
 * Normalise sparse active-cell DGA raster grid (M, nt_out) by per-snapshot
 * natural weight scale ``snap_inv_w[t]``.
 *
 * Launch geometry:
 *   block = (32, 8, 1)  [x: snapshot t (coalesced), y: active cell m]
 *   grid  = (ceil(nt_out / 32), min(65535, ceil(M / 8)), 1)
 */
extern "C" __global__ void normalise_sparse_grid_snapshot_kernel(
    const float2* __restrict__ sparse_vis,
    const float* __restrict__ sparse_wgt,
    const float* __restrict__ snap_inv_w,
    const int* __restrict__ raw_valid_chans,
    const int* __restrict__ dedisp_valid_chans,
    float2* __restrict__ out_dga,
    int m_active,
    int nt_out,
    int nchan,
    int collapse_channels,
    int min_valid_channels,
    float uv_threshold
) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nt_out) {
        return;
    }

    const bool snap_ok = is_snapshot_valid(
        t, nchan, collapse_channels, min_valid_channels, raw_valid_chans, dedisp_valid_chans
    );
    const float inv_w = snap_ok ? snap_inv_w[t] : 0.0f;
    const bool scale_ok = (inv_w > 0.0f) && isfinite(inv_w);
    const float thr_sq = (uv_threshold > 0.0f) ? (uv_threshold * uv_threshold) : 0.0f;

    for (int m = blockIdx.y * blockDim.y + threadIdx.y; m < m_active; m += gridDim.y * blockDim.y) {
        const size_t idx = static_cast<size_t>(m) * static_cast<size_t>(nt_out) + static_cast<size_t>(t);
        float2 val = make_float2(0.0f, 0.0f);

        if (scale_ok) {
            const float w = sparse_wgt[idx];
            if (w > 0.0f && isfinite(w)) {
                const float2 raw = sparse_vis[idx];
                const float rx = raw.x * inv_w;
                const float ry = raw.y * inv_w;
                if (isfinite(rx) && isfinite(ry)) {
                    if (thr_sq <= 0.0f || (rx * rx + ry * ry) > thr_sq) {
                        val = make_float2(rx, ry);
                    }
                }
            }
        }
        out_dga[idx] = val;
    }
}


/**
 * Compute per-snapshot DUCC vis2grid natural weight normalisation scale
 * ``snap_inv_w[t]`` for ``t in [0, nt_out)`` matching
 * ``fastducc.imaging.grid_chunk_uv_cube``:
 *   - For each snapshot ``t``, reduces channels into ``nsub_eff`` subbands.
 *   - Finds ``wgt_max[t] = max_{bl, s} w_sub[bl, s, t]`` and
 *     ``n_valid[t] = count_{bl, s}(w_sub[bl, s, t] > 0)``.
 *   - If ``divide_by_n != 0``: ``snap_inv_w[t] = 1.0f / (wgt_max[t] * n_valid[t])``
 *     else: ``snap_inv_w[t] = 1.0f / wgt_max[t]``.
 *
 * Launch geometry:
 *   block = (32, 8, 1)  [x: baseline tile, y: subband tile]
 *   grid  = (nt_out, 1, 1)
 */
extern "C" __global__ void compute_snapshot_vis2grid_norm_kernel(
    const float2* __restrict__ vis_3d,
    const float* __restrict__ wgt_3d,
    const double* __restrict__ uvw_3d,
    const int* __restrict__ delays,
    const int* __restrict__ sub_offsets,
    const int* __restrict__ raw_valid_chans,
    const int* __restrict__ dedisp_valid_chans,
    float* __restrict__ snap_inv_w,
    int nbl,
    int nchan,
    int nsub_eff,
    int nt_in,
    int nt_out,
    double min_uv_distance_m,
    int min_valid_channels,
    int collapse_channels,
    int divide_by_n
) {
    __shared__ float s_max_w[256];
    __shared__ int s_cnt_w[256];

    const int t = blockIdx.x;
    if (t >= nt_out) {
        return;
    }

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    float local_max = 0.0f;
    int local_cnt = 0;

    if (is_snapshot_valid(t, nchan, collapse_channels, min_valid_channels, raw_valid_chans, dedisp_valid_chans)) {
        const size_t bl_vis_stride = static_cast<size_t>(nchan) * static_cast<size_t>(nt_in);
        const size_t bl_uvw_stride = 3ULL * static_cast<size_t>(nt_in);

        for (int s = threadIdx.y; s < nsub_eff; s += blockDim.y) {
            const int c0 = sub_offsets[s];
            const int c1 = sub_offsets[s + 1];
            for (int bl = threadIdx.x; bl < nbl; bl += blockDim.x) {
                const size_t base_vis = static_cast<size_t>(bl) * bl_vis_stride;
                const size_t base_uvw = static_cast<size_t>(bl) * bl_uvw_stride;
                double sum_w = 0.0;
                for (int c = c0; c < c1; ++c) {
                    const int d = delays[c];
                    const int t_src = t + d;
                    if (t_src < 0 || t_src >= nt_in) {
                        continue;
                    }
                    const size_t vis_idx = base_vis + static_cast<size_t>(c) * static_cast<size_t>(nt_in) + static_cast<size_t>(t_src);
                    const float w = wgt_3d[vis_idx];
                    if (w <= 0.0f || !isfinite(w)) {
                        continue;
                    }
                    const float2 v = vis_3d[vis_idx];
                    if (!isfinite(v.x) || !isfinite(v.y)) {
                        continue;
                    }
                    if (min_uv_distance_m > 0.0) {
                        const double u_src = uvw_3d[base_uvw + static_cast<size_t>(t_src)];
                        const double v_src = uvw_3d[base_uvw + static_cast<size_t>(nt_in) + static_cast<size_t>(t_src)];
                        if (!passes_uv_distance(u_src, v_src, min_uv_distance_m)) {
                            continue;
                        }
                    }
                    sum_w += static_cast<double>(w);
                }
                const float wf = static_cast<float>(sum_w);
                if (wf > 0.0f) {
                    local_cnt += 1;
                    if (wf > local_max) {
                        local_max = wf;
                    }
                }
            }
        }
    }

    s_max_w[tid] = local_max;
    s_cnt_w[tid] = local_cnt;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_max_w[tid + stride] > s_max_w[tid]) {
                s_max_w[tid] = s_max_w[tid + stride];
            }
            s_cnt_w[tid] += s_cnt_w[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float wmax = s_max_w[0];
        const int nval = s_cnt_w[0];
        if (wmax > 0.0f && nval > 0) {
            double denom = static_cast<double>(wmax);
            if (divide_by_n == 1) {
                denom *= static_cast<double>(nval);
            } else if (divide_by_n == 2) {
                const int nsub_norm = (collapse_channels != 0 && nsub_eff <= 1) ? 1 : max(1, nsub_eff);
                const double nbl_eff = fmax(1.0, static_cast<double>(nval) / static_cast<double>(nsub_norm));
                denom *= nbl_eff;
            }
            snap_inv_w[t] = static_cast<float>(1.0 / denom);
        } else {
            snap_inv_w[t] = 0.0f;
        }
    }
}


/**
 * Apply DUCC w-axis correction function ``corfunc((nm1 + nshift) * dw) / ((nm1 + 1) * nu * nv)``
 * to the accumulated real image grid ``rgrid`` (grid_nu, nv) matching
 * ``Wgridder::apply_w_corrections_rgrid`` (``wgridder_impl.h:1811-1871``).
 *
 * Launch geometry:
 *   block = (32, 8, 1)  [x: jx in [0, nv), y: ix in [0, grid_nu)]
 *   grid  = (ceil(nv / 32), ceil(grid_nu / 8), 1)
 */
extern "C" __global__ void ducc_apply_w_corrections_rgrid_kernel(
    float* __restrict__ rgrid,
    const double* __restrict__ cor_x,
    const double* __restrict__ cor_wgtpsi,
    int n_cor,
    int grid_nu,
    int nv,
    int nxdirty,
    int nydirty,
    double pixsize_x,
    double pixsize_y,
    double lshift,
    double mshift,
    double nshift,
    double dw,
    int divide_by_n
) {
    const int jx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ix = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= grid_nu || jx >= nv) {
        return;
    }

    const int hnx = nxdirty / 2;
    const int hny = nydirty / 2;
    int kx = (ix <= grid_nu / 2) ? ix : (ix - grid_nu);
    kx = max(-hnx, min(hnx, kx));
    int ky = (jx <= nv / 2) ? jx : (jx - nv);
    ky = max(-hny, min(hny, ky));

    const double x = lshift + static_cast<double>(kx) * pixsize_x;
    const double y = mshift + static_cast<double>(ky) * pixsize_y;
    const double xsq = x * x;
    const double ysq = y * y;
    const double tmp = 1.0 - xsq - ysq;

    double fct = 0.0;
    if (tmp >= 0.0) {
        const double nm1 = (-xsq - ysq) / (sqrt(tmp) + 1.0);
        const double v_arg = (nm1 + nshift) * dw;
        double sum_c = 0.0;
        for (int i = 0; i < n_cor; ++i) {
            sum_c += cor_wgtpsi[i] * cos(cor_x[i] * v_arg);
        }
        fct = 1.0 / sum_c;
        if (divide_by_n != 0) {
            fct /= (nm1 + 1.0);
        }
    } else if (divide_by_n == 0) {
        const double v_arg = (-sqrt(-tmp) - 1.0 + nshift) * dw;
        double sum_c = 0.0;
        for (int i = 0; i < n_cor; ++i) {
            sum_c += cor_wgtpsi[i] * cos(cor_x[i] * v_arg);
        }
        fct = 1.0 / sum_c;
    }

    const float tfct = static_cast<float>(fct / (static_cast<double>(grid_nu) * static_cast<double>(nv)));
    const size_t idx = static_cast<size_t>(ix) * static_cast<size_t>(nv) + static_cast<size_t>(jx);
    rgrid[idx] *= tfct;
}


/**
 * Convert forward 2D complex FFT output ``fft_rgrid`` (grid_nu, nv) of real
 * ``rgrid`` into DUCC's 2D Fast Hartley Transform buffer ``fht`` (grid_nu, nv):
 *   fht(u, v) = Re(F(u, v)) - Im(F(u, v))
 * (because the 2D FHT cas(2*pi*(u*x/nu + v*y/nv)) = cos(...) + sin(...) = Re(F(u,v)) - Im(F(u,v))).
 *
 * Launch geometry:
 *   block = (32, 8, 1)  [x: v in [0, nv), y: u in [0, grid_nu)]
 *   grid  = (ceil(nv / 32), ceil(grid_nu / 8), 1)
 */
extern "C" __global__ void ducc_fft_to_fht_kernel(
    const float2* __restrict__ fft_rgrid,
    float* __restrict__ fht,
    int grid_nu,
    int nv
) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    const int u = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= grid_nu || v >= nv) {
        return;
    }
    const size_t idx = static_cast<size_t>(u) * static_cast<size_t>(nv) + static_cast<size_t>(v);
    const float2 val = fft_rgrid[idx];
    fht[idx] = val.x - val.y;
}


/**
 * Separable 1D circular box dilation along v (step 1 of DUCC apply_uv_fresnel_mask).
 */
extern "C" __global__ void ducc_fresnel_dilate_v_kernel(
    const unsigned char* __restrict__ seed,
    unsigned char* __restrict__ tmp,
    int grid_nu,
    int nv,
    int dv_f
) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    const int u = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= grid_nu || v >= nv) {
        return;
    }
    const size_t row = static_cast<size_t>(u) * static_cast<size_t>(nv);
    unsigned char hit = 0;
    for (int dv = -dv_f; dv <= dv_f; ++dv) {
        int iv = (v - dv) % nv;
        if (iv < 0) iv += nv;
        if (seed[row + static_cast<size_t>(iv)] != 0) {
            hit = 1;
            break;
        }
    }
    tmp[row + static_cast<size_t>(v)] = hit;
}


/**
 * Separable 1D circular box dilation along u + in-place Fresnel zeroing of
 * ``fht`` (step 2 of DUCC apply_uv_fresnel_mask in wgridder_impl.h:1928-1946).
 */
extern "C" __global__ void ducc_fresnel_dilate_u_and_mask_kernel(
    const unsigned char* __restrict__ tmp,
    float* __restrict__ fht,
    int grid_nu,
    int nv,
    int du_f
) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    const int u = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= grid_nu || v >= nv) {
        return;
    }
    bool hit = false;
    for (int du = -du_f; du <= du_f; ++du) {
        int iu = (u - du) % grid_nu;
        if (iu < 0) iu += grid_nu;
        if (tmp[static_cast<size_t>(iu) * static_cast<size_t>(nv) + static_cast<size_t>(v)] != 0) {
            hit = true;
            break;
        }
    }
    if (!hit) {
        fht[static_cast<size_t>(u) * static_cast<size_t>(nv) + static_cast<size_t>(v)] = 0.0f;
    }
}


/**
 * Convert masked 2D FHT buffer ``fht`` (grid_nu, nv) into Hermitian half-plane
 * complex UV grid ``(nv, u_half)`` (or sparse active-cell DGA column ``(M, nt_out)``
 * for snapshot ``t``) via DUCC ``hartley2complex``:
 *   G(u, v) = 0.5 * (fht(u, v) + fht(-u mod nu, -v mod nv))
 *           + 0.5j * (fht(-u mod nu, -v mod nv) - fht(u, v))
 * scaled by ``snap_scale`` (e.g. ``1.0f / n_valid``).
 *
 * Launch geometry:
 *   block = (32, 8, 1)  [x: u in [0, u_half), y: v in [0, nv)]
 *   grid  = (ceil(u_half / 32), ceil(nv / 8), 1)
 */
extern "C" __global__ void ducc_fht_to_halfplane_kernel(
    const float* __restrict__ fht,
    const int* __restrict__ uv_to_m,
    float2* __restrict__ out_uv,
    int t,
    int nt_out,
    int grid_nu,
    int nv,
    int u_half,
    float snap_scale,
    float uv_threshold,
    int sparse_output
) {
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= u_half || v >= nv) {
        return;
    }

    const size_t uv_offset = static_cast<size_t>(v) * static_cast<size_t>(u_half) + static_cast<size_t>(u);
    size_t dst_idx = 0ULL;
    if (uv_to_m != nullptr) {
        const int m_idx = uv_to_m[uv_offset];
        if (m_idx < 0) {
            if (sparse_output == 0) {
                out_uv[static_cast<size_t>(t) * static_cast<size_t>(nv) * static_cast<size_t>(u_half) + uv_offset] =
                    make_float2(0.0f, 0.0f);
            }
            return;
        }
        if (sparse_output != 0) {
            dst_idx = static_cast<size_t>(m_idx) * static_cast<size_t>(nt_out) + static_cast<size_t>(t);
        } else {
            dst_idx = static_cast<size_t>(t) * static_cast<size_t>(nv) * static_cast<size_t>(u_half) + uv_offset;
        }
    } else {
        dst_idx = static_cast<size_t>(t) * static_cast<size_t>(nv) * static_cast<size_t>(u_half) + uv_offset;
    }

    const int xu = (u == 0) ? 0 : (grid_nu - u);
    const int xv = (v == 0) ? 0 : (nv - v);
    const float a = fht[static_cast<size_t>(u) * static_cast<size_t>(nv) + static_cast<size_t>(v)];
    const float b = fht[static_cast<size_t>(xu) * static_cast<size_t>(nv) + static_cast<size_t>(xv)];

    const float rx = 0.5f * (a + b) * snap_scale;
    const float ry = 0.5f * (b - a) * snap_scale;
    const float thr_sq = (uv_threshold > 0.0f) ? (uv_threshold * uv_threshold) : 0.0f;

    float2 val = make_float2(0.0f, 0.0f);
    if (isfinite(rx) && isfinite(ry) && (thr_sq <= 0.0f || (rx * rx + ry * ry) > thr_sq)) {
        val = make_float2(rx, ry);
    }
    out_uv[dst_idx] = val;
}
