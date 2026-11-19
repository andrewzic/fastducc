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
