// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

// CUDA device kernels for the q2k -> k2q CSR builder (nvcc only; no ATen).

#include "build_k2q_csr_launch.h"
#include <cuda_runtime.h>

#include <cstddef>

namespace {

constexpr int kWarpSize = 32;

__device__ __forceinline__ void advance_batch_only(
    int const* __restrict__ cu_q, int B, int q_abs, int& bi)
{
    while (bi < B && cu_q[bi + 1] <= q_abs) ++bi;
}

// Atomic increment of a 16-bit half within a 32-bit SMEM word; returns the
// OLD 16-bit value (slot). Per-warp count must stay < 32768 so the low
// half does not carry into the high half.
//   base_int32 : int32 pointer; element i holds rows 2*i (low) and 2*i+1 (high).
__device__ __forceinline__ int atomic_inc_int16_packed(
    int* base_int32, int row)
{
    int idx = row >> 1;
    int shift = (row & 1) << 4;  // 0 or 16
    int delta = 1 << shift;
    int old = atomicAdd(&base_int32[idx], delta);
    return (old >> shift) & 0xFFFF;
}

// Read 16-bit half from packed int32 storage.
__device__ __forceinline__ int read_int16_packed(int const* base_int32, int row) {
    int v = base_int32[row >> 1];
    int shift = (row & 1) << 4;
    return (v >> shift) & 0xFFFF;
}

// ---------------------------------------------------------------------------
// M: round-robin row map.
// ---------------------------------------------------------------------------
template <int kBlockK>
__global__ void k2q_build_row_map_kernel(
    int const* __restrict__ cu_k,
    int* __restrict__ row_map,
    int* __restrict__ row_coords,
    int B,
    int max_kv_blocks)
{
    int level = blockIdx.x;
    if (level >= max_kv_blocks) return;
    if (threadIdx.x != 0) return;
    int rows_before = 0;
    for (int b = 0; b < B; ++b) {
        int rb = (cu_k[b + 1] - cu_k[b] + kBlockK - 1) / kBlockK;
        rows_before += (rb < level ? rb : level);
    }
    int active_before = 0;
    for (int b = 0; b < B; ++b) {
        int rb = (cu_k[b + 1] - cu_k[b] + kBlockK - 1) / kBlockK;
        if (rb > level) {
            int row_linear = rows_before + active_before;
            row_map[(size_t)b * max_kv_blocks + level] = row_linear;
            if (row_coords != nullptr) {
                row_coords[(size_t)row_linear * 2] = b;
                row_coords[(size_t)row_linear * 2 + 1] = level;
            }
            ++active_before;
        } else {
            row_map[(size_t)b * max_kv_blocks + level] = -1;
        }
    }
}

// ---------------------------------------------------------------------------
// H: per-warp histogram + tile_counts.
// kWarps warps per CTA, each owns q-sub-range = q_per_cta / kWarps.
// SMEM hist[kWarps, total_rows] int32 (stored as packed int16 cursor:
// 2 entries per int32 word). Each warp counts to its own row.
// At end-of-CTA, write tile_counts[c*kWarps + w, h, r] = smem_hist[w, r]
// and atomicAdd(row_counts[h, r], sum over w of smem_hist[w, r]).
// ---------------------------------------------------------------------------
template <int kTopK, int kBlockK, int kWarps>
__global__ void k2q_hist_kernel(
    int const* __restrict__ q2k,
    int const* __restrict__ cu_q,
    int const* __restrict__ row_map,
    int* __restrict__ row_counts,
    int* __restrict__ tile_counts,
    int H, int B, int S_Q,
    int total_rows, int max_kv_blocks,
    int q_per_cta, int q_per_warp)
{
    constexpr int kThreads = kWarps * kWarpSize;
    extern __shared__ int smem_hist_int[];
    int* smem_hist = smem_hist_int;
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;
    int c = blockIdx.x;
    int q_start_cta = c * q_per_cta;
    int q_end_cta = min(q_start_cta + q_per_cta, S_Q);
    int q_start_warp = min(q_start_cta + warp_id * q_per_warp, q_end_cta);
    int q_end_warp = min(q_start_warp + q_per_warp, q_end_cta);

    constexpr int kInt4PerToken = kTopK / 4;
    int packed_per_warp = (total_rows + 1) >> 1;
    int* my_hist = smem_hist + warp_id * packed_per_warp;

    for (int h = 0; h < H; ++h) {
        for (int i = lane; i < packed_per_warp; i += kWarpSize) my_hist[i] = 0;
        __syncthreads();

        if (q_start_warp < q_end_warp) {
            int bi = 0;
            int qi = q_start_warp + lane;
            advance_batch_only(cu_q, B, qi, bi);

            int4 const* head_topk4 =
                reinterpret_cast<int4 const*>(q2k + (size_t)h * S_Q * kTopK);

            for (; qi < q_end_warp; qi += kWarpSize) {
                advance_batch_only(cu_q, B, qi, bi);
                int const* my_row_map = row_map + (size_t)bi * max_kv_blocks;

                int4 buf[kInt4PerToken];
                #pragma unroll
                for (int v = 0; v < kInt4PerToken; ++v) {
                    buf[v] = head_topk4[(size_t)qi * kInt4PerToken + v];
                }
                #pragma unroll
                for (int t = 0; t < kTopK; ++t) {
                    int kvb_local = reinterpret_cast<int const*>(buf)[t];
                    if (kvb_local >= 0 && kvb_local < max_kv_blocks) {
                        int row = my_row_map[kvb_local];
                        if (row >= 0 && row < total_rows) {
                            atomic_inc_int16_packed(my_hist, row);
                        }
                    }
                }
            }
        }
        __syncthreads();

        int* head_row_counts = row_counts + (size_t)h * total_rows;
        // Each warp writes its own slice of tile_counts (full int32) by
        // unpacking int16 entries from SMEM.
        int* my_tile = tile_counts +
            ((size_t)(c * kWarps + warp_id) * H + h) * total_rows;
        for (int i = lane; i < total_rows; i += kWarpSize) {
            my_tile[i] = read_int16_packed(my_hist, i);
        }
        __syncthreads();

        // Sum across warps (int32 accumulator), atomicAdd to row_counts.
        for (int i = tid; i < total_rows; i += kThreads) {
            int sum = 0;
            #pragma unroll
            for (int w = 0; w < kWarps; ++w) {
                sum += read_int16_packed(smem_hist + w * packed_per_warp, i);
            }
            if (sum > 0) atomicAdd(&head_row_counts[i], sum);
        }
        if (h + 1 < H) __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// PR: row prefix. One block per head.
// ---------------------------------------------------------------------------
template <int kThreads>
__global__ void k2q_row_prefix_kernel(
    int const* __restrict__ row_counts,
    int* __restrict__ row_ptr,
    int const* __restrict__ row_coords,
    int* __restrict__ scheduler_metadata,
    int* __restrict__ work_count,
    int total_rows,
    int target_q_per_cta,
    int work_capacity)
{
    int h = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ int scan_buf[kThreads];

    int const* head_counts = row_counts + (size_t)h * total_rows;
    int* head_rowptr = row_ptr + (size_t)h * (total_rows + 1);
    int chunk = (total_rows + kThreads - 1) / kThreads;
    int lo = tid * chunk;
    int hi = min(lo + chunk, total_rows);

    int local_sum = 0;
    for (int i = lo; i < hi; ++i) local_sum += head_counts[i];
    scan_buf[tid] = local_sum;
    __syncthreads();

    for (int off = 1; off < kThreads; off <<= 1) {
        int add = (tid >= off) ? scan_buf[tid - off] : 0;
        __syncthreads();
        scan_buf[tid] += add;
        __syncthreads();
    }
    int running = scan_buf[tid] - local_sum;
    for (int i = lo; i < hi; ++i) {
        int row_count = head_counts[i];
        running += row_count;
        head_rowptr[i + 1] = running;
        if (scheduler_metadata != nullptr && work_count != nullptr && row_count > 0) {
            int num_chunks = (row_count + target_q_per_cta - 1) / target_q_per_cta;
            int base = atomicAdd(work_count, num_chunks);
            int batch_idx = row_coords[(size_t)i * 2];
            int kv_block_idx = row_coords[(size_t)i * 2 + 1];
            for (int c = 0; c < num_chunks; ++c) {
                int work_idx = base + c;
                if (work_idx < work_capacity) {
                    int q_begin = c * target_q_per_cta;
                    int q_count = min(target_q_per_cta, row_count - q_begin);
                    int* meta = scheduler_metadata + (size_t)work_idx * 6;
                    meta[0] = h;
                    meta[1] = i;
                    meta[2] = q_begin;
                    meta[3] = q_count;
                    meta[4] = batch_idx;
                    meta[5] = kv_block_idx;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PT_smem: SMEM-staged tile prefix scan.
// Each block handles kRowsPerBlock rows for one head h. Cooperative load
// of tile_counts[*, h, base_r..base_r+M) into SMEM (better coalescing
// than per-warp uncoalesced stride reads), then per-warp scan in SMEM,
// then cooperative store back. Fuses row_ptr into the base.
// ---------------------------------------------------------------------------
template <int kThreads, int kRowsPerBlock>
__global__ void k2q_tile_prefix_smem_kernel(
    int* __restrict__ tile_counts,
    int const* __restrict__ row_ptr,
    int H, int total_rows, int G_total)
{
    static_assert(kRowsPerBlock > 0, "kRowsPerBlock must be positive");
    extern __shared__ int smem_tprefix[];
    // smem layout: smem[r_off][g] for r_off in [0, M), g in [0, G_total).

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;

    // Grid: H * blocks_per_h. Each block stays within a single head h
    // and processes kRowsPerBlock contiguous rows starting at b_in_h *
    // kRowsPerBlock. (Earlier flat-grid mapping `h = block_job /
    // total_rows; base_r = block_job - h*total_rows` skipped rows when
    // total_rows was not a multiple of kRowsPerBlock and H > 1, because
    // the last partial block of head h-1 left blocks of head h starting
    // at a non-zero row offset.)
    int blocks_per_h = (total_rows + kRowsPerBlock - 1) / kRowsPerBlock;
    int h = blockIdx.x / blocks_per_h;
    int b_in_h = blockIdx.x - h * blocks_per_h;
    if (h >= H) return;
    int base_r = b_in_h * kRowsPerBlock;
    if (base_r >= total_rows) return;
    int actual_M = min(kRowsPerBlock, total_rows - base_r);

    size_t stride_g = (size_t)H * total_rows;
    int* base_ptr = tile_counts + (size_t)h * total_rows + base_r;
    int total_elems = G_total * actual_M;

    // Cooperative load. Pattern: thread tid -> (r_off=tid%M, g=tid/M),
    // then strided. 32 lanes hit M r's × (32/M) g's, giving 32/M cache
    // lines per warp (vs 32 in the naive stride-along-g pattern).
    for (int i = tid; i < total_elems; i += kThreads) {
        int r_off = i % actual_M;
        int g = i / actual_M;
        smem_tprefix[r_off * G_total + g] = base_ptr[g * stride_g + r_off];
    }
    __syncthreads();

    // Per-warp scan: warp w scans row (base_r + w) if w < actual_M.
    if (warp_id < actual_M) {
        int abs_r = base_r + warp_id;
        int rp = row_ptr[(size_t)h * (total_rows + 1) + abs_r];
        int* my_smem = smem_tprefix + warp_id * G_total;
        int running = rp;
        for (int g0 = 0; g0 < G_total; g0 += kWarpSize) {
            int g = g0 + lane;
            int v = (g < G_total) ? my_smem[g] : 0;
            int x = v;
            #pragma unroll
            for (int off = 1; off < kWarpSize; off <<= 1) {
                int nbr = __shfl_up_sync(0xFFFFFFFF, x, off);
                if (lane >= off) x += nbr;
            }
            int excl = running + x - v;
            if (g < G_total) my_smem[g] = excl;
            int chunk_sum = __shfl_sync(0xFFFFFFFF, x, 31);
            running += chunk_sum;
        }
    }
    __syncthreads();

    // Cooperative store back.
    for (int i = tid; i < total_elems; i += kThreads) {
        int r_off = i % actual_M;
        int g = i / actual_M;
        base_ptr[g * stride_g + r_off] = smem_tprefix[r_off * G_total + g];
    }
}

// ---------------------------------------------------------------------------
// S: scatter. kWarps warps per CTA, each owns q-sub-range. Per-warp SMEM
// cursor and per-warp tile_offset slot range. Within a warp, q's are
// processed sequentially; lanes 0..kTopK-1 handle the topK slots in
// lockstep. Across distinct q's in the same warp, the lockstep ordering
// guarantees q-monotonic atomicAdd on smem_cursor[r].
// ---------------------------------------------------------------------------
// kQPerIter * kTopK lanes are active per warp iter; remaining lanes idle.
// For kTopK=16, kQPerIter=2 uses all 32 lanes; for kTopK=8, kQPerIter=4.
// CORRECTNESS NOTE: relies on lane-ordered SMEM atomicAdd return values
// within a single warp instruction (verified on SM100; tests pass).
//
// SMEM cursor stored as packed int16 (two cursors per int32). Per-warp
// row count must stay < 32768 (~q_per_warp * kTopK at max sink), which
// holds for all task.md sizes up to 1024K.
template <int kTopK, int kBlockK, int kWarps>
__global__ void k2q_scatter_kernel(
    int const* __restrict__ q2k,
    int const* __restrict__ cu_q,
    int const* __restrict__ row_map,
    int const* __restrict__ abs_base,
    int* __restrict__ q_idx,
    int* __restrict__ qsplit_idx,
    int* __restrict__ split_counts,
    int H, int B, int S_Q,
    int total_rows, int max_kv_blocks,
    int q_per_cta, int q_per_warp,
    int max_seqlen_q)
{
    constexpr int kQPerIter = kWarpSize / kTopK > 0 ? kWarpSize / kTopK : 1;
    extern __shared__ int smem_cursor_int[];
    int* smem_cursor = smem_cursor_int;
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;
    int c = blockIdx.x;
    int q_start_cta = c * q_per_cta;
    int q_end_cta = min(q_start_cta + q_per_cta, S_Q);
    int q_start_warp = min(q_start_cta + warp_id * q_per_warp, q_end_cta);
    int q_end_warp = min(q_start_warp + q_per_warp, q_end_cta);

    int q_in_iter = lane / kTopK;
    int slot_in_q = lane % kTopK;
    bool lane_active = (lane < kQPerIter * kTopK);

    // Per-warp packed cursor: total_rows int16 entries -> ceil(total_rows/2) int32.
    int packed_per_warp = (total_rows + 1) >> 1;
    int* my_cursor = smem_cursor + warp_id * packed_per_warp;

    for (int h = 0; h < H; ++h) {
        for (int i = lane; i < packed_per_warp; i += kWarpSize) my_cursor[i] = 0;
        __syncwarp();

        if (q_start_warp < q_end_warp) {
            int bi = 0;
            advance_batch_only(cu_q, B, q_start_warp, bi);

            int const* head_q2k = q2k + (size_t)h * S_Q * kTopK;
            int const* my_abs_base =
                abs_base + ((size_t)(c * kWarps + warp_id) * H + h) * total_rows;
            int* head_qidx = q_idx + (size_t)h * S_Q * kTopK;

            // (Hot-row register cache experiment showed no measurable
            // benefit; relying on L1 to keep row 0 / row total_rows-1
            // hot since they're hit every iteration in sink workloads.)

            constexpr int kUnroll = 16;
            int qi_base = q_start_warp;
            for (; qi_base + kUnroll * kQPerIter <= q_end_warp;
                 qi_base += kUnroll * kQPerIter) {
                int kvb[kUnroll];
                int qloc[kUnroll];
                int batch[kUnroll];
                int const* rmap[kUnroll];

                #pragma unroll
                for (int u = 0; u < kUnroll; ++u) {
                    int qi_u = qi_base + u * kQPerIter + q_in_iter;
                    kvb[u] = -1;
                    qloc[u] = 0;
                    batch[u] = 0;
                    if (lane_active) {
                        advance_batch_only(cu_q, B, qi_u, bi);
                        qloc[u] = qi_u - cu_q[bi];
                        batch[u] = bi;
                        kvb[u] = head_q2k[(size_t)qi_u * kTopK + slot_in_q];
                    }
                    rmap[u] = row_map + (size_t)bi * max_kv_blocks;
                }

                int row[kUnroll];
                #pragma unroll
                for (int u = 0; u < kUnroll; ++u) {
                    row[u] = -1;
                    if (lane_active && kvb[u] >= 0 && kvb[u] < max_kv_blocks)
                        row[u] = rmap[u][kvb[u]];
                }

                // Pre-issue all kUnroll abs_base loads in parallel before
                // the atomic chain so memory pipeline runs concurrently
                // with SMEM atomic-adds.
                int abs_v[kUnroll];
                #pragma unroll
                for (int u = 0; u < kUnroll; ++u) {
                    abs_v[u] = (row[u] >= 0 && row[u] < total_rows)
                        ? my_abs_base[row[u]] : 0;
                }

                #pragma unroll
                for (int u = 0; u < kUnroll; ++u) {
                    int r = row[u];
                    bool valid_edge = r >= 0 && r < total_rows;
                    unsigned int valid_mask = __ballot_sync(0xFFFFFFFFu, valid_edge);
                    unsigned int group_mask = (kTopK == 32)
                        ? 0xFFFFFFFFu
                        : (((1u << kTopK) - 1u) << (q_in_iter * kTopK));
                    unsigned int lower_lane_mask = lane == 0 ? 0u : ((1u << lane) - 1u);
                    int split_slot = __popc(valid_mask & group_mask & lower_lane_mask);
                    int valid_count = __popc(valid_mask & group_mask);
                    if (split_counts != nullptr && slot_in_q == 0) {
                        int q_abs = cu_q[batch[u]] + qloc[u];
                        split_counts[(size_t)q_abs * H + h] = valid_count;
                    }
                    if (valid_edge) {
                        int slot = atomic_inc_int16_packed(my_cursor, r);
                        int out_pos = abs_v[u] + slot;
                        head_qidx[out_pos] = qloc[u];
                        if (qsplit_idx != nullptr) {
                            qsplit_idx[(size_t)h * S_Q * kTopK + out_pos] =
                                qloc[u] | ((split_slot & 0xFF) << 24);
                        }
                    }
                }
            }
            // Tail: 1-3 iters left.
            for (; qi_base < q_end_warp; qi_base += kQPerIter) {
                int my_qi = qi_base + q_in_iter;
                bool valid_q = (my_qi < q_end_warp) && lane_active;
                int kvb_local = -1;
                int q_local = 0;
                int batch_local = 0;
                if (valid_q) {
                    advance_batch_only(cu_q, B, my_qi, bi);
                    batch_local = bi;
                    q_local = my_qi - cu_q[bi];
                    kvb_local = head_q2k[(size_t)my_qi * kTopK + slot_in_q];
                }
                int const* my_row_map = row_map + (size_t)bi * max_kv_blocks;
                int row = -1;
                if (valid_q && kvb_local >= 0 && kvb_local < max_kv_blocks) {
                    row = my_row_map[kvb_local];
                }
                bool valid_edge = row >= 0 && row < total_rows;
                unsigned int valid_mask = __ballot_sync(0xFFFFFFFFu, valid_edge);
                unsigned int group_mask = (kTopK == 32)
                    ? 0xFFFFFFFFu
                    : (((1u << kTopK) - 1u) << (q_in_iter * kTopK));
                unsigned int lower_lane_mask = lane == 0 ? 0u : ((1u << lane) - 1u);
                int split_slot = __popc(valid_mask & group_mask & lower_lane_mask);
                int valid_count = __popc(valid_mask & group_mask);
                if (split_counts != nullptr && valid_q && slot_in_q == 0) {
                    split_counts[(size_t)my_qi * H + h] = valid_count;
                }
                if (valid_edge) {
                    int slot = atomic_inc_int16_packed(my_cursor, row);
                    int out_pos = my_abs_base[row] + slot;
                    head_qidx[out_pos] = q_local;
                    if (qsplit_idx != nullptr) {
                        qsplit_idx[(size_t)h * S_Q * kTopK + out_pos] =
                            q_local | ((split_slot & 0xFF) << 24);
                    }
                }
            }
        }
        if (h + 1 < H) __syncthreads();
    }
}

template <int kTopK, int kBlockK, int kWarps>
void launch_hist_kernel(
    int const* q2k,
    int const* cu_q,
    int const* row_map,
    int* row_counts,
    int* tile_counts,
    int H,
    int B,
    int S_Q,
    int total_rows,
    int max_kv_blocks,
    int q_per_cta,
    int q_per_warp,
    size_t smem_bytes,
    int G,
    cudaStream_t stream)
{
    auto hist_fn = k2q_hist_kernel<kTopK, kBlockK, kWarps>;
    cudaFuncSetAttribute(
        hist_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    hist_fn<<<G, kWarps * kWarpSize, smem_bytes, stream>>>(
        q2k, cu_q, row_map, row_counts, tile_counts,
        H, B, S_Q, total_rows, max_kv_blocks, q_per_cta, q_per_warp);
}

template <int kTopK, int kBlockK, int kWarps>
void launch_scatter_kernel(
    int const* q2k,
    int const* cu_q,
    int const* row_map,
    int const* abs_base,
    int* q_idx,
    int* qsplit_idx,
    int* split_counts,
    int H,
    int B,
    int S_Q,
    int total_rows,
    int max_kv_blocks,
    int q_per_cta,
    int q_per_warp,
    int max_seqlen_q,
    size_t smem_bytes,
    int G,
    cudaStream_t stream)
{
    auto scat_fn = k2q_scatter_kernel<kTopK, kBlockK, kWarps>;
    cudaFuncSetAttribute(
        scat_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    scat_fn<<<G, kWarps * kWarpSize, smem_bytes, stream>>>(
        q2k, cu_q, row_map, abs_base, q_idx, qsplit_idx, split_counts,
        H, B, S_Q, total_rows, max_kv_blocks, q_per_cta, q_per_warp, max_seqlen_q);
}

#define K2Q_DISPATCH_WARPS(topk, kwarps, BODY)          \
    do {                                                \
        if ((kwarps) == 4) {                            \
            BODY(topk, 4);                              \
        } else if ((kwarps) == 2) {                     \
            BODY(topk, 2);                              \
        } else {                                        \
            BODY(topk, 1);                              \
        }                                               \
    } while (0)

#define K2Q_DISPATCH_TOPK_WARPS(topk, kwarps, BODY)    \
    do {                                                \
        if ((topk) == 16) {                             \
            K2Q_DISPATCH_WARPS(16, kwarps, BODY);      \
        } else if ((topk) == 8) {                       \
            K2Q_DISPATCH_WARPS(8, kwarps, BODY);        \
        } else if ((topk) == 32) {                      \
            K2Q_DISPATCH_WARPS(32, kwarps, BODY);       \
        } else if ((topk) == 4) {                       \
            K2Q_DISPATCH_WARPS(4, kwarps, BODY);        \
        }                                               \
    } while (0)

}  // namespace

extern "C" void k2q_launch_row_map(
    int const* cu_k,
    int* row_map,
    int* row_coords,
    int B,
    int max_kv_blocks,
    cudaStream_t stream)
{
    if (max_kv_blocks <= 0) {
        return;
    }
    k2q_build_row_map_kernel<128><<<max_kv_blocks, 32, 0, stream>>>(
        cu_k, row_map, row_coords, B, max_kv_blocks);
}

extern "C" void k2q_launch_hist(
    int topk,
    int kwarps,
    int const* q2k,
    int const* cu_q,
    int const* row_map,
    int* row_counts,
    int* tile_counts,
    int H,
    int B,
    int S_Q,
    int total_rows,
    int max_kv_blocks,
    int q_per_cta,
    int q_per_warp,
    size_t smem_bytes,
    int G,
    cudaStream_t stream)
{
#define LAUNCH_HIST(TOPK, WARPS) \
    launch_hist_kernel<TOPK, 128, WARPS>( \
        q2k, cu_q, row_map, row_counts, tile_counts, \
        H, B, S_Q, total_rows, max_kv_blocks, q_per_cta, q_per_warp, \
        smem_bytes, G, stream)
    K2Q_DISPATCH_TOPK_WARPS(topk, kwarps, LAUNCH_HIST);
#undef LAUNCH_HIST
}

extern "C" void k2q_launch_row_prefix(
    int const* row_counts,
    int* row_ptr,
    int const* row_coords,
    int* scheduler_metadata,
    int* work_count,
    int total_rows,
    int target_q_per_cta,
    int work_capacity,
    int H,
    cudaStream_t stream)
{
    k2q_row_prefix_kernel<1024><<<H, 1024, 0, stream>>>(
        row_counts, row_ptr, row_coords, scheduler_metadata, work_count,
        total_rows, target_q_per_cta, work_capacity);
}

extern "C" void k2q_launch_tile_prefix(
    int* tile_counts,
    int const* row_ptr,
    int H,
    int total_rows,
    int G_total,
    cudaStream_t stream)
{
    constexpr int kPtRowsPerBlock = 8;
    constexpr int kPtThreads = 256;
    int blocks_per_h = (total_rows + kPtRowsPerBlock - 1) / kPtRowsPerBlock;
    int pt_grid = H * blocks_per_h;
    if (pt_grid < 1) {
        pt_grid = 1;
    }
    size_t pt_smem = (size_t)kPtRowsPerBlock * G_total * sizeof(int);
    auto tprefix_smem_fn = k2q_tile_prefix_smem_kernel<kPtThreads, kPtRowsPerBlock>;
    cudaFuncSetAttribute(
        tprefix_smem_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)pt_smem);
    tprefix_smem_fn<<<pt_grid, kPtThreads, pt_smem, stream>>>(
        tile_counts, row_ptr, H, total_rows, G_total);
}

extern "C" void k2q_launch_scatter(
    int topk,
    int kwarps,
    int const* q2k,
    int const* cu_q,
    int const* row_map,
    int const* abs_base,
    int* q_idx,
    int* qsplit_idx,
    int* split_counts,
    int H,
    int B,
    int S_Q,
    int total_rows,
    int max_kv_blocks,
    int q_per_cta,
    int q_per_warp,
    int max_seqlen_q,
    size_t smem_bytes,
    int G,
    cudaStream_t stream)
{
#define LAUNCH_SCATTER(TOPK, WARPS) \
    launch_scatter_kernel<TOPK, 128, WARPS>( \
        q2k, cu_q, row_map, abs_base, q_idx, qsplit_idx, split_counts, \
        H, B, S_Q, total_rows, max_kv_blocks, q_per_cta, q_per_warp, max_seqlen_q, \
        smem_bytes, G, stream)
    K2Q_DISPATCH_TOPK_WARPS(topk, kwarps, LAUNCH_SCATTER);
#undef LAUNCH_SCATTER
}
