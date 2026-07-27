// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#pragma once

#include <cuda_runtime.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void k2q_launch_row_map(
    int const* cu_k,
    int* row_map,
    int* row_coords,
    int B,
    int max_kv_blocks,
    cudaStream_t stream);

void k2q_launch_hist(
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
    cudaStream_t stream);

void k2q_launch_row_prefix(
    int const* row_counts,
    int* row_ptr,
    int const* row_coords,
    int* scheduler_metadata,
    int* work_count,
    int total_rows,
    int target_q_per_cta,
    int work_capacity,
    int H,
    cudaStream_t stream);

void k2q_launch_tile_prefix(
    int* tile_counts,
    int const* row_ptr,
    int H,
    int total_rows,
    int G_total,
    cudaStream_t stream);

void k2q_launch_scatter(
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
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
