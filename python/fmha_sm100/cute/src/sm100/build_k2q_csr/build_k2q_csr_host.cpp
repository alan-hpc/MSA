// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

// Host orchestration for the q2k -> k2q CSR builder (compiled with g++, not nvcc).

#include "build_k2q_csr_launch.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <algorithm>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INT(x) TORCH_CHECK((x).scalar_type() == at::kInt, #x " must be int32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_INT(x)

namespace {

template <int kTopK>
void launch_pipeline(
    at::Tensor q2k,
    at::Tensor cu_q,
    at::Tensor cu_k,
    at::Tensor row_ptr,
    at::Tensor q_idx,
    int total_rows,
    int max_kv_blocks,
    at::Tensor scheduler_metadata = at::Tensor(),
    at::Tensor work_count = at::Tensor(),
    at::Tensor qsplit_idx = at::Tensor(),
    at::Tensor split_counts = at::Tensor(),
    int target_q_per_cta = 1,
    int work_capacity = 0,
    int max_seqlen_q = 0)
{
    int H = (int)q2k.size(0);
    int S_Q = (int)q2k.size(1);
    int topK = (int)q2k.size(2);
    TORCH_CHECK(topK == kTopK, "topK runtime != template kTopK");
    int B = (int)cu_q.size(0) - 1;
    auto device = q2k.device();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_CUDA_CHECK(cudaMemsetAsync(
        row_ptr.data_ptr<int>(), 0,
        (size_t)H * (total_rows + 1) * sizeof(int), stream));
    AT_CUDA_CHECK(cudaMemsetAsync(
        q_idx.data_ptr<int>(), 0xFF,
        (size_t)H * S_Q * kTopK * sizeof(int), stream));

    auto opts = at::TensorOptions().dtype(at::kInt).device(device);
    auto row_counts = at::zeros({H, total_rows}, opts);
    auto row_map = at::empty({B, max_kv_blocks}, opts);
    bool emit_schedule = scheduler_metadata.defined();
    auto row_coords = emit_schedule ? at::empty({total_rows, 2}, opts) : at::Tensor();
    int* scheduler_metadata_ptr = emit_schedule ? scheduler_metadata.data_ptr<int>() : nullptr;
    int* work_count_ptr = emit_schedule ? work_count.data_ptr<int>() : nullptr;
    int* qsplit_idx_ptr = emit_schedule ? qsplit_idx.data_ptr<int>() : nullptr;
    int* split_counts_ptr = emit_schedule ? split_counts.data_ptr<int>() : nullptr;
    int* row_coords_ptr = emit_schedule ? row_coords.data_ptr<int>() : nullptr;
    if (emit_schedule) {
        AT_CUDA_CHECK(cudaMemsetAsync(work_count_ptr, 0, sizeof(int), stream));
        AT_CUDA_CHECK(cudaMemsetAsync(
            scheduler_metadata_ptr, 0,
            (size_t)work_capacity * 6 * sizeof(int), stream));
    }

    int dev = q2k.get_device();
    int num_sms = 0;
    AT_CUDA_CHECK(cudaDeviceGetAttribute(
        &num_sms, cudaDevAttrMultiProcessorCount, dev));

    int per_warp_smem = ((total_rows + 1) >> 1) * (int)sizeof(int);
    int kWarps_pick = 4;
    while (kWarps_pick > 1 && (kWarps_pick * per_warp_smem) * 2 > 228 * 1024) {
        kWarps_pick >>= 1;
    }
    if (kWarps_pick < 1) {
        kWarps_pick = 1;
    }

    int per_cta_smem_bytes = kWarps_pick * per_warp_smem;
    int max_ctas_per_sm = std::max(
        1, (228 * 1024) / std::max(1, per_cta_smem_bytes));
    if (max_ctas_per_sm > 8) {
        max_ctas_per_sm = 8;
    }
    constexpr int kMinQPerCta = 256;
    int target_g = num_sms * std::min(max_ctas_per_sm, 3);
    int max_g_for_q = (S_Q + kMinQPerCta - 1) / kMinQPerCta;
    int G = std::min({target_g, max_g_for_q, S_Q});
    if (G < 1) {
        G = 1;
    }
    int q_per_cta = (S_Q + G - 1) / G;
    G = (S_Q + q_per_cta - 1) / q_per_cta;
    int q_per_warp = (q_per_cta + kWarps_pick - 1) / kWarps_pick;
    int G_total = G * kWarps_pick;

    auto tile_counts = at::empty({G_total, H, total_rows}, opts);
    size_t smem_bytes = (size_t)kWarps_pick * per_warp_smem;

    k2q_launch_row_map(
        cu_k.data_ptr<int>(), row_map.data_ptr<int>(), row_coords_ptr,
        B, max_kv_blocks, stream);

    k2q_launch_hist(
        kTopK, kWarps_pick,
        q2k.data_ptr<int>(), cu_q.data_ptr<int>(), row_map.data_ptr<int>(),
        row_counts.data_ptr<int>(), tile_counts.data_ptr<int>(),
        H, B, S_Q, total_rows, max_kv_blocks, q_per_cta, q_per_warp,
        smem_bytes, G, stream);

    k2q_launch_row_prefix(
        row_counts.data_ptr<int>(), row_ptr.data_ptr<int>(),
        emit_schedule ? row_coords.data_ptr<int>() : nullptr,
        scheduler_metadata_ptr, work_count_ptr,
        total_rows, target_q_per_cta, work_capacity, H, stream);

    k2q_launch_tile_prefix(
        tile_counts.data_ptr<int>(), row_ptr.data_ptr<int>(),
        H, total_rows, G_total, stream);

    k2q_launch_scatter(
        kTopK, kWarps_pick,
        q2k.data_ptr<int>(), cu_q.data_ptr<int>(), row_map.data_ptr<int>(),
        tile_counts.data_ptr<int>(), q_idx.data_ptr<int>(),
        qsplit_idx_ptr, split_counts_ptr,
        H, B, S_Q, total_rows, max_kv_blocks, q_per_cta, q_per_warp,
        max_seqlen_q, smem_bytes, G, stream);
}

}  // namespace

void run_build_k2q_csr(
    at::Tensor q2k,
    at::Tensor cu_q,
    at::Tensor cu_k,
    at::Tensor row_ptr,
    at::Tensor q_idx,
    int64_t topk,
    int64_t blk_kv,
    int64_t total_rows,
    int64_t max_kv_blocks)
{
    CHECK_INPUT(q2k);
    CHECK_INPUT(cu_q);
    CHECK_INPUT(cu_k);
    CHECK_INPUT(row_ptr);
    CHECK_INPUT(q_idx);
    TORCH_CHECK(blk_kv == 128, "build_k2q_csr only supports blk_kv == 128");
    int H = (int)q2k.size(0);
    int S_Q = (int)q2k.size(1);
    int tr = (int)total_rows;
    int mkv = (int)max_kv_blocks;
    TORCH_CHECK(tr >= 0 && mkv >= 0,
                "total_rows / max_kv_blocks must be non-negative");
    TORCH_CHECK(row_ptr.size(0) == H && row_ptr.size(1) == tr + 1,
                "row_ptr shape mismatch");
    TORCH_CHECK(q_idx.size(0) == H && q_idx.size(1) == (int64_t)S_Q * (int)topk,
                "q_idx shape mismatch");
    if (S_Q == 0 || tr == 0 || H == 0 || mkv == 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        AT_CUDA_CHECK(cudaMemsetAsync(
            row_ptr.data_ptr<int>(), 0,
            (size_t)H * (tr + 1) * sizeof(int), stream));
        AT_CUDA_CHECK(cudaMemsetAsync(
            q_idx.data_ptr<int>(), 0xFF,
            (size_t)H * S_Q * (int)topk * sizeof(int), stream));
        return;
    }

    if (topk == 16) {
        launch_pipeline<16>(q2k, cu_q, cu_k, row_ptr, q_idx, tr, mkv);
    } else if (topk == 8) {
        launch_pipeline<8>(q2k, cu_q, cu_k, row_ptr, q_idx, tr, mkv);
    } else if (topk == 32) {
        launch_pipeline<32>(q2k, cu_q, cu_k, row_ptr, q_idx, tr, mkv);
    } else if (topk == 4) {
        launch_pipeline<4>(q2k, cu_q, cu_k, row_ptr, q_idx, tr, mkv);
    } else {
        TORCH_CHECK(false, "unsupported topK ", topk, " (expected 4, 8, 16, or 32)");
    }
}

void run_build_k2q_csr_with_schedule(
    at::Tensor q2k,
    at::Tensor cu_q,
    at::Tensor cu_k,
    at::Tensor row_ptr,
    at::Tensor q_idx,
    at::Tensor scheduler_metadata,
    at::Tensor work_count,
    at::Tensor qsplit_idx,
    at::Tensor split_counts,
    int64_t topk,
    int64_t blk_kv,
    int64_t total_rows,
    int64_t max_kv_blocks,
    int64_t target_q_per_cta,
    int64_t work_capacity,
    int64_t max_seqlen_q)
{
    CHECK_INPUT(q2k);
    CHECK_INPUT(cu_q);
    CHECK_INPUT(cu_k);
    CHECK_INPUT(row_ptr);
    CHECK_INPUT(q_idx);
    CHECK_INPUT(scheduler_metadata);
    CHECK_INPUT(work_count);
    CHECK_INPUT(qsplit_idx);
    CHECK_INPUT(split_counts);
    TORCH_CHECK(blk_kv == 128, "build_k2q_csr only supports blk_kv == 128");
    int H = (int)q2k.size(0);
    int S_Q = (int)q2k.size(1);
    int tr = (int)total_rows;
    int mkv = (int)max_kv_blocks;
    int target = (int)target_q_per_cta;
    int capacity = (int)work_capacity;
    int max_sq = (int)max_seqlen_q;
    TORCH_CHECK(tr >= 0 && mkv >= 0 && target > 0 && capacity > 0 && max_sq >= 0,
                "invalid schedule sizing arguments");
    TORCH_CHECK(row_ptr.size(0) == H && row_ptr.size(1) == tr + 1,
                "row_ptr shape mismatch");
    TORCH_CHECK(q_idx.size(0) == H && q_idx.size(1) == (int64_t)S_Q * (int)topk,
                "q_idx shape mismatch");
    TORCH_CHECK(qsplit_idx.sizes() == q_idx.sizes(), "qsplit_idx shape mismatch");
    TORCH_CHECK(scheduler_metadata.size(0) == capacity && scheduler_metadata.size(1) == 6,
                "scheduler_metadata shape mismatch");
    TORCH_CHECK(work_count.numel() == 1, "work_count must have one int32 element");
    TORCH_CHECK(split_counts.dim() == 2 && split_counts.size(0) == S_Q
                && split_counts.size(1) == H,
                "split_counts shape mismatch");
    if (S_Q == 0 || tr == 0 || H == 0 || mkv == 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        AT_CUDA_CHECK(cudaMemsetAsync(
            row_ptr.data_ptr<int>(), 0,
            (size_t)H * (tr + 1) * sizeof(int), stream));
        AT_CUDA_CHECK(cudaMemsetAsync(
            q_idx.data_ptr<int>(), 0xFF,
            (size_t)H * S_Q * (int)topk * sizeof(int), stream));
        AT_CUDA_CHECK(cudaMemsetAsync(work_count.data_ptr<int>(), 0, sizeof(int), stream));
        if (split_counts.numel() > 0) {
            AT_CUDA_CHECK(cudaMemsetAsync(
                split_counts.data_ptr<int>(), 0,
                (size_t)split_counts.numel() * sizeof(int), stream));
        }
        return;
    }

    if (topk == 16) {
        launch_pipeline<16>(
            q2k, cu_q, cu_k, row_ptr, q_idx, tr, mkv,
            scheduler_metadata, work_count, qsplit_idx, split_counts,
            target, capacity, max_sq);
    } else if (topk == 8) {
        launch_pipeline<8>(
            q2k, cu_q, cu_k, row_ptr, q_idx, tr, mkv,
            scheduler_metadata, work_count, qsplit_idx, split_counts,
            target, capacity, max_sq);
    } else if (topk == 32) {
        launch_pipeline<32>(
            q2k, cu_q, cu_k, row_ptr, q_idx, tr, mkv,
            scheduler_metadata, work_count, qsplit_idx, split_counts,
            target, capacity, max_sq);
    } else if (topk == 4) {
        launch_pipeline<4>(
            q2k, cu_q, cu_k, row_ptr, q_idx, tr, mkv,
            scheduler_metadata, work_count, qsplit_idx, split_counts,
            target, capacity, max_sq);
    } else {
        TORCH_CHECK(false, "unsupported topK ", topk, " (expected 4, 8, 16, or 32)");
    }
}
