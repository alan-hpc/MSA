// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#include "sparse_topk_select.cuh"
#include "tvm_ffi_utils.h"

using namespace flashinfer;
using tvm::ffi::Optional;

// v2.5_oob_clamp_in_kernel:
//   Adds num_valid_pages parameter (between topk and stream_ptr).  When
//   num_valid_pages < max_k_tiles, indices >= num_valid_pages are emitted as
//   -1 and sorted to the tail by the kernel — replacing the prod wrapper's
//   torch.where + sort + torch.where post-processing (~84-101 us measured).
//
//   To disable clamping, pass num_valid_pages = max_k_tiles (or any value
//   >= max_k_tiles).
// v3.0_msa_config:
//   * `topk` is a runtime value in [1, 64] (was pinned to 16).
//   * `head_reduce` / implicit group_size fold GQA head aggregation into the
//     transpose stage: max_score may carry `num_out_heads * group_size` head
//     rows, and output_indices.size(1) fixes `num_out_heads`.
//   * `causal_end_block` is an optional [total_qo_len] int32 tensor of
//     per-query exclusive causal block bounds used by the force_end window;
//     an empty tensor selects the legacy global `num_valid_pages` bound.
void sparse_topk_select(TensorView max_score, TensorView output_indices,
                        TensorView workspace_buffer, int64_t topk,
                        int64_t num_valid_pages,
                        int64_t force_begin_blocks, int64_t force_end_blocks,
                        int64_t head_reduce, Optional<TensorView> causal_end_block,
                        int64_t head_outermost, int64_t stream_ptr) {
  CHECK_INPUT(max_score);
  CHECK_INPUT(output_indices);
  CHECK_INPUT(workspace_buffer);
  CHECK_DIM(3, max_score);
  CHECK_DIM(3, output_indices);
  CHECK_DIM(1, workspace_buffer);

  TVM_FFI_ICHECK(encode_dlpack_dtype(max_score.dtype()) == float32_code)
      << "max_score must be float32";
  TVM_FFI_ICHECK(encode_dlpack_dtype(output_indices.dtype()) == int32_code)
      << "output_indices must be int32";
  TVM_FFI_ICHECK(encode_dlpack_dtype(workspace_buffer.dtype()) == int32_code)
      << "workspace_buffer must be int32";

  const int64_t num_in_heads = max_score.size(0);
  const int64_t max_k_tiles = max_score.size(1);
  const int64_t total_qo_len = max_score.size(2);
  // head_outermost flips the meaning of the first two output dims.
  const int64_t num_qo_heads = head_outermost ? output_indices.size(0) : output_indices.size(1);
  const int64_t out_rows = head_outermost ? output_indices.size(1) : output_indices.size(0);

  TVM_FFI_ICHECK(head_outermost == 0 || head_outermost == 1)
      << "head_outermost must be 0 or 1, got " << head_outermost;
  TVM_FFI_ICHECK(out_rows == total_qo_len)
      << "output_indices token dimension (" << out_rows << ") must equal total_qo_len ("
      << total_qo_len << ")";
  TVM_FFI_ICHECK(output_indices.size(2) == topk);
  TVM_FFI_ICHECK(topk >= 1 && topk <= static_cast<int64_t>(sparse_topk::kSparseTopkMaxK))
      << "topk must be in [1, " << sparse_topk::kSparseTopkMaxK << "], got " << topk;
  TVM_FFI_ICHECK(num_valid_pages > 0)
      << "num_valid_pages must be > 0, got " << num_valid_pages;
  TVM_FFI_ICHECK(num_qo_heads > 0 && num_in_heads % num_qo_heads == 0)
      << "max_score head count (" << num_in_heads << ") must be a multiple of the output head "
      << "count (" << num_qo_heads << ")";
  const int64_t group_size = num_in_heads / num_qo_heads;
  TVM_FFI_ICHECK(head_reduce >= 0 && head_reduce <= 2)
      << "head_reduce must be 0 (none), 1 (sum) or 2 (max), got " << head_reduce;
  TVM_FFI_ICHECK(head_reduce != 0 || group_size == 1)
      << "head_reduce=0 (none) requires max_score and output_indices to have the same head count";

  const int32_t* causal_end_ptr = nullptr;
  if (causal_end_block.has_value()) {
    TensorView ceb = causal_end_block.value();
    CHECK_INPUT(ceb);
    CHECK_DIM(1, ceb);
    TVM_FFI_ICHECK(encode_dlpack_dtype(ceb.dtype()) == int32_code)
        << "causal_end_block must be int32";
    TVM_FFI_ICHECK(ceb.size(0) == total_qo_len)
        << "causal_end_block must have total_qo_len entries";
    causal_end_ptr = static_cast<const int32_t*>(ceb.data_ptr());
  }

  const size_t needed_workspace = sparse_topk::SparseTopKWorkspaceSize(
      static_cast<uint32_t>(total_qo_len), static_cast<uint32_t>(num_qo_heads),
      static_cast<uint32_t>(max_k_tiles));
  TVM_FFI_ICHECK(static_cast<size_t>(workspace_buffer.size(0)) >= needed_workspace)
      << "workspace_buffer too small: need " << needed_workspace << " int32 elements";

  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  cudaError_t status = sparse_topk::SparseTopKSelect(
      static_cast<const float*>(max_score.data_ptr()),
      static_cast<int32_t*>(output_indices.data_ptr()),
      static_cast<int32_t*>(workspace_buffer.data_ptr()),
      static_cast<uint32_t>(total_qo_len), static_cast<uint32_t>(num_qo_heads),
      static_cast<uint32_t>(max_k_tiles), static_cast<uint32_t>(num_valid_pages),
      static_cast<uint32_t>(force_begin_blocks), static_cast<uint32_t>(force_end_blocks),
      static_cast<uint32_t>(topk), static_cast<uint32_t>(group_size),
      static_cast<int>(head_reduce), causal_end_ptr,
      static_cast<uint32_t>(head_outermost), stream);

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "sparse_topk_select failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sparse_topk_select, sparse_topk_select);
