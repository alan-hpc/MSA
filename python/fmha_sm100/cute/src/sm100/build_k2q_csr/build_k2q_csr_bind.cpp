// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

// Host-side pybind11 bindings for build_k2q_csr (compiled with g++, not nvcc).

#include <torch/extension.h>

void run_build_k2q_csr(
    torch::Tensor q2k,
    torch::Tensor cu_q,
    torch::Tensor cu_k,
    torch::Tensor row_ptr,
    torch::Tensor q_idx,
    int64_t topk,
    int64_t blk_kv,
    int64_t total_rows,
    int64_t max_kv_blocks);

void run_build_k2q_csr_with_schedule(
    torch::Tensor q2k,
    torch::Tensor cu_q,
    torch::Tensor cu_k,
    torch::Tensor row_ptr,
    torch::Tensor q_idx,
    torch::Tensor scheduler_metadata,
    torch::Tensor work_count,
    torch::Tensor qsplit_idx,
    torch::Tensor split_counts,
    int64_t topk,
    int64_t blk_kv,
    int64_t total_rows,
    int64_t max_kv_blocks,
    int64_t target_q_per_cta,
    int64_t work_capacity,
    int64_t max_seqlen_q);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_build_k2q_csr", &run_build_k2q_csr,
          "q2k -> k2q CSR build (sorted within row)",
          pybind11::arg("q2k"),
          pybind11::arg("cu_q"),
          pybind11::arg("cu_k"),
          pybind11::arg("row_ptr"),
          pybind11::arg("q_idx"),
          pybind11::arg("topk"),
          pybind11::arg("blk_kv"),
          pybind11::arg("total_rows"),
          pybind11::arg("max_kv_blocks"));
    m.def("run_build_k2q_csr_with_schedule", &run_build_k2q_csr_with_schedule,
          "q2k -> k2q CSR build with fused attention schedule metadata",
          pybind11::arg("q2k"),
          pybind11::arg("cu_q"),
          pybind11::arg("cu_k"),
          pybind11::arg("row_ptr"),
          pybind11::arg("q_idx"),
          pybind11::arg("scheduler_metadata"),
          pybind11::arg("work_count"),
          pybind11::arg("qsplit_idx"),
          pybind11::arg("split_counts"),
          pybind11::arg("topk"),
          pybind11::arg("blk_kv"),
          pybind11::arg("total_rows"),
          pybind11::arg("max_kv_blocks"),
          pybind11::arg("target_q_per_cta"),
          pybind11::arg("work_capacity"),
          pybind11::arg("max_seqlen_q"));
}
