"""Host-side paged-KV + GQA entry for sm100_blk64.

When num_heads_kv > 0 the caller hands K and V in the layout the kernel already
wants -- a page pool indexed ((page * H_kv) + kv_head) * 2 + dim_half, each plane
64x64 -- so the per-call permute/pad/contiguous rebuild of K and V is skipped
entirely, and q2k_block_index carries physical page ids.
"""
import pathlib

ROOT = pathlib.Path("/sparse/flashinfer/flashinfer/cute_dsl/sparse/sm100_blk64")

# ---------------------------------------------------------------- launch template
p = ROOT / "flash_fwd_launch_template.h"
s = p.read_text()

s = s.replace(
    "    int block_sparse_num, torch::Tensor block_sizes, float softmax_scale,\n"
    "    torch::Tensor q2k_block_nums) {",
    "    int block_sparse_num, torch::Tensor block_sizes, float softmax_scale,\n"
    "    torch::Tensor q2k_block_nums, int num_heads_kv = 0) {", 1)
assert "int num_heads_kv = 0) {" in s, "launch signature"

# K/V prep becomes conditional: in paged mode the pools are used as given.
old_k = """  // ======== Prepare K ========
  auto k_bhsd = k.permute({0, 2, 1, 3}).contiguous();"""
new_k = """  // ======== Prepare K/V ========
  // Paged mode: k and v are already the page pool the kernel indexes, so none of
  // the permute/pad/contiguous work below runs. It is the dominant host cost at
  // long sequence (hundreds of MB copied per call at 32K x 32 heads).
  bool const kv_paged = num_heads_kv > 0;
  int const num_pages = kv_paged ? static_cast<int>(k.size(0)) / (num_heads_kv * 2) : 0;
  torch::Tensor k_blocks, v_blocks;
  if (kv_paged) {
    TORCH_CHECK(k.dim() == 3 && v.dim() == 3,
                "paged blk64 expects k/v as [num_pages*H_kv*2, 64, 64]");
    TORCH_CHECK(k.size(0) % (num_heads_kv * 2) == 0, "k pool not a multiple of H_kv*2");
    k_blocks = k.contiguous();
    v_blocks = v.contiguous();
  } else {
  auto k_bhsd = k.permute({0, 2, 1, 3}).contiguous();"""
assert old_k in s, "K prep anchor"
s = s.replace(old_k, new_k, 1)

# close the else-branch after v_blocks is built
old_v_end = """          .reshape({batch * heads * total_sparse_blocks * 2, kDimHalf, kSparseBlockSize})
          .contiguous();
"""
new_v_end = """          .reshape({batch * heads * total_sparse_blocks * 2, kDimHalf, kSparseBlockSize})
          .contiguous();
  }
"""
assert old_v_end in s, "V prep end anchor"
s = s.replace(old_v_end, new_v_end, 1)

# the two block tensors are declared outside now -- drop the inner `auto`
s = s.replace("  auto k_blocks =\n      k_padded.view(", "  k_blocks =\n      k_padded.view(", 1)
s = s.replace("  auto v_blocks =\n      v_padded.view(", "  v_blocks =\n      v_padded.view(", 1)

# pass the new fields through to the kernel Arguments
s = s.replace("          ptr_q2k_block_nums,\n          raw_block_sparse_num,\n      },",
              "          ptr_q2k_block_nums,\n          raw_block_sparse_num,\n"
              "          num_heads_kv,\n          num_pages,\n      },", 1)
assert "          num_pages,\n      }," in s, "Arguments plumb"
p.write_text(s)
print("launch template patched")

# ---------------------------------------------------------------- .cu dispatcher
cu = ROOT / "flash_fwd_launch_template.cu"
c = cu.read_text()
c = c.replace("torch::Tensor q2k_block_nums)", "torch::Tensor q2k_block_nums, int num_heads_kv)")
c = c.replace("q2k_block_nums)", "q2k_block_nums, num_heads_kv)")
# the declaration's own parameter list must keep its name
c = c.replace("torch::Tensor q2k_block_nums, int num_heads_kv, int num_heads_kv)",
              "torch::Tensor q2k_block_nums, int num_heads_kv)")
cu.write_text(c)
print("dispatcher patched")

# ---------------------------------------------------------------- bindings
b = ROOT / "bindings.cpp"
t = b.read_text()
t = t.replace("torch::Tensor q2k_block_nums);", "torch::Tensor q2k_block_nums, int num_heads_kv);")
t = t.replace('py::arg("q2k_block_nums") = torch::Tensor());',
              'py::arg("q2k_block_nums") = torch::Tensor(), py::arg("num_heads_kv") = 0);')
b.write_text(t)
print("bindings patched")
