import pathlib, re
ROOT = pathlib.Path("/sparse/flashinfer/flashinfer/cute_dsl/sparse/sm100_blk64")

# --- hoist total_k_padded / total_sparse_blocks above the paged branch -------
p = ROOT / "flash_fwd_launch_template.h"
s = p.read_text()
decl = ("  int const total_k_padded = ((seq_k + kSparseBlockSize - 1) / kSparseBlockSize) * kSparseBlockSize;\n"
        "  int const total_sparse_blocks = total_k_padded / kSparseBlockSize;\n")
assert decl in s
s = s.replace(decl, "", 1)                       # remove from inside the else
s = s.replace("  // ======== Prepare K/V ========", decl + "  // ======== Prepare K/V ========", 1)
p.write_text(s)
print("hoisted total_k_padded/total_sparse_blocks")

# --- explicit instantiations must carry the new parameter -------------------
inst = ROOT / "instantiations"
for f in sorted(inst.glob("*.cu")):
    t = f.read_text()
    if "int num_heads_kv" in t:
        continue
    t2 = t.replace("torch::Tensor q2k_block_nums)", "torch::Tensor q2k_block_nums, int num_heads_kv)")
    if t2 != t:
        f.write_text(t2)
        print("patched instantiation", f.name)
