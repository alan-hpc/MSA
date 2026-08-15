import pathlib
p = pathlib.Path("/sparse/flashinfer/flashinfer/cute_dsl/sparse/sm100_blk64/flash_fwd_launch_template.cu")
s = p.read_text()
old = '  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q/k/v must be 4D");'
new = ('  // Paged mode hands K/V as a 3D page pool [num_pages*H_kv*2, 64, 64].\n'
       '  TORCH_CHECK(q.dim() == 4, "q must be 4D");\n'
       '  if (num_heads_kv > 0) {\n'
       '    TORCH_CHECK(k.dim() == 3 && v.dim() == 3, "paged k/v must be 3D page pools");\n'
       '  } else {\n'
       '    TORCH_CHECK(k.dim() == 4 && v.dim() == 4, "dense k/v must be 4D");\n'
       '  }')
assert old in s, "check anchor"
p.write_text(s.replace(old, new, 1))
print("relaxed dim check")

# seq_k comes from k.size(1); in paged mode that is the page size, not the context
h = pathlib.Path("/sparse/flashinfer/flashinfer/cute_dsl/sparse/sm100_blk64/flash_fwd_launch_template.h")
t = h.read_text()
oldk = "  const int seq_k = static_cast<int>(k.size(1));"
newk = ("  // In paged mode k is [num_pages*H_kv*2, 64, 64]; the context length is implied by\n"
        "  // the pages the index selects, so derive a seq_k that keeps the legacy sizing math\n"
        "  // well-formed without pretending it is the real context.\n"
        "  const int seq_k = (num_heads_kv > 0)\n"
        "                        ? static_cast<int>(k.size(0)) / (num_heads_kv * 2) * kSparseBlockSize\n"
        "                        : static_cast<int>(k.size(1));")
assert oldk in t, "seq_k anchor"
h.write_text(t.replace(oldk, newk, 1))
print("seq_k made paged-aware")
