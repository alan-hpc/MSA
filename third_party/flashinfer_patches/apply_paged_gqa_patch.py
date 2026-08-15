"""Patch FlashInfer's sm100_blk64 kernel for paged KV + GQA.

Three changes, all in the K/V addressing:

  before:  tma_idx = ((batch*H + head) * total_sparse_blocks + sparse_idx) * 2 + half
           - K/V indexed per (batch, query head): GQA needs KV replicated to every
             query head, and the host rebuilds the blocked layout on every call.

  after:   tma_idx = (sparse_idx * H_kv + head / qhead_per_kv) * 2 + half
           - sparse_idx is a *physical page id*, so a paged KV cache is indexed
             directly and the host does no per-call layout rebuild;
           - the query head maps to its KV head, so GQA reads KV once.

The host translates logical->physical page ids into q2k_block_index before the
call, so the kernel needs no page-table logic of its own.
"""
import re, sys, pathlib

ROOT = pathlib.Path("/sparse/flashinfer/flashinfer/cute_dsl/sparse/sm100_blk64")
ML = ROOT / "mainloop_fwd_sm100.hpp"

src = ML.read_text()
orig = src

# --- 1. Params: carry the KV-head count and the GQA ratio -------------------
src = src.replace(
    "    int kv_tma_per_head = 0;",
    "    int kv_tma_per_head = 0;\n"
    "    int num_heads_kv = 0;    // 0 => legacy dense path (num_heads_kv == num_heads)\n"
    "    int qhead_per_kv = 1;    // num_heads / num_heads_kv",
    1,
)
assert "num_heads_kv = 0;    // 0 =>" in src, "Params patch failed"

# --- 2. Arguments: same two fields, host-supplied ---------------------------
src = src.replace(
    "    int raw_block_sparse_num = 0;  // original unpadded block_sparse_num (for phantom clamping)",
    "    int raw_block_sparse_num = 0;  // original unpadded block_sparse_num (for phantom clamping)\n"
    "    int num_heads_kv = 0;          // 0 => dense/legacy: KV is indexed per query head\n"
    "    int num_pages = 0;             // physical pages in the KV cache (paged mode)",
    1,
)
assert "int num_pages = 0;" in src, "Arguments patch failed"

# --- 3. K/V addressing ------------------------------------------------------
old_base = "      int kv_tma_base = (batch * params.num_heads + head) * params.kv_tma_per_head;"
new_base = """      // Paged + GQA: K/V live in a global page pool indexed by physical page and
      // KV head, so there is no batch term and no per-query-head replication.
      // Legacy (num_heads_kv == 0) keeps the dense per-(batch, head) layout.
      bool const kv_paged = params.num_heads_kv > 0;
      int const kv_head = kv_paged ? (head / params.qhead_per_kv) : head;
      int kv_tma_base = kv_paged ? 0
                                 : (batch * params.num_heads + head) * params.kv_tma_per_head;"""
assert old_base in src, "kv_tma_base anchor not found"
src = src.replace(old_base, new_base, 1)

old_idx = "            int tma_idx = kv_tma_base + sparse_idx * kDimHalves + h;"
new_idx = ("            int tma_idx = kv_paged\n"
           "                              ? (sparse_idx * params.num_heads_kv + kv_head) * kDimHalves + h\n"
           "                              : kv_tma_base + sparse_idx * kDimHalves + h;")
n = src.count(old_idx)
assert n >= 1, "tma_idx anchor not found"
src = src.replace(old_idx, new_idx)
print(f"patched {n} tma_idx site(s)")

# --- 4. TMA descriptor extent must cover the page pool ----------------------
src = src.replace(
    "    auto shape_k =\n"
    "        make_shape(kSparseBlockSize, kDimHalf, batch * heads * total_sparse_blocks * kDimHalves);",
    "    int const kv_planes = (args.num_heads_kv > 0)\n"
    "                              ? args.num_pages * args.num_heads_kv * kDimHalves\n"
    "                              : batch * heads * total_sparse_blocks * kDimHalves;\n"
    "    auto shape_k = make_shape(kSparseBlockSize, kDimHalf, kv_planes);",
    1,
)
src = src.replace(
    "    auto shape_v =\n"
    "        make_shape(kDimHalf, kSparseBlockSize, batch * heads * total_sparse_blocks * kDimHalves);",
    "    auto shape_v = make_shape(kDimHalf, kSparseBlockSize, kv_planes);",
    1,
)

# --- 5. plumb the new fields into Params (aggregate init: field order!) -----
old_ret = "            kv_tma_per_head,\n            args.ptr_block_sizes,"
new_ret = ("            kv_tma_per_head,\n"
           "            args.num_heads_kv,\n"
           "            (args.num_heads_kv > 0 ? heads / args.num_heads_kv : 1),\n"
           "            args.ptr_block_sizes,")
assert old_ret in src, "return-init anchor not found"
src = src.replace(old_ret, new_ret, 1)

ML.write_text(src)
print("mainloop patched:", ML)
print("kv_planes present:", "kv_planes" in src)
