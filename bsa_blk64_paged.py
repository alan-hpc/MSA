#!/usr/bin/env python3
"""Paged-KV + GQA blk64 Q-outer attention on SM100, via the patched FlashInfer kernel.

Requires the patches in third_party/flashinfer_patches applied to a flashinfer
checkout. The kernel indexes K/V by (physical page, KV head):

    tma_idx = (page * H_kv + head // qhead_per_kv) * 2 + dim_half

so the page pool is laid out ``[(page * H_kv + kv_head) * 2 + half, 64, 64]`` and
``q2k_block_index`` carries physical page ids. Nothing is rebuilt per call.

Page size is fixed at 64 (kSparseBlockSize), head_dim at 128, dtype bf16.

NOT causal: the kernel attends every token of every selected page. Selecting only
pages strictly before the query's own page keeps it causal *between* pages, but
the diagonal page still leaks future tokens. See the repo notes.
"""
from __future__ import annotations

import torch

PAGE = 64
HEAD_DIM = 128
_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        from flashinfer.cute_dsl.sparse.sm100_blk64 import load_blk64_ext
        _EXT = load_blk64_ext()
    return _EXT


def build_page_pool(k_paged: torch.Tensor, v_paged: torch.Tensor):
    """Convert a paged KV cache into the kernel's page-pool layout.

    k_paged/v_paged: ``[num_pages, H_kv, 64, 128]`` bf16.
    Returns (k_pool, v_pool), each ``[num_pages * H_kv * 2, 64, 64]``.

    K keeps (seq, dim_half); V is transposed to (dim_half, seq), matching what
    the kernel's TMA expects. Build this once per layer, not per call.
    """
    np_, hkv, ps, d = k_paged.shape
    assert ps == PAGE and d == HEAD_DIM, f"expect [*, H_kv, {PAGE}, {HEAD_DIM}]"
    k_pool = (k_paged.view(np_, hkv, PAGE, 2, 64)
                     .permute(0, 1, 3, 2, 4)          # page, kv_head, half, seq, dim
                     .reshape(np_ * hkv * 2, PAGE, 64).contiguous())
    v_pool = (v_paged.view(np_, hkv, PAGE, 2, 64)
                     .permute(0, 1, 3, 4, 2)          # page, kv_head, half, dim, seq
                     .reshape(np_ * hkv * 2, 64, PAGE).contiguous())
    return k_pool, v_pool


def attn_paged(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    q2k_pages: torch.Tensor,
    *,
    num_heads_kv: int,
    block_sparse_num: int,
    block_nums: torch.Tensor | None = None,
    block_sizes: torch.Tensor | None = None,
    softmax_scale: float | None = None,
):
    """q: [B, S, H_q, 128] bf16; q2k_pages: [B, H_q, S//64, topk] int32 physical pages."""
    assert q.dtype == torch.bfloat16 and q.shape[-1] == HEAD_DIM
    if softmax_scale is None:
        softmax_scale = HEAD_DIM ** -0.5
    empty = torch.Tensor()
    if block_nums is not None and block_sizes is None:
        # Variable block counts make the kernel pad each row to a multiple of 8 and
        # fill the phantom slots with the last real page. Those slots are only
        # masked out when a block_sizes tensor is present, so hand it a full-size
        # one (every page holds PAGE tokens) exactly as the stock wrapper does.
        num_pages = k_pool.shape[0] // (num_heads_kv * 2)
        block_sizes = torch.full((num_pages,), PAGE, dtype=torch.int32,
                                 device=q.device)
    out = _ext().bsa_fused_fwd_blk64(
        q.contiguous(), k_pool, v_pool, q2k_pages.contiguous(), int(block_sparse_num),
        block_sizes if block_sizes is not None else empty,
        float(softmax_scale),
        block_nums if block_nums is not None else empty,
        int(num_heads_kv),
    )
    return out[0] if isinstance(out, (list, tuple)) else out
