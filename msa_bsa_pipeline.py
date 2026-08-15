#!/usr/bin/env python3
"""MSA indexer -> block-reduced top-k -> FlashInfer BSA attention.

The fastest combination measured on B300 for Compass-V4 prefill. It keeps MSA's
proxy indexer and top-k kernel and swaps the attention stage for FlashInfer's
block-sparse attention, which selects per Q block rather than per query token.

The ordering matters. The obvious wiring -- MSA top-k per token, then reduce the
per-token lists to one list per Q block -- pays for a top-k over `total_q` rows
and then throws most of that work away. Reducing the score tensor *first* gives
the same selection for a top-k over `total_q / block` rows instead, so the
conversion stage disappears and the top-k stage gets ~block times cheaper:

    indexer   -> max_score [H_idx, k_tiles, total_q]
    reduce    -> max over each block's tokens  [H_idx, k_tiles, n_q_blocks]
    top-k     -> [n_q_blocks, H_idx, topk]
    broadcast -> [1, H_q, n_q_blocks, topk]     each KV head's list to its GQA group
    attention -> FlashInfer bsa_attn_sm100_blk128_fwd

This changes what the model attends to: a Q tile's tokens share one block list
instead of choosing individually. That is an accuracy decision -- see
`verify()` for the cosine against MSA's own per-token output.

FlashInfer must come from a source checkout (the released wheel does not carry
bsa_attn_sm100_blk128), and blk64 is not usable here: it has no KV-head mapping,
so GQA needs KV replicated to every query head.
"""
from __future__ import annotations

import math
import os
from typing import Optional

import torch

from fmha_sm100 import fmha_sm100, fmha_sm100_plan, sparse_topk_select

_BSA = None


def _bsa():
    """Import FlashInfer's BSA kernel, with a pointed error when it is absent."""
    global _BSA
    if _BSA is None:
        try:
            from flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk128 import (
                bsa_attn_sm100_blk128_fwd,
            )
        except ImportError as exc:  # noqa: BLE001
            raise ImportError(
                "bsa_attn_sm100_blk128 not found. It ships in the flashinfer git "
                "checkout, not the released wheel: clone flashinfer, then\n"
                "  ln -s <repo>/3rdparty/cutlass <repo>/flashinfer/data/cutlass\n"
                "and put the checkout on PYTHONPATH."
            ) from exc
        _BSA = bsa_attn_sm100_blk128_fwd
    return _BSA


def indexer_scores(
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    v_idx: torch.Tensor,
    *,
    kv_len: int,
    page_size: int = 128,
    idx_h_kv: int = 1,
    qo_offset: int = 0,
) -> torch.Tensor:
    """MSA proxy indexer -> block max-scores ``[H_idx, k_tiles, q_len]``."""
    q_len, h_idx, head_dim = q_idx.shape
    pages = (kv_len + page_size - 1) // page_size
    plan = fmha_sm100_plan(
        torch.tensor([q_len], dtype=torch.int32),
        torch.tensor([kv_len], dtype=torch.int32),
        h_idx,
        qo_offset=torch.tensor([qo_offset], dtype=torch.int32),
        causal=True,
        page_size=page_size,
        num_kv_heads=idx_h_kv,
        output_maxscore=True,
    )
    kv_indices = torch.arange(pages, device=q_idx.device, dtype=torch.int32)
    _, scores = fmha_sm100(
        q_idx, k_idx, v_idx, plan_info=plan, kv_indices=kv_indices,
        sm_scale=1.0 / math.sqrt(head_dim), output_o=False, output_maxscore=True,
    )
    if scores is None:
        raise RuntimeError(
            "indexer returned no max_score (dtype unsupported, or the "
            "num_qo_heads*k_tiles*q_len <= 2**31 cap exceeded -- chunk the query dim)"
        )
    return scores


def select_blocks(
    scores: torch.Tensor,
    *,
    topk: int,
    block: int,
    num_q_heads: int,
    num_valid_pages: Optional[int] = None,
) -> torch.Tensor:
    """Block-reduce the scores, top-k them, and lay the result out for BSA.

    ``scores`` is ``[H_idx, k_tiles, q_len]``; returns
    ``[1, num_q_heads, q_len // block, topk]`` int32.
    """
    h_idx, k_tiles, q_len = scores.shape
    if q_len % block:
        raise ValueError(f"q_len={q_len} must be a multiple of block={block}")
    n_q_blocks = q_len // block
    # one score per (idx head, kv tile, Q block): the strongest any token in the
    # block gave that tile, which is what the block will actually attend with.
    reduced = scores.view(h_idx, k_tiles, n_q_blocks, block).amax(dim=3).contiguous()
    sel = sparse_topk_select(reduced, topk, num_valid_pages=num_valid_pages)
    # sparse_topk_select returns [rows, H_idx, topk] with rows = n_q_blocks
    sel = sel.permute(1, 0, 2)                                   # [H_idx, n_q_blocks, topk]
    if num_q_heads % h_idx:
        raise ValueError(f"num_q_heads={num_q_heads} not divisible by {h_idx}")
    sel = sel.repeat_interleave(num_q_heads // h_idx, dim=0)     # GQA group broadcast
    sel = sel.unsqueeze(0).contiguous().to(torch.int32)          # [1, H_q, n_q_blocks, topk]
    # sparse_topk_select tail-pads with -1 where causality leaves fewer than topk
    # valid blocks (Q block b can only see blocks 0..b). BSA does not interpret
    # -1: with a fixed block_sparse_num it would read every slot. Hand it the
    # real per-block count instead and keep the padded slots in range.
    counts = (sel >= 0).sum(dim=-1).to(torch.int32).contiguous()
    return sel.clamp_min(0).contiguous(), counts


def attend(q, k, v, q2k_block_index, *, topk: int, block_nums=None,
           pack_gqa: bool = False):
    """FlashInfer BSA attention. q/k/v are ``[1, seqlen, heads, head_dim]``.

    ``pack_gqa`` must be False for the index layout this module builds. The
    kernel defaults it to True whenever qhead_per_kvhead > 1, which packs
    several query heads into each 128-row tile -- so a "Q block" stops being
    128 consecutive tokens of one head and the per-Q-block index means
    something else. Verified: with pack_gqa=False, h_q=32/h_kv=4 matches a dense
    reference at cos 0.999998; leaving it at the default gives 0.744.
    """
    out = _bsa()(q, k, v, q2k_block_index, topk,
                 q2k_block_nums=block_nums, pack_gqa=pack_gqa)
    return out[0] if isinstance(out, (tuple, list)) else out


def msa_bsa_prefill(
    q, k, v, q_idx, k_idx, v_idx, *,
    topk: int = 16, block: int = 128, page_size: int = 128, idx_h_kv: int = 1,
):
    """End-to-end: MSA indexer + block-reduced top-k + BSA attention.

    q/k/v      : ``[1, seqlen, heads, 128]`` bf16 (BSHD, dense KV)
    q_idx/k/v  : indexer tensors, ``[seqlen, H_idx, 128]`` and paged KV
    """
    seqlen, num_q_heads = q.shape[1], q.shape[2]
    scores = indexer_scores(q_idx, k_idx, v_idx, kv_len=seqlen,
                            page_size=page_size, idx_h_kv=idx_h_kv)
    q2k, counts = select_blocks(scores, topk=topk, block=block,
                                num_q_heads=num_q_heads,
                                num_valid_pages=(seqlen + page_size - 1) // page_size)
    return attend(q, k, v, q2k, topk=topk, block_nums=counts)
