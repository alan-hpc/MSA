#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Numerical regression for MSA block-sparse attention at block sizes 32/64/128.

MSA's documented support contract pins ``blk_kv=128``, but the sweep this repo
now benchmarks also needs 32- and 64-token blocks.  ``sparse_atten_func`` takes
``blk_kv`` as a parameter and ``SparseAttentionForwardSm100`` derives its MMA
tiler, TMEM layout and split-P barrier from it, so the smaller widths are
*plausible* — this test is what turns that into a fact.

For each block size the test builds a real top-k selection, runs the CSR sparse
forward, and compares against a masked fp32 SDPA reference over exactly the
selected tokens.  A block size that miscompiles, mis-tiles or silently drops
tokens fails here rather than quietly degrading benchmark numbers.

Run directly or under pytest.  Requires an SM100/SM103 GPU.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT / "python" / "fmha_sm100" / "cute"))

from fmha_sm100.msa_config import DEFAULT_MODEL  # noqa: E402
from fmha_sm100.sparse import build_k2q_csr, sparse_atten_func  # noqa: E402

# Test at the geometry MSA is actually being targeted at, so a change to the
# target model is caught here rather than only in the benchmark.
HEAD_KV = DEFAULT_MODEL.num_kv_heads
QHEAD_PER_KV = DEFAULT_MODEL.qhead_per_kv
HEAD_DIM = DEFAULT_MODEL.bench_head_dim()


def build_selection(*, seqlen_q, seqlen_k, head_kv, blk_kv, topk, device, seed):
    """A causal top-k block selection, shaped ``[Hkv, Sq, topk]`` like production.

    Uses the same "score -> top-k -> ascending, -1 padded" contract the real
    indexer + ``sparse_topk_select`` pair produces, so the CSR builder and the
    kernel see a realistic (ragged, per-query) access pattern rather than a
    uniform prefix.
    """
    num_blocks = (seqlen_k + blk_kv - 1) // blk_kv
    gen = torch.Generator(device=device).manual_seed(seed)
    scores = torch.rand((head_kv, seqlen_q, num_blocks), generator=gen, device=device)

    # Causal bound: query q sees KV positions [0, q + seqlen_k - seqlen_q].
    q_pos = torch.arange(seqlen_q, device=device) + (seqlen_k - seqlen_q)
    last_block = (q_pos // blk_kv).clamp(min=0)                       # [Sq]
    block_idx = torch.arange(num_blocks, device=device)               # [nb]
    visible = block_idx.view(1, -1) <= last_block.view(-1, 1)         # [Sq, nb]

    scores = scores.masked_fill(~visible.unsqueeze(0), float("-inf"))
    idx = scores.topk(min(topk, num_blocks), dim=-1).indices.to(torch.int32)

    # Queries with fewer than topk visible blocks get -1 padding.
    n_visible = visible.sum(dim=-1).to(torch.int32)                   # [Sq]
    rank = torch.arange(idx.shape[-1], device=device, dtype=torch.int32)
    idx = torch.where(rank.view(1, 1, -1) < n_visible.view(1, -1, 1), idx, torch.full_like(idx, -1))

    if idx.shape[-1] < topk:
        pad = torch.full((head_kv, seqlen_q, topk - idx.shape[-1]), -1,
                         dtype=torch.int32, device=device)
        idx = torch.cat([idx, pad], dim=-1)

    sort_key = torch.where(idx < 0, torch.full_like(idx, 2**31 - 1), idx)
    idx = idx.gather(-1, sort_key.argsort(dim=-1, stable=True))
    return idx.contiguous(), num_blocks


def reference_attention(q, k, v, q2k_indices, *, blk_kv, seqlen_q, seqlen_k, softmax_scale):
    """fp32 SDPA restricted to the selected blocks, intersected with causality."""
    seqlen_q_, head_q, dim = q.shape
    head_kv = k.shape[1]
    qhead_per_kv = head_q // head_kv
    num_blocks = (seqlen_k + blk_kv - 1) // blk_kv

    # [Hkv, Sq, nb] block mask -> [Hkv, Sq, Skv] token mask.
    block_mask = torch.zeros((head_kv, seqlen_q, num_blocks), dtype=torch.bool, device=q.device)
    valid = q2k_indices >= 0
    safe = q2k_indices.clamp(min=0).to(torch.int64)
    block_mask.scatter_(2, safe, valid)

    token_mask = block_mask.repeat_interleave(blk_kv, dim=2)[:, :, :seqlen_k]
    q_pos = torch.arange(seqlen_q, device=q.device) + (seqlen_k - seqlen_q)
    causal = torch.arange(seqlen_k, device=q.device).view(1, -1) <= q_pos.view(-1, 1)
    token_mask = token_mask & causal.unsqueeze(0)

    qf = q.to(torch.float32).permute(1, 0, 2)                                  # [Hq, Sq, D]
    kf = k.to(torch.float32).permute(1, 0, 2).repeat_interleave(qhead_per_kv, 0)
    vf = v.to(torch.float32).permute(1, 0, 2).repeat_interleave(qhead_per_kv, 0)
    mask = token_mask.repeat_interleave(qhead_per_kv, 0)                       # [Hq, Sq, Skv]

    logits = torch.bmm(qf, kf.transpose(1, 2)) * softmax_scale
    logits = logits.masked_fill(~mask, float("-inf"))
    # A query with no selected block yields an all -inf row; softmax would emit
    # NaN there, so fold those rows to zero the way the kernel's combine does.
    empty = ~mask.any(dim=-1, keepdim=True)
    probs = torch.softmax(logits.masked_fill(empty, 0.0), dim=-1)
    probs = probs.masked_fill(empty, 0.0)
    out = torch.bmm(probs, vf)
    return out.permute(1, 0, 2).contiguous(), empty.squeeze(-1).permute(1, 0)


def check_block_size(
    *, blk_kv, topk, seqlen_q=2048, seqlen_k=2048,
    head_kv=HEAD_KV, qhead_per_kv=QHEAD_PER_KV, dim=HEAD_DIM, seed=0,
):
    device = "cuda"
    dtype = torch.bfloat16
    head_q = head_kv * qhead_per_kv
    softmax_scale = 1.0 / math.sqrt(dim)

    gen = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn((seqlen_q, head_q, dim), generator=gen, device=device, dtype=dtype)
    k = torch.randn((seqlen_k, head_kv, dim), generator=gen, device=device, dtype=dtype)
    v = torch.randn((seqlen_k, head_kv, dim), generator=gen, device=device, dtype=dtype)

    q2k_indices, num_blocks = build_selection(
        seqlen_q=seqlen_q, seqlen_k=seqlen_k, head_kv=head_kv,
        blk_kv=blk_kv, topk=topk, device=device, seed=seed + 1,
    )

    cu_seqlens_q = torch.tensor([0, seqlen_q], device=device, dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, seqlen_k], device=device, dtype=torch.int32)

    k2q_row_ptr, k2q_q_indices, schedule = build_k2q_csr(
        q2k_indices,
        cu_seqlens_q,
        cu_seqlens_k,
        blk_kv,
        total_k=seqlen_k,
        max_seqlen_k=seqlen_k,
        max_seqlen_q=seqlen_q,
        total_rows=num_blocks,
        qhead_per_kv=qhead_per_kv,
        return_schedule=True,
    )

    out = sparse_atten_func(
        q, k, v, k2q_row_ptr, k2q_q_indices, topk,
        blk_kv=blk_kv,
        causal=True,
        softmax_scale=softmax_scale,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=seqlen_q,
        max_seqlen_k=seqlen_k,
        schedule=schedule,
    )

    ref, empty_rows = reference_attention(
        q, k, v, q2k_indices,
        blk_kv=blk_kv, seqlen_q=seqlen_q, seqlen_k=seqlen_k, softmax_scale=softmax_scale,
    )

    got = out.to(torch.float32)
    # Rows with no selected block are unconstrained by the kernel contract.
    # `empty_rows` is already [Sq, Hq] (the reference expands KV heads first).
    keep = ~empty_rows
    diff = (got - ref).abs()[keep]
    scale = ref.abs()[keep].amax().clamp(min=1e-3)
    max_abs = float(diff.amax())
    rel = max_abs / float(scale)
    return {
        "blk_kv": blk_kv,
        "topk": topk,
        "num_blocks": num_blocks,
        "max_abs_err": max_abs,
        "rel_err": rel,
        "empty_rows": int(empty_rows.sum()),
    }


#: bf16 QK/PV accumulate in fp32 but store in bf16, so ~1e-2 relative agreement
#: against an fp32 reference is the expected floor.
REL_TOL = 2e-2


def check_pipeline_end_to_end(
    cfg, *, seqlen=2048, head_kv=HEAD_KV, qhead_per_kv=QHEAD_PER_KV, dim=HEAD_DIM, seed=0
):
    """``MsaSparseAttention.__call__`` must agree with its own selection.

    The selection kernel itself is covered exhaustively by
    ``test_msa_topk_config.py``; what this pins down is the *chaining* — that
    the config's forced windows, the derived per-query causal bounds, the CSR
    build and the attention call all describe the same sparsity pattern.  A
    mis-wired stage (wrong ``blk_kv``, stale ``total_rows``, transposed
    selection) shows up here as a large numerical error rather than as a crash.
    """
    from fmha_sm100.msa_pipeline import MsaSparseAttention  # noqa: PLC0415

    device = "cuda"
    dtype = torch.bfloat16
    head_q = head_kv * qhead_per_kv
    softmax_scale = 1.0 / math.sqrt(dim)
    num_blocks = cfg.num_blocks(seqlen)

    gen = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn((seqlen, head_q, dim), generator=gen, device=device, dtype=dtype)
    k = torch.randn((seqlen, head_kv, dim), generator=gen, device=device, dtype=dtype)
    v = torch.randn((seqlen, head_kv, dim), generator=gen, device=device, dtype=dtype)

    scorer_heads = cfg.num_scorer_heads(head_q, head_kv)
    block_scores = torch.randn(
        (scorer_heads, num_blocks, seqlen), generator=gen, device=device, dtype=torch.float32
    ).contiguous()

    cu_seqlens_q = torch.tensor([0, seqlen], device=device, dtype=torch.int32)
    cu_seqlens_k = cu_seqlens_q.clone()

    msa = MsaSparseAttention(
        cfg, num_qo_heads=head_q, num_kv_heads=head_kv, head_dim=dim, causal=True
    )
    out = msa(
        q, k, v, block_scores,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        seqlens_q=[seqlen], seqlens_k=[seqlen],
        max_seqlen_q=seqlen, max_seqlen_k=seqlen, total_k=seqlen,
        softmax_scale=softmax_scale,
    )

    # Recover the pattern the pipeline actually selected and score it directly.
    from fmha_sm100.msa_pipeline import causal_end_blocks  # noqa: PLC0415

    ceb = causal_end_blocks([seqlen], [seqlen], cfg.block_size, device=q.device)
    q2k = msa.select(block_scores, num_valid_blocks=num_blocks, causal_end_block=ceb)
    ref, empty_rows = reference_attention(
        q, k, v, q2k,
        blk_kv=cfg.block_size, seqlen_q=seqlen, seqlen_k=seqlen, softmax_scale=softmax_scale,
    )

    got = out.to(torch.float32)
    keep = ~empty_rows
    diff = (got - ref).abs()[keep]
    scale = ref.abs()[keep].amax().clamp(min=1e-3)

    # The forced windows must actually be present in the selection.
    sink_ok = True
    if cfg.force_init_blocks:
        sink = torch.arange(cfg.force_init_blocks, device=q.device, dtype=torch.int32)
        sink_ok = bool((q2k[..., : cfg.force_init_blocks] == sink).all())

    return {
        "config": cfg.resolved_name(),
        "rel_err": float(diff.amax()) / float(scale),
        "sink_present": sink_ok,
        "empty_rows": int(empty_rows.sum()),
    }


def cases():
    for blk_kv in (32, 64, 128):
        for topk in (16, 32):
            yield blk_kv, topk


def pipeline_cases():
    from fmha_sm100.msa_config import CONFIG_MATRIX  # noqa: PLC0415

    # One config per (block_size, head_mode) pair keeps the run short while
    # still covering every distinct code path through the pipeline.
    seen = set()
    for cfg in CONFIG_MATRIX:
        key = (cfg.block_size, cfg.head_mode)
        if key not in seen:
            seen.add(key)
            yield cfg


def test_msa_block_size():
    """pytest entry point."""
    assert torch.cuda.is_available(), "CUDA device required"
    for blk_kv, topk in cases():
        stats = check_block_size(blk_kv=blk_kv, topk=topk, seed=blk_kv + topk)
        assert stats["rel_err"] < REL_TOL, stats


def test_msa_pipeline_end_to_end():
    """pytest entry point."""
    assert torch.cuda.is_available(), "CUDA device required"
    for i, cfg in enumerate(pipeline_cases()):
        stats = check_pipeline_end_to_end(cfg, seed=100 + i)
        assert stats["rel_err"] < REL_TOL, stats
        assert stats["sink_present"], stats


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return 0
    torch.cuda.set_device(0)
    print(f"=== MSA sparse attention block-size regression ({torch.cuda.get_device_name(0)}) ===")
    print(f"    geometry: {DEFAULT_MODEL.name}  Hq={HEAD_KV * QHEAD_PER_KV} Hkv={HEAD_KV} "
          f"(GQA {QHEAD_PER_KV}x) D={HEAD_DIM}")
    failures = []
    for blk_kv, topk in cases():
        label = f"blk_kv={blk_kv:3d} topk={topk:2d}"
        try:
            stats = check_block_size(blk_kv=blk_kv, topk=topk, seed=blk_kv + topk)
        except Exception as exc:  # noqa: BLE001 - report every case, fail at the end
            failures.append((label, repr(exc)))
            print(f"  [FAIL] {label}: {type(exc).__name__}: {exc}")
            continue
        ok = stats["rel_err"] < REL_TOL
        status = "PASS" if ok else "FAIL"
        print(
            f"  [{status}] {label}  blocks={stats['num_blocks']:5d} "
            f"rel_err={stats['rel_err']:.3e} max_abs={stats['max_abs_err']:.3e} "
            f"empty_rows={stats['empty_rows']}"
        )
        if not ok:
            failures.append((label, f"rel_err={stats['rel_err']:.3e} >= {REL_TOL}"))

    print("\n--- MsaSparseAttention end-to-end (select -> CSR -> attend) ---")
    for i, cfg in enumerate(pipeline_cases()):
        label = f"pipeline {cfg.slug}"
        try:
            stats = check_pipeline_end_to_end(cfg, seed=100 + i)
        except Exception as exc:  # noqa: BLE001
            failures.append((label, repr(exc)))
            print(f"  [FAIL] {label}: {type(exc).__name__}: {exc}")
            continue
        ok = stats["rel_err"] < REL_TOL and stats["sink_present"]
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}  rel_err={stats['rel_err']:.3e} "
              f"sink={stats['sink_present']} empty_rows={stats['empty_rows']}")
        if not ok:
            failures.append((label, f"rel_err={stats['rel_err']:.3e} sink={stats['sink_present']}"))

    if failures:
        print(f"\n{len(failures)} case(s) FAILED")
        for label, why in failures:
            print(f"  {label}: {why}")
        return 1
    print("\nAll MSA block-size tests PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
