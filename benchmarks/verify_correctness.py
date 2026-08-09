#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Correctness for the query-chunking change, and how far it can be checked.

A first attempt compared the whole pipeline chunked against unchunked and
reported six mismatches with cosines down to -0.33.  A control -- two *identical*
unchunked runs -- reproduced the same mismatch, so the comparison was measuring
something else: top-k hands out its output slots with atomicAdd, so which member
of a tie survives depends on atomic arrival order and is not reproducible.
Comparing through that stage cannot say anything about chunking.

So the claim is checked where it is actually made, and the non-determinism is
measured separately rather than being allowed to masquerade as a bug.

  1. **Chunking is exact.**  Chunking changes only the indexer, so compare the
     indexer's output: the block-score tensors must be bitwise identical.  That
     is a stronger statement than a cosine, and it is the whole claim -- every
     later stage consumes only these scores and was not touched.

  2. **How reproducible is selection?**  Run top-k twice on one fixed score
     tensor, once with near-uniform scores (ties everywhere) and once with
     well-separated ones (ties rare), and report how many rows differ.  This
     bounds what run-to-run variation to expect, and tells whether it is a
     property of the kernel or of a synthetic input.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT / "python" / "fmha_sm100" / "cute"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))


def indexer_scores(model, seqlen, *, chunk_q, device, seed):
    from bench_msa_configs import make_fp4_indexer_inputs
    from fp4_indexer_interface import fp4_indexer_block_scores

    hkv = model.num_kv_heads
    cu_q = torch.tensor([0, seqlen], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, seqlen], dtype=torch.int32, device=device)
    inp = make_fp4_indexer_inputs(total_q=seqlen, head_q=hkv, head_kv=hkv,
                                  k_lengths=[seqlen], device=device, seed=seed)
    out = fp4_indexer_block_scores(
        inp["q_fp4"], inp["k_fp4"], inp["q_scale"], inp["k_scale"],
        cu_q, cu_k, inp["cu_page_offsets"],
        max_seqlen_q=seqlen, max_seqlen_k=seqlen,
        kv_indices=inp["kv_indices"], fp4_format="nvfp4", causal=True,
        q_chunk=chunk_q)
    torch.cuda.synchronize()
    del inp
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlens", type=lambda s: [int(x) for x in s.split(",")],
                    default=[32768, 65536, 131072, 262144])
    ap.add_argument("--model", default="n32")
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    from fmha_sm100 import sparse_topk_select
    from fmha_sm100.msa_config import model_by_name

    model = model_by_name(args.model)
    dev = torch.device("cuda")
    failures = 0

    print("=" * 78)
    print("1) 切块是否精确：indexer 输出的块分数，切块 vs 不切块")
    print("=" * 78)
    print("   切块只改 indexer，下游各级消费的就是这份分数且未被改动，")
    print("   所以逐位相同就是完整的端到端保证，比 cosine 更强。")
    print("   注：比较的是位模式而非 torch.equal —— 分数里含 NaN（随机字节喂给")
    print("   indexer 时，部分 e4m3 缩放位模式即为 NaN），而 NaN != NaN 会让")
    print("   torch.equal 对完全相同的张量返回 False。")
    print(f"\n{'上下文':>8} {'块分数张量':>18} {'逐位相同':>10} {'最大绝对差':>12} {'其中 NaN':>12}")
    print("-" * 78)

    for S in args.seqlens:
        try:
            ref = indexer_scores(model, S, chunk_q=0, device=dev, seed=args.seed)
            got = indexer_scores(model, S, chunk_q=args.chunk, device=dev, seed=args.seed)
            # torch.equal is False whenever a NaN is present, however identical
            # the tensors are, and these scores carry ~1e6 NaNs: the benchmark
            # feeds the indexer random bytes, and some e4m3 scale patterns decode
            # to NaN.  Compare bit patterns, which treats identical NaNs as equal
            # and is the stricter test anyway.
            same = torch.equal(ref.view(torch.int32), got.view(torch.int32))
            fin = torch.isfinite(ref) & torch.isfinite(got)
            dmax = float((ref[fin] - got[fin]).abs().max()) if fin.any() else 0.0
            nan_n = int(torch.isnan(ref).sum())
            failures += 0 if same else 1
            print(f"{S // 1024:>7}K {str(tuple(ref.shape)):>18} "
                  f"{('是' if same else '否'):>10} {dmax:>12.3e} {nan_n:>12}"
                  + ("" if same else "   ← 不一致"))
            del ref, got
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"{S // 1024:>7}K   失败: {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:46]}")
            torch.cuda.empty_cache()

    print()
    print("=" * 78)
    print("2) top-k 的逐次可复现性：同一份分数跑两次，有多少行选择不同")
    print("=" * 78)
    print("   与切块无关，是内核既有性质：输出槽位由 atomicAdd 分配，")
    print("   并列项保留哪个取决于原子到达顺序。分数越接近，影响越大。")
    print(f"\n{'分数分布':>26} {'并列程度':>10} {'不同的行':>14} {'占比':>9}")
    print("-" * 78)

    K, Q, topk = 2048, 4096, 32
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    cases = [
        ("均匀随机（本测试原用的输入）", "极多",
         torch.randn((1, K, Q), generator=gen, device=dev, dtype=torch.float32)),
        ("量化到 8 个档位（并列更极端）", "极端",
         (torch.randn((1, K, Q), generator=gen, device=dev,
                      dtype=torch.float32) * 2).round() / 2),
        ("分数分离（接近真实权重）", "很少",
         torch.randn((1, K, Q), generator=gen, device=dev,
                     dtype=torch.float32) * 8.0
         + torch.arange(K, device=dev, dtype=torch.float32)[None, :, None] * 0.05),
    ]
    for label, tie, sc in cases:
        sc = sc.contiguous()
        a = sparse_topk_select(sc, topk=topk, num_valid_pages=K,
                               force_begin_blocks=0, force_end_blocks=0)
        b = sparse_topk_select(sc, topk=topk, num_valid_pages=K,
                               force_begin_blocks=0, force_end_blocks=0)
        torch.cuda.synchronize()
        a2, b2 = a.reshape(Q, topk).sort(-1).values, b.reshape(Q, topk).sort(-1).values
        diff_rows = int((a2 != b2).any(-1).sum())
        print(f"{label:>26} {tie:>10} {diff_rows:>10} / {Q:<4} {diff_rows / Q:>8.1%}")
        del sc, a, b, a2, b2
        torch.cuda.empty_cache()

    print()
    print("=" * 78)
    if failures:
        print(f"结论：切块有 {failures} 处不一致 —— 性能数据在修好之前不可用。")
        return 1
    print("结论：切块逐位精确，性能数据成立。")
    print("      top-k 的逐次差异是既有性质，与本次改动无关；其影响随分数分离度下降。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
