#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Is the top-k kernel's K < 12288 limit structural, or just inherited?

`SparseTopKSelect` refuses `max_k_tiles >= 12288` with a comment saying the
radix-sort path is not implemented.  That bound is what stops block=32 above
256K and block=64 above 512K, so it is worth knowing whether it is real.

Reading the kernel suggests it may not be: shared memory is a fixed 16 KB union
(2048 staging items / 1024 histogram bins) that does not scale with K, the row
is streamed from global memory with a size_t base offset, and the four
refinement stages (10+11+11+10 bits) end in a direct fill once every remaining
tie shares one 32-bit pattern.  Nothing there is obviously bounded by K.

"Suggests" is not "is", so this measures it: raise the guard, run past it, and
check every selected index against torch.topk on the same scores.  Set equality
is the right comparison because the kernel emits ascending-by-index, not by
score, and ties may be broken differently.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))


def reference_topk(scores, topk, num_valid):
    """torch.topk over the same [heads, K, total_q] score tensor.

    Returns a [heads, total_q, topk] int64 tensor of selected block ids.
    """
    heads, K, total_q = scores.shape
    s = scores[:, :num_valid, :].permute(0, 2, 1).contiguous()   # [H, Q, K]
    return s.topk(topk, dim=-1).indices


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-tiles", type=lambda s: [int(x) for x in s.split(",")],
                    default=[8192, 12288, 16384, 32768])
    ap.add_argument("--total-q", type=int, default=2048)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from fmha_sm100 import sparse_topk_select

    dev = torch.device("cuda")
    print(f"total_q={args.total_q} heads={args.heads} topk={args.topk}")
    print(f"{'max_k_tiles':>12} {'状态':>10} {'索引集合一致':>14} {'ms':>9}   备注")
    print("-" * 76)

    for K in args.k_tiles:
        gen = torch.Generator(device=dev).manual_seed(args.seed)
        # Distinct values keep the reference unambiguous: with ties, kernel and
        # torch may legitimately pick different members of the tie set.
        scores = torch.randn((args.heads, K, args.total_q), generator=gen,
                             device=dev, dtype=torch.float32)

        try:
            out = sparse_topk_select(
                scores, topk=args.topk, num_valid_pages=K,
                force_begin_blocks=0, force_end_blocks=0,
            )
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            print(f"{K:>12} {'拒绝':>10} {'-':>14} {'-':>9}   "
                  f"{type(exc).__name__}: {str(exc).splitlines()[0][:44]}")
            continue

        ref = reference_topk(scores, args.topk, K)                 # [H, Q, topk]
        got = out.reshape(ref.shape).to(torch.int64)

        same = torch.equal(got.sort(dim=-1).values, ref.sort(dim=-1).values)
        # Where they differ, is it only a tie, or a real miss?  Compare the
        # selected *scores*: a correct selector always picks the same multiset
        # of values even when it picks different indices among equals.
        sc = scores.permute(0, 2, 1)
        got_v = sc.gather(-1, got.clamp_min(0)).sort(dim=-1).values
        ref_v = sc.gather(-1, ref).sort(dim=-1).values
        vals_same = torch.allclose(got_v, ref_v)

        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        for _ in range(3):
            sparse_topk_select(scores, topk=args.topk, num_valid_pages=K,
                               force_begin_blocks=0, force_end_blocks=0)
        torch.cuda.synchronize()
        start.record()
        for _ in range(10):
            sparse_topk_select(scores, topk=args.topk, num_valid_pages=K,
                               force_begin_blocks=0, force_end_blocks=0)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / 10

        verdict = "一致" if same else ("并列不同但取值相同" if vals_same else "*** 选错了 ***")
        print(f"{K:>12} {'跑通':>10} {verdict:>14} {ms:>9.3f}   "
              + ("超过原上限" if K >= 12288 else ""))
        del scores, out, ref, got
        torch.cuda.empty_cache()

    print()
    print("强制窗口与逐 query 因果上界的组合，由 tests/smoke/test_sparse_topk_forced.py")
    print("覆盖（含一个 K=16384 的用例，位于原上限之上）——那里有内核实际的 padding 约定，")
    print("在这里另写一份只会得到一个自造的、与约定不符的期望。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
