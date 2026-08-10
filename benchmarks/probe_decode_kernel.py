#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Does the paged decode kernel actually require qhead_per_kv == 16?

Every decode number in this repo's sweep came from the *prefill* kernel, on the
strength of a docstring on SparseDecodePagedAttentionWrapper.plan saying the
decode kernel "requires num_qo_heads / num_kv_heads == 16 at run time".  This
model is 8, so the sweep routed around it -- and decode has never beaten dense,
best case 0.39x.

But nothing enforces 16.  plan() validates only head_dim == 128 and that the
head count divides; the CuTe kernel accepts qhead_per_kv in (16, 8, 4, 2, 1).
So the restriction may be documentation that outlived the code, which would
mean every decode measurement so far used the wrong kernel.

That is worth a direct test rather than an inference: build the paged decode
call at qhead_per_kv=8, run it, and check the output against attention computed
in fp32 over the same selected blocks.  Correct output at 8 would mean the
decode path was available the whole time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT / "python" / "fmha_sm100" / "cute"))


def timed_ms(fn, iters=20, warmup=5):
    flush = torch.empty(int(256e6), dtype=torch.int8, device="cuda")
    for _ in range(warmup):
        flush.zero_()
        fn()
    torch.cuda.synchronize()
    s = []
    for _ in range(iters):
        flush.zero_()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        s.append(a.elapsed_time(b))
    s.sort()
    return s[len(s) // 2]


def reference(q_b, k_pages, v_pages, sel_b, page_size, scale, group):
    """fp32 attention over exactly the selected blocks, for one request."""
    hq, d = q_b.shape
    hkv = sel_b.shape[0]
    blocks = sorted({int(x) for x in sel_b.reshape(-1).tolist() if x >= 0})
    if not blocks:
        return torch.zeros((hq, d), device=q_b.device)
    idx = torch.tensor(blocks, device=q_b.device, dtype=torch.long)
    k = k_pages[idx].reshape(-1, hkv, d).to(torch.float32)
    v = v_pages[idx].reshape(-1, hkv, d).to(torch.float32)
    k = k.repeat_interleave(group, dim=1)
    v = v.repeat_interleave(group, dim=1)
    s = torch.einsum("hd,thd->ht", q_b.to(torch.float32), k) * scale
    p = torch.softmax(s, dim=-1)
    return torch.einsum("ht,thd->hd", p, v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--kv-len", type=int, default=32768)
    ap.add_argument("--head-q", type=int, default=32)
    ap.add_argument("--head-kv", type=int, default=4)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--blk", type=int, default=128)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    from fmha_sm100.cute.interface import SparseDecodePagedAttentionWrapper

    # The paged decode kernel takes an FP8 e4m3 KV cache -- that is a real
    # format requirement, not a guard: this path exists to read a quantised
    # cache.  Q is FP8 too.
    dev, dt = torch.device("cuda"), torch.float8_e4m3fn
    B, S, HQ, HKV, D = args.batch, args.kv_len, args.head_q, args.head_kv, args.dim
    group, scale = HQ // HKV, 1.0 / (D ** 0.5)
    page = args.blk
    npage_per_seq = S // page
    total_pages = B * npage_per_seq

    print(f"qhead_per_kv = {group}   (docstring 说必须是 16)")
    print(f"shape: batch={B} q_len=1 kv_len={S} Hq={HQ} Hkv={HKV} D={D} "
          f"topk={args.topk} page={page}\n")

    gen = torch.Generator(device=dev).manual_seed(args.seed)
    q = torch.randn((B, HQ, D), generator=gen, device=dev, dtype=torch.float32).to(dt)
    k = torch.randn((total_pages, HKV, page, D), generator=gen,
                    device=dev, dtype=torch.float32).to(dt)
    v = torch.randn((total_pages, HKV, page, D), generator=gen,
                    device=dev, dtype=torch.float32).to(dt)
    page_table = torch.arange(total_pages, dtype=torch.int32,
                              device=dev).view(B, npage_per_seq)
    seqused = torch.full((B,), S, dtype=torch.int32, device=dev)

    # A plausible selection: the sink block, the most recent blocks, and a
    # scattered middle -- the shape top-k actually produces.
    sel = torch.empty((HKV, B, args.topk), dtype=torch.int32, device=dev)
    for b in range(B):
        for h in range(HKV):
            picks = [0] + torch.randperm(npage_per_seq - 1, generator=torch.Generator().manual_seed(
                args.seed + b * 17 + h))[: args.topk - 2].add(1).tolist() + [npage_per_seq - 1]
            sel[h, b] = torch.tensor(sorted(picks[: args.topk]), dtype=torch.int32, device=dev)

    # The sparse path (q2k_indices != None) is a stub, but the *dense* paged
    # decode kernel is implemented -- and a dense kernel handed a page table that
    # lists only the selected blocks reads exactly those blocks.  That is sparse
    # decode, built from what exists.  It needs the selection to be shared across
    # KV heads, since a page table is per request, not per head; head_mode
    # max/sum produce exactly one shared selection, which is what cfg10/cfg12 use.
    shared = sel[0]                                     # [B, topk], head 0
    sel_pt = torch.empty((B, args.topk), dtype=torch.int32, device=dev)
    for b in range(B):
        sel_pt[b] = shared[b] + b * npage_per_seq       # logical -> physical
    sel_seqused = torch.full((B,), args.topk * page, dtype=torch.int32, device=dev)

    # The fp8 decode kernel packs seqlen_q * qhead_per_kv into a 128-row tile and
    # assumes the tile is full, so it wants seqlen_q = 128 / qhead_per_kv = 16.
    # It is built for speculative decode, not single-token decode.  Pad instead of
    # giving up: broadcast the one query across the tile and read the last row,
    # which under causal masking is the row that sees the whole KV run.  The waste
    # is Q-side only, and decode is overwhelmingly KV-bound -- 16 query rows
    # against topk*page*Hkv*D bytes of KV is nothing.
    qtok = 128 // group
    q_pad = q.unsqueeze(1).expand(B, qtok, HQ, D).reshape(B * qtok, HQ, D).contiguous()

    wrapper = SparseDecodePagedAttentionWrapper(blk_kv=page, causal=True)
    try:
        wrapper.plan(page_table=sel_pt, seqused_k=sel_seqused, seqlen_q=qtok,
                     max_seqlen_k=args.topk * page, q2k_indices=None,
                     num_qo_heads=HQ, num_kv_heads=HKV, head_dim=D)
    except Exception as exc:  # noqa: BLE001
        print(f"plan 失败: {type(exc).__name__}: {str(exc).splitlines()[0][:120]}")
        return 1

    try:
        out = wrapper.run(q_pad, k, v, softmax_scale=scale)
        out = (out[0] if isinstance(out, (tuple, list)) else out)
        out = out.reshape(B, qtok, HQ, D)[:, -1]     # last row sees the full run
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        print(f"run 失败: {type(exc).__name__}: {str(exc).splitlines()[0][:120]}")
        return 1

    print("跑通了。正确性（对 fp32 参考，只在选中的块上做 attention）：")
    worst = 1.0
    for b in (0, B // 2, B - 1):
        ref = reference(q[b], k[b * npage_per_seq:(b + 1) * npage_per_seq],
                        v[b * npage_per_seq:(b + 1) * npage_per_seq],
                        sel[:, b] % npage_per_seq, page, scale, group)
        c = torch.nn.functional.cosine_similarity(out[b].to(torch.float32), ref, dim=-1)
        worst = min(worst, float(c.min()))
        print(f"   请求 {b:>3}: cos 均值={float(c.mean()):.6f} 最差={float(c.min()):.6f}")

    ms = timed_ms(lambda: wrapper.run(q_pad, k, v, softmax_scale=scale))
    sel_bytes = B * args.topk * page * HKV * D * 2 * 2
    all_bytes = B * S * HKV * D * 2 * 2
    print(f"\n延迟: {ms:.4f} ms")
    print(f"   只读选中的 KV = {sel_bytes / 2**20:.1f} MiB "
          f"(全量 {all_bytes / 2**30:.2f} GiB 的 {sel_bytes / all_bytes:.2%})")
    print(f"   等效带宽 {sel_bytes / (ms * 1e-3) / 1e12:.2f} TB/s")
    print("\n判定:", "输出正确 —— decode 内核在 qhead_per_kv=8 下可用"
          if worst > 0.99 else "*** 输出不对，8 确实不被支持 ***")
    return 0 if worst > 0.99 else 1


if __name__ == "__main__":
    raise SystemExit(main())
