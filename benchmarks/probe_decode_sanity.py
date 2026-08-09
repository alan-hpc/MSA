#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Is the decode dense latency real, or is the kernel reading less than it should?

A decode step at q_len=1 is pure streaming: every kernel has to read the whole
K and V cache once, so its floor is the memory system, not the math.  That makes
the reported latency easy to sanity-check and easy to get wrong in one specific
way -- if a varlen causal kernel aligns the single query to the *start* of the
KV run instead of the end, it attends to one key instead of all of them, does
almost no work, and looks impressively fast.

So this checks three things at the same shape:

  1. what the memory system actually delivers, measured, not assumed;
  2. how long dense attention takes, and what fraction of that floor it is;
  3. whether the output is right -- full causal attention over the whole KV run,
     compared against fp32 -- because (2) only means something if the kernel
     read what it was supposed to.
"""

from __future__ import annotations

import argparse
import sys
import types

import torch


def load_fa4(path):
    if path:
        sys.path.insert(0, path)
    sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))
    from flash_attn.cute.interface import flash_attn_varlen_func
    return flash_attn_varlen_func


def timed_ms(fn, iters=50, warmup=10):
    """Median event-timed latency, L2 flushed outside the timed window."""
    flush = torch.empty(int(256e6), dtype=torch.int8, device="cuda")
    for _ in range(warmup):
        flush.zero_()
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        flush.zero_()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e))
    samples.sort()
    return samples[len(samples) // 2]


def measured_bandwidth(nbytes):
    """Achievable **read-only** bandwidth for a buffer of this size, in TB/s.

    Read-only is the right comparison here and a copy is not: a copy moves the
    data twice and reaches a lower ceiling, so dividing a read-only kernel by a
    copy-derived floor produces the nonsense of "faster than the floor".  A
    reduction reads every byte once and writes nothing, which is exactly the
    traffic pattern of a decode attention's KV stream.
    """
    buf = torch.empty(nbytes // 2, dtype=torch.bfloat16, device="cuda")
    ms = timed_ms(lambda: torch.sum(buf))
    return nbytes / (ms * 1e-3) / 1e12, ms


def reference_row(q_row, k_seq, v_seq, scale, group):
    """Exact causal attention for one decode query over its whole KV run."""
    head_q, dim = q_row.shape
    m = torch.full((head_q,), float("-inf"), device=q_row.device)
    l = torch.zeros((head_q,), device=q_row.device)
    acc = torch.zeros((head_q, dim), device=q_row.device)
    chunk = 32768
    for start in range(0, k_seq.shape[0], chunk):
        kc = k_seq[start:start + chunk].float().repeat_interleave(group, dim=1)
        vc = v_seq[start:start + chunk].float().repeat_interleave(group, dim=1)
        s = torch.einsum("hd,chd->hc", q_row.float(), kc) * scale
        m_new = torch.maximum(m, s.max(dim=-1).values)
        corr = torch.exp(m - m_new)
        p = torch.exp(s - m_new[:, None])
        acc = acc * corr[:, None] + torch.einsum("hc,chd->hd", p, vc)
        l = l * corr + p.sum(dim=-1)
        m = m_new
    return acc / l[:, None]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fa4-path", default="")
    ap.add_argument("--kv-len", type=int, default=32768)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--head-q", type=int, default=32)
    ap.add_argument("--head-kv", type=int, default=4)
    ap.add_argument("--dim", type=int, default=128)
    args = ap.parse_args()

    dev, dt = torch.device("cuda"), torch.bfloat16
    B, S, HQ, HKV, D = args.batch, args.kv_len, args.head_q, args.head_kv, args.dim
    group, scale = HQ // HKV, 1.0 / (D ** 0.5)
    kv_bytes = B * S * HKV * D * 2 * 2

    print(f"shape      : batch={B} q_len=1 kv_len={S} Hq={HQ} Hkv={HKV} D={D}")
    print(f"KV to read : {kv_bytes / 2**30:.3f} GiB  ({kv_bytes} bytes, K and V)")

    tbs, copy_ms = measured_bandwidth(kv_bytes)
    print(f"\nmeasured read bandwidth on this GPU: {tbs:.2f} TB/s "
          f"(reading {kv_bytes / 2**30:.2f} GiB took {copy_ms:.3f} ms)")
    print("   (a reduction is a loose reference, not a hard floor -- it does not\n"
          "    saturate HBM, so a well-tuned streaming kernel can and does beat it)")

    gen = torch.Generator(device=dev).manual_seed(1234)
    q = torch.randn((B, HQ, D), generator=gen, device=dev, dtype=torch.float32).to(dt)
    k = torch.randn((B * S, HKV, D), generator=gen, device=dev, dtype=torch.float32).to(dt)
    v = torch.randn((B * S, HKV, D), generator=gen, device=dev, dtype=torch.float32).to(dt)
    cu_q = torch.arange(0, B + 1, dtype=torch.int32, device=dev)
    cu_k = torch.arange(0, (B + 1) * S, S, dtype=torch.int32, device=dev)

    fa4 = load_fa4(args.fa4_path)

    def run():
        return fa4(q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                   max_seqlen_q=1, max_seqlen_k=S, causal=True)

    out = run()
    out = (out[0] if isinstance(out, (tuple, list)) else out).reshape(B, HQ, D)
    torch.cuda.synchronize()
    ms = timed_ms(run)
    achieved = kv_bytes / (ms * 1e-3) / 1e12
    print(f"\nFA4 decode : {ms:.3f} ms  -> {achieved:.2f} TB/s of KV read")
    print(f"   vs the reduction reference : {achieved / tbs:.2f}x")
    print( "   The op is pure streaming at q_len=1, so this number is the whole story:")
    print( "   there is no arithmetic left to optimise, only bytes.")

    # The check that matters: did it actually attend to the whole run?  If a
    # varlen causal kernel left-aligned the single query it would see one key,
    # be very fast, and be wrong -- which is indistinguishable from "fast"
    # unless the output is compared.
    print("\ncorrectness (this is what makes the latency meaningful):")
    for b in (0, B // 2, B - 1):
        ref = reference_row(q[b], k[b * S:(b + 1) * S], v[b * S:(b + 1) * S], scale, group)
        cos = torch.nn.functional.cosine_similarity(out[b].float(), ref, dim=-1)
        # If the kernel had attended to only the first key, the output would be
        # v[0] broadcast; report that distance too so the failure mode is named.
        v0 = v[b * S].float().repeat_interleave(group, dim=0)
        cos_v0 = torch.nn.functional.cosine_similarity(out[b].float(), v0, dim=-1)
        print(f"  seq {b:>3}: cos vs full-KV attention = {cos.mean():.6f}   "
              f"cos vs 'only first key' = {cos_v0.mean():+.4f}")
    print("\nVERDICT:", "output matches full causal attention over the whole KV run — "
          "the latency is real" if cos.mean() > 0.99 else
          "*** output does NOT match; the kernel is not reading the whole KV ***")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
