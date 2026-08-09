#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Does FlashAttention-4 share the repo's Int32 Q/O addressing limit?

The repo's dense FMHA illegal-accesses when ``total_q * num_qo_heads * head_dim``
reaches 2**31 (524288 tokens at Hq=32/D=128).  FA4 is a separate kernel and does
not go through that guard, so the question is open — and "it did not crash" is
not an answer, because a 32-bit offset overflow can just as easily read the
wrong address quietly as fault.

So this checks both: run FA4 at shapes straddling the boundary, and verify a
sample of query rows against attention computed directly in fp32 over the full
K/V.  A kernel that has silently wrapped its addressing will disagree.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import torch


def load_fa4(path: str):
    if path:
        sys.path.insert(0, path)
    sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))
    from flash_attn.cute.interface import flash_attn_varlen_func
    return flash_attn_varlen_func


def reference_rows(q, k, v, rows, *, scale):
    """Exact causal attention for a handful of query rows, in fp32.

    Computed in K-chunks with a running max so it is numerically the same
    online-softmax the kernel does, without materialising a 524288-wide score
    row in fp32 all at once.
    """
    head_q, dim = q.shape[1], q.shape[2]
    group = head_q // k.shape[1]
    out = torch.empty((len(rows), head_q, dim), dtype=torch.float32, device=q.device)
    chunk = 32768
    for i, pos in enumerate(rows):
        qi = q[pos].float()                                   # [Hq, D]
        m = torch.full((head_q,), float("-inf"), device=q.device)
        l = torch.zeros((head_q,), device=q.device)
        acc = torch.zeros((head_q, dim), device=q.device)
        for start in range(0, pos + 1, chunk):
            end = min(start + chunk, pos + 1)
            kc = k[start:end].float()                         # [C, Hkv, D]
            vc = v[start:end].float()
            kc = kc.repeat_interleave(group, dim=1)           # [C, Hq, D]
            vc = vc.repeat_interleave(group, dim=1)
            s = torch.einsum("hd,chd->hc", qi, kc) * scale    # [Hq, C]
            m_new = torch.maximum(m, s.max(dim=-1).values)
            corr = torch.exp(m - m_new)
            p = torch.exp(s - m_new[:, None])
            acc = acc * corr[:, None] + torch.einsum("hc,chd->hd", p, vc)
            l = l * corr + p.sum(dim=-1)
            m = m_new
        out[i] = acc / l[:, None]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fa4-path", default="")
    ap.add_argument("--seqlens", type=lambda s: [int(x) for x in s.split(",")],
                    default=[458752, 524288])
    ap.add_argument("--head-q", type=int, default=32)
    ap.add_argument("--head-kv", type=int, default=4)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    fa4 = load_fa4(args.fa4_path)
    device, dtype = torch.device("cuda"), torch.bfloat16
    scale = 1.0 / (args.dim ** 0.5)

    for seqlen in args.seqlens:
        elems = seqlen * args.head_q * args.dim
        print(f"\n=== total_q={seqlen}  Q/O elements = {elems} "
              f"({elems / 2**31:.3f} x 2^31) ===", flush=True)
        try:
            gen = torch.Generator(device=device).manual_seed(args.seed)
            q = torch.randn((seqlen, args.head_q, args.dim), generator=gen,
                            device=device, dtype=torch.float32).to(dtype)
            k = torch.randn((seqlen, args.head_kv, args.dim), generator=gen,
                            device=device, dtype=torch.float32).to(dtype)
            v = torch.randn((seqlen, args.head_kv, args.dim), generator=gen,
                            device=device, dtype=torch.float32).to(dtype)
            cu = torch.tensor([0, seqlen], dtype=torch.int32, device=device)

            out = fa4(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                      max_seqlen_q=seqlen, max_seqlen_k=seqlen, causal=True)
            out = (out[0] if isinstance(out, (tuple, list)) else out)
            out = out.reshape(seqlen, args.head_q, args.dim)
            torch.cuda.synchronize()
            print("  ran without faulting", flush=True)

            # Sample across the whole range: an addressing overflow shows up in
            # the high rows, so weight the sample toward the end.
            g = torch.Generator().manual_seed(args.seed)
            rows = sorted({int(x) for x in
                           (torch.rand(args.samples, generator=g) ** 0.35 * (seqlen - 1)).long()}
                          | {seqlen - 1, seqlen // 2})
            ref = reference_rows(q, k, v, rows, scale=scale)
            got = out[rows].float()
            cos = torch.nn.functional.cosine_similarity(
                got.reshape(-1, args.dim), ref.reshape(-1, args.dim), dim=-1)
            rel = ((got - ref).norm() / ref.norm()).item()
            print(f"  checked {len(rows)} query rows (max index {rows[-1]}): "
                  f"cos_min={cos.min():.6f} cos_mean={cos.mean():.6f} rel_err={rel:.3e}")
            print("  VERDICT:", "OK" if cos.min() > 0.99 else "*** OUTPUT IS WRONG ***")
            del q, k, v, out, ref, got
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {str(exc).splitlines()[0][:160]}")
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
