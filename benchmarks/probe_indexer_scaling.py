#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Which dimension makes the FP4 indexer super-quadratic?

Measured against a DSA-style FP8 indexer, this one's throughput halves with
every doubling of the context (654 -> 433 -> 251 -> 137 TFLOPS at 32K..256K)
while the FP8 path stays flat.  Work is O(S^2) for both, so something costs an
extra factor of S here — but "S" moves the query count and the key count
together, and those have very different consequences:

  * more keys  -> a longer reduction loop per query tile, and a score tensor
                  whose *stride between k-tiles* is total_q * 4 bytes;
  * more queries -> more tiles, and a wider score tensor.

Holding one fixed while growing the other separates them.  Work is linear in
whichever one moves, so anything above linear is the culprit's dimension.
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


def timed_ms(fn, iters=20, warmup=5):
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


def run_one(total_q, k_len, heads_q, heads_kv, causal, seed=1):
    from bench_msa_configs import make_fp4_indexer_inputs
    from fp4_indexer_interface import fp4_indexer_block_scores

    dev = torch.device("cuda")
    inp = make_fp4_indexer_inputs(total_q=total_q, head_q=heads_q, head_kv=heads_kv,
                                  k_lengths=[k_len], device=dev, seed=seed)
    cu_q = torch.tensor([0, total_q], dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0, k_len], dtype=torch.int32, device=dev)

    def run():
        return fp4_indexer_block_scores(
            inp["q_fp4"], inp["k_fp4"], inp["q_scale"], inp["k_scale"],
            cu_q, cu_k, inp["cu_page_offsets"],
            max_seqlen_q=total_q, max_seqlen_k=k_len,
            kv_indices=inp["kv_indices"], fp4_format="nvfp4", causal=causal)

    out = run()
    torch.cuda.synchronize()
    ms = timed_ms(run)
    del out, inp
    torch.cuda.empty_cache()
    return ms


def sweep(title, points, heads_q, heads_kv, causal, work_of):
    print(f"\n### {title}   (causal={causal})")
    print(f"{'total_q':>9} {'k_len':>9} {'ms':>9} {'x prev':>7} {'work x prev':>12} "
          f"{'TFLOPS':>9}  verdict")
    print("-" * 78)
    prev_ms = prev_w = None
    for total_q, k_len in points:
        try:
            ms = run_one(total_q, k_len, heads_q, heads_kv, causal)
        except Exception as exc:  # noqa: BLE001
            print(f"{total_q:>9} {k_len:>9} {'-':>9}  {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:50]}")
            continue
        w = work_of(total_q, k_len)
        tf = w * heads_q * 2 * 128 / (ms * 1e-3) / 1e12
        if prev_ms is None:
            print(f"{total_q:>9} {k_len:>9} {ms:>9.3f} {'-':>7} {'-':>12} {tf:>9.0f}")
        else:
            rt, rw = ms / prev_ms, w / prev_w
            verdict = ("linear-ish" if rt <= rw * 1.15 else
                       f"SUPER-linear by {rt / rw:.2f}x")
            print(f"{total_q:>9} {k_len:>9} {ms:>9.3f} {rt:>6.2f}x {rw:>11.2f}x "
                  f"{tf:>9.0f}  {verdict}")
        prev_ms, prev_w = ms, w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads-q", type=int, default=4)
    ap.add_argument("--heads-kv", type=int, default=4)
    ap.add_argument("--fixed-q", type=int, default=32768)
    ap.add_argument("--fixed-k", type=int, default=131072)
    args = ap.parse_args()

    grow = [32768, 65536, 131072, 262144]

    # Baseline: both grow together, the shape the sweep actually reports.
    sweep("both grow (q == k), the reported case", [(s, s) for s in grow],
          args.heads_q, args.heads_kv, True,
          lambda q, k: q * k / 2)

    # Keys grow, queries fixed.  Work is linear in k.
    sweep(f"keys grow, queries fixed at {args.fixed_q}",
          [(args.fixed_q, k) for k in grow],
          args.heads_q, args.heads_kv, False,
          lambda q, k: q * k)

    # Queries grow, keys fixed.  Work is linear in q.
    sweep(f"queries grow, keys fixed at {args.fixed_k}",
          [(q, args.fixed_k) for q in grow],
          args.heads_q, args.heads_kv, False,
          lambda q, k: q * k)

    # Same two sweeps with causal on.  With q much shorter than k the query
    # block is right-aligned at the end, so almost every key is visible and the
    # work is still ~q*k -- masking is on, but the *shape* is a slab, not a
    # triangle.  If these stay linear, causal masking per se is not the cost and
    # the triangular schedule is.
    sweep(f"causal, keys grow, queries fixed at {args.fixed_q}",
          [(args.fixed_q, k) for k in grow],
          args.heads_q, args.heads_kv, True,
          lambda q, k: q * k)
    sweep(f"causal, queries grow, keys fixed at {args.fixed_k}",
          [(q, args.fixed_k) for q in grow if q <= args.fixed_k],
          args.heads_q, args.heads_kv, True,
          lambda q, k: q * k - q * q / 2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
