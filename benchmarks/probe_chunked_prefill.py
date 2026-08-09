#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Can 1M prefill be reached by chunking the query dimension?

The block-score tensor is ``[Hkv, ceil(S/128), total_q]`` and top-k runs
independently per query, so splitting the queries into chunks is *exact* — each
query still sees every KV block it would have seen, and picks the same top-k.
What changes is only how many query columns of the score matrix exist at once,
which is the term that made 1M need 274 GiB on a 268 GiB card.

Chunking does not help the ``S^2`` total work, only the peak footprint: the sum
over chunks still scores every (query, block) pair.  So this measures both, and
reports peak memory next to total time, because a chunk size that fits but
serialises the machine is not a solution either.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlen", type=int, default=1048576)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--chunks", type=lambda s: [int(x) for x in s.split(",")],
                    default=[65536, 131072, 262144])
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    from bench_msa_configs import make_block_scores
    from fmha_sm100.msa_config import MsaSparseConfig
    from fmha_sm100.msa_pipeline import MsaSparseAttention, causal_end_blocks

    dev, dt = torch.device("cuda"), torch.bfloat16
    S, HQ, HKV, D = args.seqlen, 32, 4, 128
    cfg = MsaSparseConfig(block_size=args.block, topk=args.topk,
                          force_init_tokens=128, force_end_tokens=128, head_mode="keep")
    msa = MsaSparseAttention(cfg, num_qo_heads=HQ, num_kv_heads=HKV, head_dim=D, causal=True)

    print(f"seqlen={S} ({S // 1024}K)  block={args.block} topk={args.topk}  Hq={HQ} Hkv={HKV} D={D}")
    unchunked = HKV * ((S + 127) // 128) * S * 4 / 2**30
    print(f"unchunked score tensor: {unchunked:.1f} GiB  (x2 with the transpose "
          f"workspace = {2 * unchunked:.1f} GiB)")
    print(f"device capacity       : "
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.2f} GiB\n")

    gen = torch.Generator(device=dev).manual_seed(args.seed)
    k = torch.randn((S, HKV, D), generator=gen, device=dev, dtype=torch.float32).to(dt)
    v = torch.randn((S, HKV, D), generator=gen, device=dev, dtype=torch.float32).to(dt)

    print(f"{'chunk':>9} {'chunks':>7} {'peak GiB':>9} {'total ms':>9} {'ms/1K tok':>10}  status")
    print("-" * 68)
    for C in args.chunks:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        n_chunks = (S + C - 1) // C
        try:
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            torch.cuda.synchronize()
            start.record()
            for ci in range(n_chunks):
                lo = ci * C
                hi = min(lo + C, S)
                clen = hi - lo
                kv_len = hi                      # causal: this chunk sees keys [0, hi)
                nb = (kv_len + args.block - 1) // args.block

                q = torch.randn((clen, HQ, D), generator=gen, device=dev,
                                dtype=torch.float32).to(dt)
                cu_q = torch.tensor([0, clen], dtype=torch.int32, device=dev)
                cu_k = torch.tensor([0, kv_len], dtype=torch.int32, device=dev)

                # seqlens_q=clen against seqlens_k=hi right-aligns the chunk, so
                # query j of this chunk sits at absolute position lo + j — the
                # same causal extent it would have had unchunked.
                scores = make_block_scores(num_heads=cfg.num_selection_heads(HKV),
                                           num_blocks=nb, total_q=clen, valid_blocks=nb,
                                           device=dev, seed=args.seed + ci)
                ceb = causal_end_blocks([clen], [kv_len], cfg.block_size, device=dev)
                idx = msa.select(scores, num_valid_blocks=nb, causal_end_block=ceb)
                del scores

                csr = msa.build_csr(idx, cu_q, cu_k, total_k=kv_len, max_seqlen_q=clen,
                                    max_seqlen_k=kv_len, total_rows=nb)
                out = msa.attend(q, k[:kv_len], v[:kv_len], csr[0], csr[1],
                                 cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                                 max_seqlen_q=clen, max_seqlen_k=kv_len,
                                 schedule=csr[2] if len(csr) > 2 else None)
                del q, idx, csr, out
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end)
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(f"{C:>9} {n_chunks:>7} {peak:>9.1f} {ms:>9.1f} {ms / (S / 1024):>10.3f}  ok")
        except Exception as exc:  # noqa: BLE001
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(f"{C:>9} {n_chunks:>7} {peak:>9.1f} {'-':>9} {'-':>10}  "
                  f"{type(exc).__name__}: {str(exc).splitlines()[0][:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
