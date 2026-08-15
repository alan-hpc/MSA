"""FlashInfer BSA (Q-outer, per-Q-block selection) at blk128 and blk64.

Selection granularity differs from MSA: q2k_block_index is
(batch, num_heads, num_q_blocks, max_kv_blocks) -- one KV-block list per Q tile
per query head -- where MSA picks per query token. Same KV budget per query is
block x topk, so blk128x16 = 2048 tokens and blk64x16 = 1024.

Causal here is expressed through the index: Q block i is given the first
min(topk, i+1) KV blocks, mirroring the "first top-K" pattern the MSA bench uses.
"""
import os, sys, time
import numpy as np, torch
from flashinfer.cute_dsl.sparse.bsa_attn import bsa_attn_fwd
from flashinfer.cute_dsl.sparse.bsa_attn_blk64 import bsa_attn_blk64_fwd

HQ = int(os.environ.get("H_Q", "32"))
HKV = int(os.environ.get("H_KV", "4"))
D, DEV = 128, "cuda"
TOPK = int(os.environ.get("TOPK", "16"))
SEQS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "32768").split(",")]


def do_bench(fn, warmup=10, rep=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rep):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def run(S, blk, expand_kv=False):
    # blk64 has no GQA support (num_kv_heads must equal num_heads), so running
    # Compass-V4 on it means materialising KV for every query head -- 8x the KV
    # bytes. That cost is part of using this kernel at this shape.
    fwd = bsa_attn_fwd if blk == 128 else bsa_attn_blk64_fwd
    hkv = HQ if expand_kv else HKV
    q = torch.randn(1, S, HQ, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(1, S, hkv, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(1, S, hkv, D, device=DEV, dtype=torch.bfloat16)
    nqb = S // blk
    # Q block i sees the first min(TOPK, i+1) KV blocks; pad the rest by repeating
    # block 0 (a real pipeline would pass q2k_block_nums; this keeps the kernel's
    # work per Q block at exactly TOPK so the cost is the topk budget).
    if os.environ.get("RANDOM_IDX", "0") == "1":
        # each (head, q_block) draws its own causal-valid blocks -- defeats the
        # cache reuse that an identical index across Q blocks would hand the kernel
        g = torch.Generator(device="cpu").manual_seed(0)
        idx = torch.empty(1, HQ, nqb, TOPK, dtype=torch.int32)
        for qb in range(nqb):
            hi = min(qb + 1, nqb)                      # causal: blocks 0..qb
            for h in range(HQ):
                if hi >= TOPK:
                    pick = torch.randperm(hi, generator=g)[:TOPK].sort().values
                else:
                    pick = torch.arange(hi).repeat((TOPK + hi - 1) // hi)[:TOPK]
                idx[0, h, qb] = pick.to(torch.int32)
        idx = idx.to(DEV).contiguous()
    else:
        idx = torch.arange(TOPK, device=DEV, dtype=torch.int32).view(1, 1, 1, TOPK)
        idx = idx.expand(1, HQ, nqb, TOPK).contiguous()
    fn = lambda: fwd(q, k, v, idx, TOPK)
    o = fn(); torch.cuda.synchronize()
    ms = do_bench(fn)
    del q, k, v, idx, o
    torch.cuda.empty_cache()
    return ms


print(f"# FlashInfer BSA (Q-outer) | B300 | bf16 | h_q={HQ}/h_kv={HKV} d={D} topk={TOPK}")
print(f"{'seq':>8} | {'blk128x16(ms)':>13} | {'blk64x16*(ms)':>12} | {'KV budget':>18}")
print("-" * 62)
for S in SEQS:
    cells = {}
    for blk in (128, 64):
        try:
            cells[blk] = f"{run(S, blk):>13.4f}" if blk == 128 else f"{run(S, blk, expand_kv=True):>12.4f}"
        except Exception as e:
            cells[blk] = f"{'ERR':>13}" if blk == 128 else f"{'ERR':>12}"
            print(f"  blk{blk}: {repr(e)[:110]}", flush=True)
    print(f"{S:>8} | {cells[128]} | {cells[64]} | "
          f"{min(128*TOPK,S)} vs {min(64*TOPK,S)}", flush=True)
print("DONE")
