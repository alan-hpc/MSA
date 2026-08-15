"""FlashInfer BSA (main branch) blk128 / blk64 at Compass-V4, vs MSA's stages.

blk64 has no KV-head mapping, so running Compass-V4 on it needs KV materialised
for all 32 query heads -- 8x the KV bytes. That cost is part of using it here.
"""
import sys, os
import numpy as np, torch

sys.path.insert(0, "/sparse/flashinfer")
from flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk128 import bsa_attn_sm100_blk128_fwd
from flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk64 import bsa_attn_sm100_blk64_fwd

HQ, HKV, D, DEV = 32, 4, 128, "cuda"
TOPK = int(os.environ.get("TOPK", "16"))
SEQS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "32768").split(",")]


def do_bench(fn, warmup=10, rep=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(rep):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    ts.sort(); return ts[len(ts) // 2]


def run(S, blk):
    fwd = bsa_attn_sm100_blk128_fwd if blk == 128 else bsa_attn_sm100_blk64_fwd
    hkv = HKV if blk == 128 else HQ          # blk64: no GQA, replicate KV
    q = torch.randn(1, S, HQ, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(1, S, hkv, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(1, S, hkv, D, device=DEV, dtype=torch.bfloat16)
    nqb = S // blk
    idx = torch.arange(TOPK, device=DEV, dtype=torch.int32).view(1, 1, 1, TOPK)
    idx = idx.expand(1, HQ, nqb, TOPK).contiguous()
    fn = lambda: fwd(q, k, v, idx, TOPK)
    fn(); torch.cuda.synchronize()
    ms = do_bench(fn)
    del q, k, v, idx; torch.cuda.empty_cache()
    return ms


print(f"# FlashInfer BSA main | B300 | bf16 | h_q={HQ}/h_kv={HKV} d={D} topk={TOPK}")
print(f"{'seq':>8} | {'blk128(ms)':>10} | {'blk64*(ms)':>10} | KV/query")
print("-" * 52)
for S in SEQS:
    cells = []
    for blk in (128, 64):
        try:
            cells.append(f"{run(S, blk):>10.4f}")
        except Exception as e:
            cells.append(f"{'ERR':>10}")
            print(f"  blk{blk}: {repr(e)[:100]}", flush=True)
    print(f"{S:>8} | {cells[0]} | {cells[1]} | {min(128*TOPK,S)} vs {min(64*TOPK,S)}", flush=True)
print("* blk64 runs with KV replicated to 32 heads (no GQA support)")
print("DONE")
