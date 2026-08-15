"""Enumerate what FlashInfer BSA actually supports on this GPU (main branch).

Probes each (variant, dtype, heads, head_dim, topk) at a small sequence and
reports OK / the rejecting error, then times the supported combinations.
"""
import os, sys, itertools, traceback
import numpy as np, torch

sys.path.insert(0, "/sparse/flashinfer")
from flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk128 import bsa_attn_sm100_blk128_fwd
from flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk64 import bsa_attn_sm100_blk64_fwd

DEV = "cuda"
VARIANTS = {"blk128": (bsa_attn_sm100_blk128_fwd, 128),
            "blk64": (bsa_attn_sm100_blk64_fwd, 64)}


def build(S, HQ, HKV, D, blk, topk, dt):
    q = torch.randn(1, S, HQ, D, device=DEV, dtype=dt)
    k = torch.randn(1, S, HKV, D, device=DEV, dtype=dt)
    v = torch.randn(1, S, HKV, D, device=DEV, dtype=dt)
    nqb = S // blk
    idx = torch.arange(topk, device=DEV, dtype=torch.int32).view(1, 1, 1, topk)
    idx = idx.expand(1, HQ, nqb, topk).contiguous()
    return q, k, v, idx


def try_run(name, S, HQ, HKV, D, topk, dt):
    fwd, blk = VARIANTS[name]
    q, k, v, idx = build(S, HQ, HKV, D, blk, topk, dt)
    o = fwd(q, k, v, idx, topk)
    torch.cuda.synchronize()
    del q, k, v, idx, o
    torch.cuda.empty_cache()
    return True


def bench(name, S, HQ, HKV, D, topk, dt, warmup=10, rep=30):
    fwd, blk = VARIANTS[name]
    q, k, v, idx = build(S, HQ, HKV, D, blk, topk, dt)
    fn = lambda: fwd(q, k, v, idx, topk)
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(rep):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    ts.sort()
    del q, k, v, idx
    torch.cuda.empty_cache()
    return ts[len(ts) // 2]


print(f"# device: {torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability()}")
print("\n=== capability probe (S=4096) ===")
print(f"{'variant':>8} {'dtype':>6} {'HQ/HKV':>8} {'D':>4} {'topk':>5} | result")
print("-" * 74)
supported = []
for name, dt_name, (HQ, HKV), D, topk in itertools.product(
        ["blk128", "blk64"], ["bf16", "fp16"],
        [(32, 4), (32, 32), (64, 4), (16, 1), (16, 16)], [128, 64], [16]):
    dt = torch.bfloat16 if dt_name == "bf16" else torch.float16
    tag = f"{name:>8} {dt_name:>6} {HQ:>3}/{HKV:<4} {D:>4} {topk:>5}"
    try:
        try_run(name, 4096, HQ, HKV, D, topk, dt)
        print(f"{tag} | OK", flush=True)
        supported.append((name, dt_name, HQ, HKV, D, topk))
    except Exception as e:
        msg = str(e).strip().split("\n")[0][:60]
        print(f"{tag} | {type(e).__name__}: {msg}", flush=True)
    torch.cuda.empty_cache()

print("\n=== topk sweep on supported combos (S=4096) ===")
for name, dt_name, HQ, HKV, D, _ in supported:
    dt = torch.bfloat16 if dt_name == "bf16" else torch.float16
    row = []
    for topk in (4, 8, 16, 32, 64):
        try:
            try_run(name, 4096, HQ, HKV, D, topk, dt); row.append(f"{topk}:OK")
        except Exception as e:
            row.append(f"{topk}:{type(e).__name__}")
        torch.cuda.empty_cache()
    print(f"{name:>8} {dt_name:>6} {HQ:>3}/{HKV:<4} D={D:<4} | " + "  ".join(row), flush=True)
print("DONE")
