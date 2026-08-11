#!/usr/bin/env python3
"""FA4 (flash-attn cute) dense decode — single-step latency vs KV length.

Aligns with the MSA "Decode 实测" table 口径: **single-step** decode latency at
each KV length N (the cache already holds N tokens; measure ONE attention step).

Shape (default = MiniMax-M3 / MSA decode 口径): B=1, Sq=8, h_q=64/h_kv=4, d=128,
causal. The 8 query positions **share one length-N KV cache** (read once, reused).
Override via env: B, SQ, H_Q, H_KV, HD, HBM_GBS (B300 HBM peak = 8000 GB/s).

dtype = bf16 (2 B/elem). NOTE: the MSA decode table is **fp8** (1 B/elem); decode is
bandwidth-bound, so bf16 latency ≈ 2× the fp8 number — compare *shape/MBU*, not
absolute ms, against the MSA fp8 column.

FA4 is grafted onto the FA2 package by path; fmax is monkeypatched for cutlass-dsl
4.6.0.dev0 (see bench_fa4_prefill.py). Run in the dedicated fa4_venv:
  H_Q=64 H_KV=4 python bench_fa4_decode.py 32768,131072,262144,524288,1048576
"""
import os, sys, importlib.util, warnings
warnings.filterwarnings("ignore")
import torch

FA4_CUTE_INIT = os.environ.get(
    "FA4_CUTE_INIT",
    "/workspace/sparse_atten_meng/DSA/dsa-kernel-refs/flash-attention/flash_attn/cute/__init__.py",
)
HBM_GBS = float(os.environ.get("HBM_GBS", "8000"))  # B300 HBM peak GB/s


def _patch_fmax():
    import flash_attn.cute.utils as u

    def fmax(a, b, c=None, *, loc=None, ip=None):
        return u.Float32(u.nvvm.fmax(
            u.Float32(a).ir_value(loc=loc, ip=ip),
            u.Float32(b).ir_value(loc=loc, ip=ip),
            c=u.Float32(c).ir_value(loc=loc, ip=ip) if c is not None else None,
            loc=loc, ip=ip))
    u.fmax = fmax


def _load_fa4():
    import flash_attn
    spec = importlib.util.spec_from_file_location(
        "flash_attn.cute", FA4_CUTE_INIT,
        submodule_search_locations=[os.path.dirname(FA4_CUTE_INIT)])
    m = importlib.util.module_from_spec(spec)
    sys.modules["flash_attn.cute"] = m
    spec.loader.exec_module(m)
    if os.environ.get("FA4_PATCH_FMAX", "1") == "1":
        _patch_fmax()
    from flash_attn.cute.interface import flash_attn_func
    return flash_attn_func


fa4 = _load_fa4()
DEV, DT = "cuda", torch.bfloat16
B = int(os.environ.get("B", "1"))
SQ = int(os.environ.get("SQ", "8"))       # MSA decode 口径 Sq=8
H_Q = int(os.environ.get("H_Q", "64"))    # MiniMax-M3 / MSA decode shape
H_KV = int(os.environ.get("H_KV", "4"))
HD = int(os.environ.get("HD", "128"))
NS = int(os.environ.get("NUM_SPLITS", "1"))  # >1 = flash-decoding split-KV (fills SMs)
KVS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1
                        else "32768,131072,262144,524288,1048576").split(",")]


# Decode calls are sub-millisecond, so the defaults are much higher than the
# prefill script's. Still worth exposing for quick turnarounds.
FA4_WARMUP = int(os.environ.get("FA4_WARMUP", "50"))
FA4_REP = int(os.environ.get("FA4_REP", "200"))


def do_bench(fn, warmup=FA4_WARMUP, rep=FA4_REP):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rep):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


print(f"# FA4 dense decode | B300 | bf16 | B={B} Sq={SQ} h_q={H_Q}/h_kv={H_KV} d={HD} causal "
      f"| num_splits={NS} | single-step | HBM peak={HBM_GBS:.0f} GB/s", flush=True)
print(f"{'KV len':>9} | {'FA4(ms)':>10} | {'GB/s':>8} | {'MBU':>6}", flush=True)
print("-" * 44, flush=True)

for N in KVS:
    try:
        q = torch.randn(B, SQ, H_Q, HD, dtype=DT, device=DEV)
        k = torch.randn(B, N, H_KV, HD, dtype=DT, device=DEV)
        v = torch.randn(B, N, H_KV, HD, dtype=DT, device=DEV)

        # NUM_SPLITS=0 → auto: sweep candidates, keep the fastest (fills the SMs;
        # optimal split grows with N). Otherwise use the fixed NS.
        cands = [4, 8, 16, 32, 64, 128] if NS == 0 else [NS]
        best_ms, best_ns = float("inf"), cands[0]
        for ns in cands:
            def fn(ns=ns):
                return fa4(q, k, v, causal=True, num_splits=ns)
            _ = fn(); torch.cuda.synchronize()
            ms = do_bench(fn)
            if ms < best_ms:
                best_ms, best_ns = ms, ns
        ms = best_ms
        # decode is bandwidth-bound: bytes read ≈ K+V cache = B*N*H_KV*2*HD*2(bf16)
        kv_bytes = B * N * H_KV * 2 * HD * 2
        gbs = kv_bytes / (ms * 1e-3) / 1e9
        mbu = gbs / HBM_GBS
        tag = f" split={best_ns}" if NS == 0 else ""
        print(f"{N:>9} | {ms:>10.4f} | {gbs:>8.1f} | {mbu:>5.1%}{tag}", flush=True)
        del q, k, v
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"{N:>9} | ERR {repr(e)[:110]}", flush=True)
        torch.cuda.empty_cache()
print("DONE", flush=True)
