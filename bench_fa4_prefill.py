#!/usr/bin/env python3
"""FA4 (flash-attn cute) dense GQA prefill — Blackwell dense SOTA baseline.

Default shape = **Compass-v4** (Sparse-Attention.xlsx 模型总览):
  h_q=32 / h_kv=4, d=128, B=1, bf16, causal.
Override via env: H_Q, H_KV, HD, PEAK_TFLOPS (B300 bf16 peak = 2250).

FA4 replaces FA3 as the dense reference on sm100/Blackwell. It is grafted onto
the FA2 package by path (flash_attn.cute is not shipped in flash_attn 2.7.4).
Run inside msa/ (or anywhere with CUDA):
  H_Q=32 H_KV=4 HD=128 python bench_fa4_prefill.py 32768,131072,262144
"""
import os, sys, importlib.util, warnings
warnings.filterwarnings("ignore")
import torch

FA4_CUTE_INIT = os.environ.get(
    "FA4_CUTE_INIT",
    "/workspace/sparse_atten_meng/DSA/dsa-kernel-refs/flash-attention/flash_attn/cute/__init__.py",
)
PEAK_TFLOPS = float(os.environ.get("PEAK_TFLOPS", "2250"))  # B300 bf16


def _patch_fmax():
    """cutlass-dsl 4.6.0.dev0 reports CUDA_VERSION 12.9 but ships the *new* 2-arg
    nvvm.fmax, so FA4's `fmax()` wrongly takes the old 3-arg branch -> TypeError at
    JIT trace. Force the new-API wrapper here (keeps the shared checkout pristine).
    `fmax_reduce` calls the module-global `fmax`, so replacing utils.fmax suffices."""
    import flash_attn.cute.utils as u

    def fmax(a, b, c=None, *, loc=None, ip=None):
        return u.Float32(u.nvvm.fmax(
            u.Float32(a).ir_value(loc=loc, ip=ip),
            u.Float32(b).ir_value(loc=loc, ip=ip),
            c=u.Float32(c).ir_value(loc=loc, ip=ip) if c is not None else None,
            loc=loc, ip=ip))
    u.fmax = fmax


def _load_fa4():
    import flash_attn  # FA2 provides the flash_attn namespace
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
H_Q = int(os.environ.get("H_Q", "32"))    # Compass-v4 default
H_KV = int(os.environ.get("H_KV", "4"))
HD = int(os.environ.get("HD", "128"))
SEQS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1
                         else "32768,131072,262144").split(",")]


# Long sequences cost seconds per call, so 40 calls per point is minutes of
# wall time. Let the caller trade reps for turnaround.
FA4_WARMUP = int(os.environ.get("FA4_WARMUP", "10"))
FA4_REP = int(os.environ.get("FA4_REP", "30"))


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


print(f"# FA4 dense prefill | B300 | bf16 | B={B} h_q={H_Q}/h_kv={H_KV} d={HD} causal "
      f"| peak={PEAK_TFLOPS:.0f} TFLOPS", flush=True)
print(f"{'seq':>8} | {'FA4(ms)':>10} | {'TFLOP/s':>9} | {'MFU':>6}", flush=True)
print("-" * 44, flush=True)

for S in SEQS:
    try:
        q = torch.randn(B, S, H_Q, HD, dtype=DT, device=DEV)
        k = torch.randn(B, S, H_KV, HD, dtype=DT, device=DEV)
        v = torch.randn(B, S, H_KV, HD, dtype=DT, device=DEV)

        def fn():
            o = fa4(q, k, v, causal=True)
            return o

        _ = fn(); torch.cuda.synchronize()
        ms = do_bench(fn)
        # causal fwd FLOPs = 2*(QK) + 2*(AV), each B*H_Q*HD*S^2/2  ->  2*B*H_Q*HD*S^2
        flops = 2.0 * B * H_Q * HD * S * S
        tflops = flops / (ms * 1e-3) / 1e12
        mfu = tflops / PEAK_TFLOPS
        print(f"{S:>8} | {ms:>10.3f} | {tflops:>9.1f} | {mfu:>5.1%}", flush=True)
        del q, k, v
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"{S:>8} | ERR {repr(e)[:110]}", flush=True)
        torch.cuda.empty_cache()
print("DONE", flush=True)
