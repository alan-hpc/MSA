"""FP4 block-score indexer vs the fp8 FMHA-maxscore proxy, same index shape.

Both produce [Hq, ceil(kv/128), total_q] float32 block scores that feed
sparse_topk_select, so this is a like-for-like swap of stage 1.
"""
import sys, math, os, traceback
sys.path.insert(0, "python")
import numpy as np, torch

from fmha_sm100 import fmha_sm100, fmha_sm100_plan, fp4_indexer_block_scores
from fmha_sm100.bench_utils import bench_gpu_time
from fp4_indexer_interface import (normalize_fp4_format,
                                   fp4_indexer_reorder_scales_for_mma_cute)

IDX_HQ = int(os.environ.get("IDX_HQ", "4"))
IDX_HKV = int(os.environ.get("IDX_HKV", "1"))
D, PAGE, DEV = 128, 128, "cuda"
FMT = os.environ.get("FP4_FORMAT", "nvfp4")
SEQS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "32768").split(",")]

fmt = normalize_fp4_format(FMT)
G = fmt.scale_groups
SDT = fmt.torch_scale_dtype
print(f"# fp4_format={FMT} scale_groups={G} scale_dtype={SDT} | "
      f"index h_q={IDX_HQ}/h_kv={IDX_HKV} d={D} page={PAGE}", flush=True)


def run_fp4(S):
    pages = (S + PAGE - 1) // PAGE
    q = torch.randint(0, 255, (S, IDX_HQ, D // 2), device=DEV, dtype=torch.uint8)
    k = torch.randint(0, 255, (pages, IDX_HKV, PAGE, D // 2), device=DEV, dtype=torch.uint8)
    qs = torch.randint(120, 130, (S, IDX_HQ, G), device=DEV).to(SDT)
    ks = torch.randint(120, 130, (pages, IDX_HKV, PAGE, G), device=DEV).to(SDT)
    cu_q = torch.tensor([0, S], device=DEV, dtype=torch.int32)
    cu_k = torch.tensor([0, S], device=DEV, dtype=torch.int32)
    cu_p = torch.tensor([0, pages], device=DEV, dtype=torch.int32)
    kv_idx = torch.arange(pages, device=DEV, dtype=torch.int32)
    qo_off = torch.zeros(1, device=DEV, dtype=torch.int32)
    layout = os.environ.get("SCALE_LAYOUT", "preordered_mma")
    if layout == "preordered_mma":
        # hoisted: a real pipeline reorders scales once, not per call
        qs, ks = fp4_indexer_reorder_scales_for_mma_cute(qs, ks, fp4_format=FMT)

    def fn():
        return fp4_indexer_block_scores(
            q, k, qs, ks, cu_q, cu_k, cu_p,
            max_seqlen_q=S, max_seqlen_k=S, kv_indices=kv_idx,
            fp4_format=FMT, causal=True, qo_offset=qo_off,
            scale_layout=layout)

    out = fn(); torch.cuda.synchronize()
    ms = float(np.median(bench_gpu_time(fn, dry_run_time_ms=100, repeat_time_ms=400)))
    return ms, tuple(out.shape)


def run_fp8_proxy(S):
    """The current stage 1: FMHA maxscore proxy, query-chunked for the 2**31 cap."""
    dt, init = torch.float8_e4m3fn, torch.half
    pages = (S + PAGE - 1) // PAGE
    max_k_tiles = math.ceil(math.ceil(S / 128) / 128) * 128
    C = max(128, (1 << 30) // (IDX_HQ * max(max_k_tiles, 1)))
    C = min(S, (C // 128) * 128)
    kd = torch.randn(pages, IDX_HKV, PAGE, D, device=DEV, dtype=init).to(dt)
    vd = torch.randn(pages, IDX_HKV, PAGE, D, device=DEV, dtype=init).to(dt)
    kv_idx = torch.arange(pages, device=DEV, dtype=torch.int32)
    kv_lens = torch.tensor([S], dtype=torch.int32)
    total, start = 0.0, 0
    while start < S:
        c = min(C, S - start)
        qd = torch.randn(c, IDX_HQ, D, device=DEV, dtype=init).to(dt)
        plan = fmha_sm100_plan(torch.tensor([c], dtype=torch.int32), kv_lens, IDX_HQ,
                               qo_offset=torch.tensor([start], dtype=torch.int32),
                               causal=True, page_size=PAGE, num_kv_heads=IDX_HKV,
                               output_maxscore=True)
        fn = lambda: fmha_sm100(qd, kd, vd, plan_info=plan, kv_indices=kv_idx,
                                sm_scale=1.0 / math.sqrt(D), output_o=False,
                                output_maxscore=True)
        _, ms_t = fn()
        if ms_t is None:
            return None, None
        total += float(np.median(bench_gpu_time(fn, dry_run_time_ms=100, repeat_time_ms=400)))
        del qd, ms_t
        torch.cuda.empty_cache()
        start += c
    return total, None


print(f"{'seq':>8} | {'fp8 proxy(ms)':>13} | {'fp4 idx(ms)':>11} | {'fp4/fp8':>8} | shape")
print("-" * 72)
for S in SEQS:
    try:
        t8, _ = run_fp8_proxy(S)
        torch.cuda.empty_cache()
        t4, shape = run_fp4(S)
        print(f"{S:>8} | {t8:>13.4f} | {t4:>11.4f} | {t4/t8:>7.3f}x | {shape}", flush=True)
    except Exception as e:
        print(f"{S:>8} | ERR {repr(e)[:110]}", flush=True)
        traceback.print_exc(limit=4)
    torch.cuda.empty_cache()
print("DONE")
