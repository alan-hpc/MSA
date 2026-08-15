"""MSA indexer + MSA top-k -> FlashInfer BSA attention, end to end vs full MSA.

The two do not speak the same selection language, so the hybrid needs a
conversion stage that the all-MSA path does not pay:

  MSA topk out : [total_q, H_kv=4, topk]      per query TOKEN, per KV head
  BSA wants    : [1, H_q=32, n_q_blocks, topk] per query BLOCK, per query head

Reduction used: for each 128-token Q block and KV head, histogram the blocks its
tokens chose and keep the top-`topk` most-chosen, then broadcast each KV head's
list to the 8 query heads in its GQA group. That is the cheapest faithful
reduction; any choice here changes which blocks are attended, which is why this
is an accuracy decision and not only a latency one.
"""
import sys, math, os
sys.path.insert(0, "python")
import numpy as np, torch

from fmha_sm100 import fmha_sm100, fmha_sm100_plan, sparse_topk_select
from flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk128 import bsa_attn_sm100_blk128_fwd
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "benchmod", "benchmarks/bench_sparse_attention_ops.py")
bm = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(bm)

HQ, HKV, D = 32, 4, 128
IDX_HQ, IDX_HKV = 4, 1
PAGE, TOPK, DEV = 128, 16, "cuda"
SEQS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "32768").split(",")]


def bench(fn, warmup=5, rep=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(rep):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    ts.sort(); return ts[len(ts) // 2]


def stage_idx(S):
    """MSA fp8 proxy indexer, query-chunked for the 2**31 maxscore cap."""
    dt, init = torch.float8_e4m3fn, torch.half
    pages = (S + PAGE - 1) // PAGE
    mkt = math.ceil(math.ceil(S / 128) / 128) * 128
    C = min(S, (max(128, (1 << 30) // (IDX_HQ * max(mkt, 1))) // 128) * 128)
    kd = torch.randn(pages, IDX_HKV, PAGE, D, device=DEV, dtype=init).to(dt)
    vd = torch.randn(pages, IDX_HKV, PAGE, D, device=DEV, dtype=init).to(dt)
    kv_idx = torch.arange(pages, device=DEV, dtype=torch.int32)
    kv_lens = torch.tensor([S], dtype=torch.int32)
    total, start, scores = 0.0, 0, None
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
        _, ms = fn()
        total += bench(fn)
        if scores is None:
            scores = ms.contiguous()          # keep one chunk for the topk stage
        del qd
        torch.cuda.empty_cache()
        start += c
    return total, scores, pages


def stage_topk(scores, pages):
    fn = lambda: sparse_topk_select(scores, TOPK, num_valid_pages=pages)
    sel = fn()
    return bench(fn), sel


def stage_convert(sel_full, S, nblocks):
    """[T, H_kv, topk] per-token  ->  [1, H_q, T/128, topk] per-Q-block."""
    nqb = S // PAGE

    def fn():
        v = sel_full.view(nqb, PAGE, HKV, TOPK)                  # group tokens by Q block
        hist = torch.zeros(nqb, HKV, nblocks, device=DEV, dtype=torch.int32)
        idx = v.permute(0, 2, 1, 3).reshape(nqb, HKV, PAGE * TOPK).clamp_min(0)
        hist.scatter_add_(2, idx.long(), torch.ones_like(idx, dtype=torch.int32))
        top = hist.topk(TOPK, dim=2).indices.to(torch.int32)     # [nqb, H_kv, topk]
        out = top.permute(1, 0, 2).repeat_interleave(HQ // HKV, dim=0)
        return out.unsqueeze(0).contiguous()                     # [1, H_q, nqb, topk]

    return bench(fn), fn()


def stage_attn_bsa(S, q2k):
    q = torch.randn(1, S, HQ, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(1, S, HKV, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(1, S, HKV, D, device=DEV, dtype=torch.bfloat16)
    fn = lambda: bsa_attn_sm100_blk128_fwd(q, k, v, q2k, TOPK)
    fn(); torch.cuda.synchronize()
    ms = bench(fn)
    del q, k, v; torch.cuda.empty_cache()
    return ms


print(f"# hybrid vs all-MSA | B300 | h_q={HQ}/h_kv={HKV} d={D} topk={TOPK} | idx fp8, attn bf16")
hdr = (f"{'seq':>8} | {'idx':>7} {'topk':>7} {'convert':>8} {'attn':>8} {'total':>8} | "
       f"{'MSA total':>9} | {'speedup':>7}")
print(hdr); print("-" * len(hdr))
for S in SEQS:
    nblocks = S // PAGE
    t_idx, scores, pages = stage_idx(S)
    t_topk, sel = stage_topk(scores, pages)
    # the real pipeline selects for every token; the chunk above covers part of
    # the sequence, so tile it up to full length for the conversion stage
    reps = (S + sel.shape[0] - 1) // sel.shape[0]
    sel_full = sel.repeat(reps, 1, 1)[:S].contiguous()
    t_conv, q2k = stage_convert(sel_full, S, nblocks)
    t_attn = stage_attn_bsa(S, q2k)
    t_msa_attn = bm.bench_sparse(1, HQ, HKV, S, S, D, "o", True, "bf16",
                                 page_size=PAGE, topk=TOPK)[0]
    hyb = t_idx + t_topk + t_conv + t_attn
    msa = t_idx + t_topk + t_msa_attn
    print(f"{S:>8} | {t_idx:>7.3f} {t_topk:>7.3f} {t_conv:>8.3f} {t_attn:>8.3f} {hyb:>8.3f} | "
          f"{msa:>9.3f} | {msa/hyb:>6.2f}x", flush=True)
    del scores, sel, sel_full, q2k
    torch.cuda.empty_cache()
print("DONE")
