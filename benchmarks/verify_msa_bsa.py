"""Correctness + performance for the MSA-indexer / BSA-attention pipeline.

Two separate questions, deliberately kept apart:

1. Does BSA compute the right attention *for the index it is given*?
   Compared against a plain PyTorch reference driven by the same block list.
   This must match to fp-noise; anything else is a bug.

2. How far does the coarser selection move the output away from MSA's own
   per-token selection? Cosine between the two pipelines' outputs. This is NOT
   expected to be 1.0 -- the models attend to different KV -- and it is the
   number that decides whether the speedup is worth taking.
"""
import sys, math, os
sys.path.insert(0, "python")
import numpy as np, torch

from fmha_sm100 import fmha_sm100, fmha_sm100_plan
from fmha_sm100.bench_utils import bench_gpu_time
sys.path.insert(0, ".")
from msa_bsa_pipeline import indexer_scores, select_blocks, attend

HQ, HKV, D = 32, 4, 128
IDX_HQ, IDX_HKV = 4, 1
PAGE = BLOCK = 128
TOPK, DEV = 16, "cuda"


def ref_attend(q, k, v, q2k, topk, causal=False):
    """Dense reference over exactly the selected KV blocks.

    BSA is NON-CAUSAL: its kernel hardcodes is_causal=False and takes no causal
    argument, so it attends every token of every selected block. `causal=True`
    here is only for measuring how far that is from what a causal LM needs.
    """
    _, S, hq, d = q.shape
    hkv = k.shape[2]
    nqb = S // BLOCK
    out = torch.zeros_like(q, dtype=torch.float32)
    scale = 1.0 / math.sqrt(d)
    for h in range(hq):
        kv_h = h // (hq // hkv)          # verified against the kernel
        kh = k[0, :, kv_h, :].float()
        vh = v[0, :, kv_h, :].float()
        for b in range(nqb):
            qs, qe = b * BLOCK, (b + 1) * BLOCK
            n = int(cnts[0, h, b]); blocks = q2k[0, h, b, :n].tolist()
            cols = torch.cat([torch.arange(x * BLOCK, (x + 1) * BLOCK, device=q.device)
                              for x in sorted(set(int(x) for x in blocks if x >= 0))])
            qq = q[0, qs:qe, h, :].float()
            s = (qq @ kh[cols].T) * scale
            pos_q = torch.arange(qs, qe, device=q.device).view(-1, 1)
            if causal:
                s = s.masked_fill(cols.view(1, -1) > pos_q, float("-inf"))
            out[0, qs:qe, h, :] = torch.softmax(s, dim=-1) @ vh[cols]
    return out


def cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten().unsqueeze(0), b.float().flatten().unsqueeze(0)).item()


def msa_attn(q, k_paged, v_paged, kv_idx, bidx, S):
    plan = fmha_sm100_plan(torch.full((1,), S, dtype=torch.int32),
                           torch.full((1,), S, dtype=torch.int32), HQ,
                           qo_offset=torch.zeros(1, dtype=torch.int32), page_size=PAGE,
                           kv_block_num=TOPK, num_kv_heads=HKV)
    out = torch.empty(S, HQ, D, device=DEV, dtype=torch.bfloat16)
    fn = lambda: fmha_sm100(q, k_paged, v_paged, plan_info=plan, kv_indices=kv_idx,
                            out=out, kv_block_indexes=bidx)
    fn(); torch.cuda.synchronize()
    return fn, out


def build(S):
    q = torch.randn(1, S, HQ, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(1, S, HKV, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(1, S, HKV, D, device=DEV, dtype=torch.bfloat16)
    pages = S // PAGE
    dt, init = torch.float8_e4m3fn, torch.half
    qi = torch.randn(S, IDX_HQ, D, device=DEV, dtype=init).to(dt)
    ki = torch.randn(pages, IDX_HKV, PAGE, D, device=DEV, dtype=init).to(dt)
    vi = torch.randn(pages, IDX_HKV, PAGE, D, device=DEV, dtype=init).to(dt)
    return q, k, v, qi, ki, vi, pages


# ---------------------------------------------------------------- correctness
S = int(os.environ.get("VERIFY_SEQ", "2048"))
q, k, v, qi, ki, vi, pages = build(S)
scores = indexer_scores(qi, ki, vi, kv_len=S, page_size=PAGE, idx_h_kv=IDX_HKV)
q2k, cnts = select_blocks(scores, topk=TOPK, block=BLOCK, num_q_heads=HQ, num_valid_pages=pages)
o_bsa = attend(q, k, v, q2k, topk=TOPK, block_nums=cnts)
o_ref = ref_attend(q, k, v, q2k, TOPK, causal=False)
o_ref_c = ref_attend(q, k, v, q2k, TOPK, causal=True)
print(f"[1] BSA vs NON-causal reference, same index (S={S}): cos={cos(o_bsa, o_ref):.6f}")
print(f"[1b] BSA vs CAUSAL reference, same index:            cos={cos(o_bsa, o_ref_c):.6f}"
      "   <- BSA has no causal mode")

# MSA with its own per-token selection over the same scores
from fmha_sm100 import sparse_topk_select
sel_tok = sparse_topk_select(scores.contiguous(), TOPK, num_valid_pages=pages)  # [S,4,16]
k_paged = k[0].view(pages, PAGE, HKV, D).permute(0, 2, 1, 3).contiguous()
v_paged = v[0].view(pages, PAGE, HKV, D).permute(0, 2, 1, 3).contiguous()
stride = (pages + 3) // 4 * 4
kv_idx = torch.zeros(1, stride, dtype=torch.int32, device=DEV)
kv_idx[0, :pages] = torch.arange(pages, device=DEV, dtype=torch.int32)
_, o_msa = msa_attn(q[0], k_paged, v_paged, kv_idx, sel_tok.contiguous(), S)
print(f"[2] BSA(per-block) vs MSA(per-token), same indexer: cos={cos(o_bsa[0], o_msa):.6f}"
      "   <- the accuracy cost of coarser selection")

# ---------------------------------------------------------------- performance
print()
hdr = f"{'seq':>8} | {'idx':>7} {'select':>7} {'attn':>7} {'total':>8} | {'MSA total':>9} | {'speedup':>7}"
print(hdr); print("-" * len(hdr))
for S in [int(x) for x in os.environ.get("SEQS", "32768,65536").split(",")]:
    q, k, v, qi, ki, vi, pages = build(S)
    t_idx = float(np.median(bench_gpu_time(
        lambda: indexer_scores(qi, ki, vi, kv_len=S, page_size=PAGE, idx_h_kv=IDX_HKV),
        dry_run_time_ms=100, repeat_time_ms=400)))
    scores = indexer_scores(qi, ki, vi, kv_len=S, page_size=PAGE, idx_h_kv=IDX_HKV)
    t_sel = float(np.median(bench_gpu_time(
        lambda: select_blocks(scores, topk=TOPK, block=BLOCK, num_q_heads=HQ,
                              num_valid_pages=pages)[0],
        dry_run_time_ms=100, repeat_time_ms=400)))
    q2k, cnts = select_blocks(scores, topk=TOPK, block=BLOCK, num_q_heads=HQ, num_valid_pages=pages)
    t_attn = float(np.median(bench_gpu_time(
        lambda: attend(q, k, v, q2k, topk=TOPK, block_nums=cnts), dry_run_time_ms=100, repeat_time_ms=400)))
    # all-MSA baseline: same indexer, per-token top-k, MSA attention
    sel_tok = sparse_topk_select(scores.contiguous(), TOPK, num_valid_pages=pages)
    t_topk_tok = float(np.median(bench_gpu_time(
        lambda: sparse_topk_select(scores.contiguous(), TOPK, num_valid_pages=pages),
        dry_run_time_ms=100, repeat_time_ms=400)))
    k_paged = k[0].view(pages, PAGE, HKV, D).permute(0, 2, 1, 3).contiguous()
    v_paged = v[0].view(pages, PAGE, HKV, D).permute(0, 2, 1, 3).contiguous()
    stride = (pages + 3) // 4 * 4
    kv_idx = torch.zeros(1, stride, dtype=torch.int32, device=DEV)
    kv_idx[0, :pages] = torch.arange(pages, device=DEV, dtype=torch.int32)
    fn_msa, _ = msa_attn(q[0], k_paged, v_paged, kv_idx, sel_tok.contiguous(), S)
    t_msa_attn = float(np.median(bench_gpu_time(fn_msa, dry_run_time_ms=100, repeat_time_ms=400)))
    hyb = t_idx + t_sel + t_attn
    msa = t_idx + t_topk_tok + t_msa_attn
    print(f"{S:>8} | {t_idx:>7.3f} {t_sel:>7.3f} {t_attn:>7.3f} {hyb:>8.3f} | "
          f"{msa:>9.3f} | {msa/hyb:>6.2f}x", flush=True)
    del q, k, v, qi, ki, vi, scores, q2k, sel_tok, k_paged, v_paged
    torch.cuda.empty_cache()
print("DONE")
