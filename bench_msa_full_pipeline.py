#!/usr/bin/env python3
"""MSA full-pipeline prefill benchmark: dense GQA vs (indexer + topk + sparse attn).

The MSA shipped bench (`benchmarks/bench_sparse_attention_ops.py`) `sparse_prefill`
section measures ONLY the sparse-attention kernel with fixed "first top-K" block
indices — it skips the indexer (proxy max-score pass) and topk selection. That
inflates the speedup because it omits the stages that choose sparse blocks.

This script reuses the proven `bench_dense` / `bench_sparse` for the dense baseline
and the sparse-attn kernel (kernel time is mostly index-value independent), and ADDS timing
for the two missing stages:
  stage1 indexer : cheap MQA proxy pass (h_q=h_k=4, h_kv=1) with output_maxscore
  stage2 topk    : sparse_topk_select(max_score, topk=16)
full sparse latency = indexer + topk + sparse_attn.

This is an approximate engineering pipeline. The MSA repo also provides an FP4
block-score indexer, and production/paper implementations may use chunked,
fused, or lower-precision paths.

Run inside msa/ :  python bench_msa_full_pipeline.py 32768,131072,262144
"""
import sys, math, os, importlib.util
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent / "msa"
if not (REPO / "python").exists():           # allow running from inside msa/
    REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "python"))

from fmha_sm100 import fmha_sm100, fmha_sm100_plan, sparse_topk_select   # noqa: E402
from fmha_sm100.bench_utils import bench_gpu_time                        # noqa: E402

# load the shipped bench module to reuse bench_dense / bench_sparse
_spec = importlib.util.spec_from_file_location(
    "benchmod", str(REPO / "benchmarks" / "bench_sparse_attention_ops.py"))
bm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bm)

# attention shape: default = MiniMax-M3 (64/4/128); Compass-V4 = H_Q=32 H_K=4 D=128
H_Q = int(os.environ.get("H_Q", "64"))
H_K = int(os.environ.get("H_K", "4"))
D = int(os.environ.get("D", "128"))
# index (proxy indexer) shape: default = MQA proxy with h_q=H_K, h_kv=1
IDX_HQ = int(os.environ.get("IDX_HQ", str(H_K)))
IDX_HKV = int(os.environ.get("IDX_HKV", "1"))
# KV block size and top-k budget. Upstream MSA is built for page=128/topk=16;
# smaller blocks need the CSR builder and topk kernel that the
# msa-configurable-sparse-attention branch carries.
PAGE = int(os.environ.get("PAGE", "128"))
TOPK = int(os.environ.get("TOPK", "16"))
# Always-selected blocks inside the top-k budget: FORCE_BEGIN pins the sink
# blocks at the head of the sequence, FORCE_END pins the local window nearest
# the query. They occupy slots *within* topk (the result is always topk wide),
# so they change which blocks are attended, not how many -- latency-neutral.
# Default 0 = pure top-k by score, which is what the published table measured.
FORCE_BEGIN = int(os.environ.get("FORCE_BEGIN", "0"))
FORCE_END = int(os.environ.get("FORCE_END", "0"))
DTYPE = "bf16"
IDX_DTYPE = os.environ.get("IDX_DTYPE", "bf16")   # fp8 -> falls back to bf16 if maxscore unsupported
DEV = "cuda"

# Profiling: shrink the timing windows so a profiler sees a handful of launches
# per stage instead of hundreds. 0 keeps the per-stage defaults.
DRY_MS = int(os.environ.get("DRY_MS", "0"))
REP_MS = int(os.environ.get("REP_MS", "0"))
if DRY_MS:
    bm.DRY_RUN_MS = DRY_MS
if REP_MS:
    bm.REPEAT_MS = REP_MS


def _win(dry, rep):
    """Per-stage timing window, overridden by DRY_MS / REP_MS when set."""
    return (DRY_MS or dry), (REP_MS or rep)


@contextmanager
def nvtx(name):
    """Name a stage so `nsys` can attribute kernels to indexer / topk / attn."""
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()

SEQS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1
                         else "32768,131072,262144").split(",")]


def _time_indexer_and_topk_dt(s, dtype_str):
    """Time the proxy max-score pass (indexer) + sparse_topk_select for one dtype.

    The maxscore output is capped by ``num_qo_heads*max_k_tiles*total_qo_len <= 2**31``
    (api.py: "Too huge setting to output maxscore!" -> returns None). max_k_tiles is
    fixed by kv_len, so we CHUNK the query dimension (like real chunked prefill) and
    sum the per-chunk indexer + topk times. This also bounds the O(N^2) max_score memory.

    Returns (t_idx, t_topk) or (None, None) if maxscore is unsupported for this dtype.
    """
    dt = torch.float8_e4m3fn if dtype_str == "fp8" else torch.bfloat16
    init = torch.half if dt.itemsize == 1 else dt
    kv_len = s
    pages = (kv_len + PAGE - 1) // PAGE
    max_k_tiles = math.ceil(math.ceil(kv_len / 128) / 128) * 128
    # chunk so IDX_HQ*max_k_tiles*C <= 2**30 (half the 2**31 cap, safety margin)
    C = max(128, (1 << 30) // (IDX_HQ * max(max_k_tiles, 1)))
    C = min(s, (C // 128) * 128)

    kd = torch.randn(pages, IDX_HKV, PAGE, D, device=DEV, dtype=init).to(dt)   # proxy KV
    vd = torch.randn(pages, IDX_HKV, PAGE, D, device=DEV, dtype=init).to(dt)
    kv_idx = torch.arange(pages, device=DEV, dtype=torch.int32)
    kv_lens = torch.tensor([kv_len], dtype=torch.int32)

    t_idx_total, t_topk_total, nchunks = 0.0, 0.0, 0
    start = 0
    while start < s:
        c = min(C, s - start)
        qd = torch.randn(c, IDX_HQ, D, device=DEV, dtype=init).to(dt)   # proxy query chunk
        qo_lens = torch.tensor([c], dtype=torch.int32)
        qo_off = torch.tensor([start], dtype=torch.int32)          # chunk global start (causal)
        plan_d = fmha_sm100_plan(qo_lens, kv_lens, IDX_HQ, qo_offset=qo_off, causal=True,
                                 page_size=PAGE, num_kv_heads=IDX_HKV, output_maxscore=True)

        def indexer():
            return fmha_sm100(qd, kd, vd, plan_info=plan_d, kv_indices=kv_idx,
                              sm_scale=1.0 / math.sqrt(D),
                              output_o=False, output_maxscore=True)

        _, max_score = indexer()
        if max_score is None:      # dtype unsupported for maxscore, or 2**31 cap exceeded
            return None, None
        _dry, _rep = _win(100, 400)
        with nvtx("msa_idx"):
            t_idx_total += float(np.median(bench_gpu_time(
                indexer, dry_run_time_ms=_dry, repeat_time_ms=_rep)))

        max_score = max_score.contiguous()

        def topk():
            return sparse_topk_select(max_score, TOPK, num_valid_pages=pages,
                                   force_begin_blocks=FORCE_BEGIN,
                                   force_end_blocks=FORCE_END)

        topk()
        _dry, _rep = _win(50, 300)
        with nvtx("msa_topk"):
            t_topk_total += float(np.median(bench_gpu_time(
                topk, dry_run_time_ms=_dry, repeat_time_ms=_rep)))
        del qd, max_score
        torch.cuda.empty_cache()
        start += c
        nchunks += 1
    return t_idx_total, t_topk_total


def time_dense_chunked(s, chunk):
    """Dense baseline with the query dimension chunked.

    One-shot dense raises cudaErrorIllegalAddress at ``h_q * seq == 2**24`` and
    poisons the CUDA context. Chunking the query dimension keeps each launch
    well under that product (and bounds the workspace), so the point that would
    otherwise be skipped becomes measurable. Causality is preserved per chunk by
    ``qo_offset``, the same way the indexer chunking does it: chunk queries sit
    at global positions ``start .. start+c-1`` and see kv ``0 .. start+i``.

    The sum over chunks is the causal dense cost -- validate it against a
    one-shot measurement at a seq where both run before trusting a new shape.
    """
    dt = torch.bfloat16
    k = torch.randn(s, H_K, D, device=DEV, dtype=dt)
    v = torch.randn(s, H_K, D, device=DEV, dtype=dt)
    kv_lens = torch.tensor([s], dtype=torch.int32)
    total, start = 0.0, 0
    while start < s:
        c = min(chunk, s - start)
        q = torch.randn(c, H_Q, D, device=DEV, dtype=dt)
        out = torch.empty_like(q)
        plan = fmha_sm100_plan(torch.tensor([c], dtype=torch.int32), kv_lens, H_Q,
                               qo_offset=torch.tensor([start], dtype=torch.int32),
                               causal=True, num_kv_heads=H_K)

        def dense(q=q, out=out, plan=plan):
            return fmha_sm100(q, k, v, plan_info=plan, out=out, output_o=True)

        dense()
        _dry, _rep = _win(100, 400)
        with nvtx("msa_dense"):
            total += float(np.median(bench_gpu_time(
                dense, dry_run_time_ms=_dry, repeat_time_ms=_rep)))
        del q, out
        torch.cuda.empty_cache()
        start += c
    return total


def time_indexer_and_topk(s):
    """Run the indexer at IDX_DTYPE, falling back to bf16 when maxscore is unsupported."""
    used = IDX_DTYPE
    t_idx, t_topk = _time_indexer_and_topk_dt(s, IDX_DTYPE)
    if t_idx is None and IDX_DTYPE != "bf16":
        used = "bf16"
        t_idx, t_topk = _time_indexer_and_topk_dt(s, "bf16")
    if t_idx is None:
        raise RuntimeError("maxscore None (indexer dtype unsupported or 2**31 cap exceeded)")
    return t_idx, t_topk, used


hdr = (f"{'seq':>8} | {'dense(ms)':>9} | {'idx(ms)':>8} | {'topk(ms)':>8} | {'attn(ms)':>8} | "
       f"{'sparse_full':>11} | {'full x':>7} | {'attn-only x':>11} | idxdt")
print(f"# MSA full-pipeline prefill | B300 | bf16 | attn h_q={H_Q}/h_kv={H_K} d={D} | "
      f"index h_q={IDX_HQ}/h_kv={IDX_HKV} dt={IDX_DTYPE} | topk={TOPK} page={PAGE} | B=1 causal", flush=True)
print(hdr, flush=True)
print("-" * len(hdr), flush=True)

def _try(fn):
    try:
        return fn(), None
    except Exception as e:
        torch.cuda.empty_cache()
        return None, repr(e)[:80]

# dense@N ~ O(N^2); extrapolate from a measured anchor if the 1M dense kernel fails (CUDA 719).
_dense_anchor = {}  # s -> ms

import os
SKIP_DENSE = os.environ.get("SKIP_DENSE", "0") == "1"   # dense@1M crashes CUDA ctx (719) -> skip it
# DENSE_CHUNK: "auto" (default) chunks the dense query dim only where one-shot
# would hit the h_q*seq == 2**24 crash; "0"/"off" never chunks; N always chunks
# at N. Chunked and one-shot agree to 0.4% where both run (256K @ h_q=32:
# 447.536 vs 445.838), so auto leaves every other point bit-identical to before.
_DC = os.environ.get("DENSE_CHUNK", "auto").lower()
DENSE_CHUNK = 0 if _DC in ("0", "off") else (-1 if _DC in ("auto", "-1") else int(_DC))


def _dense_chunk_for(s):
    if DENSE_CHUNK == 0:
        return 0
    if DENSE_CHUNK > 0:
        return DENSE_CHUNK
    return max(1024, (1 << 23) // H_Q) if H_Q * s >= (1 << 24) else 0
DENSE_MS = float(os.environ.get("DENSE_MS", "0"))       # external extrapolated dense (ms) for full x

for s in SEQS:
    _chunk = _dense_chunk_for(s)
    if SKIP_DENSE:
        r_dense, e_d = None, "skipped"
    elif _chunk:
        r_dense, e_d = _try(lambda: (time_dense_chunked(s, _chunk),))
    else:
        with nvtx("msa_dense"):
            r_dense, e_d = _try(lambda: bm.bench_dense(1, H_Q, H_K, s, s, D, "o", True, DTYPE, use_mbu=False))
    with nvtx("msa_attn"):
        r_sparse, e_s = _try(lambda: bm.bench_sparse(1, H_Q, H_K, s, s, D, "o", True, DTYPE, page_size=PAGE, topk=TOPK))
    r_idx, e_i = _try(lambda: time_indexer_and_topk(s))

    dense_ms = r_dense[0] if r_dense else None
    dense_tag = f"~chunk{_chunk}" if (r_dense and _chunk) else ""
    if dense_ms is not None:
        _dense_anchor[s] = dense_ms
    elif _dense_anchor:                      # extrapolate O(N^2) from largest measured anchor
        a_s = max(_dense_anchor)
        dense_ms = _dense_anchor[a_s] * (s / a_s) ** 2
        dense_tag = f"~extrap(x{(s/a_s)**2:.0f})"
    elif DENSE_MS > 0:                        # external extrapolated dense
        dense_ms = DENSE_MS
        dense_tag = "~extrap"

    attn_ms = r_sparse[0] if r_sparse else None
    idx_ms, topk_ms, idxdt = (r_idx if r_idx else (None, None, "-"))

    if None in (attn_ms, idx_ms, topk_ms) or dense_ms is None:
        errs = "; ".join(x for x in (e_d, e_s, e_i) if x)
        print(f"{s:>8} | PARTIAL dense={dense_ms} idx={idx_ms} topk={topk_ms} attn={attn_ms} | {errs}", flush=True)
        continue
    full_ms = idx_ms + topk_ms + attn_ms
    print(f"{s:>8} | {dense_ms:>9.3f}{dense_tag} | {idx_ms:>8.3f} | {topk_ms:>8.3f} | {attn_ms:>8.3f} | "
          f"{full_ms:>11.3f} | {dense_ms/full_ms:>6.2f}x | {dense_ms/attn_ms:>10.2f}x | {idxdt}", flush=True)
print("DONE", flush=True)
