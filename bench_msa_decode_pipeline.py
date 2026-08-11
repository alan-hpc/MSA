#!/usr/bin/env python3
"""MSA DECODE full-pipeline: dense GQA decode vs (indexer + topk + sparse-attn) per step.

Analogous to bench_msa_full_pipeline.py but decode-shaped (small q_len, kv_len=N, paged).
The shipped `sparse_decode` section uses FIXED "first top-K" block indices -> attn-kernel only.
This ADDS a naive per-step proxy indexer (proxy max-score pass, q_len small) +
sparse_topk_select. It is useful as a sanity check for the missing stages, but is
not a reproduction of the production/paper decode pipeline with index reuse,
low-precision/fused indexer, or MTP sharing.

  stage1 indexer : MQA proxy max-score pass (h_q=h_k=4, h_kv=1), q_len=QLEN, kv_len=N
  stage2 topk    : sparse_topk_select(max_score, topk=16)
  stage3 attn    : sparse paged decode attn over selected blocks (bench_sparse)
  dense baseline : paged dense GQA decode (bench_paged)

Run inside msa/ :  DTYPE=fp8 QLEN=4 python bench_msa_decode_pipeline.py 32768,131072,262144,524288,1048576
"""
import sys, math, os, importlib.util
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent / "msa"
if not (REPO / "python").exists():
    REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "python"))

from fmha_sm100 import fmha_sm100, fmha_sm100_plan, sparse_topk_select   # noqa: E402
from fmha_sm100.bench_utils import bench_gpu_time                        # noqa: E402

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
PAGE, TOPK = 128, 16
# Always-selected blocks inside the top-k budget: FORCE_BEGIN pins the sink
# blocks at the head of the sequence, FORCE_END pins the local window nearest
# the query. They occupy slots *within* topk (the result is always topk wide),
# so they change which blocks are attended, not how many -- latency-neutral.
# Default 0 = pure top-k by score, which is what the published table measured.
FORCE_BEGIN = int(os.environ.get("FORCE_BEGIN", "0"))
FORCE_END = int(os.environ.get("FORCE_END", "0"))
DTYPE = os.environ.get("DTYPE", "fp8")
QLEN = int(os.environ.get("QLEN", "4"))          # decode query len (MSA bench uses 4)
IDX_DTYPE = os.environ.get("IDX_DTYPE", DTYPE)   # indexer proxy dtype (fall back to bf16 if maxscore unsupported)
BATCHES = [int(x) for x in os.environ.get("BATCHES", "1").split(",")]
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
KVS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1
                        else "32768,131072,262144,524288,1048576").split(",")]


def _try(fn):
    try:
        return fn(), None
    except Exception as e:
        torch.cuda.empty_cache()
        return None, repr(e)[:90]


def _proxy(batch, kv_len, dtype_str):
    dt = torch.float8_e4m3fn if dtype_str == "fp8" else torch.bfloat16
    init = torch.half if dt.itemsize == 1 else dt
    q_len = QLEN
    pages = (kv_len + PAGE - 1) // PAGE
    total_pages = batch * pages
    qd = torch.randn(batch * q_len, IDX_HQ, D, device=DEV, dtype=init).to(dt)
    kd = torch.randn(total_pages, IDX_HKV, PAGE, D, device=DEV, dtype=init).to(dt)
    vd = torch.randn(total_pages, IDX_HKV, PAGE, D, device=DEV, dtype=init).to(dt)
    stride = (pages + 3) // 4 * 4
    kv_idx = torch.zeros(batch, stride, device=DEV, dtype=torch.int32)
    page_ids = torch.arange(total_pages, device=DEV, dtype=torch.int32)
    for b in range(batch):
        kv_idx[b, :pages] = page_ids[b * pages:(b + 1) * pages]
    qo = torch.full((batch,), q_len, dtype=torch.int32)
    kv = torch.full((batch,), kv_len, dtype=torch.int32)
    qoff = torch.full((batch,), kv_len - q_len, dtype=torch.int32)
    plan = fmha_sm100_plan(qo, kv, IDX_HQ, qo_offset=qoff, causal=True,
                           page_size=PAGE, num_kv_heads=IDX_HKV, output_maxscore=True)

    def indexer():
        return fmha_sm100(qd, kd, vd, plan_info=plan, kv_indices=kv_idx,
                          sm_scale=1.0 / math.sqrt(D), output_o=False, output_maxscore=True)

    _, ms = indexer()
    return indexer, ms, pages


def time_decode_indexer_topk(batch, kv_len):
    # try requested dtype; if maxscore unsupported (None) fall back to bf16
    used = IDX_DTYPE
    indexer, ms, pages = _proxy(batch, kv_len, IDX_DTYPE)
    if ms is None and IDX_DTYPE != "bf16":
        used = "bf16"
        indexer, ms, pages = _proxy(batch, kv_len, "bf16")
    if ms is None:
        raise RuntimeError("maxscore None (both dtypes)")
    _dry, _rep = _win(100, 400)
    with nvtx("msa_idx"):
        t_idx = float(np.median(bench_gpu_time(
            indexer, dry_run_time_ms=_dry, repeat_time_ms=_rep)))
    ms = ms.contiguous()

    def topk():
        return sparse_topk_select(ms, TOPK, num_valid_pages=pages,
                                   force_begin_blocks=FORCE_BEGIN,
                                   force_end_blocks=FORCE_END)

    topk()
    _dry, _rep = _win(50, 300)
    with nvtx("msa_topk"):
        t_topk = float(np.median(bench_gpu_time(
            topk, dry_run_time_ms=_dry, repeat_time_ms=_rep)))
    return t_idx, t_topk, used


hdr = (f"{'B':>4} | {'kv_len':>8} | {'dense(ms)':>9} | {'idx(ms)':>8} | {'topk(ms)':>8} | {'attn(ms)':>8} | "
       f"{'full(ms)':>9} | {'full x':>7} | {'attn-only x':>11} | {'idx logGB/s':>11} | idxdt")
print(f"# MSA DECODE pipeline | B300 | dtype={DTYPE} | q_len={QLEN} | attn h_q={H_Q}/h_kv={H_K} d={D} | "
      f"index h_q={IDX_HQ}/h_kv={IDX_HKV} dt={IDX_DTYPE} | topk={TOPK} page={PAGE} | batches={BATCHES}", flush=True)
print(hdr, flush=True)
print("-" * len(hdr), flush=True)

for batch in BATCHES:
    for kv in KVS:
        with nvtx("msa_dense"):
            r_dense, e_d = _try(lambda: bm.bench_paged(batch, H_Q, H_K, QLEN, kv, D, "o", True, DTYPE, page_size=PAGE, use_mbu=True))
        with nvtx("msa_attn"):
            r_sp, e_s = _try(lambda: bm.bench_sparse(batch, H_Q, H_K, QLEN, kv, D, "o", True, DTYPE, page_size=PAGE, topk=TOPK))
        r_i, e_i = _try(lambda: time_decode_indexer_topk(batch, kv))
        if None in (r_dense, r_sp, r_i):
            print(f"{batch:>4} | {kv:>8} | ERR dense={e_d} sparse={e_s} idx={e_i}", flush=True)
            continue
        dense_ms, attn_ms = r_dense[0], r_sp[0]
        idx_ms, topk_ms, idxdt = r_i
        full_ms = idx_ms + topk_ms + attn_ms
        idx_elem_bytes = 1 if idxdt == "fp8" else 2
        idx_gbs = batch * kv * IDX_HKV * D * idx_elem_bytes / (idx_ms * 1e-3) / 1e9
        print(f"{batch:>4} | {kv:>8} | {dense_ms:>9.4f} | {idx_ms:>8.4f} | {topk_ms:>8.4f} | {attn_ms:>8.4f} | "
              f"{full_ms:>9.4f} | {dense_ms/full_ms:>6.2f}x | {dense_ms/attn_ms:>10.2f}x | {idx_gbs:>8.1f} | {idxdt}", flush=True)
print("DONE", flush=True)
