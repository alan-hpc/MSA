#!/usr/bin/env python3
"""FlashInfer `msa_ops.msa_sparse_attention` — sparse-attn stage A/B against MSA.

This times ONLY stage 3 of the pipeline (sparse attention over already-chosen
blocks), which is the stage the two projects actually share: FlashInfer takes
`q2k_indices` as an input, exactly like MSA's sparse path takes `kv_block_indexes`.
The indexer and top-k stages have no FlashInfer counterpart here, so a full
end-to-end number would be comparing our indexer against itself.

Index layout differs and is transposed here:
  MSA   `sparse_topk_select` -> (total_q, num_qo_heads, topk)   [per QO head]
  FI    `q2k_indices`         : (num_kv_heads, total_q, topk)   [per KV head]
For Compass-V4 the two line up (indexer h_q = 4 = h_kv), so the same selection
feeds both kernels.

Block contents follow MSA's shipped bench: the first min(topk, blocks_so_far)
causal-valid blocks, -1 padded at the tail. Kernel time is largely independent
of *which* blocks are named, so this keeps the comparison about the kernel.

  H_Q=32 H_K=4 D=128 python bench_flashinfer_msa.py 32768,131072
"""
import os, sys
import torch

from flashinfer.msa_ops import msa_sparse_attention   # noqa: E402

DEV, DT = "cuda", torch.bfloat16
H_Q = int(os.environ.get("H_Q", "32"))
H_K = int(os.environ.get("H_K", "4"))
D = int(os.environ.get("D", "128"))
PAGE = int(os.environ.get("PAGE", "128"))
TOPK = int(os.environ.get("TOPK", "16"))
MODE = os.environ.get("FI_MODE", "prefill")      # prefill | decode
BATCH = int(os.environ.get("BATCH", "32"))       # decode only
QLEN = int(os.environ.get("QLEN", "1"))          # decode only
LAYOUT = os.environ.get("FI_LAYOUT", "paged")    # paged | flat
WARMUP = int(os.environ.get("FI_WARMUP", "5"))
REP = int(os.environ.get("FI_REP", "20"))
SEQS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1
                         else "32768,131072").split(",")]


def do_bench(fn, warmup=WARMUP, rep=REP):
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


def build_q2k(q_pos, nblocks_per_q):
    """(H_K, total_q, TOPK) int32: first causal-valid blocks, -1 padded at tail."""
    total_q = q_pos.numel()
    idx = torch.arange(TOPK, device=DEV, dtype=torch.int32).view(1, TOPK)
    valid = idx < nblocks_per_q.view(total_q, 1)
    rows = torch.where(valid, idx.expand(total_q, TOPK), torch.full_like(idx, -1))
    return rows.unsqueeze(0).expand(H_K, total_q, TOPK).contiguous()


def run_prefill(S):
    total_q = S
    pages = (S + PAGE - 1) // PAGE
    q = torch.randn(total_q, H_Q, D, device=DEV, dtype=DT)
    if LAYOUT == "paged":
        k = torch.randn(pages, H_K, PAGE, D, device=DEV, dtype=DT)
        v = torch.randn(pages, H_K, PAGE, D, device=DEV, dtype=DT)
        page_table = torch.arange(pages, device=DEV, dtype=torch.int32).view(1, pages)
    else:
        k = torch.randn(S, H_K, D, device=DEV, dtype=DT)
        v = torch.randn(S, H_K, D, device=DEV, dtype=DT)
        page_table = None
    q_pos = torch.arange(total_q, device=DEV, dtype=torch.int32)
    q2k = build_q2k(q_pos, q_pos // PAGE + 1)
    cu_q = torch.tensor([0, total_q], device=DEV, dtype=torch.int32)
    cu_k = torch.tensor([0, S], device=DEV, dtype=torch.int32)
    seqused_k = torch.tensor([S], device=DEV, dtype=torch.int32)

    def fn():
        return msa_sparse_attention(
            q, k, v, q2k, cu_q, cu_seqlens_k=cu_k, causal=True,
            page_table=page_table, seqused_k=seqused_k)

    fn(); torch.cuda.synchronize()
    return do_bench(fn)


def run_decode(kv_len):
    total_q = BATCH * QLEN
    pages = (kv_len + PAGE - 1) // PAGE
    q = torch.randn(total_q, H_Q, D, device=DEV, dtype=DT)
    if LAYOUT == "paged":
        k = torch.randn(BATCH * pages, H_K, PAGE, D, device=DEV, dtype=DT)
        v = torch.randn(BATCH * pages, H_K, PAGE, D, device=DEV, dtype=DT)
        page_table = torch.arange(BATCH * pages, device=DEV,
                                  dtype=torch.int32).view(BATCH, pages)
    else:
        k = torch.randn(BATCH * kv_len, H_K, D, device=DEV, dtype=DT)
        v = torch.randn(BATCH * kv_len, H_K, D, device=DEV, dtype=DT)
        page_table = None
    # every decode step sees the whole context, so all topk slots are valid
    nblk = torch.full((total_q,), pages, device=DEV, dtype=torch.int32)
    q2k = build_q2k(torch.arange(total_q, device=DEV, dtype=torch.int32), nblk)
    cu_q = torch.arange(BATCH + 1, device=DEV, dtype=torch.int32) * QLEN
    cu_k = torch.arange(BATCH + 1, device=DEV, dtype=torch.int32) * kv_len
    seqused_k = torch.full((BATCH,), kv_len, device=DEV, dtype=torch.int32)

    def fn():
        return msa_sparse_attention(
            q, k, v, q2k, cu_q, cu_seqlens_k=cu_k, causal=True,
            page_table=page_table, seqused_k=seqused_k)

    fn(); torch.cuda.synchronize()
    return do_bench(fn)


import flashinfer  # noqa: E402
print(f"# FlashInfer msa_sparse_attention | B300 | bf16 | {MODE} | "
      f"h_q={H_Q}/h_kv={H_K} d={D} topk={TOPK} page={PAGE} layout={LAYOUT}"
      + (f" | B={BATCH} Sq={QLEN}" if MODE == "decode" else " | B=1 causal"), flush=True)
print(f"{'seq':>9} | {'FI attn(ms)':>11}", flush=True)
print("-" * 25, flush=True)

for S in SEQS:
    try:
        ms = run_prefill(S) if MODE == "prefill" else run_decode(S)
        print(f"{S:>9} | {ms:>11.4f}", flush=True)
    except Exception as e:
        print(f"{S:>9} | ERR {repr(e)[:150]}", flush=True)
    torch.cuda.empty_cache()
print("DONE", flush=True)
