"""A/B on SM103: MSA's KV-outer (CuTe CSR) vs its Q-outer (CUTLASS SparseAttnMode::Sparse).

Both kernels already exist in MSA. The router picks between them by q_len:
  sparse_kernel_mode='auto' and q_len > 32  -> sparse_fmha_plan -> KV-outer CSR
  otherwise                                 -> CUTLASS Q-outer

Q-outer never runs at prefill length because its planner expands the work list
per query token and asserts `total_q * num_qo_heads <= 65536` -- 2048 tokens at
h_q=32. That is a planner limit, not a kernel limit, so this chunks the query
dimension into <=2048-token tiles and sums the per-chunk time. Each tile is an
independent launch, so this measures the Q-outer *structure* at prefill shapes
without cross-tile scheduling.
"""
import sys, os, traceback
sys.path.insert(0, "python")
import numpy as np, torch
from fmha_sm100 import fmha_sm100, fmha_sm100_plan
from fmha_sm100.bench_utils import bench_gpu_time

H_Q = int(os.environ.get("H_Q", "32")); H_K = int(os.environ.get("H_K", "4"))
D = 128; PAGE = 128; TOPK = int(os.environ.get("TOPK", "16")); DEV = "cuda"
Q_CHUNK = 65536 // H_Q                      # the planner cap
SEQS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "32768").split(",")]


def build(S):
    q = torch.randn(S, H_Q, D, device=DEV, dtype=torch.bfloat16)
    pages = (S + PAGE - 1) // PAGE
    k = torch.randn(pages, H_K, PAGE, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(pages, H_K, PAGE, D, device=DEV, dtype=torch.bfloat16)
    stride = (pages + 3) // 4 * 4
    kv_idx = torch.zeros(1, stride, dtype=torch.int32, device=DEV)
    kv_idx[0, :pages] = torch.arange(pages, device=DEV, dtype=torch.int32)
    nblk = min(TOPK, pages)
    bidx = torch.full((S, H_K, TOPK), -1, device=DEV, dtype=torch.int32)
    bidx[:, :, :nblk] = torch.arange(nblk, device=DEV, dtype=torch.int32).view(1, 1, -1)
    return q, k, v, kv_idx, bidx, pages


def run_kvouter(S, t):
    q, k, v, kv_idx, bidx, _ = t
    plan = fmha_sm100_plan(
        torch.full((1,), S, dtype=torch.int32), torch.full((1,), S, dtype=torch.int32), H_Q,
        qo_offset=torch.zeros(1, dtype=torch.int32), page_size=PAGE,
        kv_block_num=TOPK, num_kv_heads=H_K, sparse_kernel_mode='auto')
    assert plan[3].get("MM-SA-Nv"), "expected the KV-outer plan"
    out = torch.empty(S, H_Q, D, device=DEV, dtype=torch.bfloat16)
    fn = lambda: fmha_sm100(q, k, v, plan_info=plan, kv_indices=kv_idx,
                            out=out, kv_block_indexes=bidx)
    fn(); torch.cuda.synchronize()
    return float(np.median(bench_gpu_time(fn, dry_run_time_ms=100, repeat_time_ms=400))), out


def run_qouter_chunked(S, t):
    q, k, v, kv_idx, bidx, _ = t
    out = torch.empty(S, H_Q, D, device=DEV, dtype=torch.bfloat16)
    total, start, nchunks = 0.0, 0, 0
    while start < S:
        c = min(Q_CHUNK, S - start)
        qc = q[start:start + c].contiguous()
        bc = bidx[start:start + c].contiguous()
        oc = torch.empty(c, H_Q, D, device=DEV, dtype=torch.bfloat16)
        plan = fmha_sm100_plan(
            torch.full((1,), c, dtype=torch.int32), torch.full((1,), S, dtype=torch.int32), H_Q,
            qo_offset=torch.full((1,), start, dtype=torch.int32),
            page_size=PAGE, kv_block_num=TOPK, num_kv_heads=H_K,
            sparse_kernel_mode='decode')
        assert not plan[3].get("MM-SA-Nv", False), "expected the CUTLASS Q-outer plan"
        fn = lambda: fmha_sm100(qc, k, v, plan_info=plan, kv_indices=kv_idx,
                                out=oc, kv_block_indexes=bc)
        fn(); torch.cuda.synchronize()
        total += float(np.median(bench_gpu_time(fn, dry_run_time_ms=50, repeat_time_ms=200)))
        out[start:start + c] = oc
        start += c; nchunks += 1
    return total, out, nchunks


print(f"q chunk = {Q_CHUNK} tokens (planner cap 65536 / h_q={H_Q})")
print(f"{'seq':>9} | {'KV-outer(ms)':>12} | {'Q-outer(ms)':>11} | {'chunks':>6} | {'kv/q':>6} | {'cos':>7}")
print("-" * 72)
for S in SEQS:
    try:
        t = build(S)
        kv_ms, kv_o = run_kvouter(S, t)
        q_ms, q_o, n = run_qouter_chunked(S, t)
        cos = torch.nn.functional.cosine_similarity(
            kv_o.float().flatten().unsqueeze(0), q_o.float().flatten().unsqueeze(0)).item()
        print(f"{S:>9} | {kv_ms:>12.3f} | {q_ms:>11.3f} | {n:>6} | {kv_ms/q_ms:>5.2f}x | {cos:>7.4f}")
    except Exception as e:
        print(f"{S:>9} | ERR {repr(e)[:110]}")
        traceback.print_exc(limit=4)
    torch.cuda.empty_cache()
print("AB_DONE")
