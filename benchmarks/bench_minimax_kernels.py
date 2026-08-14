"""fw-ai/minimax-kernels kvouter_attention vs MSA's sparse attn, same process.

attention : h_q=32 / h_kv=4 / d=128   (Compass-V4)
index     : topk=16, block=128        (MSA)

Both take the same per-query selected-block list -- MSA's kv_block_indexes and
minimax-kernels' `selected` are both [Tq, Hkv, topK] int32 with -1 padding -- so
the same selection drives both kernels. Run under one interpreter so the
cutlass-dsl version is identical for both.
"""
import sys, os, importlib.util, traceback
sys.path.insert(0, "python")          # MSA's fmha_sm100 from this checkout
import numpy as np, torch

_spec = importlib.util.spec_from_file_location(
    "benchmod", "benchmarks/bench_sparse_attention_ops.py")
bm = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(bm)
from fmha_sm100.bench_utils import bench_gpu_time
from minimax_kernels.m3_sparse_attention import kvouter_attention

H_Q = int(os.environ.get("H_Q", "32"))
H_K = int(os.environ.get("H_K", "4"))
D, BLOCK, TOPK, DEV = 128, 128, 16, "cuda"
PAGE = int(os.environ.get("PAGE", "128"))
DT = torch.bfloat16
SEQS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "32768").split(",")]


def run_mk(S):
    pages = (S + PAGE - 1) // PAGE
    q = torch.randn(S, H_Q, D, device=DEV, dtype=DT)
    k = torch.randn(pages, H_K, PAGE, D, device=DEV, dtype=DT)
    v = torch.randn(pages, H_K, PAGE, D, device=DEV, dtype=DT)
    nblk = min(TOPK, (S + BLOCK - 1) // BLOCK)
    sel = torch.full((S, H_K, TOPK), -1, device=DEV, dtype=torch.int32)
    sel[:, :, :nblk] = torch.arange(nblk, device=DEV, dtype=torch.int32).view(1, 1, -1)
    bt = torch.arange(pages, device=DEV, dtype=torch.int32).view(1, pages)
    cu_q = torch.tensor([0, S], device=DEV, dtype=torch.int32)
    used = torch.tensor([S], device=DEV, dtype=torch.int32)

    def fn():
        return kvouter_attention(q, k, v, sel, bt, cu_seqlens_q=cu_q,
                                 causal=True, used_kv_lens=used,
                                 block_size=BLOCK, page_size=PAGE,
                                 out_dtype=DT, return_lse=False)

    o, _ = fn(); torch.cuda.synchronize()
    ms = float(np.median(bench_gpu_time(fn, dry_run_time_ms=100, repeat_time_ms=400)))
    return ms, tuple(o.shape)


def run_msa(S):
    ms = bm.bench_sparse(1, H_Q, H_K, S, S, D, "o", True, "bf16",
                         page_size=PAGE, topk=TOPK)[0]
    return ms


backend = os.environ.get("MINIMAX_KERNELS_KVOUTER_CPP", "auto")
print(f"# attn h_q={H_Q}/h_kv={H_K} d={D} | topk={TOPK} block={BLOCK} page={PAGE} | "
      f"bf16 causal | mk backend={backend}", flush=True)
print(f"{'seq':>8} | {'MSA attn(ms)':>12} | {'minimax-kernels(ms)':>19} | {'mk/MSA':>7}")
print("-" * 60)
for S in SEQS:
    try:
        t_msa = run_msa(S)
        torch.cuda.empty_cache()
        t_mk, shape = run_mk(S)
        print(f"{S:>8} | {t_msa:>12.4f} | {t_mk:>19.4f} | {t_mk/t_msa:>6.3f}x   out={shape}",
              flush=True)
    except Exception as e:
        print(f"{S:>8} | ERR {repr(e)[:120]}", flush=True)
        traceback.print_exc(limit=4)
    torch.cuda.empty_cache()
print("DONE")
