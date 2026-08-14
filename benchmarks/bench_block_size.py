"""Sparse-attn kernel: block=64 vs block=128, topk=16, Compass-V4 shape.

Note the token budgets differ by construction: topk=16 blocks of 64 tokens
attends 1024 KV tokens per query, of 128 attends 2048. So this is the
granularity/latency tradeoff, not an apples-to-apples same-work comparison.
"""
import sys, importlib.util
sys.path.insert(0, "python")
import torch
_spec = importlib.util.spec_from_file_location("benchmod", "benchmarks/bench_sparse_attention_ops.py")
bm = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(bm)
SEQS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "32768").split(",")]
print(f"{'seq':>9} | {'blk128(ms)':>10} | {'blk64(ms)':>10} | {'64/128':>7} | tokens/q 128 vs 64")
print("-" * 70)
for S in SEQS:
    row = {}
    for page in (128, 64):
        try:
            row[page] = bm.bench_sparse(1, 32, 4, S, S, 128, "o", True, "bf16",
                                        page_size=page, topk=16)[0]
        except Exception as e:
            row[page] = None
            print(f"{S:>9} | page={page} ERR {repr(e)[:70]}")
        torch.cuda.empty_cache()
    if row.get(128) and row.get(64):
        print(f"{S:>9} | {row[128]:>10.4f} | {row[64]:>10.4f} | {row[64]/row[128]:>6.2f}x | "
              f"{min(16*128,S)} vs {min(16*64,S)}")
print("DONE")
