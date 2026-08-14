"""Sparse-attn kernel across (block, topk) configs, Compass-V4 shape.

The KV budget per query is block x topk, so configs are only comparable when
that product matches. The default grid holds it at 2048 (the shipped config)
and at 1024 two different ways, which is the question worth asking: for a given
budget, does granularity cost anything?

  CONFIGS="128x16,128x8,64x16" python benchmarks/bench_block_size.py 8192,32768

Only the attention stage is measured. On this branch the selection stage cannot
follow: sparse_topk_select asserts topk == 16, and the indexer's max_score tile
is fixed at 128 regardless of page_size, so a full pipeline at any other
(block, topk) needs the scoring path from msa-configurable-sparse-attention.
"""
import os, sys, importlib.util
sys.path.insert(0, "python")
import torch

_spec = importlib.util.spec_from_file_location(
    "benchmod", "benchmarks/bench_sparse_attention_ops.py")
bm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bm)

H_Q = int(os.environ.get("H_Q", "32"))
H_K = int(os.environ.get("H_K", "4"))
D = int(os.environ.get("D", "128"))
CONFIGS = [c.strip() for c in os.environ.get("CONFIGS", "128x16,128x8,64x16").split(",")]
SEQS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "32768").split(",")]

cfgs = []
for c in CONFIGS:
    blk, topk = (int(x) for x in c.lower().split("x"))
    cfgs.append((blk, topk))

hdr = f"{'seq':>9} | " + " | ".join(f"{f'{b}x{t}(ms)':>12}" for b, t in cfgs) + " | budget"
print(f"# sparse attn only | B300 | bf16 | h_q={H_Q}/h_kv={H_K} d={D} | B=1 causal", flush=True)
print(hdr, flush=True)
print("-" * len(hdr), flush=True)

for S in SEQS:
    cells, budgets = [], []
    for blk, topk in cfgs:
        try:
            ms = bm.bench_sparse(1, H_Q, H_K, S, S, D, "o", True, "bf16",
                                 page_size=blk, topk=topk)[0]
            cells.append(f"{ms:>12.4f}")
        except Exception as e:
            cells.append(f"{'ERR':>12}")
            print(f"  {blk}x{topk}: {repr(e)[:100]}", flush=True)
        budgets.append(min(blk * topk, S))
        torch.cuda.empty_cache()
    print(f"{S:>9} | " + " | ".join(cells) + " | " +
          ", ".join(f"{b}x{t}={g}" for (b, t), g in zip(cfgs, budgets)), flush=True)
print("DONE", flush=True)
