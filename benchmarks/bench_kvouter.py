#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""A/B the KV-outer sparse backend against this repo's CSR + Q-outer backend.

The two implementations live in checkouts that both provide a top-level
``fmha_sm100`` package, so they cannot share a process.  This script therefore
runs in two roles that communicate through a directory:

  ``--role qouter``  this repo.  Runs the FP4 indexer and ``sparse_topk_select``
                     to get a real selection, times ``build_k2q_csr`` +
                     ``sparse_atten_func``, and writes the selection and the
                     output.
  ``--role kvouter`` the fireworks checkout.  Reconstructs the *identical* Q/K/V
                     from the same seed, loads the selection, times
                     ``kvouter_attention`` (index build + forward + LSE merge),
                     and compares its output against the one on disk.

Only the selection and the outputs cross the boundary; Q/K/V are regenerated
bit-for-bit from a seeded generator on both sides, so the two backends are
measured on exactly the same inputs.

Comparability notes, stated because they bound what the numbers mean:

* KV-outer asserts ``block_size == 128`` and ``head_dim == 128``, so the
  comparison is only defined at ``block=128``.  That is where this repo's own
  sweep puts its recommended operating point, so it is the interesting column
  anyway.
* Stages 1-2 (FP4 indexer, top-k select) are identical and are *not* included in
  either total; what is compared is strictly the selection-consuming half of the
  pipeline, which is where the two designs actually differ.
* KV-outer takes a paged KV cache while this repo's forward takes a flat one.
  The pack is a setup cost and is not timed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Shared input construction — must be identical on both sides
# ---------------------------------------------------------------------------

def make_inputs(*, seqlen, head_q, head_kv, dim, dtype, device, seed):
    """Q/K/V for one shape.  Deterministic given ``seed``, on both checkouts."""
    gen = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn((seqlen, head_q, dim), generator=gen, device=device, dtype=torch.float32)
    k = torch.randn((seqlen, head_kv, dim), generator=gen, device=device, dtype=torch.float32)
    v = torch.randn((seqlen, head_kv, dim), generator=gen, device=device, dtype=torch.float32)
    return q.to(dtype), k.to(dtype), v.to(dtype)


def timed_ms(fn, *, dry_ms=40, rep_ms=150):
    """Median single-call latency, L2 flushed between iterations, no CUDA graph."""
    flush = torch.empty(int(256e6 // 4), dtype=torch.int32, device="cuda")

    def once():
        flush.zero_()
        fn()

    once()
    torch.cuda.synchronize()

    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    once()
    end.record()
    torch.cuda.synchronize()
    single = start.elapsed_time(end)

    warmup = max(1, int(dry_ms / max(single, 1e-3)))
    iters = max(3, int(rep_ms / max(single, 1e-3)))
    for _ in range(warmup):
        once()
    torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        start.record()
        once()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2]


def cosine(a, b):
    a32 = a.float().reshape(-1, a.shape[-1])
    b32 = b.float().reshape(-1, b.shape[-1])
    return torch.nn.functional.cosine_similarity(a32, b32, dim=-1)


# ---------------------------------------------------------------------------
# Role: this repo (CSR + Q-outer)
# ---------------------------------------------------------------------------

def run_qouter(args):
    sys.path.insert(0, str(REPO_ROOT / "python"))
    sys.path.insert(0, str(REPO_ROOT / "python" / "fmha_sm100" / "cute"))
    sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

    from bench_msa_configs import make_block_scores
    from fmha_sm100.msa_config import MsaSparseConfig, model_by_name
    from fmha_sm100.msa_pipeline import MsaSparseAttention, causal_end_blocks

    device = torch.device("cuda")
    dtype = torch.bfloat16
    model = model_by_name(args.model)
    head_q, head_kv, dim = model.num_qo_heads, model.num_kv_heads, model.bench_head_dim()

    out_dir = Path(args.exchange)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for seqlen in args.seqlens:
        for topk in args.topks:
            cfg = MsaSparseConfig(block_size=128, topk=topk, force_init_tokens=128,
                                  force_end_tokens=128, head_mode="keep")
            tag = f"s{seqlen}_k{topk}"
            row = {"seqlen": seqlen, "topk": topk, "block": 128}
            try:
                msa = MsaSparseAttention(cfg, num_qo_heads=head_q, num_kv_heads=head_kv,
                                         head_dim=dim, causal=True)
                num_blocks = cfg.num_blocks(seqlen)
                q, k, v = make_inputs(seqlen=seqlen, head_q=head_q, head_kv=head_kv, dim=dim,
                                      dtype=dtype, device=device, seed=args.seed)
                cu_q = torch.tensor([0, seqlen], dtype=torch.int32, device=device)
                cu_k = torch.tensor([0, seqlen], dtype=torch.int32, device=device)

                # A uniform score draw is the adversarial case for the
                # histogram-refinement stages and, more importantly here, it is
                # reproducible — both backends consume the identical selection,
                # so how it was produced does not affect the comparison.
                scores = make_block_scores(num_heads=cfg.num_selection_heads(head_kv),
                                           num_blocks=num_blocks, total_q=seqlen,
                                           valid_blocks=num_blocks, device=device,
                                           seed=args.seed + 3)
                ceb = causal_end_blocks([seqlen], [seqlen], cfg.block_size, device=device)
                idx = msa.select(scores, num_valid_blocks=num_blocks, causal_end_block=ceb)

                csr = msa.build_csr(idx, cu_q, cu_k, total_k=seqlen, max_seqlen_q=seqlen,
                                    max_seqlen_k=seqlen, total_rows=num_blocks)
                row_ptr, q_indices = csr[0], csr[1]
                schedule = csr[2] if len(csr) > 2 else None

                def do_csr():
                    return msa.build_csr(idx, cu_q, cu_k, total_k=seqlen,
                                         max_seqlen_q=seqlen, max_seqlen_k=seqlen,
                                         total_rows=num_blocks)

                def do_attn():
                    return msa.attend(q, k, v, row_ptr, q_indices,
                                      cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                                      max_seqlen_q=seqlen, max_seqlen_k=seqlen,
                                      schedule=schedule)

                out = do_attn()
                torch.cuda.synchronize()
                row["csr_ms"] = timed_ms(do_csr, dry_ms=args.dry_ms, rep_ms=args.rep_ms)
                row["attn_ms"] = timed_ms(do_attn, dry_ms=args.dry_ms, rep_ms=args.rep_ms)
                row["total_ms"] = row["csr_ms"] + row["attn_ms"]

                out_t = out[0] if isinstance(out, (tuple, list)) else out
                # [Hkv, Tq, topk] -> [Tq, Hkv, topk], which is KV-outer's layout.
                sel = idx.permute(1, 0, 2).contiguous().to(torch.int32)
                torch.save({"selected": sel.cpu(),
                            "out": out_t.reshape(seqlen, head_q, dim).float().cpu()},
                           out_dir / f"{tag}.pt")
                row["ok"] = True
            except Exception as exc:  # noqa: BLE001 - one shape failing must not kill the sweep
                row["ok"] = False
                row["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
            results.append(row)
            print(json.dumps(row), flush=True)
            torch.cuda.empty_cache()

    (out_dir / "qouter.json").write_text(json.dumps(results, indent=2))
    return 0


# ---------------------------------------------------------------------------
# Role: fireworks checkout (KV-outer)
# ---------------------------------------------------------------------------

def run_kvouter(args):
    import types

    if args.fa4_path:
        sys.path.insert(0, args.fa4_path)
    # FA4's package __init__ imports FA2's compiled extension, which is not
    # built here; nothing on the KV-outer path touches it.
    sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))
    sys.path.insert(0, str(Path(args.kvouter_path) / "python"))

    from fmha_sm100.kvouter.interface import kvouter_attention

    device = torch.device("cuda")
    dtype = torch.bfloat16
    head_q, head_kv, dim = args.head_q, args.head_kv, 128
    page_size = args.page_size

    ex = Path(args.exchange)
    results = []
    for seqlen in args.seqlens:
        for topk in args.topks:
            tag = f"s{seqlen}_k{topk}"
            row = {"seqlen": seqlen, "topk": topk, "block": 128}
            blob = ex / f"{tag}.pt"
            if not blob.exists():
                row["ok"] = False
                row["error"] = "no selection from the q-outer role (it failed at this shape)"
                results.append(row)
                print(json.dumps(row), flush=True)
                continue
            try:
                q, k, v = make_inputs(seqlen=seqlen, head_q=head_q, head_kv=head_kv, dim=dim,
                                      dtype=dtype, device=device, seed=args.seed)
                saved = torch.load(blob, map_location="cpu")
                sel = saved["selected"].to(device)
                ref = saved["out"].to(device)

                # Flat [T, Hkv, D] -> paged [pages, Hkv, page_size, D]; setup, not timed.
                pages = (seqlen + page_size - 1) // page_size
                pad = pages * page_size - seqlen
                if pad:
                    k = torch.cat([k, k.new_zeros((pad, head_kv, dim))], 0)
                    v = torch.cat([v, v.new_zeros((pad, head_kv, dim))], 0)
                k_cache = k.view(pages, page_size, head_kv, dim).permute(0, 2, 1, 3).contiguous()
                v_cache = v.view(pages, page_size, head_kv, dim).permute(0, 2, 1, 3).contiguous()
                block_tables = torch.arange(pages, dtype=torch.int32, device=device).view(1, pages)
                cu_q = torch.tensor([0, seqlen], dtype=torch.int32, device=device)
                used_kv = torch.tensor([seqlen], dtype=torch.int32, device=device)

                def run():
                    return kvouter_attention(
                        q, k_cache, v_cache, sel, block_tables,
                        cu_seqlens_q=cu_q, causal=True, used_kv_lens=used_kv,
                        block_size=128, page_size=page_size, out_dtype=dtype)

                out, _ = run()
                torch.cuda.synchronize()
                row["total_ms"] = timed_ms(run, dry_ms=args.dry_ms, rep_ms=args.rep_ms)

                cos = cosine(out.reshape(seqlen, head_q, dim), ref.reshape(seqlen, head_q, dim))
                row["cos_mean"] = float(cos.mean())
                row["cos_min"] = float(cos.min())
                row["ok"] = True
            except Exception as exc:  # noqa: BLE001
                row["ok"] = False
                row["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
            results.append(row)
            print(json.dumps(row), flush=True)
            torch.cuda.empty_cache()

    (ex / "kvouter.json").write_text(json.dumps(results, indent=2))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", required=True, choices=("qouter", "kvouter"))
    ap.add_argument("--exchange", required=True, help="directory both roles read/write")
    ap.add_argument("--seqlens", type=lambda s: [int(x) for x in s.split(",")],
                    default=[32768, 131072])
    ap.add_argument("--topks", type=lambda s: [int(x) for x in s.split(",")],
                    default=[8, 16, 32])
    ap.add_argument("--model", default="n32")
    ap.add_argument("--head-q", type=int, default=32)
    ap.add_argument("--head-kv", type=int, default=4)
    ap.add_argument("--page-size", type=int, default=64, choices=(64, 128))
    ap.add_argument("--kvouter-path", default="", help="fireworks checkout root")
    ap.add_argument("--fa4-path", default="", help="flash-attention checkout root")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dry-ms", type=float, default=40.0)
    ap.add_argument("--rep-ms", type=float, default=150.0)
    args = ap.parse_args()

    if args.role == "kvouter" and not args.kvouter_path:
        ap.error("--kvouter-path is required for --role kvouter")
    return run_qouter(args) if args.role == "qouter" else run_kvouter(args)


if __name__ == "__main__":
    raise SystemExit(main())
