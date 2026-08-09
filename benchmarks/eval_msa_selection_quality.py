#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Selection *quality* evaluation for MSA head-aggregation modes.

The configuration sweep in ``bench_msa_configs.py`` answers how fast each
configuration is.  It says nothing about whether the blocks a configuration
picks are the right ones — and `head_mode` is precisely a quality/cost trade,
so ranking the modes on latency alone is not an answer.

This script measures the quality half, on a **real model's attention** rather
than on random tensors: it captures post-RoPE Q/K from a trained checkpoint,
computes exact dense attention, and asks how much of the attention probability
mass each selection recovers.

Metric
------
For query ``q`` and query head ``h`` with dense weights ``p(q, ·)`` and selected
token set ``S(q, h)``:

    recall(q, h) = sum over j in S(q, h) of p(q, j)

Recall of attention mass is the standard measure for block-sparse attention
(NSA / MoBA / DSA all report it): it is exactly the fraction of the softmax
numerator the sparse kernel still sees, and it upper-bounds the achievable
output fidelity.  Beyond the mean, two statistics matter:

* ``p5`` — the 5th percentile over queries, i.e. how bad the tail gets.
* ``min-in-group`` — the worst-served query head inside each GQA group.  A
  shared selection is only as good as the head it serves worst, and that is the
  whole point of comparing ``sum`` against ``max``.

Modes compared
--------------
``oracle``
    Each query head selects its own top-k.  Not implementable by the kernel
    (its CSR metadata is keyed on KV heads) but it bounds what any shared
    selection can reach, so it turns "sum vs max" into a question with a scale.
``sum`` / ``max``
    The two shipped reductions, evaluated at the model's real GQA group size.
``keep (mean-Q proxy)``
    ``keep`` scores one row per KV head, which MSA expects to come from a
    separate proxy-Q projection.  Nothing in MSA trains that projection, so the
    mean of the group's query vectors stands in for it here.  **This row is a
    lower bound on `keep`, not its ceiling** — a projection fitted for the job
    would do better.  Read it as "what an untrained one-row-per-KV-head scorer
    achieves with MSA's own scoring".

Usage
-----
    python benchmarks/eval_msa_selection_quality.py --seqlen 4096
    python benchmarks/eval_msa_selection_quality.py --layer 0 --seqlen 8192 --csv quality.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from real_qk import capture_qk  # noqa: E402

from fmha_sm100 import sparse_topk_select  # noqa: E402
from fmha_sm100.msa_config import CONFIG_MATRIX, MsaSparseConfig, iter_configs  # noqa: E402


# ---------------------------------------------------------------------------
# Capturing real post-RoPE Q/K
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Selection + recall
# ---------------------------------------------------------------------------


def block_scores_and_probs(q, k, *, scale, block_size, q_chunk=512):
    """Per-head block scores, proxy-Q block scores, and dense attention probs.

    Yields ``(q_start, scores[c, Hq, nb], proxy[c, Hkv, nb], probs[c, Hq, S])``.
    Block scores are the **max logit inside each block**, matching what the FP4
    indexer reduces, with causal masking applied first so an invisible block
    scores ``-inf``.

    ``proxy`` is the ``keep`` scorer: the group's query vectors are pooled
    **before** the QK product, giving one genuinely different query per KV head.
    Pooling the per-head *scores* instead would just be ``sum`` up to a positive
    constant, and top-k is invariant to that — so it has to happen here.
    """
    seqlen, head_q, dim = q.shape
    head_kv = k.shape[1]
    group = head_q // head_kv
    num_blocks = (seqlen + block_size - 1) // block_size
    pad = num_blocks * block_size - seqlen
    kt = k.transpose(0, 1)                                     # [Hkv, S, D]
    ktt = kt.transpose(1, 2)                                   # [Hkv, D, S]
    q_proxy = q.view(seqlen, head_kv, group, dim).mean(dim=2)  # [S, Hkv, D]

    def to_block_max(logits, heads):
        padded = logits
        if pad:
            padded = torch.nn.functional.pad(logits, (0, pad), value=float("-inf"))
        return padded.view(heads, -1, num_blocks, block_size).amax(dim=-1)

    for start in range(0, seqlen, q_chunk):
        stop = min(start + q_chunk, seqlen)
        pos = torch.arange(start, stop, device=q.device).view(-1, 1)
        causal = (torch.arange(seqlen, device=q.device).view(1, -1) <= pos).unsqueeze(0)

        logits = torch.bmm(
            q[start:stop].transpose(0, 1), ktt.repeat_interleave(group, 0)
        ) * scale                                              # [Hq, c, S]
        logits = logits.masked_fill(~causal, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        scores = to_block_max(logits, head_q)
        del logits

        proxy_logits = torch.bmm(q_proxy[start:stop].transpose(0, 1), ktt) * scale  # [Hkv, c, S]
        proxy_logits = proxy_logits.masked_fill(~causal, float("-inf"))
        proxy = to_block_max(proxy_logits, head_kv)
        del proxy_logits

        yield (start,
               scores.permute(1, 0, 2).contiguous(),
               proxy.permute(1, 0, 2).contiguous(),
               probs.permute(1, 0, 2).contiguous())


def select(scores, *, cfg, head_mode, num_out_heads, num_valid_blocks, q_positions):
    """Run the production selection kernel.  ``scores`` is ``[c, Hin, nb]``."""
    # Kernel wants [Hin, nb, c].
    scores_k = scores.permute(1, 2, 0).contiguous()
    causal_end = (q_positions // cfg.block_size + 1).to(torch.int32).contiguous()
    idx = sparse_topk_select(
        scores_k,
        cfg.topk,
        num_valid_pages=num_valid_blocks,
        force_begin_blocks=cfg.force_init_blocks,
        force_end_blocks=cfg.force_end_blocks,
        head_mode=head_mode,
        num_out_heads=num_out_heads,
        causal_end_block=causal_end,
        head_outermost=True,
    )
    return idx                                                  # [Hout, c, topk]


def recall_from_selection(idx, probs, *, block_size, seqlen, group):
    """Attention-mass recall per (query, query head).

    ``idx`` is ``[Hout, c, topk]``; each output head's selection is shared by
    ``group`` consecutive query heads.  ``probs`` is ``[c, Hq, S]``.
    """
    head_out, chunk, topk = idx.shape
    num_blocks = (seqlen + block_size - 1) // block_size

    mask = torch.zeros((head_out, chunk, num_blocks), dtype=torch.bool, device=probs.device)
    valid = idx >= 0
    mask.scatter_(2, idx.clamp(min=0).long(), valid)
    token_mask = mask.repeat_interleave(block_size, dim=2)[:, :, :seqlen]   # [Hout, c, S]
    token_mask = token_mask.repeat_interleave(group, dim=0)                 # [Hq, c, S]

    kept = (probs.permute(1, 0, 2) * token_mask).sum(dim=-1)                # [Hq, c]
    return kept.transpose(0, 1).contiguous()                                # [c, Hq]


def evaluate(qk, cfg, *, head_q, head_kv, seqlen, q_chunk, device):
    """Recall statistics for every mode at one configuration, on one layer."""
    q, k, scale = qk["q"], qk["k"], qk["scale"]
    group = head_q // head_kv
    num_blocks = (seqlen + cfg.block_size - 1) // cfg.block_size

    # "@1" variants share ONE selection across the whole layer (DSA-style):
    # num_out_heads=1 instead of one per KV head.
    acc = {m: [] for m in ("oracle", "keep", "sum", "max")}
    for start, scores, proxy, probs in block_scores_and_probs(
        q, k, scale=scale, block_size=cfg.block_size, q_chunk=q_chunk
    ):
        chunk = scores.shape[0]
        pos = torch.arange(start, start + chunk, device=device)

        # oracle: every query head selects for itself.
        idx = select(scores, cfg=cfg, head_mode="keep", num_out_heads=head_q,
                     num_valid_blocks=num_blocks, q_positions=pos)
        acc["oracle"].append(recall_from_selection(
            idx, probs, block_size=cfg.block_size, seqlen=seqlen, group=1))

        # sum / max: reduce the SAME per-KV-head proxy rows `keep` uses down to
        # one selection shared by the whole layer.  The scorer therefore does
        # identical work to `keep`; only the number of selections changes.
        for mode in ("sum", "max"):
            idx = select(proxy, cfg=cfg, head_mode=mode, num_out_heads=1,
                         num_valid_blocks=num_blocks, q_positions=pos)
            acc[mode].append(recall_from_selection(
                idx, probs, block_size=cfg.block_size, seqlen=seqlen, group=head_q))

        # keep: one row per KV head, scored from the pooled proxy query.
        idx = select(proxy, cfg=cfg, head_mode="keep", num_out_heads=head_kv,
                     num_valid_blocks=num_blocks, q_positions=pos)
        acc["keep"].append(recall_from_selection(
            idx, probs, block_size=cfg.block_size, seqlen=seqlen, group=group))

        del scores, proxy, probs

    out = {}
    for mode, chunks in acc.items():
        r = torch.cat(chunks, dim=0)                                   # [S, Hq]
        # `keep` shares within a GQA group; sum/max/keep@1 share layer-wide.
        g = 1 if mode == "oracle" else (group if mode == "keep" else head_q)
        worst = r.view(-1, head_q // g, g).amin(dim=-1) if g > 1 else r
        out[mode] = {
            "recall_mean": float(r.mean()),
            "recall_p5": float(torch.quantile(r.flatten().float(), 0.05)),
            "recall_min_in_group": float(worst.mean()),
        }
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="<checkpoint>",
                   help="checkpoint to source real post-RoPE Q/K from; only its main "
                        "attention projections are read")
    p.add_argument("--seqlen", type=int, default=4096)
    p.add_argument("--layer", type=int, default=0,
                   help="which attention layer's projections to read (default 0)")
    p.add_argument("--configs", default="cfg01,cfg03,cfg02,cfg04",
                   help="configs to evaluate; head_mode in the name is ignored "
                        "because every mode is evaluated for each block/topk pair")
    p.add_argument("--q-chunk", type=int, default=512)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--csv", default=None)
    p.add_argument("--json", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return 0
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda", args.gpu)

    print("=" * 104)
    print("MSA selection-quality evaluation — attention-mass recall on real trained Q/K")
    print(f"model   : {args.model}")
    print(f"layer   : {args.layer}    seqlen: {args.seqlen}")
    print("=" * 104)

    qk = capture_qk(args.model, seqlen=args.seqlen, layer=args.layer, device=device)
    head_q, head_kv, head_dim = qk["head_q"], qk["head_kv"], qk["head_dim"]
    group = head_q // head_kv
    print(f"geometry: Hq={head_q} Hkv={head_kv} GQA group={group} head_dim={head_dim}")
    print("caveat  : single layer, hidden states = input_layernorm(embed) (layer-0 input).")
    print("          Early-layer attention is more local than mid-stack, so the spread")
    print("          between head modes here is a lower bound.")
    print()

    # De-duplicate on (block_size, topk): head_mode is swept internally.
    seen, shapes = set(), []
    for cfg in iter_configs(args.configs):
        key = (cfg.block_size, cfg.topk)
        if key not in seen:
            seen.add(key)
            shapes.append(cfg)

    rows = []
    header = (f"{'cfg':<8} {'block':>5} {'topk':>4} {'budget':>7} "
              f"{'mode':<22} {'recall':>8} {'p5':>8} {'min-in-grp':>11} {'vs oracle':>9}")
    print(header)
    print("-" * len(header))
    for cfg in shapes:
        stats = evaluate(qk, cfg, head_q=head_q, head_kv=head_kv,
                         seqlen=args.seqlen, q_chunk=args.q_chunk, device=device)
        oracle = stats["oracle"]["recall_mean"]
        for mode in ("oracle", "keep", "sum", "max"):
            st = stats[mode]
            label = {"oracle": "oracle (per-head)", "keep": "keep  4 rows -> 4 sel",
                     "sum": "sum   4 rows -> 1 sel", "max": "max   4 rows -> 1 sel"}[mode]
            rel = "—" if mode == "oracle" else f"{st['recall_mean'] / oracle:.3f}"
            print(f"{cfg.name:<8} {cfg.block_size:>5} {cfg.topk:>4} {cfg.selected_tokens:>7} "
                  f"{label:<22} {st['recall_mean']:>8.4f} {st['recall_p5']:>8.4f} "
                  f"{st['recall_min_in_group']:>11.4f} {rel:>9}")
            rows.append({
                "config": cfg.name, "block_size": cfg.block_size, "topk": cfg.topk,
                "selected_tokens": cfg.selected_tokens, "layer": args.layer, "mode": mode,
                "seqlen": args.seqlen, "gqa_group": group, **st,
                "recall_vs_oracle": None if mode == "oracle" else st["recall_mean"] / oracle,
            })
        print()
        torch.cuda.empty_cache()

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"CSV written to {args.csv}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"JSON written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
