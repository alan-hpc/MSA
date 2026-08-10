#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""MSA sparse-attention configuration sweep.

Benchmarks the MSA block-sparse prefill pipeline across the configuration space
described by :mod:`fmha_sm100.msa_config` — block size, top-k, forced
sink/local windows and GQA head-aggregation mode — at the attention geometry of
a target model (model-n16 by default).

Every stage is timed with its own real kernel at the configuration's real
shapes:

  ``indexer``  FP4 block scorer (``fp4_indexer_block_scores``)
  ``topk``     ``sparse_topk_select`` — head aggregation + forced windows fused
  ``csr``      ``build_k2q_csr`` — k2q CSR + sparse attention schedule
  ``attn``     ``sparse_atten_func`` — the block-sparse forward itself

plus a dense causal ``fmha_sm100`` row so every sparse configuration has a
same-shape dense baseline to be compared against.

Known measurement caveat, stated up front so the numbers are read correctly:
the shipped FP4 indexer reduces scores over a fixed 128-token KV page, so for
``block_size`` 32 / 64 the ``indexer`` column is the cost of producing
128-granularity scores.  Its MMA work is block-size invariant (the same QK^T is
computed either way); only the epilogue's store volume grows by
``128 / block_size``.  ``topk`` / ``csr`` / ``attn`` are exact at every block
size.  ``--indexer-mode`` controls whether that column is measured or skipped.

Usage
-----
    python benchmarks/bench_msa_configs.py --help
    python benchmarks/bench_msa_configs.py --configs all --seqlens 8192,32768
    python benchmarks/bench_msa_configs.py --configs cfg01,cfg04 --csv out.csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT / "python" / "fmha_sm100" / "cute"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from fmha_sm100 import fmha_sm100, fmha_sm100_plan  # noqa: E402
from fmha_sm100.bench_utils import attention_tflops, bench_gpu_time  # noqa: E402
from fmha_sm100.msa_config import (  # noqa: E402
    BASELINE_CONFIG,
    DEFAULT_MODEL,
    MODEL_SHAPES,
    MsaModelShape,
    iter_configs,
    model_by_name,
)
from fmha_sm100.msa_pipeline import (  # noqa: E402
    MsaSparseAttention,
    causal_end_blocks,
)

#: Mirrors the guard in fp4_indexer_interface: the score tensor's CuTe layout is
#: built from Int32 extents, so it must stay within 2**31 elements.
_SCORE_TENSOR_MAX_ELEMS = 2**31

#: Mirrors the guard in api.py: the dense FMHA's Q/O tensors are Int32-indexed.
_QO_INT32_LIMIT = 2**31

#: B300 (SM103) dense peak, used only to contextualise the dense row.
PEAK_TFLOPS_BF16 = 2250.0


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------


def make_qkv(*, total_q, total_k, head_q, head_kv, dim, dtype, device, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn((total_q, head_q, dim), generator=gen, device=device, dtype=torch.float32)
    k = torch.randn((total_k, head_kv, dim), generator=gen, device=device, dtype=torch.float32)
    v = torch.randn((total_k, head_kv, dim), generator=gen, device=device, dtype=torch.float32)
    return q.to(dtype), k.to(dtype), v.to(dtype)


def _slice_qkv(qkv, *, total_q, total_k, head_q, head_kv, dim, dtype, device, seed):
    """Q/K/V for one measurement, from a captured layer or from noise.

    A captured layer holds ``S`` tokens.  Prefill wants a prefix of them; decode
    wants ``batch`` queries against ``batch * kv_len`` keys, i.e. more KV rows
    than were captured — the capture is tiled to cover the batch, which keeps
    the value distribution real while giving each request its own KV range.
    Decode queries are taken from the *end* of the captured sequence, where a
    real decode step's query would come from.
    """
    if qkv is None:
        return make_qkv(total_q=total_q, total_k=total_k, head_q=head_q, head_kv=head_kv,
                        dim=dim, dtype=dtype, device=device, seed=seed)
    cap_q, cap_k, cap_v = qkv
    q = cap_q[-total_q:] if cap_q.shape[0] >= total_q else cap_q[:total_q]

    def tile(t):
        if t.shape[0] >= total_k:
            return t[:total_k]
        reps = -(-total_k // t.shape[0])
        return t.repeat(reps, 1, 1)[:total_k].contiguous()

    return q.contiguous(), tile(cap_k), tile(cap_v)


def make_block_scores(*, num_heads, num_blocks, total_q, valid_blocks, device, seed):
    """Synthetic block scores with the production ``-inf`` padding contract.

    Only the *distribution* matters for top-k cost, and a uniform draw is the
    adversarial case for the histogram-refinement stages (no early exit).
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    scores = torch.randn((num_heads, num_blocks, total_q), generator=gen,
                         device=device, dtype=torch.float32)
    if valid_blocks < num_blocks:
        scores[:, valid_blocks:, :] = float("-inf")
    return scores.contiguous()


def make_fp4_indexer_inputs(*, total_q, head_q, head_kv, k_lengths, device, seed):
    """Packed-FP4 Q/K plus preordered MMA scales for ``fp4_indexer_block_scores``.

    The indexer only reads bit patterns, so random bytes exercise exactly the
    same code path and cost as real quantised activations.
    """
    from fp4_indexer_interface import (  # noqa: PLC0415 - CuTe stack is heavy
        fp4_indexer_reorder_scales_for_mma_cute,
    )
    from src.sm100.fp4_indexer import normalize_fp4_format  # noqa: PLC0415

    spec = normalize_fp4_format("nvfp4")
    pages_per_batch = [(int(n) + 127) // 128 for n in k_lengths]
    page_count = sum(pages_per_batch)
    gen = torch.Generator(device=device).manual_seed(seed)

    q = torch.randint(0, 256, (total_q, head_q, 64), generator=gen,
                      dtype=torch.uint8, device=device)
    k = torch.randint(0, 256, (page_count, head_kv, 128, 64), generator=gen,
                      dtype=torch.uint8, device=device)
    # e4m3 scale bytes; the reorder kernel only permutes them.
    q_scale = torch.randint(0, 256, (total_q, head_q, spec.scale_groups), generator=gen,
                            dtype=torch.uint8, device=device).view(torch.float8_e4m3fn)
    k_scale = torch.randint(0, 256, (page_count, head_kv, 128, spec.scale_groups), generator=gen,
                            dtype=torch.uint8, device=device).view(torch.float8_e4m3fn)
    q_mma, k_mma = fp4_indexer_reorder_scales_for_mma_cute(q_scale, k_scale, fp4_format="nvfp4")
    torch.cuda.synchronize()

    def _storage(view):
        # The reorder helper returns the *logical* MMA scale view; the indexer
        # validates against the storage layout, which is this fixed permutation
        # (matches `_mma_scale_view_to_storage` in cute/test_fp4_indexer.py).
        return view.permute(5, 2, 4, 0, 1, 3)

    cu_pages = [0]
    for n in pages_per_batch:
        cu_pages.append(cu_pages[-1] + n)
    return {
        "q_fp4": q,
        "k_fp4": k,
        "q_scale": _storage(q_mma),
        "k_scale": _storage(k_mma),
        "cu_page_offsets": torch.tensor(cu_pages, dtype=torch.int32, device=device),
        "kv_indices": torch.arange(page_count, dtype=torch.int32, device=device),
    }


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------


def _host_enqueue_ms(fn, iters=20):
    """Wall time per call with no synchronisation — the host-side dispatch cost.

    Only measured under ``--timing full``; the default single-op mode reports
    the plain event-timed latency and nothing else.
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = (time.perf_counter() - t0) * 1e3 / iters
    torch.cuda.synchronize()
    return elapsed


def _graph_replay_ms(fn, *, warmup=3, iters=20):
    """Pure GPU time via CUDA-graph replay, or ``None`` if capture is impossible.

    Only measured under ``--timing full``.
    """
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        torch.cuda.synchronize()

        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters
    except Exception:  # noqa: BLE001 - capture is best-effort diagnostics
        torch.cuda.synchronize()
        return None


#: Torch appends four lines of generic advice to every CUDA error ("kernel errors
#: might be asynchronously reported...", "consider passing CUDA_LAUNCH_BLOCKING=1",
#: ...).  Repeated once per failing cell it buries the table it is annotating, and
#: it says nothing the first line did not.  Keep the first line.
_EXC_NOISE = (
    "CUDA kernel errors might be asynchronously reported",
    "For debugging consider passing CUDA_LAUNCH_BLOCKING",
    "Compile with `TORCH_USE_CUDA_DSA`",
    "Device-side assertion",
)


def _fmt_exc(exc: BaseException, *, limit: int = 160) -> str:
    """One-line, de-boilerplated rendering of an exception for a table cell.

    The guards in this stack deliberately explain themselves at length -- the
    numbers, the formula, the workarounds -- which is right when one shape
    fails and wrong when thirty do: the table disappears under prose.  The
    first sentence carries the fact; everything after it is remedy, and the
    full text is still in the CSV and JSON.
    """
    lines = [ln.strip() for ln in str(exc).splitlines() if ln.strip()]
    kept = [ln for ln in lines if not any(ln.startswith(n) for n in _EXC_NOISE)]
    msg = " ".join(kept) if kept else (lines[0] if lines else "")
    # A very short first sentence carries no numbers ("CUDA out of memory."),
    # and the size is the whole point there -- take the next one too.
    parts = msg.split(". ")
    head = parts[0].rstrip(".")
    if len(head) < 40 and len(parts) > 1:
        head = f"{head}. {parts[1].rstrip('.')}"
    if head and len(head) <= limit:
        msg = head
    elif len(msg) > limit:
        msg = msg[: limit - 1] + "\u2026"
    return f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__


def timed(fn, *, dry_ms, rep_ms, mode="simple"):
    """Latency of one operator.

    ``mode="simple"`` (the default) reports a single number: the median
    event-timed latency of one call, L2 flushed between iterations.  That is
    what a kernel comparison wants — no CUDA graphs, no host-dispatch probe,
    just the operator.

    ``mode="full"`` additionally captures the CUDA-graph replay time and the
    host enqueue cost, which is useful when diagnosing dispatch-bound stages
    but clutters a straight operator comparison.
    """
    samples = sorted(bench_gpu_time(fn, dry_run_time_ms=dry_ms, repeat_time_ms=rep_ms))
    n = len(samples)
    stat = {"ms": samples[n // 2], "ms_min": samples[0], "iters": n}
    if mode == "full":
        stat["ms_host"] = _host_enqueue_ms(fn)
        stat["ms_gpu"] = _graph_replay_ms(fn)
    return stat


def output_cosine(sparse_out, dense_out):
    """Mean cosine similarity between sparse and dense attention outputs.

    Taken per ``(token, query head)`` vector and averaged: that is the quantity
    a downstream layer actually sees, and unlike an L2 error it is scale-free,
    so heads with different output magnitudes contribute equally.
    """
    a = sparse_out.reshape(-1, sparse_out.shape[-1]).float()
    b = dense_out.reshape(-1, dense_out.shape[-1]).float()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1, eps=1e-8)
    return {
        "cos_mean": float(cos.mean()),
        "cos_p1": float(torch.quantile(cos, 0.01)),
        "cos_min": float(cos.min()),
    }


def exact_block_scores(q, k, *, scale, block_size, head_kv, q_chunk=512):
    """Block scores from the real Q/K, at the configuration's block size.

    ``score(h, blk, q) = max over the block of the causal-masked QK logit`` —
    the same reduction the FP4 indexer performs, but computed exactly and at
    any block size.  The per-KV-head query is the mean of that group's query
    heads, standing in for MSA's proxy-Q projection (which is not trained here).

    Chunked over queries: the full logit matrix never exists at once.
    """
    seqlen, head_q, dim = q.shape
    group = head_q // head_kv
    num_blocks = (seqlen + block_size - 1) // block_size
    pad = num_blocks * block_size - seqlen
    ktt = k.transpose(0, 1).transpose(1, 2).float()             # [Hkv, D, S]
    q_proxy = q.view(seqlen, head_kv, group, dim).mean(dim=2).float()

    scores = torch.empty((head_kv, num_blocks, seqlen), dtype=torch.float32, device=q.device)
    for start in range(0, seqlen, q_chunk):
        stop = min(start + q_chunk, seqlen)
        logits = torch.bmm(q_proxy[start:stop].transpose(0, 1), ktt) * scale   # [Hkv, c, S]
        pos = torch.arange(start, stop, device=q.device).view(-1, 1)
        causal = torch.arange(seqlen, device=q.device).view(1, -1) <= pos
        logits = logits.masked_fill(~causal.unsqueeze(0), float("-inf"))
        if pad:
            logits = torch.nn.functional.pad(logits, (0, pad), value=float("-inf"))
        blk = logits.view(head_kv, stop - start, num_blocks, block_size).amax(dim=-1)
        scores[:, :, start:stop] = blk.permute(0, 2, 1)
        del logits, blk
    return scores


class StageFailure(Exception):
    """A stage that cannot run in this configuration (recorded, not fatal)."""


# ---------------------------------------------------------------------------
# One (config, shape) measurement
# ---------------------------------------------------------------------------


def _auto_chunk_q(model, seqlen_k: int) -> int:
    """Largest query chunk whose block-score tensor stays Int32-addressable.

    Memory is not the binding constraint here and picking the chunk by memory
    alone silently breaks the indexer: at 1M the card can hold a 512K chunk
    (166 GiB) but the score tensor for one is 4 x 8192 x 524288 = 4x over the
    Int32 limit, so top-k, CSR and attention all run and only the indexer fails.
    Derive the chunk from the limit that actually binds.
    """
    k_tiles = (seqlen_k + 127) // 128
    denom = model.num_kv_heads * k_tiles
    if denom <= 0:
        return 0
    return max(1, _SCORE_TENSOR_MAX_ELEMS // denom)


#: build_k2q_csr keeps a 2-byte histogram entry per CSR row in shared memory.
_CSR_MAX_ROWS = 116224
#: The FP4 scale-reorder kernel's own 32-bit indexing limit.
_SCALE_REORDER_MAX_ELEMS = 2**30
#: nvfp4 scale groups per 128-token page row.
_SCALE_GROUPS = 8


def _auto_chunk_batch(cfg, model, seqlen_k, batch):
    """Largest batch group that clears the two limits that scale with batch.

    Both bind on ``batch * blocks``: the CSR builder's shared-memory histogram
    (one 2-byte counter per row, rows = batch * ceil(S/blk)) and the scale
    reorder's 32-bit indexing.  Decode reaches them because it multiplies the
    context by the batch where prefill runs one request at a time -- so the
    context is not what makes them bind, the batch is, and splitting it is the
    direct fix.  The CSR of one request is independent of another's, so this is
    exact, not an approximation.
    """
    blocks = cfg.num_blocks(seqlen_k)
    pages = (seqlen_k + 127) // 128
    sel_heads = cfg.num_selection_heads(model.num_kv_heads)
    by_csr = _CSR_MAX_ROWS // max(1, blocks)
    per_seq_scale = pages * sel_heads * 128 * _SCALE_GROUPS
    by_scale = (_SCALE_REORDER_MAX_ELEMS - 1) // max(1, per_seq_scale)
    return max(1, min(batch, by_csr, by_scale))


def _merge_batch(rows, *, batch, chunk_batch):
    """Fold per-batch-group rows into one.  Stage costs add; the groups are
    independent requests, so the result is what the unsplit call would produce."""
    base = dict(rows[0])
    base.update({"batch": batch, "chunk_batch": chunk_batch, "num_batch_chunks": len(rows)})
    stages, errors = {}, {}
    for r in rows:
        for name, st in r.get("stages", {}).items():
            acc = stages.setdefault(name, {"ms": 0.0, "iters": 0})
            acc["ms"] += st.get("ms", 0.0)
            acc["iters"] = max(acc["iters"], st.get("iters", 0))
        for k, v in r.get("errors", {}).items():
            errors.setdefault(k, v)
    base["stages"], base["errors"] = stages, errors
    ok = not {k: v for k, v in errors.items() if k != "cos"}
    base["supported"] = ok
    total = sum(st["ms"] for st in stages.values()) if ok else None
    base["pipeline_ms"] = base["pipeline_gpu_ms"] = total
    base["attn_ms_only"] = stages.get("attn", {}).get("ms") if ok else None
    for k in ("cos_mean", "cos_p1", "cos_min", "ms_host", "pipeline_host_ms"):
        base.pop(k, None)
    return base


def _merge_chunked(rows, *, seqlen_q, seqlen_k, chunk_q):
    """Fold per-chunk rows into the single row the tables expect.

    Chunking splits the *queries*; every query still scores against every KV
    block it would have seen unchunked, and top-k is per-query, so the selection
    is identical and the stage costs simply add.  Peak memory is what changes,
    which is the whole point — but latency must be reported as the sum, not the
    per-chunk figure, or a chunked row would look artificially fast next to an
    unchunked one.
    """
    base = dict(rows[0])
    base.update({"seqlen_q": seqlen_q, "seqlen_k": seqlen_k,
                 "chunk_q": chunk_q, "num_chunks": len(rows)})
    stages, errors = {}, {}
    for r in rows:
        for name, st in r.get("stages", {}).items():
            acc = stages.setdefault(name, {"ms": 0.0, "iters": 0})
            acc["ms"] += st.get("ms", 0.0)
            acc["iters"] = max(acc["iters"], st.get("iters", 0))
        for k, v in r.get("errors", {}).items():
            errors.setdefault(k, v)
    base["stages"], base["errors"] = stages, errors
    ok = not {k: v for k, v in errors.items() if k != "cos"}
    base["supported"] = ok
    total = sum(st["ms"] for st in stages.values()) if ok else None
    base["pipeline_ms"] = base["pipeline_gpu_ms"] = total
    base["attn_ms_only"] = stages.get("attn", {}).get("ms") if ok else None
    for k in ("cos_mean", "cos_p1", "cos_min", "ms_host", "pipeline_host_ms"):
        base.pop(k, None)
    return base


def bench_config(
    cfg,
    *,
    model: MsaModelShape,
    batch,
    seqlen_q,
    seqlen_k,
    dtype,
    device,
    dry_ms,
    rep_ms,
    indexer_mode,
    seed,
    timing="simple",
    dense_out=None,
    qkv=None,
):
    head_q = model.num_qo_heads
    head_kv = model.num_kv_heads
    dim = model.bench_head_dim()
    total_q = batch * seqlen_q
    total_k = batch * seqlen_k
    q_lengths = [seqlen_q] * batch
    k_lengths = [seqlen_k] * batch
    softmax_scale = 1.0 / math.sqrt(dim)

    runner = MsaSparseAttention(
        cfg, num_qo_heads=head_q, num_kv_heads=head_kv, head_dim=dim, causal=True
    )
    num_blocks = cfg.num_blocks(seqlen_k)
    total_rows = num_blocks * batch

    sel_heads = cfg.num_selection_heads(head_kv)
    row = {
        **cfg.to_dict(),
        "selection_heads": sel_heads,
        "batch": batch,
        "seqlen_q": seqlen_q,
        "seqlen_k": seqlen_k,
        "head_q": head_q,
        "head_kv": head_kv,
        "head_dim": dim,
        "num_blocks": num_blocks,
        "scorer_heads": cfg.num_scorer_heads(head_q, head_kv),
        "sparsity": cfg.sparsity(seqlen_k),
        "stages": {},
        "errors": {},
    }

    cu_seqlens_q = torch.tensor([0] + list(torch.tensor(q_lengths).cumsum(0)),
                                dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0] + list(torch.tensor(k_lengths).cumsum(0)),
                                dtype=torch.int32, device=device)

    q, k, v = _slice_qkv(qkv, total_q=total_q, total_k=total_k, head_q=head_q,
                         head_kv=head_kv, dim=dim, dtype=dtype, device=device, seed=seed)

    # ---- stage: block scores (FP4 indexer) --------------------------------
    if indexer_mode != "skip":
        try:
            from fp4_indexer_interface import fp4_indexer_block_scores  # noqa: PLC0415

            # Pad a short query up to the indexer's 128-row MMA tile.  A decode
            # step has one query per request and lands on the kernel's predicated
            # residue path, which measures 3.5-3.7x slower than the full-tile
            # path for the same work -- 11% of streaming bandwidth against 43%.
            # The extra rows are not free work: the causal convention puts the
            # last query at the end of the context, so the padded rows sit just
            # before it and their scores are discarded.  Verified bitwise: the
            # padded run's last row per request reproduces the unpadded run's
            # only row exactly, at 128K, 256K and 1M.
            _idx_q_len = seqlen_q
            if seqlen_q < _INDEXER_Q_TILE and int(seqlen_q) == 1:
                _idx_q_len = _INDEXER_Q_TILE
            _idx_total_q = batch * _idx_q_len
            _idx_cu_q = (cu_seqlens_q if _idx_q_len == seqlen_q else
                         torch.arange(0, (batch + 1) * _idx_q_len, _idx_q_len,
                                      dtype=torch.int32, device=device))
            idx_in = make_fp4_indexer_inputs(
                total_q=_idx_total_q, head_q=row["scorer_heads"], head_kv=sel_heads,
                k_lengths=k_lengths, device=device, seed=seed + 7,
            )
            row["indexer_q_pad"] = _idx_q_len if _idx_q_len != seqlen_q else None

            def run_indexer():
                return fp4_indexer_block_scores(
                    idx_in["q_fp4"], idx_in["k_fp4"], idx_in["q_scale"], idx_in["k_scale"],
                    _idx_cu_q, cu_seqlens_k, idx_in["cu_page_offsets"],
                    max_seqlen_q=_idx_q_len, max_seqlen_k=seqlen_k,
                    kv_indices=idx_in["kv_indices"], fp4_format="nvfp4", causal=True,
                    scale_layout="preordered_mma",
                )

            run_indexer()
            torch.cuda.synchronize()
            row["stages"]["indexer"] = timed(run_indexer, dry_ms=dry_ms, rep_ms=rep_ms, mode=timing)
            del idx_in
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001 - one stage failing must not kill the sweep
            row["errors"]["indexer"] = _fmt_exc(exc)

    # ---- stage: top-k selection -------------------------------------------
    # exact_block_scores assumes one self-attending sequence (q and k the same
    # rows); decode has `batch` queries against `batch * kv_len` keys, so the
    # cosine path is prefill-only.
    if dense_out is not None and not (batch == 1 and seqlen_q == seqlen_k):
        row["errors"]["cos"] = (
            "cosine is only computed for single-sequence prefill "
            f"(batch={batch}, q_len={seqlen_q}, kv_len={seqlen_k})"
        )
        dense_out = None
    if dense_out is not None:
        # cos is only meaningful if the selection comes from these very Q/K.
        scores = exact_block_scores(
            q, k, scale=softmax_scale, block_size=cfg.block_size,
            head_kv=row["scorer_heads"], q_chunk=512,
        )
    else:
        scores = make_block_scores(
            num_heads=row["scorer_heads"], num_blocks=num_blocks, total_q=total_q,
            valid_blocks=num_blocks, device=device, seed=seed + 11,
        )
    ceb = causal_end_blocks(q_lengths, k_lengths, cfg.block_size, device=device)
    q2k_buf = torch.empty((sel_heads, total_q, cfg.topk), dtype=torch.int32, device=device)

    def run_topk():
        return runner.select(
            scores, num_valid_blocks=num_blocks, causal_end_block=ceb, out=q2k_buf
        )

    try:
        q2k = run_topk()
        torch.cuda.synchronize()
        row["stages"]["topk"] = timed(run_topk, dry_ms=dry_ms, rep_ms=rep_ms, mode=timing)
    except Exception as exc:  # noqa: BLE001
        row["errors"]["topk"] = _fmt_exc(exc)
        row["supported"] = False
        return row

    # ---- stage: CSR + schedule --------------------------------------------
    def run_csr():
        return runner.build_csr(
            q2k, cu_seqlens_q, cu_seqlens_k,
            total_k=total_k, max_seqlen_q=seqlen_q, max_seqlen_k=seqlen_k,
            total_rows=total_rows,
        )

    try:
        row_ptr, q_indices, schedule = run_csr()
        torch.cuda.synchronize()
        row["stages"]["csr"] = timed(run_csr, dry_ms=dry_ms, rep_ms=rep_ms, mode=timing)
    except Exception as exc:  # noqa: BLE001
        row["errors"]["csr"] = _fmt_exc(exc)
        row["supported"] = False
        return row

    # ---- stage: sparse attention ------------------------------------------
    def run_attn():
        return runner.attend(
            q, k, v, row_ptr, q_indices,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=seqlen_q, max_seqlen_k=seqlen_k,
            schedule=schedule, softmax_scale=softmax_scale,
        )

    try:
        out = run_attn()
        torch.cuda.synchronize()
        row["stages"]["attn"] = timed(run_attn, dry_ms=dry_ms, rep_ms=rep_ms, mode=timing)
        if dense_out is not None:
            row.update(output_cosine(out, dense_out))
        del out
    except Exception as exc:  # noqa: BLE001
        row["errors"]["attn"] = _fmt_exc(exc)
        row["supported"] = False
        return row

    # ---- derived ------------------------------------------------------------
    # A row whose indexer (or any other stage) could not run is NOT comparable to
    # a complete one: summing the stages that did run would rank it as if the
    # missing work were free.  Totals are only computed for complete rows.
    fatal = {k: v for k, v in row["errors"].items() if k != "cos"}
    if fatal:
        row["supported"] = False
        return row
    row["supported"] = True
    stages = row["stages"]
    row["pipeline_ms"] = sum(s["ms"] for s in stages.values())
    row["selection_ms"] = sum(v["ms"] for kk, v in stages.items() if kk != "attn")
    # GPU-only totals fall back to the eager number for any stage that could not
    # be graph-captured, so the sum is never silently optimistic.
    row["pipeline_gpu_ms"] = sum((s.get("ms_gpu") or s["ms"]) for s in stages.values())
    row["attn_ms_only"] = stages["attn"]["ms"] if "attn" in stages else None
    row["pipeline_host_ms"] = sum(s.get("ms_host") or 0.0 for s in stages.values())
    # Useful work: each query touches at most topk*block_size KV tokens,
    # capped by its own causal bound.
    attended = min(cfg.selected_tokens, seqlen_k)
    row["attn_tflops"] = attention_tflops(
        [seqlen_q] * batch, [attended] * batch, dim, dim, head_q, False, stages["attn"]["ms"]
    )
    return row



def _load_fa4(repo_path):
    """FlashAttention-4's CuTe entry points, or ``None`` if unavailable.

    FA4 ships in the flash-attention repository under ``flash_attn/cute`` and is
    CuTe-DSL based, so it needs no nvcc build — but importing it goes through
    ``flash_attn/__init__.py``, which pulls in FA2's compiled CUDA extension.
    That extension is not built here, so it is stubbed out; nothing in the FA4
    path touches it.
    """
    import types

    if repo_path and repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))
    try:
        from flash_attn.cute.interface import flash_attn_varlen_func
    except Exception as exc:  # noqa: BLE001 - optional cross-check backend
        print(f"NOTE      : FlashAttention-4 unavailable ({type(exc).__name__}: {exc})")
        return None
    return flash_attn_varlen_func


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


PAGE_FOR_FAST_DECODE = 128

#: The FP4 indexer's MMA tile is 128 query rows.
_INDEXER_Q_TILE = 128


def _decode_fast_subprocess(python, *, model, batch, seqlen_k, block_size, topk,
                            head_mode, dry_ms, rep_ms, seed, timeout=1800):
    """Time the real paged sparse decode kernel in another interpreter.

    Two reasons it cannot run here.  The paged decode kernel only builds on
    cutlass-dsl 4.5.x while this sweep runs on 4.6+ (where FA4 is 3.3x faster),
    and the decode path wants an fp8 paged KV cache rather than the flat bf16
    one the prefill kernel takes.  Returns (ms, worst_cos, reason).

    Inputs and reference both come from the repo's own test suite rather than
    being rebuilt here.  Five separate wrong-cosine incidents on this path all
    traced to inputs I wrote myself, never to a kernel; the most recent passed
    a single query row to a kernel that reads a packed tile of
    ``seqlen_q * qhead_per_kv == 128`` rows, so it attended over uninitialised
    memory and scored cosine ~0.  Starting from the validated builder and
    changing exactly one thing -- the page table -- is what works.
    """
    group = model.num_qo_heads // model.num_kv_heads
    if PAGE_FOR_FAST_DECODE % block_size and block_size % PAGE_FOR_FAST_DECODE:
        return None, None, f"page_size ({block_size}) must equal n_block_size (128)"
    # head_mode=keep gives every KV head its own selection, and a page table is
    # per request, not per head.  Rather than skip the config, make each (request,
    # KV head) pair its own request: batch becomes batch*head_kv with head_kv=1,
    # which is exactly the shape a per-head selection describes.  It reads up to
    # head_kv times more KV than a shared selection does -- that is what keep
    # costs, not an artifact of the mapping.
    heads_as_batch = head_mode == "keep"
    if 128 % group:
        return None, None, f"qhead_per_kv ({group}) must divide the 128-row packed-q tile"

    cfg = json.dumps(dict(
        batch=batch * model.num_kv_heads if heads_as_batch else batch,
        head_kv=1 if heads_as_batch else model.num_kv_heads,
        heads_as_batch=heads_as_batch, seqlen_k=seqlen_k, topk=topk, seed=seed,
        group=group, dim=model.bench_head_dim(), dry=dry_ms, rep=rep_ms))
    body = r"""
import sys, json, types, torch
CFG = json.loads(sys.argv[1])
sys.path.insert(0, "python"); sys.path.insert(0, "python/fmha_sm100/cute")

# test_sparse_atten imports pytest at module scope purely for its decorators.
_p = types.ModuleType("pytest"); _p.skip = lambda *a, **k: None
class _M:
    def __getattr__(self, n):
        return lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
_p.mark = _M(); _p.approx = lambda x, **k: x; sys.modules["pytest"] = _p
import test_sparse_atten as T

# The builder binds its defaults from the module constants at def time, so
# assigning T.DECODE_* here would be silently ignored -- every geometry has to
# be passed as a keyword.  Getting this wrong is not loud: the builder just
# hands back the default qhead_per_kv=16 shape and the kernel then rejects the
# seqlen_q that was correct for the geometry actually asked for.
# The packed-q tile is 128 rows: seqlen_q * qhead_per_kv must fill it exactly.
PAGE = T.BLK_KV; B = CFG["batch"]; TOPK = CFG["topk"]
SQ = 128 // CFG["group"]

inp = T._build_decode_paged_dense_inputs(
    kv_tokens=CFG["seqlen_k"], batch=B, seqlen_q=SQ, head_kv=CFG["head_kv"],
    qhead_per_kv=CFG["group"], dim=CFG["dim"])
HQ = inp["q"].shape[1]; HKV = inp["k_paged"].shape[1]; D = inp["q"].shape[2]
npage = CFG["seqlen_k"] // PAGE
if TOPK >= npage:
    print("RESULT " + json.dumps({"skip": "topk covers every page; not sparse"})); raise SystemExit

g = torch.Generator().manual_seed(CFG["seed"])
sel = torch.stack([torch.sort(torch.randperm(npage, generator=g)[:TOPK]).values
                   for _ in range(B)]).cuda()

# The one and only change from the validated inputs: keep just the selected
# pages.  gather maps selection -> physical page id; computing that id by hand
# is what broke this path once already.
sp = dict(inp)
sp["page_table"] = torch.gather(inp["page_table"], 1, sel).contiguous().to(torch.int32)
sp["kv_tokens"] = TOPK * PAGE; sp["max_seqlen_k"] = TOPK * PAGE
sp["seqused_k"] = torch.full((B,), TOPK * PAGE, dtype=torch.int32, device="cuda")

fn = T._get_sparse_decode_atten_func_for_benchmark()
fn.plan(page_table=sp["page_table"], seqused_k=sp["seqused_k"], seqlen_q=SQ,
        max_seqlen_k=sp["max_seqlen_k"], num_qo_heads=HQ, num_kv_heads=HKV, head_dim=D)
run = lambda: fn.run(sp["q"], sp["k_paged"], sp["v_paged"],
                     softmax_scale=sp["softmax_scale"])

out = run(); torch.cuda.synchronize()
out = out[0] if isinstance(out, (tuple, list)) else out
ref, _ = T._decode_paged_dense_reference(sp)
cos = float(torch.nn.functional.cosine_similarity(out.float(), ref.float(), dim=-1).min())

flush = torch.empty(int(256e6), dtype=torch.int8, device="cuda")
for _ in range(max(1, CFG["dry"])): flush.zero_(); run()
torch.cuda.synchronize(); ts = []
for _ in range(max(1, CFG["rep"])):
    flush.zero_()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record(); run(); e.record(); torch.cuda.synchronize()
    ts.append(s.elapsed_time(e))
ts.sort()
print("RESULT " + json.dumps({"ms": ts[len(ts)//2], "cos": cos}))
"""
    import subprocess
    try:
        r = subprocess.run([python, "-c", body, cfg], capture_output=True, text=True,
                           timeout=timeout, cwd=str(REPO_ROOT))
    except subprocess.TimeoutExpired:
        return None, None, "timed out"
    line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT ")), None)
    if line is None:
        return None, None, _fmt_exc(r.stderr or r.stdout or "no output")
    d = json.loads(line[len("RESULT "):])
    if "skip" in d:
        return None, None, d["skip"]
    return d["ms"], d["cos"], None


def _fa4_subprocess(python, fa4_path, *, model, batch, seqlen_q, seqlen_k,
                    dry_ms, rep_ms, seed, timeout=1800, force_batch_chunk=None):
    """Measure FA4 in another interpreter, so its cutlass version is its own.

    The MSA decode kernels need cutlass-dsl 4.5.x and FA4 is 3.3x slower there;
    4.6+ reverses it.  One process cannot hold both, and FA4 is only the
    reference denominator -- so run it where it is fastest and let the sweep run
    where it must.  Returns the median latency in ms, or None with a reason.

    Decode also has to split the batch.  FA4 takes a flat unpaged KV, so it
    materialises ``2 * batch * seqlen_k * head_kv * dim`` elements -- 69 GB at
    batch 32 and 1M, which OOMs on any single device.  Requests in a batch are
    independent and the device runs the groups one after another either way, so
    timing each group and adding the results is what the unsplit call would have
    cost, the same argument that makes _merge_batch exact.  Groups are sized
    from free memory at run time, and a shape that already fits still takes the
    single-allocation path unchanged.
    """
    import json as _json
    import subprocess
    import textwrap

    src = textwrap.dedent(f"""
        import inspect, json, sys, types, torch
        sys.path.insert(0, {fa4_path!r})
        sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))
        from flash_attn.cute.interface import flash_attn_varlen_func as fa4
        B, SQ, SK = {batch}, {seqlen_q}, {seqlen_k}
        HQ, HKV, D = {model.num_qo_heads}, {model.num_kv_heads}, {model.bench_head_dim()}
        g = torch.Generator(device="cuda").manual_seed({seed})
        flush = torch.empty(int(256e6), dtype=torch.int8, device="cuda")

        # k and v together, per request in the group.  Leave the rest of free
        # memory to the output, FA4's workspace and the flush buffer.
        per_req = 2 * SK * HKV * D * 2
        free = torch.cuda.mem_get_info()[0]
        bc = max(1, min(B, int(free * 0.35) // max(1, per_req)))
        _forced = {force_batch_chunk!r}
        if _forced: bc = max(1, min(B, int(_forced)))
        groups = ([bc] * (B // bc)) + ([B % bc] if B % bc else [])

        a, b = torch.cuda.Event(True), torch.cuda.Event(True)

        def time_group(n):
            q = torch.randn((n*SQ, HQ, D), generator=g, device="cuda").to(torch.bfloat16)
            k = torch.randn((n*SK, HKV, D), generator=g, device="cuda").to(torch.bfloat16)
            v = torch.randn((n*SK, HKV, D), generator=g, device="cuda").to(torch.bfloat16)
            cq = torch.arange(0, (n+1)*SQ, SQ, dtype=torch.int32, device="cuda")
            ck = torch.arange(0, (n+1)*SK, SK, dtype=torch.int32, device="cuda")
            # num_splits would be the knob that restores occupancy for a small
            # group, but passing it makes FA4 fail to compile on this cutlass
            # version ("make every assignment to nheads_in_l2 produce the same
            # type"), and it takes the unsplit path down with it.  So a split
            # measurement here runs at whatever occupancy the group gives.
            run = lambda: fa4(q, k, v, cu_seqlens_q=cq, cu_seqlens_k=ck,
                              max_seqlen_q=SQ, max_seqlen_k=SK, causal=True)
            run(); torch.cuda.synchronize()
            flush.zero_(); a.record(); run(); b.record(); torch.cuda.synchronize()
            one = max(a.elapsed_time(b), 1e-3)
            for _ in range(max(1, int({dry_ms}/one))): flush.zero_(); run()
            torch.cuda.synchronize()
            s = []
            for _ in range(max(3, int({rep_ms}/one))):
                flush.zero_(); a.record(); run(); b.record()
                torch.cuda.synchronize(); s.append(a.elapsed_time(b))
            s.sort()
            del q, k, v, cq, ck
            torch.cuda.empty_cache()
            return s[len(s)//2]

        # One timing per distinct group size, not per group: equal groups cost
        # the same and re-timing them would only add noise.
        seen = {{}}
        total = 0.0
        for n in groups:
            if n not in seen: seen[n] = time_group(n)
            total += seen[n]
        print("RESULT " + json.dumps({{"ms": total, "batch_chunk": bc,
                                     "num_batch_chunks": len(groups)}}))
    """)
    try:
        r = subprocess.run([python, "-c", src], capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "timed out"
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            d = _json.loads(line[7:])
            if d.get("num_batch_chunks", 1) > 1:
                # Splitting keeps the shape runnable but not measurable.  A
                # decode group of a few requests leaves the device mostly idle,
                # so the groups add up to far more than the same work unsplit:
                # measured 4.11x at 256K and 4.01x at 512K against their own
                # unsplit runs.  num_splits would be the fix but it fails to
                # compile on this cutlass version.  Reporting the sum would
                # inflate every ratio built on this denominator, so decline it
                # -- the in-repo dense row is a valid denominator at this shape.
                return None, (f"needs {d['num_batch_chunks']} batch groups to fit memory; "
                              f"a split decode measures ~4x its unsplit cost, so the "
                              f"number would not be comparable")
            return d["ms"], None
    # Keep enough of the traceback to identify the failure.  A single trailing
    # line routinely cuts the message mid-sentence and sends the reader after
    # the wrong cause.
    tail = [l for l in (r.stderr or r.stdout).strip().splitlines() if l.strip()]
    if not tail:
        return None, f"exit {r.returncode}"
    err = [l for l in tail if l.startswith(("Traceback", "  File")) is False]
    return None, " | ".join(err[-3:])[:400]


def _fa4_pick_num_splits(fa4_fn, run, *, batch, seqlen_q, seqlen_k, head_kv,
                         dry_ms, rep_ms):
    """Choose FA4's KV-split factor, or ``None`` when it is not a knob.

    Only worth doing for decode-shaped work; for prefill there is already ample
    parallelism and ``num_splits=1`` is what FA4 wants.
    """
    import inspect

    try:
        if "num_splits" not in inspect.signature(fa4_fn).parameters:
            return None
    except (TypeError, ValueError):
        return None
    if seqlen_q > 8:
        return 1

    try:
        sms = torch.cuda.get_device_properties(0).multi_processor_count
    except Exception:  # noqa: BLE001
        sms = 148
    ctas = max(1, batch * head_kv)
    kv_blocks = max(1, _ceil_div(seqlen_k, 128))
    ceiling = min(kv_blocks, max(1, _ceil_div(sms * 4, ctas)))

    best, best_ms = 1, None
    cand, n = [], 1
    while n <= ceiling:
        cand.append(n)
        n *= 2
    for n in cand:
        try:
            run(n)
            torch.cuda.synchronize()
            ms = timed(lambda n=n: run(n), dry_ms=dry_ms, rep_ms=rep_ms,
                       mode="simple")["ms"]
        except Exception:  # noqa: BLE001 - an unsupported split is not fatal
            torch.cuda.synchronize()
            continue
        if best_ms is None or ms < best_ms:
            best, best_ms = n, ms
    return best


def bench_fa4(fa4_fn, *, model, batch, seqlen_q, seqlen_k, dtype, device,
              dry_ms, rep_ms, seed, timing="simple", want_output=False, qkv=None):
    """FlashAttention-4 on the same shapes — an independent dense reference.

    Comparing MSA against only its own dense kernel would leave open whether the
    speedups come from sparsity or from a weak denominator.  FA4 is the strongest
    dense attention available on this box, so it is the honest baseline.
    """
    head_q, head_kv = model.num_qo_heads, model.num_kv_heads
    dim = model.bench_head_dim()
    total_q, total_k = batch * seqlen_q, batch * seqlen_k

    q, k, v = _slice_qkv(qkv, total_q=total_q, total_k=total_k, head_q=head_q,
                         head_kv=head_kv, dim=dim, dtype=dtype, device=device, seed=seed)
    cu_q = torch.arange(0, (batch + 1) * seqlen_q, seqlen_q, dtype=torch.int32, device=device)
    cu_k = torch.arange(0, (batch + 1) * seqlen_k, seqlen_k, dtype=torch.int32, device=device)

    def run(num_splits=None):
        kw = {} if num_splits is None else {"num_splits": num_splits}
        return fa4_fn(q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                      max_seqlen_q=seqlen_q, max_seqlen_k=seqlen_k, causal=True, **kw)

    # At q_len=1 the work decomposes into batch*head_kv CTAs — 128 of them at
    # B=32/Hkv=4, under one wave on this GPU — so without a KV split FA4 leaves
    # most of the machine idle and loses to a kernel it beats everywhere else.
    # Splitting the KV dimension is what FA2/FA3/FA4 do for decode; pick the
    # split the same way a serving stack would, by measuring.
    split = _fa4_pick_num_splits(fa4_fn, run, batch=batch, seqlen_q=seqlen_q,
                                 seqlen_k=seqlen_k, head_kv=head_kv,
                                 dry_ms=dry_ms, rep_ms=rep_ms)
    out = run(split)
    torch.cuda.synchronize()
    stat = timed(lambda: run(split), dry_ms=dry_ms, rep_ms=rep_ms, mode=timing)
    fa4_out = None
    if want_output:
        fa4_out = (out[0] if isinstance(out, (tuple, list)) else out).reshape(total_q, head_q, dim)
    del out
    tflops = attention_tflops([seqlen_q] * batch, [seqlen_k] * batch, dim, dim,
                              head_q, True, stat["ms"])
    return {
        "name": "dense-fa4",
        "label": ("FlashAttention-4 causal (reference)" if split in (None, 1)
                  else f"FlashAttention-4 causal (reference, num_splits={split})"),
        "fa4_num_splits": split,
        "batch": batch, "seqlen_q": seqlen_q, "seqlen_k": seqlen_k,
        "head_q": head_q, "head_kv": head_kv, "head_dim": dim,
        "stages": {"attn": stat},
        "pipeline_ms": stat["ms"],
        "pipeline_gpu_ms": stat.get("ms_gpu") or stat["ms"],
        "pipeline_host_ms": stat.get("ms_host"),
        "attn_ms_only": stat["ms"],
        "supported": True, "cos_mean": 1.0, "cos_p1": 1.0, "cos_min": 1.0,
        "attn_tflops": tflops, "mfu": tflops / PEAK_TFLOPS_BF16, "errors": {},
    }, fa4_out


def bench_dense(*, model, batch, seqlen_q, seqlen_k, dtype, device, dry_ms, rep_ms, seed,
                timing="simple", want_output=False, qkv=None):
    """Same-shape dense causal ``fmha_sm100`` reference."""
    head_q = model.num_qo_heads
    head_kv = model.num_kv_heads
    dim = model.bench_head_dim()
    total_q = batch * seqlen_q
    total_k = batch * seqlen_k

    q, k, v = _slice_qkv(qkv, total_q=total_q, total_k=total_k, head_q=head_q,
                         head_kv=head_kv, dim=dim, dtype=dtype, device=device, seed=seed)
    qo_lens = torch.full((batch,), seqlen_q, dtype=torch.int32)
    kv_lens = torch.full((batch,), seqlen_k, dtype=torch.int32)
    qo_offset = torch.full((batch,), seqlen_k - seqlen_q, dtype=torch.int32)

    plan = fmha_sm100_plan(qo_lens, kv_lens, head_q, num_kv_heads=head_kv,
                           qo_offset=qo_offset, causal=True)

    def run_dense():
        return fmha_sm100(q, k, v, plan_info=plan)

    out = run_dense()
    torch.cuda.synchronize()
    stat = timed(run_dense, dry_ms=dry_ms, rep_ms=rep_ms, mode=timing)
    dense_out = (out[0] if isinstance(out, tuple) else out) if want_output else None
    del out
    tflops = attention_tflops([seqlen_q] * batch, [seqlen_k] * batch, dim, dim,
                              head_q, True, stat["ms"])
    return {
        "name": "dense",
        "label": "dense causal FMHA (reference)",
        "batch": batch,
        "seqlen_q": seqlen_q,
        "seqlen_k": seqlen_k,
        "head_q": head_q,
        "head_kv": head_kv,
        "head_dim": dim,
        "stages": {"attn": stat},
        "pipeline_ms": stat["ms"],
        "pipeline_gpu_ms": stat.get("ms_gpu") or stat["ms"],
        "pipeline_host_ms": stat.get("ms_host"),
        "attn_ms_only": stat["ms"],
        "supported": True,
        "cos_mean": 1.0,
        "cos_p1": 1.0,
        "cos_min": 1.0,
        "attn_tflops": tflops,
        "mfu": tflops / PEAK_TFLOPS_BF16,
        "errors": {},
    }, dense_out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_COLUMNS = [
    ("cfg", 8, "<"),
    ("block", 6, ">"),
    ("topk", 5, ">"),
    ("head", 5, ">"),
    ("sel_tok", 8, ">"),
    ("kv_len", 8, ">"),
    ("blocks", 7, ">"),
    ("indexer", 9, ">"),
    ("topk_ms", 9, ">"),
    ("csr_ms", 9, ">"),
    ("attn_ms", 9, ">"),
    ("total_ms", 9, ">"),
    ("gpu_ms", 9, ">"),
    ("host_ms", 9, ">"),
    ("vs_dense", 9, ">"),
]


def _fmt_ms(row, stage):
    stat = row.get("stages", {}).get(stage)
    return "-" if stat is None else f"{stat['ms']:.3f}"


def print_header():
    print("  ".join(f"{name:{align}{width}}" for name, width, align in _COLUMNS))
    print("-" * (sum(w for _, w, _ in _COLUMNS) + 2 * (len(_COLUMNS) - 1)))


def print_row(row, dense_ms=None):
    speedup = "-"
    if row.get("errors"):
        speedup = "N/A"
    elif dense_ms and row.get("pipeline_gpu_ms"):
        speedup = f"{dense_ms / row['pipeline_gpu_ms']:.2f}x"
    values = [
        row.get("name", "?"),
        str(row.get("block_size", "-")),
        str(row.get("topk", "-")),
        str(row.get("head_mode", "-")),
        str(row.get("selected_tokens", "-")),
        str(row.get("seqlen_k", "-")),
        str(row.get("num_blocks", "-")),
        _fmt_ms(row, "indexer"),
        _fmt_ms(row, "topk"),
        _fmt_ms(row, "csr"),
        _fmt_ms(row, "attn"),
        f"{row['pipeline_ms']:.3f}" if row.get("pipeline_ms") else "-",
        f"{row['pipeline_gpu_ms']:.3f}" if row.get("pipeline_gpu_ms") else "-",
        f"{row['pipeline_host_ms']:.3f}" if row.get("pipeline_host_ms") else "-",
        speedup,
    ]
    print("  ".join(f"{v:{a}{w}}" for v, (_, w, a) in zip(values, _COLUMNS)))
    for stage, err in row.get("errors", {}).items():
        print(f"      !! {stage}: {err}")


def write_csv(rows, path, *, quiet=False):
    import csv

    fields = [
        "name", "label", "block_size", "topk", "head_mode",
        "force_init_tokens", "force_end_tokens", "force_init_blocks", "force_end_blocks",
        "selected_tokens", "batch", "seqlen_q", "seqlen_k", "head_q", "head_kv", "head_dim",
        "num_blocks", "scorer_heads", "selection_heads", "sparsity",
        "indexer_ms", "topk_ms", "csr_ms", "attn_ms",
        "indexer_gpu_ms", "topk_gpu_ms", "csr_gpu_ms", "attn_gpu_ms",
        "indexer_host_ms", "topk_host_ms", "csr_host_ms", "attn_host_ms",
        "selection_ms", "pipeline_ms", "pipeline_gpu_ms", "pipeline_host_ms", "attn_ms_only",
        "attn_tflops", "cos_mean", "cos_p1", "cos_min", "supported", "chunk_q",
        "num_chunks", "chunk_batch", "num_batch_chunks", "errors",
    ]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            stages = row.get("stages", {})
            for stage in ("indexer", "topk", "csr", "attn"):
                stat = stages.get(stage)
                flat[f"{stage}_ms"] = stat["ms"] if stat else ""
                # ms_gpu / ms_host only exist under --timing full.
                flat[f"{stage}_gpu_ms"] = (stat.get("ms_gpu") or "") if stat else ""
                flat[f"{stage}_host_ms"] = (stat.get("ms_host") or "") if stat else ""
            flat["errors"] = json.dumps(row.get("errors", {})) if row.get("errors") else ""
            writer.writerow(flat)
    if not quiet:
        print(f"\nCSV written to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL.name,
                   help=f"target attention geometry: one of {sorted(set(MODEL_SHAPES))} "
                        f"(default: {DEFAULT_MODEL.name})")
    p.add_argument("--configs", default="all",
                   help="'all' (baseline + matrix), 'matrix', or a comma-separated list of "
                        "names/slugs/specs (default: all)")
    p.add_argument("--seqlens", default="8192,32768",
                   help="comma-separated KV lengths to sweep (default: 8192,32768)")
    p.add_argument("--batch", type=int, default=1, help="requests per measurement (default: 1)")
    p.add_argument("--qlen", type=int, default=None,
                   help="query length; defaults to the KV length (full prefill), or 1 in decode")
    p.add_argument("--mode", default="prefill", choices=("prefill", "decode"),
                   help="'prefill' sweeps q_len == kv_len; 'decode' sets q_len=1 per request, "
                        "which is where long-context serving actually spends its time and where "
                        "the indexer's Int32 score-tensor cap stops binding (total_q collapses "
                        "from the context length to the batch size)")
    p.add_argument("--dtype", default="bf16", choices=("bf16",),
                   help="Q/K/V dtype (the sparse forward supports bf16 and fp8; bf16 here)")
    p.add_argument("--indexer-mode", default="measure", choices=("measure", "skip"),
                   help="whether to time the FP4 block scorer (default: measure)")
    p.add_argument("--timing", default="simple", choices=("simple", "full"),
                   help="'simple' times each operator with CUDA events only (default); "
                        "'full' adds CUDA-graph and host-dispatch views")
    p.add_argument("--qk-source", default="random", choices=("random", "model"),
                   help="'random' (default) fills Q/K/V with noise — fine for latency, but the "
                        "resulting attention is near-uniform so the cosine table is a floor "
                        "rather than a measurement. 'model' reads one real attention layer's "
                        "projections from a checkpoint instead")
    p.add_argument("--qk-model", default="<checkpoint>",
                   help="checkpoint to source real Q/K/V from (--qk-source model)")
    p.add_argument("--qk-layer", type=int, default=0,
                   help="which attention layer to read (--qk-source model)")
    p.add_argument("--cos-max-seqlen", type=int, default=32768,
                   help="only compute the cosine column at or below this KV length; the exact "
                        "block scores it needs cost a dense attention's worth of FLOPs per "
                        "configuration (default 32768)")
    p.add_argument("--decode-python", default="",
                   help="interpreter to run decode_attend in.  The paged decode "
                        "kernel only builds on cutlass-dsl 4.5.x, so the fast "
                        "decode path is measured there and reported alongside "
                        "the in-process rows.")
    p.add_argument("--fa4-python", default="",
                   help="interpreter to measure FA4 in.  Use when the sweep's "
                        "cutlass-dsl version differs from the one FA4 is fastest "
                        "on -- FA4 is only the denominator, so it should run "
                        "where it performs, not where the kernels must.")
    p.add_argument("--chunk-q", type=int, default=0,
                   help="split prefill queries into chunks of this many tokens. "
                        "0 disables; -1 picks the largest chunk the indexer's Int32 "
                        "score tensor allows. Chunking does not change the result -- "
                        "top-k is per-query -- only the peak footprint.")
    p.add_argument("--cos", action="store_true",
                   help="also report cosine similarity of each configuration's attention "
                        "output against dense. Selections are then built from the real Q/K "
                        "(exact block scores), which costs a dense-attention's worth of FLOPs "
                        "per configuration")
    p.add_argument("--dry-ms", type=int, default=50, help="warmup milliseconds per stage")
    p.add_argument("--rep-ms", type=int, default=200, help="measurement milliseconds per stage")
    p.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--csv", default=None, help="write results to this CSV path")
    p.add_argument("--json", default=None, help="write raw results to this JSON path")
    p.add_argument("--no-dense", action="store_true", help="skip the dense reference row")
    p.add_argument("--fa4-path", default="../flash-attention",
                   help="checkout of Dao-AILab/flash-attention; its flash_attn/cute is FA4")
    p.add_argument("--no-fa4", action="store_true",
                   help="skip the FlashAttention-4 reference row")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return 0
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda", args.gpu)
    dtype = torch.bfloat16

    model = model_by_name(args.model)
    seqlens = [int(s) for s in args.seqlens.split(",") if s.strip()]

    fa4_fn = None if args.no_fa4 else _load_fa4(args.fa4_path)

    qkv = None
    qkv_seqlen = None
    if args.qk_source == "model":
        from real_qk import capture_qk  # noqa: PLC0415

        # Decode tiles K/V across the batch, so a short capture is always fine
        # there.  Prefill needs the full span, and the amount of real text on
        # hand is finite — but refusing to run at all would throw away every
        # shorter point too.  Capture what exists; points beyond it fall back to
        # random Q/K and say so.  That is sound because real Q/K only matters
        # for the cosine table (``--cos-max-seqlen``, far below this ceiling);
        # kernel *latency* does not depend on the value distribution.
        cap = capture_qk(args.qk_model, seqlen=max(seqlens), layer=args.qk_layer,
                         device=device, allow_short=True)
        qkv = tuple(t.to(dtype).contiguous() for t in (cap["q"], cap["k"], cap["v"]))
        qkv_seqlen = int(cap["seqlen"])
        print(f"Q/K/V     : layer {args.qk_layer} of {args.qk_model} "
              f"(Hq={cap['head_q']} Hkv={cap['head_kv']} D={cap['head_dim']}, "
              f"{qkv_seqlen} real tokens captured"
              + ("; tiled across the batch for longer contexts)" if args.mode == "decode"
                 else ")"))
        if args.mode != "decode" and qkv_seqlen < max(seqlens):
            over = [s for s in seqlens if s > qkv_seqlen]
            print(f"NOTE      : only {qkv_seqlen} real tokens available, so kv_len "
                  f"{', '.join(str(s) for s in over)} fall back to random Q/K "
                  f"(timing only — cosine is capped at {args.cos_max_seqlen}).")
            if args.cos and args.cos_max_seqlen > qkv_seqlen:
                # A cosine computed against random Q/K is not a weak measurement,
                # it is a different measurement (random attention is near-uniform,
                # so every selector scores ~0.33 instead of ~0.99).  Silently
                # reporting it in the same column as the real ones would be worse
                # than reporting nothing, so cap the column instead of warning.
                print(f"NOTE      : capping --cos-max-seqlen {args.cos_max_seqlen} -> "
                      f"{qkv_seqlen} so no cosine is computed on random Q/K.")
                args.cos_max_seqlen = qkv_seqlen
        del cap
        torch.cuda.empty_cache()
    configs = list(iter_configs(args.configs))

    print("=" * 118)
    print(f"MSA sparse-attention configuration sweep — {torch.cuda.get_device_name(args.gpu)}")
    print(f"model     : {model.name}  Hq={model.num_qo_heads} Hkv={model.num_kv_heads} "
          f"(GQA {model.qhead_per_kv}x) D={model.head_dim}  layers={model.num_hidden_layers} "
          f"(full-attn {model.full_attention_layers}, +{model.mtp_layers} MTP)  "
          f"ctx={model.max_position_embeddings}")
    if model.notes:
        print(f"            {model.notes}")
    for gap in model.compatibility():
        print(f"NOTE      : {gap}")
    qlen_desc = args.qlen if args.qlen is not None else ("1 (decode)" if args.mode == "decode"
                                                         else "=kv_len (prefill)")
    print(f"shapes    : mode={args.mode} batch={args.batch} kv_len={seqlens} "
          f"q_len={qlen_desc} causal=True dtype=bf16")
    print(f"configs   : {len(configs)}  ({', '.join(c.resolved_name() for c in configs)})")
    if args.indexer_mode == "measure":
        print("NOTE      : 'indexer' is the FP4 scorer at its native 128-token granularity; "
              "its MMA cost is block-size invariant.")
    print("=" * 118)

    rows = []

    def flush():
        """Persist whatever has been measured so far.

        A long sweep can die on an unrecoverable CUDA fault (a poisoned context
        makes every later config fail too), and losing an hour of completed
        measurements to that would be daft.  Called after every row and again
        from the outer finally.
        """
        if args.csv:
            write_csv(rows, args.csv, quiet=True)
        if args.json:
            Path(args.json).write_text(json.dumps(rows, indent=2, default=str))

    for seqlen_k in seqlens:
        if args.qlen is not None:
            seqlen_q = args.qlen
        else:
            seqlen_q = 1 if args.mode == "decode" else seqlen_k
        print(f"\n### kv_len={seqlen_k} q_len={seqlen_q} batch={args.batch}")
        print_header()

        # Prefill consumes the capture as a contiguous prefix, so a point longer
        # than the capture has to use random Q/K; decode tiles it and never does.
        qkv_here = qkv
        if qkv is not None and args.mode != "decode" and qkv_seqlen < seqlen_k:
            qkv_here = None

        dense_ms = None
        dense_out = None
        # The repo's dense FMHA indexes Q/O with 32-bit arithmetic, so past
        # 2**31 elements it cannot run at all -- that is a property of the
        # shape, known before launching.  Skipping it there is not hiding a
        # failure; it replaces a guaranteed multi-line guard message, repeated
        # at every long context, with one line stating the ceiling.  FA4 has no
        # such limit and still provides the reference.
        dense_elems = args.batch * seqlen_q * model.num_qo_heads * model.bench_head_dim()
        dense_reachable = dense_elems < _QO_INT32_LIMIT
        if not args.no_dense and not dense_reachable:
            ceiling = (_QO_INT32_LIMIT - 1) // (model.num_qo_heads * model.bench_head_dim())
            print(f"      -- dense (msa) skipped: Q/O would need {dense_elems} elements, "
                  f"past the {_QO_INT32_LIMIT} Int32 limit "
                  f"(ceiling {ceiling} query tokens at this head geometry)")
        if not args.no_dense:
            if dense_reachable:
                try:
                    dense, dense_out = bench_dense(
                        model=model, batch=args.batch, seqlen_q=seqlen_q, seqlen_k=seqlen_k,
                        dtype=dtype, device=device, dry_ms=args.dry_ms, rep_ms=args.rep_ms,
                        seed=args.seed, timing=args.timing,
                        want_output=args.cos and seqlen_k <= args.cos_max_seqlen, qkv=qkv_here)
                    dense_ms = dense["pipeline_gpu_ms"]
                    rows.append(dense)
                    print_row(dense)
                except Exception as exc:  # noqa: BLE001
                    dense_out = None
                    print(f"      !! dense reference failed: {_fmt_exc(exc)}")

            # FA4 is a separate kernel with no such limit, so it must not be
            # gated on the repo dense being reachable -- it is precisely at the
            # lengths the repo kernel cannot reach that its reference matters.
            if args.fa4_python:
                ms, why = _fa4_subprocess(
                    args.fa4_python, args.fa4_path, model=model, batch=args.batch,
                    seqlen_q=seqlen_q, seqlen_k=seqlen_k, dry_ms=args.dry_ms,
                    rep_ms=args.rep_ms, seed=args.seed)
                if ms is None:
                    print(f"      !! FA4 (external interpreter) failed: {why}")
                else:
                    row = {"name": "dense-fa4",
                           "label": f"FlashAttention-4 causal ({args.fa4_python})",
                           "batch": args.batch, "seqlen_q": seqlen_q,
                           "seqlen_k": seqlen_k, "head_q": model.num_qo_heads,
                           "head_kv": model.num_kv_heads,
                           "head_dim": model.bench_head_dim(),
                           "stages": {"attn": {"ms": ms}}, "pipeline_ms": ms,
                           "pipeline_gpu_ms": ms, "attn_ms_only": ms,
                           "supported": True, "errors": {}}
                    rows.append(row)
                    print_row(row, dense_ms)
                    if dense_ms is None:
                        dense_ms = ms
                        print("      -- vs_dense below is against FA4")
            elif fa4_fn is not None:
                try:
                    fa4_row, _ = bench_fa4(
                        fa4_fn, model=model, batch=args.batch, seqlen_q=seqlen_q,
                        seqlen_k=seqlen_k, dtype=dtype, device=device,
                        dry_ms=args.dry_ms, rep_ms=args.rep_ms, seed=args.seed,
                        timing=args.timing, qkv=qkv_here)
                    rows.append(fa4_row)
                    print_row(fa4_row, dense_ms)
                    if dense_ms is None:
                        # Without this the whole column reads N/A at exactly the
                        # lengths the sweep exists to measure, even though a
                        # perfectly good dense reference just ran.
                        dense_ms = fa4_row["pipeline_gpu_ms"]
                        print("      -- vs_dense below is against FA4 "
                              "(the repo's dense kernel cannot run at this shape)")
                except Exception as exc:  # noqa: BLE001
                    print(f"      !! FA4 reference failed: {type(exc).__name__}: {exc}")

        # Query chunking: the score tensor is [Hkv, blocks, total_q], so its
        # footprint falls linearly with the chunk while the result is unchanged.
        # Only prefill has queries to split.
        chunk = args.chunk_q if (args.mode != "decode" and args.chunk_q) else 0
        if chunk < 0:
            chunk = _auto_chunk_q(model, seqlen_k)
            if 0 < chunk < seqlen_q:
                print(f"NOTE      : chunking prefill queries at {chunk} "
                      f"({-(-seqlen_q // chunk)} chunks) -- the indexer's score tensor "
                      f"is Int32-addressed, which caps the chunk below what memory allows.")
        chunk_starts = (list(range(0, seqlen_q, chunk)) if 0 < chunk < seqlen_q else [])

        for cfg in configs:
            t0 = time.time()
            try:
                if chunk_starts:
                    parts = []
                    for lo in chunk_starts:
                        clen = min(chunk, seqlen_q - lo)
                        # Right-aligning clen queries against (lo + clen) keys puts
                        # chunk query j at absolute position lo + j -- the same
                        # causal extent it has unchunked.
                        parts.append(bench_config(
                            cfg, model=model, batch=args.batch, seqlen_q=clen,
                            seqlen_k=lo + clen, dtype=dtype, device=device,
                            dry_ms=args.dry_ms, rep_ms=args.rep_ms,
                            indexer_mode=args.indexer_mode, seed=args.seed,
                            timing=args.timing, dense_out=None, qkv=None,
                        ))
                        torch.cuda.empty_cache()
                    row = _merge_chunked(parts, seqlen_q=seqlen_q, seqlen_k=seqlen_k,
                                         chunk_q=chunk)
                else:
                    # Decode multiplies context by batch, which is what makes the
                    # CSR histogram and the scale reorder bind; split the batch so
                    # each call stays under both.  Requests are independent, so
                    # this is exact.
                    cb = (_auto_chunk_batch(cfg, model, seqlen_k, args.batch)
                          if args.mode == "decode" else args.batch)
                    if cb < args.batch:
                        parts = []
                        for lo in range(0, args.batch, cb):
                            n = min(cb, args.batch - lo)
                            parts.append(bench_config(
                                cfg, model=model, batch=n, seqlen_q=seqlen_q,
                                seqlen_k=seqlen_k, dtype=dtype, device=device,
                                dry_ms=args.dry_ms, rep_ms=args.rep_ms,
                                indexer_mode=args.indexer_mode, seed=args.seed,
                                timing=args.timing, dense_out=None, qkv=qkv_here,
                            ))
                            torch.cuda.empty_cache()
                        row = _merge_batch(parts, batch=args.batch, chunk_batch=cb)
                    else:
                        row = bench_config(
                            cfg, model=model, batch=args.batch, seqlen_q=seqlen_q,
                            seqlen_k=seqlen_k,
                            dtype=dtype, device=device, dry_ms=args.dry_ms,
                            rep_ms=args.rep_ms,
                            indexer_mode=args.indexer_mode, seed=args.seed,
                            timing=args.timing, dense_out=dense_out, qkv=qkv_here,
                        )
            except Exception as exc:  # noqa: BLE001 - never abort the sweep
                row = {**cfg.to_dict(), "batch": args.batch, "seqlen_q": seqlen_q,
                       "seqlen_k": seqlen_k,
                       "errors": {"setup": _fmt_exc(exc)}}
                traceback.print_exc(limit=2)
            row["wall_s"] = round(time.time() - t0, 1)
            rows.append(row)
            print_row(row, dense_ms)

            # The rows above are the prefill kernel doing decode's work, which is
            # what this sweep has always measured and why decode never beat
            # dense.  The paged decode kernel is the right one for q_len=1; it
            # lives in another environment, so measure it there and print it
            # next to its in-process counterpart rather than leaving the table
            # to imply the slow path is all there is.
            if args.mode == "decode" and args.decode_python:
                ms, cos, why = _decode_fast_subprocess(
                    args.decode_python, model=model, batch=args.batch,
                    seqlen_k=seqlen_k, block_size=cfg.block_size, topk=cfg.topk,
                    head_mode=cfg.head_mode, dry_ms=args.dry_ms,
                    rep_ms=args.rep_ms, seed=args.seed)
                if ms is None:
                    print(f"      -- fast decode unavailable: {why}")
                else:
                    fast = {**cfg.to_dict(), "name": cfg.resolved_name() + "+fastdec",
                            "batch": args.batch, "seqlen_q": seqlen_q,
                            "seqlen_k": seqlen_k, "head_q": model.num_qo_heads,
                            "head_kv": model.num_kv_heads,
                            "head_dim": model.bench_head_dim(),
                            "cos_min": cos, "supported": True, "errors": {}}
                    # Carry the parent's selection stages and swap only attn.
                    # fastdec replaces one stage of four; reporting its kernel
                    # time as the row total would compare a single kernel
                    # against other rows' end-to-end cost in the same column.
                    # Drop CSR as well as attn.  The k2q CSR exists so a
                    # prefill-shaped sparse kernel can walk K blocks and find
                    # the queries that selected them; the paged decode kernel
                    # takes a page table instead and never reads it.  The
                    # verified fast measurement builds no CSR at all and still
                    # returns cosine 0.998, so charging this row for one would
                    # bill work the path does not do.
                    _unused_by_fastdec = ("attn", "csr")
                    parent = row.get("stages", {}) if row.get("supported") else {}
                    fast["stages"] = {k: dict(v) for k, v in parent.items()
                                      if k not in _unused_by_fastdec}
                    fast["stages"]["attn"] = {"ms": ms}
                    total = sum(st["ms"] for st in fast["stages"].values())
                    fast["pipeline_ms"] = fast["pipeline_gpu_ms"] = total
                    fast["attn_ms_only"] = ms
                    rows.append(fast)
                    print_row(fast, dense_ms)
                    extra = (f"; per-KV-head selection run as {model.num_kv_heads}x "
                             f"the requests" if cfg.head_mode == "keep" else "")
                    print(f"      -- paged decode kernel, only the selected blocks{extra}; "
                          f"worst cos {cos:.5f} vs fp32")
            flush()
            try:
                torch.cuda.empty_cache()
            except Exception as exc:  # noqa: BLE001 - a poisoned context is fatal
                print(f"\nFATAL: CUDA context is unusable after "
                      f"{cfg.resolved_name()}: {_fmt_exc(exc)}")
                print("Partial results have been written; rerun the remaining "
                      "configurations in a fresh process.")
                flush()
                return 2

        dense_out = None
        torch.cuda.empty_cache()

    flush()
    if args.csv:
        print(f"\nCSV written to {args.csv}")
    if args.json:
        print(f"JSON written to {args.json}")

    failed = sum(1 for r in rows if r.get("errors"))
    print(f"\n{len(rows)} row(s), {failed} with at least one failed stage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
