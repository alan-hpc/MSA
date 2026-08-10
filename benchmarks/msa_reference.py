#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""A pure-PyTorch MSA, written stage by stage so each kernel can be checked alone.

The kernels in this repo are checked against references that live next to them --
the top-k against ``torch.topk``, the decode forward against the test suite's
paged reference -- but there has been no reference for the *pipeline*: scores,
selection and attention composed the way MSA composes them.  Without one, a
wrong selection and a wrong forward can cancel, and a plausible cosine hides
both.

MSA (arXiv 2606.13392) is blockwise sparse attention over GQA: an index branch
scores KV blocks, each GQA group independently takes a top-k, and the main
branch runs exact attention over only those blocks.  This file mirrors that in
three functions, each usable on its own:

    block_scores(q, k, ...)          -> stage 1, the index branch
    select_blocks(scores, ...)       -> stage 2, top-k plus the forced windows
    sparse_attention(q, k, v, sel)   -> stage 3, the main branch

Everything is fp32 and written for clarity, not speed -- it is the thing the
kernels are judged against, so it must be obviously right rather than fast.

Two details are easy to get wrong and are worth stating, because both cost real
debugging time in this repo:

* ``force_init_tokens`` / ``force_end_tokens`` are **tokens**, not blocks.  The
  block counts are ``ceil(tokens / block_size)``, so the same 128-token window
  is 4 blocks at ``block_size=32`` and 1 block at 128.
* The forced windows come out of the top-k budget, they do not add to it.  With
  ``topk=32`` and both windows at 128 tokens over 32-token blocks, only 24
  blocks are actually chosen by score.
"""

from __future__ import annotations

import math

import torch


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


# ---------------------------------------------------------------------------
# Stage 1 -- index branch: one score per (KV head, query, block)
# ---------------------------------------------------------------------------

def block_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    block_size: int,
    head_mode: str = "keep",
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Score every KV block for every query.

    Parameters
    ----------
    q : ``[S, Hq, D]``
    k : ``[S, Hkv, D]``
    head_mode : ``"keep"`` scores each KV head's group separately and keeps one
        row per KV head, which is MSA's group-independent selection.  ``"sum"``
        and ``"max"`` reduce those rows to a single shared selection.

    Returns ``[rows, num_blocks, S]`` where ``rows`` is ``Hkv`` for ``"keep"``
    and 1 otherwise -- the layout the kernel's top-k consumes.
    """
    S, hq, d = q.shape
    hkv = k.shape[1]
    if hq % hkv:
        raise ValueError(f"Hq={hq} must be a multiple of Hkv={hkv}")
    group = hq // hkv
    scale = softmax_scale if softmax_scale is not None else d ** -0.5
    nb = _ceil_div(S, block_size)

    qf, kf = q.float(), k.float()
    # Per KV head: the group's queries against that head's keys.  Averaging the
    # group's queries first is what makes this an *index* branch rather than
    # full attention -- one QK per group, not one per query head.
    out = torch.empty((hkv, nb, S), dtype=torch.float32, device=q.device)
    for h in range(hkv):
        q_grp = qf[:, h * group:(h + 1) * group, :].mean(dim=1)     # [S, D]
        logits = (q_grp @ kf[:, h, :].transpose(0, 1)) * scale      # [S, S] q x k
        pad = nb * block_size - S
        if pad:
            logits = torch.nn.functional.pad(logits, (0, pad), value=float("-inf"))
        # Block score = max over the block's keys.  Max, not mean: a block earns
        # its place on its best key, and this matches the kernel's epilogue.
        out[h] = logits.view(S, nb, block_size).amax(dim=-1).transpose(0, 1)

    if head_mode == "keep":
        return out
    if head_mode == "sum":
        return out.sum(dim=0, keepdim=True)
    if head_mode == "max":
        return out.amax(dim=0, keepdim=True)
    raise ValueError(f"unknown head_mode {head_mode!r}")


# ---------------------------------------------------------------------------
# Stage 2 -- top-k with the forced sink and local windows
# ---------------------------------------------------------------------------

def select_blocks(
    scores: torch.Tensor,
    *,
    block_size: int,
    topk: int,
    force_init_tokens: int = 0,
    force_end_tokens: int = 0,
    causal: bool = True,
    seqlen: int | None = None,
) -> torch.Tensor:
    """Pick ``topk`` blocks per (row, query).  Returns ``[rows, S, topk]`` int64,
    ascending by block index, ``-1`` where a query has fewer blocks available.

    ``causal`` positions the forced local window; it does not mask candidates,
    matching the kernel (see below).

    The forced windows are taken out of the budget, not added to it: the sink
    blocks at the front and the local blocks at the query's own position are
    always present, and the remainder is filled by score.
    """
    rows, nb, S = scores.shape
    seqlen = S if seqlen is None else seqlen
    fi = _ceil_div(force_init_tokens, block_size)
    fe = _ceil_div(force_end_tokens, block_size)
    if fi + fe > topk:
        raise ValueError(
            f"forced windows need {fi}+{fe} blocks but topk is {topk}; "
            "the windows come out of the budget, not on top of it"
        )

    dev = scores.device
    q_pos = torch.arange(S, device=dev)
    own = q_pos // block_size                                   # query's own block
    blk = torch.arange(nb, device=dev)

    out = torch.full((rows, S, topk), -1, dtype=torch.int64, device=dev)
    for r in range(rows):
        s = scores[r].transpose(0, 1).clone()                   # [S, nb]
        # Selection is deliberately NOT causally masked.  In the kernel,
        # causal_end_block only positions the forced local window
        # (SparseTopKIsForced); it does not restrict which blocks top-k may
        # pick, and the trivial path emits every block when topk >= num_blocks.
        # Causality is enforced downstream, by the attention kernel's mask.  A
        # reference that masks here reports a difference on almost every row and
        # says nothing about the selector.
        forced = torch.zeros((S, nb), dtype=torch.bool, device=dev)
        if fi:
            forced[:, :fi] = True
        if fe:
            # The local window ends at the query's own block, which is what
            # causal_end_block carries into the kernel.
            lo = (own - (fe - 1)).clamp_min(0)
            forced |= (blk[None, :] >= lo[:, None]) & (blk[None, :] <= own[:, None])

        # Rank forced blocks above everything else so they always survive, then
        # let the score decide the rest.
        ranked = torch.where(forced, torch.full_like(s, float("inf")), s)
        ranked = torch.where(torch.isinf(ranked) & (ranked < 0),
                             torch.full_like(ranked, float("-inf")), ranked)
        take = min(topk, nb)
        idx = ranked.topk(take, dim=-1).indices                 # [S, take]
        keep = torch.gather(ranked, 1, idx) > float("-inf")
        idx = torch.where(keep, idx, torch.full_like(idx, -1))
        idx, _ = idx.sort(dim=-1, descending=True)              # -1 to the tail
        # ascending block order among the valid entries, as the kernel emits
        n_valid = keep.sum(-1)
        for i in range(S):
            v = int(n_valid[i])
            if v:
                idx[i, :v] = idx[i, :v].sort().values
        out[r, :, :take] = idx
    return out


# ---------------------------------------------------------------------------
# Stage 3 -- main branch: exact attention over the selected blocks
# ---------------------------------------------------------------------------

def sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selection: torch.Tensor,
    *,
    block_size: int,
    causal: bool = True,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Exact attention restricted to the selected blocks.  Returns ``[S, Hq, D]``.

    ``selection`` is ``[rows, S, topk]``; ``rows`` is either ``Hkv`` (one
    selection per KV head) or 1 (shared), and a shared selection is broadcast.
    """
    S, hq, d = q.shape
    hkv = k.shape[1]
    group = hq // hkv
    scale = softmax_scale if softmax_scale is not None else d ** -0.5
    rows = selection.shape[0]
    if rows not in (1, hkv):
        raise ValueError(f"selection has {rows} rows, expected 1 or Hkv={hkv}")

    qf, kf, vf = q.float(), k.float(), v.float()
    out = torch.zeros((S, hq, d), dtype=torch.float32, device=q.device)
    pos = torch.arange(S, device=q.device)

    for h in range(hkv):
        sel_h = selection[0 if rows == 1 else h]                 # [S, topk]
        for i in range(S):
            blocks = sel_h[i]
            blocks = blocks[blocks >= 0]
            if blocks.numel() == 0:
                continue
            # token ids covered by those blocks
            offs = torch.arange(block_size, device=q.device)
            cols = (blocks[:, None] * block_size + offs[None, :]).reshape(-1)
            cols = cols[cols < S]
            if causal:
                cols = cols[cols <= i]
            if cols.numel() == 0:
                continue
            qi = qf[i, h * group:(h + 1) * group, :]             # [group, D]
            logits = (qi @ kf[cols, h, :].transpose(0, 1)) * scale
            p = torch.softmax(logits, dim=-1)
            out[i, h * group:(h + 1) * group, :] = p @ vf[cols, h, :]
    return out


# ---------------------------------------------------------------------------
# The three stages composed, for end-to-end comparison
# ---------------------------------------------------------------------------

def msa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_size: int,
    topk: int,
    head_mode: str = "keep",
    force_init_tokens: int = 0,
    force_end_tokens: int = 0,
    causal: bool = True,
    softmax_scale: float | None = None,
):
    """Full MSA in torch.  Returns ``(output, scores, selection)`` so a caller
    can compare any single stage rather than only the end."""
    scores = block_scores(q, k, block_size=block_size, head_mode=head_mode,
                          softmax_scale=softmax_scale)
    sel = select_blocks(scores, block_size=block_size, topk=topk,
                        force_init_tokens=force_init_tokens,
                        force_end_tokens=force_end_tokens, causal=causal)
    out = sparse_attention(q, k, v, sel, block_size=block_size, causal=causal,
                           softmax_scale=softmax_scale)
    return out, scores, sel


def dense_attention(q, k, v, *, causal=True, softmax_scale=None):
    """Full attention, for the "how much did sparsity cost" comparison."""
    S, hq, d = q.shape
    hkv = k.shape[1]
    group = hq // hkv
    scale = softmax_scale if softmax_scale is not None else d ** -0.5
    qf, kf, vf = q.float(), k.float(), v.float()
    out = torch.empty((S, hq, d), dtype=torch.float32, device=q.device)
    for h in range(hkv):
        qi = qf[:, h * group:(h + 1) * group, :]                 # [S, group, D]
        logits = torch.einsum("sgd,td->sgt", qi, kf[:, h, :]) * scale
        if causal:
            mask = torch.arange(S, device=q.device)[None, :] > torch.arange(S, device=q.device)[:, None]
            logits = logits.masked_fill(mask[:, None, :], float("-inf"))
        out[:, h * group:(h + 1) * group, :] = torch.einsum(
            "sgt,td->sgd", torch.softmax(logits, dim=-1), vf[:, h, :])
    return out


__all__ = [
    "block_scores",
    "select_blocks",
    "sparse_attention",
    "msa_reference",
    "dense_attention",
]
