# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Config-driven MSA block-sparse attention pipeline.

Wires the four stages of an MSA sparse attention layer behind one object whose
behaviour is fully determined by an :class:`~fmha_sm100.msa_config.MsaSparseConfig`::

    block scores            [H_idx, num_blocks, total_q] fp32   (caller/indexer)
      -> select()           top-k with head aggregation + forced windows
      -> build_csr()        k2q CSR + fused sparse attention schedule
      -> attend()           sparse_atten_func

Every stage is separately callable so a benchmark can time them individually,
and ``__call__`` runs the whole chain.

Head aggregation, the sink/local windows and the head-outermost output layout
are all handled inside ``sparse_topk_select``, so ``select()`` adds no extra
passes over the score tensor and hands ``build_k2q_csr`` a tensor it can consume
without a permute.

This module deliberately does **not** own the block scorer: production callers
feed scores from the FP4 indexer, tests feed synthetic scores, and a future
scorer with sub-128 output granularity drops in without touching this file.
See :func:`block_scores_shape` for the contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch

from .api import sparse_topk_select
from .msa_config import MsaSparseConfig

__all__ = [
    "MsaSparseAttention",
    "MsaSelection",
    "block_scores_shape",
    "causal_end_blocks",
]


def block_scores_shape(config: MsaSparseConfig, *, num_qo_heads, num_kv_heads, total_q, max_seqlen_k):
    """Shape the block scorer must produce for ``config``.

    ``(num_scorer_heads, num_blocks, total_q)`` fp32, contiguous, with entries
    past a request's own block count set to ``-inf``.  ``num_scorer_heads`` is
    ``num_kv_heads`` for ``head_mode="keep"`` and ``num_qo_heads`` otherwise.
    """
    return (
        config.num_scorer_heads(num_qo_heads, num_kv_heads),
        config.num_blocks(max_seqlen_k),
        total_q,
    )


def causal_end_blocks(
    seqlens_q: Sequence[int],
    seqlens_k: Sequence[int],
    block_size: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Per-query exclusive causal block bound for the ``force_end`` window.

    Query ``i`` of request ``b`` sits at KV position ``i + (k_len - q_len)``, so
    the last block it may attend to is ``kv_pos // block_size`` and the
    exclusive bound is one past that.  Built from host-side lengths because any
    scheduler already has them; passing device tensors would force a sync.
    """
    if len(seqlens_q) != len(seqlens_k):
        raise ValueError("seqlens_q and seqlens_k must have the same length")
    bounds = []
    for q_len, k_len in zip(seqlens_q, seqlens_k):
        offset = int(k_len) - int(q_len)
        pos = torch.arange(int(q_len), dtype=torch.int32) + offset
        bounds.append(pos.clamp_(min=0) // int(block_size) + 1)
    if not bounds:
        return torch.empty((0,), dtype=torch.int32, device=device)
    return torch.cat(bounds).to(device=device, dtype=torch.int32).contiguous()


@dataclass
class MsaSelection:
    """Output of :meth:`MsaSparseAttention.select` plus the CSR it feeds."""

    q2k_indices: torch.Tensor                       # [Hkv, total_q, topk] int32
    k2q_row_ptr: Optional[torch.Tensor] = None      # [Hkv, total_rows + 1] int32
    k2q_q_indices: Optional[torch.Tensor] = None    # [Hkv, total_q * topk] int32
    schedule: object = None
    meta: dict = field(default_factory=dict)


class MsaSparseAttention:
    """Runs one MSA sparse attention configuration end to end.

    Parameters
    ----------
    config : MsaSparseConfig
        The configuration point.  Validated on construction.
    num_qo_heads, num_kv_heads : int
        GQA geometry.  ``num_qo_heads`` must be a multiple of ``num_kv_heads``.
    head_dim : int
        Attention head dimension.  The shipped kernels support 128 only.
    causal : bool
        Whether the attention (and the ``force_end`` window) is causal.
    """

    def __init__(
        self,
        config: MsaSparseConfig,
        *,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int = 128,
        causal: bool = True,
    ):
        self.config = config.validate()
        if num_kv_heads <= 0 or num_qo_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_qo_heads ({num_qo_heads}) must be a positive multiple of "
                f"num_kv_heads ({num_kv_heads})"
            )
        self.num_qo_heads = int(num_qo_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.qhead_per_kv = self.num_qo_heads // self.num_kv_heads
        self.head_dim = int(head_dim)
        self.causal = bool(causal)
        # Import lazily: pulling in the CuTe-DSL runtime is expensive and only
        # the attention/CSR stages need it.
        self._build_k2q_csr = None
        self._sparse_atten_func = None
        self._decode_wrapper = None
        self._decode_key = None

    # ---- lazy CuTe-DSL handles --------------------------------------------

    def _sparse_api(self):
        if self._build_k2q_csr is None:
            from .sparse import build_k2q_csr, sparse_atten_func

            self._build_k2q_csr = build_k2q_csr
            self._sparse_atten_func = sparse_atten_func
        return self._build_k2q_csr, self._sparse_atten_func

    # ---- stage 1: selection ------------------------------------------------

    def select(
        self,
        block_scores: torch.Tensor,
        *,
        num_valid_blocks: int,
        causal_end_block: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Top-k block selection -> ``[Hkv, total_q, topk]`` int32.

        ``num_valid_blocks`` is the real block count (``ceil(max_seqlen_k /
        block_size)``); selections past it are emitted as ``-1``.  Pass
        ``causal_end_block`` (see :func:`causal_end_blocks`) to anchor the
        ``force_end`` window to each query's own causal position rather than to
        the end of the sequence.
        """
        cfg = self.config
        expected_heads = cfg.num_scorer_heads(self.num_qo_heads, self.num_kv_heads)
        if block_scores.shape[0] != expected_heads:
            raise ValueError(
                f"{cfg.label}: block_scores must have {expected_heads} head rows "
                f"(head_mode={cfg.head_mode!r}), got {block_scores.shape[0]}"
            )
        idx = sparse_topk_select(
            block_scores,
            cfg.topk,
            num_valid_pages=num_valid_blocks,
            output=out,
            force_begin_blocks=cfg.force_init_blocks,
            force_end_blocks=cfg.force_end_blocks,
            head_mode=cfg.head_mode,
            num_out_heads=cfg.num_selection_heads(self.num_kv_heads),
            causal_end_block=causal_end_block,
            # build_k2q_csr consumes [Hkv, total_q, topk]; writing that layout
            # directly saves a permute + copy of the whole selection.
            head_outermost=True,
        )
        if idx.shape[0] != self.num_kv_heads:
            # A single layer-wide selection (any `sum`/`max`, or `shared_index`)
            # still has to reach the attention kernel as [Hkv, ...]: its CSR
            # metadata is keyed on KV heads and the tensors must be contiguous,
            # so `expand` alone is not enough.
            idx = idx.expand(self.num_kv_heads, -1, -1).contiguous()
        return idx

    # ---- stage 2: CSR + schedule -------------------------------------------

    def build_csr(
        self,
        q2k_indices: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        *,
        total_k: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        total_rows: int,
    ):
        build_k2q_csr, _ = self._sparse_api()
        return build_k2q_csr(
            q2k_indices,
            cu_seqlens_q,
            cu_seqlens_k,
            self.config.block_size,
            total_k=total_k,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_q=max_seqlen_q,
            total_rows=total_rows,
            qhead_per_kv=self.qhead_per_kv,
            return_schedule=True,
        )

    # ---- stage 3: attention ------------------------------------------------

    def decode_attend(
        self,
        q,
        k_cache,
        v_cache,
        page_table,
        selection,
        *,
        page_size: int,
        softmax_scale: Optional[float] = None,
    ):
        """Single-token decode that reads only the selected KV blocks.

        The sweep's decode figures come from the *prefill* kernel, which at
        q_len=1 fills a 128-row tile with 16 query tokens and leaves the machine
        idle; it has never beaten dense.  The paged decode kernel suits the shape
        far better, and the two things that look like blockers are not:

        * Its sparse path (``q2k_indices``) is a stub -- but sparsity does not
          need it.  A page table listing only the selected blocks makes the dense
          kernel read exactly those blocks.  This needs one selection shared
          across KV heads, which ``head_mode`` ``sum``/``max`` produce.
        * It wants a full packed-q tile (``seqlen_q == 128 / qhead_per_kv``),
          being built for speculative decode.  Broadcasting the one query across
          the tile and keeping the row that sees the whole run satisfies it; the
          waste is Q-side and decode is bound by the KV stream.

        Measured at 128K, batch 32, topk 16: 0.685 ms reading everything against
        0.027 ms reading the selection, at cosine 0.9985 against the reference --
        25x, where the prefill-kernel path runs 3.1 ms.

        Parameters
        ----------
        page_table : torch.Tensor
            ``[batch, num_pages]`` int32, logical page -> physical page.
        selection : torch.Tensor
            ``[batch, topk]`` int64/int32 **logical** page indices, one shared
            selection per request.  Physical ids are gathered through
            ``page_table``; computing them directly assumes a contiguous layout
            that the caller's page table need not have.
        """
        if self.config.head_mode == "keep" and self.num_kv_heads > 1:
            raise ValueError(
                "decode_attend needs one selection shared across KV heads; "
                "head_mode='keep' produces one per head. Use 'sum' or 'max', or "
                "pass a single head's rows."
            )
        if q.ndim != 3:
            raise ValueError("decode_attend expects q with shape [batch, Hq, D]")
        batch, hq, dim = (int(x) for x in q.shape)
        topk = int(selection.shape[-1])

        from .cute.interface import SparseDecodePagedAttentionWrapper

        sel_phys = torch.gather(
            page_table, 1, selection.to(torch.int64)
        ).contiguous().to(torch.int32)

        q_tokens = 128 // self.qhead_per_kv
        q_packed = (q.unsqueeze(1)
                    .expand(batch, q_tokens, hq, dim)
                    .reshape(batch * q_tokens, hq, dim)
                    .contiguous())
        kv_len = topk * page_size
        seqused = torch.full((batch,), kv_len, dtype=torch.int32, device=q.device)

        # Reuse the wrapper across steps: constructing it is pure overhead, and
        # at these shapes the surrounding plan/dispatch already costs more than
        # the kernel.  plan() itself must re-run each step -- the page table is
        # what carries the selection, and that changes every token.
        key = (batch, hq, dim, topk, page_size, kv_len, q_tokens)
        if self._decode_wrapper is None or self._decode_key != key:
            self._decode_wrapper = SparseDecodePagedAttentionWrapper(
                blk_kv=page_size, causal=self.causal)
            self._decode_key = key
        wrapper = self._decode_wrapper
        wrapper.plan(
            page_table=sel_phys,
            seqused_k=seqused,
            seqlen_q=q_tokens,
            max_seqlen_k=kv_len,
            q2k_indices=None,          # sparsity rides in the page table
            num_qo_heads=hq,
            num_kv_heads=self.num_kv_heads,
            head_dim=dim,
        )
        out = wrapper.run(q_packed, k_cache, v_cache,
                          softmax_scale=softmax_scale or dim ** -0.5)
        out = out[0] if isinstance(out, (tuple, list)) else out
        # Under causal masking the last row is the one that saw the whole run.
        return out.reshape(batch, q_tokens, hq, dim)[:, -1].contiguous()

    def attend(
        self,
        q,
        k,
        v,
        k2q_row_ptr,
        k2q_q_indices,
        *,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        schedule=None,
        softmax_scale: Optional[float] = None,
        page_table=None,
        seqused_k=None,
    ):
        _, sparse_atten_func = self._sparse_api()
        return sparse_atten_func(
            q,
            k,
            v,
            k2q_row_ptr,
            k2q_q_indices,
            self.config.topk,
            blk_kv=self.config.block_size,
            causal=self.causal,
            softmax_scale=softmax_scale,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            schedule=schedule,
            page_table=page_table,
            seqused_k=seqused_k,
        )

    # ---- full chain --------------------------------------------------------

    def __call__(
        self,
        q,
        k,
        v,
        block_scores,
        *,
        cu_seqlens_q,
        cu_seqlens_k,
        seqlens_q: Sequence[int],
        seqlens_k: Sequence[int],
        max_seqlen_q: int,
        max_seqlen_k: int,
        total_k: int,
        causal_end_block: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
    ):
        cfg = self.config
        num_valid_blocks = cfg.num_blocks(max_seqlen_k)
        total_rows = sum(cfg.num_blocks(int(n)) for n in seqlens_k)
        if causal_end_block is None and self.causal and cfg.force_end_blocks > 0:
            causal_end_block = causal_end_blocks(
                seqlens_q, seqlens_k, cfg.block_size, device=q.device
            )
        q2k = self.select(
            block_scores,
            num_valid_blocks=num_valid_blocks,
            causal_end_block=causal_end_block,
        )
        row_ptr, q_indices, schedule = self.build_csr(
            q2k,
            cu_seqlens_q,
            cu_seqlens_k,
            total_k=total_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            total_rows=total_rows,
        )
        return self.attend(
            q,
            k,
            v,
            row_ptr,
            q_indices,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            schedule=schedule,
            softmax_scale=softmax_scale,
        )

    # ---- capacity planning --------------------------------------------------

    #: The FP4 indexer builds its score tensor's CuTe layout from ``Int32``
    #: extents, so it can address at most this many elements.  Mirrors
    #: ``fp4_indexer_interface._SCORE_TENSOR_MAX_ELEMS``.
    SCORE_TENSOR_MAX_ELEMS = 2**31
    #: KV tokens the shipped FP4 indexer reduces per emitted score.
    SCORER_BLOCK = 128

    def max_scorer_seqlen(self, *, batch: int = 1) -> int:
        """Longest full prefill this configuration can score, per the Int32 cap.

        The score tensor is ``[scorer_heads, ceil(S / 128), batch * S]``, so the
        limit scales as ``1 / sqrt(scorer_heads)``.  ``head_mode="keep"`` scores
        one row per KV head and reaches far further than ``sum`` / ``max``,
        which score one row per query head.

        Returns the largest ``S`` (rounded down to a multiple of 128) that fits.
        """
        heads = self.config.num_scorer_heads(self.num_qo_heads, self.num_kv_heads)
        # heads * ceil(S/128) * batch * S <= LIMIT, with ceil(S/128) ~= S/128.
        limit = self.SCORE_TENSOR_MAX_ELEMS * self.SCORER_BLOCK / (heads * max(batch, 1))
        s = int(limit**0.5)
        s -= s % self.SCORER_BLOCK
        # Walk back if the ceil made the estimate optimistic at the boundary.
        while s > 0 and heads * -(-s // self.SCORER_BLOCK) * batch * s > self.SCORE_TENSOR_MAX_ELEMS:
            s -= self.SCORER_BLOCK
        return s

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"MsaSparseAttention({self.config.label}, "
            f"Hq={self.num_qo_heads}, Hkv={self.num_kv_heads}, D={self.head_dim})"
        )
