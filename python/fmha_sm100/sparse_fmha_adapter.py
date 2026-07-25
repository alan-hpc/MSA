# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Drop-in adapter: fmha_sm100 API → Fireworks KV-outer sparse backend.

The KV-outer backend is used when ``num_qo_heads // num_kv_heads >= 8``. Smaller
GQA ratios fall back to the legacy CuTe sparse path shipped under ``fmha_sm100/cute/``.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import torch

from .kvouter import kvouter_attention

_MM_SPARSE_DIR = os.path.join(os.path.dirname(__file__), "cute")
if os.path.isdir(_MM_SPARSE_DIR) and _MM_SPARSE_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_MM_SPARSE_DIR))

BLOCK_SIZE = 128
_KVOUTER_MIN_QHEADS_PER_KV = 8


def sparse_fmha_plan(
    qo_segment_lens: torch.Tensor,
    kv_segment_lens: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int = -1,
    qo_offset: Optional[torch.Tensor] = None,
    num_kv_splits: int = -1,
    page_size: int = -1,
    output_maxscore: bool = False,
    kv_block_num: int = -1,
    causal: bool = True,
    usable_SM_count=-1,
    use_fp8_kvcache: bool = False,
) -> dict:
    """Build a reusable sparse-prefill plan."""
    assert kv_block_num in {4, 8, 16, 32}, f"kv_block_num={kv_block_num}"
    assert page_size >= 1, f"page_size={page_size}"
    assert output_maxscore is False
    assert causal is True

    batch = qo_segment_lens.shape[0]
    gpu_device = torch.device("cuda")

    cu_seqlens_q = torch.zeros(batch + 1, dtype=torch.int32, device=gpu_device)
    cu_seqlens_q[1:] = torch.cumsum(qo_segment_lens.to(torch.int32).to(gpu_device), dim=0)

    cu_seqlens_k = torch.zeros(batch + 1, dtype=torch.int32, device=gpu_device)
    cu_seqlens_k[1:] = torch.cumsum(kv_segment_lens.to(torch.int32).to(gpu_device), dim=0)

    max_seqlen_q = int(qo_segment_lens.max().item())
    max_seqlen_k = int(kv_segment_lens.max().item())
    total_k = int(kv_segment_lens.sum().item())
    blk_kv = page_size

    kv_lens = kv_segment_lens.tolist()
    total_rows = sum((int(kl) + blk_kv - 1) // blk_kv for kl in kv_lens)

    if qo_offset is not None:
        seqused_k = (qo_segment_lens + qo_offset).to(torch.int32).to(gpu_device)
    else:
        seqused_k = kv_segment_lens.to(torch.int32).to(gpu_device)

    qhead_per_kv = num_qo_heads // num_kv_heads if num_kv_heads > 0 else 1
    use_kvouter = qhead_per_kv >= _KVOUTER_MIN_QHEADS_PER_KV

    plan = {
        "qo_segment_lens": qo_segment_lens,
        "cu_seqlens_q": cu_seqlens_q,
        "qo_segment_offsets": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        "kv_segment_lens": kv_segment_lens.to(torch.int32).to(gpu_device),
        "seqused_k": seqused_k,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
        "total_k": total_k,
        "total_rows": total_rows,
        "num_qo_heads": num_qo_heads,
        "page_size": page_size,
        "blk_kv": blk_kv,
        "kv_block_num": kv_block_num,
        "causal": causal,
        "batch": batch,
        "MM-SA-Nv": True,
        "usable_SM_count": usable_SM_count,
        "num_kv_heads": num_kv_heads,
        "qhead_per_kv": qhead_per_kv,
        "use_kvouter": use_kvouter,
    }

    if use_kvouter:
        return plan

    # Legacy CuTe sparse planner metadata (qhead_per_kv < 8).
    del num_kv_splits
    from interface import sparse_atten_func  # noqa: F401
    from sparse_index_utils import build_k2q_csr  # noqa: F401
    from src.sm100.prepare_scheduler import SPARSE_SCHEDULE_MODEL
    from src.common.aot_cache import _key_to_path

    total_q = int(cu_seqlens_q[-1].item())
    target_q_per_cta = SPARSE_SCHEDULE_MODEL.balanced_target_q_per_cta(
        total_q=total_q,
        topk=kv_block_num,
        blk_kv=page_size,
        head_kv=num_kv_heads,
        qhead_per_kv=qhead_per_kv,
        device=gpu_device,
        usable_SM_count=usable_SM_count,
    )
    scheduler_metadata_capacity = SPARSE_SCHEDULE_MODEL.flat_schedule_capacity(
        total_rows=total_rows,
        total_q=total_q,
        topk=kv_block_num,
        head_kv=num_kv_heads,
        target_q_per_cta=target_q_per_cta,
    )

    def _compute_aot_kernel_paths(head_dim, n_block_size, topk, dtype, partial_dtype):
        fwd_key = (
            "sparse_forward_sm100_csr_varlen",
            head_dim,
            n_block_size,
            qhead_per_kv,
            dtype,
            partial_dtype,
            True,
            True,
            True,
            n_block_size,
            True,
            False,
        )
        k_block_size = 128 if head_dim > 64 else 64
        try:
            import cutlass

            cutlass_partial = cutlass.Float32
            cutlass_out = cutlass.BFloat16
        except ImportError:
            cutlass_partial = partial_dtype
            cutlass_out = dtype
        combine_key = (
            "combine",
            head_dim,
            k_block_size,
            64,
            topk,
            cutlass_partial,
            cutlass_out,
            True,
            False,
            True,
            False,
            True,
            True,
        )
        fwd_path = _key_to_path(fwd_key) + ".o"
        combine_path = _key_to_path(combine_key) + ".o"
        return {
            "fwd_kernel_path": fwd_path if os.path.isfile(fwd_path) else "",
            "combine_kernel_path": combine_path if os.path.isfile(combine_path) else "",
            "fwd_func_name": str(fwd_key[0]),
            "combine_func_name": str(combine_key[0]),
        }

    plan.update(
        {
            "target_q_per_cta": target_q_per_cta,
            "scheduler_metadata_capacity": scheduler_metadata_capacity,
            **_compute_aot_kernel_paths(
                head_dim=128,
                n_block_size=page_size,
                topk=kv_block_num,
                dtype=torch.float8_e4m3fn if use_fp8_kvcache else torch.bfloat16,
                partial_dtype=torch.float32,
            ),
        }
    )
    return plan


def _selected_for_kvouter(
    kv_block_indexes: torch.Tensor,
    num_kv_heads: int,
    num_qo_heads: int,
    qhead_per_kv: int,
) -> torch.Tensor:
    """Normalize ``kv_block_indexes`` to ``[total_q, H_kv, topk]``."""
    h_dim = kv_block_indexes.shape[1]
    if h_dim == num_kv_heads:
        selected = kv_block_indexes
    elif h_dim == num_qo_heads and num_qo_heads != num_kv_heads:
        selected = kv_block_indexes[:, ::qhead_per_kv, :]
    else:
        raise ValueError(
            f"kv_block_indexes head dim {h_dim} doesn't match "
            f"num_kv_heads={num_kv_heads} or num_qo_heads={num_qo_heads}"
        )
    return selected.to(torch.int32).contiguous()


def _convert_kv_block_indexes_to_q2k(
    kv_block_indexes: torch.Tensor,
    num_kv_heads: int,
    num_qo_heads: int,
    qhead_per_kv: int,
) -> torch.Tensor:
    """kv_block_indexes [total_q, H, topk] → q2k [H_kv, total_q, topk]."""
    h_dim = kv_block_indexes.shape[1]
    if h_dim == num_kv_heads:
        q2k = kv_block_indexes.permute(1, 0, 2).contiguous()
    elif h_dim == num_qo_heads and num_qo_heads != num_kv_heads:
        q2k = kv_block_indexes[:, ::qhead_per_kv, :].permute(1, 0, 2).contiguous()
    else:
        raise ValueError(
            f"kv_block_indexes head dim {h_dim} doesn't match "
            f"num_kv_heads={num_kv_heads} or num_qo_heads={num_qo_heads}"
        )
    return q2k.to(torch.int32)


def _build_page_table(
    kv_indices: torch.Tensor,
    kv_segment_lens: torch.Tensor,
    page_size: int,
    batch: int,
) -> torch.Tensor:
    """Flat kv_indices → page_table [batch, max_pages_per_seq]."""
    kv_lens = kv_segment_lens.tolist()
    pages_per_batch = [(int(kl) + page_size - 1) // page_size for kl in kv_lens]
    max_pages = max(pages_per_batch)
    total = batch * max_pages
    buf = torch.zeros(total + 4, dtype=torch.int32, device=kv_indices.device)
    shift = ((-buf.data_ptr()) % 16) // 4
    page_table = buf[shift : shift + total].view(batch, max_pages)
    assert page_table.data_ptr() % 16 == 0
    offset = 0
    for b in range(batch):
        n = pages_per_batch[b]
        page_table[b, :n] = kv_indices[offset : offset + n].to(torch.int32)
        offset += n
    return page_table


def _sparse_fmha_kvouter(
    q,
    k,
    v,
    plan_info,
    *,
    kv_indices,
    kv_block_indexes,
    seqused_k,
    sm_scale,
    out,
):
    page_size = plan_info["page_size"]
    causal = plan_info["causal"]
    batch = plan_info["batch"]
    kv_segment_lens = plan_info["kv_segment_lens"]
    cu_seqlens_q = plan_info["cu_seqlens_q"]
    num_qo_heads = plan_info["num_qo_heads"]
    num_kv_heads = k.shape[1]
    qhead_per_kv = num_qo_heads // num_kv_heads

    selected = _selected_for_kvouter(
        kv_block_indexes, num_kv_heads, num_qo_heads, qhead_per_kv,
    )
    page_table = _build_page_table(kv_indices, kv_segment_lens, page_size, batch)
    softmax_scale = sm_scale if sm_scale is not None else q.shape[-1] ** -0.5
    out_dtype = torch.bfloat16 if q.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else q.dtype

    result, _ = kvouter_attention(
        q,
        k,
        v,
        selected,
        page_table,
        cu_seqlens_q=cu_seqlens_q,
        softmax_scale=softmax_scale,
        causal=causal,
        used_kv_lens=seqused_k,
        block_size=BLOCK_SIZE,
        page_size=page_size,
        out_dtype=out_dtype,
        return_lse=False,
    )
    if out is not None:
        out.copy_(result)
        return out, None
    return result, None


def _sparse_fmha_legacy_cute(
    q,
    k,
    v,
    plan_info,
    *,
    kv_indices,
    kv_block_indexes,
    seqused_k,
    sm_scale,
    out,
):
    from interface import sparse_atten_func
    from sparse_index_utils import build_k2q_csr

    qo_segment_lens = plan_info["qo_segment_lens"]
    cu_seqlens_q = plan_info["cu_seqlens_q"]
    cu_seqlens_k = plan_info["cu_seqlens_k"]
    num_qo_heads = plan_info["num_qo_heads"]
    num_kv_heads = k.shape[1]
    qhead_per_kv = num_qo_heads // num_kv_heads
    page_size = plan_info["page_size"]
    blk_kv = plan_info["blk_kv"]
    topk = plan_info["kv_block_num"]
    causal = plan_info["causal"]
    max_seqlen_q = plan_info["max_seqlen_q"]
    max_seqlen_k = plan_info["max_seqlen_k"]
    total_k = plan_info["total_k"]
    total_rows = plan_info["total_rows"]
    batch = plan_info["batch"]
    kv_segment_lens = plan_info["kv_segment_lens"]
    usable_SM_count = int(plan_info.get("usable_SM_count", -1))

    q2k = _convert_kv_block_indexes_to_q2k(
        kv_block_indexes, num_kv_heads, num_qo_heads, qhead_per_kv,
    )
    page_table = _build_page_table(kv_indices, kv_segment_lens, page_size, batch)
    softmax_scale = sm_scale if sm_scale is not None else q.shape[-1] ** -0.5

    if usable_SM_count > 0:
        k2q_row_ptr, k2q_q_indices = build_k2q_csr(
            q2k,
            cu_seqlens_q,
            cu_seqlens_k,
            blk_kv,
            total_k=total_k,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_q=max_seqlen_q,
            total_rows=total_rows,
            qhead_per_kv=qhead_per_kv,
            return_schedule=False,
        )
        schedule = None
    else:
        k2q_row_ptr, k2q_q_indices, schedule = build_k2q_csr(
            q2k,
            cu_seqlens_q,
            cu_seqlens_k,
            blk_kv,
            total_k=total_k,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_q=max_seqlen_q,
            total_rows=total_rows,
            qhead_per_kv=qhead_per_kv,
            return_schedule=True,
        )

    result = sparse_atten_func(
        q,
        k,
        v,
        k2q_row_ptr,
        k2q_q_indices,
        topk,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        blk_kv=blk_kv,
        causal=causal,
        softmax_scale=softmax_scale,
        return_softmax_lse=False,
        page_table=page_table,
        seqused_k=seqused_k,
        schedule=schedule,
        usable_SM_count=usable_SM_count,
    )
    if out is not None:
        out.copy_(result)
        return out, None
    return result, None


def sparse_fmha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan_info: dict,
    out: Optional[torch.Tensor] = None,
    max_score: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    o_scale: Optional[float] = None,
    kv_indices: Optional[torch.Tensor] = None,
    output_maxscore: bool = True,
    output_o: bool = True,
    kv_block_indexes: Optional[torch.Tensor] = None,
    q_offset_override=None,
    check_input_valid: bool = False,
) -> Tuple[torch.Tensor, None]:
    """Run sparse prefill using KV-outer (preferred) or legacy CuTe sparse."""
    del max_score, q_scale, k_scale, v_scale, o_scale, output_maxscore, output_o, check_input_valid
    if kv_block_indexes is None:
        raise ValueError("sparse_fmha requires kv_block_indexes")
    if kv_indices is None or plan_info["page_size"] <= 0 or k.ndim != 4:
        raise ValueError("sparse_fmha requires paged KV with kv_indices")

    qo_segment_lens = plan_info["qo_segment_lens"]
    if isinstance(q_offset_override, int):
        q_offset_override = torch.full_like(qo_segment_lens, q_offset_override)
    elif q_offset_override is not None:
        assert q_offset_override.device == qo_segment_lens.device

    if q_offset_override is not None:
        seqused_k = (qo_segment_lens + q_offset_override).to(torch.int32).to(qo_segment_lens.device)
    else:
        seqused_k = plan_info["seqused_k"]

    runner = _sparse_fmha_kvouter if plan_info.get("use_kvouter", False) else _sparse_fmha_legacy_cute
    return runner(
        q,
        k,
        v,
        plan_info,
        kv_indices=kv_indices,
        kv_block_indexes=kv_block_indexes,
        seqused_k=seqused_k,
        sm_scale=sm_scale,
        out=out,
    )
